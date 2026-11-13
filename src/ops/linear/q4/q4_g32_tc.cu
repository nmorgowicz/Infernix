#include "ops/linear/q4/q4_g32_tc.cuh"

#include "core/device.h"
#include "ops/common/q4_scale_mode.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace infernix::ops::detail {
namespace {

// One byte holds two signed codes in [-8, 7]. Both are exact in BF16.
__device__ __forceinline__ unsigned q4_byte_to_bf16x2(unsigned packed) {
    const unsigned lo     = (packed & 0x0fu) ^ 0x08u;
    const unsigned hi     = ((packed >> 4) & 0x0fu) ^ 0x08u;
    const unsigned biased = (0x4300u | lo) | ((0x4300u | hi) << 16);
    union {
        unsigned bits;
        __nv_bfloat162 pair;
    } value, bias, out;
    value.bits = biased;
    bias.bits  = 0x43084308u;
    out.pair   = __hsub2(value.pair, bias.pair);
    return out.bits;
}

// Tensor-core GEMV for up to 8 tokens. Each warp owns 64 K. Codes and the stage's group
// scales move with cp.async, so no load in the MMA loop waits on DRAM. The FP16 group scale
// is applied in FP32 after that group's two m16n8k16 steps, then the warps reduce their K
// partials.
//
// kScaleMode (q4_scale_mode.h): 0 for q4_g32 (one scale per 32 K, 16-byte staged), 2 for g128 and
// 3 for g256 (one scale per 128 / 256 K). A coarse scale row is K >> (4 + mode) bytes, so its
// stride is not 16-byte aligned (84 B at K = 5376 / g128, 42 B at g256), and a 256-K stage owns
// only 1 or 2 scales of a row. The stage therefore copies the one aligned 4-byte word that holds
// them (cp.async zero-filled past the plane end) and the MMA loop picks the halfword by the
// scale's element parity. The per-32-K MMA / FP32 scale structure is the same for every mode.
template <int kRowBlocks, int kWarps, int kStages, int kScaleMode>
__global__ void __launch_bounds__(kWarps * 32, 2)
    q4g32_tc8_kernel(const __nv_bfloat16* __restrict__ x,
                                 const std::uint8_t* __restrict__ codes,
                                 const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                                 int rows, int k, int tokens) {
    constexpr int kTileK          = 64;
    constexpr int kGroupK         = kWarps * kTileK;
    constexpr int kRows           = 16 * kRowBlocks;
    constexpr int kTileTokens     = 8;
    constexpr int kRowsPerWarp    = kRows / kWarps;
    constexpr int kGroupsPerStage = kGroupK / 32;
    constexpr int kBytesPerRow    = kGroupsPerStage * 16;
    constexpr int kCodeBytes      = kRows * kBytesPerRow;
    constexpr int kActBytes = kWarps * kTileTokens * kTileK * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr int kScaleStageElems = kGroupsPerStage >> kScaleMode; // coarse scales a stage spans
    constexpr int kScaleRowBytes   = kScaleMode == 0 ? kGroupsPerStage * 2 : 4;
    constexpr int kScaleBytes    = kRows * kScaleRowBytes;
    constexpr int kStageBytes    = kCodeBytes + kActBytes + kScaleBytes;
    constexpr int kSharedBytes = kStages * kStageBytes;
    static_assert(kRows % kWarps == 0);
    static_assert(kGroupsPerStage == 8 || kGroupsPerStage == 16);
    static_assert(kScaleMode == 0 || (kGroupsPerStage % (1 << kScaleMode) == 0 && kScaleStageElems <= 2),
                  "a coarse stage must fit its scales in one aligned 4-byte word");
    extern __shared__ __align__(16) unsigned char smem[];
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRows;
    const int n_groups = k >> 5;
    // Scale units per row: coarse scales.
    const int n_scales = k >> (5 + kScaleMode);
    const int b_row      = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;

    const auto stage_bytes = [&](int stage) { return smem + stage * kStageBytes; };
    const auto stage_acts  = [&](int stage) {
        return reinterpret_cast<__nv_bfloat16*>(stage_bytes(stage) + kCodeBytes);
    };
    const auto stage_scales = [&](int stage) {
        return reinterpret_cast<const __half*>(stage_bytes(stage) + kCodeBytes + kActBytes);
    };

    const auto stage_group = [&](int group_k0, int stage) {
        auto* code_smem  = stage_bytes(stage);
        const int group0 = group_k0 >> 5;
        for (int item = lane; item < kRowsPerWarp * kGroupsPerStage; item += 32) {
            const int local_row = item / kGroupsPerStage;
            const int group     = item - local_row * kGroupsPerStage;
            const int row       = warp * kRowsPerWarp + local_row;
            const int slot      = group ^ (row & 7);
            cp_async<16, Cache::cg>(
                code_smem + row * kBytesPerRow + slot * 16,
                codes + (static_cast<long long>(row0 + row) * n_groups + group0 + group) * 16);
        }
        auto* scale_smem = stage_bytes(stage) + kCodeBytes + kActBytes;
        if constexpr (kScaleMode == 0) {
            // The stage's group scales of one row are contiguous: 16 or 32 bytes, 16-byte aligned
            // because every trunk K is a multiple of 256.
            for (int item = lane; item < kRowsPerWarp * (kScaleRowBytes / 16); item += 32) {
                const int local_row = item / (kScaleRowBytes / 16);
                const int chunk     = item - local_row * (kScaleRowBytes / 16);
                const int row       = warp * kRowsPerWarp + local_row;
                cp_async<16, Cache::cg>(
                    scale_smem + row * kScaleRowBytes + chunk * 16,
                    scales + (static_cast<long long>(row0 + row) * n_groups + group0) * 2 + chunk * 16);
            }
        } else {
            // One aligned 4-byte word per row holds the stage's 1 or 2 coarse scales. The word
            // starts at the even element at or below the stage's first scale; past the plane end
            // the bytes are zero-filled instead of read.
            const long long plane_bytes = static_cast<long long>(rows) * n_scales * 2;
            const int stage_index       = group_k0 / kGroupK;
            for (int local_row = lane; local_row < kRowsPerWarp; local_row += 32) {
                const int row = warp * kRowsPerWarp + local_row;
                const long long element =
                    static_cast<long long>(row0 + row) * n_scales + stage_index * kScaleStageElems;
                const long long byte  = (element >> 1) * 4;
                const long long left  = plane_bytes - byte;
                cp_async_zfill<4, Cache::ca>(scale_smem + row * 4, scales + byte,
                                             left >= 4 ? 4 : static_cast<int>(left));
            }
        }
        auto* activations = stage_acts(stage) + warp * kTileTokens * kTileK;
        for (int item = lane; item < kTileTokens * (kTileK / 8); item += 32) {
            const int token  = item / (kTileK / 8);
            const int k8     = item - token * (kTileK / 8);
            const int source = token < tokens ? token : 0;
            cp_async_zfill<16, Cache::ca>(
                activations + token * kTileK + fp8_a16_shared_col_64(token, k8 * 8),
                x + static_cast<long long>(source) * k + group_k0 + warp * kTileK + k8 * 8,
                token < tokens ? 16 : 0);
        }
    };

    float acc[kRowBlocks][4] = {};
    const int k_groups = k / kGroupK;
    // Keep kStages - 1 stages in flight. Every iteration commits one group (empty past the end),
    // so waiting for all but kStages - 2 groups always means the current stage has landed.
#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < k_groups) { stage_group(s * kGroupK, s); }
        cp_commit();
    }
    for (int group = 0; group < k_groups; ++group) {
        const int stage = group % kStages;
        const int next  = group + kStages - 1;
        if (next < k_groups) { stage_group(next * kGroupK, next % kStages); }
        cp_commit();
        cp_wait<kStages - 1>();
        __syncthreads();

        const auto* code_smem    = stage_bytes(stage);
        const auto* activations  = stage_acts(stage) + warp * kTileTokens * kTileK;
        const auto* scale_smem   = stage_scales(stage);
        float scale_top[kRowBlocks][2];
        float scale_bot[kRowBlocks][2];
        if constexpr (kScaleMode == 0) {
#pragma unroll
            for (int rb = 0; rb < kRowBlocks; ++rb) {
                const int top = rb * 16 + gid;
#pragma unroll
                for (int which = 0; which < 2; ++which) {
                    const int g          = warp * 2 + which;
                    scale_top[rb][which] = __half2float(scale_smem[top * kGroupsPerStage + g]);
                    scale_bot[rb][which] = __half2float(scale_smem[(top + 8) * kGroupsPerStage + g]);
                }
            }
        } else {
            // Both 32-K groups of this warp sit in one coarse group. The stage word of a row holds
            // scale elements 2w and 2w + 1 of the plane, element e = row * n_scales + scale index.
            const auto* scale_words = reinterpret_cast<const unsigned*>(scale_smem);
            const unsigned first    = static_cast<unsigned>(group * kScaleStageElems + ((warp * 2) >> kScaleMode));
            const auto scale_of     = [&](int local_row) {
                const unsigned element = static_cast<unsigned>(row0 + local_row) * static_cast<unsigned>(n_scales) + first;
                const unsigned bits    = (scale_words[local_row] >> ((element & 1u) * 16u)) & 0xffffu;
                return __half2float(__ushort_as_half(static_cast<unsigned short>(bits)));
            };
#pragma unroll
            for (int rb = 0; rb < kRowBlocks; ++rb) {
                const int top = rb * 16 + gid;
                scale_top[rb][0] = scale_top[rb][1] = scale_of(top);
                scale_bot[rb][0] = scale_bot[rb][1] = scale_of(top + 8);
            }
        }

        float step[kRowBlocks][4] = {};
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            unsigned b0, b1;
            ldmatrix_x2(b0, b1,
                        smem_addr(activations + b_row * kTileK +
                                  fp8_a16_shared_col_64(b_row, k_step * 16 + b_k_offset)));
            const int k_in_warp = k_step * 16 + lid * 2;
            const auto code_bits = [&](int row, int k_local) -> unsigned {
                const int group_in_stage = warp * 2 + (k_local >> 5);
                const int slot           = group_in_stage ^ (row & 7);
                const int byte           = (k_local & 31) >> 1;
                return q4_byte_to_bf16x2(code_smem[row * kBytesPerRow + slot * 16 + byte]);
            };
#pragma unroll
            for (int rb = 0; rb < kRowBlocks; ++rb) {
                const int r        = rb * 16 + gid;
                const unsigned a0  = code_bits(r, k_in_warp);
                const unsigned a1  = code_bits(r + 8, k_in_warp);
                const unsigned a2  = code_bits(r, k_in_warp + 8);
                const unsigned a3  = code_bits(r + 8, k_in_warp + 8);
                mma_bf16(step[rb][0], step[rb][1], step[rb][2], step[rb][3], a0, a1, a2, a3, b0, b1);
                if ((k_step & 1) == 1) {
                    const int which = k_step >> 1;
                    acc[rb][0] += step[rb][0] * scale_top[rb][which];
                    acc[rb][1] += step[rb][1] * scale_top[rb][which];
                    acc[rb][2] += step[rb][2] * scale_bot[rb][which];
                    acc[rb][3] += step[rb][3] * scale_bot[rb][which];
                    step[rb][0] = step[rb][1] = step[rb][2] = step[rb][3] = 0.0f;
                }
            }
        }
        __syncthreads();
    }

    auto* partial = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (int step = 1; step < kWarps; step *= 2) {
            const int slot = warp / (2 * step);
            if (warp % (2 * step) == step) {
                store_vec(partial + (slot * 32 + lane) * 4,
                          make_float4(acc[rb][0], acc[rb][1], acc[rb][2], acc[rb][3]));
            }
            __syncthreads();
            if (warp % (2 * step) == 0) {
                const float4 other = load_vec<float4>(partial + (slot * 32 + lane) * 4);
                acc[rb][0] += other.x;
                acc[rb][1] += other.y;
                acc[rb][2] += other.z;
                acc[rb][3] += other.w;
            }
            __syncthreads();
        }
    }

    if (warp == 0) {
#pragma unroll
        for (int rb = 0; rb < kRowBlocks; ++rb) {
            const int top    = row0 + rb * 16 + gid;
            const int token0 = 2 * lid;
            auto store       = [&](int row, int token, float value) {
                if (token < tokens && row < rows) {
                    out[row + static_cast<long long>(token) * rows] = __float2bfloat16(value);
                }
            };
            store(top, token0, acc[rb][0]);
            store(top, token0 + 1, acc[rb][1]);
            store(top + 8, token0, acc[rb][2]);
            store(top + 8, token0 + 1, acc[rb][3]);
        }
    }
    (void)kSharedBytes;
    (void)kScaleStageElems;
    (void)n_scales;
}

} // namespace

void q4_g32_tc8(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                int scale_mode) {
    auto launch_staged = [&](auto row_blocks, auto warps, auto stages, auto mode) {
        constexpr int kRowBlocks  = decltype(row_blocks)::value;
        constexpr int kWarps      = decltype(warps)::value;
        constexpr int kStages     = decltype(stages)::value;
        constexpr int kScaleMode  = decltype(mode)::value;
        constexpr int kRows       = 16 * kRowBlocks;
        constexpr int kGroupK     = kWarps * 64;
        constexpr int kScaleRow   = kScaleMode == 0 ? (kGroupK / 32) * 2 : 4;
        constexpr int kShared = kStages * (kRows * ((kGroupK / 32) * 16 + kScaleRow) + kWarps * 8 * 64 * 2);
        static const cudaError_t attribute =
            cudaFuncSetAttribute(q4g32_tc8_kernel<kRowBlocks, kWarps, kStages, kScaleMode>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kShared);
        CUDA_CHECK(attribute);
        const dim3 block(kWarps * 32);
        const dim3 grid(operands.rows / kRows);
        q4g32_tc8_kernel<kRowBlocks, kWarps, kStages, kScaleMode><<<grid, block, kShared, stream>>>(
            operands.x, operands.codes, operands.scales, out, operands.rows, operands.k, operands.tokens);
    };
    // Three stages measured best on the cold-L2 sweep of 2, 3, 4 and 6 at T=7.
    auto launch = [&](auto row_blocks, auto warps) {
        constexpr auto stages = std::integral_constant<int, 3>{};
        switch (scale_mode) {
        case 0: launch_staged(row_blocks, warps, stages, std::integral_constant<int, 0>{}); break;
        case 2: launch_staged(row_blocks, warps, stages, std::integral_constant<int, 2>{}); break;
        case 3: launch_staged(row_blocks, warps, stages, std::integral_constant<int, 3>{}); break;
        default: throw std::invalid_argument("q4 tensor-core GEMV: unsupported scale mode");
        }
    };
    // 16384 rows wants 32 per CTA. Wider outputs keep 64 so one activation tile
    // feeds four MMA row blocks. The 5376-row shapes stay at 16 rows: eight warps
    // measured no faster there (1.23-1.40 against 1.25-1.42 TB/s).
    if (operands.rows >= 18432) {
        launch(std::integral_constant<int, 4>{}, std::integral_constant<int, 4>{});
    } else if (operands.rows >= 8192) {
        launch(std::integral_constant<int, 2>{}, std::integral_constant<int, 4>{});
    } else {
        launch(std::integral_constant<int, 1>{}, std::integral_constant<int, 4>{});
    }
}

} // namespace infernix::ops::detail
