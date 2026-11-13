#pragma once

// Short-prefill (17..127 token) row-scaled E4M3 weight x BF16 activation GEMV on Tensor Cores.
//
// Like fp8_a16_ksplit_mma.cuh, but a CTA owns 16 x RowBlocks output rows, so each activation
// fragment loaded from shared memory feeds RowBlocks MMAs and the activations are read from L2
// RowBlocks times less often. Warps split K. FP8 codes are widened exactly to BF16; the BF16 row
// multiplier is applied once to the complete FP32 dot product. The K-split partials are reduced
// in a fixed pairwise tree, so results do not depend on scheduling.

#include "core/device.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

template <int TileTokens, int RowBlocks, int KWarps, int MinBlocksPerSm = 1>
struct Fp8A16WideSchedule {
    static_assert(TileTokens % 8 == 0 && TileTokens >= 8 && TileTokens <= 128);
    static_assert(RowBlocks == 1 || RowBlocks == 2 || RowBlocks == 4);
    static_assert(KWarps == 2 || KWarps == 4 || KWarps == 8);
    static constexpr int kTileTokens    = TileTokens;
    static constexpr int kRowBlocks     = RowBlocks;
    static constexpr int kKWarps        = KWarps;
    static constexpr int kMinBlocksPerSm = MinBlocksPerSm;
    static constexpr int kThreads       = KWarps * 32;
    static constexpr int kTileK         = 64;
    static constexpr int kGroupK        = KWarps * kTileK;
    static constexpr int kRows          = 16 * RowBlocks;
    static constexpr int kTokenMmas     = TileTokens / 8;
    static constexpr int kCodeBytes     = kRows * kGroupK;
    static constexpr int kActivationBytes = KWarps * TileTokens * kTileK * 2;
    static constexpr int kStageBytes    = kCodeBytes + kActivationBytes;
    static constexpr int kStages        = 2 * kStageBytes <= 96 * 1024 ? 2 : 1;
    // One row block's partials from half of the warps.
    static constexpr int kReduceBytes   = (KWarps / 2) * TileTokens * 16 * 4;
    static constexpr int kSharedBytes =
        kStages * kStageBytes > kReduceBytes ? kStages * kStageBytes : kReduceBytes;
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Geometry, class S>
__global__ __launch_bounds__(S::kThreads, S::kMinBlocksPerSm) void fp8_a16_wide_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Fp8ContiguousOutput output, int columns) {
    constexpr int kHidden = Geometry::kInputRows;
    constexpr int kTileK = S::kTileK, kGroupK = S::kGroupK, kRows = S::kRows;
    constexpr int kGroups = kHidden / kGroupK, kStages = S::kStages;
    constexpr int kTokenMmas = S::kTokenMmas, kWarps = S::kKWarps;
    constexpr int kChunks = kGroupK / 16;
    constexpr int kRowsPerWarp = kRows / kWarps;
    static_assert(kHidden % kGroupK == 0 && Geometry::kOutputRows % kRows == 0);
    static_assert(kChunks >= 8 && kRows % kWarps == 0);

    extern __shared__ __align__(16) unsigned char wide_shared[];
    const auto codes_at = [&](int stage) {
        return wide_shared + stage * S::kStageBytes;
    };
    const auto activations_at = [&](int stage) {
        return reinterpret_cast<__nv_bfloat16*>(wide_shared + stage * S::kStageBytes +
                                                S::kCodeBytes);
    };

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = static_cast<int>(blockIdx.x) * kRows;
    const int live_mmas = (columns + 7) / 8;

    const auto stage_group = [&](int group_k0, int stage) {
        auto* codes = codes_at(stage);
        for (int item = lane; item < kRowsPerWarp * kChunks; item += 32) {
            const int row   = warp * kRowsPerWarp + item / kChunks;
            const int chunk = item % kChunks;
            cp_async<16, Cache::cg>(codes + row * kGroupK + (chunk ^ (row & 7)) * 16,
                                    weight_codes + static_cast<std::int64_t>(row0 + row) * kHidden +
                                        group_k0 + chunk * 16);
        }
        auto* activations = activations_at(stage) + warp * S::kTileTokens * kTileK;
        for (int item = lane; item < S::kTileTokens * (kTileK / 8); item += 32) {
            const int token  = item / (kTileK / 8);
            const int k8     = item - token * (kTileK / 8);
            const int source = token < columns ? token : 0;
            cp_async_zfill<16, Cache::ca>(
                activations + token * kTileK + fp8_a16_shared_col_64(token, k8 * 8),
                x + static_cast<std::int64_t>(source) * kHidden + group_k0 + warp * kTileK + k8 * 8,
                token < columns ? 16 : 0);
        }
    };

    const int b_row = lane & 7;
    const int b_k_offset = ((lane >> 3) & 1) << 3;
    const int warp_k0 = warp * kTileK;
    float acc[S::kRowBlocks][kTokenMmas][4] = {};

    stage_group(0, 0);
    cp_commit();
    for (int group = 0; group < kGroups; ++group) {
        const int stage = group % kStages;
        if (kStages == 2 && group + 1 < kGroups) {
            stage_group((group + 1) * kGroupK, (group + 1) % kStages);
            cp_commit();
            cp_wait<1>();
        } else {
            cp_wait<0>();
        }
        __syncthreads();
        const auto* codes = codes_at(stage);
        const auto* activations = activations_at(stage) + warp * S::kTileTokens * kTileK;
#pragma unroll
        for (int k_step = 0; k_step < kTileK / 16; ++k_step) {
            unsigned b[kTokenMmas][2];
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                if (token_mma < live_mmas) {
                    const int row = token_mma * 8 + b_row;
                    ldmatrix_x2(b[token_mma][0], b[token_mma][1],
                                smem_addr(activations + row * kTileK +
                                          fp8_a16_shared_col_64(row, k_step * 16 + b_k_offset)));
                }
            }
            const int code_col = warp_k0 + k_step * 16 + lid * 2;
            const auto code_pair = [&](int row, int col) {
                const int offset = ((col >> 4) ^ (row & 7)) * 16 + (col & 15);
                return static_cast<unsigned>(
                    *reinterpret_cast<const std::uint16_t*>(codes + row * kGroupK + offset));
            };
#pragma unroll
            for (int rb = 0; rb < S::kRowBlocks; ++rb) {
                const int r = rb * 16 + gid;
                const unsigned a0 = fp8_e4m3x2_to_bf16x2_bits(code_pair(r, code_col));
                const unsigned a1 = fp8_e4m3x2_to_bf16x2_bits(code_pair(r + 8, code_col));
                const unsigned a2 = fp8_e4m3x2_to_bf16x2_bits(code_pair(r, code_col + 8));
                const unsigned a3 = fp8_e4m3x2_to_bf16x2_bits(code_pair(r + 8, code_col + 8));
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    if (token_mma < live_mmas) {
                        mma_bf16(acc[rb][token_mma][0], acc[rb][token_mma][1],
                                 acc[rb][token_mma][2], acc[rb][token_mma][3], a0, a1, a2, a3,
                                 b[token_mma][0], b[token_mma][1]);
                    }
                }
            }
        }
        __syncthreads();
        if (kStages == 1 && group + 1 < kGroups) {
            stage_group((group + 1) * kGroupK, 0);
            cp_commit();
        }
    }

    // Pairwise tree per row block: at step s, warp w (w % 2s == s) hands its sums to w - s.
    auto* partial = reinterpret_cast<float*>(wide_shared);
#pragma unroll
    for (int rb = 0; rb < S::kRowBlocks; ++rb) {
#pragma unroll
        for (int step = 1; step < kWarps; step *= 2) {
            const int slot = warp / (2 * step);
            if (warp % (2 * step) == step) {
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    store_vec(partial + ((slot * kTokenMmas + token_mma) * 32 + lane) * 4,
                              make_float4(acc[rb][token_mma][0], acc[rb][token_mma][1],
                                          acc[rb][token_mma][2], acc[rb][token_mma][3]));
                }
            }
            __syncthreads();
            if (warp % (2 * step) == 0) {
#pragma unroll
                for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                    const float4 other =
                        load_vec<float4>(partial + ((slot * kTokenMmas + token_mma) * 32 + lane) * 4);
                    acc[rb][token_mma][0] += other.x;
                    acc[rb][token_mma][1] += other.y;
                    acc[rb][token_mma][2] += other.z;
                    acc[rb][token_mma][3] += other.w;
                }
            }
            __syncthreads();
        }
    }

    if (warp == 0) {
#pragma unroll
        for (int rb = 0; rb < S::kRowBlocks; ++rb) {
            const int top = row0 + rb * 16 + gid;
            const float top_scale    = __bfloat162float(row_scales[top]);
            const float bottom_scale = __bfloat162float(row_scales[top + 8]);
#pragma unroll
            for (int token_mma = 0; token_mma < kTokenMmas; ++token_mma) {
                const int token0 = token_mma * 8 + 2 * lid;
                if (token0 < columns) {
                    output.store(top, token0, acc[rb][token_mma][0] * top_scale);
                    output.store(top + 8, token0, acc[rb][token_mma][2] * bottom_scale);
                }
                if (token0 + 1 < columns) {
                    output.store(top, token0 + 1, acc[rb][token_mma][1] * top_scale);
                    output.store(top + 8, token0 + 1, acc[rb][token_mma][3] * bottom_scale);
                }
            }
        }
    }
}

template <class Geometry, class S>
void launch_fp8_a16_wide_gemv(const Tensor& x, const Weight& weight, Tensor& out,
                              cudaStream_t stream) {
    if (x.ne[1] < 1 || x.ne[1] > S::kTileTokens) {
        throw std::invalid_argument("FP8 wide GEMV: token count outside the tile");
    }
    constexpr auto kernel = fp8_a16_wide_gemv_kernel<Geometry, S>;
    if constexpr (S::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, S::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    kernel<<<Geometry::kOutputRows / S::kRows, S::kThreads, S::kSharedBytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
