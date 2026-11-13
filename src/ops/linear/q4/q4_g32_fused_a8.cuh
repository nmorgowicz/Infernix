#pragma once

// Fused W4A8 prefill: the q4 rows are rounded to row-scaled E4M3 inside the GEMM, in shared
// memory, instead of being written to a workspace copy first. The bytes and the row multipliers
// are the ones q4_bake_e4m3_kernel produces, so the result matches the baked route exactly.
//
// Warp roles per CTA: one TMA warp loads the activation tile and the packed q4 codes of a stage,
// kConverterWarps round the codes into the stage's swizzled E4M3 tile, and the FP8 consumer warps
// run the unchanged FP8 MMA stage and epilogue on it.

#include "ops/common/mbarrier.cuh"
#include "ops/common/q4_scale_mode.h"
#include "ops/linear/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace infernix::ops::detail {

// Row multiplier of the bake contract: bf16_ru(8 * max|scale| / 448), 1 for an all-zero row.
// One warp per row, eight scales per load (the trunk rows hold a multiple of eight groups).
__global__ void q4_row_multiplier_kernel(const std::uint8_t* __restrict__ scales,
                                         __nv_bfloat16* __restrict__ row_scales, int rows, int groups) {
    const int row  = static_cast<int>(blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5));
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (row >= rows) { return; }
    float largest = 0.0f;
    const auto* scale_row = reinterpret_cast<const __half*>(scales) + static_cast<long long>(row) * groups;
    if (groups % 8 == 0) {
        const auto* vectors = reinterpret_cast<const uint4*>(scale_row);
        for (int v = lane; v < groups / 8; v += 32) {
            const uint4 packed = __ldg(vectors + v);
            const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (const unsigned word : words) {
                const float2 pair = __half22float2(*reinterpret_cast<const __half2*>(&word));
                largest           = fmaxf(largest, fmaxf(fabsf(pair.x), fabsf(pair.y)));
            }
        }
    } else {
        for (int g = lane; g < groups; g += 32) { largest = fmaxf(largest, fabsf(__half2float(scale_row[g]))); }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    }
    if (lane == 0) {
        const float m      = __fmul_rn(8.0f, largest);
        row_scales[row] = m > 0.0f ? __float2bfloat16_ru(__fdiv_rn(m, 448.0f)) : __ushort_as_bfloat16(0x3f80);
    }
}

inline void launch_q4_row_multipliers(const std::uint8_t* scales, __nv_bfloat16* row_scales, int rows, int k,
                                      int scale_mode, cudaStream_t stream) {
    constexpr int kWarps = 8;
    if (!q4_scale_mode_valid(scale_mode)) { throw std::invalid_argument("q4 row multipliers: invalid scale mode"); }
    q4_row_multiplier_kernel<<<(rows + kWarps - 1) / kWarps, kWarps * 32, 0, stream>>>(scales, row_scales, rows,
                                                                                         k >> (5 + scale_mode));
    CUDA_CHECK(cudaGetLastError());
}

// Packed codes as a 2D byte tensor of K/2 bytes per row. No swizzle: converter lanes read whole
// 16-byte groups in row order, which is already conflict-free.
inline CUtensorMap q4_code_tma_map(const std::uint8_t* codes, int rows, int k, int block_rows, int block_k) {
    CUtensorMap result{};
    const std::uint64_t dimensions[]{static_cast<std::uint64_t>(k / 2), static_cast<std::uint64_t>(rows)};
    const std::uint64_t strides[]{static_cast<std::uint64_t>(k / 2)};
    const std::uint32_t box[]{static_cast<std::uint32_t>(block_k / 2), static_cast<std::uint32_t>(block_rows)};
    const std::uint32_t steps[]{1, 1};
    const auto status = cuTensorMapEncodeTiled(
        &result, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, const_cast<std::uint8_t*>(codes), dimensions, strides, box,
        steps, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (status != CUDA_SUCCESS) {
        const char* name = nullptr;
        (void)cuGetErrorName(status, &name);
        throw std::runtime_error(std::string("q4 TMA descriptor: ") + (name ? name : "CUDA error"));
    }
    return result;
}

// Four nibbles (bits 0-15 of idx, element order) to four E4M3 bytes through the 16-entry table
// lut[0..3]. The low three bits pick within each half of the table, bit 3 picks the half.
__device__ __forceinline__ unsigned q4_lut_expand4(unsigned idx, const unsigned (&lut)[4]) {
    const unsigned select = idx & 0x7777u;
    const unsigned low    = __byte_perm(lut[0], lut[1], select);
    const unsigned high   = __byte_perm(lut[2], lut[3], select);
    return __byte_perm(low, high, 0x3210u | ((idx >> 1) & 0x4444u));
}

// One group of 32 codes to 32 E4M3 bytes: e4m3_rn_satfinite(code * mult), the same FP32 product
// and conversion as q4_bake_e4m3_kernel, so a table over the 16 code values gives the same bytes.
__device__ __forceinline__ void q4_round_group(uint4 packed, float mult, uint4& first, uint4& second) {
    unsigned lut[4];
#pragma unroll
    for (int part = 0; part < 4; ++part) {
        unsigned word = 0;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int a = part * 4 + pair * 2;
            const float lo = static_cast<float>(a < 8 ? a : a - 16);
            const float hi = static_cast<float>(a + 1 < 8 ? a + 1 : a + 1 - 16);
            const __nv_fp8x2_storage_t bytes = __nv_cvt_float2_to_fp8x2(
                make_float2(__fmul_rn(lo, mult), __fmul_rn(hi, mult)), __NV_SATFINITE, __NV_E4M3);
            word |= static_cast<unsigned>(bytes) << (16 * pair);
        }
        lut[part] = word;
    }
    first  = make_uint4(q4_lut_expand4(packed.x, lut), q4_lut_expand4(packed.x >> 16, lut),
                        q4_lut_expand4(packed.y, lut), q4_lut_expand4(packed.y >> 16, lut));
    second = make_uint4(q4_lut_expand4(packed.z, lut), q4_lut_expand4(packed.z >> 16, lut),
                        q4_lut_expand4(packed.w, lut), q4_lut_expand4(packed.w >> 16, lut));
}

// LoadStages of TMA tiles (activations and packed codes) feed RoundStages of rounded E4M3 tiles.
template <class Schedule, int LoadStages, int RoundStages, int ConverterWarps>
struct Q4FusedLayout {
    static constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows, BK = Schedule::kBlockK;
    static constexpr int L = LoadStages, R = RoundStages;
    static constexpr int kActivationBytes = BT * BK;
    static constexpr int kWeightBytes     = BR * BK;
    static constexpr int kCodeBytes       = BR * BK / 2;
    static constexpr int kCodeOffset      = L * kActivationBytes;
    static constexpr int kWeightOffset    = kCodeOffset + L * kCodeBytes;
    static constexpr int kStorageBytes    = kWeightOffset + R * kWeightBytes;
    static constexpr int kEpilogueBytes   = BT * (BR + 8) * 2;
    static constexpr int kBarrierOffset =
        ((kStorageBytes > kEpilogueBytes ? kStorageBytes : kEpilogueBytes) + 127) / 128 * 128;
    static constexpr int kSharedBytes      = kBarrierOffset + 2 * (L + R) * 8;
    static constexpr int kProducerThreads  = 32;
    static constexpr int kConverterThreads = ConverterWarps * 32;
    static constexpr int kThreads          = kProducerThreads + kConverterThreads + Schedule::kConsumerWarps * 32;
    // Groups per stage, spread over the converter lanes in row order; a lane keeps its group slot.
    static constexpr int kGroupsPerRow = BK / 32;
    static constexpr int kItems        = BR * kGroupsPerRow / kConverterThreads;
    static_assert(BK == 128, "the converter writes the 128-byte TMA swizzle");
    static_assert(kConverterThreads % kGroupsPerRow == 0 && (BR * kGroupsPerRow) % kConverterThreads == 0);
    static_assert(kActivationBytes % 1024 == 0 && kCodeBytes % 1024 == 0 && kWeightBytes % 1024 == 0);
    static_assert(kSharedBytes <= 99 * 1024);
};

template <class Schedule, int LoadStages, int RoundStages, int ConverterWarps, bool FullTokens, bool SplitK,
          int ScaleMode, bool InPlace = false>
__global__ void __launch_bounds__(Q4FusedLayout<Schedule, LoadStages, RoundStages, ConverterWarps>::kThreads, 1)
    q4_fused_a8_kernel(
#ifdef _WIN32
        // MSVC cannot pass the over-aligned (alignas(128)) CUtensorMap struct by value as a
        // __grid_constant__ parameter (C2719): on Windows the descriptors are pointer-passed from the
        // staging buffer (or, InPlace, read from the parameter words), as fp8_a8_tma_mma_kernel does.
        const __grid_constant__ TmaDescriptorParameter<Fp8TmaDescriptors, InPlace> descriptors_pointer,
#else
        const __grid_constant__ Fp8TmaDescriptors descriptors,
#endif
        Fp8A8Operands operands, const std::uint8_t* __restrict__ group_scales, LinearBf16Output output,
        int token_offset, int count, Fp8TmaSplitKPlan plan, float* partials) {
#ifdef _WIN32
    const Fp8TmaDescriptors& descriptors = *descriptors_pointer;
#endif
    using Layout = Q4FusedLayout<Schedule, LoadStages, RoundStages, ConverterWarps>;
    constexpr int BT = Layout::BT, BR = Layout::BR, BK = Layout::BK, L = Layout::L, R = Layout::R;
    const int k = Schedule::kStaticK ? Schedule::kStaticK : operands.k;
    int tile = static_cast<int>(blockIdx.x), partial = -1, k_begin = 0, tiles_k = k / BK;
    if constexpr (SplitK) {
        if (tile >= plan.full_tiles) {
            partial          = tile - plan.full_tiles;
            const int base   = plan.split_ctas / plan.tail_tiles;
            const int extra  = plan.split_ctas % plan.tail_tiles;
            const int longer = extra * (base + 1);
            const int tail   = partial < longer ? partial / (base + 1) : extra + (partial - longer) / base;
            const int part   = partial - plan.first_part(tail);
            const int parts  = plan.parts(tail);
            tile             = plan.full_tiles + tail;
            k_begin          = tiles_k * part / parts;
            tiles_k          = tiles_k * (part + 1) / parts - k_begin;
        }
    }
    const int row_tiles = operands.rows / BR, token_tiles = div_up(count, BT);
    int row_tile = 0, token_tile = 0;
    fp8_mma_tile_coordinates<Schedule>(tile, row_tiles, token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(1024) unsigned char q4_fused_shared[];
    auto* activation = q4_fused_shared;
    auto* weight     = q4_fused_shared + Layout::kWeightOffset;
    auto* code       = q4_fused_shared + Layout::kCodeOffset;
    // loaded/empty guard the TMA ring (empty needs the consumers, for the activations, and the
    // converters, for the codes); rounded/vacant guard the E4M3 ring.
    auto* loaded  = reinterpret_cast<std::uint64_t*>(q4_fused_shared + Layout::kBarrierOffset);
    auto* empty   = loaded + L;
    auto* rounded = empty + L;
    auto* vacant  = rounded + R;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < L; ++stage) {
            cta_mbarrier_init(loaded + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps + ConverterWarps);
        }
#pragma unroll
        for (int stage = 0; stage < R; ++stage) {
            cta_mbarrier_init(rounded + stage, ConverterWarps);
            cta_mbarrier_init(vacant + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    const int thread = static_cast<int>(threadIdx.x);
    if (thread < Layout::kProducerThreads) {
        if (thread == 0) {
#ifdef _WIN32
            // A staged tensor map was written through the generic proxy; each 128-byte map needs its own
            // acquire for the TMA (tensormap) proxy before its first use.
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % L;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / L) & 1U));
                cta_mbarrier_arrive_expect_tx(loaded + stage, Layout::kActivationBytes + Layout::kCodeBytes);
                fp8_tma_load(activation + stage * Layout::kActivationBytes, &descriptors.activation,
                             (k_begin + kt) * BK, token_begin, loaded + stage);
                fp8_tma_load(code + stage * Layout::kCodeBytes, &descriptors.weight, (k_begin + kt) * (BK / 2),
                             row_begin, loaded + stage);
            }
        }
        return;
    }

    if (thread < Layout::kProducerThreads + Layout::kConverterThreads) {
        const int converter = thread - Layout::kProducerThreads;
        const int lane      = converter & 31;
        const int group     = converter % Layout::kGroupsPerRow;
        const int groups    = k >> 5;
        const unsigned short* scale_base = nullptr;
        if constexpr (ScaleMode == 0) {
            scale_base = reinterpret_cast<const unsigned short*>(group_scales) + static_cast<long long>(row_begin) * groups +
                         group;
        } else {
            scale_base = reinterpret_cast<const unsigned short*>(group_scales) +
                         static_cast<long long>(row_begin) * (groups >> ScaleMode);
        }
        int row[Layout::kItems];
        float inverse[Layout::kItems];
        // g32 and coarse modes prefetch the binary16 scale bits.
        unsigned short next[Layout::kItems];
        const int scale_groups = groups >> ScaleMode;
        // The g32 loads are the original expressions; a coarse scale index is the 32-group index
        // shifted down. A tile holds four 32-groups, so a coarse scale covers one tile (g128) or two
        // consecutive tiles (g256).
        const auto coarse_scale = [&](int r, int tile) -> unsigned short {
            return __ldg(scale_base + static_cast<long long>(r) * scale_groups + ((tile * 4 + group) >> ScaleMode));
        };
        (void)scale_groups;
        (void)coarse_scale;
#pragma unroll
        for (int item = 0; item < Layout::kItems; ++item) {
            row[item]     = (item * Layout::kConverterThreads + converter) / Layout::kGroupsPerRow;
            inverse[item] = __frcp_rn(__bfloat162float(operands.scales[row_begin + row[item]]));
            if constexpr (ScaleMode == 0) {
                next[item] =
                    tiles_k > 0 ? __ldg(scale_base + static_cast<long long>(row[item]) * groups + k_begin * 4) : 0;
            } else {
                next[item] = tiles_k > 0 ? coarse_scale(row[item], k_begin) : 0;
            }
        }
        for (int kt = 0; kt < tiles_k; ++kt) {
            const int stage = kt % L, slot = kt % R;
            unsigned short current[Layout::kItems];
#pragma unroll
            for (int item = 0; item < Layout::kItems; ++item) {
                current[item] = next[item];
                if (kt + 1 < tiles_k) {
                    if constexpr (ScaleMode == 0) {
                        next[item] = __ldg(scale_base + static_cast<long long>(row[item]) * groups + (k_begin + kt + 1) * 4);
                    } else {
                        next[item] = coarse_scale(row[item], k_begin + kt + 1);
                    }
                }
            }
            cta_mbarrier_wait(loaded + stage, (kt / L) & 1U);
            cta_mbarrier_wait(vacant + slot, 1U ^ ((kt / R) & 1U));
            const auto* codes = code + stage * Layout::kCodeBytes;
            auto* tile        = weight + slot * Layout::kWeightBytes;
#pragma unroll
            for (int item = 0; item < Layout::kItems; ++item) {
                const int r       = row[item];
                const uint4 packed = *reinterpret_cast<const uint4*>(codes + r * (BK / 2) + group * 16);
                const unsigned short scale_bits = current[item];
                const float mult = __fmul_rn(__half2float(__ushort_as_half(scale_bits)), inverse[item]);
                uint4 first, second;
                q4_round_group(packed, mult, first, second);
                const int swizzle = r & 7;
                *reinterpret_cast<uint4*>(tile + r * BK + (((2 * group) ^ swizzle) << 4))     = first;
                *reinterpret_cast<uint4*>(tile + r * BK + (((2 * group + 1) ^ swizzle) << 4)) = second;
            }
            // The converter's ld.shared reads of the code stage must be ordered before the TMA refill that
            // the empty arrive releases (mbarrier orders generic-proxy accesses only).
            fence_reads_before_async_refill();
            __syncwarp();
            if (lane == 0) {
                cta_mbarrier_arrive(rounded + slot);
                cta_mbarrier_arrive(empty + stage);
            }
        }
        return;
    }

    const int tid  = thread - Layout::kProducerThreads - Layout::kConverterThreads;
    const int warp = tid >> 5, lane = tid & 31;
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % L, slot = kt % R;
        cta_mbarrier_wait(loaded + stage, (kt / L) & 1U);
        cta_mbarrier_wait(rounded + slot, (kt / R) & 1U);
        fp8_mma_compute_stage<Schedule>(activation + stage * Layout::kActivationBytes,
                                        weight + slot * Layout::kWeightBytes, accumulators, warp, lane);
        // The ldmatrix reads of the activation stage, likewise, before the TMA refill.
        fence_reads_before_async_refill();
        __syncwarp();
        if (lane == 0) {
            cta_mbarrier_arrive(empty + stage);
            cta_mbarrier_arrive(vacant + slot);
        }
    }
    if constexpr (SplitK) {
        if (partial >= 0) {
#pragma unroll
            for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
                const int token = warp / Schedule::kWarpsRows * Schedule::kWarpTokens + mt * 16 + lane / 4;
#pragma unroll
                for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                    const int r = warp % Schedule::kWarpsRows * Schedule::kWarpRows + mr * 8 + 2 * (lane % 4);
                    float* destination = partials + std::size_t(partial) * BT * BR + token * BR + r;
                    *reinterpret_cast<float2*>(destination) =
                        make_float2(accumulators[mt][mr][0], accumulators[mt][mr][1]);
                    *reinterpret_cast<float2*>(destination + 8 * BR) =
                        make_float2(accumulators[mt][mr][2], accumulators[mt][mr][3]);
                }
            }
            return;
        }
    }
    // Only the consumer warps are left; the epilogue reuses the staged storage.
    asm volatile("bar.sync 1, %0;" ::"n"(Schedule::kConsumerWarps * 32));
    fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8IdentityRows{}, q4_fused_shared,
                                              accumulators, operands.x_scales, operands.scales, row_begin,
                                              token_begin, operands.rows, token_offset + count, warp, lane);
}

// activations/activation_scales: the A8 codes of x. row_scales: the bake multipliers, written by
// launch_q4_row_multipliers first. group_scales: the weight's scale plane in scale mode ScaleMode
// (q4_scale_mode.h): one binary16 scale per 32 << ScaleMode columns.
// partials: Schedule::kPartialBytes when the schedule splits K.
template <class Schedule, int LoadStages, int RoundStages, int ConverterWarps, int ScaleMode>
void launch_q4_fused_a8(const std::uint8_t* activations, const float* activation_scales, const std::uint8_t* codes,
                        const std::uint8_t* group_scales, const __nv_bfloat16* row_scales, __nv_bfloat16* out,
                        float* partials, int rows, int k, int tokens, cudaStream_t stream) {
    using Layout = Q4FusedLayout<Schedule, LoadStages, RoundStages, ConverterWarps>;
    if (rows % Layout::BR != 0 || k % Layout::BK != 0 || k % q4_scale_span(ScaleMode) != 0) {
        throw std::invalid_argument("q4 fused A8 requires complete row and K tiles and K divisible by the scale group");
    }
    Fp8A8Operands operands{activations, activation_scales, codes, row_scales, rows, k, tokens};
    // Descriptors are launch-owned values, copied into kernel parameters during Graph capture.
    const Fp8TmaDescriptors descriptors{fp8_tma_map(activations, tokens, k, Layout::BT, Layout::BK),
                                        q4_code_tma_map(codes, rows, k, Layout::BR, Layout::BK)};
    LinearBf16Output output{out, rows};
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(tokens, Layout::BT, [&](int offset, int count) {
            const int blocks = rows / Layout::BR * div_up(count, Layout::BT);
            const auto plan  = fp8_tma_split_k_plan<Schedule>(blocks, k);
            const auto launch = [&]<bool Full, bool Split>() {
                constexpr auto kernel = q4_fused_a8_kernel<Schedule, LoadStages, RoundStages, ConverterWarps, Full,
                                                           Split, ScaleMode, InPlace>;
                static const cudaError_t status =
                    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Layout::kSharedBytes);
                CUDA_CHECK(status);
                const int grid = Split ? plan.full_tiles + plan.split_ctas : blocks;
                kernel<<<grid, Layout::kThreads, Layout::kSharedBytes, stream>>>(
                    parameter, operands, group_scales, output, offset, count, plan, partials);
                CUDA_CHECK(cudaGetLastError());
                if constexpr (Split) {
                    fp8_a8_tma_split_k_reduce<Schedule, LinearBf16Output, LinearIdentityEpilogue>
                        <<<plan.tail_tiles * Schedule::kReductionBlocks, 256, 0, stream>>>(
                            operands, partials, output, LinearIdentityEpilogue{}, Fp8IdentityRows{}, plan, offset,
                            count);
                    CUDA_CHECK(cudaGetLastError());
                }
            };
            if constexpr (Schedule::kSplitWaveCtas > 0) {
                if (plan.split_ctas) {
                    if (!partials) { throw std::invalid_argument("q4 fused A8 split-K requires partials"); }
                    if (count % Layout::BT == 0) launch.template operator()<true, true>();
                    else launch.template operator()<false, true>();
                    return;
                }
            }
            if (count % Layout::BT == 0) launch.template operator()<true, false>();
            else launch.template operator()<false, false>();
        });
    };
#ifdef _WIN32
    // Staged once: every token-slice launch reads the same copy, in stream order.
    with_tma_descriptor_parameter(fp8_tma_descriptor_staging(), descriptors, stream, launch_slices);
#else
    launch_slices(std::false_type{}, descriptors);
#endif
}

} // namespace infernix::ops::detail
