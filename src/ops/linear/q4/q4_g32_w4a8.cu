#include "ops/linear/q4/q4_g32_w4a8.cuh"

#include "core/layout.h"
#include "core/temporary_switches.h"
#include "ops/common/math.h"
#include "ops/common/q4_scale_mode.h"
#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/common/output.cuh"
#include "ops/linear/fp8/fp8_a8_mma_common.cuh"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/fp8/fp8_a8_tma_mma.cuh"
#include "ops/linear/fp8/fp8_shared.cuh"
#include "ops/linear/fp8/shapes/gemma4_a8.h"
#include "ops/linear/q4/q4_g32_fused_a8.cuh"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kW4A8Tokens = 32;
constexpr int kW4A8Rows   = 64;
constexpr int kW4A8K      = 32;
constexpr int kW4A8Warps  = 8;

__device__ __forceinline__ unsigned char q4_nibble_e4m3(unsigned nibble) {
    const int code = (static_cast<int>(nibble) ^ 8) - 8;
    return static_cast<unsigned char>(
        __nv_cvt_float_to_fp8(static_cast<float>(code), __NV_SATFINITE, __NV_E4M3));
}

template <int kScaleMode>
__global__ void q4g32_w4a8_kernel(const std::uint8_t* __restrict__ act,
                                  const float* __restrict__ act_scales,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ weight_scales,
                                  __nv_bfloat16* __restrict__ out, int rows, int k, int tokens) {
    constexpr int kGroup = 32;
    const int groups = k / kGroup;
    const int token0 = static_cast<int>(blockIdx.y) * kW4A8Tokens;
    const int row0   = static_cast<int>(blockIdx.x) * kW4A8Rows;
    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int warp_token = warp / 4;
    const int warp_row   = warp % 4;

    __shared__ __align__(16) std::uint8_t act_smem[kW4A8Tokens * kW4A8K];
    __shared__ __align__(16) std::uint8_t weight_smem[kW4A8Rows * kW4A8K];
    __shared__ float row_scale[kW4A8Rows];

    float acc[2][4] = {};
    const int mma_row_count = 2;

    for (int group = 0; group < groups; ++group) {
        for (int item = threadIdx.x; item < kW4A8Tokens * kW4A8K; item += blockDim.x) {
            const int token = item / kW4A8K;
            const int elem  = item - token * kW4A8K;
            const int src_token = token0 + token;
            act_smem[item] =
                src_token < tokens
                    ? act[static_cast<long long>(src_token) * k + group * kGroup + elem]
                    : 0;
        }
        for (int item = threadIdx.x; item < kW4A8Rows * 16; item += blockDim.x) {
            const int row  = item / 16;
            const int byte = item - row * 16;
            const int src_row = row0 + row;
            std::uint8_t packed = 0;
            if (src_row < rows) {
                packed = codes[(static_cast<long long>(src_row) * groups + group) * 16 + byte];
            }
            weight_smem[row * kW4A8K + byte * 2]     = q4_nibble_e4m3(packed & 0x0fu);
            weight_smem[row * kW4A8K + byte * 2 + 1] = q4_nibble_e4m3(packed >> 4);
        }
        for (int row = threadIdx.x; row < kW4A8Rows; row += blockDim.x) {
            const int src_row = row0 + row;
            float scale = 0.0f;
            if (src_row < rows) {
                const std::uint16_t bits = *reinterpret_cast<const std::uint16_t*>(
                    weight_scales +
                    (static_cast<long long>(src_row) * (groups >> kScaleMode) + (group >> kScaleMode)) * 2);
                scale = __half2float(__ushort_as_half(bits));
            }
            row_scale[row] = scale;
        }
        __syncthreads();

#pragma unroll
        for (int mma_row = 0; mma_row < mma_row_count; ++mma_row) {
            const int a_matrix     = lane >> 3;
            const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
            const int a_col        = (a_matrix >> 1) * 16;
            const int b_row        = lane & 7;
            const int b_col        = ((lane >> 3) & 1) * 16;
            const int act_row = warp_token * 16 + a_row_offset;
            const int w_row   = warp_row * 16 + mma_row * 8 + b_row;
            unsigned a_frag[4];
            unsigned b_frag[2];
            ldmatrix_x4(a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                        smem_addr(act_smem + act_row * kW4A8K + a_col));
            ldmatrix_x2(b_frag[0], b_frag[1], smem_addr(weight_smem + w_row * kW4A8K + b_col));
            float step[4] = {};
            mma_fp8_e4m3(step[0], step[1], step[2], step[3], a_frag[0], a_frag[1], a_frag[2],
                         a_frag[3], b_frag[0], b_frag[1]);
            const int scale_row = warp_row * 16 + mma_row * 8 + 2 * (lane & 3);
            const float s0 = row_scale[scale_row];
            const float s1 = row_scale[scale_row + 1];
            acc[mma_row][0] += step[0] * s0;
            acc[mma_row][1] += step[1] * s1;
            acc[mma_row][2] += step[2] * s0;
            acc[mma_row][3] += step[3] * s1;
        }
        __syncthreads();
    }

    const int token_lane = lane >> 2;
    const int row_lane   = 2 * (lane & 3);
#pragma unroll
    for (int mma_row = 0; mma_row < mma_row_count; ++mma_row) {
        const int local_row = warp_row * 16 + mma_row * 8 + row_lane;
        const int src_row0  = row0 + local_row;
        const int src_row1  = src_row0 + 1;
        const int src_token0 = token0 + warp_token * 16 + token_lane;
        const int src_token1 = src_token0 + 8;
        const float as0 = src_token0 < tokens ? act_scales[src_token0] : 0.0f;
        const float as1 = src_token1 < tokens ? act_scales[src_token1] : 0.0f;
        auto store = [&](int row, int token, float value) {
            if (row < rows && token < tokens) {
                out[row + static_cast<long long>(token) * rows] = __float2bfloat16(value);
            }
        };
        store(src_row0, src_token0, acc[mma_row][0] * as0);
        store(src_row1, src_token0, acc[mma_row][1] * as0);
        store(src_row0, src_token1, acc[mma_row][2] * as1);
        store(src_row1, src_token1, acc[mma_row][3] * as1);
    }
}

} // namespace

namespace {

// Expanded E4M3 weights are applied with the group scale inside the MMA, so no split-K partials.
constexpr std::size_t kW4A8PartialBytes = 0;

__constant__ unsigned short q4_expand_lut[256] = {
    0x0000, 0x0038, 0x0040, 0x0044, 0x0048, 0x004A, 0x004C, 0x004E, 0x00D0, 0x00CE, 0x00CC, 0x00CA, 0x00C8, 0x00C4, 0x00C0, 0x00B8,
    0x3800, 0x3838, 0x3840, 0x3844, 0x3848, 0x384A, 0x384C, 0x384E, 0x38D0, 0x38CE, 0x38CC, 0x38CA, 0x38C8, 0x38C4, 0x38C0, 0x38B8,
    0x4000, 0x4038, 0x4040, 0x4044, 0x4048, 0x404A, 0x404C, 0x404E, 0x40D0, 0x40CE, 0x40CC, 0x40CA, 0x40C8, 0x40C4, 0x40C0, 0x40B8,
    0x4400, 0x4438, 0x4440, 0x4444, 0x4448, 0x444A, 0x444C, 0x444E, 0x44D0, 0x44CE, 0x44CC, 0x44CA, 0x44C8, 0x44C4, 0x44C0, 0x44B8,
    0x4800, 0x4838, 0x4840, 0x4844, 0x4848, 0x484A, 0x484C, 0x484E, 0x48D0, 0x48CE, 0x48CC, 0x48CA, 0x48C8, 0x48C4, 0x48C0, 0x48B8,
    0x4A00, 0x4A38, 0x4A40, 0x4A44, 0x4A48, 0x4A4A, 0x4A4C, 0x4A4E, 0x4AD0, 0x4ACE, 0x4ACC, 0x4ACA, 0x4AC8, 0x4AC4, 0x4AC0, 0x4AB8,
    0x4C00, 0x4C38, 0x4C40, 0x4C44, 0x4C48, 0x4C4A, 0x4C4C, 0x4C4E, 0x4CD0, 0x4CCE, 0x4CCC, 0x4CCA, 0x4CC8, 0x4CC4, 0x4CC0, 0x4CB8,
    0x4E00, 0x4E38, 0x4E40, 0x4E44, 0x4E48, 0x4E4A, 0x4E4C, 0x4E4E, 0x4ED0, 0x4ECE, 0x4ECC, 0x4ECA, 0x4EC8, 0x4EC4, 0x4EC0, 0x4EB8,
    0xD000, 0xD038, 0xD040, 0xD044, 0xD048, 0xD04A, 0xD04C, 0xD04E, 0xD0D0, 0xD0CE, 0xD0CC, 0xD0CA, 0xD0C8, 0xD0C4, 0xD0C0, 0xD0B8,
    0xCE00, 0xCE38, 0xCE40, 0xCE44, 0xCE48, 0xCE4A, 0xCE4C, 0xCE4E, 0xCED0, 0xCECE, 0xCECC, 0xCECA, 0xCEC8, 0xCEC4, 0xCEC0, 0xCEB8,
    0xCC00, 0xCC38, 0xCC40, 0xCC44, 0xCC48, 0xCC4A, 0xCC4C, 0xCC4E, 0xCCD0, 0xCCCE, 0xCCCC, 0xCCCA, 0xCCC8, 0xCCC4, 0xCCC0, 0xCCB8,
    0xCA00, 0xCA38, 0xCA40, 0xCA44, 0xCA48, 0xCA4A, 0xCA4C, 0xCA4E, 0xCAD0, 0xCACE, 0xCACC, 0xCACA, 0xCAC8, 0xCAC4, 0xCAC0, 0xCAB8,
    0xC800, 0xC838, 0xC840, 0xC844, 0xC848, 0xC84A, 0xC84C, 0xC84E, 0xC8D0, 0xC8CE, 0xC8CC, 0xC8CA, 0xC8C8, 0xC8C4, 0xC8C0, 0xC8B8,
    0xC400, 0xC438, 0xC440, 0xC444, 0xC448, 0xC44A, 0xC44C, 0xC44E, 0xC4D0, 0xC4CE, 0xC4CC, 0xC4CA, 0xC4C8, 0xC4C4, 0xC4C0, 0xC4B8,
    0xC000, 0xC038, 0xC040, 0xC044, 0xC048, 0xC04A, 0xC04C, 0xC04E, 0xC0D0, 0xC0CE, 0xC0CC, 0xC0CA, 0xC0C8, 0xC0C4, 0xC0C0, 0xC0B8,
    0xB800, 0xB838, 0xB840, 0xB844, 0xB848, 0xB84A, 0xB84C, 0xB84E, 0xB8D0, 0xB8CE, 0xB8CC, 0xB8CA, 0xB8C8, 0xB8C4, 0xB8C0, 0xB8B8};

__global__ void q4_expand_e4m3_kernel(const std::uint8_t* __restrict__ codes, std::uint8_t* __restrict__ expanded,
                                      int rows, int k) {
    __shared__ unsigned short lut[256];
    lut[threadIdx.x] = q4_expand_lut[threadIdx.x];
    __syncthreads();
    const int groups = k >> 5;
    const int index  = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= rows * groups) { return; }
    const int row   = index / groups;
    const int group = index - row * groups;
    const uint4 packed = load_vec<uint4>(codes + (static_cast<long long>(row) * groups + group) * 16);
    auto pair          = [&](unsigned word, int byte) -> unsigned {
        const unsigned b0 = lut[(word >> (8 * byte)) & 0xffu];
        const unsigned b1 = lut[(word >> (8 * (byte + 1))) & 0xffu];
        return b0 | (b1 << 16);
    };
    uint4 low;
    uint4 high;
    low.x              = pair(packed.x, 0);
    low.y              = pair(packed.x, 2);
    low.z              = pair(packed.y, 0);
    low.w              = pair(packed.y, 2);
    high.x             = pair(packed.z, 0);
    high.y             = pair(packed.z, 2);
    high.z             = pair(packed.w, 0);
    high.w             = pair(packed.w, 2);
    auto* destination = expanded + static_cast<long long>(row) * k + group * 32;
    store_vec(destination, low);
    store_vec(destination + 16, high);
}

// Per-32 transposed scales [group][row]. `groups` counts 32-column groups; the source holds one
// scale per 32 << kScaleMode columns (expanded to every 32-group it covers).
template <int kScaleMode>
__global__ void q4_transpose_scales(const std::uint8_t* __restrict__ src, std::uint8_t* __restrict__ dst, int rows,
                                   int groups) {
    __shared__ __align__(16) std::uint16_t tile[32][33];
    const int group0 = static_cast<int>(blockIdx.x) * 32;
    const int row0   = static_cast<int>(blockIdx.y) * 32;
    const int tx     = static_cast<int>(threadIdx.x) & 31;
    const int ty     = static_cast<int>(threadIdx.x) >> 5;
    const auto* in   = reinterpret_cast<const std::uint16_t*>(src);
    auto* out        = reinterpret_cast<std::uint16_t*>(dst);
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        const int row   = row0 + ty + rep * 8;
        const int group = group0 + tx;
        std::uint16_t value = 0;
        if (row < rows && group < groups) {
            value = in[static_cast<long long>(row) * (groups >> kScaleMode) + (group >> kScaleMode)];
        }
        tile[ty + rep * 8][tx] = value;
    }
    __syncthreads();
#pragma unroll
    for (int rep = 0; rep < 4; ++rep) {
        const int group = group0 + ty + rep * 8;
        const int row   = row0 + tx;
        if (row < rows && group < groups) {
            out[static_cast<long long>(group) * rows + row] = tile[tx][ty + rep * 8];
        }
    }
}

__global__ void q4_fill_bf16_ones(__nv_bfloat16* values, int rows) {
    const int row = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (row < rows) { values[row] = __ushort_as_bfloat16(0x3f80); }
}

// Baked prefill weights, one CTA per row. The row multiplier is the BF16 value rounded up from
// 8 * max(scale) / 448, so every code times its scale fits E4M3. Each weight becomes
// e4m3_rn_satfinite(code * (scale * rcp_rn(row))), every product rounded once in FP32. The host
// test reproduces these bytes; change both together.
constexpr int kQ4BakeThreads = 256;

template <int kScaleMode>
__global__ void __launch_bounds__(kQ4BakeThreads)
    q4_bake_e4m3_kernel(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
                        std::uint8_t* __restrict__ baked, __nv_bfloat16* __restrict__ row_scales, int k) {
    const int row    = static_cast<int>(blockIdx.x);
    const int groups = k >> 5;
    const int scale_groups = groups >> kScaleMode;
    const __half* scale_row = reinterpret_cast<const __half*>(scales) + static_cast<long long>(row) * scale_groups;
    float largest = 0.0f;
    for (int g = static_cast<int>(threadIdx.x); g < scale_groups; g += kQ4BakeThreads) {
        largest = fmaxf(largest, fabsf(__half2float(scale_row[g])));
    }
    const auto group_scale = [&](int g) -> float { return __half2float(scale_row[g >> kScaleMode]); };
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffu, largest, offset));
    }
    __shared__ float warp_max[kQ4BakeThreads / 32];
    __shared__ float inverse;
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x >> 5] = largest; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = warp_max[0];
#pragma unroll
        for (int w = 1; w < kQ4BakeThreads / 32; ++w) { m = fmaxf(m, warp_max[w]); }
        m = __fmul_rn(8.0f, m);
        const __nv_bfloat16 multiplier =
            m > 0.0f ? __float2bfloat16_ru(__fdiv_rn(m, 448.0f)) : __ushort_as_bfloat16(0x3f80);
        row_scales[row] = multiplier;
        inverse         = __frcp_rn(__bfloat162float(multiplier));
    }
    __syncthreads();
    const float inv = inverse;
    for (int g = static_cast<int>(threadIdx.x); g < groups; g += kQ4BakeThreads) {
        const float mult = __fmul_rn(group_scale(g), inv);
        const uint4 packed =
            load_vec<uint4>(codes + (static_cast<long long>(row) * groups + g) * 16);
        const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
        unsigned out_words[8];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                unsigned result = 0;
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const unsigned byte = (words[w] >> (16 * half + 8 * b)) & 0xffu;
                    const float lo = static_cast<float>((static_cast<int>(byte & 0xfu) ^ 8) - 8);
                    const float hi = static_cast<float>((static_cast<int>(byte >> 4) ^ 8) - 8);
                    const __nv_fp8x2_storage_t pair = __nv_cvt_float2_to_fp8x2(
                        make_float2(__fmul_rn(lo, mult), __fmul_rn(hi, mult)), __NV_SATFINITE, __NV_E4M3);
                    result |= static_cast<unsigned>(pair) << (16 * b);
                }
                out_words[w * 2 + half] = result;
            }
        }
        auto* destination = baked + static_cast<long long>(row) * k + g * 32;
        store_vec(destination, make_uint4(out_words[0], out_words[1], out_words[2], out_words[3]));
        store_vec(destination + 16, make_uint4(out_words[4], out_words[5], out_words[6], out_words[7]));
    }
}

// The q4 scale mode (q4_scale_mode.h) as a template argument: 0 (g32), 2 (g128), 3 (g256).
template <class F>
void with_scale_mode(int scale_mode, F&& f) {
    switch (scale_mode) {
    case 0: f.template operator()<0>(); return;
    case 2: f.template operator()<2>(); return;
    case 3: f.template operator()<3>(); return;
    default: throw std::invalid_argument("q4 W4A8: unsupported scale mode");
    }
}

// The six Gemma 4 trunk geometries, routed to the FP8 A8 GEMM on baked weights.
template <class F>
void with_gemma4_trunk(int rows, int k, F&& f) {
    if (rows == 16384 && k == 5376) { f.template operator()<16384, 5376>(); }
    if (rows == 18432 && k == 5376) { f.template operator()<18432, 5376>(); }
    if (rows == 43008 && k == 5376) { f.template operator()<43008, 5376>(); }
    if (rows == 5376 && k == 8192) { f.template operator()<5376, 8192>(); }
    if (rows == 5376 && k == 16384) { f.template operator()<5376, 16384>(); }
    if (rows == 5376 && k == 21504) { f.template operator()<5376, 21504>(); }
}

// The trunk projections from 9 tokens take an A8 route on row-scaled E4M3 weights: the fused
// kernel up to kQ4FusedMaxTokens, above that the bake pass plus the FP8 A8 GEMM. Model prefill
// on the RTX PRO 6000 (Gemma 4 31B QAT): 9-16 tokens 19 ms against 24 ms on two 8-token
// tensor-core passes; 384 tokens 89.5 ms fused against 90.9 ms baked; 512 tokens 116.8 against
// 108.3. Below 128 the FP8 tile choice still picks the split-K bulk tile for some shapes, and
// its partials helper sizes nothing there, so the baked partials are sized as for 128 tokens.
std::size_t q4_bake_partial_bytes(std::int32_t tokens) {
    return gemma4_fp8::partial_capacity_bytes(std::max(tokens, 128));
}

// Fused route: 128-row tiles of 32 or 64 tokens, four load stages feeding two rounded stages,
// four converter warps, split-K over one wave. The MMA work is per tile, so the 32-token tile
// halves it where the prompt fits.
using Q4FusedSmall  = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<32, 128, 128, 1, 8, 2, 1>, 188, 8, 8>;
using Q4FusedNarrow = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>, 188, 8, 8>;
constexpr int kQ4FusedLoadStages  = 4;
constexpr int kQ4FusedRoundStages = 2;
constexpr int kQ4FusedConverters  = 4;
constexpr std::int32_t kQ4FusedSmallTokens = 32;
constexpr std::size_t kQ4FusedPartialBytes = std::max(Q4FusedSmall::kPartialBytes, Q4FusedNarrow::kPartialBytes);

void q4_fused(const Q4LinearOperands& operands, __nv_bfloat16* out, const std::uint8_t* codes,
              const float* code_scales, float* partials, WorkspaceArena& workspace, cudaStream_t stream) {
    const __nv_bfloat16* row_scales = operands.row_scales;
    if (row_scales == nullptr) {
        auto* derived = static_cast<__nv_bfloat16*>(
            workspace.alloc_bytes(static_cast<std::size_t>(operands.rows) * sizeof(__nv_bfloat16), 256).data);
        launch_q4_row_multipliers(operands.scales, derived, operands.rows, operands.k, operands.scale_mode, stream);
        row_scales = derived;
    }
    const auto launch = [&]<class Schedule>() {
        with_scale_mode(operands.scale_mode, [&]<int S>() {
            launch_q4_fused_a8<Schedule, kQ4FusedLoadStages, kQ4FusedRoundStages, kQ4FusedConverters, S>(
                codes, code_scales, operands.codes, operands.scales, row_scales, out, partials, operands.rows,
                operands.k, operands.tokens, stream);
        });
    };
    if (operands.tokens <= kQ4FusedSmallTokens) {
        launch.template operator()<Q4FusedSmall>();
    } else {
        launch.template operator()<Q4FusedNarrow>();
    }
}

template <class Schedule>
__device__ void q4_mma_scaled_stage(const std::uint8_t* activation_stage, const std::uint8_t* weight_stage,
                                    const __half* tile_scale,
                                    float (&accumulators)[Schedule::kMmaTokens][Schedule::kMmaRows][4], int warp,
                                    int lane) {
    constexpr int BK       = Schedule::kBlockK;
    const int warp_token   = warp / Schedule::kWarpsRows;
    const int warp_row     = warp % Schedule::kWarpsRows;
    const int a_matrix     = lane >> 3;
    const int a_row_offset = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column     = (a_matrix >> 1) * 16;
    const int b_row_offset = lane & 7;
    const int b_column     = ((lane >> 3) & 1) * 16;
    const int owned_row    = 2 * (lane & 3);
    unsigned a_fragments[2][Schedule::kMmaTokens][4];
    unsigned b_fragments[2][Schedule::kMmaRows][2];
    auto load_fragments = [&](int k_step, int slot) {
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
            const int row          = warp_token * Schedule::kWarpTokens + mma_token * 16 + a_row_offset;
            const int logical      = k_step * 32 + a_column;
            const int physical     = fp8_mma_shared_byte<Schedule>(row, logical);
            ldmatrix_x4(a_fragments[slot][mma_token][0], a_fragments[slot][mma_token][1],
                        a_fragments[slot][mma_token][2], a_fragments[slot][mma_token][3],
                        smem_addr(activation_stage + row * BK + physical));
        }
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int row      = warp_row * Schedule::kWarpRows + mma_row * 8 + b_row_offset;
            const int logical  = k_step * 32 + b_column;
            const int physical = fp8_mma_shared_byte<Schedule>(row, logical);
            ldmatrix_x2(b_fragments[slot][mma_row][0], b_fragments[slot][mma_row][1],
                        smem_addr(weight_stage + row * BK + physical));
        }
    };
    load_fragments(0, 0);
#pragma unroll
    for (int k_step = 0; k_step < Schedule::kMmaK; ++k_step) {
        const int slot = k_step & 1;
        if (k_step + 1 < Schedule::kMmaK) { load_fragments(k_step + 1, slot ^ 1); }
        float top[Schedule::kMmaRows];
        float bottom[Schedule::kMmaRows];
#pragma unroll
        for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
            const int local = warp_row * Schedule::kWarpRows + mma_row * 8 + owned_row;
            top[mma_row]    = __half2float(tile_scale[k_step * Schedule::kBlockRows + local]);
            bottom[mma_row] = __half2float(tile_scale[k_step * Schedule::kBlockRows + local + 1]);
        }
#pragma unroll
        for (int mma_token = 0; mma_token < Schedule::kMmaTokens; ++mma_token) {
#pragma unroll
            for (int mma_row = 0; mma_row < Schedule::kMmaRows; ++mma_row) {
                float step0 = 0.0f, step1 = 0.0f, step2 = 0.0f, step3 = 0.0f;
                mma_fp8_e4m3(step0, step1, step2, step3, a_fragments[slot][mma_token][0],
                             a_fragments[slot][mma_token][1], a_fragments[slot][mma_token][2],
                             a_fragments[slot][mma_token][3], b_fragments[slot][mma_row][0],
                             b_fragments[slot][mma_row][1]);
                accumulators[mma_token][mma_row][0] += step0 * top[mma_row];
                accumulators[mma_token][mma_row][1] += step1 * bottom[mma_row];
                accumulators[mma_token][mma_row][2] += step2 * top[mma_row];
                accumulators[mma_token][mma_row][3] += step3 * bottom[mma_row];
            }
        }
    }
}

template <class Schedule, bool FullTokens, bool SplitK, bool InPlace = false>
__global__ void __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
    q4_w4a8_tma_kernel(
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
    constexpr int BT = Schedule::kBlockTokens, BR = Schedule::kBlockRows;
    constexpr int BK = Schedule::kBlockK, S = Schedule::kStages;
    const int k = operands.k;
    int tile = static_cast<int>(blockIdx.x), partial = -1, k_begin = 0, tiles_k = k / BK;
    if constexpr (SplitK) {
        if (tile >= plan.full_tiles) {
            partial          = tile - plan.full_tiles;
            const int base   = plan.split_ctas / plan.tail_tiles;
            const int extra  = plan.split_ctas % plan.tail_tiles;
            const int longer = extra * (base + 1);
            const int tail =
                partial < longer ? partial / (base + 1) : extra + (partial - longer) / base;
            const int part  = partial - plan.first_part(tail);
            const int parts = plan.parts(tail);
            tile            = plan.full_tiles + tail;
            k_begin         = tiles_k * part / parts;
            tiles_k         = tiles_k * (part + 1) / parts - k_begin;
        }
    }
    const int row_tiles = operands.rows / BR, token_tiles = div_up(count, BT);
    int row_tile = 0, token_tile = 0;
    fp8_mma_tile_coordinates<Schedule>(tile, row_tiles, token_tiles, row_tile, token_tile);
    const int row_begin   = row_tile * BR;
    const int token_begin = token_offset + token_tile * BT;

    extern __shared__ __align__(128) unsigned char q4_tma_shared[];
    auto* activation = q4_tma_shared;
    auto* weight     = activation + S * BT * BK;
    auto* full       = reinterpret_cast<std::uint64_t*>(q4_tma_shared + fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue>);
    auto* empty      = full + S;
    if (threadIdx.x == 0) {
#pragma unroll
        for (int stage = 0; stage < S; ++stage) {
            cta_mbarrier_init(full + stage, 1);
            cta_mbarrier_init(empty + stage, Schedule::kConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    if (threadIdx.x < Schedule::kProducerThreads) {
        if (threadIdx.x == 0) {
#ifdef _WIN32
            // A staged tensor map was written through the generic proxy; each 128-byte map needs its own
            // acquire for the TMA (tensormap) proxy before its first use.
            acquire_staged_tensor_map<InPlace>(&descriptors.activation);
            acquire_staged_tensor_map<InPlace>(&descriptors.weight);
#endif
            for (int kt = 0; kt < tiles_k; ++kt) {
                const int stage = kt % S;
                cta_mbarrier_wait(empty + stage, 1U ^ ((kt / S) & 1U));
                cta_mbarrier_arrive_expect_tx(full + stage, (BT + BR) * BK);
                fp8_tma_load(activation + stage * BT * BK, &descriptors.activation, (k_begin + kt) * BK,
                             token_begin, full + stage);
                fp8_tma_load(weight + stage * BR * BK, &descriptors.weight, (k_begin + kt) * BK, row_begin,
                             full + stage);
            }
        }
        return;
    }

    const int tid  = static_cast<int>(threadIdx.x) - Schedule::kProducerThreads;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    constexpr int kConsumers = Schedule::kThreads - Schedule::kProducerThreads;
    constexpr int kGroups    = BK / 32;
    constexpr int kScaleTile = BR * kGroups;
    auto* tile_scale = reinterpret_cast<__half*>(
        q4_tma_shared + fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes);
    auto issue_scales = [&](int kt, int buf) {
        const int group_base = ((k_begin + kt) * BK) >> 5;
        auto* destination    = tile_scale + buf * kScaleTile;
        constexpr int kVec   = 8;
        for (int item = tid; item < kGroups * (BR / kVec); item += kConsumers) {
            const int group = item / (BR / kVec);
            const int chunk = item - group * (BR / kVec);
            cp_async<16, Cache::cg>(destination + group * BR + chunk * kVec,
                                    group_scales + ((static_cast<long long>(group_base + group) * operands.rows +
                                                     row_begin + chunk * kVec) *
                                                    2));
        }
        cp_commit();
    };
    float accumulators[Schedule::kMmaTokens][Schedule::kMmaRows][4] = {};
    if (tiles_k > 0) { issue_scales(0, 0); }
    for (int kt = 0; kt < tiles_k; ++kt) {
        const int stage = kt % S;
        cta_mbarrier_wait(full + stage, (kt / S) & 1U);
        cp_wait<0>();
        asm volatile("bar.sync 1, %0;" ::"n"(kConsumers));
        q4_mma_scaled_stage<Schedule>(activation + stage * BT * BK, weight + stage * BR * BK,
                                      tile_scale + (kt & 1) * kScaleTile, accumulators, warp, lane);
        if (kt + 1 < tiles_k) { issue_scales(kt + 1, (kt + 1) & 1); }
        __syncwarp();
        if (lane == 0) { cta_mbarrier_arrive(empty + stage); }
    }
    if constexpr (SplitK) {
        if (partial >= 0) {
#pragma unroll
            for (int mt = 0; mt < Schedule::kMmaTokens; ++mt) {
                const int token = warp / Schedule::kWarpsRows * Schedule::kWarpTokens + mt * 16 + lane / 4;
#pragma unroll
                for (int mr = 0; mr < Schedule::kMmaRows; ++mr) {
                    const int row = warp % Schedule::kWarpsRows * Schedule::kWarpRows + mr * 8 + 2 * (lane % 4);
                    float* destination = partials + std::size_t(partial) * BT * BR + token * BR + row;
                    *reinterpret_cast<float2*>(destination) =
                        make_float2(accumulators[mt][mr][0], accumulators[mt][mr][1]);
                    *reinterpret_cast<float2*>(destination + 8 * BR) =
                        make_float2(accumulators[mt][mr][2], accumulators[mt][mr][3]);
                }
            }
            return;
        }
    }
    __syncthreads();
    fp8_finish_mma_tile<Schedule, FullTokens>(output, LinearIdentityEpilogue{}, Fp8IdentityRows{}, q4_tma_shared,
                                              accumulators, operands.x_scales, operands.scales, row_begin, token_begin,
                                              operands.rows, token_offset + count, warp, lane);
}

template <class Schedule>
void launch_q4_w4a8_tma(const std::uint8_t* activations, const float* activation_scales,
                        const std::uint8_t* expanded, const std::uint8_t* group_scales,
                        const __nv_bfloat16* ones, __nv_bfloat16* out, float* partials, int rows, int k, int tokens,
                        cudaStream_t stream) {
    Fp8A8Operands operands{activations, activation_scales, expanded, ones, rows, k, tokens};
    constexpr int weight_span = Schedule::kBlockRows;
    const Fp8TmaDescriptors descriptors{fp8_tma_map(activations, tokens, k, Schedule::kBlockTokens, Schedule::kBlockK),
                                        fp8_tma_map(expanded, rows, k, weight_span, Schedule::kBlockK)};
    LinearBf16Output output{out, rows};
    const auto launch_slices = [&]<bool InPlace>(std::bool_constant<InPlace>, const auto& parameter) {
        for_each_token_slice(tokens, Schedule::kBlockTokens, [&](int offset, int count) {
            const int blocks = rows / Schedule::kBlockRows * div_up(count, Schedule::kBlockTokens);
            const auto plan  = fp8_tma_split_k_plan<Schedule>(blocks, k);
            const auto launch = [&]<bool Full, bool Split>() {
                constexpr auto kernel = q4_w4a8_tma_kernel<Schedule, Full, Split, InPlace>;
                constexpr int kScaleBytes =
                    2 * Schedule::kBlockRows * (Schedule::kBlockK / 32) * static_cast<int>(sizeof(__half));
                constexpr int kRaw =
                    fp8_tma_scratch_bytes<Schedule, LinearIdentityEpilogue> + Schedule::kBarrierBytes + kScaleBytes;
                constexpr int bytes = (kRaw + 127) & ~127;
                const int dynamic = fp8_prepare_shared<bytes, kernel, true>();
                const int grid    = Split ? plan.full_tiles + plan.split_ctas : blocks;
                kernel<<<grid, Schedule::kThreads, dynamic, stream>>>(parameter, operands, group_scales, output,
                                                                      offset, count, plan, partials);
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
                    if (count % Schedule::kBlockTokens == 0) launch.template operator()<true, true>();
                    else launch.template operator()<false, true>();
                    return;
                }
            }
            if (count % Schedule::kBlockTokens == 0) launch.template operator()<true, false>();
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

using Q4W4A8Narrow = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Q4W4A8Mid   = Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>;
using Q4W4A8Bulk  = Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>;

// Rounds the q4 rows to row-scaled E4M3 in the workspace and describes them as an FP8 weight.
Weight q4_bake(const Q4LinearOperands& operands, WorkspaceArena& workspace, cudaStream_t stream) {
    auto* baked = static_cast<std::uint8_t*>(
        workspace.alloc_bytes(static_cast<std::size_t>(operands.rows) * static_cast<std::size_t>(operands.k), 256)
            .data);
    auto* row_scales = static_cast<__nv_bfloat16*>(
        workspace.alloc_bytes(static_cast<std::size_t>(operands.rows) * sizeof(__nv_bfloat16), 256).data);
    with_scale_mode(operands.scale_mode, [&]<int S>() {
        q4_bake_e4m3_kernel<S><<<operands.rows, kQ4BakeThreads, 0, stream>>>(operands.codes, operands.scales, baked,
                                                                            row_scales, operands.k);
    });
    CUDA_CHECK(cudaGetLastError());
    Weight fp8{};
    fp8.qtype  = QType::FP8_E4M3FN_ROW_BF16;
    fp8.qdata  = baked;
    fp8.scales = row_scales;
    fp8.n      = operands.rows;
    fp8.k      = operands.k;
    return fp8;
}

} // namespace

bool q4_exact_prefill() {
    static const bool exact = temporary_switches::flag("INFERNIX_TMP_Q4_EXACT");
    return exact;
}

void q4_g32_row_multipliers(const std::uint8_t* scales, void* row_scales, std::int32_t rows, std::int32_t k,
                            std::int32_t scale_mode, cudaStream_t stream) {
    launch_q4_row_multipliers(scales, static_cast<__nv_bfloat16*>(row_scales), rows, k, scale_mode, stream);
}

bool q4_g32_takes_ready_codes(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    const Q4Route route = q4_g32_route(rows, k, tokens, LinearPolicy::AllowA8, q4_exact_prefill());
    return rows == 5376 && k == 21504 && (route == Q4Route::Fused || route == Q4Route::Baked);
}

void q4_g32_w4a8_ready(const Q4LinearOperands& operands, __nv_bfloat16* out, std::uint8_t* codes,
                       float* scales, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!q4_g32_takes_ready_codes(operands.rows, operands.k, operands.tokens)) {
        throw std::invalid_argument("q4_g32 ready: only the baked Gemma down projection takes ready codes");
    }
    const auto scope    = workspace.scope();
    if (operands.tokens <= kQ4FusedMaxTokens) {
        float* partials = static_cast<float*>(workspace.alloc_bytes(kQ4FusedPartialBytes, 256).data);
        q4_fused(operands, out, codes, scales, partials, workspace, stream);
        return;
    }
    const auto partial  = q4_bake_partial_bytes(operands.tokens);
    float* partials     = partial != 0 ? static_cast<float*>(workspace.alloc_bytes(partial, 256).data) : nullptr;
    const Weight fp8    = q4_bake(operands, workspace, stream);
    Tensor x(const_cast<__nv_bfloat16*>(operands.x), DType::BF16, {operands.k, operands.tokens});
    Tensor y(out, DType::BF16, {operands.rows, operands.tokens});
    gemma4_fp8::launch_down_ready(x, fp8, y, Fp8A8Workspace{codes, scales, partials}, stream);
}

std::size_t q4_g32_w4a8_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder exact;
    (void)allocate_fp8_a8_workspace(exact, tokens, k, kW4A8PartialBytes);
    (void)exact.alloc_bytes(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), 256);
    (void)exact.alloc_bytes(static_cast<std::size_t>(rows) * sizeof(__nv_bfloat16), 256);
    (void)exact.alloc_bytes(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k / 32) * sizeof(std::uint16_t),
                            256);
    std::size_t bytes = tokens >= kQ4ExactMinTokens ? exact.peak_bytes(1) : 0;
    if (tokens >= kQ4A8MinTokens && q4_trunk_shape(rows, k)) {
        WorkspaceLayoutBuilder fused;
        (void)allocate_fp8_a8_workspace(fused, tokens, k, kQ4FusedPartialBytes);
        (void)fused.alloc_bytes(static_cast<std::size_t>(rows) * sizeof(__nv_bfloat16), 256);
        bytes = std::max(bytes, fused.peak_bytes(1));
        if (tokens > kQ4FusedMaxTokens) {
            bytes = std::max(bytes, q4_g32_baked_workspace_bytes(rows, k, tokens));
        }
    }
    return bytes;
}

std::size_t q4_g32_baked_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens) {
    WorkspaceLayoutBuilder baked;
    (void)allocate_fp8_a8_workspace(baked, tokens, k, q4_bake_partial_bytes(tokens));
    (void)baked.alloc_bytes(static_cast<std::size_t>(rows) * static_cast<std::size_t>(k), 256);
    (void)baked.alloc_bytes(static_cast<std::size_t>(rows) * sizeof(__nv_bfloat16), 256);
    return baked.peak_bytes(1);
}

void q4_g32_w4a8_baked(const Q4LinearOperands& operands, __nv_bfloat16* out, WorkspaceArena& workspace,
                       cudaStream_t stream) {
    if (operands.tokens < kQ4A8MinTokens || !q4_trunk_shape(operands.rows, operands.k)) {
        throw std::invalid_argument("q4_g32 baked: only Gemma trunk shapes from 9 tokens");
    }
    const auto scope = workspace.scope();
    const Fp8A8Workspace scratch =
        allocate_fp8_a8_workspace(workspace, operands.tokens, operands.k, q4_bake_partial_bytes(operands.tokens));
    const Weight fp8 = q4_bake(operands, workspace, stream);
    Tensor x(const_cast<__nv_bfloat16*>(operands.x), DType::BF16, {operands.k, operands.tokens});
    Tensor y(out, DType::BF16, {operands.rows, operands.tokens});
    with_gemma4_trunk(operands.rows, operands.k,
                      [&]<int N, int K>() { gemma4_fp8::launch_a8<N, K>(x, fp8, y, scratch, stream); });
}

void q4_g32_w4a8(const Q4LinearOperands& operands, __nv_bfloat16* out, WorkspaceArena& workspace,
                 cudaStream_t stream) {
    const Q4Route route =
        q4_g32_route(operands.rows, operands.k, operands.tokens, LinearPolicy::AllowA8, q4_exact_prefill());
    if (route == Q4Route::Baked) { return q4_g32_w4a8_baked(operands, out, workspace, stream); }
    if (route == Q4Route::Fused) {
        const auto scope = workspace.scope();
        const Fp8A8Workspace scratch =
            allocate_fp8_a8_workspace(workspace, operands.tokens, operands.k, kQ4FusedPartialBytes);
        Tensor x(const_cast<__nv_bfloat16*>(operands.x), DType::BF16, {operands.k, operands.tokens});
        Weight shape{};
        shape.k = operands.k;
        launch_fp8_a8_quantize(x, shape, scratch, stream);
        q4_fused(operands, out, scratch.codes, scratch.scales, scratch.partials, workspace, stream);
        return;
    }
    const auto scope = workspace.scope();
    const bool tma = operands.rows % 128 == 0 && operands.k % 128 == 0;
    Fp8A8Workspace scratch = allocate_fp8_a8_workspace(workspace, operands.tokens, operands.k,
                                                       tma ? kW4A8PartialBytes : 0);
    std::uint8_t* expanded     = nullptr;
    __nv_bfloat16* ones        = nullptr;
    std::uint8_t* scale_groups = nullptr;
    if (tma) {
        expanded = static_cast<std::uint8_t*>(
            workspace.alloc_bytes(static_cast<std::size_t>(operands.rows) * static_cast<std::size_t>(operands.k), 256)
                .data);
        ones = static_cast<__nv_bfloat16*>(
            workspace.alloc_bytes(static_cast<std::size_t>(operands.rows) * sizeof(__nv_bfloat16), 256).data);
        scale_groups = static_cast<std::uint8_t*>(
            workspace
                .alloc_bytes(static_cast<std::size_t>(operands.rows) * static_cast<std::size_t>(operands.k / 32) *
                                 sizeof(std::uint16_t),
                             256)
                .data);
    }
    Tensor view;
    view.data  = const_cast<__nv_bfloat16*>(operands.x);
    view.dtype = DType::BF16;
    view.ne[0] = operands.k;
    view.ne[1] = operands.tokens;
    view.ne[2] = 1;
    view.ne[3] = 1;
    Weight weight;
    weight.k = operands.k;
    launch_fp8_a8_quantize(view, weight, scratch, stream);
    if (!tma) {
        const dim3 block(kW4A8Warps * 32);
        const dim3 grid((operands.rows + kW4A8Rows - 1) / kW4A8Rows,
                        (operands.tokens + kW4A8Tokens - 1) / kW4A8Tokens);
        with_scale_mode(operands.scale_mode, [&]<int S>() {
            q4g32_w4a8_kernel<S><<<grid, block, 0, stream>>>(scratch.codes, scratch.scales, operands.codes,
                                                             operands.scales, out, operands.rows, operands.k,
                                                             operands.tokens);
        });
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int groups = operands.rows * (operands.k >> 5);
    const dim3 transpose_grid((static_cast<unsigned>(operands.k >> 5) + 31) / 32,
                              (static_cast<unsigned>(operands.rows) + 31) / 32);
    q4_expand_e4m3_kernel<<<(groups + 255) / 256, 256, 0, stream>>>(operands.codes, expanded, operands.rows, operands.k);
    CUDA_CHECK(cudaGetLastError());
    // The TMA kernel reads one scale per 32 columns: a coarse scale is expanded here.
    with_scale_mode(operands.scale_mode, [&]<int S>() {
        q4_transpose_scales<S><<<transpose_grid, 256, 0, stream>>>(operands.scales, scale_groups, operands.rows,
                                                                  operands.k >> 5);
    });
    CUDA_CHECK(cudaGetLastError());
    q4_fill_bf16_ones<<<(operands.rows + 255) / 256, 256, 0, stream>>>(ones, operands.rows);
    CUDA_CHECK(cudaGetLastError());
    const bool narrow = operands.tokens <= 128 && operands.rows != 43008 &&
                        !(operands.rows == 5376 && (operands.k == 16384 || operands.k == 21504));
    if (narrow) {
        launch_q4_w4a8_tma<Q4W4A8Narrow>(scratch.codes, scratch.scales, expanded, scale_groups, ones, out,
                                         scratch.partials, operands.rows, operands.k, operands.tokens, stream);
    } else {
        launch_q4_w4a8_tma<Q4W4A8Mid>(scratch.codes, scratch.scales, expanded, scale_groups, ones, out,
                                      scratch.partials, operands.rows, operands.k, operands.tokens, stream);
    }
}

} // namespace infernix::ops::detail
