#include "ops/linear/q4/q4_g32_gemv.cuh"

#include "ops/common/q4_scale_mode.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/q4/q4_g32_tc.cuh"
#include "ops/linear/q4/q4_g32_wide.cuh"
#include "ops/linear/q4/q4_g32_w4a8.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_g32_route.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <stdexcept>

#include <type_traits>

namespace infernix::ops::detail {
namespace {

constexpr int kQ4G32TileGroups = 32;
// 34 bf16 between lanes: stride/2 = 17 is coprime with 32, so a warp reading one
// element of its own group hits 32 different shared-memory banks.
constexpr int kQ4G32ActStride = 34;

__device__ __forceinline__ void q4g32_decode_group(std::uint32_t packed, float (&w)[8]) {
    const std::uint16_t scale_unused = 0;
    (void)scale_unused;
    // decode_eight applies a scale. The caller scales in FP32, so decode at scale 1.
    Q4SimtDecodeAtom::decode_eight(packed, 0x3c00u, w);
}

template <int kTokens, int kRowsPerWarp, int kWarps, int kScaleMode>
__global__ void q4g32_gemv_fast(const __nv_bfloat16* __restrict__ x,
                                const std::uint8_t* __restrict__ codes,
                                const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
                                int rows, int k) {
    constexpr int kGroup = 32;
    constexpr int kRows = kWarps * kRowsPerWarp;
    const int groups = k / kGroup;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base = (static_cast<int>(blockIdx.x) * kRows) + warp * kRowsPerWarp;
    __shared__ __align__(16) __nv_bfloat16 tile[kTokens][kQ4G32TileGroups * kQ4G32ActStride];

    float acc[kRowsPerWarp][kTokens];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int token = 0; token < kTokens; ++token) { acc[row][token] = 0.0f; }
    }

    for (int group0 = 0; group0 < groups; group0 += kQ4G32TileGroups) {
        const int live = min(kQ4G32TileGroups, groups - group0);
        const int tile_k = live * kGroup;
        const int vecs = tile_k >> 3;
        for (int item = static_cast<int>(threadIdx.x); item < kTokens * vecs; item += kWarps * 32) {
            const int token = item / vecs;
            const int elem  = (item - token * vecs) << 3;
            const uint4 bits = *reinterpret_cast<const uint4*>(
                x + static_cast<long long>(token) * k + group0 * kGroup + elem);
            __nv_bfloat16 unpacked[8];
            *reinterpret_cast<uint4*>(unpacked) = bits;
            const int owner = elem >> 5;
            const int local = elem & 31;
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                *reinterpret_cast<std::uint32_t*>(
                    &tile[token][owner * kQ4G32ActStride + local + pair * 2]) =
                    *reinterpret_cast<std::uint32_t*>(&unpacked[pair * 2]);
            }
        }
        __syncthreads();

        if (lane < live) {
#pragma unroll
            for (int local = 0; local < kRowsPerWarp; ++local) {
                const int row = row_base + local;
                if (row >= rows) { continue; }
                const int group = group0 + lane;
                const std::uint8_t* code =
                    codes + (static_cast<long long>(row) * groups + group) * 16;
                const std::uint16_t scale_bits = *reinterpret_cast<const std::uint16_t*>(
                    scales + (static_cast<long long>(row) * (groups >> kScaleMode) + (group >> kScaleMode)) * 2);
                const float scale = __half2float(__ushort_as_half(scale_bits));
                float weights[32];
#pragma unroll
                for (int word = 0; word < 4; ++word) {
                    float piece[8];
                    q4g32_decode_group(*reinterpret_cast<const std::uint32_t*>(code + word * 4), piece);
#pragma unroll
                    for (int i = 0; i < 8; ++i) { weights[word * 8 + i] = piece[i] * scale; }
                }
#pragma unroll
                for (int token = 0; token < kTokens; ++token) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < 32; i += 2) {
                        const float2 act = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(
                            &tile[token][lane * kQ4G32ActStride + i]));
                        dot = fmaf(weights[i], act.x, dot);
                        dot = fmaf(weights[i + 1], act.y, dot);
                    }
                    acc[local][token] += dot;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int local = 0; local < kRowsPerWarp; ++local) {
        const int row = row_base + local;
#pragma unroll
        for (int token = 0; token < kTokens; ++token) {
            float sum = acc[local][token];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0 && row < rows) {
                out[row + static_cast<long long>(token) * rows] = __float2bfloat16(sum);
            }
        }
    }
}

// T=1 from DRAM. One group of 32 per lane per phase with K fixed at compile time, so every phase
// unrolls and the 16-byte code loads of all rows and phases can be in flight together. The
// activation comes straight from L1 (the whole vector is at most 43 KB), no shared tile or barrier.
// kScaleMode (q4_scale_mode.h): codes are indexed by 32-group as ever; for modes 0/2/3 the binary16
// scale of a 32-group is element (group >> kScaleMode) of the row's K >> (5 + kScaleMode) scales.
// Scale row strides of coarse formats are not 16-byte aligned, so scales load as 2-byte words.
constexpr int kQ4T1Warps = 8;

template <int K, int kRowsPerWarp, int kScaleMode>
__global__ void __launch_bounds__(kQ4T1Warps * 32)
    q4g32_gemv_t1(const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
                  const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kGroups = K / 32;
    constexpr int kScaleGroups = K >> (5 + kScaleMode);
    static_assert(K % q4_scale_span(kScaleMode) == 0);
    constexpr int kPhases = (kGroups + 31) / 32;
    const int lane        = static_cast<int>(threadIdx.x) & 31;
    const int warp        = static_cast<int>(threadIdx.x) >> 5;
    const int row_base    = (static_cast<int>(blockIdx.x) * kQ4T1Warps + warp) * kRowsPerWarp;
    const auto* scale_words = reinterpret_cast<const unsigned short*>(scales);
    float acc[kRowsPerWarp] = {};
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        const int group = phase * 32 + lane;
        if (kGroups % 32 != 0 && phase == kPhases - 1 && group >= kGroups) { break; }
        uint4 packed[kRowsPerWarp];
        unsigned short scale_bits[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const long long index = static_cast<long long>(row_base + r) * kGroups + group;
            const long long scale_index = static_cast<long long>(row_base + r) * kScaleGroups + (group >> kScaleMode);
            packed[r]     = __ldcs(reinterpret_cast<const uint4*>(codes) + index);
            scale_bits[r] = __ldcs(scale_words + scale_index);
        }
        const auto* activation = reinterpret_cast<const uint4*>(x + group * 32);
        uint4 act[4];
#pragma unroll
        for (int v = 0; v < 4; ++v) { act[v] = __ldg(activation + v); }
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            const std::uint32_t words[4] = {packed[r].x, packed[r].y, packed[r].z, packed[r].w};
            float even = 0.0f;
            float odd  = 0.0f;
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                float value[8];
                q4g32_decode_group(words[w], value);
                const std::uint32_t pairs[4] = {act[w].x, act[w].y, act[w].z, act[w].w};
#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&pairs[p]));
                    even           = fmaf(value[2 * p], a.x, even);
                    odd            = fmaf(value[2 * p + 1], a.y, odd);
                }
            }
            const float scale = __half2float(__ushort_as_half(scale_bits[r]));
            acc[r] = fmaf(even + odd, scale, acc[r]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        float sum = acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) { sum += __shfl_xor_sync(0xffffffffu, sum, offset); }
        if (lane == 0) { out[row_base + r] = __float2bfloat16(sum); }
    }
}

template <int K, int kScaleMode>
bool launch_q4g32_t1_rows(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream) {
    const auto launch = [&]<int kRowsPerWarp>() {
        constexpr int kRows = kQ4T1Warps * kRowsPerWarp;
        if (operands.rows % kRows != 0) { return false; }
        q4g32_gemv_t1<K, kRowsPerWarp, kScaleMode><<<operands.rows / kRows, kQ4T1Warps * 32, 0, stream>>>(
            operands.x, operands.codes, operands.scales, out);
        return true;
    };
    // Cold-L2 sweep of 2, 4, 8 rows per warp on the trunk shapes: 8 on the wide outputs (1.45-1.52
    // TB/s), 2 on the 5376-row outputs (1.45-1.50 TB/s; 1 row per warp left them at 1.30).
    if (operands.rows >= 8192 && launch.template operator()<8>()) { return true; }
    return launch.template operator()<2>();
}

template <int kScaleMode>
bool launch_q4g32_t1_mode(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream) {
    switch (operands.k) {
    case 5376: return launch_q4g32_t1_rows<5376, kScaleMode>(operands, out, stream);
    case 8192: return launch_q4g32_t1_rows<8192, kScaleMode>(operands, out, stream);
    case 16384: return launch_q4g32_t1_rows<16384, kScaleMode>(operands, out, stream);
    case 21504: return launch_q4g32_t1_rows<21504, kScaleMode>(operands, out, stream);
    default: return false;
    }
}

bool launch_q4g32_t1(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                     int scale_mode) {
    switch (scale_mode) {
    case 0: return launch_q4g32_t1_mode<0>(operands, out, stream);
    case 2: return launch_q4g32_t1_mode<2>(operands, out, stream);
    case 3: return launch_q4g32_t1_mode<3>(operands, out, stream);
    default: return false;
    }
}

template <int kTokens, int kRowsPerWarp, int kWarps>
void launch_q4g32_gemv_fast(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                            int scale_mode) {
    if (operands.k % q4_scale_span(scale_mode) != 0) {
        throw std::invalid_argument("q4 gemv requires K divisible by its scale group size");
    }
    constexpr int kRows = kWarps * kRowsPerWarp;
    const dim3 block(kWarps * 32);
    const dim3 grid((operands.rows + kRows - 1) / kRows);
    const auto launch = [&]<int kScaleMode>() {
        q4g32_gemv_fast<kTokens, kRowsPerWarp, kWarps, kScaleMode>
            <<<grid, block, 0, stream>>>(operands.x, operands.codes, operands.scales, out,
                                         operands.rows, operands.k);
    };
    switch (scale_mode) {
    case 0: launch.template operator()<0>(); break;
    case 2: launch.template operator()<2>(); break;
    case 3: launch.template operator()<3>(); break;
    default: throw std::invalid_argument("q4 gemv: unsupported scale group size");
    }
}

template <int kTokens>
void launch_q4g32_gemv_tokens(const Q4LinearOperands& operands, __nv_bfloat16* out,
                              cudaStream_t stream, int scale_mode) {
    // One row per warp keeps the token accumulators in registers. Wide output shapes can
    // afford four rows per warp at T=1, which is what saturates the bus there.
    if (kTokens == 1 && operands.rows >= 8192) {
        launch_q4g32_gemv_fast<1, 4, 8>(operands, out, stream, scale_mode);
        return;
    }
    if (kTokens == 1) {
        launch_q4g32_gemv_fast<1, 1, 8>(operands, out, stream, scale_mode);
        return;
    }
    // T>1 reuses one weight read across tokens. Four rows per warp keeps enough
    // weight bytes in flight to cover the extra FMAs on the trunk shapes.
    if (kTokens <= 8) {
        launch_q4g32_gemv_fast<kTokens, 4, 8>(operands, out, stream, scale_mode);
        return;
    }
    if (operands.rows >= 8192) {
        launch_q4g32_gemv_fast<kTokens, 2, 8>(operands, out, stream, scale_mode);
        return;
    }
    launch_q4g32_gemv_fast<kTokens, 1, 8>(operands, out, stream, scale_mode);
}

} // namespace

void q4_g32_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                     WorkspaceArena* workspace, cudaStream_t stream) {
    const Q4LinearOperands operands = q4_linear_operands(x, weight);
    validate_q4_operands(operands);
    // q4_g32 / q4_g128 / q4_g256: the code plane is identical, only the scale plane differs
    // (coarser binary16 scales).
    const int mode = q4_scale_mode(weight.qtype);
    if (!q4_scale_mode_valid(mode) || weight.group_size != (32u << mode) || operands.k % q4_scale_span(mode) != 0) {
        throw std::invalid_argument(
            "q4 fp16 dispatch requires a q4_g32/g128/g256 weight with K divisible by its scale span");
    }
    // Weights from DRAM (cold-L2 bench, INFERNIX_Q4_COLD): T=1 on the unrolled GEMV is
    // 1.45-1.53 TB/s on the six trunk shapes and the head. T=2..8 uses the K-split
    // tensor-core kernel when the shape divides its 16-row, 256-K tile; T=7 is
    // 1.25-1.42 TB/s. With A8 allowed, the Gemma trunk shapes take W4A8 from 9 tokens
    // (q4_g32_route.h); the 8-token passes then serve other shapes and A16-only calls.
    auto* output = static_cast<__nv_bfloat16*>(out.data);
    const int tokens = operands.tokens;
    const Q4Route route = q4_g32_route(operands.rows, operands.k, tokens, policy, q4_exact_prefill());
    switch (route) {
    case Q4Route::Fused:
    case Q4Route::Baked:
    case Q4Route::Exact:
        if (workspace == nullptr) {
            throw std::invalid_argument("q4_g32 W4A8 requires workspace");
        }
        q4_g32_w4a8(operands, output, *workspace, stream);
        return;
    case Q4Route::TensorCore:
        // Two 8-token passes for 9..16 tokens.
        for (int first = 0; first < tokens; first += 8) {
            Q4LinearOperands chunk = operands;
            chunk.x                = operands.x + static_cast<long long>(first) * operands.k;
            chunk.tokens           = std::min(8, tokens - first);
            q4_g32_tc8(chunk, output + static_cast<long long>(first) * operands.rows, stream, mode);
        }
        return;
    case Q4Route::Wide: q4_g32_wide(operands, output, stream, mode); return;
    case Q4Route::Gemv: break;
    }
    if (tokens == 1 && launch_q4g32_t1(operands, output, stream, mode)) { return; }
    switch (tokens) {
    case 1: launch_q4g32_gemv_tokens<1>(operands, output, stream, mode); return;
    case 2: launch_q4g32_gemv_tokens<2>(operands, output, stream, mode); return;
    case 3: launch_q4g32_gemv_tokens<3>(operands, output, stream, mode); return;
    case 4: launch_q4g32_gemv_tokens<4>(operands, output, stream, mode); return;
    case 5: launch_q4g32_gemv_tokens<5>(operands, output, stream, mode); return;
    case 6: launch_q4g32_gemv_tokens<6>(operands, output, stream, mode); return;
    case 7: launch_q4g32_gemv_tokens<7>(operands, output, stream, mode); return;
    case 8: launch_q4g32_gemv_tokens<8>(operands, output, stream, mode); return;
    case 16: launch_q4g32_gemv_tokens<16>(operands, output, stream, mode); return;
    default: throw std::logic_error("q4_g32: no GEMV for this token count");
    }
}

} // namespace infernix::ops::detail
