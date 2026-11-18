#pragma once

#include "infernix/ops/gemma4_cache.h"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

// Scalar row addresses are shared by SIMT, MMA staging and fused stores. Code rows have D
// columns (512 global, 256 sliding); scale rows have D / 256 columns, with identical page order.
__host__ __device__ inline std::int64_t gemma4_kv_row(int physical, int heads, int head, int slot) {
    return (static_cast<std::int64_t>(physical) * heads + head) * kGemma4KvPageTokens + slot;
}

// No FP16 product: both reader paths consume exactly this BF16 represented value.
__device__ __forceinline__ __nv_bfloat16 gemma4_fp8_expand(std::uint8_t code, __half scale) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return __float2bfloat16_rn(__fmul_rn(static_cast<float>(value), __half2float(scale)));
}

// The RoPE frequency of pair i (dimensions i and i + rotary_dim / 2). gemma4_qkv_prep rotates by the
// FP32 angle position x this value, and key-derived V readers form that same FP32 angle.
__device__ __forceinline__ float gemma4_rope_frequency(int pair, int rotary_dim, float theta) {
    const float exponent = -2.0F * static_cast<float>(pair) / static_cast<float>(rotary_dim);
    return powf(theta, exponent);
}

// Turns a rotated pair back (Gemma4ValueSource::Key): RoPE wrote a' = a c - b s, b' = b c + a s at
// the FP32 angle position x frequency, so a = a' c + b' s and b = b' c - a' s. The angle is reduced
// to [-pi, pi] (|error| about 2e-7 rad below 2^18 rad) before the hardware sine/cosine, whose
// absolute error there is about 4e-7: the inverse rotation is exact to ~1e-6 relative, far below
// the BF16 rounding the rebuilt V then takes.
__device__ __forceinline__ void gemma4_unrope_pair(float rotated_a, float rotated_b, int position,
                                                   float frequency, float& a, float& b) {
    constexpr float kTwoPiHigh = 6.28318548202514648f;      // float(2 pi)
    constexpr float kTwoPiLow  = -1.74845553146951529e-7f;  // 2 pi - float(2 pi)
    const float angle = static_cast<float>(position) * frequency;
    const float turns = rintf(angle * 0.159154943091895336f);
    const float reduced = fmaf(-turns, kTwoPiLow, fmaf(-turns, kTwoPiHigh, angle));
    float s, c;
    __sincosf(reduced, &s, &c);
    a = fmaf(rotated_a, c, rotated_b * s);
    b = fmaf(rotated_b, c, -rotated_a * s);
}

// One warp owns BF16 pairs lane + 32*s of a D = 64 * Pairs row (Pairs 8: D512 global, 4: D256
// sliding). Each 256-column group spans four slots and gets its own FP16 scale.
// Nonfinite policy is explicit: any NaN or infinity poisons its complete 256-column group
// with canonical E4M3 NaNs and unit scale; never silently turn it into a zero row.
template <int Pairs>
__device__ __forceinline__ void gemma4_fp8_store(const __nv_bfloat162 (&values)[Pairs],
                                                std::uint8_t* codes, __half* scales,
                                                std::int64_t row, int lane) {
    static_assert(Pairs == 4 || Pairs == 8, "Gemma4 FP8 rows are D256 or D512");
    constexpr int Groups = Pairs / 4;
#pragma unroll
    for (int group = 0; group < Groups; ++group) {
        float amax = 0.0f;
        bool nonfinite = false;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float2 x = __bfloat1622float2(values[group * 4 + s]);
            nonfinite |= !isfinite(x.x) || !isfinite(x.y);
            amax = fmaxf(amax, fmaxf(fabsf(x.x), fabsf(x.y)));
        }
        const bool poison = __any_sync(0xffffffffU, nonfinite);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffU, amax, offset));
        }
        const auto params = kv_cache_fp8_quant_params(amax);
        if (lane == 0) { scales[row * Groups + group] = poison ? __float2half_rn(1.0f) : params.scale; }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float2 x = __bfloat1622float2(values[group * 4 + s]);
            const auto packed = poison ? std::uint16_t{0x7f7f} :
                kv_cache_fp8_quant_code2(x.x, x.y, params.inverse_scale);
            reinterpret_cast<std::uint16_t*>(codes + row * (Pairs * 64))[lane + 32 * (group * 4 + s)] = packed;
        }
    }
}

} // namespace infernix::ops::detail
