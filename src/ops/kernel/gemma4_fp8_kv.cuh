#pragma once

#include "infernix/ops/gemma4_cache.h"
#include "ops/kv_cache/fp8_e4m3_row_codec.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

// Scalar row addresses are shared by SIMT, MMA staging and fused stores. Code rows
// have 512 columns; scale rows have 512 / 256 columns, with identical page order.
__host__ __device__ inline std::int64_t gemma4_kv_row(int physical, int heads, int head, int slot) {
    return (static_cast<std::int64_t>(physical) * heads + head) * kGemma4KvPageTokens + slot;
}

// No FP16 product: both reader paths consume exactly this BF16 represented value.
__device__ __forceinline__ __nv_bfloat16 gemma4_fp8_expand(std::uint8_t code, __half scale) {
    __nv_fp8_e4m3 value;
    value.__x = code;
    return __float2bfloat16_rn(__fmul_rn(static_cast<float>(value), __half2float(scale)));
}

// One warp owns BF16 pairs lane + 32*s. Each D512 half spans four slots.
// Nonfinite policy is explicit: any NaN or infinity poisons its complete row-half
// with canonical E4M3 NaNs and unit scale; never silently turn it into a zero row.
__device__ __forceinline__ void gemma4_fp8_store(const __nv_bfloat162 (&values)[8],
                                                std::uint8_t* codes, __half* scales,
                                                std::int64_t row, int lane) {
#pragma unroll
    for (int group = 0; group < 2; ++group) {
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
        if (lane == 0) { scales[row * 2 + group] = poison ? __float2half_rn(1.0f) : params.scale; }
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float2 x = __bfloat1622float2(values[group * 4 + s]);
            const auto packed = poison ? std::uint16_t{0x7f7f} :
                kv_cache_fp8_quant_code2(x.x, x.y, params.inverse_scale);
            reinterpret_cast<std::uint16_t*>(codes + row * 512)[lane + 32 * (group * 4 + s)] = packed;
        }
    }
}

} // namespace infernix::ops::detail
