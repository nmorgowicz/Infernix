#pragma once

#include "ops/common/embed_ids.cuh" // EmbedIds
#include "ops/common/math.h"
#include "ops/common/q4_scale_mode.h"
#include "core/pdl.cuh"

// infernix::ops - scaled embedding kernels for the Gemma 4 tables. Included only by the non-RDC Gemma
// archive launcher (launcher/gemma4_embed_gather.cu): the kernels are templates, so no extern symbol
// can collide with the unscaled kernels of embed_gather.cuh in infernix_ops.
//
// Gemma's embedding table is also its output head, and the gather multiplies by sqrt(hidden). The
// factor is a BF16 value: each decoded value is rounded to BF16, multiplied by it, and rounded
// again, like a BF16 table times a BF16 scalar. A factor of 1 leaves the rows unchanged. An id
// outside the table embeds zeros and reports the fault, as in the unscaled kernels.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace infernix::ops {

template <int D, int BlocksPerToken, int Threads>
__launch_bounds__(Threads) __global__
    void embed_gather_scaled_fp8_kernel(EmbedIds ids, const std::uint8_t* codes,
                                        const __nv_bfloat16* scales, float factor,
                                        __nv_bfloat16* out) {
    pdl::enter();
    static_assert(D % BlocksPerToken == 0);
    constexpr int kValuesPerBlock = D / BlocksPerToken;
    static_assert(kValuesPerBlock % 4 == 0);
    constexpr int kWordsPerBlock = kValuesPerBlock / 4;

    const int token        = static_cast<int>(blockIdx.x) / BlocksPerToken;
    const int split        = static_cast<int>(blockIdx.x) - token * BlocksPerToken;
    const int row          = ids.row(token);
    const int split_offset = split * kValuesPerBlock;
    auto* output_column    = out + static_cast<std::int64_t>(token) * D;
    if (row < 0) {
        for (int i = static_cast<int>(threadIdx.x); i < kValuesPerBlock; i += Threads) {
            output_column[split_offset + i] = __float2bfloat16(0.0f);
        }
        return;
    }
    const float scale    = __bfloat162float(scales[row]);
    const auto* code_row = codes + static_cast<std::int64_t>(row) * D;
    for (int word_index = static_cast<int>(threadIdx.x); word_index < kWordsPerBlock;
         word_index += Threads) {
        const int offset    = split_offset + word_index * 4;
        const unsigned word = *reinterpret_cast<const unsigned*>(code_row + offset);
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            __nv_fp8x2_e4m3 values;
            values.__x           = static_cast<std::uint16_t>(word >> (pair * 16));
            const float2 decoded = static_cast<float2>(values);
            const __nv_bfloat162 rounded =
                __floats2bfloat162_rn(decoded.x * scale, decoded.y * scale);
            reinterpret_cast<__nv_bfloat162*>(output_column + offset + pair * 2)[0] =
                __floats2bfloat162_rn(__low2float(rounded) * factor, __high2float(rounded) * factor);
        }
    }
}

// One thread per (token, group of 32 output columns): 16 code bytes, one group scale, 32 BF16
// outputs. kScaleMode (q4_scale_mode.h): the code plane is K/2 bytes per row for every format, only
// the scale plane differs; one binary16 scale per 32 << kScaleMode columns. Even lanes are the low
// nibble, odd lanes the high nibble, both two's complement.
template <int kScaleMode>
__global__ void embed_gather_q4_kernel(EmbedIds ids, const std::uint8_t* codes,
                                       const std::uint8_t* scales, float factor,
                                       __nv_bfloat16* out, std::int32_t d, std::int32_t T,
                                       std::int32_t padded_d) {
    pdl::enter();
    constexpr std::int32_t kGroup = 32;
    const std::int32_t kg         = padded_d / kGroup;
    const std::int32_t kscale     = kg >> kScaleMode;
    const std::int32_t live       = (d + kGroup - 1) / kGroup;
    const std::int64_t n          = static_cast<std::int64_t>(live) * T;
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const std::int32_t t = static_cast<std::int32_t>(i / live);
        const std::int32_t g = static_cast<std::int32_t>(i - static_cast<std::int64_t>(t) * live);
        const std::int32_t row   = ids.row(t);
        __nv_bfloat16* column    = out + static_cast<std::int64_t>(t) * d + g * kGroup;
        const std::int32_t lanes = min(kGroup, d - g * kGroup);
        if (row < 0) {
            for (std::int32_t lane = 0; lane < lanes; ++lane) { column[lane] = __float2bfloat16(0.0f); }
            continue;
        }
        const std::int64_t group_index = static_cast<std::int64_t>(row) * kg + g;
        const std::int64_t scale_index = static_cast<std::int64_t>(row) * kscale + (g >> kScaleMode);
        const std::uint16_t scale_bits =
            static_cast<std::uint16_t>(scales[scale_index * 2]) |
            static_cast<std::uint16_t>(static_cast<std::uint16_t>(scales[scale_index * 2 + 1]) << 8);
        const float scale        = __half2float(__ushort_as_half(scale_bits));
        const std::uint8_t* code = codes + group_index * (kGroup / 2);
        for (std::int32_t lane = 0; lane < lanes; ++lane) {
            const std::int32_t nibble = (code[lane >> 1] >> ((lane & 1) * 4)) & 0xf;
            const std::int32_t value  = nibble >= 8 ? nibble - 16 : nibble;
            const float rounded = __bfloat162float(__float2bfloat16(static_cast<float>(value) * scale));
            column[lane]        = __float2bfloat16(rounded * factor);
        }
    }
}

} // namespace infernix::ops
