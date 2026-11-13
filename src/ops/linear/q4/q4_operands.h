#pragma once

#include "core/tensor.h"
#include "core/weight.h"
#include "ops/common/q4_scale_mode.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {

struct Q4LinearOperands {
    const __nv_bfloat16* x;
    const std::uint8_t* codes;
    const std::uint8_t* scales;
    std::int32_t rows;
    std::int32_t k;
    std::int32_t tokens;
    std::int32_t padded_k;
    // Row multipliers of the weight (Weight::row_scales), or null: the Gemma W4A8 route derives them
    // into the workspace then.
    const __nv_bfloat16* row_scales = nullptr;
    // Scale mode of the weight (q4_scale_mode.h): log2(G / 32) for q4_g32/g128/g256.
    std::int32_t scale_mode = 0;
};

inline Q4LinearOperands q4_linear_operands(const Tensor& x, const Weight& weight) {
    return {static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            weight.n,
            weight.k,
            x.ne[1],
            weight.padded_shape[1],
            static_cast<const __nv_bfloat16*>(weight.row_scales),
            q4_scale_mode(weight.qtype) > 0 ? q4_scale_mode(weight.qtype) : 0};
}

// Private template domain. Public shape admission remains in q4_dispatch.
inline void validate_q4_operands(const Q4LinearOperands& operands) {
    const auto aligned = [](const void* pointer, std::uintptr_t alignment) {
        return pointer && (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0;
    };
    if (operands.rows <= 0 || operands.tokens <= 0 || operands.k <= 0 || operands.k % 128 != 0 ||
        operands.padded_k < operands.k || operands.padded_k % 128 != 0) {
        throw std::invalid_argument(
            "Q4 templates require positive N/T and K/padded K aligned to 128");
    }
    if (!aligned(operands.x, 16) || !aligned(operands.codes, 16) || !aligned(operands.scales, 4)) {
        throw std::invalid_argument("Q4 templates require aligned activation and weight planes");
    }
}

} // namespace infernix::ops::detail
