#pragma once

#include "core/weight.h"
#include "core/tensor.h"

#include <cstdint>

namespace infernix::ops::detail {

struct Fp8WeightGeometry {
    std::uint64_t code_plane_bytes;
    std::uint64_t scale_plane_offset;
    std::uint64_t scale_plane_bytes;
    std::uint64_t required_payload_bytes;
};

// Plane geometry of an [n, k] row-scaled matrix whose row words are `scale_word_bytes` wide (2: BF16,
// 4: FP32): the code plane, then the 256-aligned row-word plane.
[[nodiscard]] Fp8WeightGeometry fp8_row_geometry(std::int32_t n, std::int32_t k,
                                                 const char* operation,
                                                 std::int64_t scale_word_bytes = 2);
Fp8WeightGeometry validate_fp8_weight(const Weight& weight, const char* operation);

} // namespace infernix::ops::detail
