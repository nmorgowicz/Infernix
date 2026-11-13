#pragma once

#include "core/weight.h"

#include <cstdint>

#if defined(__CUDACC__)
#    define INFERNIX_Q4_HD __host__ __device__
#else
#    define INFERNIX_Q4_HD
#endif

namespace infernix::ops::detail {

// The q4 fp16 family keeps K/2 code bytes per row for every format, in the same order: 16 bytes
// per 32-weight group. Only the scale plane differs. A kernel names the plane by its scale mode:
//
//   0, 2, 3    one binary16 scale per 32 << mode weights (q4_g32, q4_g128, q4_g256). The scale
//              index of the weight at column k of row r is r * (K >> (5 + mode)) + (k >> (5 + mode)).
//
// Kernels take the mode as a template argument.

INFERNIX_Q4_HD constexpr bool is_q4_fp16_family(QType qtype) {
    return qtype == QType::Q4_G32_FP16 || qtype == QType::Q4_G128_FP16 || qtype == QType::Q4_G256_FP16;
}

// 0 for g32, 2 for g128, 3 for g256; -1 for any other qtype.
INFERNIX_Q4_HD constexpr int q4_scale_mode(QType qtype) {
    switch (qtype) {
    case QType::Q4_G32_FP16: return 0;
    case QType::Q4_G128_FP16: return 2;
    case QType::Q4_G256_FP16: return 3;
    default: return -1;
    }
}

INFERNIX_Q4_HD constexpr bool q4_scale_mode_valid(int mode) { return mode == 0 || mode == 2 || mode == 3; }

// K columns one binary16 scale covers (32 << mode). K of a weight in this mode is a multiple of
// it, except mode 0 (K is a multiple of 128 by the layout).
INFERNIX_Q4_HD constexpr int q4_scale_span(int mode) { return 32 << mode; }

// Bytes of one row's scale plane for K columns.
INFERNIX_Q4_HD constexpr long long q4_scale_row_bytes(int mode, long long k) { return (k >> (5 + mode)) * 2; }

} // namespace infernix::ops::detail
