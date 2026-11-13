#pragma once

#include "infernix/ops/linear.h"

#include <cstdint>

namespace infernix::ops::detail {

// Every q4_g32 kernel choice. The dispatch calls q4_g32_route and launches what it names.
enum class Q4Route : std::uint8_t {
    Gemv,       // T=1, and T 2..8 or 16 on shapes the tensor-core GEMV cannot tile
    TensorCore, // T 2..16 in 8-token passes (rows % 16 == 0, k % 256 == 0)
    Wide,       // A16 from 17 tokens, and 9..15 on shapes the tensor-core GEMV cannot tile
    Fused,      // Gemma trunk, A8, 9..384 tokens: codes rounded to E4M3 inside the FP8 GEMM
    Baked,      // Gemma trunk, A8, from 385 tokens: a bake pass, then the FP8 A8 GEMM
    Exact,      // A8 from 128 tokens off the trunk, or on it with INFERNIX_Q4_EXACT: FP32 group scales
};

inline constexpr std::int32_t kQ4A8MinTokens    = 9;
inline constexpr std::int32_t kQ4FusedMaxTokens = 384;
inline constexpr std::int32_t kQ4ExactMinTokens = 128;

// The six Gemma 4 31B trunk projections: qkv, o, gate/up, and down at the three widths.
constexpr bool q4_trunk_shape(std::int32_t rows, std::int32_t k) {
    if (k == 5376) { return rows == 16384 || rows == 18432 || rows == 43008; }
    return rows == 5376 && (k == 8192 || k == 16384 || k == 21504);
}

// `exact_prefill` is INFERNIX_Q4_EXACT, read by the caller.
constexpr Q4Route q4_g32_route(std::int32_t rows, std::int32_t k, std::int32_t tokens, LinearPolicy policy,
                               bool exact_prefill) {
    if (allows_a8(policy)) {
        if (!exact_prefill && tokens >= kQ4A8MinTokens && q4_trunk_shape(rows, k)) {
            return tokens <= kQ4FusedMaxTokens ? Q4Route::Fused : Q4Route::Baked;
        }
        if (tokens >= kQ4ExactMinTokens) { return Q4Route::Exact; }
    }
    if (tokens > 1 && tokens <= 16 && rows % 16 == 0 && k % 256 == 0) { return Q4Route::TensorCore; }
    if (tokens > 8 && tokens != 16) { return Q4Route::Wide; }
    return Q4Route::Gemv;
}

bool q4_exact_prefill();

} // namespace infernix::ops::detail
