#include "ops/linear/q5/q5_shapes.h"

namespace infernix::ops::detail {
namespace {

// Only the six registered Gemma 4 31B trunk projections use this schedule. Reuse the
// existing runtime-K A16 launchers; these choices are not a measured performance claim.
Q5Launch select_trunk_tokens(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_wide_t128;
}

} // namespace

// Parameters::geometry joins sliding Q/K/V, global Q/K (V is K), and gate/up.
Q5Launch select_q5_n16384_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n18432_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n43008_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n5376_k8192(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n5376_k16384(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n5376_k21504(std::int32_t tokens) { return select_trunk_tokens(tokens); }

} // namespace infernix::ops::detail
