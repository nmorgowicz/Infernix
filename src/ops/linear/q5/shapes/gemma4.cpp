#include "ops/linear/q5/q5_shapes.h"

namespace infernix::ops::detail {
namespace {

// Only the six registered Gemma 4 31B trunk projections use this schedule. The tokens<=4 decode
// arm is the S26-measured choice (q5-decode-kernel-20261128): at one token the static-K
// one-row-per-warp instance wins the down shape (-14.8 %, clean DRAM condition) and the static-K
// two-rows-per-warp instance wins the two o shapes; at two or more tokens every candidate was a
// tie or a loss on at least one shape, so those widths keep the shipped arm. Q5A16DirectRows
// schedules keep the per-lane K map and accumulation order of the shipped kernel, so outputs are
// bit-identical at every token count.
Q5Launch select_trunk_tokens(std::int32_t tokens) {
    if (tokens <= 4) return launch_q5_a16_direct_r2_t4_w2_g8_b4;
    if (tokens <= 8) return launch_q5_a16_sliced_r16_t8_w4_s2;
    if (tokens <= 16) return launch_q5_a16_sliced_r16_t16_w4_s2;
    if (tokens <= 32) return launch_q5_a16_sliced_r32_t32_w4_s2;
    if (tokens <= 64) return launch_q5_a16_sliced_r32_t64_w2_s2;
    if (tokens <= 128) return launch_q5_a16_mma_r32_t128;
    return launch_q5_a16_wide_t128;
}

Q5Launch select_trunk_decode(Q5Launch decode, std::int32_t tokens) {
    return tokens <= 1 ? decode : select_trunk_tokens(tokens);
}

} // namespace

// Parameters::geometry joins sliding Q/K/V, global Q/K (V is K), and gate/up.
Q5Launch select_q5_n16384_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n18432_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n43008_k5376(std::int32_t tokens) { return select_trunk_tokens(tokens); }
Q5Launch select_q5_n5376_k8192(std::int32_t tokens) {
    return select_trunk_decode(launch_q5_a16_rows_m2_t1_k8192, tokens);
}
Q5Launch select_q5_n5376_k16384(std::int32_t tokens) {
    return select_trunk_decode(launch_q5_a16_rows_m2_t1_k16384, tokens);
}
Q5Launch select_q5_n5376_k21504(std::int32_t tokens) {
    return select_trunk_decode(launch_q5_a16_rows_m1_t1_k21504, tokens);
}

} // namespace infernix::ops::detail
