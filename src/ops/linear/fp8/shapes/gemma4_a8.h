#pragma once
#include "ops/linear/fp8/fp8_a8_plan.h"

// A8 routes of the Gemma 4 trunk projections (gemma4_a8.cu): from 33 tokens, per-token FP8
// activations on the TMA tensor-core GEMM. Instantiated for the six trunk geometries only.
namespace infernix::ops::detail::gemma4_fp8 {
template <int N, int K>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream);
// Same tile as launch_a8<5376, 21504>, without quantizing. The caller already wrote the codes.
void launch_down_ready(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream);
bool uses_a8(std::int32_t min_tokens, std::int32_t max_tokens);
std::size_t partial_capacity_bytes(std::int32_t max_tokens);
} // namespace infernix::ops::detail::gemma4_fp8
