#pragma once

#include "core/arena.h"
#include "ops/linear/q4/q4_g32_route.h"
#include "ops/linear/q4/q4_operands.h"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

// The Fused, Baked and Exact routes of q4_g32_route (per-token FP8 activations on q4 weights), for
// the q4_g32/g128/g256 family (Q4LinearOperands::scale_mode). Fused and Baked use row-scaled
// E4M3 weights, rounded inside the GEMM or by a bake pass; Exact widens the int4 codes to E4M3 and
// applies FP32 group scales. A coarse weight gives the same bytes and outputs as the g32
// weight carrying its decoded per-32 scales.
void q4_g32_w4a8(const Q4LinearOperands& operands, __nv_bfloat16* out, WorkspaceArena& workspace,
                 cudaStream_t stream);

// Workspace for every route a call of at most `tokens` tokens can take.
std::size_t q4_g32_w4a8_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens);

// The bake route on any trunk shape from 9 tokens, with its own workspace size. The fused route
// must give the same values.
void q4_g32_w4a8_baked(const Q4LinearOperands& operands, __nv_bfloat16* out, WorkspaceArena& workspace,
                       cudaStream_t stream);
std::size_t q4_g32_baked_workspace_bytes(std::int32_t rows, std::int32_t k, std::int32_t tokens);

// BF16 row multipliers of a trunk weight (rows x 2 bytes), written on `stream` from the FP16
// group scales. `scale_mode` is the weight's q4 scale mode (q4_scale_mode.h): 0 for q4_g32, 2 for
// q4_g128, 3 for q4_g256.
void q4_g32_row_multipliers(const std::uint8_t* scales, void* row_scales, std::int32_t rows, std::int32_t k,
                            std::int32_t scale_mode, cudaStream_t stream);

// The Gemma down projection can take A8 codes the GeGLU already wrote.
bool q4_g32_takes_ready_codes(std::int32_t rows, std::int32_t k, std::int32_t tokens);
void q4_g32_w4a8_ready(const Q4LinearOperands& operands, __nv_bfloat16* out, std::uint8_t* codes,
                       float* scales, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops::detail
