#pragma once

// The Gemma 4 MTP assistant (Gemma4Assistant) forward: query-only layers over the target's K/V.

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/linear.h"
#include "models/gemma4/execution/text_forward.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <span>
#include <vector>

namespace infernix::models::gemma4 {

// The assistant's weights. Each layer reuses Gemma4LayerWeights with qkv holding the query rows
// only (k_norm unused); kv_source[i] is the target layer whose cache layer i reads.
struct Gemma4DraftWeights {
    Weight input_projection;  // [hidden, 2*backbone]
    Weight output_projection; // [backbone, hidden]
    Weight output_head;       // [vocab, hidden]
    const void* final_norm = nullptr;
    std::vector<Gemma4LayerWeights> layers;
    std::vector<int> kv_source;
    int hidden = 0;
    int backbone = 0;
    int vocab = 0;
    float rms_eps = 1.0e-6f;
};

// Arena capacity of gemma4_draft_step: every allocation of one step with its projection and
// attention scratch.
[[nodiscard]] std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft);

// One assistant step for one column. stem = [embed(token) * token_embedding.scale ; carry] is
// projected, normed and run through every layer at `position` (the last target position: every
// step of a round queries there), reading target_caches[kv_source[i]] without writing K/V. Then
// next = argmax(output_head(final_norm(x))) (no softcap: argmax is invariant to it) and
// carry = output_projection(final_norm(x)). token/position/next are I32 [1]; carry is BF16
// [backbone,1] and is updated in place.
void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Gemma4Embedding& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches, ops::Fp8RowCublasContext& gemm,
                       WorkspaceArena& workspace, DeviceExecutionView execution);

} // namespace infernix::models::gemma4
