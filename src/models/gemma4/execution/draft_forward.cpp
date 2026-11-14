#include "models/gemma4/execution/draft_forward.h"

#include "core/device.h"
#include "models/gemma4/execution/projection.h"
#include "infernix/ops/argmax.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/gemma4_attention.h"
#include "infernix/ops/gemma4_residual_norm.h"
#include "infernix/ops/geglu.h"
#include "infernix/ops/rmsnorm.h"
#include "infernix/ops/rope.h"

#include <algorithm>
#include <stdexcept>

namespace infernix::models::gemma4 {
namespace {

constexpr std::size_t kAllocations = 16;
constexpr std::size_t kAlignment = 256;

// hn holds rmsnorm(x, input_norm) on entry; with next_input_norm it leaves the next layer's
// input norm there.
void draft_layer(Tensor& x, Tensor& hn, const Gemma4LayerWeights& layer, const void* next_input_norm,
                 const Tensor& position, const Gemma4CacheLayer& cache, ops::Fp8RowCublasContext& gemm,
                 WorkspaceArena& workspace, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const int hidden = layer.hidden;
    const int dim = layer.head_dim;
    const int heads = layer.q_heads;
    const int inter = layer.intermediate;
    if (layer.qkv.n != heads * dim || layer.qkv.k != hidden) {
        throw std::invalid_argument("gemma4 draft: query must be [heads*dim,hidden]");
    }
    auto scope = workspace.scope();
    auto q = workspace.alloc(DType::BF16, {dim, heads, 1});
    auto qn = workspace.alloc(DType::BF16, {dim, heads, 1});
    auto attn = workspace.alloc(DType::BF16, {dim, heads, 1});
    auto projected = workspace.alloc(DType::BF16, {hidden, 1});
    Tensor q_column(q.data, DType::BF16, {heads * dim, 1});
    project(hn, layer.qkv, layer.qkv_policy, q_column, gemm, workspace, stream);
    ops::rmsnorm(q, norm_weight(layer.q_norm, dim), layer.rms_eps, false, qn, stream);
    ops::rope(position, layer.rotary_dim, layer.rope_theta, qn, execution);
    ops::gemma4_attention(qn, cache, position, layer.window, 1.0f, attn, workspace, stream);
    Tensor attn_column(attn.data, DType::BF16, {heads * dim, 1});
    project(attn_column, layer.o_proj, layer.o_policy, projected, gemm, workspace, stream);
    const Tensor pre_ff_norm = norm_weight(layer.pre_ff_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_attn_norm, hidden), 1.0f, &pre_ff_norm,
                              layer.rms_eps, x, &hn, stream);
    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, 1});
    project(hn, layer.gate_up, layer.gate_up_policy, gate_up, gemm, workspace, stream);
    const Tensor gate = gate_up.slice(0, 0, inter);
    const Tensor up = gate_up.slice(0, inter, inter);
    auto activated = workspace.alloc(DType::BF16, {inter, 1});
    ops::geglu(gate, up, activated, stream);
    project(activated, layer.down, layer.down_policy, projected, gemm, workspace, stream);
    const Tensor next_norm = norm_weight(next_input_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(layer.post_ff_norm, hidden), layer.layer_scalar,
                              next_input_norm != nullptr ? &next_norm : nullptr, layer.rms_eps, x,
                              next_input_norm != nullptr ? &hn : nullptr, stream);
}

} // namespace

std::size_t gemma4_draft_workspace_bytes(const Gemma4DraftWeights& draft) {
    std::size_t widest = 0;
    for (const Gemma4LayerWeights& layer : draft.layers) {
        const auto q = static_cast<std::size_t>(layer.q_heads) * layer.head_dim;
        const auto inter = static_cast<std::size_t>(layer.intermediate);
        const std::size_t scratch = std::max(
            {projection_bytes(layer.qkv, layer.qkv_policy, 1), projection_bytes(layer.o_proj, layer.o_policy, 1),
             projection_bytes(layer.gate_up, layer.gate_up_policy, 1),
             projection_bytes(layer.down, layer.down_policy, 1),
             ops::gemma4_attention_workspace_capacity_bytes(layer.head_dim, layer.q_heads, layer.kv_heads, 1)});
        widest = std::max(widest, (3 * q + 3 * inter + static_cast<std::size_t>(layer.hidden)) * 2 + scratch);
    }
    const std::size_t outer =
        (2U * draft.backbone + 3U * draft.hidden + static_cast<std::size_t>(draft.vocab)) * 2U +
        std::max({projection_bytes(draft.input_projection, ops::LinearPolicy::A16Only, 1),
                  projection_bytes(draft.output_projection, ops::LinearPolicy::A16Only, 1),
                  projection_bytes(draft.output_head, ops::LinearPolicy::A16Only, 1)});
    return outer + widest + kAllocations * kAlignment;
}

void gemma4_draft_step(const Tensor& token, const Tensor& position, Tensor& carry, Tensor& next,
                       const Gemma4Embedding& token_embedding, const Gemma4DraftWeights& draft,
                       std::span<const Gemma4CacheLayer> target_caches, ops::Fp8RowCublasContext& gemm,
                       WorkspaceArena& workspace, DeviceExecutionView execution) {
    const cudaStream_t stream = execution.stream;
    const int backbone = draft.backbone;
    const int hidden = draft.hidden;
    if (token.ne[0] != 1 || position.ne[0] != 1 || next.ne[0] != 1 || carry.ne[0] != backbone ||
        carry.ne[1] != 1 || draft.layers.empty() || draft.kv_source.size() != draft.layers.size()) {
        throw std::invalid_argument("gemma4 draft: step shape");
    }
    auto scope = workspace.scope();
    auto stem = workspace.alloc(DType::BF16, {2 * backbone, 1});
    // [embedding ; carry]: the two halves of the one stem column.
    Tensor embedded = stem.slice(0, 0, backbone);
    ops::embedding(token, token_embedding.table, token_embedding.scale, embedded, stream);
    CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(stem.data) + static_cast<std::size_t>(backbone) * 2,
                               carry.data, static_cast<std::size_t>(backbone) * 2, cudaMemcpyDeviceToDevice,
                               stream));
    auto x = workspace.alloc(DType::BF16, {hidden, 1});
    auto hn = workspace.alloc(DType::BF16, {hidden, 1});
    project(stem, draft.input_projection, ops::LinearPolicy::A16Only, x, gemm, workspace, stream);
    ops::rmsnorm(x, norm_weight(draft.layers[0].input_norm, hidden), draft.rms_eps, false, hn, stream);
    for (std::size_t index = 0; index < draft.layers.size(); ++index) {
        const void* following = index + 1 < draft.layers.size() ? draft.layers[index + 1].input_norm : nullptr;
        const auto source = static_cast<std::size_t>(draft.kv_source[index]);
        if (source >= target_caches.size()) {
            throw std::invalid_argument("gemma4 draft: KV source is outside the target");
        }
        draft_layer(x, hn, draft.layers[index], following, position, target_caches[source], gemm, workspace,
                    execution);
    }
    ops::rmsnorm(x, norm_weight(draft.final_norm, hidden), draft.rms_eps, false, hn, stream);
    auto logits = workspace.alloc(DType::BF16, {draft.vocab, 1});
    project(hn, draft.output_head, ops::LinearPolicy::A16Only, logits, gemm, workspace, stream);
    ops::argmax(logits, next, draft.vocab, stream);
    project(hn, draft.output_projection, ops::LinearPolicy::A16Only, carry, gemm, workspace, stream);
}

} // namespace infernix::models::gemma4
