#include "models/gemma4/execution/text_forward.h"

#include "models/gemma4/execution/projection.h"
#include "infernix/ops/embedding.h"
#include "infernix/ops/gemma4_attention.h"
#include "infernix/ops/gemma4_qkv_prep.h"
#include "infernix/ops/gemma4_residual_norm.h"
#include "infernix/ops/geglu.h"
#include "infernix/ops/logit_softcap.h"
#include "infernix/ops/rmsnorm.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace infernix::models::gemma4 {
namespace {

constexpr std::size_t kAllocations = 18;
constexpr std::size_t kAlignment = 256;

std::size_t add_bytes(std::size_t a, std::size_t b) {
    if (a > std::numeric_limits<std::size_t>::max() - b) {
        throw std::overflow_error("Gemma4 forward: workspace size overflow");
    }
    return a + b;
}

std::size_t multiply_bytes(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error("Gemma4 forward: workspace size overflow");
    }
    return a * b;
}

void require_matrix(const Weight& weight, int rows, int cols, const char* name) {
    if (weight.n != rows || weight.k != cols) {
        throw std::invalid_argument(std::string("gemma4_layer: ") + name + " must be [" +
                                    std::to_string(rows) + "," + std::to_string(cols) + "]");
    }
}

} // namespace

Weight bf16_matrix(const void* data, int rows, int cols) {
    Weight weight;
    weight.qtype = QType::BF16;
    weight.layout = QuantLayout::Contiguous;
    weight.qdata = data;
    weight.n = rows;
    weight.k = cols;
    weight.ndim = 2;
    weight.shape[0] = rows;
    weight.shape[1] = cols;
    return weight;
}

std::size_t gemma4_layer_workspace_bytes(const Gemma4LayerWeights& weights, int tokens) {
    if (tokens <= 0 || weights.q_heads <= 0 || weights.kv_heads <= 0 || weights.head_dim <= 0 ||
        weights.hidden <= 0 || weights.intermediate <= 0) {
        throw std::invalid_argument("Gemma4 forward: workspace extents must be positive");
    }
    const auto q = static_cast<std::size_t>(weights.q_heads) * weights.head_dim;
    const auto k = static_cast<std::size_t>(weights.kv_heads) * weights.head_dim;
    const auto h = static_cast<std::size_t>(weights.hidden);
    const auto inter = static_cast<std::size_t>(weights.intermediate);
    // qkv (the sliding upper bound), qn, attention; hidden_n, projected; gate_up, activated.
    const auto elements = add_bytes(add_bytes(3 * q, 2 * k), add_bytes(2 * h, 3 * inter));
    auto down_scratch = projection_bytes(weights.down, weights.down_policy, tokens);
    if (ops::allows_a8(weights.down_policy)) {
        // GeGLU's ready codes/scales stay live while linear_a8_ready takes its own scratch.
        down_scratch = add_bytes(down_scratch, down_a8_bytes(weights.intermediate, tokens));
    }
    const auto scratch = std::max({projection_bytes(weights.qkv, weights.qkv_policy, tokens),
                                   projection_bytes(weights.o_proj, weights.o_policy, tokens),
                                   projection_bytes(weights.gate_up, weights.gate_up_policy, tokens),
                                   down_scratch,
                                   ops::gemma4_attention_workspace_capacity_bytes(
                                       weights.head_dim, weights.q_heads, weights.kv_heads, tokens)});
    return add_bytes(add_bytes(multiply_bytes(elements, static_cast<std::size_t>(tokens) * 2), scratch),
                     kAllocations * kAlignment);
}

std::size_t gemma4_logits_workspace_bytes(const Weight& output_head, int tokens) {
    if (tokens <= 0 || output_head.k <= 0) {
        throw std::invalid_argument("Gemma4 logits: workspace extents must be positive");
    }
    return add_bytes(add_bytes(multiply_bytes(static_cast<std::size_t>(output_head.k),
                                              static_cast<std::size_t>(tokens) * 2),
                                projection_bytes(output_head, ops::LinearPolicy::A16Only, tokens)),
                     2 * kAlignment);
}

namespace {

// hidden_n is rmsnorm(x,input_norm) on entry. The FFN tail leaves the next layer's input norm
// there when next_input_norm is present; otherwise it is scratch.
void layer_body(Tensor& x, Tensor& hidden_n, const Gemma4LayerWeights& weights,
                const void* next_input_norm, const Tensor& positions,
                const ops::Gemma4CacheView& cache, ops::Fp8RowCublasContext& gemm,
                WorkspaceArena& workspace, cudaStream_t stream) {
    const int tokens = x.ne[1];
    const int hidden = weights.hidden;
    const int dim = weights.head_dim;
    const int q_rows = weights.q_heads * dim;
    const int k_rows = weights.kv_heads * dim;
    const int inter = weights.intermediate;
    if (x.dtype != DType::BF16 || !x.is_contiguous() || x.ne[0] != hidden || tokens <= 0 ||
        x.ne[2] != 1 || x.ne[3] != 1) {
        throw std::invalid_argument("gemma4_layer: x must be contiguous BF16 [hidden,T]");
    }
    const int qkv_rows = q_rows + (weights.global_layer ? 1 : 2) * k_rows;
    require_matrix(weights.qkv, qkv_rows, hidden, "qkv");
    require_matrix(weights.o_proj, hidden, q_rows, "o_proj");
    require_matrix(weights.gate_up, 2 * inter, hidden, "gate_up");
    require_matrix(weights.down, hidden, inter, "down");

    auto qkv = workspace.alloc(DType::BF16, {qkv_rows, tokens});
    auto qn = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto attn = workspace.alloc(DType::BF16, {q_rows, tokens});
    auto projected = workspace.alloc(DType::BF16, {hidden, tokens});

    project(hidden_n, weights.qkv, weights.qkv_policy, qkv, gemm, workspace, stream);
    Tensor qn_heads(qn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_qkv_prep(qkv, norm_weight(weights.q_norm, dim), norm_weight(weights.k_norm, dim),
                         positions, weights.rotary_dim, weights.rope_theta, weights.rms_eps,
                         weights.global_layer, qn_heads, cache, stream);
    Tensor attn_heads(attn.data, DType::BF16, {dim, weights.q_heads, tokens});
    ops::gemma4_attention(qn_heads, cache, positions, weights.window, 1.0f, attn_heads,
                          workspace, stream);
    project(attn, weights.o_proj, weights.o_policy, projected, gemm, workspace, stream);
    const Tensor pre_ff_norm = norm_weight(weights.pre_ff_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(weights.post_attn_norm, hidden), 1.0f,
                              &pre_ff_norm, weights.rms_eps, x, &hidden_n, stream);

    auto gate_up = workspace.alloc(DType::BF16, {2 * inter, tokens});
    project(hidden_n, weights.gate_up, weights.gate_up_policy, gate_up, gemm, workspace, stream);
    const Tensor gate = gate_up.slice(0, 0, inter);
    const Tensor up = gate_up.slice(0, inter, inter);
    auto activated = workspace.alloc(DType::BF16, {inter, tokens});
    const bool fused_down = ops::linear_takes_a8_ready(weights.down, tokens, weights.down_policy);
    const DownA8 down_codes = fused_down ? prepare_down_a8(workspace, inter, tokens) : DownA8{};
    ops::geglu(gate, up, activated, stream, down_codes.codes, down_codes.scales);
    if (fused_down) {
        project_down_ready(activated, weights.down, weights.down_policy, projected, down_codes,
                           gemm, workspace, stream);
    } else {
        project(activated, weights.down, weights.down_policy, projected, gemm, workspace, stream);
    }
    const Tensor next_norm = norm_weight(next_input_norm, hidden);
    ops::gemma4_residual_norm(projected, norm_weight(weights.post_ff_norm, hidden),
                              weights.layer_scalar,
                              next_input_norm != nullptr ? &next_norm : nullptr, weights.rms_eps, x,
                              next_input_norm != nullptr ? &hidden_n : nullptr, stream);
}

} // namespace

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  const ops::Gemma4CacheView& cache, ops::Fp8RowCublasContext& gemm,
                  WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    auto hidden_n = workspace.alloc(DType::BF16, {weights.hidden, x.ne[1]});
    ops::rmsnorm(x, norm_weight(weights.input_norm, weights.hidden), weights.rms_eps, false,
                 hidden_n, stream);
    layer_body(x, hidden_n, weights, nullptr, positions, cache, gemm, workspace, stream);
}

void gemma4_layers(const Tensor& ids, Tensor& hidden, const Gemma4Embedding& token_embedding,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (layers.size() != caches.size()) {
        throw std::invalid_argument("gemma4_layers: one cache per layer");
    }
    ops::embedding(ids, token_embedding.table, token_embedding.scale, hidden, stream);
    if (layers.empty()) { return; }
    auto scope = workspace.scope();
    auto hidden_n = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(hidden, norm_weight(layers[0].input_norm, hidden.ne[0]), layers[0].rms_eps, false,
                 hidden_n, stream);
    for (std::size_t layer = 0; layer < layers.size(); ++layer) {
        auto layer_scope = workspace.scope();
        const void* next = layer + 1 < layers.size() ? layers[layer + 1].input_norm : nullptr;
        layer_body(hidden, hidden_n, layers[layer], next, positions, caches[layer], gemm,
                   workspace, stream);
    }
}

void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const Weight& output_head, float softcap, float rms_eps,
                   ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (hidden.ne[1] != logits.ne[1]) {
        throw std::invalid_argument("gemma4_logits: one logits column per hidden column");
    }
    if (output_head.n != logits.ne[0] || output_head.k != hidden.ne[0]) {
        throw std::invalid_argument("gemma4_logits: output head must be [vocab,hidden]");
    }
    auto scope = workspace.scope();
    auto normed = workspace.alloc(DType::BF16, {hidden.ne[0], hidden.ne[1]});
    ops::rmsnorm(hidden, norm_weight(final_norm, hidden.ne[0]), rms_eps, false, normed, stream);
    project(normed, output_head, ops::LinearPolicy::A16Only, logits, gemm, workspace, stream);
    ops::logit_softcap(logits, softcap, stream);
}

void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits,
                    const Gemma4Embedding& token_embedding, const void* final_norm,
                    const Weight& output_head, float softcap, float rms_eps,
                    std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    gemma4_layers(ids, hidden, token_embedding, layers, caches, positions, gemm, workspace, stream);
    gemma4_logits(hidden, logits, final_norm, output_head, softcap, rms_eps, gemm, workspace, stream);
}

} // namespace infernix::models::gemma4
