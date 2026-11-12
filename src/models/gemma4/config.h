#pragma once

// Gemma 4 instance configuration as recorded by the converter (tools/convert/gemma4.py
// text_config / assistant_config). The mathematics is fixed by this implementation; the config only
// supplies dimensions, RoPE geometry and the MTP assistant's wiring to the target's KV.

#include "models/load_options.h"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace infernix::artifact {
struct Directory;
}

namespace infernix::models::gemma4 {

// Sliding-window attention (head_dim, num_key_value_heads, window) or global attention
// (global_head_dim, num_global_key_value_heads, K == V, no window).
enum class LayerKind : std::uint8_t { Sliding, Global };

struct TextConfig {
    std::uint32_t hidden_size             = 0;
    std::uint32_t vocab_size              = 0;
    std::uint32_t num_hidden_layers       = 0;
    std::uint32_t max_position_embeddings = 0;
    std::uint32_t intermediate_size       = 0;
    std::uint32_t num_attention_heads     = 0;
    // Sliding layers (rotated over the whole head) and global layers (rotated over prope_rotary_dim).
    std::uint32_t head_dim                   = 0;
    std::uint32_t global_head_dim            = 0;
    std::uint32_t num_key_value_heads        = 0;
    std::uint32_t num_global_key_value_heads = 0;
    std::vector<LayerKind> layer_types;
    std::uint32_t sliding_window = 0;
    float rms_norm_eps           = 0;
    float final_logit_softcapping = 0;
    // bf16(sqrt(hidden_size)): the forward multiplies the gathered rows by it when the token table is
    // the unscaled one the output head shares (TextWeights::embedding_shared).
    float embed_normalizer = 0;
    float rope_sliding_theta = 0;
    float rope_global_theta  = 0;
    float partial_rotary_factor = 0;
    // Global layers: the proportional RoPE's pairs, moved by the converter onto a contiguous prefix
    // of prope_rotary_dim = 2 * prope_pairs dimensions, rotating with prope_theta_eff.
    std::uint32_t prope_pairs      = 0;
    std::uint32_t prope_rotary_dim = 0;
    float prope_theta_eff          = 0;
    // Global layers whose key norm weight is one constant (layer index -> value); parsed and kept.
    std::map<std::uint32_t, float> global_k_norm_scalar;
    // Computed from layer_types; empty when the model has no layer of that kind.
    std::optional<std::uint32_t> last_sliding_layer;
    std::optional<std::uint32_t> last_global_layer;

    [[nodiscard]] std::uint32_t head_dim_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? global_head_dim : head_dim;
    }
    [[nodiscard]] std::uint32_t kv_heads_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? num_global_key_value_heads : num_key_value_heads;
    }
    // The window of a sliding layer; 0 (no window) for a global one.
    [[nodiscard]] std::uint32_t window_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? 0 : sliding_window;
    }
    [[nodiscard]] std::uint32_t query_width(LayerKind kind) const noexcept {
        return num_attention_heads * head_dim_of(kind);
    }
    [[nodiscard]] std::uint32_t key_width(LayerKind kind) const noexcept {
        return kv_heads_of(kind) * head_dim_of(kind);
    }
};

// The MTP assistant (component `mtp`, architecture Gemma4Assistant): every layer is query-only and
// attends to the K/V of one target layer, kv_source_layers[i], of its own kind.
struct AssistantConfig {
    std::uint32_t hidden_size          = 0;
    std::uint32_t backbone_hidden_size = 0; // the target's hidden size
    std::uint32_t intermediate_size    = 0;
    std::uint32_t num_hidden_layers    = 0;
    std::uint32_t num_attention_heads  = 0;
    std::uint32_t head_dim                   = 0;
    std::uint32_t global_head_dim            = 0;
    std::uint32_t num_key_value_heads        = 0;
    std::uint32_t num_global_key_value_heads = 0;
    std::vector<LayerKind> layer_types;
    std::vector<std::uint32_t> kv_source_layers; // target layer index per assistant layer
    std::uint32_t sliding_window = 0;
    float rms_norm_eps           = 0;
    std::uint32_t vocab_size     = 0; // the target's
    float rope_sliding_theta     = 0;
    std::uint32_t prope_rotary_dim = 0;
    float prope_theta_eff          = 0;

    [[nodiscard]] std::uint32_t head_dim_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? global_head_dim : head_dim;
    }
    [[nodiscard]] std::uint32_t kv_heads_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? num_global_key_value_heads : num_key_value_heads;
    }
    [[nodiscard]] std::uint32_t window_of(LayerKind kind) const noexcept {
        return kind == LayerKind::Global ? 0 : sliding_window;
    }
    [[nodiscard]] std::uint32_t query_width(LayerKind kind) const noexcept {
        return num_attention_heads * head_dim_of(kind);
    }
};

struct Config {
    TextConfig text;
    // Present when the load selects the MTP assistant (LoadOptions::mtp()).
    std::optional<AssistantConfig> mtp;
};

// Strict parse of the artifact's `text` component and, with LoadOptions::mtp(), its `mtp` component.
// Options Gemma 4 does not implement (vision, the lossless route, other speculative backends, ...)
// are refused here.
[[nodiscard]] Config parse_config(const artifact::Directory& directory, const LoadOptions& options);

} // namespace infernix::models::gemma4
