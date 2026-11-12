#pragma once

// Logical parameter handles of a loaded Gemma 4 model. Names follow the converter
// (tools/convert/gemma4.py); handles index the frozen Model's bound weights.

#include "core/weight_view.h"
#include "infernix/ops/linear.h"

#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace infernix::models::gemma4 {

struct WeightId {
    std::size_t index                          = std::numeric_limits<std::size_t>::max();
    friend bool operator==(WeightId, WeightId) = default;
    [[nodiscard]] bool valid() const noexcept { return index != std::numeric_limits<std::size_t>::max(); }
};

struct WeightUseId {
    WeightId parameter;
    std::size_t use_index = std::numeric_limits<std::size_t>::max();
};

// One mathematical use of a parameter, as the artifact declares it. Gemma 4 formats carry no Use
// auxiliaries (no activation divisors or scales).
struct WeightUse {
    std::string input;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

struct BoundWeight {
    std::string name;
    std::vector<std::string> source_objects; // artifact object ID of each view part
    WeightView view;
    std::vector<WeightUse> uses;
};

// Query, key and value (rows of one stored parent, in this order) of a sliding layer.
struct SlidingAttentionWeights {
    WeightId query, key, value;
    WeightId query_norm, key_norm, output;
};

// Global layers store no value projection: V is K (the key's rows are both).
struct GlobalAttentionWeights {
    WeightId query, key;
    WeightId query_norm, key_norm, output;
};

struct LayerWeights {
    std::variant<SlidingAttentionWeights, GlobalAttentionWeights> attention;
    // gate and up are the consecutive row ranges of one stored parent.
    WeightId gate, up, down;
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    // FP32 [1], Host-resident; layer_scalar_value is its value.
    WeightId layer_scalar;
    float layer_scalar_value = 1.0F;
};

struct TextWeights {
    // The vocabulary table and the output head: two parameters (the head's Use is text/final_hidden,
    // A16Only) over the same stored object when embedding_shared. A shared table is the unscaled
    // one: the embedding gather applies TextConfig::embed_normalizer. Otherwise the table is stored
    // prescaled.
    WeightId token_embedding, output_head, final_norm;
    WeightUseId output_head_use;
    bool embedding_shared = false;
    std::vector<LayerWeights> layers;
};

// A query-only assistant layer: its K/V are the target's (AssistantConfig::kv_source_layers).
struct MtpLayerWeights {
    WeightId query, output;
    WeightId gate, up, down;
    WeightId query_norm;
    WeightId input_norm, post_attention_norm, pre_feedforward_norm, post_feedforward_norm;
    WeightId layer_scalar;
    float layer_scalar_value = 1.0F;
};

// Every Use is A16Only. output_projection and output_head each have one Use (mtp/final_hidden).
struct MtpWeights {
    WeightId input_projection, output_projection, output_head, final_norm;
    WeightUseId output_head_use;
    std::vector<MtpLayerWeights> layers;
};

// Handles refer to the frozen model's weight array. No artifact ID lookup is needed in execution.
struct ModelWeights {
    TextWeights text;
    std::optional<MtpWeights> mtp;
};

} // namespace infernix::models::gemma4
