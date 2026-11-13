#include "models/gemma4/execution/parameters.h"

#include "core/weight_view.h"
#include "infernix/ops/weight_input.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace infernix::models::gemma4::execution {
namespace {

int dimension(std::uint64_t value) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("Gemma4 parameters: dimension exceeds native tensor range");
    }
    return static_cast<int>(value);
}

// Workspace planning needs codec/extent metadata, not resident addresses. Never manufacture
// device pointers for plan_model(): the execution bridge deliberately rejects that Model.
Weight matrix_geometry(const WeightView& view) {
    const auto region = contiguous_weight_region(view);
    const auto& g = region.parent->geometry;
    if (view.shape.size() != 2 ||
        (g.layout != QuantLayout::Contiguous && (g.shape.size() != 2 || view.shape[1] != g.shape[1])) ||
        (g.layout == QuantLayout::RowScale && !is_complete_weight(view))) {
        throw std::invalid_argument("Gemma4 parameters: unsupported logical matrix geometry");
    }
    Weight out;
    out.qtype = g.format;
    out.layout = g.layout;
    out.ndim = 2;
    out.n = out.shape[0] = out.padded_shape[0] = dimension(view.shape[0]);
    out.k = out.shape[1] = dimension(view.shape[1]);
    out.padded_shape[1] = g.layout == QuantLayout::Contiguous ? out.k : dimension(g.padded_columns);
    out.group = dimension(g.group_size);
    out.group_size = static_cast<std::uint32_t>(out.group);
    return out;
}
class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    Weight matrix(const WeightView& view) const {
        return model_.weights_resident() ? device_native_weight(view) : matrix_geometry(view);
    }

    ops::WeightInput input(WeightId id, std::string_view input_name) const {
        const auto& bound = model_.weight(id);
        const auto found = std::find_if(bound.uses.begin(), bound.uses.end(), [&](const WeightUse& use) {
            return use.input == input_name;
        });
        if (found == bound.uses.end()) {
            throw std::invalid_argument(bound.name + ": missing Use " + std::string(input_name));
        }
        // P4 implements A16 and the artifact-permitted A8 routes, never calibrated A4.
        if (found->policy != ops::LinearPolicy::A16Only && found->policy != ops::LinearPolicy::AllowA8) {
            throw std::invalid_argument(bound.name + ": Gemma4 execution supports only A16Only or AllowA8");
        }
        return {bound.view, found->policy, std::nullopt, std::nullopt};
    }

    ops::SingleProjectionWeight linear(std::initializer_list<WeightId> ids,
                                       std::string_view input_name) const {
        std::vector<ops::WeightInput> rows;
        rows.reserve(ids.size());
        for (const auto id : ids) { rows.push_back(input(id, input_name)); }
        if (model_.weights_resident()) {
            return ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows));
        }
        if (rows.empty()) { throw std::invalid_argument("Gemma4 parameters: empty projection"); }
        WeightView joined;
        joined.shape = {0, rows.front().weight.shape.at(1)};
        for (const auto& row : rows) {
            if (row.weight.shape.size() != 2 || row.weight.shape[1] != joined.shape[1] ||
                row.policy != rows.front().policy) {
                throw std::invalid_argument("Gemma4 parameters: fused rows need one K and policy");
            }
            joined.shape[0] += row.weight.shape[0];
            joined.parts.insert(joined.parts.end(), row.weight.parts.begin(), row.weight.parts.end());
        }
        return {matrix_geometry(joined), rows.front().policy};
    }

    const void* norm(WeightId id, int width) const {
        const auto& bound = model_.weight(id);
        const auto region = contiguous_weight_region(bound.view);
        const auto& g = region.parent->geometry;
        if (g.format != QType::BF16 || g.layout != QuantLayout::Contiguous ||
            bound.view.shape != std::vector<std::uint64_t>{static_cast<std::uint64_t>(width)}) {
            throw std::invalid_argument(bound.name + ": expected contiguous BF16 norm vector");
        }
        if (!model_.weights_resident()) { return nullptr; }
        if (!region.parent->device) { throw std::invalid_argument(bound.name + ": norm is not device-visible"); }
        return region.parent->device + region.begin * sizeof(std::uint16_t);
    }

private:
    const Model& model_;
};

} // namespace

ParameterGeometry Parameters::geometry(const Model& source) {
    const Prepare prepare(source);
    const auto& weights = source.weights().text;
    const auto& config = source.config().text;
    if (source.config().mtp) {
        throw std::invalid_argument("Gemma4 P4 execution requires speculative decoding off");
    }
    if (weights.layers.size() != config.layer_types.size() ||
        weights.layers.size() != config.num_hidden_layers || weights.layers.empty()) {
        throw std::invalid_argument("Gemma4 parameters: one weight block per configured text layer");
    }
    ParameterGeometry out;
    out.embedding.table = prepare.matrix(source.weight(weights.token_embedding).view);
    out.embedding.scale = weights.embedding_shared ? config.embed_normalizer : 1.0f;
    out.final_norm = prepare.norm(weights.final_norm, dimension(config.hidden_size));
    if (weights.output_head_use.parameter != weights.output_head) {
        throw std::invalid_argument("Gemma4 parameters: output head Use refers to another parameter");
    }
    const auto& head_bound = source.weight(weights.output_head);
    const auto& head_use = head_bound.uses.at(weights.output_head_use.use_index);
    if (head_use.policy != ops::LinearPolicy::A16Only || head_use.input != "text/final_hidden") {
        throw std::invalid_argument("Gemma4 parameters: output head must use A16Only text/final_hidden");
    }
    out.output_head = prepare.linear({weights.output_head}, head_use.input).weight;
    out.layers.reserve(weights.layers.size());
    for (std::size_t i = 0; i < weights.layers.size(); ++i) {
        const auto& stored = weights.layers[i];
        const auto kind = config.layer_types[i];
        const auto prefix = "text/layers/" + std::to_string(i) + "/";
        Gemma4LayerWeights layer;
        layer.hidden = dimension(config.hidden_size);
        layer.intermediate = dimension(config.intermediate_size);
        layer.head_dim = dimension(config.head_dim_of(kind));
        layer.q_heads = dimension(config.num_attention_heads);
        layer.kv_heads = dimension(config.kv_heads_of(kind));
        layer.window = dimension(config.window_of(kind));
        layer.rotary_dim = dimension(kind == LayerKind::Global ? config.prope_rotary_dim :
                                                                         config.head_dim);
        layer.rope_theta = kind == LayerKind::Global ? config.prope_theta_eff : config.rope_sliding_theta;
        layer.rms_eps = config.rms_norm_eps;
        layer.global_layer = kind == LayerKind::Global;
        ops::SingleProjectionWeight qkv;
        WeightId q_norm, k_norm, output;
        if (layer.global_layer) {
            const auto& attention = std::get<GlobalAttentionWeights>(stored.attention);
            qkv = prepare.linear({attention.query, attention.key}, prefix + "mixer_input");
            q_norm = attention.query_norm;
            k_norm = attention.key_norm;
            output = attention.output;
        } else {
            const auto& attention = std::get<SlidingAttentionWeights>(stored.attention);
            qkv = prepare.linear({attention.query, attention.key, attention.value}, prefix + "mixer_input");
            q_norm = attention.query_norm;
            k_norm = attention.key_norm;
            output = attention.output;
        }
        const auto o = prepare.linear({output}, prefix + "attention_output");
        const auto gate_up = prepare.linear({stored.gate, stored.up}, prefix + "ffn_input");
        const auto down = prepare.linear({stored.down}, prefix + "mlp/product");
        layer.qkv = qkv.weight;
        layer.qkv_policy = qkv.policy;
        layer.q_norm = prepare.norm(q_norm, layer.head_dim);
        layer.k_norm = prepare.norm(k_norm, layer.head_dim);
        layer.o_proj = o.weight;
        layer.o_policy = o.policy;
        layer.gate_up = gate_up.weight;
        layer.gate_up_policy = gate_up.policy;
        layer.down = down.weight;
        layer.down_policy = down.policy;
        layer.input_norm = prepare.norm(stored.input_norm, layer.hidden);
        layer.post_attn_norm = prepare.norm(stored.post_attention_norm, layer.hidden);
        layer.pre_ff_norm = prepare.norm(stored.pre_feedforward_norm, layer.hidden);
        layer.post_ff_norm = prepare.norm(stored.post_feedforward_norm, layer.hidden);
        layer.layer_scalar = stored.layer_scalar_value;
        for (const Weight* projection : {&layer.qkv, &layer.o_proj, &layer.gate_up, &layer.down}) {
            const auto bytes = ops::linear_a8_rows_bytes(*projection);
            if (bytes > std::numeric_limits<std::size_t>::max() - out.derived_device_bytes) {
                throw std::overflow_error("Gemma4 parameters: prepared scale size overflow");
            }
            out.derived_device_bytes += bytes;
        }
        out.layers.push_back(layer);
    }
    return out;
}

Parameters::Parameters(const Model& source, cudaStream_t stream) : model(source) {
    if (!source.weights_resident()) {
        throw std::invalid_argument("Gemma4 parameters: resident execution requires materialized weights");
    }
    auto prepared = geometry(source);
    embedding = prepared.embedding;
    layers = std::move(prepared.layers);
    final_norm = prepared.final_norm;
    output_head = prepared.output_head;
    row_scales_.reserve(layers.size() * 4);
    for (auto& layer : layers) {
        for (Weight* projection : {&layer.qkv, &layer.o_proj, &layer.gate_up, &layer.down}) {
            ops::linear_prepare_a8_rows(*projection, row_scales_.emplace_back(), stream);
            derived_device_bytes_ += row_scales_.back().bytes;
        }
    }
}

} // namespace infernix::models::gemma4::execution
