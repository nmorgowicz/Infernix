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

// A prepared projection: the native Weight (or, for GGUF blocks, a descriptor of [N,K]), its policy and,
// for GGUF blocks, the segments ops::ggml_linear runs.
struct PreparedProjection {
    Weight weight;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    GgmlProjection ggml;
};

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

    PreparedProjection linear(std::initializer_list<WeightId> ids, std::string_view input_name) const {
        std::vector<ops::WeightInput> rows;
        rows.reserve(ids.size());
        for (const auto id : ids) { rows.push_back(input(id, input_name)); }
        if (rows.empty()) { throw std::invalid_argument("Gemma4 parameters: empty projection"); }
        if (ggml(ids)) { return ggml_linear(ids, rows); }
        if (model_.weights_resident()) {
            const auto single = ops::prepare_linear_weight(std::span<const ops::WeightInput>(rows));
            return {single.weight, single.policy, {}};
        }
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
        return {matrix_geometry(joined), rows.front().policy, {}};
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
    // Whether the parameters are stored as GGUF blocks (ggml_rows_v1); a projection mixing them with native
    // formats is refused.
    bool ggml(std::initializer_list<WeightId> ids) const {
        bool any = false, all = true;
        for (const auto id : ids) {
            for (const auto& part : model_.weight(id).view.parts) {
                const bool g = is_ggml_matrix_layout(part.parent->geometry.layout);
                any = any || g;
                all = all && g;
            }
        }
        if (any && !all) {
            throw std::invalid_argument(model_.weight(*ids.begin()).name +
                                        ": a projection mixes GGUF blocks with native formats");
        }
        return any;
    }

    // The projection's rows in order: one segment per region of a GGUF parent (no device address for a
    // geometry-only Model).
    PreparedProjection ggml_linear(std::initializer_list<WeightId> ids, const std::vector<ops::WeightInput>& rows) const {
        // input() admits A16Only and AllowA8; the routes follow the policy, so a fused projection needs one.
        for (const auto& row : rows) {
            if (row.policy != rows.front().policy) {
                throw std::invalid_argument(model_.weight(*ids.begin()).name +
                                            ": a fused GGUF projection's parameters need one activation policy");
            }
        }
        PreparedProjection out;
        std::int32_t row = 0;
        std::uint64_t k  = 0;
        for (const auto id : ids) {
            const auto& view = model_.weight(id).view;
            if (view.shape.size() != 2) { throw std::invalid_argument("a GGUF projection parameter must be a matrix"); }
            if (k == 0) { k = view.shape[1]; }
            if (view.shape[1] != k) { throw std::invalid_argument("a GGUF projection's parameters must share K"); }
            for (const auto& part : view.parts) {
                const auto& parent = *part.parent;
                const auto& g      = parent.geometry;
                if (g.layout != QuantLayout::GgmlRows) {
                    throw std::invalid_argument(model_.weight(id).name + ": Gemma4 binds GGUF blocks as ggml_rows_v1");
                }
                if (g.shape[1] != k || part.begin % k || part.end % k) {
                    throw std::invalid_argument("a GGUF projection region must be whole rows");
                }
                ops::GgmlSegment s;
                s.format = g.format;
                if (model_.weights_resident()) {
                    const std::byte* base = parent.device != nullptr ? parent.device : parent.data;
                    if (base == nullptr) { throw std::invalid_argument("a GGUF projection needs device-visible weights"); }
                    s.data = reinterpret_cast<const std::uint8_t*>(base + part.begin / k * g.code_bytes_per_row);
                }
                s.rows       = static_cast<std::int32_t>((part.end - part.begin) / k);
                s.out_offset = row;
                s.layout     = ops::GgmlLayout::Rows;
                row += s.rows;
                out.ggml.segments.push_back(s);
            }
        }
        out.weight.qtype         = out.ggml.segments.front().format;
        out.weight.layout        = QuantLayout::GgmlRows;
        out.weight.ndim          = 2;
        out.weight.n             = out.weight.shape[0] = out.weight.padded_shape[0] = row;
        out.weight.k             = out.weight.shape[1] = out.weight.padded_shape[1] = dimension(k);
        out.policy               = rows.front().policy;
        return out;
    }

    const Model& model_;
};

} // namespace

std::vector<std::int32_t> global_output_input_perm(const TextConfig& c) {
    const int dim = static_cast<int>(c.global_head_dim), pairs = static_cast<int>(c.prope_pairs), half = dim / 2;
    std::vector<int> perm;
    for (int i = 0; i < pairs; ++i) { perm.push_back(i); }
    for (int i = 0; i < pairs; ++i) { perm.push_back(half + i); }
    for (int i = 0; i < dim; ++i) {
        if (!(i < pairs || (i >= half && i < half + pairs))) { perm.push_back(i); }
    }
    std::vector<int> position(dim);
    for (int p = 0; p < dim; ++p) { position[perm[p]] = p; }
    if (dim % kGlobalOutputPermBlock != 0) {
        throw std::invalid_argument("Gemma4 parameters: global head dim is not whole permutation blocks");
    }
    const int per_head = dim / kGlobalOutputPermBlock;
    std::vector<std::int32_t> blocks;
    for (int b = 0; b < per_head; ++b) {
        const int start = position[b * kGlobalOutputPermBlock];
        for (int e = 0; e < kGlobalOutputPermBlock; ++e) {
            if (position[b * kGlobalOutputPermBlock + e] != start + e || start % kGlobalOutputPermBlock != 0) {
                throw std::invalid_argument("Gemma4 parameters: prope_perm does not move whole 64-element blocks");
            }
        }
        blocks.push_back(start / kGlobalOutputPermBlock);
    }
    std::vector<std::int32_t> out;
    for (std::uint32_t h = 0; h < c.num_attention_heads; ++h) {
        for (const auto b : blocks) { out.push_back(static_cast<std::int32_t>(h * per_head + b)); }
    }
    return out;
}

ParameterGeometry Parameters::geometry(const Model& source) {
    const Prepare prepare(source);
    const auto& weights = source.weights().text;
    const auto& config = source.config().text;
    if (source.config().mtp.has_value() != source.weights().mtp.has_value()) {
        throw std::invalid_argument("Gemma4 parameters: the MTP config and weights disagree");
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
    const auto q5_routes = q5_prefill_routes(source.options().gemma4_q5_prefill);
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
        layer.q5_routes = q5_routes;
        PreparedProjection qkv;
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
        layer.qkv_ggml = qkv.ggml;
        layer.o_ggml = o.ggml;
        layer.gate_up_ggml = gate_up.ggml;
        layer.down_ggml = down.ggml;
        if (layer.global_layer && config.global_output_source_order != layer.o_ggml.active()) {
            throw std::invalid_argument(prefix + "attention/output: global_output_order \"source\" and a GGUF-block "
                                                 "global output projection come together");
        }
        // The permutation's device array is uploaded by the resident constructor.
        if (layer.global_layer && layer.o_ggml.active()) { layer.o_ggml.perm_block = kGlobalOutputPermBlock; }
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
    if (config.global_output_source_order) {
        out.derived_device_bytes += global_output_input_perm(config).size() * sizeof(std::int32_t);
    }
    if (const auto& assistant = source.config().mtp) {
        const auto& mtp = *source.weights().mtp;
        if (mtp.layers.size() != assistant->num_hidden_layers || mtp.layers.empty() ||
            assistant->kv_source_layers.size() != mtp.layers.size()) {
            throw std::invalid_argument("Gemma4 parameters: one weight block per configured MTP layer");
        }
        if (mtp.output_head_use.parameter != mtp.output_head) {
            throw std::invalid_argument("Gemma4 parameters: MTP head Use refers to another parameter");
        }
        Gemma4DraftWeights draft;
        draft.hidden = dimension(assistant->hidden_size);
        draft.backbone = dimension(assistant->backbone_hidden_size);
        draft.vocab = dimension(assistant->vocab_size);
        draft.rms_eps = assistant->rms_norm_eps;
        draft.input_projection = prepare.linear({mtp.input_projection}, "mtp/stem_input").weight;
        draft.output_projection = prepare.linear({mtp.output_projection}, "mtp/final_hidden").weight;
        draft.output_head = prepare.linear({mtp.output_head}, "mtp/final_hidden").weight;
        draft.final_norm = prepare.norm(mtp.final_norm, draft.hidden);
        draft.layers.reserve(mtp.layers.size());
        for (std::size_t i = 0; i < mtp.layers.size(); ++i) {
            const auto& stored = mtp.layers[i];
            const auto kind = assistant->layer_types[i];
            const auto prefix = "mtp/layers/" + std::to_string(i) + "/";
            Gemma4LayerWeights layer;
            layer.hidden = draft.hidden;
            layer.intermediate = dimension(assistant->intermediate_size);
            layer.head_dim = dimension(assistant->head_dim_of(kind));
            layer.q_heads = dimension(assistant->num_attention_heads);
            layer.kv_heads = dimension(assistant->kv_heads_of(kind));
            layer.window = dimension(assistant->window_of(kind));
            layer.rotary_dim = dimension(kind == LayerKind::Global ? assistant->prope_rotary_dim : assistant->head_dim);
            layer.rope_theta = kind == LayerKind::Global ? assistant->prope_theta_eff : assistant->rope_sliding_theta;
            layer.rms_eps = assistant->rms_norm_eps;
            layer.global_layer = kind == LayerKind::Global;
            const auto query = prepare.linear({stored.query}, prefix + "mixer_input");
            const auto o = prepare.linear({stored.output}, prefix + "attention_output");
            const auto gate_up = prepare.linear({stored.gate, stored.up}, prefix + "ffn_input");
            const auto down = prepare.linear({stored.down}, prefix + "mlp/product");
            layer.qkv = query.weight;
            layer.qkv_policy = query.policy;
            layer.o_proj = o.weight;
            layer.o_policy = o.policy;
            layer.gate_up = gate_up.weight;
            layer.gate_up_policy = gate_up.policy;
            layer.down = down.weight;
            layer.down_policy = down.policy;
            layer.q_norm = prepare.norm(stored.query_norm, layer.head_dim);
            layer.input_norm = prepare.norm(stored.input_norm, layer.hidden);
            layer.post_attn_norm = prepare.norm(stored.post_attention_norm, layer.hidden);
            layer.pre_ff_norm = prepare.norm(stored.pre_feedforward_norm, layer.hidden);
            layer.post_ff_norm = prepare.norm(stored.post_feedforward_norm, layer.hidden);
            layer.layer_scalar = stored.layer_scalar_value;
            draft.layers.push_back(layer);
            draft.kv_source.push_back(dimension(assistant->kv_source_layers[i]));
        }
        out.draft = std::move(draft);
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
    draft = std::move(prepared.draft);
    for (auto& layer : layers) {
        for (Weight* projection : {&layer.qkv, &layer.o_proj, &layer.gate_up, &layer.down}) {
            ops::linear_prepare_a8_rows(*projection, row_scales_.emplace_back(), stream);
            derived_device_bytes_ += row_scales_.back().bytes;
        }
    }
    if (source.config().text.global_output_source_order) {
        const auto perm = global_output_input_perm(source.config().text);
        output_perm_ = DeviceBuffer(perm.size() * sizeof(std::int32_t));
        output_perm_.copy_from_host(perm.data(), perm.size() * sizeof(std::int32_t));
        derived_device_bytes_ += output_perm_.bytes;
        for (auto& layer : layers) {
            if (layer.o_ggml.perm_block != 0) {
                layer.o_ggml.input_perm = static_cast<const std::int32_t*>(output_perm_.p);
            }
        }
    }
}

} // namespace infernix::models::gemma4::execution
