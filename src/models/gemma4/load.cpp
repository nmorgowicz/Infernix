#include "models/gemma4/load.h"

#include "artifact/binder.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "artifact/views.h"
#include "core/weight_view.h"
#include "models/registry.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::gemma4 {
namespace {

using artifact::ArtifactError;
using artifact::Residency;
using artifact::Shape;

// The stored formats of a projection, the vocabulary table and the MTP head: BF16, the FP8 row table
// (the tied vocabulary) and the coarse-group Q4 weights.
constexpr QType kMatrixFormats[] = {QType::BF16, QType::FP8_E4M3FN_ROW_BF16, QType::Q4_G32_FP16,
                                    QType::Q4_G128_FP16, QType::Q4_G256_FP16};

std::string allowed_formats() {
    std::string out;
    for (const QType format : kMatrixFormats) {
        out += out.empty() ? "" : ", ";
        out += artifact::format_name(format);
    }
    return out;
}

ops::LinearPolicy linear_policy(artifact::ActivationPolicy policy) {
    switch (policy) {
    case artifact::ActivationPolicy::A16Only:
        return ops::LinearPolicy::A16Only;
    case artifact::ActivationPolicy::AllowA8:
        return ops::LinearPolicy::AllowA8;
    case artifact::ActivationPolicy::AllowA4:
        return ops::LinearPolicy::AllowA4;
    }
    throw ArtifactError("unknown activation policy");
}

struct PendingWeight {
    artifact::ParameterReference reference;
    std::vector<WeightUse> uses;
    std::vector<std::string> source_objects;
};

// Logical demands of the selected model; each parameter is declared exactly once.
class Bindings {
public:
    explicit Bindings(artifact::Binder& binder) : binder(binder) {}

    // A stored matrix [rows, K] (a projection, the vocabulary table or a head): device-resident, one of
    // kMatrixFormats, whole rows, and a complete stored parent where its layout needs one. `inputs` are
    // the Uses the artifact declares for it (each with an activation policy and no auxiliaries).
    WeightId matrix(std::string name, Shape shape, std::vector<std::string> inputs = {},
                    bool grouped = false) {
        if (shape.size() != 2) { throw ArtifactError(name + ": a matrix parameter is two-dimensional"); }
        PendingWeight pending;
        pending.reference = binder.parameter(name, shape);
        const auto& reader = binder.reader();
        for (const auto& part : pending.reference.binding.parts) {
            const QType format = reader.geometry(part.object).format;
            if (std::find(std::begin(kMatrixFormats), std::end(kMatrixFormats), format) ==
                std::end(kMatrixFormats)) {
                throw ArtifactError(name + ": stored format " + std::string(artifact::format_name(format)) +
                                    " is not supported (Gemma4 binds " + allowed_formats() + ")");
            }
            if (part.begin % shape[1] != 0 || part.end % shape[1] != 0) {
                throw ArtifactError(name + ": a matrix binds whole rows of its stored parent");
            }
        }
        for (const auto& input : inputs) {
            const auto& use = binder.use(name, input);
            if (!use.activation_policy) {
                throw ArtifactError(name + "@" + input + ": missing activation policy");
            }
            if (!use.auxiliaries.empty()) {
                throw ArtifactError(name + "@" + input + ": Gemma4 weights carry no Use auxiliaries");
            }
            pending.uses.push_back({input, linear_policy(*use.activation_policy)});
        }
        const WeightId id = declare(std::move(name), std::move(pending));
        if (!grouped) { group({id}); }
        return id;
    }

    // A vector read directly by an Op: stored in exactly `format` (BF16 norm weights).
    WeightId direct(std::string name, Shape shape, QType format = QType::BF16) {
        PendingWeight pending;
        pending.reference = binder.parameter(name, std::move(shape), Residency::Device, format);
        return declare(std::move(name), std::move(pending));
    }

    // A layer's FP32 scalar: Host-resident, its value read here.
    WeightId scalar(std::string name, float& value) {
        PendingWeight pending;
        pending.reference = binder.parameter(name, {1}, Residency::Host, QType::FP32);
        value             = binder.values(pending.reference.binding, QType::FP32).scalar_f32();
        if (!std::isfinite(value)) { throw ArtifactError(name + ": layer scalar must be finite"); }
        return declare(std::move(name), std::move(pending));
    }

    [[nodiscard]] const PendingWeight& at(WeightId id) const { return weights.at(id.index); }

    [[nodiscard]] WeightUseId use(WeightId id, std::string_view input) const {
        const auto& parameter = at(id);
        for (std::size_t i = 0; i < parameter.uses.size(); ++i) {
            if (parameter.uses[i].input == input) { return {id, i}; }
        }
        throw ArtifactError(parameter.reference.name + ": unresolved Use " + std::string(input));
    }

    // Every Use of `id` is A16Only.
    void require_a16(WeightId id) const {
        for (const auto& use : at(id).uses) {
            if (use.policy != ops::LinearPolicy::A16Only) {
                throw ArtifactError(at(id).reference.name + "@" + use.input + ": must be A16Only");
            }
        }
    }

    // The parameters, in order, are consecutive row ranges of ONE stored matrix (the rows q/k/(v) or
    // gate/up share), which one native Weight spans (contiguous_weight_region). A stored layout that
    // addresses only its complete parent (FP8 rows) must be filled exactly.
    void group(std::initializer_list<WeightId> ids) const {
        std::optional<artifact::Part> run;
        std::string label;
        for (const WeightId id : ids) {
            const auto& reference = at(id).reference;
            label += label.empty() ? "" : "+";
            label += reference.name;
            for (const auto& part : reference.binding.parts) {
                if (!run) {
                    run = part;
                } else if (part.object != run->object || part.begin != run->end) {
                    throw ArtifactError(label + ": must be consecutive row ranges of one stored matrix");
                } else {
                    run->end = part.end;
                }
            }
        }
        if (!run) { throw ArtifactError("empty parameter group"); }
        const auto& geometry = binder.reader().geometry(run->object);
        if (geometry.layout != QuantLayout::Contiguous && geometry.layout != QuantLayout::RowSplit &&
            (run->begin != 0 || run->end != geometry.elements)) {
            throw ArtifactError(label + ": a " + std::string(artifact::format_name(geometry.format)) +
                                " matrix binds its complete stored parent");
        }
    }

    artifact::Binder& binder;
    std::vector<PendingWeight> weights;

private:
    WeightId declare(std::string name, PendingWeight pending) {
        if (ids_.contains(name)) { throw ArtifactError(name + ": duplicate parameter declaration"); }
        for (const auto& part : pending.reference.binding.parts) {
            pending.source_objects.push_back(
                artifact::object_id(binder.reader().directory().object(part.object)));
        }
        const WeightId id{weights.size()};
        ids_.emplace(std::move(name), id);
        weights.push_back(std::move(pending));
        return id;
    }

    std::map<std::string, WeightId, std::less<>> ids_;
};

bool same_object(const artifact::Binding& a, const artifact::Binding& b) {
    return a.parts.size() == 1 && b.parts.size() == 1 && a.parts[0].object == b.parts[0].object &&
           a.parts[0].begin == b.parts[0].begin && a.parts[0].end == b.parts[0].end;
}

TextWeights bind_text(Bindings& b, const TextConfig& c) {
    const std::uint64_t h = c.hidden_size, v = c.vocab_size, inter = c.intermediate_size;
    TextWeights out;
    out.token_embedding = b.matrix("text/token_embedding", {v, h});
    out.output_head     = b.matrix("text/output_head", {v, h}, {"text/final_hidden"});
    out.final_norm      = b.direct("text/final_norm", {h});
    out.output_head_use = b.use(out.output_head, "text/final_hidden");
    b.require_a16(out.output_head);
    out.embedding_shared = same_object(b.at(out.token_embedding).reference.binding,
                                       b.at(out.output_head).reference.binding);
    out.layers.reserve(c.num_hidden_layers);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const LayerKind kind      = c.layer_types[i];
        const std::uint64_t dim   = c.head_dim_of(kind);
        const std::uint64_t q     = c.query_width(kind), k = c.key_width(kind);
        const std::string p       = "text/layers/" + std::to_string(i) + "/";
        const std::string input   = p + "mixer_input";
        LayerWeights layer;
        if (kind == LayerKind::Sliding) {
            SlidingAttentionWeights a;
            a.query = b.matrix(p + "attention/query", {q, h}, {input}, true);
            a.key   = b.matrix(p + "attention/key", {k, h}, {input}, true);
            a.value = b.matrix(p + "attention/value", {k, h}, {input}, true);
            b.group({a.query, a.key, a.value});
            a.query_norm = b.direct(p + "attention/query_norm", {dim});
            a.key_norm   = b.direct(p + "attention/key_norm", {dim});
            a.output     = b.matrix(p + "attention/output", {h, q}, {p + "attention_output"});
            layer.attention = a;
        } else {
            // Global layers have no value projection: V is K.
            GlobalAttentionWeights a;
            a.query = b.matrix(p + "attention/query", {q, h}, {input}, true);
            a.key   = b.matrix(p + "attention/key", {k, h}, {input}, true);
            b.group({a.query, a.key});
            a.query_norm = b.direct(p + "attention/query_norm", {dim});
            a.key_norm   = b.direct(p + "attention/key_norm", {dim});
            a.output     = b.matrix(p + "attention/output", {h, q}, {p + "attention_output"});
            layer.attention = a;
        }
        layer.gate = b.matrix(p + "mlp/gate", {inter, h}, {p + "ffn_input"}, true);
        layer.up   = b.matrix(p + "mlp/up", {inter, h}, {p + "ffn_input"}, true);
        b.group({layer.gate, layer.up});
        layer.down                  = b.matrix(p + "mlp/down", {h, inter}, {p + "mlp/product"});
        layer.input_norm            = b.direct(p + "input_norm", {h});
        layer.post_attention_norm   = b.direct(p + "post_attention_norm", {h});
        layer.pre_feedforward_norm  = b.direct(p + "pre_feedforward_norm", {h});
        layer.post_feedforward_norm = b.direct(p + "post_feedforward_norm", {h});
        layer.layer_scalar          = b.scalar(p + "layer_scalar", layer.layer_scalar_value);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

// The MTP assistant's weights (tools/convert/gemma4.py _add_assistant). Its attention is
// query-only (the K/V are the target's) and every Use is A16Only.
MtpWeights bind_mtp(Bindings& b, const AssistantConfig& c) {
    const std::uint64_t h = c.hidden_size, backbone = c.backbone_hidden_size, v = c.vocab_size,
                        inter = c.intermediate_size;
    MtpWeights out;
    out.input_projection  = b.matrix("mtp/input_projection", {h, 2 * backbone}, {"mtp/stem_input"});
    out.output_projection = b.matrix("mtp/output_projection", {backbone, h}, {"mtp/final_hidden"});
    out.output_head       = b.matrix("mtp/output_head", {v, h}, {"mtp/final_hidden"});
    out.final_norm        = b.direct("mtp/final_norm", {h});
    out.output_head_use   = b.use(out.output_head, "mtp/final_hidden");
    for (const WeightId id : {out.input_projection, out.output_projection, out.output_head}) {
        b.require_a16(id);
    }
    out.layers.reserve(c.num_hidden_layers);
    for (std::uint32_t i = 0; i < c.num_hidden_layers; ++i) {
        const LayerKind kind    = c.layer_types[i];
        const std::uint64_t dim = c.head_dim_of(kind), q = c.query_width(kind);
        const std::string p     = "mtp/layers/" + std::to_string(i) + "/";
        MtpLayerWeights layer;
        layer.query = b.matrix(p + "attention/query", {q, h}, {p + "mixer_input"});
        layer.output = b.matrix(p + "attention/output", {h, q}, {p + "attention_output"});
        layer.gate  = b.matrix(p + "mlp/gate", {inter, h}, {p + "ffn_input"}, true);
        layer.up    = b.matrix(p + "mlp/up", {inter, h}, {p + "ffn_input"}, true);
        b.group({layer.gate, layer.up});
        layer.down       = b.matrix(p + "mlp/down", {h, inter}, {p + "mlp/product"});
        layer.query_norm = b.direct(p + "attention/query_norm", {dim});
        for (const WeightId id : {layer.query, layer.output, layer.gate, layer.up, layer.down}) {
            b.require_a16(id);
        }
        layer.input_norm            = b.direct(p + "input_norm", {h});
        layer.post_attention_norm   = b.direct(p + "post_attention_norm", {h});
        layer.pre_feedforward_norm  = b.direct(p + "pre_feedforward_norm", {h});
        layer.post_feedforward_norm = b.direct(p + "post_feedforward_norm", {h});
        layer.layer_scalar          = b.scalar(p + "layer_scalar", layer.layer_scalar_value);
        out.layers.push_back(std::move(layer));
    }
    return out;
}

// The text component's raw resources as byte views of Host objects.
TextResources bind_resources(artifact::Binder& binder) {
    const auto resource = [&](std::string_view role) {
        const auto bytes = binder.host_object(binder.resource("text", role));
        return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    TextResources out;
    out.tokenizer_json         = resource("tokenizer.json");
    out.tokenizer_config_json  = resource("tokenizer_config.json");
    out.chat_template_jinja    = resource("chat_template.jinja");
    out.generation_config_json = resource("generation_config.json");
    if (binder.has_resource("text", "sampling.json")) { out.sampling_json = resource("sampling.json"); }
    return out;
}

std::vector<BoundWeight> resolve_weights(std::vector<PendingWeight>&& pending,
                                         const artifact::MaterializedArtifact& backing) {
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (auto& item : pending) {
        auto view = artifact::bind_view(item.reference, backing);
        out.push_back({std::move(item.reference.name), std::move(item.source_objects), std::move(view),
                       std::move(item.uses)});
    }
    return out;
}

} // namespace

struct LoadPlan::Impl {
    Config config;
    LoadOptions options;
    ModelWeights weights;
    std::vector<PendingWeight> pending;
    artifact::MaterializationPlan materialization;
    TextResources resources;
    InstanceInfo info;
};

LoadPlan::LoadPlan(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
LoadPlan::~LoadPlan()                              = default;
LoadPlan::LoadPlan(LoadPlan&&) noexcept            = default;
LoadPlan& LoadPlan::operator=(LoadPlan&&) noexcept = default;

const Config& LoadPlan::config() const { return impl_->config; }

const ModelWeights& LoadPlan::weights() const { return impl_->weights; }

const TextResources& LoadPlan::resources() const { return impl_->resources; }

const artifact::MaterializationPlan& LoadPlan::materialization() const {
    return impl_->materialization;
}

const artifact::ParameterReference& LoadPlan::parameter(WeightId id) const {
    return impl_->pending.at(id.index).reference;
}

std::span<const WeightUse> LoadPlan::uses(WeightId id) const {
    return impl_->pending.at(id.index).uses;
}

std::uint64_t LoadPlan::device_bytes() const {
    return impl_->materialization.device_capacity_bytes;
}

bool is_gemma4(const artifact::Reader& reader) {
    const auto& config = reader.directory().component("text").config;
    if (!config.contains("architectures") || !config.contains("model_type")) { return false; }
    const auto& names = config.at("architectures");
    if (!names.is_array() || names.size() != 1 || !names[0].is_string() ||
        !config.at("model_type").is_string()) {
        return false;
    }
    try {
        return resolve_architecture(names[0].get<std::string>(),
                                    config.at("model_type").get<std::string>()) ==
               Architecture::Gemma4;
    } catch (const std::invalid_argument&) {
        return false;
    }
}

LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options) {
    auto out     = std::make_unique<LoadPlan::Impl>();
    out->options = options;
    out->config  = parse_config(reader.directory(), options);
    artifact::Binder binder(reader);
    out->resources = bind_resources(binder);
    Bindings bindings(binder);
    out->weights.text = bind_text(bindings, out->config.text);
    if (out->config.mtp) { out->weights.mtp = bind_mtp(bindings, *out->config.mtp); }
    out->pending         = std::move(bindings.weights);
    out->materialization = std::move(binder).finish();
    out->info.name       = reader.directory().metadata.value(
        "name", std::string(architecture_name(Architecture::Gemma4)));
    out->info.metadata_json   = reader.directory().metadata.dump();
    out->info.provenance_json = reader.directory().provenance.dump();
    out->info.artifact_id     = reader.artifact_id();
    return LoadPlan(std::move(out));
}

Model::Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
             TextResources resources, InstanceInfo info, artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(resources),
      info_(std::move(info)) {}

Model::~Model() = default;

std::unique_ptr<Model> plan_model(LoadPlan&& plan) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data                      = std::move(plan.impl_);
    const artifact::Reader& reader = *data->materialization.source;
    auto backing                   = artifact::materialize_geometry(reader, std::move(data->materialization));
    auto bound                     = resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(std::move(data->config), data->options,
                                            std::move(data->weights), std::move(bound),
                                            data->resources, std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                         const StartupObserver* observer) {
    if (!plan.impl_) { throw ArtifactError("load plan was already consumed"); }
    auto data                      = std::move(plan.impl_);
    const artifact::Reader& reader = *data->materialization.source;
    auto backing = artifact::materialize(reader, std::move(data->materialization), device, observer);
    auto bound   = resolve_weights(std::move(data->pending), backing);
    return std::unique_ptr<Model>(new Model(std::move(data->config), data->options,
                                            std::move(data->weights), std::move(bound),
                                            data->resources, std::move(data->info), std::move(backing)));
}

std::unique_ptr<Model> load_model(const std::filesystem::path& path, LoadOptions options,
                                  DeviceContext& device, const StartupObserver* observer) {
    artifact::Reader reader(path);
    return materialize_model(plan_load(reader, options), device, observer);
}

} // namespace infernix::models::gemma4
