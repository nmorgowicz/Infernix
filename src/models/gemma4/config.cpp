#include "models/gemma4/config.h"

#include "artifact/schema.h"
#include "models/registry.h"

#include <charconv>
#include <cmath>
#include <limits>
#include <string>
#include <string_view>

namespace infernix::models::gemma4 {
namespace {

using artifact::ArtifactError;
using artifact::Json;
using artifact::require_members;

std::uint32_t integer(const Json& value, std::string_view label, bool positive = true) {
    const auto n = artifact::require_u64(value, label, positive);
    if (n > std::numeric_limits<std::uint32_t>::max()) {
        throw ArtifactError(std::string(label) + ": config dimension exceeds u32");
    }
    return static_cast<std::uint32_t>(n);
}

std::uint32_t dimension(const Json& config, const char* name) {
    return integer(config.at(name), name);
}

// A finite real value; positive: strictly greater than zero.
float real(const Json& config, const char* name, bool positive = true) {
    const auto& value = config.at(name);
    if (!value.is_number()) { throw ArtifactError(std::string(name) + " must be a real value"); }
    const float result = static_cast<float>(value.get<double>());
    if (!std::isfinite(result) || (positive && result <= 0)) {
        throw ArtifactError(std::string(name) + (positive ? " must be positive finite FP32"
                                                          : " must be finite FP32"));
    }
    return result;
}

void require_string(const Json& config, const char* name, std::string_view expected) {
    const auto& value = config.at(name);
    if (!value.is_string() || value.get<std::string>() != expected) {
        throw ArtifactError(std::string(name) + " must be " + std::string(expected));
    }
}

std::vector<LayerKind> layer_kinds(const Json& value, std::uint32_t count, const char* label) {
    if (!value.is_array() || value.size() != count) {
        throw ArtifactError(std::string(label) + " must name every layer");
    }
    std::vector<LayerKind> out;
    out.reserve(count);
    for (const auto& kind : value) {
        if (kind == "sliding_attention") {
            out.push_back(LayerKind::Sliding);
        } else if (kind == "full_attention") {
            out.push_back(LayerKind::Global);
        } else {
            throw ArtifactError(std::string(label) + ": unknown layer type");
        }
    }
    return out;
}

void require_attention_geometry(std::uint32_t heads, std::uint32_t head_dim, std::uint32_t global_head_dim,
                                std::uint32_t kv_heads, std::uint32_t global_kv_heads,
                                std::string_view label) {
    if (heads % kv_heads != 0 || heads % global_kv_heads != 0) {
        throw ArtifactError(std::string(label) + ": query heads must be divisible by the KV head counts");
    }
    // Sliding layers rotate the whole head: half-split pairs.
    if (head_dim % 2 != 0 || global_head_dim % 2 != 0) {
        throw ArtifactError(std::string(label) + ": head dimensions must be even");
    }
}

void require_prope(std::uint32_t rotary_dim, std::uint32_t global_head_dim, float theta_eff,
                   std::string_view label) {
    if (rotary_dim % 2 != 0 || rotary_dim > global_head_dim) {
        throw ArtifactError(std::string(label) + ": prope_rotary_dim must be even and at most global_head_dim");
    }
    if (!std::isfinite(theta_eff) || theta_eff <= 0) {
        throw ArtifactError(std::string(label) + ": prope_theta_eff must be positive finite FP32");
    }
}

TextConfig text(const Json& value) {
    require_members(
        value,
        {"architectures", "model_type", "hidden_size", "vocab_size", "num_hidden_layers",
         "max_position_embeddings", "intermediate_size", "num_attention_heads", "head_dim",
         "global_head_dim", "num_key_value_heads", "num_global_key_value_heads", "layer_types",
         "sliding_window", "rms_norm_eps", "final_logit_softcapping", "tie_word_embeddings",
         "hidden_activation", "attention_scale", "embed_normalizer", "rope_sliding_theta",
         "rope_global_theta", "partial_rotary_factor", "prope_pairs", "prope_rotary_dim",
         "prope_theta_eff"},
        {"global_k_norm_scalar"}, "text config");
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1) {
        throw ArtifactError("architectures must name one implementation");
    }
    if (resolve_architecture(artifact::require_id(names[0], "architecture"),
                             artifact::require_id(value.at("model_type"), "model_type")) !=
        Architecture::Gemma4) {
        throw ArtifactError("text component is not Gemma4");
    }
    if (value.at("tie_word_embeddings") != true) {
        throw ArtifactError("Gemma4 requires tied word embeddings");
    }
    require_string(value, "hidden_activation", "gelu_pytorch_tanh");
    if (!value.at("attention_scale").is_number() || value.at("attention_scale").get<double>() != 1.0) {
        throw ArtifactError("Gemma4 attention_scale must be 1.0");
    }
    TextConfig out;
    out.hidden_size                 = dimension(value, "hidden_size");
    out.vocab_size                  = dimension(value, "vocab_size");
    out.num_hidden_layers           = dimension(value, "num_hidden_layers");
    out.max_position_embeddings     = dimension(value, "max_position_embeddings");
    out.intermediate_size           = dimension(value, "intermediate_size");
    out.num_attention_heads         = dimension(value, "num_attention_heads");
    out.head_dim                    = dimension(value, "head_dim");
    out.global_head_dim             = dimension(value, "global_head_dim");
    out.num_key_value_heads         = dimension(value, "num_key_value_heads");
    out.num_global_key_value_heads  = dimension(value, "num_global_key_value_heads");
    out.sliding_window              = dimension(value, "sliding_window");
    out.rms_norm_eps                = real(value, "rms_norm_eps");
    out.final_logit_softcapping     = real(value, "final_logit_softcapping");
    out.embed_normalizer            = real(value, "embed_normalizer");
    out.rope_sliding_theta          = real(value, "rope_sliding_theta");
    out.rope_global_theta           = real(value, "rope_global_theta");
    out.partial_rotary_factor       = real(value, "partial_rotary_factor");
    out.prope_pairs                 = dimension(value, "prope_pairs");
    out.prope_rotary_dim            = dimension(value, "prope_rotary_dim");
    out.prope_theta_eff             = real(value, "prope_theta_eff");
    out.layer_types                 = layer_kinds(value.at("layer_types"), out.num_hidden_layers, "layer_types");
    require_attention_geometry(out.num_attention_heads, out.head_dim, out.global_head_dim,
                               out.num_key_value_heads, out.num_global_key_value_heads, "text config");
    if (out.partial_rotary_factor > 1.0F) {
        throw ArtifactError("partial_rotary_factor must be in (0, 1]");
    }
    if (std::uint64_t(out.prope_pairs) * 2 != out.prope_rotary_dim) {
        throw ArtifactError("prope_rotary_dim must be twice prope_pairs");
    }
    require_prope(out.prope_rotary_dim, out.global_head_dim, out.prope_theta_eff, "text config");
    for (std::uint32_t i = 0; i < out.num_hidden_layers; ++i) {
        (out.layer_types[i] == LayerKind::Global ? out.last_global_layer : out.last_sliding_layer) = i;
    }
    if (value.contains("global_k_norm_scalar")) {
        const auto& scalars = value.at("global_k_norm_scalar");
        if (!scalars.is_object()) { throw ArtifactError("global_k_norm_scalar must be an object"); }
        for (const auto& [key, scalar] : scalars.items()) {
            std::uint32_t layer = 0;
            const auto parsed   = std::from_chars(key.data(), key.data() + key.size(), layer);
            if (key.empty() || parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() ||
                layer >= out.num_hidden_layers || out.layer_types[layer] != LayerKind::Global) {
                throw ArtifactError("global_k_norm_scalar keys must be global layer indices");
            }
            if (!scalar.is_number() || !std::isfinite(scalar.get<double>())) {
                throw ArtifactError("global_k_norm_scalar values must be finite reals");
            }
            out.global_k_norm_scalar.emplace(layer, static_cast<float>(scalar.get<double>()));
        }
    }
    return out;
}

AssistantConfig assistant(const Json& value, const TextConfig& target) {
    require_members(
        value,
        {"architectures", "hidden_size", "backbone_hidden_size", "intermediate_size",
         "num_hidden_layers", "num_attention_heads", "head_dim", "global_head_dim",
         "num_key_value_heads", "num_global_key_value_heads", "layer_types", "kv_source_layers",
         "sliding_window", "rms_norm_eps", "vocab_size", "rope_sliding_theta", "prope_rotary_dim",
         "prope_theta_eff"},
        {}, "MTP config");
    const auto& names = value.at("architectures");
    if (!names.is_array() || names.size() != 1 ||
        artifact::require_id(names[0], "architecture") != "Gemma4Assistant") {
        throw ArtifactError("MTP component is not Gemma4Assistant");
    }
    AssistantConfig out;
    out.hidden_size                 = dimension(value, "hidden_size");
    out.backbone_hidden_size        = dimension(value, "backbone_hidden_size");
    out.intermediate_size           = dimension(value, "intermediate_size");
    out.num_hidden_layers           = dimension(value, "num_hidden_layers");
    out.num_attention_heads         = dimension(value, "num_attention_heads");
    out.head_dim                    = dimension(value, "head_dim");
    out.global_head_dim             = dimension(value, "global_head_dim");
    out.num_key_value_heads         = dimension(value, "num_key_value_heads");
    out.num_global_key_value_heads  = dimension(value, "num_global_key_value_heads");
    out.sliding_window              = dimension(value, "sliding_window");
    out.rms_norm_eps                = real(value, "rms_norm_eps");
    out.vocab_size                  = dimension(value, "vocab_size");
    out.rope_sliding_theta          = real(value, "rope_sliding_theta");
    out.prope_rotary_dim            = dimension(value, "prope_rotary_dim");
    out.prope_theta_eff             = real(value, "prope_theta_eff");
    out.layer_types                 = layer_kinds(value.at("layer_types"), out.num_hidden_layers, "layer_types");
    if (out.backbone_hidden_size != target.hidden_size) {
        throw ArtifactError("MTP backbone_hidden_size differs from the target hidden size");
    }
    if (out.vocab_size != target.vocab_size) {
        throw ArtifactError("MTP vocab_size differs from the target vocabulary");
    }
    require_attention_geometry(out.num_attention_heads, out.head_dim, out.global_head_dim,
                               out.num_key_value_heads, out.num_global_key_value_heads, "MTP config");
    require_prope(out.prope_rotary_dim, out.global_head_dim, out.prope_theta_eff, "MTP config");
    const auto& sources = value.at("kv_source_layers");
    if (!sources.is_array() || sources.size() != out.num_hidden_layers) {
        throw ArtifactError("kv_source_layers must name a target layer for every MTP layer");
    }
    for (std::uint32_t i = 0; i < out.num_hidden_layers; ++i) {
        const auto source = integer(sources[i], "kv_source_layers entry", false);
        if (source >= target.num_hidden_layers) {
            throw ArtifactError("MTP KV source is outside the target");
        }
        // The assistant's query reads the source layer's K/V, so the two attentions must match.
        const LayerKind kind = out.layer_types[i], source_kind = target.layer_types[source];
        if (kind != source_kind || out.head_dim_of(kind) != target.head_dim_of(source_kind) ||
            out.kv_heads_of(kind) != target.kv_heads_of(source_kind) ||
            out.window_of(kind) != target.window_of(source_kind)) {
            throw ArtifactError("MTP layer and its KV source layer disagree");
        }
        out.kv_source_layers.push_back(source);
    }
    return out;
}

} // namespace

Config parse_config(const artifact::Directory& directory, const LoadOptions& options) {
    try {
        if (options.purpose != EnginePurpose::Generation &&
            options.purpose != EnginePurpose::CausalScoring) {
            throw ArtifactError("unknown loading purpose");
        }
        if (options.vision) {
            throw ArtifactError("vision is not supported");
        }
        if (options.speculative != SpeculativeBackend::None && !options.mtp()) {
            throw ArtifactError("speculative decoding uses the MTP assistant (--spec mtp)");
        }
        if (options.mtp() && options.purpose != EnginePurpose::Generation) {
            throw ArtifactError("CausalScoring loads the text backbone only");
        }
        if (!options.draft_model.empty()) { throw ArtifactError("--draft-model is not supported"); }
        if (options.proposal_head != ProposalHead::Full) {
            throw ArtifactError("proposal heads are not supported");
        }
        if (options.rope_yarn_factor != 1.0F) { throw ArtifactError("YaRN is not supported"); }
        if (options.lossless || options.reference) {
            throw ArtifactError("the lossless route is not supported");
        }
        if (options.embedding_host || options.stream_experts) {
            throw ArtifactError("the token embedding and weights are device-resident");
        }
        Config out;
        out.text = text(directory.component("text").config);
        if (options.mtp()) {
            if (!directory.components.contains("mtp")) {
                throw ArtifactError("the artifact has no MTP component");
            }
            const auto& component = directory.component("mtp");
            if (component.target != "text") { throw ArtifactError("MTP target must be text"); }
            out.mtp = assistant(component.config, out.text);
        }
        return out;
    } catch (const std::exception& error) {
        throw ArtifactError(std::string("Gemma4 config: ") + error.what());
    }
}

} // namespace infernix::models::gemma4
