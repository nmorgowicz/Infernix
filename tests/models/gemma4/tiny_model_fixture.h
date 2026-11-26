#pragma once

#include "artifact/fixture.h"
#include "core/weight_view.h"
#include "artifact/formats.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace infernix::test::gemma4_fixture {

using namespace infernix::test::artifact_fixture;

inline constexpr std::uint64_t H = 128, V = 8, INTER = 24;

// A 2-layer Gemma 4 text artifact (layer 0 sliding: head_dim `sliding_dim`, 2 heads, 1 KV head; layer 1 global:
// global_head_dim `global_dim`, 1 KV head), with an optional 2-layer MTP assistant. The defaults (4 and 8) keep the
// payload tiny; a test that needs the real FP8 geometry passes 256 and 512.
struct ModelFixture {
    Fixture file;
    std::uint64_t sliding_dim, global_dim;

    void tensor(const std::string& id, std::vector<std::uint64_t> shape, QType format = QType::BF16,
                QuantLayout layout = QuantLayout::Contiguous) {
        const auto geometry = weight_geometry(format, layout, shape);
        const auto offset   = (file.payload.size() + 255) / 256 * 256;
        file.payload.resize(offset + geometry.bytes);
        file.root["objects"].push_back({{"id", id},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(format)},
                                        {"layout", artifact::layout_name(layout)},
                                        {"offset", offset},
                                        {"bytes", geometry.bytes}});
    }

    void resource(const std::string& role, const std::string& bytes) {
        const auto offset = file.payload.size();
        for (const auto byte : bytes) { file.payload.push_back(std::byte(byte)); }
        file.root["objects"].push_back({{"id", role},
                                        {"kind", "resource"},
                                        {"encoding", "raw_bytes_v1"},
                                        {"offset", offset},
                                        {"bytes", bytes.size()}});
        file.root["components"]["text"]["resources"][role] = role;
    }

    void use(const std::string& name, const std::string& input) {
        file.root["uses"].push_back(
            {{"parameter", name}, {"input", input}, {"activation_policy", "A16Only"}});
    }

    void parameter(const std::string& name, std::vector<std::uint64_t> shape,
                   const std::string& input = {}, QType format = QType::BF16,
                   QuantLayout layout = QuantLayout::Contiguous) {
        tensor(name, shape, format, layout);
        file.root["bindings"][name] = {{"object", name}};
        if (!input.empty()) { use(name, input); }
    }

    // Consecutive row ranges {name, rows, input} of one stored parent.
    void group(const std::string& parent, std::uint64_t k,
               const std::vector<std::tuple<std::string, std::uint64_t, std::string>>& rows) {
        std::uint64_t total = 0;
        for (const auto& row : rows) { total += std::get<1>(row); }
        tensor(parent, {total, k});
        std::uint64_t begin = 0;
        for (const auto& [name, count, input] : rows) {
            file.root["bindings"][name] = {
                {"parts", Json::array({{{"object", parent},
                                        {"range", {begin * k, (begin + count) * k}}}})}};
            use(name, input);
            begin += count;
        }
    }

    void scalar(const std::string& name) {
        tensor(name, {1}, QType::FP32);
        put_word(file.payload, file.root["objects"].back()["offset"].get<std::size_t>(), 0x3e800000, 4);
        file.root["bindings"][name] = {{"object", name}};
    }

    void layer(const std::string& p, bool sliding, QType down_format) {
        const std::uint64_t dim = sliding ? sliding_dim : global_dim, q = 2 * dim, k = dim;
        const std::string a = p + "attention/";
        if (sliding) {
            group(p + "qkv", H, {{a + "query", q, p + "mixer_input"},
                                 {a + "key", k, p + "mixer_input"},
                                 {a + "value", k, p + "mixer_input"}});
        } else {
            group(p + "qk", H, {{a + "query", q, p + "mixer_input"}, {a + "key", k, p + "mixer_input"}});
        }
        parameter(a + "query_norm", {dim});
        parameter(a + "key_norm", {dim});
        parameter(a + "output", {H, q}, p + "attention_output");
        group(p + "gateup", H, {{p + "mlp/gate", INTER, p + "ffn_input"}, {p + "mlp/up", INTER, p + "ffn_input"}});
        parameter(p + "mlp/down", {H, INTER}, p + "mlp/product", down_format,
                  down_format == QType::BF16 ? QuantLayout::Contiguous : QuantLayout::RowSplit);
        for (const char* norm : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                 "post_feedforward_norm"}) {
            parameter(p + norm, {H});
        }
        scalar(p + "layer_scalar");
    }

    void mtp_layer(const std::string& p, bool sliding) {
        const std::uint64_t dim = sliding ? sliding_dim : global_dim, q = 2 * dim, h = 8;
        parameter(p + "attention/query", {q, h}, p + "mixer_input");
        parameter(p + "attention/output", {h, q}, p + "attention_output");
        group(p + "gateup", h, {{p + "mlp/gate", 8, p + "ffn_input"}, {p + "mlp/up", 8, p + "ffn_input"}});
        parameter(p + "mlp/down", {h, 8}, p + "mlp/product");
        parameter(p + "attention/query_norm", {dim});
        for (const char* norm : {"input_norm", "post_attention_norm", "pre_feedforward_norm",
                                 "post_feedforward_norm"}) {
            parameter(p + norm, {h});
        }
        scalar(p + "layer_scalar");
    }

    explicit ModelFixture(bool with_mtp, QType down_format = QType::BF16, std::uint64_t sliding_head_dim = 4,
                          std::uint64_t global_head_dim = 8)
        : sliding_dim(sliding_head_dim), global_dim(global_head_dim) {
        file.payload.clear();
        file.root = {{"components",
                      {{"text",
                        {{"config",
                          {{"architectures", {"Gemma4ForCausalLM"}},
                           {"model_type", "gemma4_text"},
                           {"hidden_size", H},
                           {"vocab_size", V},
                           {"num_hidden_layers", 2},
                           {"max_position_embeddings", 64},
                           {"intermediate_size", INTER},
                           {"num_attention_heads", 2},
                           {"head_dim", sliding_dim},
                           {"global_head_dim", global_dim},
                           {"num_key_value_heads", 1},
                           {"num_global_key_value_heads", 1},
                           {"layer_types", {"sliding_attention", "full_attention"}},
                           {"sliding_window", 4},
                           {"rms_norm_eps", 1e-6},
                           {"final_logit_softcapping", 30.0},
                           {"tie_word_embeddings", true},
                           {"hidden_activation", "gelu_pytorch_tanh"},
                           {"attention_scale", 1.0},
                           {"embed_normalizer", 11.3125},
                           {"rope_sliding_theta", 10000.0},
                           {"rope_global_theta", 1000000.0},
                           {"partial_rotary_factor", 0.25},
                           {"prope_pairs", 2},
                           {"prope_rotary_dim", 4},
                           {"prope_theta_eff", 31.6227}}}}}}},
                     {"objects", Json::array()},
                     {"bindings", Json::object()},
                     {"uses", Json::array()},
                     {"metadata", {{"name", "tiny-gemma4"}}}};
        tensor("embedding-storage", {V, H});
        file.root["bindings"]["text/token_embedding"] = {{"object", "embedding-storage"}};
        file.root["bindings"]["text/output_head"]     = {{"object", "embedding-storage"}};
        use("text/output_head", "text/final_hidden");
        parameter("text/final_norm", {H});
        layer("text/layers/0/", true, down_format);
        layer("text/layers/1/", false, QType::BF16);
        if (with_mtp) {
            file.root["components"]["mtp"] = {
                {"target", "text"},
                {"config",
                 {{"architectures", {"Gemma4Assistant"}},
                  {"hidden_size", 8},
                  {"backbone_hidden_size", H},
                  {"intermediate_size", 8},
                  {"num_hidden_layers", 2},
                  {"num_attention_heads", 2},
                  {"head_dim", sliding_dim},
                  {"global_head_dim", global_dim},
                  {"num_key_value_heads", 1},
                  {"num_global_key_value_heads", 1},
                  {"layer_types", {"sliding_attention", "full_attention"}},
                  {"kv_source_layers", {0, 1}},
                  {"sliding_window", 4},
                  {"rms_norm_eps", 1e-6},
                  {"vocab_size", V},
                  {"rope_sliding_theta", 10000.0},
                  {"prope_rotary_dim", 4},
                  {"prope_theta_eff", 31.6227}}}};
            parameter("mtp/input_projection", {8, 2 * H}, "mtp/stem_input");
            parameter("mtp/output_projection", {H, 8}, "mtp/final_hidden");
            parameter("mtp/output_head", {V, 8}, "mtp/final_hidden");
            parameter("mtp/final_norm", {8});
            mtp_layer("mtp/layers/0/", true);
            mtp_layer("mtp/layers/1/", false);
        }
        resource("tokenizer.json", "{}");
        resource("tokenizer_config.json", "{}");
        resource("generation_config.json", "{}");
        resource("chat_template.jinja", "a template, preserved as data");
        file.root["files"] = Json::array({{{"path", nullptr}, {"payload_bytes", file.payload.size()}}});
    }
};

} // namespace infernix::test::gemma4_fixture
