#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/gemma4/load.h"
#include "models/qwen3_5/load.h"

#include <iostream>
#include <tuple>
#include <string>
#include <utility>
#include <variant>

namespace {

using namespace infernix;
using namespace infernix::models;
namespace g4 = infernix::models::gemma4;
using namespace infernix::test::artifact_fixture;

constexpr std::uint64_t H = 128, V = 8, INTER = 24;

// A 2-layer Gemma 4 text artifact (layer 0 sliding: head_dim 4, 2 heads, 1 KV head; layer 1 global:
// global_head_dim 8, 1 KV head), with an optional 2-layer MTP assistant.
struct ModelFixture {
    Fixture file;

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
        const std::uint64_t dim = sliding ? 4 : 8, q = 2 * dim, k = dim;
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
        const std::uint64_t dim = sliding ? 4 : 8, q = 2 * dim, h = 8;
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

    explicit ModelFixture(bool with_mtp, QType down_format = QType::BF16) {
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
                           {"head_dim", 4},
                           {"global_head_dim", 8},
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
                  {"head_dim", 4},
                  {"global_head_dim", 8},
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

// (a) binding and geometry, (b) shared head, (c) MTP selection, (e) the guards between architectures.
void binds_and_selects() {
    ModelFixture fixture(true);
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    require(g4::is_gemma4(reader), "a Gemma4 artifact was not recognized");
    {
        auto plan = g4::plan_load(reader);
        const auto& text = plan.config().text;
        require(text.last_sliding_layer == 0 && text.last_global_layer == 1,
                "last sliding/global layers were not derived from layer_types");
        require(!plan.weights().mtp, "MTP was bound without being requested");
        const auto& layers = plan.weights().text.layers;
        require(layers.size() == 2, "layer count differs from the config");
        const auto& sliding = std::get<g4::SlidingAttentionWeights>(layers[0].attention);
        require(std::holds_alternative<g4::GlobalAttentionWeights>(layers[1].attention),
                "a global layer bound a value projection");
        const auto rows = [&](g4::WeightId id) {
            const auto& part = plan.parameter(id).binding.parts.at(0);
            return std::pair{part.begin, part.end};
        };
        require(rows(sliding.query) == std::pair<std::uint64_t, std::uint64_t>{0, 8 * H} &&
                    rows(sliding.key) == std::pair<std::uint64_t, std::uint64_t>{8 * H, 12 * H} &&
                    rows(sliding.value) == std::pair<std::uint64_t, std::uint64_t>{12 * H, 16 * H},
                "sliding q/k/v rows do not follow head_dim");
        const auto& global = std::get<g4::GlobalAttentionWeights>(layers[1].attention);
        require(rows(global.query) == std::pair<std::uint64_t, std::uint64_t>{0, 16 * H} &&
                    rows(global.key) == std::pair<std::uint64_t, std::uint64_t>{16 * H, 24 * H},
                "global q/k rows do not follow global_head_dim");
        require(layers[0].layer_scalar_value == 0.25F, "layer scalar value was not read");

        const auto& text_weights = plan.weights().text;
        require(text_weights.embedding_shared, "tied table and head were not recognized as shared");
        require(plan.uses(text_weights.output_head)[text_weights.output_head_use.use_index].policy ==
                    ops::LinearPolicy::A16Only,
                "output head Use is not A16Only");
        auto model = g4::plan_model(std::move(plan));
        require(!model->weights_resident() && model->weights().text.layers.size() == 2,
                "plan_model did not produce a geometry-only model");
    }
    {
        auto plan = g4::plan_load(reader, {.speculative = SpeculativeBackend::Mtp});
        require(plan.weights().mtp && plan.weights().mtp->layers.size() == 2 &&
                    plan.config().mtp->kv_source_layers == std::vector<std::uint32_t>{0, 1},
                "requested MTP was not bound");
    }
    rejects([&] { (void)qwen3_5::plan_load(reader); }, "the Qwen3.5 loader accepted a Gemma4 artifact");
}

void absent_mtp_and_foreign_artifact() {
    ModelFixture absent(false);
    absent.file.write();
    {
        artifact::Reader reader(absent.file.entry);
        rejects([&] { (void)g4::plan_load(reader, {.speculative = SpeculativeBackend::Mtp}); },
                "requested MTP was accepted without the component");
    }
    absent.file.root["components"]["text"]["config"]["architectures"] = {"Qwen3_5ForCausalLM"};
    absent.file.root["components"]["text"]["config"]["model_type"]    = "qwen3_5_text";
    absent.file.write();
    artifact::Reader reader(absent.file.entry);
    require(!g4::is_gemma4(reader), "a Qwen3.5 config was recognized as Gemma4");
    rejects([&] { (void)g4::plan_load(reader); }, "the Gemma4 loader accepted a Qwen3.5 config");
}

// (d) a stored projection format Gemma4 does not bind is refused and the message lists what it binds.
void unsupported_format() {
    ModelFixture fixture(false, QType::Q6_G64_FP16);
    fixture.file.write();
    artifact::Reader reader(fixture.file.entry);
    try {
        (void)g4::plan_load(reader);
    } catch (const artifact::ArtifactError& error) {
        const std::string message = error.what();
        for (const char* allowed : {"bf16", "fp8_e4m3fn_row_bf16", "q4_g32_fp16", "q4_g128_fp16", "q4_g256_fp16"}) {
            require(message.find(allowed) != std::string::npos,
                    "the format error does not name the allowed formats");
        }
        return;
    }
    throw std::runtime_error("an unsupported projection format was accepted");
}

} // namespace

int main() {
    try {
        binds_and_selects();
        absent_mtp_and_foreign_artifact();
        unsupported_format();
        std::cout << "gemma4 config, binding, MTP selection and format checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
