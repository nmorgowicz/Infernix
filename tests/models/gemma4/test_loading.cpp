#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "tiny_model_fixture.h"
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

using namespace infernix::test::gemma4_fixture;

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

// (f) GGUF blocks (a llama.cpp GGUF import): text-layer projections bind ggml_rows_v1 parents, a fused
// projection may span one parent per member but never mixes GGUF blocks with native formats, the vocabulary
// table refuses them, and the text config's global_output_order is parsed.
void gguf_projections() {
    // A tensor appended after the resources: the file's payload size follows it.
    const auto ggml_tensor = [](ModelFixture& fixture, const std::string& id, std::vector<std::uint64_t> shape) {
        fixture.tensor(id, shape, QType::GGML_IQ4_NL, QuantLayout::GgmlRows);
        fixture.file.root["files"][0]["payload_bytes"] = fixture.file.payload.size();
    };
    const auto ggml_parameter = [&](ModelFixture& fixture, const std::string& name, std::vector<std::uint64_t> shape) {
        ggml_tensor(fixture, name + "#gguf", shape);
        fixture.file.root["bindings"][name] = {{"object", name + "#gguf"}};
    };
    const std::string p = "text/layers/0/";
    {
        ModelFixture fixture(false);
        for (const auto& [role, rows] : {std::pair{"attention/query", 8}, std::pair{"attention/key", 4},
                                         std::pair{"attention/value", 4}, std::pair{"mlp/gate", 24},
                                         std::pair{"mlp/up", 24}}) {
            ggml_parameter(fixture, p + role, {static_cast<std::uint64_t>(rows), H});
        }
        fixture.file.root["components"]["text"]["config"]["global_output_order"] = "source";
        fixture.file.write();
        artifact::Reader reader(fixture.file.entry);
        const auto plan = g4::plan_load(reader);
        require(plan.config().text.global_output_source_order, "global_output_order source was not parsed");
        const auto& sliding = std::get<g4::SlidingAttentionWeights>(plan.weights().text.layers[0].attention);
        require(plan.parameter(sliding.query).binding.parts.at(0).object !=
                    plan.parameter(sliding.key).binding.parts.at(0).object,
                "GGUF q/k were not bound to their own parents");
    }
    {
        ModelFixture fixture(false);
        ggml_parameter(fixture, p + "attention/key", {4, H}); // q and v stay in the BF16 qkv parent
        fixture.file.write();
        artifact::Reader reader(fixture.file.entry);
        rejects([&] { (void)g4::plan_load(reader); }, "a fused projection mixing GGUF blocks and BF16 was accepted");
    }
    {
        ModelFixture fixture(false);
        ggml_tensor(fixture, "embedding-gguf", {V, H});
        fixture.file.root["bindings"]["text/token_embedding"] = {{"object", "embedding-gguf"}};
        fixture.file.root["bindings"]["text/output_head"]     = {{"object", "embedding-gguf"}};
        fixture.file.write();
        artifact::Reader reader(fixture.file.entry);
        rejects([&] { (void)g4::plan_load(reader); }, "a GGUF-block vocabulary table was accepted");
    }
    {
        ModelFixture fixture(false);
        fixture.file.root["components"]["text"]["config"]["global_output_order"] = "ggml";
        fixture.file.write();
        artifact::Reader reader(fixture.file.entry);
        rejects([&] { (void)g4::plan_load(reader); }, "an unknown global_output_order was accepted");
    }
}

} // namespace

int main() {
    try {
        binds_and_selects();
        absent_mtp_and_foreign_artifact();
        unsupported_format();
        gguf_projections();
        std::cout << "gemma4 config, binding, MTP selection and format checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
