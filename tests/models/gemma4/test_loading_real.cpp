// A real Gemma 4 text(+mtp) artifact by explicit path: host-only binding (plan_load + plan_model, no
// device), every object validated, and optionally the converter's conversion receipt compared with the
// container. The container and the receipts record no per-object SHA-256: the receipt carries the
// artifact_id, the payload size and each tensor object's format/layout/shape (see --receipt).
#include "artifact/binder.h"
#include "artifact/fixture.h"
#include "artifact/formats.h"
#include "artifact/reader.h"
#include "models/gemma4/load.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <variant>

namespace {

using namespace infernix;
using namespace infernix::models;
namespace g4 = infernix::models::gemma4;
using infernix::artifact::Json;
using infernix::test::artifact_fixture::require;

bool allowed(QType format) {
    return format == QType::BF16 || format == QType::FP8_E4M3FN_ROW_BF16 ||
           format == QType::Q4_G32_FP16 || format == QType::Q4_G128_FP16 ||
           format == QType::Q4_G256_FP16;
}

void check_receipt(const artifact::Reader& reader, const std::filesystem::path& path) {
    std::ifstream in(path);
    require(in.good(), "cannot open the receipt");
    const Json receipt = Json::parse(in);
    const auto& dir    = reader.directory();
    std::string id;
    for (const auto byte : reader.artifact_id()) {
        static constexpr char hex[] = "0123456789abcdef";
        id += hex[std::to_integer<unsigned>(byte) >> 4];
        id += hex[std::to_integer<unsigned>(byte) & 15];
    }
    require(receipt.at("artifact_id") == id, "artifact_id differs from the receipt");
    require(receipt.at("payload_bytes").get<std::uint64_t>() == dir.payload_bytes,
            "payload bytes differ from the receipt");
    std::map<std::string, const Json*> methods;
    for (const auto& method : receipt.at("methods")) { methods[method.at("object")] = &method; }
    std::size_t tensors = 0;
    for (const auto& object : dir.objects) {
        const auto* tensor = std::get_if<artifact::TensorObject>(&object);
        if (!tensor) { continue; }
        ++tensors;
        const auto found = methods.find(tensor->id);
        require(found != methods.end(), "a tensor object is missing from the receipt");
        const Json& method = *found->second;
        require(method.at("format") == tensor->format && method.at("layout") == tensor->layout &&
                    method.at("shape") == Json(tensor->shape),
                "a tensor object differs from its receipt method");
    }
    require(tensors == methods.size(), "the receipt describes other tensor objects than the container");
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path, receipt;
        LoadOptions options;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--receipt") {
                receipt = value();
            } else if (arg == "--mtp") {
                options.speculative = SpeculativeBackend::Mtp;
            } else if (arg == "--host-only") {
                // Always: this test never touches the device.
            } else {
                throw std::invalid_argument("usage: --artifact PATH [--mtp] [--host-only] [--receipt PATH]");
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        artifact::Reader reader(path);
        for (std::size_t i = 0; i < reader.directory().objects.size(); ++i) { reader.validate_object({i}); }
        if (!receipt.empty()) { check_receipt(reader, receipt); }
        require(g4::is_gemma4(reader), "the artifact is not Gemma4");

        auto plan = g4::plan_load(reader, options);
        const auto& config  = plan.config();
        const auto& weights = plan.weights();
        require(config.text.num_hidden_layers == 60 && weights.text.layers.size() == 60,
                "layer count is not 60");
        require(config.text.hidden_size == 5376 && config.text.vocab_size == 262144,
                "hidden size or vocabulary differs");
        require(weights.mtp.has_value() == options.mtp(), "MTP selection differs");
        require(weights.text.embedding_shared, "token table and head are not shared");

        std::size_t global = 0;
        std::set<std::size_t> projections;
        const auto add = [&](g4::WeightId id) {
            for (const auto& part : plan.parameter(id).binding.parts) { projections.insert(part.object.index); }
        };
        for (std::size_t i = 0; i < weights.text.layers.size(); ++i) {
            const auto& layer = weights.text.layers[i];
            const bool is_global = std::holds_alternative<g4::GlobalAttentionWeights>(layer.attention);
            require(is_global == (config.text.layer_types[i] == g4::LayerKind::Global),
                    "a layer's attention differs from layer_types");
            global += is_global;
            std::visit([&](const auto& a) { add(a.query); add(a.key); add(a.output); }, layer.attention);
            add(layer.gate);
            add(layer.up);
            add(layer.down);
        }
        require(global == 10, "the artifact does not have 10 global layers");
        std::map<std::string, std::size_t> formats;
        for (const auto object : projections) {
            const QType format = reader.geometry({object}).format;
            require(allowed(format), "a projection is stored in a format Gemma4 does not bind");
            ++formats[std::string(artifact::format_name(format))];
        }
        auto model = g4::plan_model(std::move(plan));
        require(!model->weights_resident(), "plan_model produced resident weights");

        std::cout << path.filename().string() << ": layers=60 global=10 sliding=50 embedding_shared formats:";
        for (const auto& [name, count] : formats) { std::cout << ' ' << name << " x" << count; }
        std::cout << (receipt.empty() ? "" : " (receipt matched)") << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
