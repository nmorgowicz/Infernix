#pragma once

#include "artifact/framing.h"
#include "artifact/materializer.h"
#include "models/gemma4/config.h"
#include "models/gemma4/weights.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::gemma4 {

// The text component's raw resources. The byte views borrow the model's Host backing; the
// tokenizer, chat template and output parsing are interpreted by the frontend.
struct TextResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view chat_template_jinja;
    std::string_view generation_config_json;
    // Optional: the checkpoint's sampling presets (text resource sampling.json); empty when absent.
    std::string_view sampling_json;
};

struct InstanceInfo {
    std::string name;
    std::string metadata_json;
    std::string provenance_json;
    artifact::ArtifactId artifact_id{};
};

class LoadPlan;

class Model {
public:
    ~Model();
    Model(const Model&)            = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&)                 = delete;
    Model& operator=(Model&&)      = delete;

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const LoadOptions& options() const noexcept { return options_; }
    [[nodiscard]] const ModelWeights& weights() const noexcept { return weights_; }
    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }
    [[nodiscard]] std::span<const BoundWeight> weight_data() const noexcept { return bound_; }
    [[nodiscard]] const TextResources& resources() const noexcept { return resources_; }
    [[nodiscard]] const InstanceInfo& info() const noexcept { return info_; }
    // False for a plan-only model (plan_model): its device weights are address ranges with no data.
    [[nodiscard]] bool weights_resident() const noexcept { return !backing_.geometry_only(); }
    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const noexcept {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> plan_model(LoadPlan&&);
    Model(Config config, LoadOptions options, ModelWeights weights, std::vector<BoundWeight> bound,
          TextResources resources, InstanceInfo info, artifact::MaterializedArtifact backing);

    // Borrowers are declared after the backing they view, so they are destroyed first.
    artifact::MaterializedArtifact backing_;
    Config config_;
    LoadOptions options_;
    ModelWeights weights_;
    std::vector<BoundWeight> bound_; // views into backing_
    TextResources resources_;        // views into backing_
    InstanceInfo info_;
};

} // namespace infernix::models::gemma4
