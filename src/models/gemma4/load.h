#pragma once

#include "models/gemma4/model.h"

#include <filesystem>
#include <memory>
#include <span>

namespace infernix::artifact {
class Reader;
struct ParameterReference;
}

namespace infernix::models::gemma4 {

// Cold load plan; it borrows its Reader until materialization.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    // The raw resources the Model will hold.
    [[nodiscard]] const TextResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;
    // Device bytes of the weights.
    [[nodiscard]] std::uint64_t device_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::unique_ptr<Model> plan_model(LoadPlan&&);
};

// True when the artifact's text component is Gemma4.
[[nodiscard]] bool is_gemma4(const artifact::Reader& reader);

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
// Geometry-only materialization (no device, no uploads): every device weight is an address range
// with no data behind it, so a real artifact's binding, formats and byte counts can be checked
// without a GPU. The Model's weights_resident() is false.
[[nodiscard]] std::unique_ptr<Model> plan_model(LoadPlan&& plan);
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

} // namespace infernix::models::gemma4
