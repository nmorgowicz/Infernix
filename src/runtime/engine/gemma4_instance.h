#pragma once

#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/program.h"
#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <memory>

namespace infernix { struct DeviceContext; }

namespace infernix::runtime {

// Native Gemma operands and frontend remain owned until after the Program is destroyed.
struct Gemma4Instance {
    using ModelContract = models::gemma4::RuntimeTypes;

    std::unique_ptr<models::gemma4::Model> model;
    const models::gemma4::execution::Parameters parameters;
    models::gemma4::Frontend frontend;
    KvCapacityResolution kv_capacity_resolution;
    const std::uint32_t capacity;
    std::unique_ptr<models::gemma4::Program> program;

    Gemma4Instance(std::unique_ptr<models::gemma4::Model> source,
                   const EngineOptions& options, DeviceContext& device);
    ~Gemma4Instance();
    Gemma4Instance(const Gemma4Instance&) = delete;
    Gemma4Instance& operator=(const Gemma4Instance&) = delete;
};

struct ConstructedGemma4 {
    std::unique_ptr<Gemma4Instance> instance;
    LoadSummary load;
    ModelMetadata model_metadata;
    EngineOptions options;
};

// P4 is a real cold-cache, single-lane BF16 route. Other routes must fail before loading.
void validate_gemma4_options(const EngineOptions& options);

// Planning uses the same model-aware reservation curve without uploading weights or
// constructing Parameters/Program. A memory refusal populates plan.fits/error.
[[nodiscard]] ConstructedGemma4 construct_gemma4(const EngineOptions& options,
                                               DeviceContext& device, EnginePlan* plan = nullptr);

} // namespace infernix::runtime
