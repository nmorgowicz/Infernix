#pragma once

#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/program.h"
#include "infernix/types.h"
#include "runtime/contract/resources.h"

#include <functional>
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

// P5 is a real cold-cache, single-lane route with BF16 or FP8 global KV. Other routes must fail before loading.
void validate_gemma4_options(const EngineOptions& options);

// Page-aligned largest context, rebuilding the native reservation for each candidate (workspace,
// tables and rings included). The callback returns bytes for the complete candidate context.
// allow_shared_memory (--allow-shared-memory): when nothing fits, the 9216-token minimum is returned
// (what does not fit then runs in shared GPU memory) instead of refusing.
[[nodiscard]] std::uint32_t resolve_gemma4_auto_context(
    std::uint32_t upper, std::size_t available, std::size_t headroom,
    const std::function<std::size_t(std::uint32_t)>& reservation, bool allow_shared_memory = false);

// --allow-shared-memory: the room a capacity check may count, `free` or, when that is less, what the
// native reservation of `context` tokens needs plus `headroom`; the shortfall is what spills.
[[nodiscard]] std::size_t gemma4_shared_memory_room(const SequenceCapacityCurve& curve, std::size_t free,
                                                    std::uint32_t context, std::size_t headroom);

// Planning uses the same model-aware reservation curve without uploading weights or
// constructing Parameters/Program. A memory refusal populates plan.fits/error.
[[nodiscard]] ConstructedGemma4 construct_gemma4(const EngineOptions& options,
                                               DeviceContext& device, EnginePlan* plan = nullptr);

} // namespace infernix::runtime
