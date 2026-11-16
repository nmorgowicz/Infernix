#pragma once

#include "core/arena.h"
#include "models/gemma4/execution/draft_forward.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/model.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace infernix::models::gemma4::execution {

// The block width of a global output projection's input permutation (GGUF imports keep its columns in
// source order): prope_perm moves 64-wide runs of each 512-wide head.
inline constexpr std::int32_t kGlobalOutputPermBlock = 64;

// That permutation for every query head: source block j of the stored columns reads block result[j] of the
// attention output (written in the converter's prope order). Refused unless prope_perm moves whole blocks.
[[nodiscard]] std::vector<std::int32_t> global_output_input_perm(const TextConfig& config);

// Native operand geometry and policies for allocation-complete planning. Views may refer to a
// geometry-only Model and must not be executed then. Constructing this allocates host metadata only.
struct ParameterGeometry {
    Gemma4Embedding embedding;
    std::vector<Gemma4LayerWeights> layers;
    const void* final_norm = nullptr;
    Weight output_head;
    // The MTP assistant, present when the load selected it (LoadOptions::mtp()).
    std::optional<Gemma4DraftWeights> draft;
    std::size_t derived_device_bytes = 0;
};

class Parameters {
private:
    // Backing precedes its borrowers so prepared scales outlive every native descriptor.
    std::vector<DeviceBuffer> row_scales_;
    DeviceBuffer output_perm_;
    std::size_t derived_device_bytes_ = 0;

public:
    Parameters(const Model& model, cudaStream_t stream);
    Parameters(const Parameters&) = delete;
    Parameters& operator=(const Parameters&) = delete;

    // No upload, device allocation or device read. Uses the same grouping/policies as execution.
    [[nodiscard]] static ParameterGeometry geometry(const Model& model);
    [[nodiscard]] std::size_t derived_device_bytes() const noexcept { return derived_device_bytes_; }

    const Model& model;
    Gemma4Embedding embedding;
    std::vector<Gemma4LayerWeights> layers;
    const void* final_norm = nullptr;
    Weight output_head;
    std::optional<Gemma4DraftWeights> draft;
};

} // namespace infernix::models::gemma4::execution
