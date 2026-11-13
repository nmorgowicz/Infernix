#pragma once

#include "core/arena.h"
#include "models/gemma4/execution/text_forward.h"
#include "models/gemma4/model.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace infernix::models::gemma4::execution {

// Native operand geometry and policies for allocation-complete planning. Views may refer to a
// geometry-only Model and must not be executed then. Constructing this allocates host metadata only.
struct ParameterGeometry {
    Gemma4Embedding embedding;
    std::vector<Gemma4LayerWeights> layers;
    const void* final_norm = nullptr;
    Weight output_head;
    std::size_t derived_device_bytes = 0;
};

class Parameters {
private:
    // Backing precedes its borrowers so prepared scales outlive every native descriptor.
    std::vector<DeviceBuffer> row_scales_;
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
};

} // namespace infernix::models::gemma4::execution
