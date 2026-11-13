#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void geglu_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream,
                  std::uint8_t* fp8_codes = nullptr, float* fp8_scales = nullptr);

} // namespace infernix::ops::detail
