#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void residual_scale_launch(Tensor& x, float scale, cudaStream_t stream);

} // namespace infernix::ops::detail
