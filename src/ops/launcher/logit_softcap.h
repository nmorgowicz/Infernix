#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void logit_softcap_launch(Tensor& x, float cap, cudaStream_t stream);

} // namespace infernix::ops::detail
