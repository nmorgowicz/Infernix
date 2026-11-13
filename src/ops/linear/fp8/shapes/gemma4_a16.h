#pragma once
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

// A16 routes of the Gemma 4 trunk projections (gemma4_a16.cu). Instantiated for the six trunk
// geometries only.
namespace infernix::ops::detail::gemma4_fp8 {
template <int N, int K>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
} // namespace infernix::ops::detail::gemma4_fp8
