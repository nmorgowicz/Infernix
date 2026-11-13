#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

inline constexpr int kGemma4ResidualNormBlock        = 256;
inline constexpr int kGemma4ResidualNormMaxPerThread = 24;

void gemma4_residual_norm_launch(const Tensor& projected, const Tensor& post_norm, float scale,
                                 const Tensor* next_norm, float eps, Tensor& x, Tensor* next_out,
                                 cudaStream_t stream);

} // namespace infernix::ops::detail
