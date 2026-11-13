#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Gemma 4 sublayer tail in one pass. For every column t of the contiguous BF16 [H,T] tensors:
 *
 *   x[:,t]        = (x[:,t] + rmsnorm(projected[:,t], post_norm)) * scale
 *   next_out[:,t] = rmsnorm(x[:,t], next_norm)            (only when next_norm is given)
 *
 * rmsnorm is v * (mean(v^2) + eps)^-1/2 * weight. The normed projection, the sum, the scaled sum
 * and next_out are each rounded to BF16, so for H = 5376 the result is bit-identical to rmsnorm,
 * residual_add, residual_scale (skipped when scale is 1) and rmsnorm run one after another.
 * H is a multiple of 256 and at most 6144. post_norm and next_norm are BF16 [H]. projected, x and
 * next_out must not alias. No workspace.
 */
void gemma4_residual_norm(const Tensor& projected, const Tensor& post_norm, float scale,
                          const Tensor* next_norm, float eps, Tensor& x, Tensor* next_out,
                          cudaStream_t stream);

} // namespace infernix::ops
