#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * In-place logit softcap:
 *
 *   ideal[i] = cap * tanh(x[i] / cap).
 *
 * `x` is a contiguous BF16 tensor and `cap` is positive and finite. The oracle evaluates `ideal`
 * in FP64 from the represented input. The updated BF16 x is promoted and compared directly with
 * that result; output storage rounding belongs to the Op's numerical criterion, not the oracle.
 * The Op mutates only x and uses no workspace.
 */
void logit_softcap(Tensor& x, float cap, cudaStream_t stream);

} // namespace infernix::ops
