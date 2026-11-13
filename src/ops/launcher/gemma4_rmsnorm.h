#pragma once

// infernix::ops::detail - Gemma 4 hidden-5376 RMSNorm launch. It lives in the non-RDC Gemma archive
// (src/ops/gemma4_small_ops.cmake); rmsnorm_launch calls it for BF16 d == 5376 rows.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// x, weight and out are contiguous BF16 with 4-byte aligned data; rows of d = 5376. The sum keeps
// rmsnorm_generic_kernel's order, so the result equals the generic route bit for bit.
void rmsnorm_gemma4_hidden_launch(const Tensor& x, const Tensor& weight, bool unit_offset,
                                  Tensor& out, std::int64_t rows, float eps, cudaStream_t stream);

} // namespace infernix::ops::detail
