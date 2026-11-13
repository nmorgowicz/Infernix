#pragma once

#include "ops/linear/q4/q4_operands.h"

#include <cuda_bf16.h>

namespace infernix::ops::detail {

// A16 q4 fp16-scale GEMV for more than 16 tokens. Activations stay BF16. `scale_mode` is the scale
// mode of the weight (q4_scale_mode.h): 0 for q4_g32, 2 for q4_g128, 3 for q4_g256.
void q4_g32_wide(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                 int scale_mode);

} // namespace infernix::ops::detail
