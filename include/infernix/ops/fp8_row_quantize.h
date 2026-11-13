#pragma once

#include "core/arena.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Replace a resident BF16 [N,K] Weight with row-scaled E4M3FN in `storage`.
 *
 * Each row's scale is BF16(maxabs / 448), or the BF16 min-subnormal when that rounds to zero on a
 * nonzero row. Codes are SATFINITE round-to-nearest-even E4M3 of value/scale. A Weight that is
 * already FP8_E4M3FN_ROW_BF16 is left unchanged. `storage` owns the new payload and must outlive
 * `weight`. The conversion finishes before return.
 */
void quantize_fp8_row_bf16(Weight& weight, DeviceBuffer& storage, cudaStream_t stream);

} // namespace infernix::ops
