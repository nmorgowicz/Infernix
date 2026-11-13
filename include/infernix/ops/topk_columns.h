#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops {

/**
 * Per-column top-k of BF16 logits.
 *
 * ids[r, t] is the token with the r-th largest float(logits[v, t]) over
 * 0 <= v < valid_rows, ordered by descending value. Equal values keep the lower
 * index, so ids[0, t] equals ops::argmax. As in argmax, a NaN in row 0 ranks first;
 * a NaN in any other row ranks after every number. logits is contiguous BF16
 * [physical_rows, T] and ids is contiguous I32 [k, T], with 1 <= k <= 8. Physical rows
 * at and above valid_rows do not participate; ranks past valid_rows are -1. No
 * workspace. Graph-capturable.
 */
void topk_columns(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                  cudaStream_t stream);

} // namespace infernix::ops
