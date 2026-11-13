#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void topk_columns_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                         Tensor& ids, cudaStream_t stream);

} // namespace infernix::ops::detail
