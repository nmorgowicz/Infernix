#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void gemma4_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                             cudaStream_t stream);

} // namespace infernix::ops::detail
