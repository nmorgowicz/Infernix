#pragma once

#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

void gemma4_qkv_prep_launch(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                            const Tensor& positions, int rotary_dim, float theta, float eps,
                            bool v_from_k, Tensor& q_out, const Gemma4CacheView& cache,
                            cudaStream_t stream);

} // namespace infernix::ops::detail
