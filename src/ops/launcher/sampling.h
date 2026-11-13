#pragma once

// infernix::ops::detail - private launch prototype for sample.

#include "core/tensor.h"
#include "infernix/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// top_k_capacity is kSamplerFastCandidates or kSamplerWideCandidates (validated by the wrapper).
void sample_batch_launch(const Tensor& logits, Tensor& out, std::int32_t token_domain,
                         const SamplingConfig* configs, const Tensor& logical_positions,
                         std::int32_t purpose, DeviceSpan workspace, cudaStream_t stream,
                         std::int32_t top_k_capacity);

void increment_token_counts_launch(const Tensor& token_ids, Tensor& token_counts,
                                   cudaStream_t stream);

void increment_licensed_token_counts_launch(const Tensor& licensed_tokens, const Tensor& licensed_counts,
                                            const SamplingConfig* configs, cudaStream_t stream);

[[nodiscard]] std::size_t sampling_workspace_exact_bytes(std::int32_t token_domain,
                                                         std::int32_t columns,
                                                         std::int32_t top_k_capacity);

} // namespace infernix::ops::detail
