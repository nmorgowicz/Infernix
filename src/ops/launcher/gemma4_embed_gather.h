#pragma once

// infernix::ops::detail - scaled embedding gathers for the Gemma 4 tables (FP8_E4M3FN_ROW_BF16 at
// [262144,5376] or the Qwen4 shapes, and the q4 fp16 family). They live in the non-RDC Gemma archive
// (src/ops/gemma4_small_ops.cmake). An id outside the table embeds zeros and reports `fault`.

#include "core/device.h" // DeviceFault
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// out[d,t] = bf16(bf16(dequantize(table)[ids[t],d]) * bf16(factor)); out.ne[0] is 5376 or 5120.
void embed_gather_scaled_fp8_launch(const Tensor& ids, const Weight& table, float factor, Tensor& out,
                                    cudaStream_t stream, DeviceFault fault);

// Q4_G32/G128/G256_FP16 table, same formula.
void embed_gather_q4_launch(const Tensor& ids, const Weight& table, float factor, Tensor& out,
                            cudaStream_t stream, DeviceFault fault);

} // namespace infernix::ops::detail
