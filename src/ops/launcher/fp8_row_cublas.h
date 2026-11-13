#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// codes[count] E4M3 -> widened[count] BF16, exact. codes and widened are 16-byte aligned.
void fp8_widen_codes_launch(const std::uint8_t* codes, __nv_bfloat16* widened, std::int64_t count,
                            cudaStream_t stream);

// out[i] = BF16(accumulators[i] * scales[i % rows]) over a column-major [rows, columns] tensor.
void fp8_scale_rows_launch(const float* accumulators, const __nv_bfloat16* scales,
                           __nv_bfloat16* out, std::int32_t rows, std::int64_t count,
                           cudaStream_t stream);

} // namespace infernix::ops::detail
