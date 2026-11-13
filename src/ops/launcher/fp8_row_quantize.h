#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

void fp8_row_quantize_launch(const __nv_bfloat16* src, std::uint8_t* codes, __nv_bfloat16* scales,
                             std::int32_t n, std::int32_t k, cudaStream_t stream);

} // namespace infernix::ops::detail
