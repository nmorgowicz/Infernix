#include "ops/launcher/logit_softcap.h"

#include "core/device.h"
#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace infernix::ops::detail {
namespace {

__global__ void logit_softcap_kernel(__nv_bfloat16* x, std::int64_t n, float cap) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const float inv           = 1.0f / cap;
    for (std::int64_t i = start; i < n; i += stride) {
        const float value = __bfloat162float(x[i]);
        x[i]              = __float2bfloat16_rn(cap * tanhf(value * inv));
    }
}

} // namespace

void logit_softcap_launch(Tensor& x, float cap, cudaStream_t stream) {
    constexpr int block  = 256;
    const std::int64_t n = x.numel();
    const int grid       = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(block)), std::numeric_limits<int>::max()));
    logit_softcap_kernel<<<grid, block, 0, stream>>>(static_cast<__nv_bfloat16*>(x.data), n, cap);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
