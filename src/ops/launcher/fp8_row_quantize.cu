#include "ops/launcher/fp8_row_quantize.h"

#include "core/device.h"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kBlock          = 256;
constexpr int kWarps          = kBlock / kWarpSize;
constexpr float kE4M3fnMax    = 448.0F;
constexpr std::uint16_t kBf16MinSubnormal = 1;

__global__ __launch_bounds__(kBlock) void fp8_row_quantize_kernel(
    const __nv_bfloat16* __restrict__ src, std::uint8_t* __restrict__ codes,
    __nv_bfloat16* __restrict__ scales, int k) {
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * k;
    float amax             = 0.0F;
    for (int col = static_cast<int>(threadIdx.x); col < k; col += kBlock) {
        amax = fmaxf(amax, fabsf(__bfloat162float(src[row + col])));
    }
    amax = warp_max(amax);

    __shared__ float warp_maxima[kWarps];
    __shared__ float inverse;
    if ((static_cast<int>(threadIdx.x) & (kWarpSize - 1)) == 0) {
        warp_maxima[threadIdx.x / kWarpSize] = amax;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float reduced = 0.0F;
        for (int warp = 0; warp < kWarps; ++warp) reduced = fmaxf(reduced, warp_maxima[warp]);
        __nv_bfloat16 scale = __float2bfloat16_rn(0.0F);
        float inv           = 0.0F;
        if (reduced != 0.0F) {
            scale = __float2bfloat16_rn(reduced / kE4M3fnMax);
            if (__bfloat16_as_ushort(scale) == 0) {
                scale = __ushort_as_bfloat16(kBf16MinSubnormal);
            }
            inv = 1.0F / __bfloat162float(scale);
        }
        scales[blockIdx.x] = scale;
        inverse            = inv;
    }
    __syncthreads();
    const float inv = inverse;
    for (int col = static_cast<int>(threadIdx.x); col < k; col += kBlock) {
        codes[row + col] =
            inv == 0.0F ? 0
                        : static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(
                              __bfloat162float(src[row + col]) * inv, __NV_SATFINITE, __NV_E4M3));
    }
}

} // namespace

void fp8_row_quantize_launch(const __nv_bfloat16* src, std::uint8_t* codes, __nv_bfloat16* scales,
                             std::int32_t n, std::int32_t k, cudaStream_t stream) {
    fp8_row_quantize_kernel<<<n, kBlock, 0, stream>>>(src, codes, scales, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
