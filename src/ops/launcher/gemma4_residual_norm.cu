#include "ops/launcher/gemma4_residual_norm.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kBlock        = kGemma4ResidualNormBlock;
constexpr int kMaxPerThread = kGemma4ResidualNormMaxPerThread;
static_assert(kBlock == kRmsGenericBlock);

// One CTA per column; thread t holds elements t + 256*k. Both norms sum and normalize as the
// hidden-5376 rmsnorm route (rmsnorm_generic_order_kernel), and the add matches residual_add, so
// each step rounds as the unfused ops do. Every input is loaded before the first reduction, so the
// loads overlap instead of waiting on the stores to x.
template <bool Scale, bool Next>
__global__ __launch_bounds__(kBlock) void gemma4_residual_norm_kernel(
    const __nv_bfloat16* __restrict__ projected, const __nv_bfloat16* __restrict__ post_norm,
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ next_norm,
    __nv_bfloat16* __restrict__ next_out, int d, int count, float eps, float scale) {
    const std::int64_t base = static_cast<std::int64_t>(blockIdx.x) * d;
    const int thread        = static_cast<int>(threadIdx.x);
    float values[kMaxPerThread];
    float residuals[kMaxPerThread];
    float post_gains[kMaxPerThread];
    float next_gains[kMaxPerThread];
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        if (k < count) {
            const int i   = thread + k * kBlock;
            values[k]     = __bfloat162float(projected[base + i]);
            residuals[k]  = __bfloat162float(x[base + i]);
            post_gains[k] = __bfloat162float(post_norm[i]);
            if constexpr (Next) { next_gains[k] = __bfloat162float(next_norm[i]); }
            sum += values[k] * values[k];
        }
    }
    __shared__ float scratch[kBlock + 1];
    const float inv = rsqrtf(rmsnorm_generic_order_sum(sum, scratch) / static_cast<float>(d) + eps);

    float next_sum = 0.0f;
#pragma unroll
    for (int k = 0; k < kMaxPerThread; ++k) {
        if (k < count) {
            const int i        = thread + k * kBlock;
            const float normed = __bfloat162float(__float2bfloat16_rn(
                rmsnorm_epilogue<RmsEpilogue::Plain>(values[k], inv, post_gains[k], 0.0f)));
            float residual     = __bfloat162float(__float2bfloat16_rn(residuals[k] + normed));
            if constexpr (Scale) {
                residual = __bfloat162float(__float2bfloat16_rn(residual * scale));
            }
            x[base + i] = __float2bfloat16_rn(residual);
            values[k]   = residual;
            if constexpr (Next) { next_sum += residual * residual; }
        }
    }
    if constexpr (Next) {
        const float next_inv =
            rsqrtf(rmsnorm_generic_order_sum(next_sum, scratch) / static_cast<float>(d) + eps);
#pragma unroll
        for (int k = 0; k < kMaxPerThread; ++k) {
            if (k < count) {
                const int i        = thread + k * kBlock;
                next_out[base + i] = __float2bfloat16_rn(
                    rmsnorm_epilogue<RmsEpilogue::Plain>(values[k], next_inv, next_gains[k], 0.0f));
            }
        }
    }
}

template <bool Scale, bool Next>
void launch(const Tensor& projected, const Tensor& post_norm, float scale, const Tensor* next_norm,
            float eps, Tensor& x, Tensor* next_out, cudaStream_t stream) {
    const int d     = projected.ne[0];
    const int count = d / kBlock;
    gemma4_residual_norm_kernel<Scale, Next>
        <<<static_cast<unsigned>(projected.ne[1]), kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(post_norm.data), static_cast<__nv_bfloat16*>(x.data),
            Next ? static_cast<const __nv_bfloat16*>(next_norm->data) : nullptr,
            Next ? static_cast<__nv_bfloat16*>(next_out->data) : nullptr, d, count, eps, scale);
}

} // namespace

void gemma4_residual_norm_launch(const Tensor& projected, const Tensor& post_norm, float scale,
                                 const Tensor* next_norm, float eps, Tensor& x, Tensor* next_out,
                                 cudaStream_t stream) {
    const bool scaled = scale != 1.0f;
    if (next_norm != nullptr) {
        if (scaled) {
            launch<true, true>(projected, post_norm, scale, next_norm, eps, x, next_out, stream);
        } else {
            launch<false, true>(projected, post_norm, scale, next_norm, eps, x, next_out, stream);
        }
    } else if (scaled) {
        launch<true, false>(projected, post_norm, scale, next_norm, eps, x, next_out, stream);
    } else {
        launch<false, false>(projected, post_norm, scale, next_norm, eps, x, next_out, stream);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
