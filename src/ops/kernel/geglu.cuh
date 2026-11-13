#pragma once

// Implements: include/infernix/ops/geglu.h
// Match: BF16 GeGLU with the tanh GELU approximation. out is contiguous; gate and
// up may use arbitrary valid Tensor strides.

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops {

inline constexpr int kGegluPairsPerThread = 4;

__device__ __forceinline__ float geglu_gelu_tanh(float value) {
    constexpr float kSqrt2Pi = 0.79788456080286535588f;
    return 0.5f * value * (1.0f + tanhf(kSqrt2Pi * (value + 0.044715f * value * value * value)));
}

__device__ __forceinline__ __nv_bfloat162 geglu_pair(__nv_bfloat162 gate, __nv_bfloat162 up) {
    const float r0 = geglu_gelu_tanh(__low2float(gate)) * __low2float(up);
    const float r1 = geglu_gelu_tanh(__high2float(gate)) * __high2float(up);
    return __floats2bfloat162_rn(r0, r1);
}

__global__ void geglu_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                             std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const float activated = geglu_gelu_tanh(__bfloat162float(gate[i]));
        out[i]                = __float2bfloat16_rn(activated * __bfloat162float(up[i]));
    }
}

__global__ void geglu_strided_input_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                           __nv_bfloat16* out, std::int64_t n, std::int32_t ne0,
                                           std::int32_t ne1, std::int32_t ne2, std::int64_t gnb0,
                                           std::int64_t gnb1, std::int64_t gnb2, std::int64_t gnb3,
                                           std::int64_t unb0, std::int64_t unb1, std::int64_t unb2,
                                           std::int64_t unb3) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const auto* gate_bytes    = reinterpret_cast<const unsigned char*>(gate);
    const auto* up_bytes      = reinterpret_cast<const unsigned char*>(up);
    for (std::int64_t i = start; i < n; i += stride) {
        std::int64_t rem = i;
        const auto d0    = static_cast<std::int32_t>(rem % ne0);
        rem /= ne0;
        const auto d1 = static_cast<std::int32_t>(rem % ne1);
        rem /= ne1;
        const auto d2 = static_cast<std::int32_t>(rem % ne2);
        const auto d3 = static_cast<std::int32_t>(rem / ne2);

        const std::int64_t goff =
            static_cast<std::int64_t>(d0) * gnb0 + static_cast<std::int64_t>(d1) * gnb1 +
            static_cast<std::int64_t>(d2) * gnb2 + static_cast<std::int64_t>(d3) * gnb3;
        const std::int64_t uoff =
            static_cast<std::int64_t>(d0) * unb0 + static_cast<std::int64_t>(d1) * unb1 +
            static_cast<std::int64_t>(d2) * unb2 + static_cast<std::int64_t>(d3) * unb3;
        const auto gv         = *reinterpret_cast<const __nv_bfloat16*>(gate_bytes + goff);
        const auto uv         = *reinterpret_cast<const __nv_bfloat16*>(up_bytes + uoff);
        const float activated = geglu_gelu_tanh(__bfloat162float(gv));
        out[i]                = __float2bfloat16_rn(activated * __bfloat162float(uv));
    }
}

__launch_bounds__(256) __global__
    void geglu_dim0_split_kernel(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                                 __nv_bfloat16* out, std::int32_t rows,
                                 std::int64_t gate_col_stride, std::int64_t up_col_stride) {
    const int col            = blockIdx.y;
    const std::int64_t pair0 = (blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x) *
                               kGegluPairsPerThread;
    const std::int64_t row_pairs = rows / 2;

    const auto* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate + col * gate_col_stride);
    const auto* up2   = reinterpret_cast<const __nv_bfloat162*>(up + col * up_col_stride);
    auto* out2 = reinterpret_cast<__nv_bfloat162*>(out + col * static_cast<std::int64_t>(rows));

    for (int p = 0; p < kGegluPairsPerThread; ++p) {
        const std::int64_t j = pair0 + p;
        if (j < row_pairs) { out2[j] = geglu_pair(gate2[j], up2[j]); }
    }
}

} // namespace infernix::ops
