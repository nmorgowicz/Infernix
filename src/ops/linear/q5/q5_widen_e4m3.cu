#include "infernix/ops/linear.h"

#include "core/device.h"
#include "ops/linear/q5/q5_rowsplit_storage.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256;

// Eight consecutive logical values, without the BF16 cast used by the A16 MMA decoder.
__device__ __forceinline__ void decode_chunk(const std::uint8_t* codes, const std::uint8_t* high,
                                             const __half* scales, std::int64_t row_group,
                                             int chunk, float (&values)[8]) {
    const auto group = row_group + chunk / 8;
    const int within = chunk % 8;
    const unsigned packed = *reinterpret_cast<const unsigned*>(codes + group * 32 + within * 4);
    const unsigned upper = high[group * 8 + within];
    const float scale = __half2float(scales[group]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int raw = static_cast<int>(((packed >> (4 * i)) & 15U) | (((upper >> i) & 1U) << 4));
        const int code = (raw ^ 16) - 16;
        values[i] = __fmul_rn(static_cast<float>(code), scale);
    }
}

__global__ __launch_bounds__(kThreads)
void widen_kernel(const std::uint8_t* codes, const std::uint8_t* high, const __half* scales,
                  int k, int padded_k, std::uint8_t* baked, __nv_bfloat16* row_scales) {
    const int row = static_cast<int>(blockIdx.x);
    const auto row_group = static_cast<std::int64_t>(row) * (padded_k / detail::Q5RowSplitStorage::kGroupK);
    const int chunks = k / 8;
    float largest = 0.0F;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += kThreads) {
        float values[8];
        decode_chunk(codes, high, scales, row_group, chunk, values);
#pragma unroll
        for (int i = 0; i < 8; ++i) { largest = fmaxf(largest, fabsf(values[i])); }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        largest = fmaxf(largest, __shfl_xor_sync(0xffffffffU, largest, offset));
    }
    __shared__ float warp_max[kThreads / 32];
    __shared__ float inverse;
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x / 32] = largest; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = warp_max[0];
#pragma unroll
        for (int i = 1; i < kThreads / 32; ++i) { maximum = fmaxf(maximum, warp_max[i]); }
        const __nv_bfloat16 multiplier = maximum > 0.0F
                                            ? __float2bfloat16_ru(__fdiv_rn(maximum, 448.0F))
                                            : __float2bfloat16(1.0F);
        row_scales[row] = multiplier;
        inverse = __frcp_rn(__bfloat162float(multiplier));
    }
    __syncthreads();
    const float reciprocal = inverse;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += kThreads) {
        float values[8];
        decode_chunk(codes, high, scales, row_group, chunk, values);
        __align__(8) __nv_fp8x2_storage_t encoded[4];
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            encoded[pair] = __nv_cvt_float2_to_fp8x2(
                make_float2(__fmul_rn(values[2 * pair], reciprocal),
                            __fmul_rn(values[2 * pair + 1], reciprocal)),
                __NV_SATFINITE, __NV_E4M3);
        }
        *reinterpret_cast<uint2*>(baked + static_cast<std::int64_t>(row) * k + chunk * 8) =
            *reinterpret_cast<const uint2*>(encoded);
    }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

} // namespace

void q5_widen_e4m3(const Weight& weight, Tensor& codes, Tensor& row_scales, cudaStream_t stream) {
    if (weight.qtype != QType::Q5_G64_FP16 || weight.layout != QuantLayout::RowSplit ||
        weight.group != 64 || weight.group_size != 64 || weight.scale_dtype != DType::FP16 ||
        weight.n <= 0 || weight.k <= 0 || weight.k % 64 != 0 ||
        weight.padded_shape[1] < weight.k || weight.padded_shape[1] % 128 != 0 ||
        !aligned(weight.qdata, 4) || !aligned(weight.qhigh, 1) || !aligned(weight.scales, 2)) {
        throw std::invalid_argument("q5_widen_e4m3: expected native Q5 group64 RowSplit matrix");
    }
    if (!aligned(codes.data, 8) || codes.dtype != DType::U8 || !codes.is_contiguous() ||
        codes.ne[0] != weight.k || codes.ne[1] != weight.n || codes.ne[2] != 1 || codes.ne[3] != 1) {
        throw std::invalid_argument("q5_widen_e4m3: codes must be contiguous aligned U8 [K,N]");
    }
    if (!aligned(row_scales.data, 2) || row_scales.dtype != DType::BF16 || !row_scales.is_contiguous() ||
        row_scales.ne[0] != weight.n || row_scales.ne[1] != 1 || row_scales.ne[2] != 1 || row_scales.ne[3] != 1) {
        throw std::invalid_argument("q5_widen_e4m3: row_scales must be contiguous BF16 [N]");
    }
    widen_kernel<<<weight.n, kThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const __half*>(weight.scales), weight.k, weight.padded_shape[1],
        static_cast<std::uint8_t*>(codes.data), static_cast<__nv_bfloat16*>(row_scales.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops
