#include "ops/launcher/fp8_row_cublas.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_codec.cuh"

#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kBlock = 256;

// Eight codes per thread: one 8-byte load, one 16-byte store.
__global__ __launch_bounds__(kBlock) void fp8_widen_codes_kernel(const std::uint8_t* codes,
                                                                 __nv_bfloat16* widened,
                                                                 std::int64_t count) {
    const std::int64_t first = (static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x) * 8;
    if (first + 8 <= count) {
        const uint2 packed = *reinterpret_cast<const uint2*>(codes + first);
        uint4 out;
        out.x                                      = fp8_e4m3x2_to_bf16x2_bits(packed.x & 0xffffu);
        out.y                                      = fp8_e4m3x2_to_bf16x2_bits(packed.x >> 16);
        out.z                                      = fp8_e4m3x2_to_bf16x2_bits(packed.y & 0xffffu);
        out.w                                      = fp8_e4m3x2_to_bf16x2_bits(packed.y >> 16);
        *reinterpret_cast<uint4*>(widened + first) = out;
    } else {
        for (std::int64_t i = first; i < count; ++i) {
            const unsigned bits = fp8_e4m3x2_to_bf16x2_bits(codes[i]);
            widened[i]          = __ushort_as_bfloat16(static_cast<unsigned short>(bits & 0xffffu));
        }
    }
}

// Four outputs per thread when the row extent allows it, so a float4 never crosses a column.
__global__ __launch_bounds__(kBlock) void fp8_scale_rows_kernel(const float* accumulators,
                                                                const __nv_bfloat16* scales,
                                                                __nv_bfloat16* out,
                                                                std::int32_t rows,
                                                                std::int64_t count) {
    const std::int64_t first = (static_cast<std::int64_t>(blockIdx.x) * kBlock + threadIdx.x) * 4;
    if (first >= count) { return; }
    const auto row = static_cast<std::int32_t>(first % rows);
    if (rows % 4 == 0) {
        const float4 v = *reinterpret_cast<const float4*>(accumulators + first);
        *reinterpret_cast<__nv_bfloat162*>(out + first) = __floats2bfloat162_rn(
            v.x * __bfloat162float(scales[row]), v.y * __bfloat162float(scales[row + 1]));
        *reinterpret_cast<__nv_bfloat162*>(out + first + 2) = __floats2bfloat162_rn(
            v.z * __bfloat162float(scales[row + 2]), v.w * __bfloat162float(scales[row + 3]));
        return;
    }
    for (std::int64_t i = first; i < first + 4 && i < count; ++i) {
        out[i] = __float2bfloat16_rn(accumulators[i] *
                                     __bfloat162float(scales[static_cast<std::int32_t>(i % rows)]));
    }
}

unsigned blocks_for(std::int64_t count, int per_thread) {
    return static_cast<unsigned>((count + static_cast<std::int64_t>(kBlock) * per_thread - 1) /
                                 (static_cast<std::int64_t>(kBlock) * per_thread));
}

} // namespace

void fp8_widen_codes_launch(const std::uint8_t* codes, __nv_bfloat16* widened, std::int64_t count,
                            cudaStream_t stream) {
    fp8_widen_codes_kernel<<<blocks_for(count, 8), kBlock, 0, stream>>>(codes, widened, count);
    CUDA_CHECK(cudaGetLastError());
}

void fp8_scale_rows_launch(const float* accumulators, const __nv_bfloat16* scales,
                           __nv_bfloat16* out, std::int32_t rows, std::int64_t count,
                           cudaStream_t stream) {
    fp8_scale_rows_kernel<<<blocks_for(count, 4), kBlock, 0, stream>>>(accumulators, scales, out,
                                                                       rows, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
