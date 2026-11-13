// infernix::ops - scaled embedding launcher for the Gemma 4 tables.
#include "ops/launcher/gemma4_embed_gather.h"

#include "ops/common/math.h"
#include "ops/common/q4_scale_mode.h"
#include "ops/kernel/gemma4_embed_gather.cuh"
#include "core/device.h" // CUDA_CHECK
#include "core/pdl.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kBlock = 128;

float bf16_factor(float factor) { return __bfloat162float(__float2bfloat16(factor)); }

template <int D, int BlocksPerToken, int Threads>
void launch_fp8(const Tensor& ids, const EmbedIds& guard, const Weight& table, float factor,
                Tensor& out, cudaStream_t stream) {
    const int grid = ids.ne[0] * BlocksPerToken;
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(grid), dim3(Threads), 0, stream},
        embed_gather_scaled_fp8_kernel<D, BlocksPerToken, Threads>, guard,
        static_cast<const std::uint8_t*>(table.qdata),
        static_cast<const __nv_bfloat16*>(table.scales), bf16_factor(factor),
        static_cast<__nv_bfloat16*>(out.data)));
}

} // namespace

void embed_gather_scaled_fp8_launch(const Tensor& ids, const Weight& table, float factor, Tensor& out,
                                    cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.n, fault};
    const std::int32_t T = ids.ne[0];
    if (out.ne[0] == 5376) {
        if (T <= 176) {
            launch_fp8<5376, 14, kBlock>(ids, guard, table, factor, out, stream);
        } else {
            launch_fp8<5376, 7, kBlock>(ids, guard, table, factor, out, stream);
        }
    } else if (out.ne[0] == 5120) {
        if (T <= 176) {
            launch_fp8<5120, 10, kBlock>(ids, guard, table, factor, out, stream);
        } else {
            launch_fp8<5120, 5, kBlock>(ids, guard, table, factor, out, stream);
        }
    } else {
        throw std::invalid_argument("embed_gather: fp8 table d must be 5376 or 5120");
    }
    CUDA_CHECK(cudaGetLastError());
}

void embed_gather_q4_launch(const Tensor& ids, const Weight& table, float factor, Tensor& out,
                            cudaStream_t stream, DeviceFault fault) {
    const EmbedIds guard{static_cast<const std::int32_t*>(ids.data), table.shape[0], fault};
    const std::int32_t d = out.ne[0];
    const std::int32_t T = ids.ne[0];
    const std::int64_t n = div_up(static_cast<std::int64_t>(d), std::int64_t{32}) * T;
    const int grid =
        static_cast<int>(std::max<std::int64_t>(1, div_up(n, static_cast<std::int64_t>(kBlock))));
    const auto launch = [&]<int kScaleMode>() {
        CUDA_CHECK(pdl::launch_consumer(
            {dim3(grid), dim3(kBlock), 0, stream}, embed_gather_q4_kernel<kScaleMode>, guard,
            static_cast<const std::uint8_t*>(table.qdata),
            static_cast<const std::uint8_t*>(table.scales), bf16_factor(factor),
            static_cast<__nv_bfloat16*>(out.data), d, T,
            static_cast<std::int32_t>(table.padded_shape[1])));
    };
    switch (table.qtype) {
    case QType::Q4_G32_FP16: launch.template operator()<0>(); break;
    case QType::Q4_G128_FP16: launch.template operator()<2>(); break;
    case QType::Q4_G256_FP16: launch.template operator()<3>(); break;
    default: throw std::invalid_argument("embed_gather: not a q4 fp16 table");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
