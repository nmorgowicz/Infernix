// infernix::ops - Gemma 4 hidden-5376 RMSNorm launcher (order-preserving single-pass kernel).
#include "ops/launcher/gemma4_rmsnorm.h"

#include "ops/kernel/rmsnorm.cuh"
#include "core/device.h"
#include "core/pdl.cuh"

namespace infernix::ops::detail {
namespace {

template <RmsEpilogue Epilogue>
void launch_order(const Tensor& x, const Tensor& weight, Tensor& out, std::int64_t rows, float eps,
                  cudaStream_t stream) {
    // The generic kernel took 10.7 us per one-token row. Its summation order is kept: a prose
    // parity prompt amplifies a reordered sum at one token from 2.5% to 66% hidden error within
    // two layers, although each layer stays exact.
    CUDA_CHECK(pdl::launch_consumer(
        {dim3(static_cast<unsigned>(rows)), dim3(kRmsGenericBlock), 0, stream},
        rmsnorm_generic_order_kernel<Epilogue, 21>, static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(weight.data), static_cast<__nv_bfloat16*>(out.data), rows,
        eps));
}

} // namespace

void rmsnorm_gemma4_hidden_launch(const Tensor& x, const Tensor& weight, bool unit_offset,
                                  Tensor& out, std::int64_t rows, float eps, cudaStream_t stream) {
    if (unit_offset) {
        launch_order<RmsEpilogue::Offset>(x, weight, out, rows, eps, stream);
    } else {
        launch_order<RmsEpilogue::Plain>(x, weight, out, rows, eps, stream);
    }
}

} // namespace infernix::ops::detail
