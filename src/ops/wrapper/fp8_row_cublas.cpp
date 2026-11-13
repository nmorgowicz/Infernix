#include "infernix/ops/fp8_row_cublas.h"

#include "core/device.h"
#include "ops/launcher/fp8_row_cublas.h"
#include "ops/linear/fp8/fp8_format.h"

#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::size_t kAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

void check_cublas(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) { return; }
    std::ostringstream message;
    message << operation << " failed with cuBLAS status " << static_cast<int>(status);
    throw std::runtime_error(message.str());
}

void require_matrix(const Tensor& tensor, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* name) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr ||
        tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("fp8_row_cublas: ") + name + " must be contiguous " +
                                    (dtype == DType::BF16 ? "BF16" : "FP32") + " [" +
                                    std::to_string(rows) + "," + std::to_string(columns) + "]");
    }
}

} // namespace

struct Fp8RowCublasContext::Impl {
    cublasHandle_t handle = nullptr;
    cudaStream_t stream   = nullptr;
};

Fp8RowCublasContext::Fp8RowCublasContext(cudaStream_t stream) : impl_(std::make_unique<Impl>()) {
    if (stream == nullptr) { throw std::invalid_argument("fp8_row_cublas: the context requires a stream"); }
    check_cublas(cublasCreate(&impl_->handle), "cublasCreate");
    try {
        check_cublas(cublasSetStream(impl_->handle, stream), "cublasSetStream");
        check_cublas(cublasSetMathMode(impl_->handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
        // No workspace: fixed algorithms, the setting the stored greedy prefixes were made with.
        check_cublas(cublasSetWorkspace(impl_->handle, nullptr, 0), "cublasSetWorkspace");
        impl_->stream = stream;
    } catch (...) {
        (void)cublasDestroy(impl_->handle);
        throw;
    }
}

Fp8RowCublasContext::~Fp8RowCublasContext() {
    if (impl_ && impl_->handle != nullptr) { (void)cublasDestroy(impl_->handle); }
}

void Fp8RowCublasContext::gemm_bf16_fp32(const Tensor& x, const Tensor& widened, Tensor& partials,
                                         cudaStream_t stream) {
    if (stream != impl_->stream) {
        throw std::invalid_argument("fp8_row_cublas: the context was used with a different stream");
    }
    const std::int32_t k      = widened.ne[0];
    const std::int32_t n      = widened.ne[1];
    const std::int32_t tokens = x.ne[1];
    require_matrix(widened, DType::BF16, k, n, "widened weight");
    require_matrix(x, DType::BF16, k, tokens, "x");
    require_matrix(partials, DType::FP32, n, tokens, "partials");
    const float alpha = 1.0F;
    const float beta  = 0.0F;
    check_cublas(cublasGemmEx(impl_->handle, CUBLAS_OP_T, CUBLAS_OP_N, n, tokens, k, &alpha,
                              widened.data, CUDA_R_16BF, k, x.data, CUDA_R_16BF, k, &beta,
                              partials.data, CUDA_R_32F, n, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                 "cublasGemmEx");
}

std::size_t fp8_row_cublas_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens) {
    if (n <= 0 || k <= 0 || tokens <= 0) {
        throw std::invalid_argument("fp8_row_cublas: extents must be positive");
    }
    const auto rows = static_cast<std::size_t>(n);
    return aligned(rows * static_cast<std::size_t>(k) * 2) +
           aligned(rows * static_cast<std::size_t>(tokens) * 4) + kAlignment;
}

void fp8_row_cublas(const Tensor& x, const Weight& w, Tensor& out, Fp8RowCublasContext& gemm,
                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (w.qtype != QType::FP8_E4M3FN_ROW_BF16) {
        throw std::invalid_argument("fp8_row_cublas: weight must be FP8_E4M3FN_ROW_BF16");
    }
    (void)detail::validate_fp8_weight(w, "fp8_row_cublas");
    const std::int32_t tokens = x.ne[1];
    require_matrix(x, DType::BF16, w.k, tokens, "x");
    require_matrix(out, DType::BF16, w.n, tokens, "out");
    if (tokens <= 0) { throw std::invalid_argument("fp8_row_cublas: T must be positive"); }

    const std::int64_t codes = static_cast<std::int64_t>(w.n) * w.k;
    const auto scope         = workspace.scope();
    Tensor widened           = workspace.alloc(DType::BF16, {w.k, w.n});
    Tensor partials          = workspace.alloc(DType::FP32, {w.n, tokens});
    detail::fp8_widen_codes_launch(static_cast<const std::uint8_t*>(w.qdata),
                                   static_cast<__nv_bfloat16*>(widened.data), codes, stream);
    gemm.gemm_bf16_fp32(x, widened, partials, stream);
    detail::fp8_scale_rows_launch(static_cast<const float*>(partials.data),
                                  static_cast<const __nv_bfloat16*>(w.scales),
                                  static_cast<__nv_bfloat16*>(out.data), w.n,
                                  static_cast<std::int64_t>(w.n) * tokens, stream);
}

} // namespace infernix::ops
