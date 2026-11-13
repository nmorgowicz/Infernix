#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace infernix::ops {

/**
 * Program-owned cuBLAS execution resource of fp8_row_cublas: one cuBLAS handle bound to one stream.
 *
 * The handle is created here and nowhere else; no Op holds a handle of its own. It runs with no cuBLAS
 * workspace (fixed algorithms, so the summation order does not depend on the algorithm cuBLAS picks per
 * call) and is used on its stream only. The owner outlives every call that takes it and keeps CUDA
 * Graph capture of those calls on its own stream.
 */
class Fp8RowCublasContext {
public:
    explicit Fp8RowCublasContext(cudaStream_t stream);
    ~Fp8RowCublasContext();

    Fp8RowCublasContext(const Fp8RowCublasContext&)            = delete;
    Fp8RowCublasContext& operator=(const Fp8RowCublasContext&) = delete;

    // partials[n,t] = FP32 sum_k widened[n,k] * x[k,t]: widened is BF16 [K,N] holding the N rows of K
    // contiguous values (ne[0] = K), x is BF16 [K,T], partials is FP32 [N,T]. `stream` must be the
    // context's stream.
    void gemm_bf16_fp32(const Tensor& x, const Tensor& widened, Tensor& partials, cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/**
 * Op: fp8_row_cublas
 *
 * Math / indexing:
 *   out[n,t] = BF16(scale[n] * sum_k e4m3[n,k] * x[k,t])
 *
 * The E4M3 codes are widened to BF16 (exact) in workspace, cuBLAS accumulates and stores FP32,
 * and the BF16 row scale is applied once before the BF16 store: the arithmetic of the FP8 A16
 * kernels behind ops::linear, in cuBLAS's summation order.
 *
 * Logical shapes:
 *   w is FP8_E4M3FN_ROW_BF16 in the RowScale layout with BF16 row scales [N,K], x is contiguous BF16
 *   [K,T] and out contiguous BF16 [N,T]; N, K and T are positive.
 *
 * Effects:
 *   Writes out. out must not overlap x or the weight.
 *
 * Workspace:
 *   fp8_row_cublas_workspace_bytes(N, K, T); it must not overlap x, w or out.
 *
 * Execution:
 *   `gemm` is bound to `stream`; the call enqueues the widening kernel, the cuBLAS GEMM and the scaling
 *   kernel on it.
 */
[[nodiscard]] std::size_t fp8_row_cublas_workspace_bytes(std::int32_t n, std::int32_t k,
                                                         std::int32_t tokens);

void fp8_row_cublas(const Tensor& x, const Weight& w, Tensor& out, Fp8RowCublasContext& gemm,
                    WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
