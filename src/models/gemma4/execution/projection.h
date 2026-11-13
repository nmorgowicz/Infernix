#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::models::gemma4 {

[[nodiscard]] inline Tensor norm_weight(const void* data, int width) {
    return Tensor(const_cast<void*>(data), DType::BF16, {width});
}

// out = weight * x for contiguous BF16 x [K,T]. BF16 uses native ops::linear; wide FP8 A16
// with BF16 output uses caller-owned cuBLAS resources and per-call widening in the arena.
void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Producer-owned ready E4M3 activation [rows,tokens] and one FP32 scale per column.
struct DownA8 {
    std::uint8_t* codes = nullptr;
    float* scales = nullptr;
};

[[nodiscard]] DownA8 prepare_down_a8(WorkspaceArena& workspace, int rows, int tokens);
[[nodiscard]] std::size_t down_a8_bytes(int rows, int tokens);

void project_down_ready(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
                        DownA8 ready, ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace,
                        cudaStream_t stream);

// Complete arena capacity project can take for any positive column count up to tokens.
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy, int tokens);

} // namespace infernix::models::gemma4
