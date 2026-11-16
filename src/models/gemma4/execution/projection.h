#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace infernix::models::gemma4 {

[[nodiscard]] inline Tensor norm_weight(const void* data, int width) {
    return Tensor(const_cast<void*>(data), DType::BF16, {width});
}

// A projection stored as GGUF blocks (a llama.cpp GGUF import, tools/convert/gemma4_gguf.py), run by
// ops::ggml_linear: its segments in output-row order (one per stored parent region) and, for a global
// layer's output projection, the block permutation that reads the attention output (written in the
// converter's prope order) in the stored matrix's source column order. Empty for every other format.
struct GgmlProjection {
    std::vector<ops::GgmlSegment> segments;
    const std::int32_t* input_perm = nullptr; // device, K / perm_block entries
    std::int32_t perm_block = 0;
    [[nodiscard]] bool active() const noexcept { return !segments.empty(); }
};

// GGUF projections take the exact A16 route at decode widths (and ggml_linear's BF16 tensor-core tiles up to
// kGgmlDenseMinTokens - 1 columns); from kGgmlDenseMinTokens columns, prefill decodes the blocks into a dense BF16
// matrix in the workspace and runs one cuBLAS GEMM (Fp8RowCublasContext::gemm_bf16_fp32, FP32 sums).
inline constexpr ops::LinearRoutes kGgmlRoutes = ops::LinearRoutes::GgmlBf16Wide;
inline constexpr int kGgmlDenseMinTokens = 64;

// out = weight * x for contiguous BF16 x [K,T]. BF16 uses native ops::linear; wide FP8 A16
// with BF16 output uses caller-owned cuBLAS resources and per-call widening in the arena; an active
// GgmlProjection runs ops::ggml_linear instead (weight then only describes its [N,K]).
void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);
void project(const Tensor& x, const Weight& weight, const GgmlProjection& ggml, ops::LinearPolicy policy,
             Tensor& out, ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

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
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, const GgmlProjection& ggml,
                                           ops::LinearPolicy policy, int tokens);

} // namespace infernix::models::gemma4
