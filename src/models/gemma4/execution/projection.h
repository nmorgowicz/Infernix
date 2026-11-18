#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/ggml_linear.h"
#include "infernix/ops/linear.h"
#include "infernix/types.h"

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

// GGUF projections by the artifact's input policy. AllowA8 (the converter's default for --weights q4_g32, as on q4
// trunks): calls of up to kGgmlDenseMinTokens - 1 columns take ggml_linear's 8-bit activation route (decode, MTP
// verification); wider calls widen the blocks to E4M3 rows (ops::ggml_widen_e4m3) and run the FP8 A8 GEMM. A16Only
// (--weights q4_g32_a16): narrow calls take the exact A16 route (BF16 tensor-core tiles from 32 columns); wider calls
// decode the blocks into a dense BF16 matrix and run one cuBLAS GEMM (Fp8RowCublasContext::gemm_bf16_fp32, FP32 sums).
inline constexpr ops::LinearRoutes kGgmlRoutes = ops::LinearRoutes::GgmlBf16Wide;
inline constexpr int kGgmlDenseMinTokens = 64;

// Explicit AllowA8 native Q5 trunks bake row-scaled E4M3 only at prefill widths. Narrow
// decode/MTP and every A16Only call retain the stored-code native Q5 A16 route.
inline constexpr int kQ5DenseMinTokens = 64;

// --gemma4-q5-prefill: the route flag the layer's native Q5 projections of kQ5DenseMinTokens or more columns carry
// (ops::LinearRoutes::Q5Int8G32 / Q5Int8G64); a16 is no flag. It takes precedence over an AllowA8 Use's row-FP8 bake
// at those widths; narrower calls never use it.
[[nodiscard]] constexpr ops::LinearRoutes q5_prefill_routes(Gemma4Q5Prefill mode) noexcept {
    switch (mode) {
    case Gemma4Q5Prefill::A16: return ops::LinearRoutes::Exact;
    case Gemma4Q5Prefill::Int8G32: return ops::LinearRoutes::Q5Int8G32;
    case Gemma4Q5Prefill::Int8G64: return ops::LinearRoutes::Q5Int8G64;
    }
    return ops::LinearRoutes::Exact;
}

// out = weight * x for contiguous BF16 x [K,T]. Wide AllowA8 native Q5 uses a transient row-FP8
// bake and the existing FP8 A8 GEMM; wide FP8 A16 with BF16 output uses caller-owned cuBLAS
// resources and per-call widening in the arena. An active GgmlProjection runs its routes below
// (weight then only describes [N,K]); all other calls use ops::linear. `q5_routes` (q5_prefill_routes) selects the
// opt-in INT8-activation route of a native Q5 trunk projection of 64 or more columns, whose scratch is the arena's.
void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream,
             ops::LinearRoutes q5_routes = ops::LinearRoutes::Exact);
void project(const Tensor& x, const Weight& weight, const GgmlProjection& ggml, ops::LinearPolicy policy,
             Tensor& out, ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream,
             ops::LinearRoutes q5_routes = ops::LinearRoutes::Exact);

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
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy, int tokens,
                                           ops::LinearRoutes q5_routes = ops::LinearRoutes::Exact);
[[nodiscard]] std::size_t projection_bytes(const Weight& weight, const GgmlProjection& ggml,
                                           ops::LinearPolicy policy, int tokens,
                                           ops::LinearRoutes q5_routes = ops::LinearRoutes::Exact);

} // namespace infernix::models::gemma4
