#include "models/gemma4/execution/projection.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace infernix::models::gemma4 {
namespace {

// At this width per-call code widening amortizes the FP8 A16 tensor-core gap.
constexpr int kFp8CublasMinTokens = 512;
constexpr std::size_t kAlignment = 256;

bool fp8_through_cublas(const Weight& weight, ops::LinearPolicy policy, int tokens) {
    // Q4 stays in ops::linear; it must not widen through the FP8 route.
    return weight.qtype == QType::FP8_E4M3FN_ROW_BF16 &&
           policy == ops::LinearPolicy::A16Only && tokens >= kFp8CublasMinTokens;
}

std::size_t checked_bytes(int rows, int tokens, std::size_t element_bytes) {
    if (rows <= 0 || tokens <= 0) {
        throw std::invalid_argument("Gemma4 projection: extents must be positive");
    }
    const auto width = static_cast<std::size_t>(rows);
    const auto columns = static_cast<std::size_t>(tokens);
    if (width > (std::numeric_limits<std::size_t>::max() - kAlignment) / columns / element_bytes) {
        throw std::overflow_error("Gemma4 projection: workspace size overflow");
    }
    return width * columns * element_bytes;
}

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace

void project(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
             ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (out.dtype == DType::BF16 && fp8_through_cublas(weight, policy, x.ne[1])) {
        ops::fp8_row_cublas(x, weight, out, gemm, workspace, stream);
    } else {
        ops::linear(x, weight, out, policy, workspace, stream);
    }
}

DownA8 prepare_down_a8(WorkspaceArena& workspace, int rows, int tokens) {
    const auto codes = workspace.alloc_bytes(checked_bytes(rows, tokens, 1), kAlignment);
    const auto scales = workspace.alloc_bytes(checked_bytes(1, tokens, sizeof(float)), kAlignment);
    return {static_cast<std::uint8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t down_a8_bytes(int rows, int tokens) {
    const auto codes = aligned(checked_bytes(rows, tokens, 1));
    const auto scales = aligned(checked_bytes(1, tokens, sizeof(float)));
    if (codes > std::numeric_limits<std::size_t>::max() - scales - kAlignment) {
        throw std::overflow_error("Gemma4 projection: workspace size overflow");
    }
    return codes + scales + kAlignment;
}

void project_down_ready(const Tensor& x, const Weight& weight, ops::LinearPolicy policy, Tensor& out,
                        DownA8 ready, ops::Fp8RowCublasContext&, WorkspaceArena& workspace,
                        cudaStream_t stream) {
    ops::linear_a8_ready(x, weight, out, policy, ready.codes, ready.scales, workspace, stream);
}

std::size_t projection_bytes(const Weight& weight, ops::LinearPolicy policy, int tokens) {
    // Query BF16 too: its native fallback owns its workspace contract, not this family.
    const auto linear = ops::linear_workspace_capacity_bytes(weight.qtype, weight.n, weight.k,
                                                             policy, 1, tokens);
    if (!fp8_through_cublas(weight, policy, tokens)) { return linear; }
    return std::max(linear, ops::fp8_row_cublas_workspace_bytes(weight.n, weight.k, tokens));
}

} // namespace infernix::models::gemma4
