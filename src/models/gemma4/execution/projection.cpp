#include "models/gemma4/execution/projection.h"

#include "core/temporary_switches.h"
#include "infernix/ops/cast.h"

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

// INFERNIX_TMP_GGML_A8: GGUF projections at prefill widths widen to E4M3 rows and take the FP8 A8 GEMM.
bool ggml_a8() {
    static const bool enabled = temporary_switches::flag("INFERNIX_TMP_GGML_A8");
    return enabled;
}

// INFERNIX_TMP_GGML_DECODE_A8: GGUF projections at decode widths add the 8-bit activation route.
ops::LinearRoutes ggml_narrow_routes() {
    static const bool a8 = temporary_switches::flag("INFERNIX_TMP_GGML_DECODE_A8");
    return a8 ? kGgmlRoutes | ops::LinearRoutes::GgmlA8 : kGgmlRoutes;
}

// The FP8_E4M3FN_ROW_BF16 descriptor of `rows` x `k` codes at `payload` with the row multipliers at the
// format's scale plane offset (the layout fp8_row_geometry checks).
Weight fp8_row_weight(void* payload, int rows, int k) {
    const std::size_t scale_offset = aligned(checked_bytes(rows, k, 1));
    Weight w;
    w.qtype = QType::FP8_E4M3FN_ROW_BF16;
    w.layout = QuantLayout::RowScale;
    w.scale_dtype = DType::BF16;
    w.payload = payload;
    w.payload_bytes = scale_offset + checked_bytes(rows, 1, 2);
    w.qdata = payload;
    w.scales = static_cast<std::uint8_t*>(payload) + scale_offset;
    w.n = rows;
    w.k = k;
    w.group = k;
    w.group_size = static_cast<std::uint32_t>(k);
    w.ndim = 2;
    w.shape[0] = w.padded_shape[0] = rows;
    w.shape[1] = w.padded_shape[1] = k;
    w.shape[2] = w.shape[3] = w.padded_shape[2] = w.padded_shape[3] = 1;
    w.scale_ne[0] = rows;
    w.scale_ne[1] = w.scale_ne[2] = w.scale_ne[3] = 1;
    w.scale_nb[0] = 2;
    w.scale_nb[1] = w.scale_nb[2] = w.scale_nb[3] = static_cast<std::size_t>(rows) * 2;
    return w;
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

void project(const Tensor& x, const Weight& weight, const GgmlProjection& ggml, ops::LinearPolicy policy,
             Tensor& out, ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream) {
    if (!ggml.active()) {
        project(x, weight, policy, out, gemm, workspace, stream);
        return;
    }
    if (policy != ops::LinearPolicy::A16Only) {
        throw std::invalid_argument("Gemma4 projection: GGUF blocks run on 16-bit activations (A16Only)");
    }
    if (x.ne[1] >= kGgmlDenseMinTokens && out.dtype == DType::BF16 && ggml_a8()) {
        // Experiment (INFERNIX_TMP_GGML_A8): E4M3 rows with BF16 row multipliers, then the FP8 A8 GEMM.
        auto scope = workspace.scope();
        const auto payload = workspace.alloc_bytes(aligned(checked_bytes(weight.n, weight.k, 1)) +
                                                       checked_bytes(weight.n, 1, 2),
                                                   kAlignment);
        const Weight fp8 = fp8_row_weight(payload.data, weight.n, weight.k);
        Tensor codes(const_cast<void*>(fp8.qdata), DType::U8, {weight.k, weight.n});
        Tensor scales(const_cast<void*>(fp8.scales), DType::BF16, {weight.n});
        ops::ggml_widen_e4m3(ggml.segments, weight.k, codes, scales, ggml.input_perm, ggml.perm_block, stream);
        ops::linear(x, fp8, out, ops::LinearPolicy::AllowA8, workspace, stream);
        return;
    }
    if (x.ne[1] >= kGgmlDenseMinTokens && out.dtype == DType::BF16) {
        // Prefill: decode the blocks once into a dense BF16 matrix (global output columns already in the attention
        // output's order), one cuBLAS GEMM with FP32 sums, one rounding to BF16.
        auto scope = workspace.scope();
        Tensor widened = workspace.alloc(DType::BF16, {weight.k, weight.n});
        ops::ggml_dequantize_bf16(ggml.segments, weight.k, widened, ggml.input_perm, ggml.perm_block, stream);
        Tensor partials = workspace.alloc(DType::FP32, {weight.n, x.ne[1]});
        gemm.gemm_bf16_fp32(x, widened, partials, stream);
        ops::cast_fp32_to_bf16(partials, out, stream);
        return;
    }
    ops::ggml_linear(x, ggml.segments, out, ggml.input_perm, ggml.perm_block, workspace, stream, ggml_narrow_routes());
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

std::size_t projection_bytes(const Weight& weight, const GgmlProjection& ggml, ops::LinearPolicy policy, int tokens) {
    if (!ggml.active()) { return projection_bytes(weight, policy, tokens); }
    // ggml_linear's workspace depends on the call's width (encoding batches, wide-route planes): the largest
    // over every width it takes; from kGgmlDenseMinTokens the dense route's matrix and FP32 partials.
    std::size_t bytes = 0;
    for (int columns = 1; columns <= std::min(tokens, kGgmlDenseMinTokens - 1); ++columns) {
        bytes = std::max(bytes, ops::ggml_linear_workspace_bytes(weight.k, columns));
    }
    if (tokens >= kGgmlDenseMinTokens) {
        bytes = std::max(bytes, aligned(checked_bytes(weight.n, weight.k, 2)) +
                                    aligned(checked_bytes(weight.n, tokens, 4)) + 2 * kAlignment);
        // The INFERNIX_TMP_GGML_A8 arm (Gemma trunk shapes): E4M3 rows and multipliers, then the FP8 A8 GEMM's own
        // workspace.
        if (ggml_a8()) {
            bytes = std::max(bytes, aligned(aligned(checked_bytes(weight.n, weight.k, 1)) + checked_bytes(weight.n, 1, 2)) +
                                        ops::linear_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16, weight.n,
                                                                             weight.k, ops::LinearPolicy::AllowA8,
                                                                             kGgmlDenseMinTokens, tokens) +
                                        2 * kAlignment);
        }
    }
    return bytes;
}

} // namespace infernix::models::gemma4
