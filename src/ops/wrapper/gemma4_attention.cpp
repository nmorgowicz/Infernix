#include "infernix/ops/gemma4_attention.h"

#include "ops/launcher/gemma4_attention.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr std::int32_t kPage = 64;

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_attention: ") + name +
                                    " must be contiguous non-null BF16");
    }
}

constexpr std::size_t kAlignment = 256;

std::size_t aligned(std::size_t bytes) {
    return (bytes + kAlignment - 1) / kAlignment * kAlignment;
}

} // namespace

std::size_t gemma4_attention_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                      std::int32_t kv_heads, std::int32_t tokens) {
    std::size_t widest = 0;
    if (head_dim <= 0 || q_heads <= 0 || kv_heads <= 0 || tokens <= 0) { return widest; }
    for (std::int32_t count = 1; count <= tokens; ++count) {
        int parts = 1;
        if (count < detail::kGemma4AttentionMmaMinTokens) {
            parts = detail::gemma4_attention_chunks(count, kv_heads);
            // Bound both decode kernels so the workspace never depends on the switch.
            if (head_dim == 512 && q_heads % kv_heads == 0 &&
                q_heads / kv_heads <= detail::kGemma4DecodeD512MaxGroup) {
                parts = std::max(parts, detail::gemma4_decode_d512_chunks(count, kv_heads));
            }
        } else if (head_dim == 512) {
            // Unclamped bound covers every context length a graph bucket can reach. The
            // wave-aware segment search is not monotonic in the width, so scan every
            // count; the loop runs once per program at chunk-width scale.
            parts = detail::gemma4_attention_mma_d512_segments(count, q_heads, 0, 0);
        }
        const std::size_t rows =
            parts > 1 ? static_cast<std::size_t>(parts) * q_heads * count : 0;
        if (rows > 0) {
            widest = std::max(widest, aligned(rows * head_dim * sizeof(float)) +
                                          aligned(rows * 2 * sizeof(float)));
        }
    }
    return widest;
}

void gemma4_attention(const Tensor& q, const Gemma4CacheView& cache,
                      const Tensor& positions, int window, float scale,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream) {
    require_bf16(q, "q");
    detail::validate_gemma4_cache(cache, q.ne[0], window == 0, "gemma4_attention");
    const Tensor& k_pages = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    require_bf16(out, "out");
    if (positions.dtype != DType::I32 || block_table.dtype != DType::I32 ||
        !positions.is_contiguous() || !block_table.is_contiguous() || positions.data == nullptr ||
        block_table.data == nullptr) {
        throw std::invalid_argument("gemma4_attention: positions and block_table must be I32");
    }
    if (window < 0) { throw std::invalid_argument("gemma4_attention: window must be nonnegative"); }
    if (!(scale > 0.0f) || !std::isfinite(scale)) {
        throw std::invalid_argument("gemma4_attention: scale must be positive and finite");
    }
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = k_pages.ne[2];
    if ((head_dim != 256 && head_dim != 512) || q_heads <= 0 || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || tokens <= 0 || q.ne[3] != 1) {
        throw std::invalid_argument("gemma4_attention: unsupported q shape");
    }
    if (out.ne[0] != head_dim || out.ne[1] != q_heads || out.ne[2] != tokens || out.ne[3] != 1) {
        throw std::invalid_argument("gemma4_attention: out shape must match q");
    }
    if (positions.ne[0] != tokens || positions.ne[1] != 1) {
        throw std::invalid_argument("gemma4_attention: positions must be [T]");
    }
    // Plane geometry and represented storage are validated together above.
    if (block_table.ne[0] <= 0) {
        throw std::invalid_argument("gemma4_attention: block_table must be nonempty");
    }
    const auto q_bytes   = static_cast<const char*>(q.data);
    const auto out_bytes = static_cast<const char*>(out.data);
    const auto bytes     = static_cast<std::int64_t>(head_dim) * q_heads * tokens * 2;
    if ((out_bytes >= q_bytes && out_bytes < q_bytes + bytes) ||
        (q_bytes >= out_bytes && q_bytes < out_bytes + bytes)) {
        throw std::invalid_argument("gemma4_attention: q and out must not overlap");
    }
    auto scope       = workspace.scope();
    float* partial   = nullptr;
    float* stats     = nullptr;
    if (head_dim == 512 && tokens >= detail::kGemma4AttentionMmaMinTokens) {
        // The D=512 tensor-core kernel splits the context; the stride must match the
        // segments the launcher computes for this exact context.
        const int segments = detail::gemma4_attention_mma_d512_segments(
            tokens, q_heads, block_table.ne[0], window);
        if (segments > 1) {
            partial = static_cast<float*>(
                workspace.alloc(DType::FP32, {head_dim, segments, q_heads, tokens}).data);
            stats = static_cast<float*>(
                workspace.alloc(DType::FP32, {2, segments, q_heads, tokens}).data);
        }
    } else if (tokens < detail::kGemma4AttentionMmaMinTokens) {
        const int chunks =
            detail::gemma4_attention_simt_chunks(tokens, q_heads, kv_heads, head_dim);
        if (chunks > 1) {
            partial = static_cast<float*>(
                workspace.alloc(DType::FP32, {head_dim, chunks, q_heads, tokens}).data);
            stats =
                static_cast<float*>(workspace.alloc(DType::FP32, {2, chunks, q_heads, tokens}).data);
        }
    }
    detail::gemma4_attention_launch(q, cache, positions, window, scale, out,
                                    partial, stats, stream);
}

} // namespace infernix::ops
