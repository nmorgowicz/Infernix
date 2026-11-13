#pragma once

#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace infernix::ops::detail {

// From one full 64-row query tile on, prefill goes through the tensor-core kernel.
inline constexpr std::int32_t kGemma4AttentionMmaMinTokens = 64;

// Number of query rows in one D=512 tensor-core query tile.
inline constexpr std::int32_t kGemma4AttentionD512TileRows = 64;

// Number of context split-K segments the D=512 tensor-core kernel will use; 1 means the
// single-kernel path. pages <= 0 returns the context-unclamped bound used for workspace
// sizing. INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K is read once before capture: 0 (unset)
// selects the wave-aware heuristic, 1 disables splitting, and 2..13 pins that many
// segments without a context clamp. The same selection sizes the workspace.
int gemma4_attention_mma_d512_segments(std::int32_t tokens, std::int32_t q_heads,
                                       std::int32_t pages, std::int32_t window);

// Number of key chunks per query; 1 means no split and no workspace.
int gemma4_attention_chunks(std::int32_t tokens, std::int32_t kv_heads);

// Widest query group (query heads per KV head) the KV-head-shared D=512 decode kernel serves.
inline constexpr std::int32_t kGemma4DecodeD512MaxGroup = 8;

// True when (head_dim, q_heads, kv_heads) fits the
// KV-head-shared D=512 decode kernel (head_dim 512, group <= kGemma4DecodeD512MaxGroup).
bool gemma4_decode_d512_selected(std::int32_t head_dim, std::int32_t q_heads,
                                 std::int32_t kv_heads);

// Key chunks per query for the KV-head-shared D=512 decode kernel: the largest count whose
// (tokens x kv_heads x chunks) grid fits one wave of resident CTAs on this device, at most
// 128. Depends only on (tokens, kv_heads) and the device, never on the live context, so the
// grid and workspace stay fixed per CUDA-graph capture. 1 means no split and no workspace.
int gemma4_decode_d512_chunks(std::int32_t tokens, std::int32_t kv_heads);

// Chunk count the SIMT decode path (query widths < 64) uses for this shape: the shared
// D=512 kernel's when selected, else gemma4_attention_chunks.
int gemma4_attention_simt_chunks(std::int32_t tokens, std::int32_t q_heads,
                                 std::int32_t kv_heads, std::int32_t head_dim);

// KV-head-shared D=512 decode kernel (BF16 or FP8 KV). partial/stats use the layout of
// gemma4_attention_launch with gemma4_decode_d512_chunks parts.
void gemma4_attention_decode_d512_launch(const Tensor& q, const Gemma4CacheView& cache,
                                         const Tensor& positions, int window, float scale,
                                         Tensor& out, float* partial, float* partial_stats,
                                         cudaStream_t stream);

// partial is FP32 [D, parts, Hq, T] and stats FP32 [2, parts, Hq, T], unused for parts=1.
// parts is gemma4_attention_chunks for SIMT or gemma4_attention_mma_d512_segments for D512 MMA.
void gemma4_attention_launch(const Tensor& q, const Gemma4CacheView& cache,
                             const Tensor& positions, int window, float scale, Tensor& out,
                             float* partial, float* stats, cudaStream_t stream);

// Tensor-core path for wide prefill chunks (D=256 or 512). The D=512 kernel may split the
// context into segments and then needs the same partial/stats workspace as above.
void gemma4_attention_mma_launch(const Tensor& q, const Gemma4CacheView& cache,
                                 const Tensor& positions, int window, float scale, Tensor& out,
                                 float* partial, float* stats, cudaStream_t stream);

// The D=512 kernel behind gemma4_attention_mma_launch.
void gemma4_attention_mma_d512_launch(const Tensor& q, const Gemma4CacheView& cache,
                                      const Tensor& positions, int window, float scale,
                                      Tensor& out, float* partial, float* partial_stats,
                                      cudaStream_t stream);

// Diagnostic query for the exact D512 specialization selected by the cache and
// QK mode. Call outside capture/timing; occupancy is the CUDA API limit,
// not a measured resident-warp guarantee.
struct Gemma4AttentionD512Resources {
    cudaFuncAttributes attributes{};
    std::size_t dynamic_shared_bytes = 0;
    int threads = 0;
    int active_blocks_per_sm = 0;
};
Gemma4AttentionD512Resources gemma4_attention_mma_d512_resources(const Gemma4CacheView& cache);

} // namespace infernix::ops::detail
