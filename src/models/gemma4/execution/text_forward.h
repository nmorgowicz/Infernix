#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/gemma4_cache.h"
#include "infernix/ops/linear.h"

#include <cuda_runtime.h>

#include <span>

namespace infernix::models::gemma4 {

// Contiguous BF16 matrix [rows, cols] stored with K fastest: index = k + cols * n.
[[nodiscard]] Weight bf16_matrix(const void* data, int rows, int cols);

// One Gemma 4 text block. x is contiguous BF16 [hidden, T] and is updated in place.
// qkv holds query rows, then key rows, then (sliding layers only) value rows;
// gate_up holds gate rows, then up rows. Policies are the artifact's input permissions.
struct Gemma4LayerWeights {
    const void* input_norm = nullptr;
    Weight qkv;
    const void* q_norm = nullptr;
    const void* k_norm = nullptr;
    Weight o_proj;
    const void* post_attn_norm = nullptr;
    const void* pre_ff_norm = nullptr;
    Weight gate_up;
    Weight down;
    ops::LinearPolicy qkv_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy o_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy gate_up_policy = ops::LinearPolicy::A16Only;
    ops::LinearPolicy down_policy = ops::LinearPolicy::A16Only;
    const void* post_ff_norm = nullptr;
    float layer_scalar = 1.0f;
    int hidden = 0;
    int intermediate = 0;
    int head_dim = 0;
    int q_heads = 0;
    int kv_heads = 0;
    int window = 0;
    int rotary_dim = 0;
    float rope_theta = 10000.0f;
    float rms_eps = 1.0e-6f;
    bool global_layer = false;
};

void gemma4_layer(Tensor& x, const Gemma4LayerWeights& weights, const Tensor& positions,
                  const ops::Gemma4CacheView& cache, ops::Fp8RowCublasContext& gemm,
                  WorkspaceArena& workspace, cudaStream_t stream);

// Complete arena capacity for every positive column count up to tokens, including native BF16
// projection scratch, ready-A8 producer storage and the attention context-split bound. Geometry-only
// Weight descriptors suffice; attention sizing requires the selected device and runtime switches.
[[nodiscard]] std::size_t gemma4_layer_workspace_bytes(const Gemma4LayerWeights& weights, int tokens);
[[nodiscard]] std::size_t gemma4_logits_workspace_bytes(const Weight& output_head, int tokens);

// The gather multiplies rows by this BF16-valued factor: one for prescaled stored rows,
// sqrt(hidden) when the output head shares the unscaled table.
struct Gemma4Embedding {
    Weight table;
    float scale = 1.0f;
};

using Gemma4CacheLayer = ops::Gemma4CacheView;

// Embed ids into hidden [hidden,T] and run every layer, appending K/V at positions.
void gemma4_layers(const Tensor& ids, Tensor& hidden, const Gemma4Embedding& token_embedding,
                   std::span<const Gemma4LayerWeights> layers,
                   std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                   ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Final norm, A16-only output head and logit softcap for hidden [hidden,C] into logits [vocab,C].
// hidden is not modified. The Gemma FP8 vocabulary projection writes BF16 logits.
void gemma4_logits(const Tensor& hidden, Tensor& logits, const void* final_norm,
                   const Weight& output_head, float softcap, float rms_eps,
                   ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

// Embed, run every layer, final norm, output head, then logit softcap.
// hidden and logits are caller storage, [hidden,T] and [vocab,T].
void gemma4_prefill(const Tensor& ids, Tensor& hidden, Tensor& logits,
                    const Gemma4Embedding& token_embedding, const void* final_norm,
                    const Weight& output_head, float softcap, float rms_eps,
                    std::span<const Gemma4LayerWeights> layers,
                    std::span<const Gemma4CacheLayer> caches, const Tensor& positions,
                    ops::Fp8RowCublasContext& gemm, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::models::gemma4
