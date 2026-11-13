#pragma once

#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Gemma 4 attention input preparation in one pass over the fused projection output.
 *
 * qkv is contiguous BF16 [Hq*D + Hkv*D (+ Hkv*D), T]: Q heads, then K heads, then V heads on
 * sliding layers. Global layers have no V rows (v_from_k): V is the raw K. For each token t at
 * position p = positions[t]:
 *
 *   q_out[:,h,t]  = rope(rmsnorm(Q_h, q_norm))
 *   K_h           = rope(rmsnorm(K_h, k_norm))
 *   V_h           = rmsnorm(V_h or raw K_h, 1)
 *
 * rmsnorm is x * (mean(x^2) + eps)^-1/2 * weight with each result rounded to BF16, and rope is the
 * split-half NeoX rotation of dimensions [0, rotary_dim) at angle p * theta^(-2i/rotary_dim),
 * as in ops::rmsnorm and ops::rope. K_h and V_h are stored at physical page block_table[p/64],
 * slot p%64 of cache.k_pages and cache.v_pages; tokens with a negative position or a
 * page outside the table write no K/V. D is 256 (rotary_dim 256) or 512 (rotary_dim 128).
 * BF16 output is bit-identical to rmsnorm, rope and gemma4_kv_append in sequence.
 * Global FP8 (D512, v_from_k) quantizes these final BF16-valued K/V per row-half,
 * using FP16-rounded bounded scales before reciprocal/encoding. A row-half containing
 * any nonfinite value stores canonical NaN codes with unit scale (including infinities);
 * finite rows are the supported domain. q_out stays BF16 [D,Hq,T]. No workspace.
 */
void gemma4_qkv_prep(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                     const Tensor& positions, int rotary_dim, float theta, float eps, bool v_from_k,
                     Tensor& q_out, const Gemma4CacheView& cache, cudaStream_t stream);

} // namespace infernix::ops
