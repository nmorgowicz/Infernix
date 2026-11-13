#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

// Workspace gemma4_attention takes from the arena for any call with at most T query tokens.
// Nonzero only when the grid is too small to fill the device and keys are split across blocks.
// D512 tensor-core splits reserve the context-independent bound before graph capture.
// The active device and runtime switches must be selected before sizing/capture.
[[nodiscard]] std::size_t gemma4_attention_workspace_capacity_bytes(std::int32_t head_dim,
                                                                    std::int32_t q_heads,
                                                                    std::int32_t kv_heads,
                                                                    std::int32_t tokens);

/**
 * Causal grouped-query attention over a Gemma paged KV cache.
 *
 *   score[j] = scale * dot(q[:,h,i], k[:,kv,j])
 *   out[:,h,i] = softmax(score) * v[:,kv,j]
 *
 * Key j is visible to query position p when j <= p and (window == 0 or p - j < window).
 * Query head h reads KV head floor(h / (Hq/Hkv)). Admitted profiles are D=256 with any
 * positive Hq divisible by Hkv, and D=512 with the same rule. Gemma 4 uses Hq=32, Hkv=16
 * (sliding, window 1024) and Hq=32, Hkv=4 (global, window 0), both with scale 1.
 *
 * q and out are contiguous, disjoint BF16 [D,Hq,T]. positions is contiguous I32 [T]
 * containing nonnegative absolute query positions. window is nonnegative; scale is
 * positive and finite. Output storage must not alias positions or any cache plane.
 *
 * cache follows Gemma4CacheView's plane layout. FP8 is accepted only for D512/global
 * (window zero). With BF16 QK, each FP8 K/V code times its stored FP16 group scale
 * is multiplied in FP32 and rounded to BF16 before use. Gemma4QkMode::Fp8 requires
 * an FP8 cache; for T >= 64 it quantizes each query into two independent 256-element
 * E4M3 groups using the cache codec's represented FP16 scales and bounded reciprocal.
 * A zero group uses the codec's zero-row rule; any nonfinite query element poisons
 * its entire group with canonical E4M3 NaNs and unit scale. Cached K codes/scales
 * are consumed directly, without re-encoding. The QK dot is the sum over groups g
 * of dot(Q_codes[g], K_codes[g]) * (Q_scale[g] * K_scale[g]), accumulated in FP32;
 * scale multiplies that score. V retains the BF16 expansion described above.
 * For T < 64, QK uses the BF16 query and expanded K even when FP8 QK is selected.
 *
 * block_table maps absolute key position j to physical page block_table[j/64],
 * slot j%64. Keys beyond the logical table, or in negative/out-of-range physical
 * pages, do not participate in softmax. A query with no visible mapped keys writes
 * an all-zero row. The Op writes all of out and never mutates the cache or inputs.
 * Context splitting can change FP32 reduction order; results need not be bit-exact
 * across split selections.
 *
 * Workspace allocations are scoped to this call; the caller owns the arena storage
 * and must reserve the capacity above on the selected device before capture.
 * Keep device and temporary-switch selection fixed between sizing and capture.
 * All input, output and workspace storage must remain alive until stream work
 * completes, and captured addresses must remain valid for every graph replay.
 * Concurrent calls/replays must not reuse overlapping output or workspace storage.
 */
void gemma4_attention(const Tensor& q, const Gemma4CacheView& cache,
                      const Tensor& positions, int window, float scale,
                      Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace infernix::ops
