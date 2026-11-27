#pragma once

#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <cuda_runtime.h>

namespace infernix::ops {

// Which K/V rows gemma4_ring_copy moves, as absolute positions, from the first position of `positions`
// (device value F; a prompt call's positions are F, F+1, ..., F+tokens-1):
//   Window: [max(0, F - window_rows), F)                  the rows before the call a chunk attends to;
//   Tail:   [max(F, F + tokens - capacity_rows), F + tokens)   the call's own rows a ring of capacity_rows keeps.
enum class Gemma4RingCopyRows { Window, Tail };

/**
 * Row-exact copy of sliding-ring K/V rows between two page layouts of the same layer geometry.
 *
 * `src` and `dst` are Gemma4CacheView descriptors of the same format (BF16 or FP8 row-256 with scale
 * planes), head dimension and KV head count; each has its own pages and its own block_table (logical
 * page p/64 -> physical page). Row p is read at src.block_table[p/64], slot p%64 and written at
 * dst.block_table[p/64], slot p%64, for K, V and (FP8) both scale planes: bytes are copied, never
 * re-encoded. Only the selected rows are written; every other destination byte is unchanged. The row
 * range is computed on the device from positions[0], so the launch is graph-capturable and replays with
 * a changed frontier. `tokens` is the call's width (Tail range and launch bound); `window_rows` the
 * Window range's length and bound; `capacity_rows` the destination ring's rows (Tail).
 *
 * Preconditions, not checked on the device: the selected rows map to distinct destination slots, src and
 * dst planes do not overlap, positions is contiguous (positions[i] = positions[0] + i), no key-derived
 * values. Storage must stay valid until work on stream completes.
 */
void gemma4_ring_copy(const Gemma4CacheView& src, const Gemma4CacheView& dst, const Tensor& positions,
                      Gemma4RingCopyRows rows, int tokens, int window_rows, int capacity_rows,
                      cudaStream_t stream);

} // namespace infernix::ops
