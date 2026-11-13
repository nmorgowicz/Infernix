#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * Copy BF16 K and V into a paged cache.
 *
 * k and v are contiguous BF16 [D,Hkv,T]. positions is contiguous I32 [T]. k_pages and v_pages
 * are contiguous BF16 [D,64,Hkv,pages]. block_table is contiguous I32 [logical_pages]. Absolute
 * position p is stored at physical page block_table[p/64], slot p%64. D is 256 or 512. The Op
 * writes only the addressed K and V elements; all other cache elements are unchanged.
 * Negative positions, positions beyond the logical block table, and negative or
 * out-of-range physical page mappings are skipped without forming a cache address.
 *
 * The caller must provide disjoint destination planes, neither overlapping k, v,
 * positions or block_table. Each token must map to a distinct physical page/slot;
 * duplicate destinations would race and have no defined ordering. These alias and
 * uniqueness preconditions are not checked by the Op. Inputs and page mappings are
 * not modified. All storage must remain valid until work on stream completes.
 */
void gemma4_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, Tensor& k_pages,
                      Tensor& v_pages, const Tensor& block_table, cudaStream_t stream);

} // namespace infernix::ops
