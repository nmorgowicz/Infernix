#pragma once

#include "infernix/ops/gemma4_ring_copy.h"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace infernix::ops::detail {

// Four planes in Gemma4CacheView order: K codes, V codes, K scales, V scales. Scale planes are absent
// (planes == 2) for a BF16 ring.
struct Gemma4RingCopyPlanes {
    const std::byte* src[4] = {};
    std::byte* dst[4] = {};
    std::int32_t row_bytes[4] = {};   // bytes of one row of one KV head in the plane
    std::int32_t planes = 0;
};

struct Gemma4RingCopyLaunch {
    Gemma4RingCopyPlanes planes;
    const std::int32_t* src_table = nullptr;
    const std::int32_t* dst_table = nullptr;
    const std::int32_t* positions = nullptr;
    std::int32_t table_pages = 0;      // logical pages both tables cover
    std::int32_t src_pages = 0;
    std::int32_t dst_pages = 0;
    std::int32_t kv_heads = 0;
    bool window = false;
    std::int32_t tokens = 0;
    std::int32_t window_rows = 0;
    std::int32_t capacity_rows = 0;
};

void gemma4_ring_copy_launch(const Gemma4RingCopyLaunch& launch, cudaStream_t stream);

} // namespace infernix::ops::detail
