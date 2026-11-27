#include "ops/launcher/gemma4_ring_copy.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <algorithm>

namespace infernix::ops::detail {
namespace {

constexpr int kThreads = 256;

// One block copies one (logical page, KV head, plane) span: the rows of that page inside [lo, hi) are
// contiguous bytes in every plane (layout [D,64,Hkv,pages]: D fastest, then the 64 rows, head, page).
__global__ void gemma4_ring_copy_kernel(Gemma4RingCopyLaunch launch) {
    const std::int32_t first = launch.positions[0];
    std::int32_t lo = 0;
    std::int32_t hi = 0;
    if (launch.window) {
        hi = first;
        lo = max(0, first - launch.window_rows);
    } else {
        hi = first + launch.tokens;
        lo = max(first, hi - launch.capacity_rows);
    }
    if (first < 0 || lo >= hi) { return; }
    const std::int32_t page = (lo >> kPagedKVPageShift) + static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t page_first = page << kPagedKVPageShift;
    if (page_first >= hi || page >= launch.table_pages) { return; }
    const std::int32_t src_page = launch.src_table[page];
    const std::int32_t dst_page = launch.dst_table[page];
    if (src_page < 0 || src_page >= launch.src_pages || dst_page < 0 || dst_page >= launch.dst_pages) { return; }
    const std::int32_t row_begin = max(lo, page_first) - page_first;
    const std::int32_t row_end = min(hi, page_first + kPagedKVPageSize) - page_first;
    const std::int32_t plane = static_cast<std::int32_t>(blockIdx.z);
    const std::int64_t row_bytes = launch.planes.row_bytes[plane];
    const std::int64_t head = blockIdx.y;
    const std::int64_t heads = launch.kv_heads;
    const std::int64_t src_base = row_bytes * (kPagedKVPageSize * (head + heads * src_page)) + row_bytes * row_begin;
    const std::int64_t dst_base = row_bytes * (kPagedKVPageSize * (head + heads * dst_page)) + row_bytes * row_begin;
    const std::int64_t bytes = row_bytes * (row_end - row_begin);
    const std::byte* src = launch.planes.src[plane] + src_base;
    std::byte* dst = launch.planes.dst[plane] + dst_base;
    if (row_bytes % 16 == 0) {
        const auto* s = reinterpret_cast<const uint4*>(src);
        auto* d = reinterpret_cast<uint4*>(dst);
        for (std::int64_t i = threadIdx.x; i < bytes / 16; i += blockDim.x) { d[i] = s[i]; }
    } else {
        // FP16 scale rows: 2 bytes per row and head.
        const auto* s = reinterpret_cast<const unsigned short*>(src);
        auto* d = reinterpret_cast<unsigned short*>(dst);
        for (std::int64_t i = threadIdx.x; i < bytes / 2; i += blockDim.x) { d[i] = s[i]; }
    }
}

} // namespace

void gemma4_ring_copy_launch(const Gemma4RingCopyLaunch& launch, cudaStream_t stream) {
    const std::int32_t rows = launch.window ? launch.window_rows : std::min(launch.tokens, launch.capacity_rows);
    // A span of `rows` rows touches at most ceil(rows / 64) + 1 pages.
    const std::int32_t pages = (rows + kPagedKVPageSize - 1) / kPagedKVPageSize + 1;
    const dim3 grid(static_cast<unsigned>(pages), static_cast<unsigned>(launch.kv_heads),
                    static_cast<unsigned>(launch.planes.planes));
    gemma4_ring_copy_kernel<<<grid, kThreads, 0, stream>>>(launch);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
