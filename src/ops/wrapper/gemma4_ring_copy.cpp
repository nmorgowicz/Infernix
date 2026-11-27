#include "infernix/ops/gemma4_ring_copy.h"

#include "ops/launcher/gemma4_ring_copy.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

[[noreturn]] void fail(const std::string& reason) {
    throw std::invalid_argument("gemma4_ring_copy: " + reason);
}

std::int32_t row_bytes(const Tensor& plane) {
    const std::size_t bytes = static_cast<std::size_t>(plane.ne[0]) * dtype_size(plane.dtype);
    return static_cast<std::int32_t>(bytes);
}

} // namespace

void gemma4_ring_copy(const Gemma4CacheView& src, const Gemma4CacheView& dst, const Tensor& positions,
                      Gemma4RingCopyRows rows, int tokens, int window_rows, int capacity_rows,
                      cudaStream_t stream) {
    if (!src.k_pages) { fail("source cache has no K pages"); }
    const int head_dim = src.k_pages->ne[0];
    detail::validate_gemma4_cache(src, head_dim, false, "gemma4_ring_copy");
    detail::validate_gemma4_cache(dst, head_dim, false, "gemma4_ring_copy");
    if (src.value_source != Gemma4ValueSource::Stored || dst.value_source != Gemma4ValueSource::Stored) {
        fail("sliding rings store both K and V");
    }
    if (src.format != dst.format) { fail("source and destination formats differ"); }
    if (src.k_pages->ne[2] != dst.k_pages->ne[2]) { fail("source and destination KV head counts differ"); }
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || positions.data == nullptr ||
        positions.ne[0] < 1) {
        fail("positions must be contiguous non-null I32");
    }
    if (tokens <= 0 || window_rows < 0 || capacity_rows <= 0) { fail("invalid row counts"); }
    const bool window = rows == Gemma4RingCopyRows::Window;
    if (window && window_rows == 0) { return; }
    detail::Gemma4RingCopyLaunch launch;
    launch.planes.planes = src.format == Gemma4KvFormat::Fp8E4m3Row256 ? 4 : 2;
    const Tensor* src_planes[4] = {src.k_pages, src.v_pages, src.k_scales, src.v_scales};
    const Tensor* dst_planes[4] = {dst.k_pages, dst.v_pages, dst.k_scales, dst.v_scales};
    for (int i = 0; i < launch.planes.planes; ++i) {
        if (!src_planes[i] || !dst_planes[i]) { fail("missing plane"); }
        if (src_planes[i]->ne[0] != dst_planes[i]->ne[0] || src_planes[i]->dtype != dst_planes[i]->dtype) {
            fail("plane row geometry differs");
        }
        launch.planes.src[i] = static_cast<const std::byte*>(src_planes[i]->data);
        launch.planes.dst[i] = static_cast<std::byte*>(dst_planes[i]->data);
        launch.planes.row_bytes[i] = row_bytes(*src_planes[i]);
    }
    const Tensor& src_table = *src.block_table;
    const Tensor& dst_table = *dst.block_table;
    if (src_table.dtype != DType::I32 || dst_table.dtype != DType::I32) { fail("block tables must be I32"); }
    launch.src_table = static_cast<const std::int32_t*>(src_table.data);
    launch.dst_table = static_cast<const std::int32_t*>(dst_table.data);
    launch.table_pages = std::min(src_table.ne[0], dst_table.ne[0]);
    launch.src_pages = src.k_pages->ne[3];
    launch.dst_pages = dst.k_pages->ne[3];
    launch.kv_heads = src.k_pages->ne[2];
    launch.positions = static_cast<const std::int32_t*>(positions.data);
    launch.window = window;
    launch.tokens = tokens;
    launch.window_rows = window_rows;
    launch.capacity_rows = capacity_rows;
    detail::gemma4_ring_copy_launch(launch, stream);
}

} // namespace infernix::ops
