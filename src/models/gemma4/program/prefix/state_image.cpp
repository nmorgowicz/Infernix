#include "models/gemma4/program/prefix/state_image.h"

#include "core/device.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace infernix::models::gemma4::prefix {
namespace {

constexpr std::uint64_t kHeaderMagic = 0x31474d494d334547ULL; // "GE3MIMG1" little-endian
// Header field offsets (all little-endian fixed-width integers).
constexpr std::size_t kMagicAt      = 0;   // u64
constexpr std::size_t kSizeAt       = 8;   // u32 header bytes
constexpr std::size_t kVersionAt    = 12;  // u32 packing version
constexpr std::size_t kPrintAt      = 16;  // u64 identity fingerprint
constexpr std::size_t kFrontierAt   = 24;  // u32
constexpr std::size_t kStartAt      = 28;  // u32 window start
constexpr std::size_t kFlagsAt      = 32;  // u32: bit 0 opener, bit 1 lineage echo
constexpr std::size_t kIdentityAt   = 40;  // 16 x u32, then 2 x u64
constexpr std::size_t kIdentityWords = 16;
constexpr std::size_t kIdentityBytesAt = kIdentityAt + kIdentityWords * sizeof(std::uint32_t);
static_assert(kIdentityBytesAt + 2 * sizeof(std::uint64_t) <= kStateImageHeaderBytes);

// A 2-D copy's row count the driver accepts comfortably.
constexpr std::size_t kMaxCopyHeight = 65535;

template <class T>
void put(std::byte* base, std::size_t offset, T value) {
    std::memcpy(base + offset, &value, sizeof(T));
}
template <class T>
T get(const std::byte* base, std::size_t offset) {
    T value;
    std::memcpy(&value, base + offset, sizeof(T));
    return value;
}

void put_identity(std::byte* base, const LayoutIdentity& id) {
    const std::uint32_t words[kIdentityWords] = {
        static_cast<std::uint32_t>(id.kv_format), static_cast<std::uint32_t>(id.code_dtype),
        id.scale_group_columns, id.scale_element_bytes, static_cast<std::uint32_t>(id.storage),
        static_cast<std::uint32_t>(id.qk_mode), id.global_layers, id.global_kv_heads,
        id.global_head_dim, id.sliding_layers, id.sliding_kv_heads, id.sliding_head_dim,
        id.window_tokens, id.prefill_chunk, id.packing_version, id.page_tokens};
    for (std::size_t i = 0; i < kIdentityWords; ++i) {
        put<std::uint32_t>(base, kIdentityAt + i * sizeof(std::uint32_t), words[i]);
    }
    put<std::uint64_t>(base, kIdentityBytesAt, id.block_bytes);
    put<std::uint64_t>(base, kIdentityBytesAt + sizeof(std::uint64_t), id.image_bytes);
}

LayoutIdentity get_identity(const std::byte* base) {
    std::uint32_t w[kIdentityWords];
    for (std::size_t i = 0; i < kIdentityWords; ++i) {
        w[i] = get<std::uint32_t>(base, kIdentityAt + i * sizeof(std::uint32_t));
    }
    LayoutIdentity id;
    id.kv_format = static_cast<ops::Gemma4KvFormat>(w[0]);
    id.code_dtype = static_cast<DType>(w[1]);
    id.scale_group_columns = w[2];
    id.scale_element_bytes = w[3];
    id.storage = static_cast<KvStorageModel>(w[4]);
    id.qk_mode = static_cast<ops::Gemma4QkMode>(w[5]);
    id.global_layers = w[6];
    id.global_kv_heads = w[7];
    id.global_head_dim = w[8];
    id.sliding_layers = w[9];
    id.sliding_kv_heads = w[10];
    id.sliding_head_dim = w[11];
    id.window_tokens = w[12];
    id.prefill_chunk = w[13];
    id.packing_version = w[14];
    id.page_tokens = w[15];
    id.block_bytes = get<std::uint64_t>(base, kIdentityBytesAt);
    id.image_bytes = get<std::uint64_t>(base, kIdentityBytesAt + sizeof(std::uint64_t));
    return id;
}

std::ptrdiff_t distance_between(const std::byte* a, const std::byte* b) noexcept {
    return static_cast<std::ptrdiff_t>(reinterpret_cast<std::intptr_t>(a) -
                                       reinterpret_cast<std::intptr_t>(b));
}

} // namespace

// ---- layouts ---------------------------------------------------------------------------------

BlockLayout plan_block_layout(std::span<const KvLayerFact> layers, KvCacheStorage global_format) {
    BlockLayout out;
    for (const auto& layer : layers) {
        if (!layer.global) { continue; }
        const auto layout = require_implemented_storage(
            kv_layer_layout(global_format, true, layer.head_dim, layer.kv_heads));
        const auto planes = layout.planes();
        for (std::uint32_t plane = 0; plane < layout.plane_count(); ++plane) {
            out.parts.push_back({out.global_layers, plane, out.record_bytes, planes[plane].page_bytes});
            out.record_bytes += planes[plane].page_bytes;
        }
        ++out.global_layers;
    }
    return out;
}

StateImageLayout plan_state_image(std::span<const KvLayerFact> layers, std::uint32_t window,
                                  KvCacheStorage global_format) {
    (void)global_format; // sliding rings are always BF16, whatever the global format
    StateImageLayout out;
    out.window_tokens = window > 0 ? window - 1U : 0U;
    std::size_t offset = kStateImageHeaderBytes;
    std::uint32_t group = 0;
    for (const auto& layer : layers) {
        if (layer.global) { continue; }
        const auto layout = require_implemented_storage(
            kv_layer_layout(KvCacheStorage::BFloat16, false, layer.head_dim, layer.kv_heads));
        const auto planes = layout.planes();
        out.group_begin.push_back(static_cast<std::uint32_t>(out.parts.size()));
        const std::size_t heads = static_cast<std::size_t>(layer.kv_heads);
        for (std::uint32_t plane = 0; plane < layout.plane_count(); ++plane) {
            ImagePart part;
            part.group = group;
            part.plane = plane;
            part.offset = offset;
            part.kv_heads = static_cast<std::uint32_t>(layer.kv_heads);
            part.page_bytes = planes[plane].page_bytes;
            part.row_bytes = planes[plane].page_bytes / (static_cast<std::size_t>(kPageTokens) * heads);
            part.bytes = align_bytes(static_cast<std::size_t>(out.window_tokens) * heads * part.row_bytes,
                                     kStateImageAlignment);
            offset += part.bytes;
            out.parts.push_back(part);
        }
        ++group;
    }
    out.group_begin.push_back(static_cast<std::uint32_t>(out.parts.size()));
    out.image_bytes = offset;
    return out;
}

// ---- the ring window -------------------------------------------------------------------------

std::vector<WindowRun> window_runs(std::uint32_t start, std::uint32_t frontier, std::uint32_t ring_pages) {
    if (ring_pages == 0) { throw std::invalid_argument("Gemma4 prefix cache: a ring needs pages"); }
    if (frontier < start) { throw std::invalid_argument("Gemma4 prefix cache: window ends before it starts"); }
    std::vector<WindowRun> runs;
    for (std::uint32_t position = start; position < frontier;) {
        const std::uint32_t page = position / kPageTokens;
        const std::uint32_t row = position % kPageTokens;
        const std::uint32_t take = std::min(kPageTokens - row, frontier - position);
        runs.push_back({position - start, take, page % ring_pages, row});
        position += take;
    }
    return runs;
}

// ---- header ----------------------------------------------------------------------------------

const char* header_status_name(HeaderStatus status) noexcept {
    switch (status) {
    case HeaderStatus::Ok: return "ok";
    case HeaderStatus::Magic: return "not a Gemma4 state image";
    case HeaderStatus::Size: return "header size differs";
    case HeaderStatus::Version: return "packing version differs";
    case HeaderStatus::Identity: return "layout identity differs";
    case HeaderStatus::Frontier: return "window start does not match the frontier";
    }
    return "unknown";
}

void write_state_image_header(const LayoutIdentity& identity, const StateImageHeader& header,
                              std::byte* image_begin) {
    std::memset(image_begin, 0, kStateImageHeaderBytes);
    put<std::uint64_t>(image_begin, kMagicAt, kHeaderMagic);
    put<std::uint32_t>(image_begin, kSizeAt, static_cast<std::uint32_t>(kStateImageHeaderBytes));
    put<std::uint32_t>(image_begin, kVersionAt, identity.packing_version);
    put<std::uint64_t>(image_begin, kPrintAt, identity.fingerprint());
    put<std::uint32_t>(image_begin, kFrontierAt, header.frontier);
    put<std::uint32_t>(image_begin, kStartAt, header.image_start);
    put<std::uint32_t>(image_begin, kFlagsAt,
                       (header.opener ? 1U : 0U) | (header.lineage_echo ? 2U : 0U));
    put_identity(image_begin, identity);
}

HeaderCheck read_state_image_header(const LayoutIdentity& expected, const std::byte* image_begin) {
    HeaderCheck out;
    if (get<std::uint64_t>(image_begin, kMagicAt) != kHeaderMagic) { out.status = HeaderStatus::Magic; return out; }
    if (get<std::uint32_t>(image_begin, kSizeAt) != kStateImageHeaderBytes) { out.status = HeaderStatus::Size; return out; }
    if (get<std::uint32_t>(image_begin, kVersionAt) != expected.packing_version) {
        out.status = HeaderStatus::Version;
        return out;
    }
    if (get_identity(image_begin) != expected || get<std::uint64_t>(image_begin, kPrintAt) != expected.fingerprint()) {
        out.status = HeaderStatus::Identity;
        return out;
    }
    out.header.frontier = get<std::uint32_t>(image_begin, kFrontierAt);
    out.header.image_start = get<std::uint32_t>(image_begin, kStartAt);
    const auto flags = get<std::uint32_t>(image_begin, kFlagsAt);
    out.header.opener = (flags & 1U) != 0;
    out.header.lineage_echo = (flags & 2U) != 0;
    const std::uint32_t kept = expected.window_tokens > 0 ? expected.window_tokens - 1U : 0U;
    if (out.header.image_start != image_window_start(out.header.frontier, kept)) {
        out.status = HeaderStatus::Frontier;
        return out;
    }
    out.status = HeaderStatus::Ok;
    return out;
}

// ---- copies ----------------------------------------------------------------------------------

std::vector<CopyOp> plan_copy_ops(std::span<const CopyPiece> pieces) {
    std::vector<CopyPiece> merged;
    merged.reserve(pieces.size());
    for (const CopyPiece& piece : pieces) {
        if (piece.bytes == 0) { continue; }
        if (!merged.empty() && merged.back().device + merged.back().bytes == piece.device &&
            merged.back().host + merged.back().bytes == piece.host && merged.back().host_chunk == piece.host_chunk &&
            merged.back().device_plane == piece.device_plane) {
            merged.back().bytes += piece.bytes;
        } else {
            merged.push_back(piece);
        }
    }
    std::vector<CopyOp> ops;
    for (std::size_t i = 0; i < merged.size();) {
        CopyOp op;
        op.device = merged[i].device;
        op.host = merged[i].host;
        op.width = merged[i].bytes;
        std::size_t j = i + 1;
        if (j < merged.size() && merged[j].bytes == op.width && merged[j].host_chunk == merged[i].host_chunk &&
            merged[j].device_plane == merged[i].device_plane) {
            const std::ptrdiff_t device_stride = distance_between(merged[j].device, merged[i].device);
            const std::ptrdiff_t host_stride = distance_between(merged[j].host, merged[i].host);
            const auto width = static_cast<std::ptrdiff_t>(op.width);
            if (device_stride >= width && host_stride >= width) {
                while (j < merged.size() && j - i < kMaxCopyHeight && merged[j].bytes == op.width &&
                       merged[j].host_chunk == merged[i].host_chunk &&
                       merged[j].device_plane == merged[i].device_plane &&
                       distance_between(merged[j].device, merged[j - 1].device) == device_stride &&
                       distance_between(merged[j].host, merged[j - 1].host) == host_stride) {
                    ++j;
                }
                op.height = j - i;
                op.device_pitch = static_cast<std::size_t>(device_stride);
                op.host_pitch = static_cast<std::size_t>(host_stride);
            }
        }
        if (op.height == 1) { j = i + 1; }
        ops.push_back(op);
        i = j;
    }
    return ops;
}

void submit_copy_ops(std::span<const CopyOp> ops, cudaMemcpyKind kind, cudaStream_t stream) {
    if (kind != cudaMemcpyDeviceToHost && kind != cudaMemcpyHostToDevice) {
        throw std::invalid_argument("Gemma4 prefix cache copies move between Device and Host only");
    }
    const bool save = kind == cudaMemcpyDeviceToHost;
    for (const CopyOp& op : ops) {
        if (op.height == 1) {
            CUDA_CHECK(cudaMemcpyAsync(save ? static_cast<void*>(op.host) : static_cast<void*>(op.device),
                                       save ? static_cast<const void*>(op.device) : static_cast<const void*>(op.host),
                                       op.width, kind, stream));
        } else {
            CUDA_CHECK(cudaMemcpy2DAsync(
                save ? static_cast<void*>(op.host) : static_cast<void*>(op.device),
                save ? op.host_pitch : op.device_pitch,
                save ? static_cast<const void*>(op.device) : static_cast<const void*>(op.host),
                save ? op.device_pitch : op.host_pitch, op.width, op.height, kind, stream));
        }
    }
}

void append_image_pieces(const HostImage& image, std::size_t offset, std::size_t bytes,
                         std::byte* device, std::vector<CopyPiece>& out, std::uint32_t device_plane) {
    if (image.segment_bytes == 0 || offset + bytes > image.segments.size() * image.segment_bytes) {
        throw std::out_of_range("Gemma4 prefix cache: an image range lies outside its slabs");
    }
    while (bytes != 0) {
        const std::size_t segment = offset / image.segment_bytes;
        const std::size_t inside = offset % image.segment_bytes;
        const std::size_t take = std::min(bytes, image.segment_bytes - inside);
        const std::uint32_t chunk = segment < image.chunks.size() ? image.chunks[segment] : 0U;
        out.push_back({device, image.segments[segment] + inside, take, chunk, device_plane});
        device += take;
        offset += take;
        bytes -= take;
    }
}

// ---- Device side ----------------------------------------------------------------------------

RingImage::RingImage(const StateImageLayout& layout, std::vector<std::vector<DevicePlane>> sliding,
                     std::uint32_t ring_pages)
    : layout_(&layout), sliding_(std::move(sliding)), ring_pages_(ring_pages) {
    if (ring_pages_ == 0 || sliding_.size() != layout.groups()) {
        throw std::invalid_argument("Gemma4 prefix cache: the rings do not match the image layout");
    }
    for (std::uint32_t group = 0; group < layout.groups(); ++group) {
        const std::uint32_t first = layout.group_begin[group], end = layout.group_begin[group + 1];
        if (sliding_[group].size() != end - first) {
            throw std::invalid_argument("Gemma4 prefix cache: a ring's planes do not match the image layout");
        }
        for (std::uint32_t i = first; i < end; ++i) {
            const ImagePart& part = layout.parts[i];
            if (sliding_[group][part.plane].base == nullptr ||
                sliding_[group][part.plane].page_bytes != part.page_bytes) {
                throw std::invalid_argument("Gemma4 prefix cache: a ring plane differs from its image part");
            }
        }
    }
    // Every plane is its own Device allocation: a copy never joins two.
    std::uint32_t allocation = 0;
    for (auto& planes : sliding_) {
        for (DevicePlane& plane : planes) { plane.allocation = allocation++; }
    }
}

void RingImage::gather_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                             const HostImage& image, cudaStream_t stream) const {
    copy_group(group, start, frontier, image, cudaMemcpyDeviceToHost, stream);
}

void RingImage::scatter_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                              const HostImage& image, cudaStream_t stream) const {
    copy_group(group, start, frontier, image, cudaMemcpyHostToDevice, stream);
}

void RingImage::copy_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                           const HostImage& image, cudaMemcpyKind kind, cudaStream_t stream) const {
    const StateImageLayout& layout = *layout_;
    if (group >= layout.groups() || frontier < start || frontier - start > layout.window_tokens) {
        throw std::invalid_argument("Gemma4 prefix cache: a ring copy lies outside the image window");
    }
    const std::vector<WindowRun> runs = window_runs(start, frontier, ring_pages_);
    std::vector<CopyPiece> pieces;
    for (std::uint32_t i = layout.group_begin[group]; i < layout.group_begin[group + 1]; ++i) {
        const ImagePart& part = layout.parts[i];
        const DevicePlane& plane = sliding_[group][part.plane];
        const std::size_t head_stride = static_cast<std::size_t>(kPageTokens) * part.row_bytes;
        pieces.clear();
        for (const WindowRun& run : runs) {
            const std::size_t run_bytes = static_cast<std::size_t>(run.tokens) * part.row_bytes;
            for (std::uint32_t head = 0; head < part.kv_heads; ++head) {
                std::byte* device = plane.base + static_cast<std::size_t>(run.physical_page) * plane.page_bytes +
                                    head * head_stride + static_cast<std::size_t>(run.first_row) * part.row_bytes;
                const std::size_t offset = part.offset +
                                           static_cast<std::size_t>(part.kv_heads) * run.first * part.row_bytes +
                                           head * run_bytes;
                append_image_pieces(image, offset, run_bytes, device, pieces, plane.allocation);
            }
        }
        const std::vector<CopyOp> ops = plan_copy_ops(pieces);
        submit_copy_ops(ops, kind, stream);
    }
}

} // namespace infernix::models::gemma4::prefix
