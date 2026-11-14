#pragma once

// The state image of one Gemma4 sequence at a frontier F (P7 scope contract 1, HPC-min packing)
// and the Host record of one global KV block. Both layouts are computed from the per-layer facts
// (kv_layout.h), never from a model size:
//
//   image = header (256 B) | sliding layer 0: K part [, K scale part], V part [, V scale part] |
//           sliding layer 1 ... | sliding layer S-1
//
// One restore group per sliding layer (S groups). Every part is 256-byte aligned and sized
// align((W-1) * kv_heads * row_bytes, 256), the size prefix_cache_geometry sums: the image keeps
// the ring window [F-(W-1), F) and nothing else. Inside a part the window is cut into runs at the
// 64-token page boundaries of the logical positions (the physical ring page is the logical page %
// ring pages, so a run never straddles the ring's wrap); run r, whose first token is window
// index i0 and which holds n_r tokens, occupies kv_heads * n_r * row_bytes bytes at
// i0 * kv_heads * row_bytes laid out [head][token][row]. A run is therefore one strided copy, and
// consecutive full ring pages are one contiguous copy.
//
// The header carries the LayoutIdentity (layout_identity.h) and is validated before any restore.
// Host bytes of an image live in slabs that are not contiguous: byte o is at
// segments[o / segment_bytes] + o % segment_bytes.

#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/prefix/layout_identity.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace infernix::models::gemma4::prefix {

inline constexpr std::size_t kStateImageHeaderBytes = 256;
inline constexpr std::size_t kStateImageAlignment   = 256;
inline constexpr std::uint32_t kPageTokens          = static_cast<std::uint32_t>(ops::kGemma4KvPageTokens);

// ---- layouts ---------------------------------------------------------------------------------

// One plane of a global layer's page bundle inside a block record.
struct BlockPart {
    std::uint32_t layer = 0;  // ordinal among the global layers
    std::uint32_t plane = 0;  // index in KvLayerLayout::planes()
    std::size_t offset  = 0;  // in the record
    std::size_t bytes   = 0;  // the plane's bytes of one page
};

// The Host record of one global block (and of a snapshot's tail page): every global layer's
// page bundle, planes in KvLayerLayout::planes() order, layers in order.
struct BlockLayout {
    std::vector<BlockPart> parts;
    std::size_t record_bytes     = 0; // sum of the parts (the slab is this, 4 KiB aligned)
    std::uint32_t global_layers  = 0;
};

[[nodiscard]] BlockLayout plan_block_layout(std::span<const KvLayerFact> layers,
                                            KvCacheStorage global_format);

struct ImagePart {
    std::uint32_t group      = 0; // sliding layer ordinal
    std::uint32_t plane      = 0; // index in the layer's KvLayerLayout::planes()
    std::size_t offset       = 0; // in the image
    std::size_t bytes        = 0; // reserved (256-aligned)
    std::uint32_t kv_heads   = 0;
    std::size_t row_bytes    = 0; // one head's bytes of one token in this plane
    std::size_t page_bytes   = 0; // one ring page of this plane on the Device
};

struct StateImageLayout {
    std::vector<ImagePart> parts;
    std::vector<std::uint32_t> group_begin; // group g's parts are [group_begin[g], group_begin[g + 1])
    std::uint32_t window_tokens = 0;        // W - 1: the ring window an image keeps
    std::size_t image_bytes     = 0;        // header + parts

    [[nodiscard]] std::uint32_t groups() const noexcept {
        return group_begin.empty() ? 0U : static_cast<std::uint32_t>(group_begin.size()) - 1U;
    }
};

[[nodiscard]] StateImageLayout plan_state_image(std::span<const KvLayerFact> layers,
                                                std::uint32_t window, KvCacheStorage global_format);

// ---- the ring window -------------------------------------------------------------------------

// First logical position an image taken at `frontier` keeps: max(0, F - (W-1)).
[[nodiscard]] constexpr std::uint32_t image_window_start(std::uint32_t frontier,
                                                         std::uint32_t window_tokens) noexcept {
    return frontier > window_tokens ? frontier - window_tokens : 0U;
}

// A maximal run of window tokens inside one logical page.
struct WindowRun {
    std::uint32_t first         = 0; // window index of its first token (position - start)
    std::uint32_t tokens        = 0;
    std::uint32_t physical_page = 0; // logical page % ring_pages
    std::uint32_t first_row     = 0; // token row inside the page
};

// The runs covering logical positions [start, frontier); wrap-aware through the physical page.
[[nodiscard]] std::vector<WindowRun> window_runs(std::uint32_t start, std::uint32_t frontier,
                                                 std::uint32_t ring_pages);

// ---- header ----------------------------------------------------------------------------------

struct StateImageHeader {
    std::uint32_t frontier    = 0;
    std::uint32_t image_start = 0; // first logical position of the kept window
    bool opener               = false; // the snapshot is a generation-opener tap
    bool lineage_echo         = false; // the capturing request resumed from an endpoint
};

enum class HeaderStatus : std::uint8_t { Ok, Magic, Size, Version, Identity, Frontier };

struct HeaderCheck {
    HeaderStatus status = HeaderStatus::Magic;
    StateImageHeader header;
    [[nodiscard]] bool ok() const noexcept { return status == HeaderStatus::Ok; }
};

[[nodiscard]] const char* header_status_name(HeaderStatus status) noexcept;

// Writes the kStateImageHeaderBytes of `image_begin`: magic, identity block and fingerprint,
// frontier, window start, flags; the rest is zero.
void write_state_image_header(const LayoutIdentity& identity, const StateImageHeader& header,
                              std::byte* image_begin);
// Validates the header against the identity this process caches under, before any restore reads
// the image: magic, size, packing version, the full identity block and its fingerprint, and that
// the window start is the one the frontier implies.
[[nodiscard]] HeaderCheck read_state_image_header(const LayoutIdentity& expected,
                                                  const std::byte* image_begin);

// ---- Host placement of an image --------------------------------------------------------------

struct HostImage {
    std::span<std::byte* const> segments;
    std::size_t segment_bytes = 0;
    // The registered Host chunk each segment (slab) lies in (HostSlabStore::chunk_of). A copy never
    // spans two chunks, so pieces of different chunks are never joined; empty: one chunk.
    std::span<const std::uint32_t> chunks;
};

// ---- copies ----------------------------------------------------------------------------------

// One contiguous Device range paired with one contiguous Host range, the Host range wholly inside
// registered chunk `host_chunk` and the Device range inside one Device allocation `device_plane`
// (DevicePlane::allocation; separate allocations are never adjacent in the address space the copy
// may touch). Both default to 0: "the same".
struct CopyPiece {
    std::byte* device           = nullptr;
    std::byte* host             = nullptr;
    std::size_t bytes           = 0;
    std::uint32_t host_chunk    = 0;
    std::uint32_t device_plane  = 0;
};

// A copy as the driver sees it: height rows of `width` bytes, `height == 1` a plain copy.
struct CopyOp {
    std::byte* device           = nullptr;
    std::byte* host             = nullptr;
    std::size_t width           = 0;
    std::size_t height          = 1;
    std::size_t device_pitch    = 0;
    std::size_t host_pitch      = 0;
};

// Merges neighbouring pieces contiguous on both sides and in one Host chunk, then folds runs of
// equal-width pieces with constant strides and one Host chunk into one strided op. Order is
// preserved; no op spans two chunks. Pure.
[[nodiscard]] std::vector<CopyOp> plan_copy_ops(std::span<const CopyPiece> pieces);

// Enqueues `ops` on `stream` (`kind` is DeviceToHost or HostToDevice).
void submit_copy_ops(std::span<const CopyOp> ops, cudaMemcpyKind kind, cudaStream_t stream);

// Appends the host ranges of image bytes [offset, offset + bytes) paired with consecutive Device
// bytes from `device`, split at segment boundaries.
void append_image_pieces(const HostImage& image, std::size_t offset, std::size_t bytes,
                         std::byte* device, std::vector<CopyPiece>& out, std::uint32_t device_plane = 0);

// ---- Device side ----------------------------------------------------------------------------

// One plane of a layer's page bundle on the Device: the base of its pages and one page's bytes.
struct DevicePlane {
    std::byte* base          = nullptr;
    std::size_t page_bytes   = 0;
    // Identity of this plane's Device allocation, unique among the planes one cache plans copies
    // over (assigned by RingImage and PrefixCache; the caller's value is overwritten).
    std::uint32_t allocation = 0;
};

// Gathers and scatters the sliding rings' window planes of one frontier. `sliding[g]` is sliding
// layer g's planes (KvLayerLayout::planes() order) over `ring_pages` physical pages.
class RingImage {
public:
    RingImage(const StateImageLayout& layout, std::vector<std::vector<DevicePlane>> sliding,
              std::uint32_t ring_pages);

    [[nodiscard]] const StateImageLayout& layout() const noexcept { return *layout_; }
    [[nodiscard]] std::uint32_t ring_pages() const noexcept { return ring_pages_; }

    // Group g's window [start, frontier) to the image (DeviceToHost) or back (HostToDevice), in
    // stream order. The header is the caller's.
    void gather_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                      const HostImage& image, cudaStream_t stream) const;
    void scatter_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                       const HostImage& image, cudaStream_t stream) const;

private:
    void copy_group(std::uint32_t group, std::uint32_t start, std::uint32_t frontier,
                    const HostImage& image, cudaMemcpyKind kind, cudaStream_t stream) const;

    const StateImageLayout* layout_ = nullptr;
    std::vector<std::vector<DevicePlane>> sliding_;
    std::uint32_t ring_pages_ = 0;
};

} // namespace infernix::models::gemma4::prefix
