#pragma once

// Physical binding of the generic prefix index (runtime/prefix_cache) to Gemma4's contiguous KV
// buffers and pinned Host slab pool (P7 scope, docs/plans 20261111 §8.4). The index decides; this
// class holds what its ids name and moves the bytes its decisions imply.
//
// - There is no Device block pool. Gemma's global KV is per-layer contiguous buffers addressed
//   through an identity page table, so page b of the live sequence IS block b. A block's "Device
//   copy" in the index's sense is the sequence's own page (its device id is the page index): it
//   lives exactly as long as the sequence's pins, and every Device copy is evicted at release.
//   Blocks therefore persist only as Host slabs, written through from the sequence's pages.
// - Snapshots are Host-born: the ring window (S groups) and, when the frontier is not a block
//   boundary, the partial page are copied from the live sequence straight into reserved slabs.
// - Every copy runs on the Program's compute stream. Capture-before-overwrite, overwrite-vs-
//   pending-copy and restore-before-first-forward are stream order (no cross-stream hazard, so no
//   page-reuse ordering is needed); one event per batch retires it on the host (poll).

#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/prefix/layout_identity.h"
#include "models/gemma4/program/prefix/state_image.h"
#include "infernix/types.h"
#include "runtime/prefix_cache/cost.h"
#include "runtime/prefix_cache/host_slab_store.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace infernix::models::gemma4::prefix {

struct PrefixCacheCounters {
    std::uint64_t admissions                 = 0;
    std::uint64_t reused_tokens              = 0;
    std::uint64_t blocks_inserted            = 0;
    std::uint64_t blocks_duplicate           = 0;
    std::uint64_t taps_created               = 0;
    std::uint64_t taps_skipped               = 0;
    std::uint64_t endpoints_created          = 0;
    std::uint64_t host_image_writes          = 0;
    std::uint64_t host_block_writes          = 0;
    std::uint64_t host_block_write_refusals  = 0;
    std::uint64_t host_image_restores        = 0;
    std::uint64_t host_block_restores        = 0;
    std::uint64_t host_tail_restores         = 0;
    std::uint64_t host_write_bytes           = 0;
    std::uint64_t host_restore_bytes         = 0;
    std::uint64_t identity_refusals          = 0; // images whose header failed validation
};

// The Host tier: one slab per block record (4 KiB aligned, PrefixCacheGeometry::slab_bytes); an
// image spans image_slabs slabs and a snapshot's tail page one more.
struct PrefixHostLayout {
    std::size_t slab_bytes    = 0;
    std::size_t block_bytes   = 0;
    std::size_t image_bytes   = 0;
    std::uint32_t image_slabs = 0;
    std::uint32_t slabs       = 0;
};

// The refusal the Host tier owes a budget below the floor, its MiB computed from `geometry`.
[[nodiscard]] std::string host_floor_refusal(const PrefixCacheGeometry& geometry);

// Slabs from the byte budget; throws std::invalid_argument with host_floor_refusal's text when the
// budget holds fewer than image_slabs + 2 slabs (one live snapshot, one staging, one slab of
// headroom), and for a tier too large to index.
[[nodiscard]] PrefixHostLayout plan_prefix_host_layout(const PrefixCacheGeometry& geometry,
                                                       std::uint64_t host_bytes);

// The tap ladder retuned for the instance's image: every tap is one image DtoH, so the planner's
// defaults (tuned for ~192 MiB images) widen and thin with image_bytes. Pure.
[[nodiscard]] runtime::prefix_cache::TapPlannerConfig derive_tap_config(std::size_t image_bytes) noexcept;

// Exclusive ends of the prefill calls covering [frontier, prompt_tokens): `chunk`-token calls and
// one remainder call between consecutive cuts (sorted exact taps inside the range) and the end.
// Pure.
[[nodiscard]] std::vector<std::uint32_t> plan_call_ends(std::uint32_t frontier, std::uint32_t prompt_tokens,
                                                        std::uint32_t chunk,
                                                        std::span<const std::uint32_t> cuts);

struct PrefixCacheConfig {
    runtime::prefix_cache::CacheCostModel cost;
    runtime::prefix_cache::TapPlannerConfig taps;
    std::uint32_t kv_pages = 0; // global page groups of the sequence (upper bound of its Device copies)
};

struct RestoreResult {
    bool restored = false;
    // Why a given snapshot was refused (header validation, before any copy was enqueued).
    HeaderStatus refusal = HeaderStatus::Ok;
    bool refused_host    = false; // a block or image had no Resident Host copy
    bool image           = false;
    bool tail            = false;
};

class PrefixCache final : public runtime::prefix_cache::PrefixIndexBackend {
public:
    // `global[l]` is global layer l's planes and `sliding[g]` sliding layer g's (KvLayerLayout::
    // planes() order) over `ring_pages` physical ring pages. `stream` is the compute stream.
    PrefixCache(const PrefixCacheConfig& config, const PrefixCacheGeometry& geometry,
                const LayoutIdentity& identity, BlockLayout blocks, StateImageLayout image,
                std::vector<std::vector<DevicePlane>> global, std::vector<std::vector<DevicePlane>> sliding,
                std::uint32_t ring_pages, std::uint64_t host_bytes, cudaStream_t stream);
    ~PrefixCache() override;

    PrefixCache(const PrefixCache&)            = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    [[nodiscard]] runtime::prefix_cache::PrefixCacheIndex& index() noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::PrefixCacheIndex& index() const noexcept { return *index_; }
    [[nodiscard]] const runtime::prefix_cache::TapPlannerConfig& taps() const noexcept { return taps_; }
    [[nodiscard]] PrefixCacheCounters& counters() noexcept { return counters_; }
    [[nodiscard]] const PrefixCacheCounters& counters() const noexcept { return counters_; }
    [[nodiscard]] const PrefixHostLayout& host_layout() const noexcept { return host_; }
    [[nodiscard]] const StateImageLayout& image_layout() const noexcept { return image_; }
    [[nodiscard]] const BlockLayout& block_layout() const noexcept { return blocks_; }
    [[nodiscard]] const LayoutIdentity& identity() const noexcept { return identity_; }
    [[nodiscard]] std::uint64_t pinned_bytes() const noexcept { return store_.pinned_bytes(); }

    // ---- blocks ------------------------------------------------------------------------------
    // Inserts the sequence's committed full block whose page is `page` (its virtual device id).
    // A new node is Device Resident and pinned by the sequence (device_attached); an identical
    // existing node is only pinned and the page stays the sequence's private page (a duplicate).
    [[nodiscard]] runtime::prefix_cache::InsertResult
    insert_block(runtime::prefix_cache::NodeRef parent, std::uint64_t lookup_hash,
                 std::span<const TokenId> tokens, std::uint32_t page);
    // Writes the Device-only nodes of `nodes` (page pages[i]) through to Host slabs on the stream.
    void write_blocks(std::span<const runtime::prefix_cache::NodeRef> nodes, std::span<const std::uint32_t> pages);

    // ---- snapshot meta -----------------------------------------------------------------------
    [[nodiscard]] const StateImageHeader& meta(runtime::prefix_cache::SnapshotRef snapshot) const;
    // Whether the snapshot's Host image carries a header this process can restore (identity,
    // packing version and window start validated): the check a restore makes first.
    [[nodiscard]] bool image_valid(runtime::prefix_cache::SnapshotRef snapshot) const;

    // ---- Host-born captures ------------------------------------------------------------------
    struct Capture {
        std::uint64_t id = 0; // the caller's name for capture_result
        runtime::prefix_cache::NodeRef anchor;
        std::uint32_t frontier = 0;
        std::vector<TokenId> tail;       // the tokens of the partial page, empty at a block boundary
        std::uint32_t tail_page = 0;     // the sequence's page holding them
        runtime::prefix_cache::SnapshotKind kind = runtime::prefix_cache::SnapshotKind::Tap;
        double claim = 0.0;              // the new snapshot's estimate_priority
        StateImageHeader meta;
        std::optional<runtime::prefix_cache::SnapshotRef> supersedes;
        std::vector<TokenId> continuation;
    };
    // Enqueues the ring window of `frontier` (and the tail page) into reserved Host slabs behind
    // the sequence's work on the stream, and publishes the snapshot once the copies land (poll).
    // False when no slabs could be reserved for its claim (the capture is skipped).
    [[nodiscard]] bool capture(Capture&& capture);
    [[nodiscard]] std::optional<runtime::prefix_cache::PublishResult> capture_result(std::uint64_t id) const;
    void forget_capture(std::uint64_t id) noexcept;

    // ---- restore -----------------------------------------------------------------------------
    // Validates the snapshot's header and every Host copy first, then enqueues, in stream order,
    // the `path` blocks into the sequence's pages 0..k-1, the tail page into page k and the ring
    // window into the rings, pinning the sources until the batch lands. Nothing is enqueued when
    // the result is not restored.
    [[nodiscard]] RestoreResult restore(std::span<const runtime::prefix_cache::NodeRef> path,
                                        std::optional<runtime::prefix_cache::SnapshotRef> snapshot);

    // ---- bookkeeping -------------------------------------------------------------------------
    // Publishes landed captures, completes landed writes, retires landed restores and evicts the
    // Device copies nothing pins any more (their content is on the Host). Non-blocking.
    void poll();
    // Stream synchronize, then poll: every copy in flight has landed.
    void drain();
    [[nodiscard]] bool transfers_pending() const noexcept {
        return !captures_.empty() || !writes_.empty() || !landing_.empty();
    }
    // After the sequence released its pins: no Device copy outlives it. Backed copies go first;
    // an unbacked one (its write was refused) is lost with its subtree.
    void sweep_device_copies();
    // Releases every reference; the index is rebuilt empty (recovery after a failure).
    void clear() noexcept;

    // PrefixIndexBackend
    void release_device_block(std::uint32_t device_id) noexcept override;
    void release_snapshot(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;
    void drop_snapshot_device_image(runtime::prefix_cache::SnapshotRef snapshot) noexcept override;

private:
    struct PendingCapture {
        Capture capture;
        runtime::prefix_cache::HostImageReservation image;
        cudaEvent_t event = nullptr;
    };
    struct PendingWrite {
        std::vector<runtime::prefix_cache::NodeRef> nodes;
        cudaEvent_t event = nullptr;
    };
    struct RestoreBatch {
        std::vector<runtime::prefix_cache::NodeRef> pins;
        std::optional<runtime::prefix_cache::SnapshotRef> snapshot;
        cudaEvent_t event = nullptr;
    };

    [[nodiscard]] std::byte* slab(std::uint32_t id) const noexcept { return store_.slab(id); }
    // The registered chunk a slab lies in: one copy never spans two.
    [[nodiscard]] std::uint32_t slab_chunk(std::uint32_t id) const noexcept { return store_.chunk_of(id); }
    // An image's slabs as HostImage takes them: pointers and the chunk of each.
    struct ImageSegments {
        std::vector<std::byte*> pointers;
        std::vector<std::uint32_t> chunks;
        [[nodiscard]] HostImage view(std::size_t segment_bytes) const {
            return HostImage{pointers, segment_bytes, chunks};
        }
    };
    [[nodiscard]] ImageSegments image_segments(const std::uint32_t* slabs) const;
    [[nodiscard]] cudaEvent_t take_event();
    void recycle(cudaEvent_t event) noexcept;
    // Pieces moving page `pages[i]` of every global plane to/from the record in slab `slab_ids[i]`.
    void append_block_pieces(std::span<const std::uint32_t> pages, std::span<const std::uint32_t> slab_ids,
                             std::vector<CopyPiece>& out) const;

    runtime::prefix_cache::TapPlannerConfig taps_;
    LayoutIdentity identity_;
    BlockLayout blocks_;
    StateImageLayout image_;
    std::vector<std::vector<DevicePlane>> global_;
    PrefixHostLayout host_;
    runtime::prefix_cache::HostSlabStore store_;
    std::optional<RingImage> rings_;
    cudaStream_t stream_ = nullptr; // not owned
    std::vector<StateImageHeader> snapshot_meta_; // by snapshot index
    std::deque<PendingCapture> captures_;
    std::deque<PendingWrite> writes_;
    std::deque<RestoreBatch> landing_;
    std::unordered_map<std::uint64_t, runtime::prefix_cache::PublishResult> published_;
    std::vector<cudaEvent_t> spare_events_;
    PrefixCacheCounters counters_;
    std::optional<runtime::prefix_cache::PrefixCacheIndex> index_;
};

} // namespace infernix::models::gemma4::prefix
