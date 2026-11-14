#include "models/gemma4/program/prefix/prefix_cache.h"

#include "core/device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::models::gemma4::prefix {

namespace pc = runtime::prefix_cache;

namespace {

// The image size the default tap ladder (TapPlannerConfig) was tuned for.
constexpr double kReferenceImageBytes = 192.0 * 1048576.0;
// The most DtoH a prompt's new taps may add, in reference images.
constexpr double kTapImageBudget = 16.0;
constexpr std::uint32_t kMinimumNewTaps = 3;

} // namespace

std::string host_floor_refusal(const PrefixCacheGeometry& geometry) {
    const std::uint64_t mib = (geometry.host_floor_bytes + (1ULL << 20) - 1) >> 20;
    return "--host-context-mib is too small for the Gemma4 prefix cache: one ring snapshot needs " +
           std::to_string(mib) + " MiB (" + std::to_string(geometry.image_slabs) + "+2 slabs of " +
           std::to_string((geometry.slab_bytes + 1023) >> 10) + " KiB from this artifact's geometry)";
}

PrefixHostLayout plan_prefix_host_layout(const PrefixCacheGeometry& geometry, std::uint64_t host_bytes) {
    if (geometry.slab_bytes == 0 || geometry.image_slabs == 0) {
        throw std::invalid_argument("Gemma4 prefix cache: the cache geometry is empty");
    }
    PrefixHostLayout out;
    out.slab_bytes  = geometry.slab_bytes;
    out.block_bytes = geometry.block_bytes;
    out.image_bytes = geometry.image_bytes;
    out.image_slabs = geometry.image_slabs;
    const std::uint64_t slabs = host_bytes / geometry.slab_bytes;
    if (slabs > pc::kNoId / 4U) { throw std::invalid_argument("Gemma4 prefix cache: the Host tier is too large"); }
    if (slabs < static_cast<std::uint64_t>(geometry.image_slabs) + 2U) {
        throw std::invalid_argument(host_floor_refusal(geometry));
    }
    out.slabs = static_cast<std::uint32_t>(slabs);
    return out;
}

pc::TapPlannerConfig derive_tap_config(std::size_t image_bytes) noexcept {
    const double scale = std::max(1.0, static_cast<double>(image_bytes) / kReferenceImageBytes);
    pc::TapPlannerConfig defaults;
    pc::TapPlannerConfig out;
    const double taps = std::floor(kTapImageBudget / scale);
    out.max_new_taps = static_cast<std::uint32_t>(
        std::clamp(taps, static_cast<double>(kMinimumNewTaps), static_cast<double>(defaults.max_new_taps)));
    const auto widen = static_cast<std::uint32_t>(std::ceil(scale));
    out.ladder_tokens  = defaults.ladder_tokens * widen;
    out.min_gap_tokens = defaults.min_gap_tokens * widen;
    return out;
}

std::vector<std::uint32_t> plan_call_ends(std::uint32_t frontier, std::uint32_t prompt_tokens,
                                          std::uint32_t chunk, std::span<const std::uint32_t> cuts) {
    if (chunk == 0) { throw std::invalid_argument("Gemma4 prefix cache: a prefill chunk is empty"); }
    std::vector<std::uint32_t> ends;
    std::uint32_t begin = frontier;
    const auto segment = [&](std::uint32_t from, std::uint32_t to) {
        for (std::uint32_t position = from; position < to;) {
            position = std::min(to, position + chunk);
            ends.push_back(position);
        }
    };
    for (const std::uint32_t cut : cuts) {
        if (cut <= begin || cut >= prompt_tokens) { continue; }
        segment(begin, cut);
        begin = cut;
    }
    segment(begin, prompt_tokens);
    return ends;
}

// ---- construction ----------------------------------------------------------------------------

PrefixCache::PrefixCache(const PrefixCacheConfig& config, const PrefixCacheGeometry& geometry,
                         const LayoutIdentity& identity, BlockLayout blocks, StateImageLayout image,
                         std::vector<std::vector<DevicePlane>> global,
                         std::vector<std::vector<DevicePlane>> sliding, std::uint32_t ring_pages,
                         std::uint64_t host_bytes, cudaStream_t stream)
    : taps_(config.taps), identity_(identity), blocks_(std::move(blocks)), image_(std::move(image)),
      global_(std::move(global)), host_(plan_prefix_host_layout(geometry, host_bytes)), stream_(stream) {
    // The layouts must be the ones the geometry (and the boot refusal) sums.
    if (image_.image_bytes != geometry.image_bytes || blocks_.record_bytes > geometry.block_bytes ||
        identity_.block_bytes != geometry.block_bytes || identity_.image_bytes != geometry.image_bytes) {
        throw std::logic_error("Gemma4 prefix cache: the state image or block layout disagrees with its geometry");
    }
    if (global_.size() != blocks_.global_layers) {
        throw std::invalid_argument("Gemma4 prefix cache: the global layers do not match the block layout");
    }
    for (const BlockPart& part : blocks_.parts) {
        if (global_[part.layer].size() <= part.plane || global_[part.layer][part.plane].base == nullptr ||
            global_[part.layer][part.plane].page_bytes != part.bytes) {
            throw std::invalid_argument("Gemma4 prefix cache: a global plane differs from its block part");
        }
    }
    // Every plane of every global layer is its own Device allocation: no copy joins two.
    std::uint32_t allocation = 0;
    for (auto& planes : global_) {
        for (DevicePlane& plane : planes) { plane.allocation = allocation++; }
    }
    pc::PrefixIndexConfig index;
    index.cost                  = config.cost;
    index.max_nodes             = config.kv_pages + host_.slabs + 1U;
    index.max_snapshots         = host_.slabs / host_.image_slabs + 1U;
    index.host_slabs            = host_.slabs;
    index.image_slabs           = host_.image_slabs;
    index.device_snapshot_slots = 0;
    index.block_bytes           = host_.block_bytes;
    index.image_bytes           = host_.image_bytes;
    index.host_blocks           = true;
    store_ = pc::HostSlabStore(host_.slab_bytes, host_.slabs, pc::HostSlabStore::kChunkBytes,
                               "Gemma4 prefix cache Host tier");
    snapshot_meta_.assign(index.max_snapshots, StateImageHeader{});
    rings_.emplace(image_, std::move(sliding), ring_pages);
    index_.emplace(index, *this);
}

PrefixCache::~PrefixCache() {
    // Copies in flight still read or write slabs and Device pages: they land first.
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    for (PendingCapture& c : captures_) {
        if (c.event != nullptr) { (void)cudaEventDestroy(c.event); }
    }
    for (PendingWrite& w : writes_) {
        if (w.event != nullptr) { (void)cudaEventDestroy(w.event); }
    }
    for (RestoreBatch& batch : landing_) {
        if (batch.event != nullptr) { (void)cudaEventDestroy(batch.event); }
    }
    for (cudaEvent_t event : spare_events_) { (void)cudaEventDestroy(event); }
    index_.reset();
}

cudaEvent_t PrefixCache::take_event() {
    if (!spare_events_.empty()) {
        cudaEvent_t event = spare_events_.back();
        spare_events_.pop_back();
        return event;
    }
    cudaEvent_t event = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

void PrefixCache::recycle(cudaEvent_t event) noexcept {
    if (event != nullptr) { spare_events_.push_back(event); }
}

PrefixCache::ImageSegments PrefixCache::image_segments(const std::uint32_t* slabs) const {
    ImageSegments segments;
    segments.pointers.reserve(host_.image_slabs);
    segments.chunks.reserve(host_.image_slabs);
    for (std::uint32_t i = 0; i < host_.image_slabs; ++i) {
        segments.pointers.push_back(slab(slabs[i]));
        segments.chunks.push_back(slab_chunk(slabs[i]));
    }
    return segments;
}

void PrefixCache::append_block_pieces(std::span<const std::uint32_t> pages,
                                      std::span<const std::uint32_t> slab_ids, std::vector<CopyPiece>& out) const {
    // Plane-major, block-minor: consecutive pages of one plane are one contiguous Device range and
    // consecutive slabs of one chunk one constant stride, so plan_copy_ops folds a run of blocks into
    // one copy. A record lies wholly inside its slab, and a slab inside one registered chunk.
    for (const BlockPart& part : blocks_.parts) {
        const DevicePlane& plane = global_[part.layer][part.plane];
        for (std::size_t i = 0; i < pages.size(); ++i) {
            out.push_back({plane.base + static_cast<std::size_t>(pages[i]) * plane.page_bytes,
                           slab(slab_ids[i]) + part.offset, part.bytes, slab_chunk(slab_ids[i]), plane.allocation});
        }
    }
}

// ---- blocks ----------------------------------------------------------------------------------

pc::InsertResult PrefixCache::insert_block(pc::NodeRef parent, std::uint64_t lookup_hash,
                                           std::span<const TokenId> tokens, std::uint32_t page) {
    if (tokens.size() != pc::kBlockTokens) {
        throw std::invalid_argument("Gemma4 prefix cache: a block is one full page of tokens");
    }
    // attach = false: an existing Host twin keeps its bytes (no write, no replacement); the
    // sequence's page stays its own.
    const pc::InsertResult result = index_->insert_block(parent, lookup_hash, tokens, 0, page, false);
    ++(result.inserted ? counters_.blocks_inserted : counters_.blocks_duplicate);
    return result;
}

void PrefixCache::write_blocks(std::span<const pc::NodeRef> nodes, std::span<const std::uint32_t> pages) {
    if (nodes.size() != pages.size()) {
        throw std::invalid_argument("Gemma4 prefix cache: every written block needs its page");
    }
    std::vector<pc::NodeRef> written;
    std::vector<std::uint32_t> written_pages;
    std::vector<std::uint32_t> slab_ids;
    const auto abandon = [&]() noexcept {
        for (const pc::NodeRef node : written) {
            index_->abort_host_fill(node);
            index_->unpin_node(node);
        }
    };
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (!index_->valid(nodes[i])) { continue; }
        const pc::NodeView view = index_->node(nodes[i]);
        if (view.device != pc::CopyState::Resident || view.host != pc::CopyState::Absent) { continue; }
        index_->pin_node(nodes[i]);
        const auto slab_id = index_->begin_host_fill(nodes[i]);
        if (!slab_id) {
            index_->unpin_node(nodes[i]);
            ++counters_.host_block_write_refusals;
            break; // the Host tier cannot free a slab for anything worth less
        }
        written.push_back(nodes[i]);
        written_pages.push_back(pages[i]);
        slab_ids.push_back(*slab_id);
    }
    if (written.empty()) { return; }
    cudaEvent_t event = nullptr;
    try {
        std::vector<CopyPiece> pieces;
        append_block_pieces(written_pages, slab_ids, pieces);
        submit_copy_ops(plan_copy_ops(pieces), cudaMemcpyDeviceToHost, stream_);
        event = take_event();
        CUDA_CHECK(cudaEventRecord(event, stream_));
    } catch (...) {
        recycle(event);
        abandon();
        throw;
    }
    counters_.host_block_writes += written.size();
    counters_.host_write_bytes += written.size() * blocks_.record_bytes;
    writes_.push_back(PendingWrite{std::move(written), event});
}

// ---- snapshots -------------------------------------------------------------------------------

const StateImageHeader& PrefixCache::meta(pc::SnapshotRef snapshot) const {
    if (!index_->valid(snapshot)) { throw std::logic_error("Gemma4 prefix cache: stale snapshot"); }
    return snapshot_meta_[snapshot.index];
}

bool PrefixCache::image_valid(pc::SnapshotRef snapshot) const {
    if (!index_->valid(snapshot)) { return false; }
    const pc::SnapshotView view = index_->snapshot(snapshot);
    if (view.host != pc::CopyState::Resident || view.host_slabs.size() < host_.image_slabs) { return false; }
    const HeaderCheck check = read_state_image_header(identity_, slab(view.host_slabs[0]));
    return check.ok() && check.header.frontier == view.frontier;
}

bool PrefixCache::capture(Capture&& capture) {
    const bool tail = !capture.tail.empty();
    if (capture.frontier == 0 || capture.tail.size() >= pc::kBlockTokens ||
        capture.frontier % pc::kBlockTokens != capture.tail.size()) {
        throw std::invalid_argument("Gemma4 prefix cache: a capture's tail does not match its frontier");
    }
    auto reservation = index_->reserve_host_image(tail, capture.claim);
    if (!reservation) {
        ++counters_.taps_skipped;
        return false;
    }
    // The anchor must outlive the copy: the capturing sequence may release its path before it lands.
    if (capture.anchor.valid()) { index_->pin_node(capture.anchor); }
    capture.meta.frontier    = capture.frontier;
    capture.meta.image_start = image_window_start(capture.frontier, image_.window_tokens);
    cudaEvent_t event        = nullptr;
    try {
        const ImageSegments segments = image_segments(reservation->slabs.data());
        // The header goes first and is the identity a restore validates before reading a byte.
        write_state_image_header(identity_, capture.meta, segments.pointers.front());
        const HostImage host = segments.view(host_.slab_bytes);
        for (std::uint32_t group = 0; group < image_.groups(); ++group) {
            rings_->gather_group(group, capture.meta.image_start, capture.frontier, host, stream_);
        }
        if (tail) {
            const std::uint32_t pages[1] = {capture.tail_page};
            const std::uint32_t tail_slab[1] = {reservation->slabs.back()};
            std::vector<CopyPiece> pieces;
            append_block_pieces(pages, tail_slab, pieces);
            submit_copy_ops(plan_copy_ops(pieces), cudaMemcpyDeviceToHost, stream_);
        }
        event = take_event();
        CUDA_CHECK(cudaEventRecord(event, stream_));
    } catch (...) {
        recycle(event);
        if (capture.anchor.valid()) { index_->unpin_node(capture.anchor); }
        index_->release_host_image(std::move(*reservation));
        throw;
    }
    counters_.host_write_bytes += host_.image_bytes + (tail ? blocks_.record_bytes : 0U);
    ++counters_.host_image_writes;
    captures_.push_back(PendingCapture{std::move(capture), std::move(*reservation), event});
    return true;
}

std::optional<pc::PublishResult> PrefixCache::capture_result(std::uint64_t id) const {
    const auto found = published_.find(id);
    if (found == published_.end()) { return std::nullopt; }
    return found->second;
}

void PrefixCache::forget_capture(std::uint64_t id) noexcept { published_.erase(id); }

// ---- restore ---------------------------------------------------------------------------------

RestoreResult PrefixCache::restore(std::span<const pc::NodeRef> path, std::optional<pc::SnapshotRef> snapshot) {
    RestoreResult result;
    // Everything is checked before the first byte is enqueued or a pin taken.
    std::vector<std::uint32_t> pages;
    std::vector<std::uint32_t> slab_ids;
    pages.reserve(path.size());
    slab_ids.reserve(path.size());
    for (std::size_t j = 0; j < path.size(); ++j) {
        if (!index_->valid(path[j])) { result.refused_host = true; return result; }
        const pc::NodeView view = index_->node(path[j]);
        if (view.host != pc::CopyState::Resident) { result.refused_host = true; return result; }
        pages.push_back(static_cast<std::uint32_t>(j));
        slab_ids.push_back(view.host_slab);
    }
    pc::SnapshotView view{};
    ImageSegments segments;
    std::uint32_t start = 0;
    if (snapshot) {
        if (!index_->valid(*snapshot)) { result.refused_host = true; return result; }
        view = index_->snapshot(*snapshot);
        const std::size_t slabs = host_.image_slabs + (view.tail_len != 0 ? 1U : 0U);
        if (view.host != pc::CopyState::Resident || view.host_slabs.size() < slabs) {
            result.refused_host = true;
            return result;
        }
        segments = image_segments(view.host_slabs.data());
        const HeaderCheck check = read_state_image_header(identity_, segments.pointers.front());
        if (!check.ok()) {
            ++counters_.identity_refusals;
            result.refusal = check.status;
            return result;
        }
        if (check.header.frontier != view.frontier || view.frontier / pc::kBlockTokens != path.size() ||
            view.tail_len != view.frontier % pc::kBlockTokens) {
            ++counters_.identity_refusals;
            result.refusal = HeaderStatus::Frontier;
            return result;
        }
        start = check.header.image_start;
    }
    RestoreBatch batch;
    cudaEvent_t event = nullptr;
    const auto unpin_all = [&]() noexcept {
        for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
        if (batch.snapshot) { index_->unpin_snapshot(*batch.snapshot); }
        batch.pins.clear();
        batch.snapshot.reset();
    };
    try {
        for (const pc::NodeRef node : path) {
            index_->pin_node(node);
            batch.pins.push_back(node);
        }
        if (snapshot) {
            index_->pin_snapshot(*snapshot);
            batch.snapshot = snapshot;
        }
        std::vector<CopyPiece> pieces;
        append_block_pieces(pages, slab_ids, pieces);
        if (snapshot && view.tail_len != 0) {
            const std::uint32_t tail_page[1] = {static_cast<std::uint32_t>(path.size())};
            const std::uint32_t tail_slab[1] = {view.host_slabs.back()};
            append_block_pieces(tail_page, tail_slab, pieces);
            result.tail = true;
        }
        submit_copy_ops(plan_copy_ops(pieces), cudaMemcpyHostToDevice, stream_);
        if (snapshot) {
            const HostImage host = segments.view(host_.slab_bytes);
            for (std::uint32_t group = 0; group < image_.groups(); ++group) {
                rings_->scatter_group(group, start, view.frontier, host, stream_);
            }
            result.image = true;
        }
        event = take_event();
        CUDA_CHECK(cudaEventRecord(event, stream_));
    } catch (...) {
        recycle(event);
        unpin_all();
        throw;
    }
    batch.event = event;
    counters_.host_block_restores += path.size();
    counters_.host_restore_bytes += path.size() * blocks_.record_bytes + (result.tail ? blocks_.record_bytes : 0U) +
                                    (result.image ? host_.image_bytes : 0U);
    if (result.tail) { ++counters_.host_tail_restores; }
    if (result.image) { ++counters_.host_image_restores; }
    landing_.push_back(std::move(batch));
    result.restored = true;
    return result;
}

// ---- bookkeeping -----------------------------------------------------------------------------

void PrefixCache::poll() {
    while (!captures_.empty()) {
        PendingCapture& c = captures_.front();
        const cudaError_t status = cudaEventQuery(c.event);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        const pc::PublishResult result = index_->publish_host_snapshot(
            c.capture.anchor, c.capture.frontier, c.capture.tail, std::nullopt, std::move(c.image), c.capture.kind);
        if (result.created) {
            snapshot_meta_[result.snapshot.index] = c.capture.meta;
            ++(c.capture.kind == pc::SnapshotKind::Endpoint ? counters_.endpoints_created : counters_.taps_created);
        }
        if (c.capture.supersedes && index_->valid(*c.capture.supersedes) &&
            (!result.snapshot.valid() || *c.capture.supersedes != result.snapshot)) {
            index_->supersede(*c.capture.supersedes, c.capture.continuation);
        }
        if (c.capture.anchor.valid()) { index_->unpin_node(c.capture.anchor); }
        published_[c.capture.id] = result;
        recycle(c.event);
        captures_.pop_front();
    }
    while (!writes_.empty()) {
        PendingWrite& w = writes_.front();
        const cudaError_t status = cudaEventQuery(w.event);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        for (const pc::NodeRef node : w.nodes) {
            index_->complete_host_fill(node);
            index_->unpin_node(node);
        }
        recycle(w.event);
        writes_.pop_front();
    }
    while (!landing_.empty()) {
        RestoreBatch& batch = landing_.front();
        const cudaError_t status = cudaEventQuery(batch.event);
        if (status == cudaErrorNotReady) { break; }
        CUDA_CHECK(status);
        for (const pc::NodeRef node : batch.pins) { index_->unpin_node(node); }
        if (batch.snapshot) { index_->unpin_snapshot(*batch.snapshot); }
        recycle(batch.event);
        landing_.pop_front();
    }
    // A written block whose sequence has gone holds no Device copy worth keeping: its page is
    // gone with the sequence and its bytes are on the Host.
    (void)index_->evict_backed_device_blocks(std::numeric_limits<std::uint32_t>::max());
}

void PrefixCache::drain() {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    poll();
}

void PrefixCache::sweep_device_copies() {
    (void)index_->evict_backed_device_blocks(std::numeric_limits<std::uint32_t>::max());
    (void)index_->evict_device_blocks(std::numeric_limits<std::uint32_t>::max());
}

void PrefixCache::clear() noexcept {
    try {
        drain();
    } catch (...) {}
    for (PendingCapture& c : captures_) { recycle(c.event); }
    for (PendingWrite& w : writes_) { recycle(w.event); }
    for (RestoreBatch& batch : landing_) { recycle(batch.event); }
    captures_.clear();
    writes_.clear();
    landing_.clear();
    published_.clear();
    const pc::PrefixIndexConfig config = index_->config();
    index_.reset();
    std::fill(snapshot_meta_.begin(), snapshot_meta_.end(), StateImageHeader{});
    index_.emplace(config, *this);
}

// ---- PrefixIndexBackend ----------------------------------------------------------------------

void PrefixCache::release_device_block(std::uint32_t) noexcept {
    // A Device copy is the sequence's own page (its device id is the page index): there is no
    // lease to return, and the page is rewritten by the next sequence in stream order.
}

void PrefixCache::release_snapshot(pc::SnapshotRef snapshot) noexcept {
    if (snapshot.index < snapshot_meta_.size()) { snapshot_meta_[snapshot.index] = StateImageHeader{}; }
}

void PrefixCache::drop_snapshot_device_image(pc::SnapshotRef) noexcept {
    // Snapshots are Host-born; there are no Device images to drop.
}

} // namespace infernix::models::gemma4::prefix
