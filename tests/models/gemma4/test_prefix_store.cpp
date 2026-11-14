// P7 prefix-cache store mechanics on the pure helpers of prefix/state_image.h, prefix_cache.h and
// layout_identity.h, over two synthetic descriptors: 31B-like (10 global x 4 KV heads x D512 FP8; 50
// sliding x 16 x D256, W=1024) and 26B-A4B-like (5 x 2 x D512 FP8; sliding Hs=8). CPU only: no GPU,
// no artifact, no model.
//
// The Device side is emulated on host memory: rings and block pages are byte vectors laid out as the
// kernels address them ([head][64 tokens][row] per page), and a copy is the planned CopyOp list
// replayed with memcpy. RingImage and PrefixCache themselves need the CUDA driver (RingImage enqueues
// cudaMemcpyAsync, the Host tier is page-locked by HostSlabStore, restore/capture record events), so
// the in-memory PrefixCache scenarios (capture -> publish -> restore, withdrawal, pins) are covered
// by the real tests (test_prefix_scenarios_real.cpp, test_prefix_recovery_real.cpp) and the generic
// index by tests/runtime/test_prefix_cache_index.cpp. What this file checks:
//   * image/block layouts equal prefix_cache_geometry's bytes;
//   * header round trip, foreign identity refused in both directions (and a value-only flag), every
//     ignored header byte is padding;
//   * window_runs across a ring wrap and for a frontier below the window;
//   * gather/scatter round trip over the ring emulation (frontiers around 1/63/64/65 and the wrap)
//     against an oracle built from the documented layout, plus the committed window surviving the
//     unaccepted writes a chunk can make;
//   * plan_copy_ops merging full consecutive pages while partial runs stay correct;
//   * the Host floor refusal text, derive_tap_config and plan_call_ends.

#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/prefix/layout_identity.h"
#include "models/gemma4/program/prefix/prefix_cache.h"
#include "models/gemma4/program/prefix/state_image.h"
#include "prefix_geometry_support.h"
#include "prefix_host_emulation.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace g4  = infernix::models::gemma4;
namespace g4p = infernix::models::gemma4::prefix;
namespace ops = infernix::ops;
namespace pc  = infernix::runtime::prefix_cache;
using namespace prefix_test;

constexpr std::uint32_t kWindow = 1024;          // W
constexpr std::uint32_t kKept   = kWindow - 1;   // an image keeps W-1 tokens
constexpr auto kFp8 = infernix::KvCacheStorage::Fp8E4M3Row256;
constexpr auto kBf16 = infernix::KvCacheStorage::BFloat16;
constexpr std::uint32_t kEmulatedSliding = 2; // sliding layers in the ring emulation (groups > 1)

struct Shape {
    const char* name;
    std::size_t global_layers, global_heads, sliding_layers, sliding_heads;
    std::uint32_t chunk;
};
constexpr Shape kShapes[] = {{"31B-like", 10, 4, 50, 16, 256}, {"26B-A4B-like", 5, 2, 50, 8, 512}};

std::string label(const Shape& s, const char* what) { return std::string(s.name) + " " + what; }

std::uint32_t ring_pages_for(std::uint32_t chunk) { return (chunk + kWindow - 1 + kPage - 1) / kPage; }

template <class E>
bool throws(const std::function<void()>& work, std::string* what = nullptr) {
    try { work(); }
    catch (const E& error) {
        if (what) { *what = error.what(); }
        return true;
    }
    return false;
}

// ---- the descriptors -------------------------------------------------------------------------

struct Descriptor {
    Shape shape;
    std::vector<g4::KvLayerFact> full;   // all layers of the model-sized descriptor
    g4::PrefixCacheGeometry geometry;
    GeometryExpectation want;
    g4p::LayoutIdentity identity;
};

Descriptor describe(const Shape& s) {
    Descriptor d{s, layer_facts(s.global_layers, s.global_heads, s.sliding_layers, s.sliding_heads), {}, {}, {}};
    d.geometry = g4::prefix_cache_geometry(d.full, kWindow, kFp8);
    d.want = expected_geometry(s.global_layers, s.global_heads, s.sliding_layers, s.sliding_heads, kWindow);
    d.identity = g4p::make_layout_identity(d.full, kWindow, s.chunk, kFp8, ops::Gemma4QkMode::BFloat16);
    return d;
}

// ---- layouts equal the geometry --------------------------------------------------------------

void test_layouts(const Descriptor& d) {
    const Shape& s = d.shape;
    const g4p::StateImageLayout image = g4p::plan_state_image(d.full, kWindow, kFp8);
    check(image.groups() == s.sliding_layers, label(s, "one restore group per sliding layer"),
          std::to_string(image.groups()));
    check(image.parts.size() == 2U * s.sliding_layers, label(s, "K and V part per group"));
    check(image.image_bytes == d.geometry.image_bytes && image.image_bytes == d.want.image_bytes,
          label(s, "image layout bytes equal prefix_cache_geometry"), std::to_string(image.image_bytes));
    check(image.window_tokens == kKept, label(s, "an image keeps W-1 tokens"));
    const std::size_t row = 256U * 2U; // BF16 sliding row
    std::size_t offset = g4p::kStateImageHeaderBytes;
    bool parts_ok = true;
    for (std::size_t i = 0; i < image.parts.size(); ++i) {
        const g4p::ImagePart& p = image.parts[i];
        parts_ok = parts_ok && p.group == i / 2 && p.plane == i % 2 && p.kv_heads == s.sliding_heads &&
                   p.row_bytes == row && p.page_bytes == kPage * s.sliding_heads * row &&
                   p.bytes == g4::align_bytes(static_cast<std::size_t>(kKept) * s.sliding_heads * row, 256) &&
                   p.offset == offset && p.offset % 256 == 0 && image.group_begin[i / 2] == (i / 2) * 2;
        offset += p.bytes;
    }
    check(parts_ok && offset == image.image_bytes && image.group_begin.back() == image.parts.size(),
          label(s, "image parts are 256-aligned, contiguous and sized align((W-1)*heads*row)"));
    // The sliding rings stay BF16 whatever the global format.
    check(g4p::plan_state_image(d.full, kWindow, kBf16).image_bytes == image.image_bytes,
          label(s, "image bytes do not depend on the global format"));

    const g4p::BlockLayout block = g4p::plan_block_layout(d.full, kFp8);
    const std::size_t code = static_cast<std::size_t>(kPage) * s.global_heads * 512U;
    const std::size_t scale = static_cast<std::size_t>(kPage) * s.global_heads * (512U / 256U) * 2U;
    check(block.global_layers == s.global_layers && block.parts.size() == 4U * s.global_layers,
          label(s, "block record has four FP8 planes per global layer"));
    check(block.record_bytes == d.want.global_bytes_per_token * kPage &&
              block.record_bytes <= d.geometry.block_bytes && d.geometry.block_bytes == d.want.block_bytes,
          label(s, "block record bytes equal the geometry's page"), std::to_string(block.record_bytes));
    std::size_t at = 0;
    bool block_ok = true;
    for (std::size_t i = 0; i < block.parts.size(); ++i) {
        const g4p::BlockPart& p = block.parts[i];
        const std::size_t bytes = i % 2 == 0 ? code : scale;
        block_ok = block_ok && p.layer == i / 4 && p.plane == i % 4 && p.offset == at && p.bytes == bytes;
        at += p.bytes;
    }
    check(block_ok && at == block.record_bytes, label(s, "block parts: layer-major, K codes/scales then V"));
    const g4p::BlockLayout bf16 = g4p::plan_block_layout(d.full, kBf16);
    const g4::PrefixCacheGeometry bf16_geometry = g4::prefix_cache_geometry(d.full, kWindow, kBf16);
    check(bf16.parts.size() == 2U * s.global_layers &&
              bf16.record_bytes == bf16_geometry.global_bytes_per_token * kPage &&
              bf16.record_bytes == s.global_layers * 2U * code * 2U,
          label(s, "BF16 block record has two planes per global layer"));

    // The identity is the geometry, never a literal.
    const g4p::LayoutIdentity& id = d.identity;
    check(id.global_layers == s.global_layers && id.global_kv_heads == s.global_heads &&
              id.global_head_dim == 512 && id.sliding_layers == s.sliding_layers &&
              id.sliding_kv_heads == s.sliding_heads && id.sliding_head_dim == 256 &&
              id.window_tokens == kWindow && id.prefill_chunk == s.chunk && id.page_tokens == kPage &&
              id.block_bytes == d.geometry.block_bytes && id.image_bytes == d.geometry.image_bytes &&
              id.scale_group_columns == 256 && id.scale_element_bytes == 2 &&
              id.storage == g4::KvStorageModel::KeyEqualsValue,
          label(s, "make_layout_identity carries the geometry"), id.describe());
    check(g4p::make_layout_identity(d.full, kWindow, s.chunk, kFp8, ops::Gemma4QkMode::Fp8).fingerprint() !=
              id.fingerprint(),
          label(s, "QK mode is part of the identity"));
    auto ragged = d.full;
    ragged.push_back({true, 512, static_cast<int>(s.global_heads) + 1});
    check(throws<std::invalid_argument>([&] {
              (void)g4p::make_layout_identity(ragged, kWindow, s.chunk, kFp8, ops::Gemma4QkMode::BFloat16);
          }),
          label(s, "global layers of unequal heads have no single identity"));
}

// ---- header ----------------------------------------------------------------------------------

Bytes header_bytes(const g4p::LayoutIdentity& id, const g4p::StateImageHeader& header) {
    Bytes out(g4p::kStateImageHeaderBytes); // exactly the header: a restore reads nothing beyond it
    g4p::write_state_image_header(id, header, out.data());
    return out;
}

g4p::StateImageHeader header_for(std::uint32_t frontier, bool opener = false, bool echo = false) {
    return {frontier, g4p::image_window_start(frontier, kKept), opener, echo};
}

void test_headers(const Descriptor& a, const Descriptor& b) {
    const std::string who = std::string(a.shape.name) + " vs " + b.shape.name;
    for (const std::uint32_t frontier : {1U, 63U, 64U, 65U, 1023U, 1024U, 2000U, 5000U}) {
        for (int flags = 0; flags < 4; ++flags) {
            const auto h = header_for(frontier, (flags & 1) != 0, (flags & 2) != 0);
            const auto check_a = g4p::read_state_image_header(a.identity, header_bytes(a.identity, h).data());
            check(check_a.ok() && check_a.header.frontier == h.frontier &&
                      check_a.header.image_start == h.image_start && check_a.header.opener == h.opener &&
                      check_a.header.lineage_echo == h.lineage_echo,
                  label(a.shape, "header round trip"), std::to_string(frontier));
        }
    }
    // A foreign identity is refused in both directions, from the header alone.
    const auto h = header_for(2000);
    for (const auto& [writer, reader] : {std::pair{&a, &b}, std::pair{&b, &a}}) {
        const auto verdict = g4p::read_state_image_header(reader->identity, header_bytes(writer->identity, h).data());
        check(!verdict.ok() && verdict.status == g4p::HeaderStatus::Identity,
              who + ": image of " + writer->shape.name + " refused by " + reader->shape.name,
              g4p::header_status_name(verdict.status));
    }
    // Value-only is its own identity: a K==V image never restores into it and the reverse.
    g4p::LayoutIdentity value_only = a.identity;
    value_only.storage = g4::KvStorageModel::ValueOnly;
    check(g4p::read_state_image_header(value_only, header_bytes(a.identity, h).data()).status ==
                  g4p::HeaderStatus::Identity &&
              g4p::read_state_image_header(a.identity, header_bytes(value_only, h).data()).status ==
                  g4p::HeaderStatus::Identity,
          label(a.shape, "value-only flag is refused against K==V in both directions"));
    check(g4p::read_state_image_header(value_only, header_bytes(value_only, h).data()).ok(),
          label(a.shape, "a value-only identity round-trips its own header"));
    // Every other identity field refuses; a packing version refuses by its own status.
    const std::pair<const char*, std::function<void(g4p::LayoutIdentity&)>> mutations[] = {
        {"qk_mode", [](g4p::LayoutIdentity& i) { i.qk_mode = ops::Gemma4QkMode::Fp8; }},
        {"kv_format", [](g4p::LayoutIdentity& i) { i.kv_format = ops::Gemma4KvFormat::BFloat16; }},
        {"prefill_chunk", [](g4p::LayoutIdentity& i) { ++i.prefill_chunk; }},
        {"window_tokens", [](g4p::LayoutIdentity& i) { ++i.window_tokens; }},
        {"global_kv_heads", [](g4p::LayoutIdentity& i) { ++i.global_kv_heads; }},
        {"sliding_kv_heads", [](g4p::LayoutIdentity& i) { ++i.sliding_kv_heads; }},
        {"sliding_layers", [](g4p::LayoutIdentity& i) { ++i.sliding_layers; }},
        {"scale_group_columns", [](g4p::LayoutIdentity& i) { i.scale_group_columns = 128; }},
        {"block_bytes", [](g4p::LayoutIdentity& i) { ++i.block_bytes; }},
        {"image_bytes", [](g4p::LayoutIdentity& i) { ++i.image_bytes; }},
    };
    for (const auto& [name, mutate] : mutations) {
        g4p::LayoutIdentity other = a.identity;
        mutate(other);
        check(g4p::read_state_image_header(a.identity, header_bytes(other, h).data()).status ==
                  g4p::HeaderStatus::Identity,
              label(a.shape, (std::string("a changed ") + name + " is refused").c_str()));
    }
    g4p::LayoutIdentity newer = a.identity;
    ++newer.packing_version;
    check(g4p::read_state_image_header(a.identity, header_bytes(newer, h).data()).status ==
              g4p::HeaderStatus::Version,
          label(a.shape, "another packing version is refused as a version"));
    // Magic, and a window start the frontier does not imply.
    const Bytes zeros(g4p::kStateImageHeaderBytes);
    check(g4p::read_state_image_header(a.identity, zeros.data()).status == g4p::HeaderStatus::Magic,
          label(a.shape, "a zero page is not an image"));
    g4p::StateImageHeader skewed = header_for(2000);
    ++skewed.image_start;
    check(g4p::read_state_image_header(a.identity, header_bytes(a.identity, skewed).data()).status ==
              g4p::HeaderStatus::Frontier,
          label(a.shape, "a window start that does not follow the frontier is refused"));
    g4p::StateImageHeader low{100, 1, false, false};
    check(g4p::read_state_image_header(a.identity, header_bytes(a.identity, low).data()).status ==
              g4p::HeaderStatus::Frontier,
          label(a.shape, "a window start past 0 below W-1 is refused"));
    // Corruption: a flipped byte the reader ignores must be padding (zero in the original).
    const auto original = header_for(2000, true, false);
    const Bytes clean = header_bytes(a.identity, original);
    int ignored_nonzero = 0;
    for (std::size_t i = 0; i < clean.size(); ++i) {
        Bytes flipped = clean;
        flipped[i] ^= std::byte{0x5A};
        const auto verdict = g4p::read_state_image_header(a.identity, flipped.data());
        const bool same = verdict.ok() && verdict.header.frontier == original.frontier &&
                          verdict.header.image_start == original.image_start &&
                          verdict.header.opener == original.opener &&
                          verdict.header.lineage_echo == original.lineage_echo;
        if (same && clean[i] != std::byte{0}) { ++ignored_nonzero; }
    }
    check(ignored_nonzero == 0, label(a.shape, "no non-padding header byte is ignored by validation"),
          std::to_string(ignored_nonzero));
}

// ---- the ring window -------------------------------------------------------------------------

void check_runs(const std::string& what, std::uint32_t start, std::uint32_t frontier, std::uint32_t ring_pages) {
    const auto runs = g4p::window_runs(start, frontier, ring_pages);
    std::vector<std::array<std::uint32_t, 3>> got, want;
    std::uint32_t covered = 0;
    bool shaped = true;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const auto& r = runs[i];
        shaped = shaped && r.first == covered && r.tokens >= 1 && r.first_row + r.tokens <= kPage &&
                 (i == 0 || r.first_row == 0) && (i + 1 == runs.size() || r.first_row + r.tokens == kPage);
        covered += r.tokens;
        for (std::uint32_t t = 0; t < r.tokens; ++t) {
            got.push_back({start + r.first + t, r.physical_page, r.first_row + t});
        }
    }
    for (std::uint32_t p = start; p < frontier; ++p) { want.push_back({p, (p / kPage) % ring_pages, p % kPage}); }
    check(shaped && covered == frontier - start && got == want, "window_runs " + what,
          "[" + std::to_string(start) + "," + std::to_string(frontier) + ") ring " + std::to_string(ring_pages));
}

void test_window_runs() {
    constexpr std::uint32_t ring = 24; // 512-token chunk + 1023
    check_runs("empty", 0, 0, ring);
    for (const std::uint32_t f : {1U, 63U, 64U, 65U, 127U, 500U, 1022U, 1023U}) { // F < W-1: start 0
        check_runs("F<W-1 " + std::to_string(f), g4p::image_window_start(f, kKept), f, ring);
    }
    const std::uint32_t capacity = ring * kPage;
    for (const std::uint32_t f : {1024U, 1025U, capacity - 1, capacity, capacity + 1, capacity + 63, capacity + 64,
                                  capacity + 65, 2 * capacity + 1}) {
        check_runs("F=" + std::to_string(f), g4p::image_window_start(f, kKept), f, ring);
    }
    // Wrap is visible in the physical pages: the window of F=capacity+100 crosses the ring end once.
    const std::uint32_t f = capacity + 100;
    const auto runs = g4p::window_runs(g4p::image_window_start(f, kKept), f, ring);
    int drops = 0;
    for (std::size_t i = 1; i < runs.size(); ++i) { drops += runs[i].physical_page < runs[i - 1].physical_page; }
    check(drops == 1 && runs.back().physical_page == 1, "window_runs wraps once to physical page 0", std::to_string(drops));
    check(throws<std::invalid_argument>([] { (void)g4p::window_runs(0, 10, 0); }) &&
              throws<std::invalid_argument>([] { (void)g4p::window_runs(10, 9, ring); }),
          "window_runs refuses an empty ring and a window ending before it starts");
    check(g4p::image_window_start(1023, kKept) == 0 && g4p::image_window_start(1024, kKept) == 1 &&
              g4p::image_window_start(5000, kKept) == 5000 - kKept,
          "image_window_start is max(0, F-(W-1))");
}

void test_ring_transfers(const Shape& s) {
    const auto facts = layer_facts(s.global_layers, s.global_heads, kEmulatedSliding, s.sliding_heads);
    const g4p::StateImageLayout layout = g4p::plan_state_image(facts, kWindow, kFp8);
    const g4::PrefixCacheGeometry geometry = g4::prefix_cache_geometry(facts, kWindow, kFp8);
    const g4p::LayoutIdentity id = g4p::make_layout_identity(facts, kWindow, s.chunk, kFp8, ops::Gemma4QkMode::BFloat16);
    const std::uint32_t ring_pages = ring_pages_for(s.chunk), capacity = ring_pages * kPage;
    Slabs slabs(geometry.slab_bytes, geometry.image_slabs);
    Rings live(layout, ring_pages), restored(layout, ring_pages), want(layout, ring_pages);

    std::vector<std::uint32_t> frontiers = {1, 63, 64, 65, 127, 128, 129, 1022, 1023, 1024, 1025, capacity - 1,
                                            capacity, capacity + 1, capacity + 63, capacity + 64, capacity + 65,
                                            2 * capacity + 1};
    for (const std::uint32_t frontier : frontiers) {
        const std::string at = label(s, ("ring round trip at F=" + std::to_string(frontier)).c_str());
        const g4p::StateImageHeader header = header_for(frontier, frontier % 2 == 0, false);
        live.reset();
        for (std::uint32_t p = frontier > capacity ? frontier - capacity : 0; p < frontier; ++p) { live.write(layout, p, 0); }
        slabs.zero();
        g4p::write_state_image_header(id, header, slabs.pointers[0]);
        transfer(layout, live, slabs.image(), header.image_start, frontier, true);
        check(slabs.flatten(layout.image_bytes) == expected_image(layout, id, header, 0), at + ": gather equals the layout oracle");
        check(g4p::read_state_image_header(id, slabs.pointers[0]).ok(), at + ": the gather leaves the header valid");

        restored.reset();
        transfer(layout, restored, slabs.image(), header.image_start, frontier, false);
        want.reset();
        for (std::uint32_t p = header.image_start; p < frontier; ++p) { want.write(layout, p, 0); }
        check(restored == want, at + ": scatter restores exactly the window and nothing else");
    }

    // The committed window survives the unaccepted writes a chunk can make past it (ring capacity
    // >= chunk + W-1), and the one write past that bound reaches it (the check is sensitive).
    for (const std::uint32_t frontier : {kWindow + 76, capacity + 65}) {
        const g4p::StateImageHeader header = header_for(frontier);
        live.reset();
        for (std::uint32_t p = frontier > capacity ? frontier - capacity : 0; p < frontier; ++p) { live.write(layout, p, 0); }
        const auto gather = [&] {
            slabs.zero();
            g4p::write_state_image_header(id, header, slabs.pointers[0]);
            transfer(layout, live, slabs.image(), header.image_start, frontier, true);
            return slabs.flatten(layout.image_bytes);
        };
        const Bytes before = gather();
        const std::uint32_t safe = capacity - kKept; // positions F .. F+safe-1 overwrite only < start
        for (std::uint32_t p = frontier; p < frontier + safe; ++p) { live.write(layout, p, 1); }
        check(gather() == before, label(s, "unaccepted writes up to the ring slack leave the committed window"),
              std::to_string(frontier));
        live.write(layout, frontier + safe, 1);
        check(gather() != before, label(s, "one write past the slack reaches the window start (control)"),
              std::to_string(frontier));
    }
}

// ---- copy planning ---------------------------------------------------------------------------

void test_plan_copy_ops() {
    constexpr std::size_t page = 96, count = 10, host_stride = 160;
    Bytes device(count * page + 64), host(count * host_stride + 64);
    const auto at = [&](std::size_t d, std::size_t h, std::size_t bytes) {
        return g4p::CopyPiece{device.data() + d, host.data() + h, bytes};
    };
    std::vector<g4p::CopyPiece> pieces;
    for (std::size_t i = 0; i < count; ++i) { pieces.push_back(at(i * page, i * page, page)); }
    auto ops = g4p::plan_copy_ops(pieces);
    check(ops.size() == 1 && ops[0].height == 1 && ops[0].width == count * page, "contiguous pages merge into one copy");

    pieces.clear();
    for (std::size_t i = 0; i < count; ++i) { pieces.push_back(at(i * page, i * host_stride, page)); }
    ops = g4p::plan_copy_ops(pieces);
    check(ops.size() == 1 && ops[0].height == count && ops[0].width == page && ops[0].device_pitch == page &&
              ops[0].host_pitch == host_stride,
          "equal pieces with constant strides fold into one 2-D copy");

    pieces.push_back(at(count * page, count * host_stride, page / 3)); // a partial run at the end
    ops = g4p::plan_copy_ops(pieces);
    check(ops.size() == 2 && ops[0].height == count && ops[1].width == page / 3 && ops[1].height == 1,
          "a partial trailing run stays its own copy");

    pieces = {at(0, 0, page), at(page + 8, page, page), at(2 * page + 8, 2 * page, page)}; // device gap after the first
    ops = g4p::plan_copy_ops(pieces);
    check(ops.size() == 2 && ops[0].width == page && ops[1].width == 2 * page, "a gap breaks a merge on that side only");
    check(g4p::plan_copy_ops({}).empty() && g4p::plan_copy_ops(std::vector{at(0, 0, 0)}).empty(), "empty pieces plan nothing");

    // Replaying the planned copies reproduces piece-by-piece copying, both directions.
    std::uint64_t state = 7;
    const auto next = [&](std::uint64_t modulo) { state = mix(state); return state % modulo; };
    Bytes src_device(1 << 16), dst_device(1 << 16), dst_host(1 << 17), want_host(1 << 17), src_host(1 << 17);
    for (std::size_t i = 0; i < src_device.size(); ++i) { src_device[i] = std::byte(mix(i) & 0xFF); }
    for (std::size_t i = 0; i < src_host.size(); ++i) { src_host[i] = std::byte(mix(i + 99) & 0xFF); }
    std::vector<g4p::CopyPiece> save, load;
    std::size_t d = 0, h = 0, total = 0;
    while (d + 1024 < src_device.size() && h + 2048 < dst_host.size()) {
        const std::size_t width = next(2) ? 32 : 64, repeat = 1 + next(5), dgap = next(2) * 16, hgap = next(2) * 48;
        for (std::size_t r = 0; r < repeat; ++r) {
            save.push_back({src_device.data() + d, dst_host.data() + h, width});
            load.push_back({dst_device.data() + d, src_host.data() + h, width});
            std::memcpy(want_host.data() + h, src_device.data() + d, width);
            d += width + dgap;
            h += width + hgap;
            total += width;
        }
    }
    const auto save_ops = g4p::plan_copy_ops(save);
    const auto load_ops = g4p::plan_copy_ops(load);
    std::size_t moved = 0;
    for (const auto& op : save_ops) { moved += op.width * op.height; }
    replay(save_ops, true);
    replay(load_ops, false);
    bool loaded = true;
    for (const auto& piece : load) { loaded = loaded && std::memcmp(piece.device, piece.host, piece.bytes) == 0; }
    check(dst_host == want_host && moved == total && save_ops.size() < save.size() && loaded,
          "planned copies replay to the same bytes as the pieces, in both directions");

    // append_image_pieces cuts a range at the slab boundaries and nowhere else.
    Bytes slab_a(4096), slab_b(4096), slab_c(4096);
    std::byte* segments[] = {slab_a.data(), slab_b.data(), slab_c.data()};
    const g4p::HostImage image{segments, 4096};
    Bytes target(4300);
    std::vector<g4p::CopyPiece> cut;
    g4p::append_image_pieces(image, 4000, 4300, target.data(), cut);
    check(cut.size() == 3 && cut[0].host == slab_a.data() + 4000 && cut[0].bytes == 96 && cut[1].host == slab_b.data() &&
              cut[1].bytes == 4096 && cut[2].host == slab_c.data() && cut[2].bytes == 108 &&
              cut[1].device == target.data() + 96 && cut[2].device == target.data() + 96 + 4096,
          "append_image_pieces splits at segment boundaries");
    cut.clear();
    g4p::append_image_pieces(image, 100, 0, target.data(), cut);
    check(cut.empty() && throws<std::out_of_range>([&] { g4p::append_image_pieces(image, 12000, 300, target.data(), cut); }) &&
              throws<std::out_of_range>([&] { g4p::append_image_pieces({}, 0, 1, target.data(), cut); }),
          "append_image_pieces refuses ranges outside the slabs");

    // A copy never spans two registered Host chunks: pieces whose host ranges are address-adjacent but
    // lie in different chunks stay apart; adjacent pieces of one chunk still merge.
    {
        constexpr std::size_t piece_bytes = 64;
        Bytes dev(8 * piece_bytes), hst(8 * piece_bytes);
        const auto piece = [&](std::size_t i, std::uint32_t chunk) {
            return g4p::CopyPiece{dev.data() + i * piece_bytes, hst.data() + i * piece_bytes, piece_bytes, chunk};
        };
        const auto same = g4p::plan_copy_ops(std::vector{piece(0, 3), piece(1, 3), piece(2, 3), piece(3, 3)});
        check(same.size() == 1 && same[0].height == 1 && same[0].width == 4 * piece_bytes,
              "adjacent pieces of one host chunk still merge");
        const auto cross = g4p::plan_copy_ops(std::vector{piece(0, 3), piece(1, 3), piece(2, 4), piece(3, 4)});
        check(cross.size() == 2 && cross[0].width == 2 * piece_bytes && cross[1].width == 2 * piece_bytes &&
                  cross[0].host == hst.data() && cross[1].host == hst.data() + 2 * piece_bytes,
              "address-adjacent pieces in different host chunks are not merged");
        const auto alone = g4p::plan_copy_ops(std::vector{piece(0, 1), piece(1, 2)});
        check(alone.size() == 2 && alone[0].width == piece_bytes && alone[1].width == piece_bytes,
              "two adjacent pieces each alone in their chunk stay two copies");

        // Folding equal-stride runs needs one chunk across every row of the 2-D copy.
        constexpr std::size_t stride = 160, rows = 6;
        Bytes strided(rows * stride);
        std::vector<g4p::CopyPiece> rows_same, rows_split;
        for (std::size_t i = 0; i < rows; ++i) {
            rows_same.push_back({dev.data() + i * piece_bytes, strided.data() + i * stride, piece_bytes, 9U});
            rows_split.push_back({dev.data() + i * piece_bytes, strided.data() + i * stride, piece_bytes,
                                  i < rows / 2 ? 9U : 10U});
        }
        const auto folded = g4p::plan_copy_ops(rows_same);
        check(folded.size() == 1 && folded[0].height == rows && folded[0].host_pitch == stride &&
                  folded[0].device_pitch == piece_bytes,
              "equal-stride rows of one chunk fold into one 2-D copy");
        const auto halves = g4p::plan_copy_ops(rows_split);
        check(halves.size() == 2 && halves[0].height == rows / 2 && halves[1].height == rows / 2,
              "a 2-D copy never has rows in two host chunks");
    }

    // A copy never spans two Device allocations: adjacent bytes or equal strides across different
    // device_plane ids (a layer's K and V buffers, layer i and i+1) are neither merged nor folded.
    {
        constexpr std::size_t piece_bytes = 64, gap = 4096;
        Bytes dev(8 * piece_bytes), far(3 * gap), hst(8 * piece_bytes);
        const auto adjacent = [&](std::size_t i, std::uint32_t plane) {
            return g4p::CopyPiece{dev.data() + i * piece_bytes, hst.data() + i * piece_bytes, piece_bytes, 0, plane};
        };
        const auto same = g4p::plan_copy_ops(std::vector{adjacent(0, 1), adjacent(1, 1), adjacent(2, 1), adjacent(3, 1)});
        check(same.size() == 1 && same[0].height == 1 && same[0].width == 4 * piece_bytes,
              "adjacent pieces of one device plane still merge");
        const auto cross = g4p::plan_copy_ops(std::vector{adjacent(0, 1), adjacent(1, 1), adjacent(2, 2), adjacent(3, 2)});
        check(cross.size() == 2 && cross[0].width == 2 * piece_bytes && cross[1].width == 2 * piece_bytes,
              "byte-adjacent pieces of different device planes are not merged");
        const auto strided = [&](std::size_t i, std::uint32_t plane) {
            return g4p::CopyPiece{far.data() + i * gap, hst.data() + i * piece_bytes, piece_bytes, 0, plane};
        };
        const auto folded = g4p::plan_copy_ops(std::vector{strided(0, 4), strided(1, 4), strided(2, 4)});
        check(folded.size() == 1 && folded[0].height == 3 && folded[0].device_pitch == gap &&
                  folded[0].host_pitch == piece_bytes,
              "equal-stride rows of one device plane fold into one 2-D copy");
        const auto apart = g4p::plan_copy_ops(std::vector{strided(0, 4), strided(1, 4), strided(2, 5)});
        check(apart.size() == 2 && apart[0].height == 2 && apart[1].height == 1,
              "rows of different device planes never share a 2-D copy");
        const auto lone = g4p::plan_copy_ops(std::vector{strided(0, 6), strided(1, 7)});
        check(lone.size() == 2 && lone[0].height == 1 && lone[1].height == 1,
              "one row per device plane stays plain copies");
    }

    // append_image_pieces tags every piece with its segment's chunk.
    {
        Bytes seg_a(4096), seg_b(4096), seg_c(4096);
        std::byte* seg_pointers[] = {seg_a.data(), seg_b.data(), seg_c.data()};
        const std::uint32_t seg_chunks[] = {5, 5, 7};
        const g4p::HostImage tagged{seg_pointers, 4096, seg_chunks};
        Bytes sink(9000);
        std::vector<g4p::CopyPiece> tagged_pieces;
        g4p::append_image_pieces(tagged, 4000, 4300, sink.data(), tagged_pieces);
        check(tagged_pieces.size() == 3 && tagged_pieces[0].host_chunk == 5 && tagged_pieces[1].host_chunk == 5 &&
                  tagged_pieces[2].host_chunk == 7,
              "append_image_pieces carries each segment's host chunk");
    }
}

// A block's DtoH/HtoD (PrefixCache::append_block_pieces): plane-major, block-minor over consecutive pages.
void test_block_copies(const Descriptor& d, infernix::KvCacheStorage format, const char* what) {
    const Shape& s = d.shape;
    const g4p::BlockLayout layout = g4p::plan_block_layout(d.full, format);
    constexpr std::uint32_t pages = 5, first_page = 3; // pages 3..7 of the sequence, written to records 0..4
    std::size_t slab_bytes = g4::align_bytes(layout.record_bytes, 4096);
    std::vector<Bytes> device(layout.parts.size());
    for (std::size_t i = 0; i < device.size(); ++i) {
        device[i].resize((first_page + pages) * layout.parts[i].bytes);
        for (std::size_t b = 0; b < device[i].size(); ++b) { device[i][b] = std::byte(mix(i * 7919 + b) & 0xFF); }
    }
    Bytes host(pages * slab_bytes + 1);
    const auto pieces_for = [&](Bytes* base_host, std::vector<Bytes>& dev) {
        std::vector<g4p::CopyPiece> out;
        for (std::size_t i = 0; i < layout.parts.size(); ++i) {
            for (std::uint32_t k = 0; k < pages; ++k) {
                out.push_back({dev[i].data() + (first_page + k) * layout.parts[i].bytes,
                               base_host->data() + k * slab_bytes + layout.parts[i].offset, layout.parts[i].bytes});
            }
        }
        return out;
    };
    const auto save_ops = g4p::plan_copy_ops(pieces_for(&host, device));
    check(save_ops.size() == layout.parts.size() &&
              std::all_of(save_ops.begin(), save_ops.end(), [&](const g4p::CopyOp& op) {
                  return op.height == pages && op.host_pitch == slab_bytes && op.device_pitch == op.width;
              }),
          label(s, (std::string(what) + " blocks: one folded copy per plane").c_str()));
    replay(save_ops, true);
    bool saved = true;
    for (std::size_t i = 0; i < layout.parts.size(); ++i) {
        for (std::uint32_t k = 0; k < pages; ++k) {
            saved = saved && std::memcmp(host.data() + k * slab_bytes + layout.parts[i].offset,
                                         device[i].data() + (first_page + k) * layout.parts[i].bytes, layout.parts[i].bytes) == 0;
        }
    }
    std::vector<Bytes> restored(layout.parts.size());
    for (std::size_t i = 0; i < restored.size(); ++i) { restored[i].assign(device[i].size(), std::byte{0}); }
    replay(g4p::plan_copy_ops(pieces_for(&host, restored)), false);
    bool round_trip = true;
    for (std::size_t i = 0; i < restored.size(); ++i) {
        round_trip = round_trip && std::equal(restored[i].begin() + first_page * layout.parts[i].bytes, restored[i].end(),
                                              device[i].begin() + first_page * layout.parts[i].bytes) &&
                     std::all_of(restored[i].begin(), restored[i].begin() + first_page * layout.parts[i].bytes,
                                 [](std::byte b) { return b == std::byte{0}; });
    }
    check(saved && round_trip, label(s, (std::string(what) + " block bytes round-trip through the planned copies").c_str()));
}

// Full consecutive ring pages are one contiguous copy per plane; partial runs are strided copies.
void test_ring_copy_shape(const Shape& s) {
    const auto facts = layer_facts(s.global_layers, s.global_heads, 1, s.sliding_heads);
    const g4p::StateImageLayout layout = g4p::plan_state_image(facts, kWindow, kFp8);
    const std::uint32_t ring_pages = ring_pages_for(s.chunk);
    const g4p::ImagePart& part = layout.parts[0];
    Bytes ring(static_cast<std::size_t>(ring_pages) * part.page_bytes), image_bytes(layout.image_bytes);
    std::byte* segment[] = {image_bytes.data()}; // one segment: merges are not cut by slab edges
    const g4p::HostImage image{segment, image_bytes.size()};
    const g4p::DevicePlane plane{ring.data(), part.page_bytes};
    struct Case { std::uint32_t frontier; std::size_t ops; const char* what; };
    const std::uint32_t capacity = ring_pages * kPage;
    for (const Case& c : {Case{130, 2, "F<W-1: two full pages (merged) then a partial run"},
                          Case{kWindow + 76, 3, "partial, full pages merged, partial"},
                          Case{1088, 2, "partial, full pages merged (page-aligned frontier)"},
                          Case{capacity + 100, 4, "a wrap splits the full pages into two merged copies"}}) {
        const std::uint32_t start = g4p::image_window_start(c.frontier, kKept);
        const auto runs = g4p::window_runs(start, c.frontier, ring_pages);
        const auto ops = g4p::plan_copy_ops(part_pieces(part, plane, runs, image));
        std::size_t moved = 0;
        for (const auto& op : ops) { moved += op.width * op.height; }
        check(ops.size() == c.ops && moved == static_cast<std::size_t>(c.frontier - start) * part.kv_heads * part.row_bytes,
              label(s, c.what), "ops " + std::to_string(ops.size()));
    }
}

// ---- Host floor, tap ladder, call planning ---------------------------------------------------

void test_host_floor(const Descriptor& d) {
    const Shape& s = d.shape;
    const std::size_t mib = (d.want.host_floor_bytes + (1U << 20) - 1) >> 20;
    const std::string expected = "needs " + std::to_string(mib) + " MiB (" + std::to_string(d.want.image_slabs) +
                                 "+2 slabs of " + std::to_string((d.want.block_bytes + 1023) >> 10) + " KiB";
    const std::string message = g4p::host_floor_refusal(d.geometry);
    check(message.find(expected) != std::string::npos && message.find("--host-context-mib") != std::string::npos,
          label(s, "host floor refusal names (image_slabs+2) slabs in MiB"), message);
    const std::uint64_t floor = (static_cast<std::uint64_t>(d.want.image_slabs) + 2U) * d.want.block_bytes;
    const g4p::PrefixHostLayout at_floor = g4p::plan_prefix_host_layout(d.geometry, floor);
    check(at_floor.slabs == d.want.image_slabs + 2 && at_floor.image_slabs == d.want.image_slabs &&
              at_floor.slab_bytes == d.want.block_bytes && at_floor.block_bytes == d.want.block_bytes &&
              at_floor.image_bytes == d.want.image_bytes,
          label(s, "the floor is accepted with image_slabs+2 slabs"));
    check(g4p::plan_prefix_host_layout(d.geometry, floor + 3U * d.want.block_bytes + 17).slabs == d.want.image_slabs + 5,
          label(s, "slabs are the whole slabs of the budget"));
    std::string what;
    check(throws<std::invalid_argument>([&] { (void)g4p::plan_prefix_host_layout(d.geometry, floor - 1); }, &what) &&
              what == message,
          label(s, "one byte below the floor refuses with the refusal text"), what);
    check(throws<std::invalid_argument>([&] { (void)g4p::plan_prefix_host_layout(d.geometry, 0); }),
          label(s, "a zero budget is below the floor"));
    check(throws<std::invalid_argument>([&] {
              (void)g4p::plan_prefix_host_layout(
                  d.geometry, (static_cast<std::uint64_t>(pc::kNoId / 4U) + 1U) * d.want.block_bytes);
          }),
          label(s, "a tier too large to index is refused"));
}

void test_tap_config(const Descriptor& big, const Descriptor& small) {
    const pc::TapPlannerConfig defaults{};
    constexpr std::size_t reference = 192U << 20;
    const std::size_t sizes[] = {0, 1U << 20, 96U << 20, reference, reference + 1, 300U << 20, small.want.image_bytes,
                                 big.want.image_bytes, std::size_t{2} << 30, std::size_t{8} << 30};
    pc::TapPlannerConfig previous = g4p::derive_tap_config(sizes[0]);
    for (const std::size_t bytes : sizes) {
        const pc::TapPlannerConfig c = g4p::derive_tap_config(bytes);
        const std::string at = "derive_tap_config(" + std::to_string(bytes) + ")";
        check(c.max_new_taps >= 3 && c.max_new_taps <= defaults.max_new_taps && c.ladder_tokens >= defaults.ladder_tokens &&
                  c.min_gap_tokens >= defaults.min_gap_tokens && c.ladder_tokens % defaults.ladder_tokens == 0 &&
                  c.min_gap_tokens % defaults.min_gap_tokens == 0,
              at + " stays within the planner's bounds");
        check(c.max_new_taps <= previous.max_new_taps && c.ladder_tokens >= previous.ladder_tokens &&
                  c.min_gap_tokens >= previous.min_gap_tokens,
              at + " is monotonic in the image size");
        check(c.max_new_taps == 3 || static_cast<double>(c.max_new_taps) * static_cast<double>(bytes) <= 16.0 * reference,
              at + " bounds a prompt's tap DtoH to 16 reference images unless at the minimum taps");
        if (bytes <= reference) {
            check(c.max_new_taps == defaults.max_new_taps && c.ladder_tokens == defaults.ladder_tokens &&
                      c.min_gap_tokens == defaults.min_gap_tokens,
                  at + " keeps the defaults for an image the defaults were tuned for");
        }
        previous = c;
    }
    const auto c_big = g4p::derive_tap_config(big.want.image_bytes), c_small = g4p::derive_tap_config(small.want.image_bytes);
    check(c_small.max_new_taps >= c_big.max_new_taps && c_small.ladder_tokens <= c_big.ladder_tokens,
          "the smaller geometry's tap ladder is no sparser than the larger one's");
}

void test_call_ends() {
    const std::vector<std::uint32_t> none;
    check(g4p::plan_call_ends(0, 0, 128, none).empty() && g4p::plan_call_ends(500, 500, 128, none).empty(),
          "plan_call_ends: nothing to prefill, no calls");
    check(g4p::plan_call_ends(0, 300, 128, none) == std::vector<std::uint32_t>{128, 256, 300} &&
              g4p::plan_call_ends(256, 300, 128, none) == std::vector<std::uint32_t>{300} &&
              g4p::plan_call_ends(0, 128, 128, none) == std::vector<std::uint32_t>{128},
          "plan_call_ends: chunk calls and a remainder");
    const std::vector<std::uint32_t> cuts = {100, 300, 400};
    check(g4p::plan_call_ends(0, 450, 128, cuts) == std::vector<std::uint32_t>{100, 228, 300, 400, 450},
          "plan_call_ends: every exact tap ends a call, chunks restart after it");
    check(g4p::plan_call_ends(200, 450, 128, std::vector<std::uint32_t>{100, 200, 450, 900}) ==
              std::vector<std::uint32_t>{328, 450},
          "plan_call_ends: cuts at or before the frontier and at or past the prompt are ignored");
    check(throws<std::invalid_argument>([&] { (void)g4p::plan_call_ends(0, 10, 0, none); }), "plan_call_ends: an empty chunk is refused");
    int broken = 0;
    for (const std::uint32_t frontier : {0U, 1U, 63U, 64U, 65U, 1000U}) {
        for (const std::uint32_t prompt : {frontier + 1, frontier + 64, frontier + 1537, 4097U}) {
            for (const std::uint32_t chunk : {1U, 64U, 128U, 2048U}) {
                const std::vector<std::uint32_t> taps = {frontier + 1, frontier + 65, prompt - 1, prompt / 2};
                std::vector<std::uint32_t> sorted = taps;
                std::sort(sorted.begin(), sorted.end());
                const auto ends = g4p::plan_call_ends(frontier, prompt, chunk, sorted);
                std::uint32_t at = frontier;
                bool ok = !ends.empty() && ends.back() == prompt;
                for (const std::uint32_t end : ends) { ok = ok && end > at && end - at <= chunk; at = end; }
                for (const std::uint32_t cut : sorted) {
                    if (cut > frontier && cut < prompt) { ok = ok && std::binary_search(ends.begin(), ends.end(), cut); }
                }
                broken += ok ? 0 : 1;
            }
        }
    }
    check(broken == 0, "plan_call_ends: increasing ends within a chunk, every cut an end, the prompt the last", std::to_string(broken));
}

} // namespace

int main() {
    const Descriptor big = describe(kShapes[0]), small = describe(kShapes[1]);
    for (const Descriptor* d : {&big, &small}) {
        test_layouts(*d);
        test_host_floor(*d);
        test_ring_transfers(d->shape);
        test_ring_copy_shape(d->shape);
        test_block_copies(*d, kFp8, "FP8");
        test_block_copies(*d, kBf16, "BF16");
    }
    test_headers(big, small);
    test_headers(small, big);
    test_window_runs();
    test_plan_copy_ops();
    test_tap_config(big, small);
    test_call_ends();
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 prefix store\n";
    return failures ? 1 : 0;
}
