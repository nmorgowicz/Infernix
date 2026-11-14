#include "models/gemma4/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/startup.h"
#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/program/kv_layout.h"
#include "infernix/types.h"
#include "models/gemma4/execution/text_forward.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/target_logprobs.h"
#include "infernix/ops/top_logprobs.h"
#include "models/gemma4/program/prefix/layout_identity.h"
#include "models/gemma4/program/prefix/prefix_cache.h"
#include "models/gemma4/program/prefix/state_image.h"
#include "runtime/prefix_cache/block_hash.h"
#include "runtime/prefix_cache/prefix_index.h"
#include "runtime/prefix_cache/tap_planner.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace infernix::models::gemma4 {
namespace pc = runtime::prefix_cache;
namespace {
constexpr std::uint32_t kPage = ops::kGemma4KvPageTokens;
constexpr int kScoreTile = 128;
constexpr std::array<std::uint32_t, 6> kGraphWidths{32, 64, 128, 256, 512, 1024};
constexpr std::uint64_t kGraphAllowance = 64ULL << 20;
// Ranking only (CacheCostModel never decides feasibility): what a prefilled token costs against a
// restore's copy time when the cache chooses between sources and values snapshots.
constexpr double kCacheTokenSeconds = 1.0 / 1500.0;
PrefixReusePath reuse_path_for(pc::SnapshotKind kind) noexcept {
    return kind == pc::SnapshotKind::Endpoint ? PrefixReusePath::HybridEndpoint : PrefixReusePath::HybridSnapshot;
}
std::uint64_t aligned(std::uint64_t bytes) { return (bytes + 255) & ~std::uint64_t{255}; }
std::uint32_t pages_for(std::uint32_t tokens) { return (tokens + kPage - 1) / kPage; }
std::uint32_t ring_pages(std::uint32_t chunk, std::uint32_t window) {
    return pages_for(chunk + window - 1);
}
std::uint64_t io_bytes(const TextConfig& c, std::uint32_t chunk, std::uint32_t columns) {
    // Each item is a separate aligned arena allocation. Head and distribution staging are shared
    // by prefill readout, decode and scoring; no hot-path device allocation is required.
    return aligned(4ULL * chunk) * 2 + aligned(2ULL * c.hidden_size * chunk) +
        aligned(2ULL * c.vocab_size * columns) + aligned(sizeof(ops::SamplingConfig)) +
        aligned(4ULL * c.vocab_size) + aligned(4ULL * ((c.vocab_size + 31) / 32)) +
        aligned(4 * sizeof(std::int32_t)) +
        2 * aligned(4ULL * kMaximumScoreTopTokens * columns);
}
void validate_options(const Model& model, const ProgramOptions& o, std::uint32_t public_tokens) {
    const auto& c = model.config().text;
    if (o.max_context == 0 || o.max_context > c.max_position_embeddings ||
        o.max_context > static_cast<std::uint32_t>(INT32_MAX) - 64 ||
        public_tokens != c.vocab_size || c.hidden_size == 0 || c.layer_types.empty()) {
        throw std::invalid_argument("Gemma4: invalid context or model geometry");
    }
    if (o.max_concurrency != 1) { throw std::invalid_argument("Gemma4 requires max_concurrency=1"); }
    if (o.reproducible) { throw std::invalid_argument("Gemma4 does not support --reproducible yet"); }
    if (o.prefix_cache) {
        if (o.host_bytes == 0) {
            throw std::invalid_argument(
                "Gemma4's prefix cache needs a Host tier: --host-context-mib must be positive "
                "(0 is the off switch)");
        }
    } else if (o.host_bytes != 0) {
        throw std::invalid_argument("Gemma4 host context bytes require the prefix cache to be enabled");
    }
    (void)gemma4_kv_format(o.kv_cache);
    validate_gemma4_qk(o.qk_mode, true, o.kv_cache);
    if (model.config().mtp) { throw std::invalid_argument("Gemma4 P4 does not execute MTP"); }
    if (o.prefill_chunk < 16 || o.prefill_chunk > 16384) {
        throw std::invalid_argument("Gemma4 prefill_chunk must be in [16,16384]");
    }
    if (o.kv_capacity_tokens != 0 && (o.kv_capacity_tokens > pages_for(o.max_context) * kPage ||
        o.kv_capacity_tokens % kPage != 0)) {
        throw std::invalid_argument("Gemma4 KV capacity must be page aligned and bounded by max_context");
    }
}
} // namespace

ProgramDevicePlan Program::plan_device(const Model& model, const ProgramOptions& options,
                                        std::uint32_t public_tokens) {
    validate_options(model, options, public_tokens);
    const auto& c = model.config().text;
    const auto geometry = execution::Parameters::geometry(model);
    ProgramDevicePlan plan;
    const std::uint32_t maximum_pages = pages_for(options.max_context);
    plan.kv_pages = options.kv_capacity_tokens ? options.kv_capacity_tokens / kPage : maximum_pages;
    const auto rings = std::min(maximum_pages, ring_pages(options.prefill_chunk, c.sliding_window));
    std::uint64_t global_page_bytes = 0;
    std::uint64_t sliding_page_bytes = 0;
    for (const auto& layer : geometry.layers) {
        // Global layers follow the requested format (FP8 codes plus FP16 scales); rings stay BF16.
        const std::uint64_t bytes =
            require_implemented_storage(
                kv_layer_layout(options.kv_cache, layer.global_layer, layer.head_dim, layer.kv_heads))
                .page_bytes();
        (layer.global_layer ? global_page_bytes : sliding_page_bytes) += bytes;
        plan.workspace = std::max<std::uint64_t>(plan.workspace,
            gemma4_layer_workspace_bytes(layer, static_cast<int>(options.prefill_chunk)));
    }
    // One logits tile and the sampler share the arena, not concurrent layer intermediates. The
    // arena is sized for the 64-wide sampler, which serves requests with top_k above 20.
    plan.workspace = std::max<std::uint64_t>(plan.workspace,
        gemma4_logits_workspace_bytes(geometry.output_head, options.causal_scoring ? kScoreTile : 1));
    plan.workspace = std::max<std::uint64_t>(plan.workspace,
        gemma4_logits_workspace_bytes(geometry.output_head, 1) +
        ops::sampling_workspace_capacity_bytes(static_cast<int>(public_tokens), 1, 1, ops::kSamplingMaxTopK));
    plan.workspace = aligned(plan.workspace + 4096);
    plan.prepared_row_scales = geometry.derived_device_bytes;
    // The affine capacity curve starts where all sliding rings have reached their fixed size.
    // Below that point both the global payload and every ring grow, so it is not one slope.
    const auto minimum_pages = sliding_page_bytes ? rings : 1U;
    if (plan.kv_pages < minimum_pages) {
        throw std::invalid_argument("Gemma4 KV capacity is below the sliding-ring capacity curve minimum");
    }
    plan.kv = global_page_bytes * plan.kv_pages + sliding_page_bytes * rings + 8ULL * plan.kv_pages;
    plan.kv_max = global_page_bytes * maximum_pages + sliding_page_bytes * rings + 8ULL * maximum_pages;
    plan.io = io_bytes(c, options.prefill_chunk, options.causal_scoring ? kScoreTile : 1);
    if (options.use_cuda_graph) {
        plan.graph_bound = 1;
        for (const auto width : kGraphWidths) {
            if (width <= options.prefill_chunk && width <= std::min(options.max_context, plan.kv_pages * kPage)) {
                ++plan.graph_bound;
            }
        }
        plan.graph_allowance = kGraphAllowance;
    }
    plan.curve.main_page_tokens = kPage;
    plan.curve.minimum_main_page_groups = minimum_pages;
    plan.curve.maximum_main_page_groups = maximum_pages;
    plan.curve.bytes_per_additional_main_page_group = static_cast<std::size_t>(global_page_bytes + 8);
    const auto without_global_tables = plan.fixed_bytes() - (global_page_bytes + 8) * plan.kv_pages;
    plan.curve.minimum_device_reservation_bytes = static_cast<std::size_t>(
        without_global_tables + (global_page_bytes + 8) * minimum_pages);
    return plan;
}

namespace detail {
struct RequestPlan {
    const void* owner = nullptr;
    runtime::RequestPlanSummary summary;
    std::vector<TokenId> tokens;
    ResolvedSamplingParameters sampling;
    std::vector<TokenId> readout_tokens;
    // The prefix cache serves this request (the cache exists and the request allows reuse), its
    // chained block hashes over `tokens` and the Frontend's tap hints.
    bool reuse = false;
    std::vector<std::uint64_t> block_hashes;
    std::vector<pc::TapHint> hints;
};
struct ResumeData {
    const void* owner = nullptr;
    std::shared_ptr<const RequestPlan> plan;
    std::vector<TokenId> ledger;
    std::uint32_t frontier = 0;
    bool prefilled = false;
    bool finishable = false;
    GenerationTimings timings;
    // The Begin the Engine committed at the request's first admission: it never changes, so a
    // resumed binding reports it again whatever source the resume restored from.
    runtime::BeginSummary begin;
};
struct ProgramImpl {
    const execution::Parameters& parameters;
    DeviceContext& device;
    ProgramOptions options;
    ProgramDevicePlan plan;
    cudaStream_t stream;
    ops::Fp8RowCublasContext gemm;
    CudaEventTimer prefill_timer;
    DeviceArena workspace;
    DeviceArena io;
    DeviceBuffer global_table_storage, ring_table_storage;
    Tensor global_table, ring_table;
    std::vector<DeviceBuffer> keys, values, key_scales, value_scales;
    std::vector<Tensor> key_views, value_views, key_scale_views, value_scale_views;
    std::uint64_t global_payload_bytes = 0, sliding_payload_bytes = 0;
    std::vector<Gemma4CacheLayer> caches;
    Tensor ids, positions, hidden, head, counts, mask, step, distribution_ids, distribution_values;
    ops::SamplingConfig* sampling_device = nullptr;
    ops::SamplingConfig sampling_host;
    PinnedHostBuffer host_step{4 * sizeof(std::int32_t)};
    std::vector<std::int32_t> host_positions;
    std::vector<std::int32_t> host_counts;
    std::vector<std::uint32_t> host_mask;
    std::vector<std::uint16_t> host_logits;
    DecodeGraphDefinition decode_definition;
    DecodeGraphExecutable decode_graph;
    struct PrefillGraph { DecodeGraphDefinition definition; DecodeGraphExecutable executable; };
    std::array<PrefillGraph, kGraphWidths.size()> prefill_graphs;
    std::size_t graph_measured_bytes = 0;
    // Sampling pipeline width: the 20-wide path serves requests with top_k <= 20 (every request the
    // fast path served before), the 64-wide path the rest. The decode graph embeds the sampler, so a
    // change recaptures it (select_sampler).
    std::int32_t sampler_capacity = ops::kSamplingFastTopK;

    std::shared_ptr<const RequestPlan> request;
    std::unique_ptr<ResumeData> binding_resume;
    std::optional<ResumeState> paused;
    bool binding = false, occupied = false, prefilled = false, replaying = false, finishable = false;
    bool pending = false, state_valid = true;
    std::uint64_t epoch = 1, transaction = 0;
    // The ledger is only committed input/output. fed is the valid target-state frontier. Physical
    // writes can run ahead on failure; they never license a ledger entry or cache publication.
    std::vector<TokenId> ledger;
    std::uint32_t fed = 0, physical_written = 0, replay_target = 0;
    TokenId pending_token = 0;
    runtime::PendingRowFailure pending_failure = runtime::PendingRowFailure::None;
    bool pending_decode = false;
    GenerationTimings timings;
    // The Begin the Engine committed for this binding (its prompt, the source it quoted and that
    // source's path): every prefill progress reports it back unchanged.
    runtime::BeginSummary begin;

    // ---- prefix cache ---------------------------------------------------------------------------
    // The snapshot a binding resumes from (its `path` is the k = frontier / 64 full blocks).
    struct PrefixSelection {
        std::optional<pc::SnapshotRef> snapshot;
        pc::SnapshotKind kind = pc::SnapshotKind::Tap;
        std::uint32_t frontier = 0;
        std::vector<pc::NodeRef> path;
    };
    struct PrefixCandidates {
        std::vector<PrefixSelection> list;     // preference order, the root excluded
        std::uint32_t cached_tokens = 0;       // the prompt prefix held as cached blocks
        std::vector<std::uint32_t> existing;   // frontiers of the usable snapshots on the matched path
    };
    // What the live sequence owns in the cache. Every pin a sequence holds is released in
    // release_prefix, exactly once, whichever way the sequence ends.
    struct PrefixLane {
        bool active = false;
        std::vector<pc::NodeRef> path;                   // pinned: matched blocks, then published or found ones
        std::vector<std::uint64_t> hashes;               // chained over the ledger, extended as blocks publish
        std::vector<pc::TapHint> hints;
        std::vector<pc::PlannedTap> taps;                // not yet realized
        std::vector<std::uint32_t> call_ends;            // exclusive prefill call ends, ascending
        std::size_t next_call = 0;
        std::optional<pc::SnapshotRef> restore_snapshot; // selected at binding, restored by poll_context
        std::optional<pc::SnapshotRef> pinned_snapshot;  // the selection's pin, until the restore holds its own
        std::optional<pc::SnapshotRef> resume;           // the snapshot the sequence was restored from
        std::optional<pc::SnapshotRef> superseded;       // the snapshot its first capture superseded
        std::uint32_t reused = 0;                        // the resume frontier
        std::uint32_t deepest = 0;                       // the deepest snapshot frontier of this lineage
        // The end of the prefix computed only by full, grid-aligned prefill calls ([kC,(k+1)C) for chunk C), from
        // the root or from a restore at a grid frontier: the arithmetic a cold prefill of any longer prompt
        // applies to those positions, bit for bit. Only blocks and snapshots below it are published, so a
        // restore reproduces a cold prefill exactly; a partial final call, decode, forced feeds or an
        // off-grid replay never reach the cache (plan §8.4, §17.6 grid-aligned policy).
        std::uint32_t exact_through = 0;
        std::vector<pc::NodeRef> owned;                  // blocks whose Device copy this sequence inserted
        std::vector<std::uint64_t> captures;             // every capture it asked for
        bool captured = false;
        bool withdraw = false;                           // its state may be wrong: take back what it published
        bool close_endpoint = true;                      // the next turn may resume at this one's endpoint
    } pl;
    pc::Withdrawal prefix_withdrawal;                    // entries a pin kept from an earlier withdrawal
    std::uint64_t next_capture = 0;
    bool clear_prefix_after_release = false;
    std::uint32_t ring_pages_ = 0;
    // Last: destroyed first, waiting for its copies before the Device buffers it reads are freed.
    std::unique_ptr<prefix::PrefixCache> prefix_cache;

    ProgramImpl(const execution::Parameters& p, DeviceContext& d, ProgramOptions o)
        : parameters(p), device(d), options(std::move(o)),
          plan(Program::plan_device(p.model, options, p.model.config().text.vocab_size)),
          stream(d.stream), gemm(stream), prefill_timer(d), workspace(static_cast<std::size_t>(plan.workspace)),
          io(static_cast<std::size_t>(plan.io)),
          global_table_storage(plan.kv_pages * sizeof(std::int32_t)),
          ring_table_storage(plan.kv_pages * sizeof(std::int32_t)) {
        const auto& c = config();
        const int head_columns = options.causal_scoring ? kScoreTile : 1;
        ids = io.alloc(DType::I32, {static_cast<int>(options.prefill_chunk)});
        positions = io.alloc(DType::I32, {static_cast<int>(options.prefill_chunk)});
        hidden = io.alloc(DType::BF16, {static_cast<int>(c.hidden_size), static_cast<int>(options.prefill_chunk)});
        head = io.alloc(DType::BF16, {static_cast<int>(c.vocab_size), head_columns});
        sampling_device = static_cast<ops::SamplingConfig*>(io.alloc_bytes(sizeof(ops::SamplingConfig)).data);
        counts = io.alloc(DType::I32, {static_cast<int>(c.vocab_size)});
        mask = io.alloc(DType::I32, {static_cast<int>((c.vocab_size + 31) / 32)});
        step = io.alloc(DType::I32, {4});
        distribution_ids = io.alloc(DType::I32, {static_cast<int>(kMaximumScoreTopTokens), head_columns});
        distribution_values = io.alloc(DType::FP32, {static_cast<int>(kMaximumScoreTopTokens), head_columns});
        host_positions.resize(options.prefill_chunk);
        host_counts.resize(c.vocab_size);
        host_mask.resize((c.vocab_size + 31) / 32);
        if (options.causal_scoring) { host_logits.resize(static_cast<std::size_t>(c.vocab_size)); }
        ledger.reserve(capacity() + 1ULL);
        const auto ring_count = std::min(plan.kv_pages, ring_pages(options.prefill_chunk, c.sliding_window));
        ring_pages_ = static_cast<std::uint32_t>(ring_count);
        std::vector<std::int32_t> table(plan.kv_pages);
        std::iota(table.begin(), table.end(), 0);
        global_table_storage.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        for (auto& page : table) { page %= static_cast<int>(ring_count); }
        ring_table_storage.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        global_table = Tensor(global_table_storage.p, DType::I32, {static_cast<int>(plan.kv_pages)});
        ring_table = Tensor(ring_table_storage.p, DType::I32, {static_cast<int>(plan.kv_pages)});
        const auto layers = p.layers.size();
        for (auto* storage : {&keys, &values, &key_scales, &value_scales}) { storage->reserve(layers); }
        for (auto* views : {&key_views, &value_views, &key_scale_views, &value_scale_views}) { views->reserve(layers); }
        caches.reserve(layers);
        for (const auto& layer : p.layers) {
            const auto pages = layer.global_layer ? plan.kv_pages : ring_count;
            const auto layout = kv_layer_layout(options.kv_cache, layer.global_layer, layer.head_dim, layer.kv_heads);
            const auto code_bytes = layout.code_page_bytes * pages;
            (layer.global_layer ? global_payload_bytes : sliding_payload_bytes) += layout.page_bytes() * pages;
            // Never-written rows must read as finite zeros (code 0, scale 0), including after recovery.
            keys.emplace_back(code_bytes); values.emplace_back(code_bytes);
            keys.back().fill(0); values.back().fill(0);
            const std::initializer_list<std::int32_t> shape{
                layer.head_dim, static_cast<int>(kPage), layer.kv_heads, static_cast<int>(pages)};
            key_views.emplace_back(keys.back().p, layout.code_dtype, shape);
            value_views.emplace_back(values.back().p, layout.code_dtype, shape);
            key_scales.emplace_back(); value_scales.emplace_back();
            key_scale_views.emplace_back(); value_scale_views.emplace_back();
            if (layout.scale_groups != 0) {
                const auto scale_bytes = layout.scale_page_bytes * pages;
                key_scales.back() = DeviceBuffer(scale_bytes); value_scales.back() = DeviceBuffer(scale_bytes);
                key_scales.back().fill(0); value_scales.back().fill(0);
                const std::initializer_list<std::int32_t> scale_shape{
                    layout.scale_groups, static_cast<int>(kPage), layer.kv_heads, static_cast<int>(pages)};
                key_scale_views.back() = Tensor(key_scales.back().p, DType::FP16, scale_shape);
                value_scale_views.back() = Tensor(value_scales.back().p, DType::FP16, scale_shape);
            }
        }
        // Every reserve() above is complete; the descriptors now point at stable Tensor storage.
        for (std::size_t i = 0; i < layers; ++i) {
            const bool global = p.layers[i].global_layer;
            const bool scaled = key_scale_views[i].data != nullptr;
            caches.push_back({.format = scaled ? ops::Gemma4KvFormat::Fp8E4m3Row256 : ops::Gemma4KvFormat::BFloat16,
                .k_pages = &key_views[i], .v_pages = &value_views[i],
                .k_scales = scaled ? &key_scale_views[i] : nullptr,
                .v_scales = scaled ? &value_scale_views[i] : nullptr,
                .block_table = global ? &global_table : &ring_table,
                // Sliding descriptors retain BF16 QK.
                .qk_mode = global ? options.qk_mode : ops::Gemma4QkMode::BFloat16});
        }
        if (options.prefix_cache) { create_prefix_cache(); }
        if (p.derived_device_bytes() != plan.prepared_row_scales) {
            throw std::logic_error("Gemma4 Parameters and device plan disagree on prepared row scales");
        }
        try {
            if (options.use_cuda_graph) {
                StartupPhaseScope graph_phase(options.startup_observer, StartupPhase::CudaGraphPrepare);
                const auto free_before = device.free_bytes();
                prepare_graphs();
                const auto free_after = device.free_bytes();
                graph_measured_bytes = free_before > free_after ? free_before - free_after : 0;
                if (graph_measured_bytes > plan.graph_allowance) {
                    throw std::runtime_error("Gemma4 CUDA graph preparation exceeded its planned device allowance; reduce prefill_chunk or disable CUDA graphs");
                }
                graph_phase.complete();
            }
        }
        catch (...) { (void)settle_after_memory_exhausted(); throw; }
    }
    const TextConfig& config() const noexcept { return parameters.model.config().text; }
    std::uint32_t capacity() const noexcept { return std::min(options.max_context, plan.kv_pages * kPage); }
    SequenceHandle handle() const noexcept {
        SequenceHandle h; h.owner_ = this; h.epoch_ = epoch; return h;
    }
    bool owns(SequenceHandle h) const noexcept {
        return occupied && h.owner_ == this && h.lane_ == 0 && h.epoch_ == epoch;
    }
    bool owns(const PendingBatch& batch) const noexcept {
        return pending && batch.owner_ == this && batch.transaction_ == transaction && owns(batch.sequence_);
    }
    void require(SequenceHandle sequence) const {
        if (!owns(sequence) || !state_valid) { throw std::logic_error("Gemma4 stale or invalid sequence"); }
    }
    void clear() noexcept {
        // A sequence that ends with its state unknown (an exception left it invalid) takes back
        // what it published; every pin it holds goes back before the request itself is dropped.
        release_prefix(!state_valid);
        binding = occupied = prefilled = replaying = finishable = pending = false;
        state_valid = true;
        fed = physical_written = replay_target = 0;
        ledger.clear(); request.reset(); binding_resume.reset(); paused.reset();
        pending_failure = runtime::PendingRowFailure::None;
        timings = {};
        ++epoch; ++transaction;
    }
    Tensor hidden_columns(int first, int n) const { return hidden.slice(1, first, n); }
    Tensor logits(int n) const { return head.slice(1, 0, n); }
    void layers_body(std::uint32_t width) {
        workspace.reset();
        auto input = ids.slice(0, 0, static_cast<int>(width));
        auto pos = positions.slice(0, 0, static_cast<int>(width));
        auto out = hidden_columns(0, static_cast<int>(width));
        gemma4_layers(input, out, parameters.embedding, parameters.layers, caches, pos,
            gemm, workspace, stream);
    }
    void head_body(int first, int n, int output_first = 0) {
        workspace.reset();
        auto out = head.slice(1, output_first, n);
        gemma4_logits(hidden_columns(first, n), out, parameters.final_norm, parameters.output_head,
            config().final_logit_softcapping, config().rms_norm_eps, gemm, workspace, stream);
    }
    void sample_body(std::int32_t purpose) {
        auto out = step.slice(0, 0, 1);
        auto logical = step.slice(0, 1, 1);
        auto values = logits(1);
        ops::sample(values, out, static_cast<int>(config().vocab_size), sampling_device,
            logical, purpose, workspace, stream, sampler_capacity);
    }
    void decode_body() {
        layers_body(1);
        head_body(0, 1);
        sample_body(ops::kSamplePurposeDecode);
    }
    void prepare_graphs() {
        // All graphs are instantiated/uploaded at startup. Ordinary units only replay them.
        // Warm eager calls settle cuBLAS and module initialization before stream capture.
        CUDA_CHECK(cudaMemsetAsync(io.base(), 0, io.capacity(), stream));
        sampling_host.nonfinite = static_cast<std::int32_t*>(step.data) + 2;
        upload_pinned(sampling_device, &sampling_host, sizeof(sampling_host), stream);
        const auto width = std::min(options.prefill_chunk, capacity());
        std::iota(host_positions.begin(), host_positions.end(), 0);
        upload_pinned(positions.data, host_positions.data(), width * sizeof(std::int32_t), stream);
        layers_body(width);
        head_body(0, std::min<int>(width, options.causal_scoring ? kScoreTile : 1));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Decode uses position zero during preparation; no live state exists yet.
        CUDA_CHECK(cudaMemsetAsync(positions.data, 0, sizeof(std::int32_t), stream));
        decode_body();
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Settle the wide sampler's module load eagerly: a later recapture of the decode graph for
        // a top_k above 20 must not meet a first kernel launch inside stream capture.
        sampler_capacity = ops::kSamplingMaxTopK;
        sample_body(ops::kSamplePurposeDecode);
        sampler_capacity = ops::kSamplingFastTopK;
        CUDA_CHECK(cudaStreamSynchronize(stream));
        decode_definition.capture(stream, [this] { decode_body(); });
        decode_graph.instantiate(decode_definition);
        decode_graph.upload(stream);
        decode_graph.launch(stream);
        for (std::size_t i = 0; i < kGraphWidths.size(); ++i) {
            if (kGraphWidths[i] > options.prefill_chunk || kGraphWidths[i] > capacity()) { continue; }
            auto& graph = prefill_graphs[i];
            graph.definition.capture(stream, [this, i] { layers_body(kGraphWidths[i]); });
            graph.executable.instantiate(graph.definition);
            graph.executable.upload(stream);
            graph.executable.launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        workspace.reset();
    }
    void select_sampler(const ResolvedSamplingParameters& s) {
        const std::int32_t needed =
            s.top_k > ops::kSamplingFastTopK ? ops::kSamplingMaxTopK : ops::kSamplingFastTopK;
        if (needed == sampler_capacity) { return; }
        sampler_capacity = needed;
        if (!options.use_cuda_graph) { return; }
        // Earlier work on the stream finishes before the old executable is released.
        CUDA_CHECK(cudaStreamSynchronize(stream));
        decode_graph.reset();
        decode_definition.reset();
        decode_definition.capture(stream, [this] { decode_body(); });
        decode_graph.instantiate(decode_definition);
        decode_graph.upload(stream);
    }
    void upload_sampler() {
        const auto& s = request->sampling;
        select_sampler(s);
        sampling_host = {};
        sampling_host.temperature = s.temperature; sampling_host.top_k = s.top_k;
        sampling_host.top_p = s.top_p; sampling_host.min_p = s.min_p;
        sampling_host.presence_penalty = s.presence_penalty;
        sampling_host.frequency_penalty = s.frequency_penalty; sampling_host.seed = s.seed;
        sampling_host.nonfinite = static_cast<std::int32_t*>(step.data) + 2;
        std::fill(host_counts.begin(), host_counts.end(), 0);
        for (std::size_t i = request->tokens.size(); i < ledger.size(); ++i) { ++host_counts[ledger[i]]; }
        upload_pinned(counts.data, host_counts.data(), counts.bytes(), stream);
        if (s.presence_penalty != 0 || s.frequency_penalty != 0) {
            sampling_host.token_counts = static_cast<std::int32_t*>(counts.data);
        }
        upload_pinned(sampling_device, &sampling_host, sizeof(sampling_host), stream);
    }
    void synchronize(runtime::ExecutionTimingRecorder* timing = nullptr) {
        if (timing) { timing->begin_wait(); }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (timing) { timing->end_wait(); }
    }
    runtime::ExecutionTiming charge(runtime::ExecutionTiming timing, bool decode_round) noexcept {
        const double seconds = static_cast<double>(timing.elapsed_ns()) * 1e-9;
        if (decode_round) {
            timings.decode_seconds += seconds;
            timings.decode_share_seconds += seconds; // This Program has one lane.
        } else { timings.prefill_seconds += seconds; }
        return timing;
    }
    void diagnostic(const std::string& text, DiagnosticLevel level = DiagnosticLevel::Info) const noexcept {
        try {
            if (options.diagnostics.callback) {
                options.diagnostics.callback(Diagnostic{.level = level, .message = text});
            } else {
                std::fprintf(stderr, "[engine] %s\n", text.c_str());
            }
        } catch (...) {}
    }

    // ---- prefix cache: construction -----------------------------------------------------------
    void create_prefix_cache() {
        const auto& c = config();
        std::vector<KvLayerFact> facts;
        facts.reserve(parameters.layers.size());
        for (const auto& layer : parameters.layers) {
            facts.push_back({layer.global_layer, layer.head_dim, layer.kv_heads});
        }
        const auto geometry = prefix_cache_geometry(facts, c.sliding_window, options.kv_cache);
        const auto identity = prefix::make_layout_identity(facts, c.sliding_window, options.prefill_chunk,
                                                           options.kv_cache, options.qk_mode);
        std::vector<std::vector<prefix::DevicePlane>> global, sliding;
        for (std::size_t i = 0; i < parameters.layers.size(); ++i) {
            const auto& layer = parameters.layers[i];
            const auto layout = kv_layer_layout(options.kv_cache, layer.global_layer, layer.head_dim, layer.kv_heads);
            // KvLayerLayout::planes() order: K codes, K scales (when scaled), V codes, V scales.
            std::vector<prefix::DevicePlane> planes;
            planes.push_back({static_cast<std::byte*>(keys[i].p), layout.code_page_bytes});
            if (layout.scale_page_bytes != 0) {
                planes.push_back({static_cast<std::byte*>(key_scales[i].p), layout.scale_page_bytes});
            }
            planes.push_back({static_cast<std::byte*>(values[i].p), layout.code_page_bytes});
            if (layout.scale_page_bytes != 0) {
                planes.push_back({static_cast<std::byte*>(value_scales[i].p), layout.scale_page_bytes});
            }
            (layer.global_layer ? global : sliding).push_back(std::move(planes));
        }
        prefix::PrefixCacheConfig cache;
        cache.kv_pages = plan.kv_pages;
        cache.taps = prefix::derive_tap_config(geometry.image_bytes);
        cache.cost.chunk_tokens = options.prefill_chunk;
        cache.cost.token_seconds = kCacheTokenSeconds;
        prefix_cache = std::make_unique<prefix::PrefixCache>(
            cache, geometry, identity, prefix::plan_block_layout(facts, options.kv_cache),
            prefix::plan_state_image(facts, c.sliding_window, options.kv_cache), std::move(global),
            std::move(sliding), ring_pages_, options.host_bytes, stream);
    }

    // ---- prefix cache: admission --------------------------------------------------------------
    // The usable snapshots of `tokens` in preference order (the index's choice first; a resumed
    // request takes its deepest), none deeper than `maximum_frontier`. A candidate is usable when
    // its blocks are on the Host (none still being written), it is not on an SSD tier and its
    // image header validates against this process's layout identity.
    PrefixCandidates prefix_candidates(std::span<const TokenId> tokens, std::span<const std::uint64_t> hashes,
                                       std::uint32_t maximum_frontier, bool resumed) {
        PrefixCandidates out;
        const auto n = static_cast<std::uint32_t>(tokens.size());
        if (!prefix_cache || n <= 1 || hashes.size() != n / pc::kBlockTokens) { return out; }
        pc::PrefixCacheIndex& index = prefix_cache->index();
        prefix_cache->poll();
        // With no sequence mapping the cache, a Device copy is a leftover (a write refused its
        // block): none is worth keeping.
        if (!occupied && !binding) { prefix_cache->sweep_device_copies(); }
        pc::MatchResult match = index.match(tokens, hashes, {}, n);
        const auto filling = [](const pc::MatchResult& m) {
            return std::any_of(m.candidates.begin(), m.candidates.end(),
                               [](const pc::MatchCandidate& c) { return c.filling_blocks != 0; });
        };
        if (prefix_cache->transfers_pending() && filling(match)) {
            // Blocks being written land within a copy time: a cached source is not given up for them.
            prefix_cache->drain();
            match = index.match(tokens, hashes, {}, n);
        }
        out.cached_tokens = static_cast<std::uint32_t>(match.path.size()) * pc::kBlockTokens;
        // Only a grid frontier restores to a cold prefill's state (PrefixLane::exact_through); a snapshot
        // elsewhere is skipped.
        std::erase_if(match.candidates, [&](const pc::MatchCandidate& c) {
            return c.filling_blocks != 0 || c.ssd_only_blocks != 0 || c.image_on_ssd_only || c.tail_on_ssd_only ||
                   c.frontier > maximum_frontier || c.frontier % options.prefill_chunk != 0 ||
                   !prefix_cache->image_valid(c.snapshot);
        });
        for (const pc::MatchCandidate& c : match.candidates) { out.existing.push_back(c.frontier); }
        const pc::AdmissionChoice choice = index.choose(match, n);
        std::vector<std::size_t> order;
        if (choice.candidate) { order.push_back(*choice.candidate); }
        for (std::size_t i = 0; i < match.candidates.size(); ++i) {
            if (!choice.candidate || i != *choice.candidate) { order.push_back(i); }
        }
        // A resumed request takes its deepest state: a shallower one would replay the tokens past
        // it with prefill arithmetic.
        if (resumed) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return match.candidates[a].frontier > match.candidates[b].frontier;
            });
        }
        for (const std::size_t i : order) {
            const pc::MatchCandidate& c = match.candidates[i];
            PrefixSelection s;
            s.snapshot = c.snapshot;
            s.kind = index.snapshot(c.snapshot).kind;
            s.frontier = c.frontier;
            s.path.assign(match.path.begin(), match.path.begin() + c.path_blocks);
            out.list.push_back(std::move(s));
        }
        return out;
    }

    // The prompt's prefill calls from the grid frontier `frontier`: the plain chunk grid, as a cold
    // prefill runs. The taps the planner places (the generation opener moves forward unless the
    // lineage echoes its replies) are realized at the first full-call boundary at or past them; none
    // cuts a call, since a narrower call changes the arithmetic of the positions it covers.
    void plan_prefill(std::uint32_t frontier, std::span<const std::uint32_t> existing, bool follow_endpoint) {
        const std::uint32_t n = request->summary.prompt_tokens;
        pl.taps = pc::plan_taps(n, frontier, pl.hints, existing, {}, prefix_cache->taps(), !follow_endpoint);
        pl.call_ends = prefix::plan_call_ends(frontier, n, options.prefill_chunk, {});
        pl.next_call = 0;
    }

    // Records a prefill or replay call over [first, first + width): a full call on the chunk grid that
    // continues the exact prefix extends it.
    void note_grid_call(std::uint32_t first, std::uint32_t width) noexcept {
        if (first == pl.exact_through && first % options.prefill_chunk == 0 && width == options.prefill_chunk) {
            pl.exact_through = first + width;
        }
    }

    // Exclusive end of the prefill call that starts at `fed`. advance_prefill takes every prefill call's
    // end from here (a cache-off or non-reusing request has no planned calls and ends at the prompt, then
    // chunk-capped there).
    std::uint32_t next_call_end() {
        const std::uint32_t prompt = request->summary.prompt_tokens;
        while (pl.next_call < pl.call_ends.size() && pl.call_ends[pl.next_call] <= fed) { ++pl.next_call; }
        return pl.next_call < pl.call_ends.size() ? pl.call_ends[pl.next_call] : prompt;
    }

    // ---- prefix cache: publication ------------------------------------------------------------
    // Publishes the full blocks below `frontier` and below the exact prefix (PrefixLane::exact_through):
    // positions whose input token is in the committed ledger and whose KV full grid prefill calls wrote.
    // Nothing past `fed` is ever published, so a rejected or uncommitted write reaches no block and no
    // snapshot (plan §8.4 invariant 1). Each new block is written through to the Host at once; a block
    // already cached stays as it is.
    void publish_blocks(std::uint32_t frontier) {
        if (!prefix_cache || !pl.active) { return; }
        frontier = std::min({frontier, pl.exact_through, static_cast<std::uint32_t>(ledger.size())});
        const std::uint32_t blocks = frontier / pc::kBlockTokens;
        std::vector<pc::NodeRef> fresh;
        std::vector<std::uint32_t> pages;
        while (pl.path.size() < blocks) {
            const auto b = static_cast<std::uint32_t>(pl.path.size());
            const std::span<const TokenId> tokens(ledger.data() + static_cast<std::size_t>(b) * pc::kBlockTokens,
                                                  pc::kBlockTokens);
            while (pl.hashes.size() <= b) {
                const std::size_t at = pl.hashes.size();
                const std::uint64_t previous = at == 0 ? pc::kRootLookupHash : pl.hashes[at - 1];
                pl.hashes.push_back(pc::block_lookup_hash(
                    previous, std::span<const TokenId>(ledger.data() + at * pc::kBlockTokens, pc::kBlockTokens), 0));
            }
            const pc::NodeRef parent = b == 0 ? pc::NodeRef{} : pl.path.back();
            const pc::InsertResult result = prefix_cache->insert_block(parent, pl.hashes[b], tokens, b);
            pl.path.push_back(result.node);
            if (result.device_attached) {
                pl.owned.push_back(result.node);
                fresh.push_back(result.node);
                pages.push_back(b);
            }
        }
        prefix_cache->write_blocks(fresh, pages);
    }

    // A snapshot of the sequence's state at its frontier `fed`, Host-born; only at the end of the exact
    // prefix, the one state a cold prefill reaches too.
    void prefix_capture(pc::SnapshotKind kind, bool resume_point) {
        const std::uint32_t F = fed;
        const std::uint32_t k = F / pc::kBlockTokens;
        if (F == 0 || F != pl.exact_through || pl.path.size() < k || F > ledger.size()) { return; }
        pc::PrefixCacheIndex& index = prefix_cache->index();
        prefix::PrefixCache::Capture c;
        c.id = ++next_capture;
        c.anchor = k == 0 ? pc::NodeRef{} : pl.path[k - 1];
        c.frontier = F;
        c.tail.assign(ledger.begin() + static_cast<std::ptrdiff_t>(k) * pc::kBlockTokens,
                      ledger.begin() + static_cast<std::ptrdiff_t>(F));
        c.tail_page = k;
        c.kind = kind;
        // A paused request's frontier is its own resume point: its image is admitted over any
        // retained snapshot that is not pinned. The claim only admits it.
        c.claim = resume_point ? std::numeric_limits<double>::infinity()
                               : index.estimate_priority(pl.deepest, F, !c.tail.empty());
        c.meta.opener = kind != pc::SnapshotKind::Endpoint &&
                        std::any_of(pl.hints.begin(), pl.hints.end(), [&](const pc::TapHint& hint) {
                            return hint.kind == pc::TapHintKind::GenerationOpener && hint.position == F;
                        });
        c.meta.lineage_echo = pl.resume && index.valid(*pl.resume) &&
                              index.snapshot(*pl.resume).kind == pc::SnapshotKind::Endpoint;
        // The request's first capture supersedes the snapshot it resumed from, which then serves
        // only requests diverging before this one.
        const bool supersede = !pl.captured && pl.resume && index.valid(*pl.resume);
        if (supersede) {
            c.supersedes = *pl.resume;
            const std::uint32_t from = index.snapshot(*pl.resume).frontier;
            if (from < F) { c.continuation.assign(ledger.begin() + from, ledger.begin() + F); }
        }
        const std::uint64_t id = c.id;
        if (prefix_cache->capture(std::move(c))) {
            pl.captures.push_back(id);
            pl.captured = true;
            pl.deepest = F;
            if (supersede) { pl.superseded = pl.resume; }
        }
    }

    // Whether a snapshot at `boundary` still pays given the deepest one of the lineage.
    [[nodiscard]] bool tap_far_enough(std::uint32_t boundary, bool forward) const noexcept {
        return boundary >= pl.deepest + pc::kBlockTokens || (forward && boundary > pl.deepest);
    }

    // The planned taps due at call boundary `boundary`: an exact tap ends a call; a flexible one is
    // realized at the first boundary at or past it, or at the start of the prompt's final call.
    bool take_due_taps(std::uint32_t boundary, bool final_next, bool& is_boundary, bool& forward) {
        bool due = false;
        std::erase_if(pl.taps, [&](const pc::PlannedTap& tap) {
            if (tap.position > boundary && !(final_next && tap.placement == pc::TapPlacement::Flexible)) {
                return false;
            }
            due = true;
            is_boundary = is_boundary || tap.boundary;
            forward = forward || tap.forward;
            return true;
        });
        return due;
    }

    // After a prefill call ending at `fed`: publish its blocks and realize the taps due here. The
    // prompt's deepest grid state is always offered as a tap, so a next turn resumes there and replays
    // at most one chunk of this prompt: the end of the prompt when it ends on the grid, else the start
    // of its final (partial) call.
    void after_prefill_call() {
        if (!pl.active) { return; }
        publish_blocks(fed);
        const std::uint32_t prompt = request->summary.prompt_tokens;
        bool is_boundary = false, forward = false;
        if (fed >= prompt) {
            if (fed == pl.exact_through && tap_far_enough(fed, true)) { prefix_capture(pc::SnapshotKind::Tap, false); }
            return;
        }
        const bool final_next = next_call_end() == prompt;
        const bool deepest = final_next && prompt % options.prefill_chunk != 0;
        if (deepest) { forward = true; }
        if ((take_due_taps(fed, final_next, is_boundary, forward) || deepest) && tap_far_enough(fed, forward)) {
            prefix_capture(is_boundary ? pc::SnapshotKind::Boundary : pc::SnapshotKind::Tap, false);
        }
    }

    // The end of a consistent sequence (finish, abort, pause): its remaining blocks and an endpoint
    // snapshot at the committed frontier when that is the exact prefix's end (a pausing sequence past it
    // resumes from its last grid snapshot and replays the rest on the chunk grid).
    void prefix_finish(bool pausing) noexcept {
        if (!prefix_cache || !pl.active || !state_valid) { return; }
        try {
            publish_blocks(fed);
            const std::uint32_t F = fed;
            const bool far_enough = F >= pl.deepest + pc::kBlockTokens || (pausing && F > pl.deepest) ||
                                    (pl.close_endpoint && F >= pl.deepest + pc::kMinimumForwardTokens);
            if (far_enough && pl.path.size() >= F / pc::kBlockTokens) {
                prefix_capture(pc::SnapshotKind::Endpoint, pausing);
            }
        } catch (const std::exception& error) {
            diagnostic(std::string("Gemma4: publishing the prefix cache failed: ") + error.what(),
                       DiagnosticLevel::Warning);
            pl.withdraw = true;
            clear_prefix_after_release = true;
        } catch (...) {
            pl.withdraw = true;
            clear_prefix_after_release = true;
        }
    }

    // The prompt tokens past the reused prefix that this request's accepted captures made
    // reusable for a later request (Anthropic's cache_creation_input_tokens).
    [[nodiscard]] std::optional<std::uint32_t> prefix_created_tokens() const noexcept {
        if (!prefix_cache || !pl.active) { return std::nullopt; }
        const std::uint32_t deepest = std::min(pl.deepest, request->summary.prompt_tokens);
        return deepest > pl.reused ? deepest - pl.reused : 0U;
    }

    // ---- prefix cache: release ----------------------------------------------------------------
    // After a non-finite failure or an invalid state, once the pins are released
    // (hybrid-prefix-cache-spec §7.8): the sequence's captures land, then its blocks (with what
    // later requests built on them) and its snapshots leave the cache, so a resend resumes from
    // what was cached before it. What another pin still holds waits in prefix_withdrawal.
    void prefix_withdraw() noexcept {
        try {
            prefix_cache->drain(); // each capture pins its anchor until it has landed and published
            for (const std::uint64_t id : pl.captures) {
                if (const auto result = prefix_cache->capture_result(id); result && result->created) {
                    prefix_withdrawal.snapshots.push_back(result->snapshot);
                }
                prefix_cache->forget_capture(id);
            }
            const bool published = !pl.captures.empty() || !pl.owned.empty();
            prefix_withdrawal.nodes.insert(prefix_withdrawal.nodes.end(), pl.owned.begin(), pl.owned.end());
            const std::uint32_t removed = prefix_cache->index().withdraw(prefix_withdrawal);
            // The snapshot its first capture superseded serves again.
            if (pl.superseded && prefix_cache->index().valid(*pl.superseded)) {
                prefix_cache->index().unsupersede(*pl.superseded);
            }
            if (published) {
                diagnostic("Gemma4: withdrew " + std::to_string(removed) +
                               " prefix-cache entries a failed request published",
                           DiagnosticLevel::Warning);
            }
        } catch (...) {
            // What it published cannot be told apart now: the cache is emptied.
            clear_prefix_after_release = true;
        }
    }

    // Gives back everything the sequence holds in the cache: the selection's pins, the mapped
    // path, its Device copies (the pages go with the sequence; their bytes are on the Host) and,
    // for a failed sequence, what it published.
    void release_prefix(bool invalid) noexcept {
        if (!prefix_cache) { return; }
        if (invalid && pl.active) { pl.withdraw = true; }
        pc::PrefixCacheIndex& index = prefix_cache->index();
        try {
            if (pl.pinned_snapshot) { index.unpin_snapshot(*pl.pinned_snapshot); }
            if (!pl.path.empty()) { index.release_path(pl.path); }
        } catch (...) { clear_prefix_after_release = true; }
        try {
            if (pl.withdraw) {
                prefix_withdraw();
            } else {
                for (const std::uint64_t id : pl.captures) { prefix_cache->forget_capture(id); }
                // Entries an earlier withdrawal found pinned go once nothing pins them.
                if (!prefix_withdrawal.empty()) { (void)index.withdraw(prefix_withdrawal); }
            }
            prefix_cache->poll();
            prefix_cache->sweep_device_copies();
        } catch (...) { clear_prefix_after_release = true; }
        pl = PrefixLane{};
        if (clear_prefix_after_release) { reset_prefix_cache(); }
    }

    // The cache is emptied: after a failure it may hold state no one can vouch for.
    void reset_prefix_cache() noexcept {
        if (!prefix_cache) { return; }
        prefix_withdrawal = {};
        clear_prefix_after_release = false;
        pl = PrefixLane{};
        prefix_cache->clear();
    }

    [[nodiscard]] HybridPrefixCacheStats prefix_stats() const noexcept {
        HybridPrefixCacheStats out;
        if (!prefix_cache) { return out; }
        const pc::PrefixIndexStats index = prefix_cache->index().stats();
        const prefix::PrefixCacheCounters& count = prefix_cache->counters();
        out.nodes = index.nodes;
        out.snapshots = index.snapshots;
        out.device_resident_blocks = index.device_resident_blocks;
        out.device_evictable_blocks = index.device_evictable_blocks;
        out.host_slabs = index.host_slab_limit;
        out.host_free_slabs = index.host_free_slabs;
        out.host_slab_bytes = prefix_cache->host_layout().slab_bytes;
        out.snapshot_hits = index.snapshot_hits;
        out.reused_tokens = count.reused_tokens;
        out.blocks_inserted = count.blocks_inserted;
        out.blocks_duplicate = count.blocks_duplicate;
        out.taps_created = count.taps_created;
        out.taps_skipped = count.taps_skipped;
        out.endpoints_created = count.endpoints_created;
        out.host_image_writes = count.host_image_writes;
        out.host_block_writes = count.host_block_writes;
        out.host_image_restores = count.host_image_restores;
        out.host_block_restores = count.host_block_restores;
        out.host_write_bytes = count.host_write_bytes;
        out.host_restore_bytes = count.host_restore_bytes;
        out.evicted_blocks = index.device_block_evictions;
        out.host_snapshot_evictions = index.host_snapshot_evictions;
        out.host_dead_reclaims = index.host_dead_reclaims;
        out.unbacked_node_losses = index.unbacked_node_losses;
        out.held_snapshots = index.held_snapshots;
        out.held_device_evictions = index.held_device_evictions;
        out.held_snapshot_losses = index.held_snapshot_losses;
        out.held_host_refusals = index.held_host_refusals;
        return out;
    }
    void run_layers(std::span<const TokenId> tokens, std::uint32_t first,
                    runtime::ExecutionTimingRecorder* timing = nullptr) {
        if (tokens.empty() || tokens.size() > options.prefill_chunk || first != fed ||
            tokens.size() > capacity() - first) {
            throw std::logic_error("Gemma4 forward exceeds the valid frontier or capacity");
        }
        if (timing) { timing->resume_submit(); prefill_timer.start(); }
        for (std::size_t i = 0; i < tokens.size(); ++i) { host_positions[i] = static_cast<int>(first + i); }
        upload_pinned(ids.data, tokens.data(), tokens.size_bytes(), stream);
        upload_pinned(positions.data, host_positions.data(), tokens.size() * sizeof(std::int32_t), stream);
        physical_written = first + static_cast<std::uint32_t>(tokens.size());
        bool graphed = false;
        for (std::size_t i = 0; i < kGraphWidths.size(); ++i) {
            if (tokens.size() == kGraphWidths[i] && prefill_graphs[i].executable.ready()) {
                prefill_graphs[i].executable.launch(stream); graphed = true; break;
            }
        }
        if (!graphed) { layers_body(static_cast<std::uint32_t>(tokens.size())); }
        if (timing) { prefill_timer.record_stop(); }
        synchronize(timing);
        if (timing) {
            runtime::ExecutionTiming gpu;
            gpu.gpu_elapsed_ns = static_cast<std::uint64_t>(prefill_timer.elapsed_ms() * 1e6);
            timing->include(gpu);
        }
        fed = physical_written;
    }
    runtime::PendingRowFailure prepare_sample(runtime::TokenMaskProvider* masks, std::uint32_t logical) {
        runtime::PendingRowFailure failure = runtime::PendingRowFailure::None;
        sampling_host.mask = {};
        sampling_host.reasoning_close = -1;
        sampling_host.reasoning_ends[0] = sampling_host.reasoning_ends[1] = -1;
        sampling_host.reasoning_withheld = -1;
        if (masks) {
            const auto ban = masks->reasoning_ban(0);
            sampling_host.reasoning_close = ban.close;
            sampling_host.reasoning_ends[0] = ban.ends[0]; sampling_host.reasoning_ends[1] = ban.ends[1];
            sampling_host.reasoning_withheld = ban.withheld;
            if (masks->constrained(0)) {
                if (masks->fill(0, {}, host_mask) != 0) { failure = runtime::PendingRowFailure::ConstraintDeadEnd; }
                upload_pinned(mask.data, host_mask.data(), mask.bytes(), stream);
                masks->uploaded(0, mask.bytes());
                sampling_host.mask = {static_cast<const std::uint32_t*>(mask.data), static_cast<int>(host_mask.size())};
            }
        }
        auto* words = static_cast<std::int32_t*>(host_step.data());
        words[0] = 0; words[1] = static_cast<int>(logical); words[2] = 0; words[3] = 0;
        upload_pinned(step.data, words, step.bytes(), stream);
        upload_pinned(sampling_device, &sampling_host, sizeof(sampling_host), stream);
        return failure;
    }
    TokenId collect_sample(runtime::PendingRowFailure& failure, runtime::ExecutionTimingRecorder* timing) {
        timing->resume_submit();
        CUDA_CHECK(cudaMemcpyAsync(host_step.data(), step.data, step.bytes(), cudaMemcpyDeviceToHost, stream));
        synchronize(timing);
        const auto* words = static_cast<const std::int32_t*>(host_step.data());
        if (words[2] != 0 || runtime::testing::take_nonfinite_fault()) {
            failure = runtime::PendingRowFailure::NonFiniteLogits;
        }
        if (words[0] < 0 || static_cast<std::uint32_t>(words[0]) >= config().vocab_size) {
            throw runtime::RecoverableExecutionError("Gemma4 sampler returned an invalid token");
        }
        return words[0];
    }
    PendingBatch stage(TokenId token, runtime::PendingRowFailure failure, runtime::ExecutionTiming timing,
                       bool decode_round = false) {
        if (pending) { throw std::logic_error("Gemma4 already has a pending output transaction"); }
        pending = true; pending_token = token; pending_failure = failure; pending_decode = decode_round;
        PendingBatch result;
        result.owner_ = this; result.transaction_ = ++transaction; result.sequence_ = handle();
        result.tokens_ = {&pending_token, 1}; result.failure_ = failure; result.timing_ = timing;
        return result;
    }
    void consume(PendingBatch& batch) noexcept {
        batch.owner_ = nullptr; batch.transaction_ = 0; batch.tokens_ = {}; pending = false;
    }
    void target_scores(const Tensor& values, std::span<const TokenId> targets, int per_column,
                       std::vector<float>& output, runtime::ExecutionTimingRecorder* timing = nullptr) {
        if (timing) { timing->resume_submit(); }
        const int columns = values.ne[1];
        Tensor target(distribution_ids.data, DType::I32, {per_column, columns});
        Tensor scores(distribution_values.data, DType::FP32, {per_column, columns});
        upload_pinned(target.data, targets.data(), targets.size_bytes(), stream);
        ops::target_logprobs(values, target, static_cast<int>(config().vocab_size), scores, stream);
        const auto offset = output.size(); output.resize(offset + targets.size());
        CUDA_CHECK(cudaMemcpyAsync(output.data() + offset, scores.data, scores.bytes(), cudaMemcpyDeviceToHost, stream));
        synchronize(timing);
        for (std::size_t i = offset; i < output.size(); ++i) {
            if (!std::isfinite(output[i])) {
                throw runtime::RecoverableExecutionError("Gemma4 scoring found non-finite logits");
            }
        }
    }
    void top_scores(const Tensor& values, std::uint32_t k, std::vector<TokenId>& out_ids,
                    std::vector<float>& out_values, runtime::ExecutionTimingRecorder* timing = nullptr) {
        if (timing) { timing->resume_submit(); }
        Tensor top_ids(distribution_ids.data, DType::I32, {static_cast<int>(k), values.ne[1]});
        Tensor top_values(distribution_values.data, DType::FP32, {static_cast<int>(k), values.ne[1]});
        ops::top_logprobs(values, static_cast<int>(config().vocab_size), top_ids, top_values, stream);
        const auto offset = out_ids.size();
        out_ids.resize(offset + static_cast<std::size_t>(top_ids.numel()));
        out_values.resize(out_ids.size());
        CUDA_CHECK(cudaMemcpyAsync(out_ids.data() + offset, top_ids.data, top_ids.bytes(), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(out_values.data() + offset, top_values.data, top_values.bytes(), cudaMemcpyDeviceToHost, stream));
        synchronize(timing);
    }
};
} // namespace detail

const runtime::RequestPlanSummary& RequestBasePlan::summary() const noexcept { return impl_->summary; }
ResumeState::ResumeState(std::unique_ptr<detail::ResumeData> data) noexcept : impl_(std::move(data)) {}
ResumeState::ResumeState(ResumeState&&) noexcept = default;
ResumeState& ResumeState::operator=(ResumeState&&) noexcept = default;
ResumeState::~ResumeState() = default;
std::uint32_t ResumeState::frontier() const noexcept { return impl_ ? impl_->frontier : 0; }
Program::Program(const execution::Parameters& parameters, DeviceContext& device, ProgramOptions options)
    : impl_(std::make_unique<detail::ProgramImpl>(parameters, device, std::move(options))) {}
Program::~Program() noexcept { shutdown_cleanup(); }
const ProgramDevicePlan& Program::device_plan() const noexcept { return impl_->plan; }
std::uint64_t Program::host_bytes() const noexcept { return impl_->options.host_bytes; }

RequestBasePlan Program::plan_request(PreparedPrompt&& prompt, const runtime::ResolvedExecutionOptions& options) {
    const auto& tokens = prompt.tokens();
    if (tokens.empty() || tokens.size() > impl_->capacity() || options.requested_output_tokens == 0) {
        throw std::invalid_argument("Gemma4 request needs a nonempty prompt fitting the KV capacity and an output budget");
    }
    const auto vocab = impl_->config().vocab_size;
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocab) {
            throw std::invalid_argument("Gemma4 prompt token is outside the public vocabulary");
        }
    }
    for (const auto token : options.readout_tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocab) {
            throw std::invalid_argument("Gemma4 readout token is outside the public vocabulary");
        }
    }
    const auto& s = options.sampling;
    if (!std::isfinite(s.temperature) || s.temperature < 0 || s.top_k < 1 || s.top_k > ops::kSamplingMaxTopK ||
        !std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1 ||
        !std::isfinite(s.min_p) || s.min_p < 0 || s.min_p > 1 ||
        !std::isfinite(s.presence_penalty) || !std::isfinite(s.frequency_penalty)) {
        throw std::invalid_argument("Gemma4 sampling requires finite parameters and top_k in [1,64]");
    }
    auto plan = std::make_shared<detail::RequestPlan>();
    plan->owner = impl_.get();
    plan->reuse = impl_->prefix_cache != nullptr && options.allow_prefix_reuse;
    // The Frontend keys the prompt; a prompt whose keys do not cover its tokens (changed after
    // preparation) is keyed here.
    plan->block_hashes = prompt.block_hashes_.size() == tokens.size() / pc::kBlockTokens
        ? std::move(prompt.block_hashes_) : pc::block_lookup_hashes(tokens, {});
    plan->hints = std::move(prompt.tap_hints_);
    plan->tokens = std::move(prompt.tokens_);
    plan->sampling = s; plan->readout_tokens = options.readout_tokens;
    auto& summary = plan->summary;
    summary.prompt_tokens = static_cast<std::uint32_t>(plan->tokens.size());
    summary.requested_output_tokens = options.requested_output_tokens;
    summary.effective_output_tokens = std::min(options.requested_output_tokens,
        impl_->capacity() - summary.prompt_tokens + 1);
    summary.effective_limit_reason = summary.effective_output_tokens < options.requested_output_tokens
        ? FinishReason::ContextCapacity : FinishReason::OutputLimit;
    summary.publish_continuation = plan->reuse;
    return RequestBasePlan(std::move(plan));
}
std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, std::uint32_t maximum_frontier) {
    if (!base.impl_ || base.impl_->owner != impl_.get()) { throw std::logic_error("Gemma4 foreign request plan"); }
    auto& p = *impl_;
    const auto& plan = *base.impl_;
    std::uint32_t cached = 0;
    std::vector<SourceCandidate> sources;
    if (p.prefix_cache && plan.reuse) {
        // The index's choice first, then a root start. A resumed request's quote is bounded by its
        // frontier; its binding re-selects over its whole ledger.
        const auto candidates = p.prefix_candidates(plan.tokens, plan.block_hashes, maximum_frontier,
                                                    maximum_frontier != UINT32_MAX);
        cached = candidates.cached_tokens;
        if (!candidates.list.empty()) {
            const auto& best = candidates.list.front();
            sources.push_back(SourceCandidate{best.frontier, reuse_path_for(best.kind), cached});
        }
    }
    sources.push_back(SourceCandidate{0, PrefixReusePath::Root, cached});
    return sources;
}
BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
    const SourceCandidate& source, ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    auto& p = *impl_;
    if (!base.impl_ || base.impl_->owner != impl_.get()) { throw std::logic_error("Gemma4 foreign request plan"); }
    if (lane.value != 0) { throw std::invalid_argument("Gemma4 has only lane zero"); }
    const auto& plan = *base.impl_;
    const bool cached = p.prefix_cache && plan.reuse;
    if (source.reused_tokens != 0 && !cached) { return {.source_valid = false}; }
    if (p.binding || p.occupied || p.pending || p.paused) { return {}; }
    if (resume && (!resume->impl_ || resume->impl_->owner != &p || resume->impl_->plan != base.impl_)) {
        throw std::logic_error("Gemma4 foreign or consumed replay state");
    }
    // Stage all potentially throwing host copies before taking ownership of the binding.
    auto ledger = resume ? resume->impl_->ledger : plan.tokens;
    ledger.reserve(p.capacity() + 1ULL);
    // Re-select against the cache as it is now: the quote the Engine holds may be stale. A
    // resumed request re-selects over its whole ledger (its pause published the frontier).
    detail::ProgramImpl::PrefixCandidates candidates;
    std::optional<detail::ProgramImpl::PrefixSelection> selection;
    if (cached) {
        std::vector<std::uint64_t> ledger_hashes;
        if (resume) { ledger_hashes = pc::block_lookup_hashes(ledger, {}); }
        candidates = p.prefix_candidates(ledger, resume ? std::span<const std::uint64_t>(ledger_hashes)
                                                         : std::span<const std::uint64_t>(plan.block_hashes),
                                         resume ? resume->impl_->frontier : UINT32_MAX, resume != nullptr);
        if (source.reused_tokens != 0) {
            for (auto& candidate : candidates.list) {
                if (resume || candidate.frontier == source.reused_tokens) {
                    selection = std::move(candidate);
                    break;
                }
            }
            // The quoted snapshot changed (evicted, superseded, its image no longer valid): the
            // Engine moves to the next source, the root.
            if (!selection) { return {.source_valid = false}; }
        }
        // Pin what the restore reads before anything can evict it.
        if (selection) {
            auto& index = p.prefix_cache->index();
            if (!selection->path.empty()) { index.acquire_path(selection->path); }
            try {
                index.pin_snapshot(*selection->snapshot);
            } catch (...) {
                if (!selection->path.empty()) { index.release_path(selection->path); }
                throw;
            }
        }
    }
    p.request = base.impl_; p.ledger = std::move(ledger);
    if (resume) { p.binding_resume = std::move(resume->impl_); }
    p.pl = {};
    if (cached) {
        p.pl.active = true;
        p.pl.hashes = plan.block_hashes;
        p.pl.hints = plan.hints;
        std::uint32_t frontier = 0;
        bool follow_endpoint = false;
        if (selection) {
            frontier = selection->frontier;
            follow_endpoint = selection->kind == pc::SnapshotKind::Endpoint;
            p.pl.path = std::move(selection->path);
            p.pl.restore_snapshot = selection->snapshot;
            p.pl.pinned_snapshot = selection->snapshot;
        }
        // A resumed request publishes blocks, not taps (its ledger is already cached up to where
        // it was); the call grid then is the plain chunk grid.
        if (!p.binding_resume) { p.plan_prefill(frontier, candidates.existing, follow_endpoint); }
    }
    // The Engine committed the Begin once, at the first admission: a resume keeps it (its own source
    // only decides where the state is rebuilt from), and a first binding reports the quote it got.
    p.begin = p.binding_resume ? p.binding_resume->begin
        : runtime::BeginSummary{.prompt_tokens = plan.summary.prompt_tokens,
                                .reused_prompt_tokens = source.reused_tokens,
                                .prefix_reuse_path = source.reused_tokens ? source.reuse_path : PrefixReusePath::Root};
    p.binding = true;
    return {.reserved = true};
}
ContextProgress Program::poll_context(runtime::CancellationFlagView cancellation) {
    auto& p = *impl_;
    ContextProgress progress;
    if (p.paused) {
        progress.kind = ContextOperationKind::Pause; progress.advanced = progress.complete = true;
        progress.paused.emplace(std::move(*p.paused)); p.paused.reset(); return progress;
    }
    if (!p.binding) { return progress; }
    progress.advanced = progress.complete = true;
    if (cancellation.requested()) { p.clear(); return progress; }
    try {
        p.upload_sampler();
        p.occupied = true; p.binding = false; p.fed = p.physical_written = 0;
        p.prefilled = false; p.finishable = false; p.timings = {};
        if (p.pl.restore_snapshot) {
            // The restore batch: blocks, tail page and ring window, enqueued in stream order ahead
            // of the first forward (the image header is validated again before any copy).
            auto& cache = *p.prefix_cache;
            pc::PrefixCacheIndex& index = cache.index();
            const pc::SnapshotRef snapshot = *p.pl.restore_snapshot;
            const prefix::RestoreResult restore = cache.restore(p.pl.path, snapshot);
            if (restore.restored) {
                const std::uint32_t frontier = index.snapshot(snapshot).frontier;
                p.fed = p.physical_written = frontier;
                progress.operations.state_restores = 1;
                progress.operations.partial_tail_cow_pages = restore.tail ? 1U : 0U;
                index.note_hit(snapshot);
                index.unpin_snapshot(snapshot); // the restore batch holds its own pin while it lands
                p.pl.pinned_snapshot.reset();
                p.pl.resume = snapshot;
                p.pl.reused = p.pl.deepest = p.pl.exact_through = frontier;
                ++cache.counters().admissions;
                cache.counters().reused_tokens += frontier;
            } else {
                // Refused before any copy (an image that no longer validates): start at the root.
                index.unpin_snapshot(snapshot);
                p.pl.pinned_snapshot.reset();
                if (!p.pl.path.empty()) { index.release_path(p.pl.path); }
                p.pl.path.clear();
                if (!p.binding_resume) { p.plan_prefill(0, {}, false); }
            }
            p.pl.restore_snapshot.reset();
        }
        CUDA_CHECK(cudaStreamSynchronize(p.stream));
        if (p.prefix_cache) { p.prefix_cache->poll(); }
        if (p.binding_resume) {
            p.replay_target = p.binding_resume->frontier;
            p.prefilled = p.binding_resume->prefilled; p.finishable = p.binding_resume->finishable;
            p.timings = p.binding_resume->timings; p.replaying = p.replay_target > p.fed;
            p.binding_resume.reset();
        }
        progress.published = true; progress.sequence = p.handle(); progress.replaying = p.replaying;
        return progress;
    } catch (...) { p.state_valid = false; throw; }
}
bool Program::has_context_transaction() const noexcept { return impl_->binding || impl_->paused.has_value(); }
bool Program::context_blocks(SequenceHandle h) const noexcept {
    return impl_->owns(h) && (has_context_transaction() || impl_->pending);
}
runtime::ResourceReservation Program::reserve_units(std::span<const ExecutionUnit> units) {
    auto& p = *impl_;
    if (units.size() != 1) { throw std::logic_error("Gemma4 reserves exactly one execution lane"); }
    p.require(units[0].sequence);
    if (p.pending || has_context_transaction()) { throw std::logic_error("Gemma4 reservation overlaps a transaction"); }
    const auto& unit = units[0];
    if ((unit.kind == ExecutionUnitKind::Decode && (unit.tokens == 0 || !p.prefilled || p.finishable ||
        p.replaying || p.ledger.size() != p.fed + 1ULL)) ||
        (unit.kind == ExecutionUnitKind::Prefill && (p.prefilled || p.replaying)) ||
        (unit.kind == ExecutionUnitKind::Replay && !p.replaying)) {
        throw std::logic_error("Gemma4 unit does not match its sequence phase");
    }
    if (unit.kind == ExecutionUnitKind::Normalize) {
        throw std::invalid_argument("Gemma4 P4 has no normalization unit");
    }
    const auto needed = unit.kind == ExecutionUnitKind::Decode ? p.fed + 1ULL :
        unit.kind == ExecutionUnitKind::Control ? p.ledger.size() + unit.tokens : 0;
    if (needed > p.capacity()) {
        return {.shortage = {.main_kv_pages = pages_for(static_cast<std::uint32_t>(needed)) - p.plan.kv_pages}};
    }
    return {.reserved = true};
}
bool Program::start_pause(SequenceHandle sequence, runtime::ExecutionTiming* observed_timing) {
    auto& p = *impl_; p.require(sequence);
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, observed_timing);
    if (p.pending || has_context_transaction() || p.replaying) { return false; }
    // Publish what the exact prefix reached (and an endpoint when the committed frontier is its end), so
    // the resume restores the deepest grid state instead of replaying from the root.
    p.prefix_finish(true);
    auto data = std::make_unique<detail::ResumeData>();
    data->owner = &p; data->plan = p.request; data->ledger = p.ledger;
    data->frontier = p.fed; data->prefilled = p.prefilled; data->finishable = p.finishable; data->timings = p.timings;
    data->begin = p.begin;
    // The deepest published grid snapshot is the state the resume restores from; replay then rebuilds the
    // exact committed target frontier on the chunk grid.
    p.clear(); p.paused.emplace(ResumeState(std::move(data))); return true;
}
ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing) {
    auto& p = *impl_; p.require(sequence);
    if (!p.replaying || p.pending) { throw std::logic_error("Gemma4 sequence is not replaying"); }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    try {
        const auto n = std::min(p.options.prefill_chunk, p.replay_target - p.fed);
        const std::uint32_t first = p.fed;
        p.run_layers(std::span<const TokenId>(p.ledger).subspan(p.fed, n), p.fed, &timing);
        p.note_grid_call(first, n);
        p.publish_blocks(p.fed);
        p.replaying = p.fed != p.replay_target;
        return {.processed_tokens = n, .complete = !p.replaying, .timing = timing.finish()};
    } catch (...) { p.state_valid = false; throw; }
}
PrefillProgress Program::advance_prefill(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing,
    runtime::TokenMaskProvider* masks) {
    auto& p = *impl_; p.require(sequence);
    if (p.prefilled || p.replaying || p.pending || p.finishable) { throw std::logic_error("Gemma4 sequence is not prefilling"); }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    try {
        const auto n = std::min(p.options.prefill_chunk, p.next_call_end() - p.fed);
        const std::uint32_t first = p.fed;
        p.run_layers(std::span<const TokenId>(p.ledger).subspan(p.fed, n), p.fed, &timing);
        p.note_grid_call(first, n);
        p.after_prefill_call();
        PrefillProgress progress;
        progress.summary = p.begin;
        progress.processed_prompt_tokens = n;
        progress.complete = p.fed == p.request->summary.prompt_tokens;
        if (progress.complete) {
            timing.resume_submit();
            p.head_body(static_cast<int>(n) - 1, 1);
            if (!p.request->readout_tokens.empty()) {
                PromptReadout readout;
                // Readout can name more than one staging column's 256 targets: tile the list.
                for (std::size_t first = 0; first < p.request->readout_tokens.size(); first += kMaximumScoreTopTokens) {
                    const auto targets = std::span<const TokenId>(p.request->readout_tokens).subspan(first,
                        std::min<std::size_t>(kMaximumScoreTopTokens, p.request->readout_tokens.size() - first));
                    p.target_scores(p.logits(1), targets, static_cast<int>(targets.size()), readout.logprobs, &timing);
                }
                std::vector<TokenId> ids; std::vector<float> values;
                p.top_scores(p.logits(1), 1, ids, values, &timing);
                readout.top_token = ids[0]; readout.top_logprob = values[0];
                progress.readout = std::move(readout);
            }
            timing.resume_submit();
            auto failure = p.prepare_sample(masks, p.fed);
            p.sample_body(ops::kSamplePurposePrefill);
            const auto token = p.collect_sample(failure, &timing);
            p.prefilled = true;
            progress.timing = p.charge(timing.finish(), false);
            progress.pending.emplace(p.stage(token, failure, progress.timing));
        } else { progress.timing = p.charge(timing.finish(), false); }
        return progress;
    } catch (...) {
        p.state_valid = false;
        const auto measured = p.charge(timing.finish(), false);
        if (failed_timing) { *failed_timing += measured; }
        throw;
    }
}
PendingBatch Program::decode(std::span<const SequenceHandle> sequences, std::span<const runtime::RoundBudget> budgets,
    runtime::ExecutionTiming* failed_timing, runtime::TokenMaskProvider* masks) {
    auto& p = *impl_;
    if (sequences.size() != 1 || budgets.size() != 1) { throw std::logic_error("Gemma4 decode needs one row"); }
    p.require(sequences[0]);
    if (p.pending || !p.prefilled || p.replaying || p.finishable || p.ledger.size() != p.fed + 1ULL ||
        budgets[0].generated_tokens_remaining == 0 || p.fed >= p.capacity()) {
        throw std::logic_error("Gemma4 decode has no licensed input or capacity");
    }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    try {
        auto failure = p.prepare_sample(masks, p.fed + 1);
        const auto token = p.ledger.back();
        const auto position = static_cast<std::int32_t>(p.fed);
        upload_pinned(p.ids.data, &token, sizeof(token), p.stream);
        upload_pinned(p.positions.data, &position, sizeof(position), p.stream);
        p.physical_written = p.fed + 1;
        if (p.options.use_cuda_graph) { p.decode_graph.launch(p.stream); }
        else { p.decode_body(); }
        const auto next = p.collect_sample(failure, &timing);
        p.fed = p.physical_written;
        return p.stage(next, failure, p.charge(timing.finish(), true), true);
    } catch (...) {
        p.state_valid = false;
        const auto measured = p.charge(timing.finish(), true);
        if (failed_timing) { *failed_timing += measured; }
        throw;
    }
}
runtime::ExecutionTiming Program::append_forced_tokens(std::span<const SequenceHandle> sequences,
    std::span<const TokenId> tokens, std::uint32_t stride,
    std::span<const std::optional<std::uint32_t>> splits, runtime::ExecutionTiming* failed_timing) {
    auto& p = *impl_;
    if (sequences.size() != 1 || tokens.empty() || tokens.size() != stride ||
        (!splits.empty() && splits.size() != 1)) { throw std::invalid_argument("Gemma4 forced-token row shape"); }
    p.require(sequences[0]);
    if (p.pending || !p.prefilled || p.finishable || p.replaying || p.ledger.size() != p.fed + 1ULL ||
        p.ledger.size() + tokens.size() > p.capacity() + 1ULL) {
        throw std::logic_error("Gemma4 cannot append forced tokens in this state");
    }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= p.config().vocab_size) {
            throw std::invalid_argument("Gemma4 forced token outside the public vocabulary");
        }
    }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    try {
        // Host capacity was reserved at startup. Once forced output is licensed, feed the previous
        // anchor and all but its last token; the final token remains the next decode's anchor.
        p.ledger.insert(p.ledger.end(), tokens.begin(), tokens.end());
        while (p.fed + 1ULL < p.ledger.size()) {
            const auto n = std::min<std::size_t>(p.options.prefill_chunk, p.ledger.size() - p.fed - 1);
            p.run_layers(std::span<const TokenId>(p.ledger).subspan(p.fed, n), p.fed, &timing);
        }
        timing.resume_submit();
        p.upload_sampler(); p.synchronize(&timing);
        return p.charge(timing.finish(), true);
    } catch (...) {
        p.state_valid = false;
        const auto measured = p.charge(timing.finish(), true);
        if (failed_timing) { *failed_timing += measured; }
        throw;
    }
}
CommitResult Program::commit(PendingBatch&& batch, std::span<const runtime::CommitDecision> decisions,
    runtime::CommitObservation observation, runtime::ExecutionTiming* failed_timing) {
    auto& p = *impl_;
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Post, failed_timing);
    if (!p.owns(batch) || decisions.size() != 1 || !p.state_valid) {
        throw std::logic_error("Gemma4 commit does not own the pending transaction");
    }
    const auto& decision = decisions[0];
    const bool failed = decision.failed || p.pending_failure != runtime::PendingRowFailure::None;
    if (decision.accepted_tokens > 1 || ((!failed && !decision.cancelled && !decision.terminal) &&
        decision.accepted_tokens != 1) || (decision.prefix_execution_split_after &&
        *decision.prefix_execution_split_after > decision.accepted_tokens)) {
        throw std::logic_error("Gemma4 commit has an invalid accepted extent");
    }
    CommitResult result; result.row_count = 1;
    if (!failed && !decision.cancelled && decision.accepted_tokens == 1) { p.ledger.push_back(p.pending_token); }
    p.consume(batch);
    auto& row = result.rows[0];
    row.disposition = failed ? runtime::CommitDisposition::FailedReleased :
        decision.cancelled ? runtime::CommitDisposition::CancelledReleased :
        decision.terminal ? runtime::CommitDisposition::Finishable : runtime::CommitDisposition::Active;
    result.timing = p.charge(timing.finish(), p.pending_decode);
    if (observation == runtime::CommitObservation::AllRows || failed || decision.cancelled) { row.timings = p.timings; }
    if (failed || decision.cancelled) {
        // Non-finite logits make the sequence's state suspect: what it published is taken back.
        if (p.pending_failure == runtime::PendingRowFailure::NonFiniteLogits) { p.pl.withdraw = true; }
        p.clear();
    }
    else { p.finishable = decision.terminal; }
    return result;
}
DiscardResult Program::abort_pending(PendingBatch&& batch) noexcept {
    auto& p = *impl_;
    if (!p.owns(batch)) { return {}; }
    if (p.pending_failure == runtime::PendingRowFailure::NonFiniteLogits) { p.pl.withdraw = true; }
    p.consume(batch); p.clear();
    return {.status = runtime::ConsumeStatus::Consumed, .row_count = 1};
}
FinishResult Program::finish(SequenceHandle sequence) noexcept {
    auto& p = *impl_;
    if (!p.owns(sequence) || p.pending || !p.finishable || !p.state_valid) { return {}; }
    FinishResult result; result.status = runtime::ConsumeStatus::Consumed; result.timings = p.timings;
    p.prefix_finish(false);
    result.cache_created_prompt_tokens = p.prefix_created_tokens();
    p.clear(); return result;
}
AbortResult Program::abort(SequenceHandle sequence) noexcept {
    auto& p = *impl_;
    if (!p.owns(sequence)) { return {}; }
    AbortResult result; result.status = runtime::ConsumeStatus::Consumed; result.timings = p.timings;
    // A consistent abort (no unit in flight) leaves an endpoint a resend resumes from.
    if (!p.pending) { p.prefix_finish(false); }
    p.clear(); return result;
}
void Program::fail_all_cleanup() noexcept {
    (void)settle_after_memory_exhausted();
    impl_->clear();
    impl_->reset_prefix_cache();
}
bool Program::recover_after_failure() noexcept {
    // The caller has settled an OOM once already; this also checks the real stream/device state
    // for other recoverable failures. Never claim recovery while earlier writes are still live.
    const bool usable = settle_after_memory_exhausted();
    impl_->clear(); impl_->reset_prefix_cache(); return usable;
}
void Program::shutdown_cleanup() noexcept {
    if (!impl_) { return; }
    const auto status = cudaStreamSynchronize(impl_->stream);
    if (status != cudaSuccess) { report_cuda_cleanup_failure("Gemma4 shutdown synchronize", status); }
    impl_->clear();
}
void Program::hybrid_hold_queue(std::span<const RequestBasePlan* const> queue) noexcept {
    auto& p = *impl_;
    if (!p.prefix_cache) { return; }
    try {
        std::vector<pc::SnapshotRef> holds;
        for (const RequestBasePlan* base : queue) {
            if (holds.size() == pc::kMaxQueueHolds) { break; }
            if (base == nullptr || !base->impl_ || !base->impl_->reuse) { continue; }
            const auto& plan = *base->impl_;
            const auto candidates = p.prefix_candidates(plan.tokens, plan.block_hashes, UINT32_MAX, false);
            if (!candidates.list.empty()) { holds.push_back(*candidates.list.front().snapshot); }
        }
        p.prefix_cache->index().set_queue_holds(holds);
    } catch (...) {}
}
void Program::hybrid_hold_continuation(std::chrono::steady_clock::time_point) {
    // HybridPrefixCacheOptions::tool_call_hold is refused for Gemma4 at startup (model_instance.cpp),
    // so the Engine never asks.
    throw std::logic_error("Gemma4's prefix cache has no tool-call holds (refused at startup)");
}
std::uint64_t Program::hybrid_cache_epoch() const noexcept {
    return impl_->prefix_cache ? impl_->prefix_cache->index().snapshot_epoch() : 0U;
}
HybridPrefixCacheStats Program::hybrid_stats() const noexcept { return impl_->prefix_stats(); }
runtime::prefix_cache::PersistResult Program::attach_prefix_cache_file(const std::filesystem::path&,
    std::string, const StartupObserver&) {
    throw std::invalid_argument("Gemma4's prefix cache is not persisted: --prefix-cache-file is unsupported");
}
PhysicalUsageSnapshot Program::physical_usage() const noexcept {
    PhysicalUsageSnapshot result;
    result.capacity.main_kv_pages = impl_->plan.kv_pages;
    result.capacity.state_slots = 1;
    if (impl_->occupied || impl_->binding) {
        // The lane has exclusive ownership of the complete fixed allocation from admission.
        result.occupied.main_kv_pages = impl_->plan.kv_pages; result.occupied.state_slots = 1;
    }
    return result;
}
MemorySummary Program::memory_summary() const noexcept {
    const auto& p = *impl_; MemorySummary result;
    result.device = p.device.device; result.max_context = p.options.max_context;
    result.weights.capacity_bytes = static_cast<std::size_t>(p.parameters.model.storage_stats().device_capacity_bytes);
    result.weights.used_bytes = result.weights.capacity_bytes;
    result.weights.peak_used_bytes = result.weights.capacity_bytes;
    result.kv_capacity = p.plan.kv_pages * kPage; result.kv_capacity_page_groups = p.plan.kv_pages;
    result.kv_capacity_max_page_groups = pages_for(p.options.max_context);
    // kv_cache is the global format; gemma4_kv states the BF16 sliding rings explicitly.
    result.kv_cache = p.options.kv_cache;
    result.gemma4_kv = Gemma4KvMemorySummary{
        .requested = p.options.kv_cache, .global = p.options.kv_cache, .sliding = KvCacheStorage::BFloat16,
        .qk_mode = p.options.qk_mode, .global_payload_bytes = p.global_payload_bytes,
        .sliding_payload_bytes = p.sliding_payload_bytes};
    result.kv_payload_bytes = p.global_payload_bytes + p.sliding_payload_bytes;
    result.sequence.capacity_bytes = static_cast<std::size_t>(p.plan.kv + p.plan.io);
    result.sequence.used_bytes = result.sequence.capacity_bytes;
    result.sequence.peak_used_bytes = result.sequence.capacity_bytes;
    result.workspace.capacity_bytes = p.workspace.capacity();
    result.workspace.used_bytes = p.workspace.used(); result.workspace.peak_used_bytes = p.workspace.peak_used();
    result.workspace_logical_peak_bytes = p.workspace.peak_used();
    result.cuda_graph_allowance_bytes = static_cast<std::size_t>(p.plan.graph_allowance);
    result.cuda_graph_measured_bytes = p.graph_measured_bytes;
    result.runtime_reservation_bytes = static_cast<std::size_t>(p.plan.fixed_bytes());
    result.minimum_runtime_reservation_bytes = p.plan.curve.minimum_device_reservation_bytes;
    result.kv_capacity_increment_bytes = p.plan.curve.bytes_per_additional_main_page_group;
    return result;
}
MemoryUsage Program::memory_usage() const noexcept {
    const auto& p = *impl_; MemoryUsage out;
    const auto& storage = p.parameters.model.storage_stats();
    // The materialized arena plus the row scales Parameters prepared from the FP8 projections.
    out.device.weights = storage.device_capacity_bytes + p.parameters.derived_device_bytes();
    // KV: every global and ring payload (codes and FP16 scales) and both page tables, all reserved at construction.
    out.device.kv = p.plan.kv;
    if (p.occupied) {
        // The live request's tokens: global pages up to its frontier and the rings' filled share.
        const std::uint32_t pages = p.plan.kv_pages;
        const std::uint32_t used = std::min(pages, pages_for(std::max(p.fed, p.physical_written)));
        const std::uint32_t rings = std::min(pages, ring_pages(p.options.prefill_chunk, p.config().sliding_window));
        std::uint64_t bytes = 8ULL * used;
        if (pages != 0) { bytes += p.global_payload_bytes * used / pages; }
        if (rings != 0) { bytes += p.sliding_payload_bytes * std::min(rings, used) / rings; }
        out.device.kv_used = std::min<std::uint64_t>(bytes, out.device.kv);
    }
    out.device.state = p.plan.io; // call inputs and outputs (hidden, head, sampler and distribution buffers)
    out.device.workspace = p.workspace.capacity();
    out.device.cuda_graphs = p.graph_measured_bytes;
    out.host.weights = storage.pinned_bytes + storage.retained_host_bytes;
    out.host.staging = kUploadStagingRingBytes + p.host_step.size();
    return out;
}
void Program::reset_memory_peaks() noexcept { impl_->workspace.reset_peak(); }

ScoreResult Program::causal_score(PreparedPrompt&& prompt, std::uint32_t first_target, const ScoreOptions& options) {
    auto& p = *impl_;
    if (!p.options.causal_scoring) { throw std::logic_error("Gemma4 Program was not planned for causal scoring"); }
    if (p.occupied || has_context_transaction() || p.pending) { throw std::logic_error("Gemma4 scoring needs an idle Program"); }
    const auto& tokens = prompt.tokens(); const auto n = tokens.size();
    if (n < 2 || n > p.capacity() || first_target == 0 || first_target >= n) {
        throw std::invalid_argument("Gemma4 causal scoring window or first_target is invalid");
    }
    const auto scored = n - first_target;
    if (options.top_k > std::min(kMaximumScoreTopTokens, p.config().vocab_size) ||
        options.candidates_per_position > kMaximumScoreTopTokens ||
        options.candidates.size() != scored * options.candidates_per_position ||
        options.logits_positions.size() > kMaximumScoreLogitPositions) {
        throw std::invalid_argument("Gemma4 causal scoring distribution shape is invalid");
    }
    for (const auto token : tokens) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= p.config().vocab_size) {
            throw std::invalid_argument("Gemma4 score token outside public vocabulary");
        }
    }
    for (const auto token : options.candidates) {
        if (token < 0 || static_cast<std::uint32_t>(token) >= p.config().vocab_size) {
            throw std::invalid_argument("Gemma4 score candidate outside public vocabulary");
        }
    }
    for (std::size_t i = 0; i < options.logits_positions.size(); ++i) {
        if (options.logits_positions[i] >= scored ||
            (i != 0 && options.logits_positions[i - 1] >= options.logits_positions[i])) {
            throw std::invalid_argument("Gemma4 score logits positions must be strictly ascending within the scored range");
        }
    }
    ScoreResult result;
    result.logprobs.reserve(scored); result.top_ids.reserve(scored * options.top_k);
    result.top_logprobs.reserve(scored * options.top_k);
    result.candidate_logprobs.reserve(options.candidates.size());
    result.logits.reserve(options.logits_positions.size() * static_cast<std::size_t>(p.config().vocab_size));
    std::size_t selected = 0;
    p.occupied = true; p.fed = p.physical_written = 0;
    try {
        for (std::uint32_t cursor = 0; cursor < n - 1;) {
            const auto width = std::min<std::uint32_t>(p.options.prefill_chunk, static_cast<std::uint32_t>(n - 1) - cursor);
            p.run_layers(std::span<const TokenId>(tokens).subspan(cursor, width), cursor);
            const auto first = std::max(cursor, first_target - 1);
            for (auto predictor = first; predictor < cursor + width;) {
                const auto columns = std::min<std::uint32_t>(kScoreTile, cursor + width - predictor);
                p.head_body(static_cast<int>(predictor - cursor), static_cast<int>(columns));
                const auto values = p.logits(static_cast<int>(columns));
                const auto position = predictor + 1 - first_target;
                p.target_scores(values, std::span<const TokenId>(tokens).subspan(predictor + 1, columns), 1, result.logprobs);
                if (options.top_k) { p.top_scores(values, options.top_k, result.top_ids, result.top_logprobs); }
                if (options.candidates_per_position) {
                    const auto count = options.candidates_per_position;
                    p.target_scores(values, std::span<const TokenId>(options.candidates).subspan(
                        static_cast<std::size_t>(position) * count, static_cast<std::size_t>(columns) * count),
                        static_cast<int>(count), result.candidate_logprobs);
                }
                while (selected < options.logits_positions.size() && options.logits_positions[selected] < position + columns) {
                    const auto column = options.logits_positions[selected] - position;
                    const auto* source = static_cast<const std::uint16_t*>(values.data) +
                        static_cast<std::size_t>(column) * p.config().vocab_size;
                    CUDA_CHECK(cudaMemcpyAsync(p.host_logits.data(), source, p.host_logits.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, p.stream));
                    CUDA_CHECK(cudaStreamSynchronize(p.stream));
                    for (const auto bits : p.host_logits) {
                        result.logits.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16));
                    }
                    ++selected;
                }
                predictor += columns;
            }
            cursor += width;
        }
        p.clear();
        return result;
    } catch (...) {
        // Preserve the original recoverable CUDA exception; settlement must finish before a new
        // scoring call can reuse the rings, graph operands or staging addresses.
        const bool recovered = recover_after_failure();
        if (!recovered) { throw std::runtime_error("Gemma4 scoring failed and the device could not recover"); }
        throw;
    }
}

void Program::maintain() noexcept {
    if (!impl_->prefix_cache) { return; }
    try {
        impl_->prefix_cache->poll();
    } catch (const std::exception& error) {
        impl_->clear_prefix_after_release = true;
        try {
            impl_->diagnostic(std::string("Gemma4: retiring prefix-cache Host transfers failed: ") + error.what(),
                DiagnosticLevel::Warning);
        } catch (...) {}
    } catch (...) {
        impl_->clear_prefix_after_release = true;
        impl_->diagnostic("Gemma4: retiring prefix-cache Host transfers failed", DiagnosticLevel::Warning);
    }
}
void Program::maintain_busy() noexcept { maintain(); }

ScoreResult Program::diagnostic_score_bound(SequenceHandle sequence, std::uint32_t first_target,
    const std::function<void(std::span<const std::uint16_t>, std::int32_t, std::uint32_t)>& observer) {
    auto& p = *impl_;
    p.require(sequence);
    const auto n = static_cast<std::uint32_t>(p.ledger.size());
    if (!p.options.causal_scoring || !p.request || p.prefilled || p.replaying || p.pending ||
        p.finishable || has_context_transaction() || !observer || first_target == 0 ||
        first_target >= n || p.fed > first_target - 1) {
        throw std::logic_error("Gemma4 diagnostic scoring needs a bound prefill before its first predictor");
    }
    ScoreResult result;
    result.logprobs.reserve(n - first_target);
    try {
        // Unlike causal_score, do not reset fed: poll_context's Host restore is the input state.
        // Feed the full prompt, as serving does, including its final non-predictor input token.
        while (p.fed < n) {
            const auto cursor = p.fed;
            const auto width = std::min(p.options.prefill_chunk, p.next_call_end() - cursor);
            if (width == 0) { throw std::logic_error("Gemma4 diagnostic prefill made no progress"); }
            p.run_layers(std::span<const TokenId>(p.ledger).subspan(cursor, width), cursor);
            p.note_grid_call(cursor, width);
            p.after_prefill_call();
            for (auto predictor = std::max(cursor, first_target - 1);
                 predictor < std::min(cursor + width, n - 1);) {
                const auto columns = std::min<std::uint32_t>(kScoreTile,
                    std::min(cursor + width, n - 1) - predictor);
                // Keep the serving GEMV head for every predictor; batch only transfer/reduction.
                for (std::uint32_t c = 0; c < columns; ++c) {
                    p.head_body(static_cast<int>(predictor - cursor + c), 1, static_cast<int>(c));
                }
                const auto values = p.logits(static_cast<int>(columns));
                p.target_scores(values, std::span<const TokenId>(p.ledger).subspan(predictor + 1, columns),
                    1, result.logprobs);
                const auto cells = static_cast<std::size_t>(columns) * p.config().vocab_size;
                if (p.host_logits.size() < cells) { p.host_logits.resize(cells); }
                CUDA_CHECK(cudaMemcpyAsync(p.host_logits.data(), values.data,
                    cells * sizeof(std::uint16_t), cudaMemcpyDeviceToHost, p.stream));
                p.synchronize();
                observer(std::span<const std::uint16_t>(p.host_logits).first(cells),
                    static_cast<std::int32_t>(p.config().vocab_size),
                    predictor + 1 - first_target);
                predictor += columns;
            }
        }
        if (result.logprobs.size() != n - first_target) {
            throw std::logic_error("Gemma4 diagnostic scoring omitted suffix targets");
        }
        // Ordinary selected-logit scoring reuses this vector as exactly one vocabulary row.
        p.host_logits.resize(p.config().vocab_size);
        p.prefix_finish(false);
        p.clear();
        return result;
    } catch (...) {
        p.host_logits.resize(p.config().vocab_size);
        const bool recovered = recover_after_failure();
        if (!recovered) { throw std::runtime_error("Gemma4 diagnostic scoring failed and the device could not recover"); }
        throw;
    }
}
} // namespace infernix::models::gemma4
