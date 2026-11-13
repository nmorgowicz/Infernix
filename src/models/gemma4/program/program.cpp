#include "models/gemma4/program/program.h"

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/startup.h"
#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/execution/text_forward.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "infernix/ops/sampling.h"
#include "infernix/ops/target_logprobs.h"
#include "infernix/ops/top_logprobs.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace infernix::models::gemma4 {
namespace {
constexpr std::uint32_t kPage = ops::kGemma4KvPageTokens;
constexpr int kScoreTile = 128;
constexpr std::array<std::uint32_t, 6> kGraphWidths{32, 64, 128, 256, 512, 1024};
constexpr std::uint64_t kGraphAllowance = 64ULL << 20;
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
    if (o.max_concurrency != 1) { throw std::invalid_argument("Gemma4 P4 requires max_concurrency=1"); }
    if (o.reproducible) { throw std::invalid_argument("Gemma4 P4 does not support --reproducible"); }
    if (o.prefix_cache) { throw std::invalid_argument("Gemma4 P4 requires prefix_cache=false"); }
    if (o.kv_cache != KvCacheStorage::BFloat16) {
        throw std::invalid_argument("Gemma4 P4 requires BF16 K and V storage");
    }
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
        const std::uint64_t bytes = 4ULL * layer.head_dim * layer.kv_heads * kPage;
        (layer.global_layer ? global_page_bytes : sliding_page_bytes) += bytes;
        plan.workspace = std::max<std::uint64_t>(plan.workspace,
            gemma4_layer_workspace_bytes(layer, static_cast<int>(options.prefill_chunk)));
    }
    // One logits tile and the sampler share the arena, not concurrent layer intermediates.
    plan.workspace = std::max<std::uint64_t>(plan.workspace,
        gemma4_logits_workspace_bytes(geometry.output_head, options.causal_scoring ? kScoreTile : 1));
    plan.workspace = std::max<std::uint64_t>(plan.workspace,
        gemma4_logits_workspace_bytes(geometry.output_head, 1) +
        ops::sampling_workspace_capacity_bytes(static_cast<int>(public_tokens), 1, 1));
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
};
struct ResumeData {
    const void* owner = nullptr;
    std::shared_ptr<const RequestPlan> plan;
    std::vector<TokenId> ledger;
    std::uint32_t frontier = 0;
    bool prefilled = false;
    bool finishable = false;
    GenerationTimings timings;
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
    std::vector<DeviceBuffer> keys, values;
    std::vector<Tensor> key_views, value_views;
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
        std::vector<std::int32_t> table(plan.kv_pages);
        std::iota(table.begin(), table.end(), 0);
        global_table_storage.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        for (auto& page : table) { page %= static_cast<int>(ring_count); }
        ring_table_storage.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        global_table = Tensor(global_table_storage.p, DType::I32, {static_cast<int>(plan.kv_pages)});
        ring_table = Tensor(ring_table_storage.p, DType::I32, {static_cast<int>(plan.kv_pages)});
        const auto layers = p.layers.size();
        keys.reserve(layers); values.reserve(layers); key_views.reserve(layers); value_views.reserve(layers);
        caches.reserve(layers);
        for (const auto& layer : p.layers) {
            const auto pages = layer.global_layer ? plan.kv_pages : ring_count;
            const auto bytes = 2ULL * layer.head_dim * kPage * layer.kv_heads * pages;
            keys.emplace_back(static_cast<std::size_t>(bytes));
            values.emplace_back(static_cast<std::size_t>(bytes));
            key_views.emplace_back(keys.back().p, DType::BF16,
                std::initializer_list<std::int32_t>{layer.head_dim, static_cast<int>(kPage), layer.kv_heads, static_cast<int>(pages)});
            value_views.emplace_back(values.back().p, DType::BF16,
                std::initializer_list<std::int32_t>{layer.head_dim, static_cast<int>(kPage), layer.kv_heads, static_cast<int>(pages)});
        }
        for (std::size_t i = 0; i < layers; ++i) {
            caches.push_back({.format = ops::Gemma4KvFormat::BFloat16,
                .k_pages = &key_views[i], .v_pages = &value_views[i],
                .block_table = p.layers[i].global_layer ? &global_table : &ring_table,
                .qk_mode = ops::Gemma4QkMode::BFloat16});
        }
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
    void head_body(int first, int n) {
        workspace.reset();
        auto out = logits(n);
        gemma4_logits(hidden_columns(first, n), out, parameters.final_norm, parameters.output_head,
            config().final_logit_softcapping, config().rms_norm_eps, gemm, workspace, stream);
    }
    void sample_body(std::int32_t purpose) {
        auto out = step.slice(0, 0, 1);
        auto logical = step.slice(0, 1, 1);
        auto values = logits(1);
        ops::sample(values, out, static_cast<int>(config().vocab_size), sampling_device,
            logical, purpose, workspace, stream);
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
    void upload_sampler() {
        const auto& s = request->sampling;
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
    if (!std::isfinite(s.temperature) || s.temperature < 0 || s.top_k < 1 || s.top_k > 20 ||
        !std::isfinite(s.top_p) || s.top_p <= 0 || s.top_p > 1 ||
        !std::isfinite(s.min_p) || s.min_p < 0 || s.min_p > 1 ||
        !std::isfinite(s.presence_penalty) || !std::isfinite(s.frequency_penalty)) {
        throw std::invalid_argument("Gemma4 P4 sampling requires finite parameters and top_k in [1,20]");
    }
    auto plan = std::make_shared<detail::RequestPlan>();
    plan->owner = impl_.get(); plan->tokens = std::move(prompt.tokens_);
    plan->sampling = s; plan->readout_tokens = options.readout_tokens;
    auto& summary = plan->summary;
    summary.prompt_tokens = static_cast<std::uint32_t>(plan->tokens.size());
    summary.requested_output_tokens = options.requested_output_tokens;
    summary.effective_output_tokens = std::min(options.requested_output_tokens,
        impl_->capacity() - summary.prompt_tokens + 1);
    summary.effective_limit_reason = summary.effective_output_tokens < options.requested_output_tokens
        ? FinishReason::ContextCapacity : FinishReason::OutputLimit;
    summary.publish_continuation = false;
    return RequestBasePlan(std::move(plan));
}
std::vector<SourceCandidate> Program::hybrid_sources(const RequestBasePlan& base, std::uint32_t) {
    if (!base.impl_ || base.impl_->owner != impl_.get()) { throw std::logic_error("Gemma4 foreign request plan"); }
    return {SourceCandidate{}};
}
BindingReservation Program::start_binding(const RequestBasePlan& base, runtime::LaneId lane,
    const SourceCandidate& source, ResumeState* resume, ExecutionUnitKind, std::uint32_t) {
    auto& p = *impl_;
    if (!base.impl_ || base.impl_->owner != impl_.get()) { throw std::logic_error("Gemma4 foreign request plan"); }
    if (lane.value != 0) { throw std::invalid_argument("Gemma4 has only lane zero"); }
    if (source.reused_tokens != 0 || source.cached_block_tokens != 0 || source.reuse_path != PrefixReusePath::Root) {
        return {.source_valid = false};
    }
    if (p.binding || p.occupied || p.pending || p.paused) { return {}; }
    if (resume && (!resume->impl_ || resume->impl_->owner != &p || resume->impl_->plan != base.impl_)) {
        throw std::logic_error("Gemma4 foreign or consumed replay state");
    }
    // Stage all potentially throwing host copies before taking ownership of the binding.
    auto ledger = resume ? resume->impl_->ledger : base.impl_->tokens;
    ledger.reserve(p.capacity() + 1ULL);
    p.request = base.impl_; p.ledger = std::move(ledger);
    if (resume) { p.binding_resume = std::move(resume->impl_); }
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
        CUDA_CHECK(cudaStreamSynchronize(p.stream));
        p.occupied = true; p.binding = false; p.fed = p.physical_written = 0;
        p.prefilled = false; p.finishable = false; p.timings = {};
        if (p.binding_resume) {
            p.replay_target = p.binding_resume->frontier;
            p.prefilled = p.binding_resume->prefilled; p.finishable = p.binding_resume->finishable;
            p.timings = p.binding_resume->timings; p.replaying = p.replay_target != 0;
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
    auto data = std::make_unique<detail::ResumeData>();
    data->owner = &p; data->plan = p.request; data->ledger = p.ledger;
    data->frontier = p.fed; data->prefilled = p.prefilled; data->finishable = p.finishable; data->timings = p.timings;
    // No snapshot is claimed. Replay will rebuild the exact committed target frontier from root.
    p.clear(); p.paused.emplace(ResumeState(std::move(data))); return true;
}
ReplayProgress Program::advance_replay(SequenceHandle sequence, runtime::ExecutionTiming* failed_timing) {
    auto& p = *impl_; p.require(sequence);
    if (!p.replaying || p.pending) { throw std::logic_error("Gemma4 sequence is not replaying"); }
    runtime::ExecutionTimingRecorder timing(runtime::ExecutionTimingPhase::Submit, failed_timing);
    try {
        const auto n = std::min(p.options.prefill_chunk, p.replay_target - p.fed);
        p.run_layers(std::span<const TokenId>(p.ledger).subspan(p.fed, n), p.fed, &timing);
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
        const auto n = std::min(p.options.prefill_chunk, p.request->summary.prompt_tokens - p.fed);
        p.run_layers(std::span<const TokenId>(p.ledger).subspan(p.fed, n), p.fed, &timing);
        PrefillProgress progress;
        progress.summary.prompt_tokens = p.request->summary.prompt_tokens;
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
    if (failed || decision.cancelled) { p.clear(); }
    else { p.finishable = decision.terminal; }
    return result;
}
DiscardResult Program::abort_pending(PendingBatch&& batch) noexcept {
    auto& p = *impl_;
    if (!p.owns(batch)) { return {}; }
    p.consume(batch); p.clear();
    return {.status = runtime::ConsumeStatus::Consumed, .row_count = 1};
}
FinishResult Program::finish(SequenceHandle sequence) noexcept {
    auto& p = *impl_;
    if (!p.owns(sequence) || p.pending || !p.finishable || !p.state_valid) { return {}; }
    FinishResult result; result.status = runtime::ConsumeStatus::Consumed; result.timings = p.timings;
    p.clear(); return result;
}
AbortResult Program::abort(SequenceHandle sequence) noexcept {
    auto& p = *impl_;
    if (!p.owns(sequence)) { return {}; }
    AbortResult result; result.status = runtime::ConsumeStatus::Consumed; result.timings = p.timings;
    p.clear(); return result;
}
void Program::fail_all_cleanup() noexcept {
    (void)settle_after_memory_exhausted();
    impl_->clear();
}
bool Program::recover_after_failure() noexcept {
    // The caller has settled an OOM once already; this also checks the real stream/device state
    // for other recoverable failures. Never claim recovery while earlier writes are still live.
    const bool usable = settle_after_memory_exhausted();
    impl_->clear(); return usable;
}
void Program::shutdown_cleanup() noexcept {
    if (!impl_) { return; }
    const auto status = cudaStreamSynchronize(impl_->stream);
    if (status != cudaSuccess) { report_cuda_cleanup_failure("Gemma4 shutdown synchronize", status); }
    impl_->clear();
}
void Program::hybrid_hold_continuation(std::chrono::steady_clock::time_point) {
    throw std::logic_error("Gemma4 P4 prefix cache is disabled; continuation holds are unsupported");
}
runtime::prefix_cache::PersistResult Program::attach_prefix_cache_file(const std::filesystem::path&,
    std::string, const StartupObserver&) {
    throw std::invalid_argument("Gemma4 P4 prefix cache is disabled; persistence is unsupported");
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
    result.kv_cache = KvCacheStorage::BFloat16;
    result.kv_payload_bytes = p.plan.kv - 8ULL * p.plan.kv_pages;
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
} // namespace infernix::models::gemma4
