// Real-artifact final-layer early-exit evidence; no downloads. Missing artifact skips (77).
// EVIDENCE fingerprints can be compared with retained pre-cutover baseline runs.
// INFERNIX_GEMMA4_ARTIFACT includes MTP unless --mtp 0; INFERNIX_GEMMA4_HOST_MIB sizes the Host tier.
// Optional arguments: --chunk 128|2048 --graphs 0|1 --mtp 0|1 (default: all combinations).
// Direct execution additionally compares full and KV-only calls exactly within each process.
#include "prefix_real_support.h"
#include "models/gemma4/program/kv_layout.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>
#include <utility>

namespace {
using namespace prefix_real;
using namespace infernix;
constexpr std::uint32_t kContext = 4352;
constexpr std::uint32_t kOutputs = 12;

struct Hash {
    std::uint64_t value = 14695981039346656037ULL;
    void bytes(const void* data, std::size_t size) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) { value = (value ^ p[i]) * 1099511628211ULL; }
    }
    template<class T> void scalar(const T& v) { bytes(&v, sizeof(v)); }
    template<class T> void vector(const std::vector<T>& v) {
        scalar(static_cast<std::uint64_t>(v.size()));
        bytes(v.data(), v.size() * sizeof(T));
    }
};
void emit(const char* name, std::uint32_t chunk, bool graphs, const Hash& hash) {
    std::printf("EVIDENCE %s chunk=%u graphs=%u %016llx\n", name, chunk, graphs ? 1U : 0U,
                static_cast<unsigned long long>(hash.value));
}

// A single native cache, with ordinary contiguous pages even for sliding layers. This makes every
// byte (including unwritten zero rows) inspectable without reaching into Program's private arena.
// The Program scenarios below separately exercise its real sliding rings and Host snapshots.
void execution_kv(const g4::execution::Parameters& p, DeviceContext& device, std::uint32_t chunk, KvCacheStorage kv) {
    const auto& config = p.model.config().text;
    const int pages = static_cast<int>((chunk + 1 + 63) / 64);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    DeviceBuffer table_memory(table.size() * sizeof(std::int32_t));
    table_memory.copy_from_host(table.data(), table_memory.bytes);
    Tensor table_view(table_memory.p, DType::I32, {pages});
    std::vector<DeviceBuffer> keys, values, key_scales, value_scales;
    std::vector<Tensor> key_views, value_views, key_scale_views, value_scale_views;
    std::vector<g4::Gemma4CacheLayer> caches;
    const auto count = p.layers.size();
    keys.reserve(count); values.reserve(count); key_views.reserve(count); value_views.reserve(count);
    key_scales.reserve(count); value_scales.reserve(count);
    key_scale_views.reserve(count); value_scale_views.reserve(count);
    caches.reserve(count);
    std::size_t workspace_bytes = g4::gemma4_logits_workspace_bytes(p.output_head, 1);
    for (const auto& layer : p.layers) {
        const auto layout = g4::kv_layer_layout(kv, layer.global_layer, layer.head_dim, layer.kv_heads,
                                                g4::KvStorageModel::KeyEqualsValue);
        keys.emplace_back(layout.code_page_bytes * pages); values.emplace_back(layout.code_page_bytes * pages);
        key_views.emplace_back(keys.back().p, layout.code_dtype, std::initializer_list<std::int32_t>{
            layer.head_dim, 64, layer.kv_heads, pages});
        value_views.emplace_back(values.back().p, layout.code_dtype, std::initializer_list<std::int32_t>{
            layer.head_dim, 64, layer.kv_heads, pages});
        key_scales.emplace_back(); value_scales.emplace_back();
        key_scale_views.emplace_back(); value_scale_views.emplace_back();
        if (layout.scale_groups != 0) {
            key_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages);
            value_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages);
            key_scale_views.back() = Tensor(key_scales.back().p, DType::FP16, {layout.scale_groups, 64, layer.kv_heads, pages});
            value_scale_views.back() = Tensor(value_scales.back().p, DType::FP16, {layout.scale_groups, 64, layer.kv_heads, pages});
        }
        caches.push_back({layout.format, &key_views.back(), &value_views.back(),
                          layout.scale_groups ? &key_scale_views.back() : nullptr,
                          layout.scale_groups ? &value_scale_views.back() : nullptr, &table_view});
        workspace_bytes = std::max(workspace_bytes, g4::gemma4_layer_workspace_bytes(layer, chunk));
    }
    WorkspaceArena workspace(workspace_bytes);
    ops::Fp8RowCublasContext gemm(device.stream);
    DeviceBuffer input_memory(4ULL * chunk), position_memory(4ULL * chunk);
    DeviceBuffer hidden_memory(2ULL * config.hidden_size * chunk), logits_memory(2ULL * config.vocab_size);
    const auto prompt = tokens(chunk + 1, 101);
    // The reference retains exact bytes at both the skipped-tail boundary and the final boundary.
    std::array<std::vector<std::vector<unsigned char>>, 2> reference;
    std::vector<std::uint16_t> reference_logits;
    Hash evidence;
    for (const bool exit : {false, true}) {
        device.synchronize();
        for (auto* buffers : {&keys, &values, &key_scales, &value_scales}) {
            for (auto& buffer : *buffers) { if (buffer.bytes != 0) { buffer.fill(); } }
        }
        for (int step = 0; step < 2; ++step) {
            const int width = step == 0 ? static_cast<int>(chunk) : 1;
            const int first = step == 0 ? 0 : static_cast<int>(chunk);
            std::vector<std::int32_t> positions(width);
            std::iota(positions.begin(), positions.end(), first);
            input_memory.copy_from_host(prompt.data() + first, 4ULL * width);
            position_memory.copy_from_host(positions.data(), 4ULL * width);
            Tensor input(input_memory.p, DType::I32, {width});
            Tensor pos(position_memory.p, DType::I32, {width});
            Tensor hidden(hidden_memory.p, DType::BF16, {static_cast<int>(config.hidden_size), width});
            g4::gemma4_layers(input, hidden, p.embedding, p.layers, caches, pos, gemm, workspace,
                             device.stream, !exit || step == 1);
            device.synchronize();
            std::size_t plane = 0;
            for (std::size_t layer = 0; layer < count; ++layer) {
                for (const auto* buffer : {&keys[layer], &values[layer], &key_scales[layer], &value_scales[layer]}) {
                    if (buffer->bytes == 0) { continue; }
                    std::vector<unsigned char> bytes(buffer->bytes);
                    buffer->copy_to_host(bytes.data(), bytes.size());
                    if (!exit) { reference[step].push_back(std::move(bytes)); }
                    else {
                        require(bytes == reference[step][plane], "KV-only forward changed a layer's K/V bytes");
                        evidence.vector(bytes);
                    }
                    ++plane;
                }
            }
            if (step == 1) {
                Tensor logits(logits_memory.p, DType::BF16, {static_cast<int>(config.vocab_size), 1});
                g4::gemma4_logits(hidden, logits, p.final_norm, p.output_head, config.final_logit_softcapping,
                                 config.rms_norm_eps, gemm, workspace, device.stream);
                device.synchronize();
                std::vector<std::uint16_t> result(config.vocab_size);
                logits_memory.copy_to_host(result.data(), logits_memory.bytes);
                if (!exit) { reference_logits = std::move(result); }
                else {
                    require(result == reference_logits, "KV-only prefix changed final full-vocabulary logits");
                    evidence.vector(result);
                }
            }
        }
    }
    emit(kv == KvCacheStorage::BFloat16 ? "execution-bf16-kv-and-logits" : "execution-fp8-kv-and-logits",
         chunk, false, evidence);
}

struct Trace {
    Tokens output;
    std::vector<std::uint16_t> logits;
    std::vector<std::uint32_t> columns;
    std::vector<std::uint64_t> counters;
    PromptReadout readout;
    Hash hash() const {
        Hash h; h.vector(output); h.vector(logits); h.vector(columns); h.vector(counters);
        h.vector(readout.logprobs); h.scalar(readout.top_token); h.scalar(readout.top_logprob);
        return h;
    }
};

Trace finish_trace(g4::Program& program, g4::SequenceHandle sequence, std::size_t prompt_size,
                   std::size_t vocab, std::vector<std::uint16_t>& selecting,
                   std::uint32_t& predictor, std::uint32_t& width) {
    Trace trace;
    bool readout = false;
    while (trace.output.size() < kOutputs) {
        auto pending = [&]() -> g4::PendingBatch {
            if (!trace.output.empty()) {
                const std::array sequences{sequence};
                const std::array budgets{rt::RoundBudget{kOutputs - static_cast<std::uint32_t>(trace.output.size())}};
                return program.decode(sequences, budgets);
            }
            for (;;) {
                auto progress = program.advance_prefill(sequence);
                if (progress.readout) { trace.readout = std::move(*progress.readout); readout = true; }
                if (progress.pending) { return std::move(*progress.pending); }
                require(!progress.complete, "prefill completed without a selecting distribution");
            }
        }();
        require(pending.row_failure(0) == rt::PendingRowFailure::None, "non-finite selecting logits");
        const auto n = pending.row_counts().empty() ? 1U : static_cast<std::uint32_t>(pending.row_counts()[0]);
        require(n > 0 && n <= kOutputs - trace.output.size(), "invalid licensed token count");
        require(width >= n && selecting.size() == vocab * width, "observer omitted full-vocabulary columns");
        require(predictor + 1 == prompt_size + trace.output.size(), "selecting column position mismatch");
        if (trace.output.empty()) {
            require(readout && trace.readout.logprobs.size() == 3, "prompt readout is absent");
            require(trace.readout.top_token == pending.tokens()[0], "greedy first token differs from prompt readout");
            require(std::isfinite(trace.readout.top_logprob), "prompt readout is not finite");
        }
        trace.columns.push_back(predictor); trace.columns.push_back(width); trace.columns.push_back(n);
        // Include unlicensed verification columns too: equal tokens alone do not prove carry equality.
        trace.logits.insert(trace.logits.end(), selecting.begin(), selecting.end());
        trace.output.insert(trace.output.end(), pending.tokens().begin(), pending.tokens().begin() + n);
        rt::CommitDecision decision;
        decision.accepted_tokens = n; decision.terminal = trace.output.size() == kOutputs;
        const std::array decisions{decision};
        const auto committed = program.commit(std::move(pending), decisions);
        const auto& c = committed.rows[0].speculative_counters;
        trace.counters.insert(trace.counters.end(), {c.rounds, c.drafted_tokens, c.accepted_tokens, c.fallback_steps});
    }
    const auto finished = program.finish(sequence);
    require(finished.status == rt::ConsumeStatus::Consumed, "sequence did not finish");
    const auto& s = finished.speculative;
    trace.counters.insert(trace.counters.end(), {s.rounds, s.drafted_tokens, s.accepted_tokens, s.fallback_steps});
    trace.counters.insert(trace.counters.end(), s.accepted_per_position.begin(), s.accepted_per_position.end());
    trace.counters.insert(trace.counters.end(), s.rounds_by_draft_length.begin(), s.rounds_by_draft_length.end());
    require(!s.enabled || (s.rounds > 0 && s.drafted_tokens > 0), "MTP did not exercise any draft rounds");
    return trace;
}

g4::SequenceHandle bind(g4::Program& program, const g4::RequestBasePlan& plan, g4::ResumeState* resume = nullptr) {
    require(static_cast<bool>(program.start_binding(plan, rt::LaneId{0}, {}, resume)), "binding refused");
    auto progress = program.poll_context({});
    require(progress.published && progress.sequence.has_value(), "binding did not publish");
    require(resume == nullptr || progress.replaying, "paused root binding did not require replay");
    if (progress.replaying) {
        while (!program.advance_replay(*progress.sequence).complete) {}
    }
    return *progress.sequence;
}

void generation(const g4::execution::Parameters& parameters, DeviceContext& device, const g4::Frontend& frontend,
                std::uint32_t chunk, bool graphs, bool mtp) {
    std::vector<std::uint16_t> selecting;
    std::uint32_t predictor = 0, width = 0;
    g4::ProgramOptions options;
    options.max_context = kContext; options.prefill_chunk = chunk; options.use_cuda_graph = graphs;
    options.kv_cache = KvCacheStorage::Fp8E4M3Row256; options.qk_mode = ops::Gemma4QkMode::Fp8;
    options.draft_tokens = mtp ? 3 : 0;
    options.prefix_cache = true; options.host_bytes = host_bytes_from_env();
    options.logits_observer = [&](std::uint32_t first, std::uint32_t columns, std::span<const std::uint16_t> rows) {
        predictor = first; width = columns; selecting.assign(rows.begin(), rows.end());
    };
    g4::Program program(parameters, device, options);
    const auto prompt = tokens(std::max(2101U, 2 * chunk + 3), 17);
    auto execution = greedy_execution(kOutputs, false);
    execution.readout_tokens = {7, 1000, 30000};
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), execution);
    const auto finish = [&](g4::SequenceHandle sequence) {
        return finish_trace(program, sequence, prompt.size(), parameters.model.config().text.vocab_size,
                            selecting, predictor, width);
    };
    const auto reference = finish(bind(program, plan));
    emit(mtp ? "mtp-cold" : "plain-cold", chunk, graphs, reference.hash());
    const auto repeat = finish(bind(program, plan));
    require(reference.hash().value == repeat.hash().value, "clean generation is not deterministic");

    // Pause at a full intermediate call, with no source candidate: this MUST replay from root.
    auto sequence = bind(program, plan);
    for (int call = 0; call < 2; ++call) {
        const auto partial = program.advance_prefill(sequence);
        require(!partial.complete && !partial.pending && partial.processed_prompt_tokens == chunk,
                "pause was not after an intermediate chunk");
    }
    require(program.start_pause(sequence), "pause refused");
    auto paused = program.poll_context({});
    require(paused.paused && paused.paused->frontier() == 2 * chunk, "pause lost its frontier");
    const auto replayed = finish(bind(program, plan, &*paused.paused));
    require(reference.hash().value == replayed.hash().value, "root replay changed logits, readout, tokens or MTP counters");
    emit(mtp ? "mtp-root-replay" : "plain-root-replay", chunk, graphs, replayed.hash());

    // A separate reusable plan publishes a snapshot while still stopped between prompt chunks.
    execution.allow_prefix_reuse = true;
    auto reusable = program.plan_request(frontend.prepare_tokens(prompt), execution);
    sequence = bind(program, reusable);
    const auto writes = program.hybrid_stats().host_image_writes;
    std::uint32_t frontier = 0;
    while (program.hybrid_stats().host_image_writes == writes) {
        const auto progress = program.advance_prefill(sequence);
        require(!progress.complete && !progress.pending, "no snapshot captured before the final prompt chunk");
        frontier += progress.processed_prompt_tokens;
    }
    require(program.abort(sequence).status == rt::ConsumeStatus::Consumed, "capture abort did not consume sequence");
    // Overwrite the live lane before restoring the captured Host image.
    const auto other = start(program, frontend, tokens(chunk + 1, 99), 1, false);
    require(other.sequence.has_value(), "overwrite did not bind");
    (void)complete(program, *other.sequence, 1);
    program.maintain();
    bool restored = false;
    for (const auto& source : program.hybrid_sources(reusable, nullptr)) {
        if (source.reused_tokens == 0) { continue; }
        require(source.reused_tokens <= frontier, "snapshot passed the interrupted frontier");
        if (!program.start_binding(reusable, rt::LaneId{0}, source)) { continue; }
        auto progress = program.poll_context({});
        require(progress.published && progress.sequence, "snapshot restore did not publish");
        require(progress.operations.state_restores == 1 && !progress.replaying,
                "snapshot binding fell back to root instead of restoring its captured state");
        const auto resumed = finish(*progress.sequence);
        require(reference.hash().value == resumed.hash().value,
                "snapshot resume changed logits, readout, tokens or MTP counters");
        emit(mtp ? "mtp-snapshot-resume" : "plain-snapshot-resume", chunk, graphs, resumed.hash());
        restored = true;
        break;
    }
    require(restored, "no intermediate captured snapshot was restored");
}

Hash score_hash(const ScoreResult& result) {
    Hash h; h.vector(result.logprobs); h.vector(result.top_ids); h.vector(result.top_logprobs);
    h.vector(result.candidate_logprobs); h.vector(result.logits); return h;
}
void scoring(const g4::execution::Parameters& parameters, DeviceContext& device, const g4::Frontend& frontend,
             std::uint32_t chunk, bool graphs) {
    g4::ProgramOptions options;
    options.max_context = kContext; options.prefill_chunk = chunk; options.use_cuda_graph = graphs;
    options.kv_cache = KvCacheStorage::Fp8E4M3Row256; options.qk_mode = ops::Gemma4QkMode::Fp8;
    options.causal_scoring = true;
    g4::Program program(parameters, device, options);
    const auto prompt = tokens(2 * chunk + 3, 91);
    // A whole call has no scored predictor. Then score across the next call boundary, including
    // the last column of the penultimate call and first two columns of the final call.
    const std::uint32_t first_target = chunk + 2;
    ScoreOptions score;
    score.top_k = 3; score.candidates_per_position = 2;
    score.logits_positions = {0, chunk - 3, chunk - 2, chunk - 1, chunk};
    for (std::size_t i = first_target; i < prompt.size(); ++i) {
        score.candidates.push_back(prompt[i]); score.candidates.push_back(7);
    }
    const auto result = program.causal_score(frontend.prepare_tokens(prompt), first_target, score);
    require(result.logprobs.size() == chunk + 1 && result.top_ids.size() == 3 * (chunk + 1),
            "scored range has the wrong size");
    require(result.candidate_logprobs.size() == 2 * result.logprobs.size(), "missing candidate columns");
    const auto vocab = parameters.model.config().text.vocab_size;
    require(result.logits.size() == score.logits_positions.size() * vocab, "missing full-vocabulary boundary logits");
    for (std::size_t i = 0; i < result.logprobs.size(); ++i) {
        require(std::isfinite(result.logprobs[i]) && result.logprobs[i] <= 0, "invalid teacher-forced logprob");
        require(std::abs(result.logprobs[i] - result.candidate_logprobs[2 * i]) < 1e-5F,
                "candidate target and teacher-forced column are misaligned");
    }
    for (std::size_t row = 0; row < score.logits_positions.size(); ++row) {
        const auto begin = result.logits.begin() + row * vocab;
        require(std::all_of(begin, begin + vocab, [](float v) { return std::isfinite(v); }), "non-finite scoring logits");
        const auto winner = static_cast<TokenId>(std::max_element(begin, begin + vocab) - begin);
        require(winner == result.top_ids[3 * score.logits_positions[row]], "boundary top token is misaligned");
    }
    const auto repeat = program.causal_score(frontend.prepare_tokens(prompt), first_target, score);
    require(score_hash(result).value == score_hash(repeat).value, "causal scoring is not deterministic");
    emit("causal-boundaries", chunk, graphs, score_hash(result));

    // The bound diagnostic route preserves serving's GEMV head and full-prompt call partition.
    // Its arithmetic is not the tiled causal-score head's oracle; retain its own fingerprint.
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), greedy_execution(1, false));
    const auto sequence = bind(program, plan);
    Hash diagnostic;
    std::uint32_t observed = 0;
    const auto bound = program.diagnostic_score_bound(sequence, first_target,
        [&](std::span<const std::uint16_t> rows, std::int32_t row_vocab, std::uint32_t position) {
            require(row_vocab == static_cast<std::int32_t>(vocab) && !rows.empty() && rows.size() % vocab == 0,
                    "bound diagnostic omitted full-vocabulary rows");
            require(position == observed, "bound diagnostic skipped or repeated scored positions");
            const auto columns = static_cast<std::uint32_t>(rows.size() / vocab);
            diagnostic.scalar(position); diagnostic.scalar(columns);
            diagnostic.bytes(rows.data(), rows.size_bytes());
            observed += columns;
        });
    require(observed == chunk + 1 && bound.logprobs.size() == observed, "bound diagnostic scored the wrong range");
    require(std::all_of(bound.logprobs.begin(), bound.logprobs.end(),
                       [](float v) { return std::isfinite(v) && v <= 0; }), "invalid bound diagnostic logprob");
    diagnostic.vector(bound.logprobs);
    emit("diagnostic-boundaries", chunk, graphs, diagnostic);
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set"); return 77;
    }
    try {
        std::uint32_t selected_chunk = 0;
        int selected_graphs = -1;
        int selected_mtp = -1;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "expected --chunk 128|2048, --graphs 0|1 or --mtp 0|1");
            const std::string option(argv[i]), value(argv[i + 1]);
            if (option == "--chunk" && (value == "128" || value == "2048")) {
                selected_chunk = value == "128" ? 128 : 2048;
            } else if (option == "--graphs" && (value == "0" || value == "1")) {
                selected_graphs = value == "0" ? 0 : 1;
            } else if (option == "--mtp" && (value == "0" || value == "1")) {
                selected_mtp = value == "0" ? 0 : 1;
            } else { throw std::runtime_error("expected --chunk 128|2048, --graphs 0|1 or --mtp 0|1"); }
        }
        DeviceContext device(0);
        for (const bool mtp : {false, true}) {
            if (selected_mtp >= 0 && selected_mtp != static_cast<int>(mtp)) { continue; }
            models::LoadOptions load;
            load.speculative = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
            auto model = g4::load_model(artifact, load, device);
            g4::execution::Parameters parameters(*model, device.stream);
            require(!mtp || parameters.draft.has_value(), "artifact must include the MTP assistant");
            g4::Frontend frontend(model->resources(), {.max_context = kContext});
            for (const std::uint32_t chunk : {128U, 2048U}) {
                if (selected_chunk != 0 && selected_chunk != chunk) { continue; }
                if (!mtp) {
                    execution_kv(parameters, device, chunk, KvCacheStorage::BFloat16);
                    execution_kv(parameters, device, chunk, KvCacheStorage::Fp8E4M3Row256);
                }
                for (const bool graphs : {false, true}) {
                    if (selected_graphs >= 0 && selected_graphs != static_cast<int>(graphs)) { continue; }
                    generation(parameters, device, frontend, chunk, graphs, mtp);
                    if (!mtp) { scoring(parameters, device, frontend, chunk, graphs); }
                }
            }
            device.synchronize();
        }
        std::puts("PASS early-exit invariants");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
