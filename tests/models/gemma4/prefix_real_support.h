#pragma once

// Shared by the real Gemma4 prefix-cache tests (test_prefix_scenarios_real.cpp,
// test_prefix_recovery_real.cpp): soft expectations, deterministic synthetic prompts, the Program
// rig (one model on one device, Programs created from it), the public Program transaction driver
// (plan -> hybrid_sources -> start_binding -> poll_context -> prefill/decode/commit/finish) and the
// public Engine helpers (settled statistics, idle gauges, cancellable streaming requests).
// INFERNIX_GEMMA4_ARTIFACT selects the artifact; INFERNIX_GEMMA4_HOST_MIB the prefix cache's Host tier
// (default 8192: several 31B images beside the blocks; the 26B floor is lower).

#include "core/device.h"
#include "infernix/engine.h"
#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/program.h"
#include "runtime/contract/execution.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace prefix_real {

namespace g4 = infernix::models::gemma4;
namespace rt = infernix::runtime;
using Tokens = std::vector<infernix::TokenId>;

inline int failures = 0;

// A soft expectation: reported, counted, the run continues.
inline void expect(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "  ok  " : "  FAIL", what.c_str());
    if (!ok) { ++failures; }
}
// A precondition the rest of a scenario cannot go on without.
inline void require(bool ok, const std::string& what) {
    if (!ok) { throw std::runtime_error(what); }
}

inline Tokens tokens(std::size_t count, std::uint32_t seed) {
    Tokens result;
    for (std::size_t i = 0; i < count; ++i) {
        seed = seed * 1664525U + 1013904223U;
        result.push_back(static_cast<infernix::TokenId>(1000U + (seed >> 8U) % 30000U));
    }
    return result;
}

inline Tokens concat(Tokens a, const Tokens& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

inline std::string difference(const Tokens& a, const Tokens& b) {
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i] != b[i]) { return "first difference at " + std::to_string(i); }
    }
    return a.size() == b.size() ? "identical" : "lengths differ";
}

inline std::uint64_t host_bytes_from_env() {
    const char* mib = std::getenv("INFERNIX_GEMMA4_HOST_MIB");
    return (mib != nullptr && *mib != '\0' ? std::strtoull(mib, nullptr, 10) : 8192ULL) << 20U;
}

// ---- the Program path --------------------------------------------------------------------------

struct Rig {
    infernix::DeviceContext device{0};
    std::unique_ptr<g4::Model> model;
    std::unique_ptr<g4::execution::Parameters> parameters;
    std::unique_ptr<g4::Frontend> frontend;

    Rig(const char* artifact, std::uint32_t max_context)
        : model(g4::load_model(artifact, {}, device)),
          parameters(std::make_unique<g4::execution::Parameters>(*model, device.stream)),
          frontend(std::make_unique<g4::Frontend>(model->resources(), g4::FrontendOptions{.max_context = max_context})) {}

    // A Program with the prefix cache on (`host_bytes` > 0) or off (0: the cold lane).
    [[nodiscard]] std::unique_ptr<g4::Program> program(std::uint32_t max_context, std::uint32_t chunk,
                                                       std::uint64_t host_bytes, infernix::KvCacheStorage kv,
                                                       infernix::ops::Gemma4QkMode qk) {
        g4::ProgramOptions options;
        options.max_context   = max_context;
        options.prefill_chunk = chunk;
        options.kv_cache      = kv;
        options.qk_mode       = qk;
        options.prefix_cache  = host_bytes != 0;
        options.host_bytes    = host_bytes;
        options.use_cuda_graph = false;
        return std::make_unique<g4::Program>(*parameters, device, options);
    }
};

inline rt::ResolvedExecutionOptions greedy_execution(std::uint32_t outputs, bool reuse) {
    rt::ResolvedExecutionOptions options;
    options.requested_output_tokens = outputs;
    options.sampling.temperature    = 0;
    options.sampling.top_k          = 20;
    options.sampling.top_p          = 1;
    options.sampling.min_p          = 0;
    options.sampling.seed           = 7;
    options.allow_prefix_reuse      = reuse;
    return options;
}

// One binding the way the Engine drives it: the sources in the Program's order until one binds.
struct Started {
    std::optional<g4::SequenceHandle> sequence; // empty when `cancel` stopped the restore
    std::uint32_t quoted = 0;                   // the frontier of the source that bound
    std::uint64_t reused = 0;                   // tokens the restore actually brought back
};

inline Started start(g4::Program& program, const g4::Frontend& frontend, const Tokens& prompt, std::uint32_t outputs,
                     bool reuse, rt::CancellationFlagView cancel = {}) {
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), greedy_execution(outputs, reuse));
    const std::uint64_t before = program.hybrid_stats().reused_tokens;
    for (const g4::SourceCandidate& source : program.hybrid_sources(plan, UINT32_MAX)) {
        const auto reservation = program.start_binding(plan, rt::LaneId{0}, source);
        if (!reservation) {
            require(!reservation.source_valid, "the lane refused a binding (busy or out of capacity)");
            continue;
        }
        Started out;
        out.quoted      = source.reused_tokens;
        auto progress   = program.poll_context(cancel);
        out.reused      = program.hybrid_stats().reused_tokens - before;
        if (progress.published && progress.sequence) { out.sequence = progress.sequence; }
        return out;
    }
    throw std::runtime_error("no source bound the request");
}

// Prefill to the first token's transaction (the pending batch is the caller's to commit or abort).
inline g4::PendingBatch first_token(g4::Program& program, g4::SequenceHandle sequence) {
    for (;;) {
        auto progress = program.advance_prefill(sequence);
        if (progress.pending) { return std::move(*progress.pending); }
        require(!progress.complete, "prefill completed without its first-token transaction");
    }
}

inline rt::CommitDecision decision(bool terminal = false, bool cancelled = false, bool failed = false) {
    rt::CommitDecision d;
    d.accepted_tokens = cancelled || failed ? 0U : 1U;
    d.terminal        = terminal;
    d.cancelled       = cancelled;
    d.failed          = failed;
    return d;
}

inline rt::CommitDisposition commit_one(g4::Program& program, g4::PendingBatch&& pending, rt::CommitDecision d) {
    const std::array decisions{d};
    return program.commit(std::move(pending), decisions).rows[0].disposition;
}

// `outputs` greedy tokens from a bound sequence, then a terminal finish.
inline Tokens complete(g4::Program& program, g4::SequenceHandle sequence, std::uint32_t outputs) {
    Tokens result;
    const std::array sequences{sequence};
    for (std::uint32_t i = 0; i < outputs; ++i) {
        const std::array budgets{rt::RoundBudget{outputs - i}};
        auto pending = i == 0 ? first_token(program, sequence) : program.decode(sequences, budgets);
        require(pending.row_failure(0) == rt::PendingRowFailure::None, "logits failed");
        result.push_back(pending.tokens()[0]);
        (void)commit_one(program, std::move(pending), decision(i + 1 == outputs));
    }
    require(program.finish(sequence).status == rt::ConsumeStatus::Consumed, "terminal sequence did not finish");
    return result;
}

// One whole request: the generated tokens, how many prompt tokens were restored and the quote.
struct Outcome {
    Tokens tokens;
    std::uint64_t reused = 0;
    std::uint32_t quoted = 0;
};

inline Outcome run(g4::Program& program, const g4::Frontend& frontend, const Tokens& prompt, std::uint32_t outputs,
                   bool reuse) {
    const Started started = start(program, frontend, prompt, outputs, reuse);
    require(started.sequence.has_value(), "the binding did not publish");
    return {complete(program, *started.sequence, outputs), started.reused, started.quoted};
}

// A release polls the Host copies without waiting for them, so a Device copy whose write-through was
// still in flight rests until the next call that polls. A trivial cold request (its binding waits for
// the stream and polls the cache, its abort releases) completes them: the gauges are then at rest.
inline void settle(g4::Program& program, const g4::Frontend& frontend) {
    const Started started = start(program, frontend, tokens(8, 1), 1, false);
    require(started.sequence.has_value(), "the settling request did not bind");
    (void)program.abort(*started.sequence);
}

// Nothing is leased, pinned or held after a request's release: the lane is free, no cached block
// keeps a Device copy (each is pinned by its sequence) and no snapshot is held for a queued request.
inline void expect_released(g4::Program& program, const g4::Frontend& frontend, const std::string& what) {
    settle(program, frontend);
    const auto usage = program.physical_usage();
    const auto stats = program.hybrid_stats();
    expect(usage.occupied == rt::ContextResourceUsage{} && stats.device_resident_blocks == 0 && stats.held_snapshots == 0,
           what + ": lane free, no Device copy, no hold (resident " + std::to_string(stats.device_resident_blocks) +
               ", held " + std::to_string(stats.held_snapshots) + ")");
}

// The cache counters of a Program at one moment, for a failure message.
inline std::string describe_stats(const g4::HybridPrefixCacheStats& s) {
    return "nodes " + std::to_string(s.nodes) + ", snapshots " + std::to_string(s.snapshots) + ", blocks inserted " +
           std::to_string(s.blocks_inserted) + " (duplicate " + std::to_string(s.blocks_duplicate) + "), taps created " +
           std::to_string(s.taps_created) + " skipped " + std::to_string(s.taps_skipped) + ", endpoints " +
           std::to_string(s.endpoints_created) + ", host image writes " + std::to_string(s.host_image_writes) +
           " block writes " + std::to_string(s.host_block_writes) + ", snapshot evictions " +
           std::to_string(s.host_snapshot_evictions) + ", free slabs " + std::to_string(s.host_free_slabs) + "/" +
           std::to_string(s.host_slabs) + ", restored tokens " + std::to_string(s.reused_tokens) +
           " (identity refusals are not exposed by HybridPrefixCacheStats)";
}

// The sources the Program would offer for `prompt` now: "frontier/path/cached-block-tokens" each, in order.
inline std::string describe_sources(g4::Program& program, const g4::Frontend& frontend, const Tokens& prompt) {
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), greedy_execution(1, true));
    std::string out = "sources";
    for (const g4::SourceCandidate& source : program.hybrid_sources(plan, UINT32_MAX)) {
        out += " [" + std::to_string(source.reused_tokens) + "/path " + std::to_string(static_cast<int>(source.reuse_path)) +
               "/blocks " + std::to_string(source.cached_block_tokens) + "]";
    }
    return out;
}

// ---- the public Engine -------------------------------------------------------------------------

// `max_context` is the test's own need (its largest prompt plus generation), not the model maximum, so
// the KV capacity fits beside the weights. Every Program and model an earlier section created has been
// destroyed by the caller; the device is synchronized here so their memory is returned before the
// Engine plans its capacity from the free VRAM.
inline infernix::EngineOptions engine_options(const char* artifact, std::uint32_t chunk, std::size_t host_bytes,
                                              std::uint32_t max_context) {
    (void)cudaDeviceSynchronize();
    infernix::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_context     = max_context;
    options.prefill_chunk   = chunk;
    options.max_concurrency = 1;
    options.kv_cache        = infernix::KvCacheStorage::BFloat16;
    options.logits_dtype    = infernix::LogitsDtype::BF16;
    options.speculative.backend_auto = false;
    options.speculative.backend      = infernix::SpeculativeBackend::None;
    // KvCapacityPolicy defaults to an explicit 2048 tokens: a larger --max-context needs its own capacity.
    options.kv_capacity = infernix::KvCapacityPolicy::explicit_capacity(max_context);
    options.context_cache.host_capacity_bytes = host_bytes; // 0: the cache off
    return options;
}

// What the Engine's startup sizing and the driver say about `options`, printed when a start fails.
inline void print_engine_plan(const infernix::EngineOptions& options) {
    constexpr double mib = 1048576.0;
    std::size_t free_bytes = 0, total_bytes = 0;
    (void)cudaMemGetInfo(&free_bytes, &total_bytes);
    std::fprintf(stderr, "engine start failed; cudaMemGetInfo free %.0f MiB of %.0f MiB\n", free_bytes / mib, total_bytes / mib);
    try {
        const infernix::EnginePlan plan = infernix::Engine::plan(options);
        std::fprintf(stderr,
                     "plan: fits %d, error '%s', gpu %s total %.0f MiB free %.0f MiB budget %.0f MiB\n"
                     "      weights %.0f MiB, kv %.0f MiB for %u tokens, minimum reservation %.0f MiB, workspace %.0f MiB, "
                     "graph allowance %.0f MiB, headroom %.0f MiB, planned %.0f MiB, left %.0f MiB, largest max-context %u "
                     "(requested %u, chunk %u)\n",
                     plan.fits ? 1 : 0, plan.error.c_str(), plan.gpu.name.c_str(), plan.gpu.total_bytes / mib,
                     plan.gpu.free_bytes / mib, plan.gpu.budget_bytes.value_or(0) / mib, plan.vram.weights_bytes / mib,
                     plan.vram.kv_bytes / mib, plan.vram.kv_tokens, plan.vram.minimum_reservation_bytes / mib,
                     plan.vram.workspace_bytes.value_or(0) / mib, plan.vram.graph_allowance_bytes.value_or(0) / mib,
                     plan.vram.headroom_bytes / mib, plan.vram.planned_bytes / mib, plan.vram.left_bytes / mib,
                     plan.max_context.value_or(0), options.max_context, options.prefill_chunk);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "plan: Engine::plan threw: %s\n", error.what());
    }
}

// Starts an Engine; if the start fails, prints the plan (the failure path only) and rethrows.
inline infernix::Engine start_engine(infernix::EngineOptions options) {
    try {
        return infernix::Engine(options);
    } catch (...) {
        print_engine_plan(options);
        throw;
    }
}

inline infernix::RequestOptions greedy(std::uint32_t outputs, bool reuse, bool model_stops = false) {
    infernix::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.top_k          = 20;
    options.execution.allow_prefix_reuse      = reuse;
    options.stop.include_model_defaults       = model_stops;
    return options;
}

struct Served {
    Tokens tokens;
    std::uint32_t reused = 0, prompt = 0;
    std::string content;
};

inline Served serve(infernix::GenerationResult result) {
    return {result.generated_token_ids, result.reused_prompt_tokens, result.prompt.prompt_tokens, result.content};
}
inline Served generate(infernix::Engine& engine, const Tokens& prompt, std::uint32_t outputs, bool reuse) {
    return serve(engine.generate(engine.prepare_tokens(prompt), greedy(outputs, reuse)));
}

// The Engine publishes its statistics after it answers a request: waits (at most 10 s) until `ready` holds.
inline infernix::RuntimeStats stats_when(infernix::Engine& engine,
                                         const std::function<bool(const infernix::RuntimeStats&)>& ready) {
    infernix::RuntimeStats stats = engine.runtime_stats();
    for (int i = 0; i < 500 && !ready(stats); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        stats = engine.runtime_stats();
    }
    return stats;
}

// No request anywhere, no lane resources, no Device copy of a cached block, no hold.
inline bool idle(const infernix::RuntimeStats& s) {
    return s.running_requests == 0 && s.prefilling_requests == 0 && s.decode_ready_requests == 0 &&
           s.waiting_requests == 0 && s.paused_requests == 0 && s.replaying_requests == 0 &&
           s.materializing_requests == 0 && s.terminal_pending_requests == 0 && s.device_main_kv_occupied_pages == 0 &&
           s.device_state_occupied_slots == 0 && s.hybrid_held_snapshots == 0;
}
// Idle at baseline. The trivial cold request in between polls the cache (as Program's settle does), so
// that Device copies whose write-through had not been retired are gone from the gauges.
inline infernix::RuntimeStats settled(infernix::Engine& engine, const std::string& what) {
    (void)stats_when(engine, idle);
    (void)generate(engine, tokens(8, 3), 1, false);
    const auto at_rest = [](const infernix::RuntimeStats& s) { return idle(s) && s.hybrid_cached_blocks == 0; };
    const auto stats = stats_when(engine, at_rest);
    expect(at_rest(stats), what + ": Engine idle at baseline (running " + std::to_string(stats.running_requests) +
                               ", kv pages " + std::to_string(stats.device_main_kv_occupied_pages) + ", slots " +
                               std::to_string(stats.device_state_occupied_slots) + ", device blocks " +
                               std::to_string(stats.hybrid_cached_blocks) + ", held " +
                               std::to_string(stats.hybrid_held_snapshots) + ")");
    return stats;
}

// A streaming consumer that raises a cancellation once the prompt or the output reaches a position.
struct CancelSink final : infernix::OutputSink {
    std::atomic<bool> cancel{false};
    std::uint32_t prompt_threshold = 0, token_threshold = 0;
    void start(infernix::GenerationStart) override {}
    void progress(infernix::PromptProgress p) override {
        if (prompt_threshold != 0 && p.processed_prompt_tokens >= prompt_threshold) { cancel = true; }
    }
    void timing(infernix::GenerationTimingObservation t) override {
        if (token_threshold != 0 && t.generated_tokens >= token_threshold) { cancel = true; }
    }
    void publish(infernix::OutputDelta) override {}
    void tool_call(infernix::ToolCallDelta) override {}
};

inline infernix::GenerationResult generate_cancelled(infernix::Engine& engine, const Tokens& prompt,
                                                     std::uint32_t outputs, CancelSink& sink) {
    const infernix::GenerationObservationOptions observation{.live_timings = true, .prompt_progress = true};
    auto handle = engine.submit(engine.prepare_tokens(prompt), greedy(outputs, true),
                                infernix::OutputConsumerMode::Streaming, observation);
    return handle.wait(&sink, infernix::CancellationView([&sink] { return sink.cancel.load(); }));
}

} // namespace prefix_real
