// Real-artifact Program transaction coverage. No downloads; absent artifact skips (77).
// Compare only identical speculative/pause schedules, never plain and wide verification forwards.
#include "prefix_real_support.h"

#include <algorithm>
#include <numeric>

namespace {
using namespace prefix_real;
using infernix::SpeculativeStats;

void metrics(const SpeculativeStats& s, std::uint32_t drafts) {
    require(s.enabled && s.backend == infernix::SpeculativeBackend::Mtp && s.draft_window == drafts,
            "MTP identity lost");
    require(s.accepted_tokens <= s.drafted_tokens, "accepted exceeds drafted");
    require(std::accumulate(s.accepted_per_position.begin(), s.accepted_per_position.end(), std::uint64_t{0}) ==
                s.accepted_tokens, "position histogram lost accepted drafts");
    require(std::accumulate(s.rounds_by_draft_length.begin(), s.rounds_by_draft_length.end(), std::uint64_t{0}) ==
                s.rounds && s.round_steps.size() == s.rounds, "round history lost");
    std::uint64_t proposed = 0, accepted = 0;
    for (const auto& step : s.round_steps) {
        require(step.drafted > 0 && step.drafted <= drafts && step.accepted <= step.drafted && !step.ngram,
                "invalid MTP round history");
        proposed += step.drafted;
        accepted += step.accepted;
    }
    require(proposed == s.drafted_tokens && accepted == s.accepted_tokens, "round history counters disagree");
}

bool same_metrics(const SpeculativeStats& a, const SpeculativeStats& b) {
    if (a.backend != b.backend || a.enabled != b.enabled || a.draft_window != b.draft_window ||
        a.rounds != b.rounds || a.drafted_tokens != b.drafted_tokens || a.accepted_tokens != b.accepted_tokens ||
        a.fallback_steps != b.fallback_steps || a.accepted_per_position != b.accepted_per_position ||
        a.rounds_by_draft_length != b.rounds_by_draft_length || a.round_steps.size() != b.round_steps.size()) { return false; }
    for (std::size_t i = 0; i < a.round_steps.size(); ++i) {
        if (a.round_steps[i].drafted != b.round_steps[i].drafted ||
            a.round_steps[i].accepted != b.round_steps[i].accepted || a.round_steps[i].ngram != b.round_steps[i].ngram) {
            return false;
        }
    }
    return true;
}

void released(g4::Program& p) {
    require(p.physical_usage().occupied == rt::ContextResourceUsage{}, "transaction leaked lane resources");
    const auto s = p.hybrid_stats();
    require(s.held_snapshots == 0 && s.device_resident_blocks == 0 && !p.has_context_transaction(),
            "transaction leaked cache/context resources");
}

g4::CommitRowResult commit(g4::Program& p, g4::PendingBatch&& batch, std::uint32_t count,
                            bool terminal = false, bool cancelled = false, bool failed = false) {
    rt::CommitDecision d;
    d.accepted_tokens = count;
    d.terminal = terminal;
    d.cancelled = cancelled;
    d.failed = failed;
    const std::array decisions{d};
    auto result = p.commit(std::move(batch), decisions);
    require(result.row_count == 1 && batch.row_count() == 0, "commit did not consume pending capability");
    return std::move(result.rows[0]);
}

std::uint32_t extent(const g4::PendingBatch& batch) {
    require(batch.row_count() == 1 && batch.row_failure(0) == rt::PendingRowFailure::None &&
            (batch.row_counts().empty() || (batch.row_counts().size() == 1 && batch.row_counts()[0] > 0)),
            "invalid pending row");
    const auto n = batch.row_counts().empty() ? 1U : static_cast<std::uint32_t>(batch.row_counts()[0]);
    require(n <= batch.row_stride() && batch.tokens().size() >= n, "invalid pending extent");
    return n;
}

g4::PendingBatch decode(g4::Program& p, g4::SequenceHandle sequence, std::uint32_t budget) {
    const std::array sequences{sequence};
    const std::array budgets{rt::RoundBudget{budget}};
    return p.decode(sequences, budgets);
}

g4::SequenceHandle pause_replay(g4::Program& p, const g4::RequestBasePlan& plan,
                                g4::SequenceHandle sequence, std::uint32_t frontier, std::uint32_t expected_replay) {
    const auto predicted = p.pause_replay_tokens(sequence);
    require(predicted == expected_replay, "pause replay quote disagrees with the checkpoint schedule");
    require(p.start_pause(sequence), "pause refused");
    auto paused = p.poll_context({});
    require(paused.paused && paused.paused->frontier() == frontier, "pause frontier changed");
    CUDA_CHECK(cudaDeviceSynchronize());
    p.maintain();
    released(p);
    for (const auto& source : p.hybrid_sources(plan, &*paused.paused)) {
        if (!p.start_binding(plan, rt::LaneId{0}, source, &*paused.paused)) { continue; }
        auto bound = p.poll_context({});
        require(bound.published && bound.sequence.has_value(), "resume did not publish");
        std::uint32_t replayed = 0;
        if (bound.replaying) {
            for (;;) {
                auto progress = p.advance_replay(*bound.sequence);
                replayed += progress.processed_tokens;
                if (progress.complete) { break; }
            }
        }
        require(replayed == predicted, "resume did not replay exactly the quoted tokens");
        return *bound.sequence;
    }
    throw std::runtime_error("no resume source bound");
}

struct Outcome { Tokens tokens; SpeculativeStats stats; };

Outcome continuation(g4::Program& p, const g4::Frontend& frontend, const Tokens& prompt,
                     std::uint32_t drafts, bool pause) {
    constexpr std::uint32_t outputs = 65;
    auto plan = p.plan_request(frontend.prepare_tokens(prompt), greedy_execution(outputs, false));
    std::optional<g4::SequenceHandle> bound;
    for (const auto& source : p.hybrid_sources(plan, nullptr)) {
        if (!p.start_binding(plan, rt::LaneId{0}, source)) { continue; }
        auto progress = p.poll_context({});
        require(progress.published && progress.sequence.has_value(), "cold binding did not publish");
        bound = progress.sequence;
        break;
    }
    require(bound.has_value(), "cold request did not bind");
    auto sequence = *bound;
    auto first = first_token(p, sequence);
    Outcome out;
    out.tokens.push_back(first.tokens()[0]);
    (void)commit(p, std::move(first), 1);
    if (pause) {
        const auto frontier = static_cast<std::uint32_t>(prompt.size());
        sequence = pause_replay(p, plan, sequence, frontier, frontier);
    }
    bool paused_round = false;
    std::uint64_t previous_rounds = 0;
    while (out.tokens.size() < outputs) {
        auto batch = decode(p, sequence, outputs - static_cast<std::uint32_t>(out.tokens.size()));
        const auto n = extent(batch);
        out.tokens.insert(out.tokens.end(), batch.tokens().begin(), batch.tokens().begin() + n);
        const bool terminal = out.tokens.size() == outputs;
        const auto row = commit(p, std::move(batch), n, terminal);
        require(row.speculative_counters.rounds >= previous_rounds, "resume reset speculative counters");
        previous_rounds = row.speculative_counters.rounds;
        if (pause && !paused_round && !terminal && previous_rounds >= 2) {
            const auto frontier = static_cast<std::uint32_t>(prompt.size() + out.tokens.size() - 1);
            sequence = pause_replay(p, plan, sequence, frontier, frontier);
            paused_round = true;
        }
    }
    auto finished = p.finish(sequence);
    require(finished.status == rt::ConsumeStatus::Consumed, "terminal finish refused");
    out.stats = std::move(finished.speculative);
    metrics(out.stats, drafts);
    require(out.stats.rounds == previous_rounds && out.stats.rounds > 0, "finish lost speculative counters");
    if (pause) { require(paused_round, "no speculative frontier was paused"); }
    released(p);
    return out;
}

// A committed first token supplies the anchor; every exit below discards real wide verification work.
void exits(g4::Program& p, const g4::Frontend& frontend, const Tokens& prompt, std::uint32_t drafts) {
    const auto reference = continuation(p, frontend, prompt, drafts, false);
    for (int mode = 0; mode < 5; ++mode) {
        auto started = start(p, frontend, prompt, 32, false);
        require(started.sequence.has_value(), "exit request did not bind");
        const auto sequence = *started.sequence;
        auto first = first_token(p, sequence);
        (void)commit(p, std::move(first), 1);
        auto batch = decode(p, sequence, 32);
        const auto n = extent(batch);
        // Refused decisions must leave the pending capability intact for the valid operation below.
        for (const auto invalid : {rt::CommitDecision{.accepted_tokens = n + 1},
                                   rt::CommitDecision{.accepted_tokens = 1, .terminal = true, .cancelled = true},
                                   rt::CommitDecision{.accepted_tokens = 0, .failed = true}}) {
            bool refused = false;
            try {
                const std::array decisions{invalid};
                (void)p.commit(std::move(batch), decisions);
            } catch (const std::logic_error&) { refused = true; }
            require(refused && batch.row_count() == 1, "invalid decision consumed a pending capability");
        }
        if (mode == 4) {
            require(p.abort_pending(std::move(batch)).status == rt::ConsumeStatus::Consumed, "abort_pending refused");
            require(p.abort_pending(std::move(batch)).status == rt::ConsumeStatus::InvariantMismatch,
                    "aborted capability remained reusable");
        } else {
            // Zero-token and strict partial terminal commits both exercise post-verification truncation.
            const auto kept = mode == 0 ? 0U : mode == 1 ? n - 1U : 0U;
            const auto row = commit(p, std::move(batch), kept, true, mode == 2, mode == 3);
            const auto expected = mode < 2 ? rt::CommitDisposition::Finishable :
                mode == 2 ? rt::CommitDisposition::CancelledReleased : rt::CommitDisposition::FailedReleased;
            require(row.disposition == expected, "wrong terminal/cancel/failure disposition");
            if (mode < 2) {
                const auto finish = p.finish(sequence);
                require(finish.status == rt::ConsumeStatus::Consumed, "partial terminal finish refused");
                metrics(finish.speculative, drafts);
                require(finish.speculative.rounds == 1 && finish.speculative.accepted_tokens == kept,
                        "partial terminal counted uncommitted drafts");
            } else {
                metrics(row.speculative, drafts);
                require(row.speculative.rounds == 1 && row.speculative.accepted_tokens == 0,
                        "discarded round counted accepted drafts");
            }
        }
        released(p);
        const auto a = continuation(p, frontend, prompt, drafts, false);
        require(a.tokens == reference.tokens && same_metrics(a.stats, reference.stats),
                "exit contaminated next valid cold request");
    }
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { std::puts("skip: INFERNIX_GEMMA4_ARTIFACT is not set"); return 77; }
    try {
        infernix::DeviceContext device(0);
        infernix::models::LoadOptions load;
        load.speculative = infernix::SpeculativeBackend::Mtp;
        auto model = g4::load_model(artifact, load, device);
        const g4::execution::Parameters parameters(*model, device.stream);
        // --chunk N (default 64): a chunk of 1,024 or more runs the prompt calls wider than 513 tokens on the staged ring's
        // staging ring (ProgramOptions::staged_sliding_ring); the prompts below wrap the full-size ring either way.
        std::uint32_t chunk = 64;
        for (int i = 1; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--chunk") { chunk = static_cast<std::uint32_t>(std::stoul(argv[i + 1])); }
        }
        const auto wrap = ((chunk + model->config().text.sliding_window - 1 + 63) / 64) * 64;
        const auto capacity = std::max(wrap + 256, 3 * chunk + 64);
        g4::Frontend frontend(model->resources(), {.max_context = capacity});
        for (const auto drafts : {1U, 7U}) {
            g4::ProgramOptions options;
            options.max_context = capacity;
            options.prefill_chunk = chunk;
            options.draft_tokens = drafts;
            options.use_cuda_graph = false;
            const char* kv = std::getenv("INFERNIX_GEMMA4_KV");
            if (kv && std::string(kv) == "fp8") { options.kv_cache = infernix::KvCacheStorage::Fp8E4M3Row256; }
            const char* qk = std::getenv("INFERNIX_GEMMA4_QK");
            if (qk && std::string(qk) == "fp8") { options.qk_mode = infernix::ops::Gemma4QkMode::Fp8; }
            g4::Program program(parameters, device, options);
            std::uint64_t accepted = 0, rejected = 0;
            for (const auto length : {1U, 63U, 64U, 65U, wrap - 1, wrap, wrap + 1}) {
                std::printf("MTP transactions d%u prompt/frontier %u (ring wrap %u)\n", drafts, length, wrap);
                const auto prompt = tokens(length, length + 17);
                for (const bool pause : {false, true}) {
                    const auto a = continuation(program, frontend, prompt, drafts, pause);
                    const auto b = continuation(program, frontend, prompt, drafts, pause);
                    require(a.tokens == b.tokens && same_metrics(a.stats, b.stats), "same schedule continuation differs");
                    accepted += a.stats.accepted_tokens;
                    rejected += a.stats.drafted_tokens - a.stats.accepted_tokens;
                }
                exits(program, frontend, prompt, drafts);
            }
            require(accepted > 0 && rejected > 0, "matrix failed to exercise both draft acceptance and rejection");
            for (const bool free_resume : {false, true}) {
                auto cached_options = options;
                cached_options.prefix_cache = true;
                cached_options.host_bytes = host_bytes_from_env();
                cached_options.free_cache_resume = free_resume;
                g4::Program cached(parameters, device, cached_options);
                const auto prompt = tokens(3 * chunk + 1, 93);
                auto plan = cached.plan_request(frontend.prepare_tokens(prompt), greedy_execution(16, true));
                const auto sources = cached.hybrid_sources(plan, nullptr);
                require(sources.size() == 1 && sources.front().reused_tokens == 0,
                        "an empty Host cache offered a non-root source");
                require(static_cast<bool>(cached.start_binding(plan, rt::LaneId{0}, sources.front())),
                        "cached pause test did not bind");
                auto bound = cached.poll_context({});
                require(bound.published && bound.sequence, "cached pause binding did not publish");
                auto sequence = *bound.sequence;
                std::uint64_t seen_publications = 0;
                bool notified = false;
                const auto observe_publication = [&](const g4::PrefillProgress& progress) {
                    const auto stats = cached.hybrid_stats();
                    const auto now = stats.taps_created + stats.endpoints_created;
                    require(progress.snapshot_published == (now != seen_publications),
                            "prefill notification disagrees with actual Host snapshot publication");
                    seen_publications = now;
                    notified = notified || progress.snapshot_published;
                };
                for (int call = 0; call < 2; ++call) {
                    auto progress = cached.advance_prefill(sequence);
                    observe_publication(progress);
                    require(!progress.pending && !progress.complete, "prefill reached the output too early");
                }
                sequence = pause_replay(cached, plan, sequence, 2 * chunk, 0);
                for (;;) {
                    auto progress = cached.advance_prefill(sequence);
                    observe_publication(progress);
                    if (progress.pending) {
                        (void)commit(cached, std::move(*progress.pending), 1);
                        break;
                    }
                    require(!progress.complete, "prefill completed without its first token");
                }
                require(notified, "no actual snapshot publication exercised the admission notification");
                sequence = pause_replay(cached, plan, sequence, static_cast<std::uint32_t>(prompt.size()),
                                        free_resume ? 0U : 1U);
                const auto aborted = cached.abort(sequence);
                require(aborted.status == rt::ConsumeStatus::Consumed && !aborted.device_error,
                        "cached pause test did not release its lane");
                released(cached);
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts("all MTP transaction checks passed");
    return 0;
}
