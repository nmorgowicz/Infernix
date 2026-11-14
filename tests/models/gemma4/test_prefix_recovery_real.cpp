// Gemma4 prefix-cache recovery on a real artifact (plan §8.4 invariants 1, 4 and 5; hybrid-prefix-cache-spec
// §7.8). INFERNIX_GEMMA4_ARTIFACT selects the artifact (absent: skip, 77); INFERNIX_GEMMA4_HOST_MIB the Host
// tier (default 8192). A cache-off Program generates every reference; after each recovery the lane is free,
// no Device copy or hold is left, and the next request equals the reference (cold, and resumed).
//
//   Program path (the Engine's own transaction calls, so every failure point is exact):
//     cancel mid-prefill    abort between prefill calls: valid published state stays reusable
//     cancel mid-decode     abort between rounds, a cancelled commit of a staged round, abort_pending
//     non-finite withdrawal take_nonfinite_fault on the prompt's first token and on a decode row: what the
//                           request published (blocks, taps) leaves the cache, what was there stays, a resend
//                           resumes from nothing and then reuses
//     abort during restore  a binding cancelled in poll_context (pins released, source intact), an abort
//                           right after the restore landed
//     capture then cancel   a tap capture in flight when the request aborts, then an unrelated cold request
//                           overwrites the pages and rings: the captured tap still restores the reference
//     memory fault          an out-of-memory forward after a restore: recover_after_failure empties the
//                           cache and the next request equals the reference
//   Engine path (public): cancel mid-prefill, cancel mid-decode, a non-finite first token (withdrawal in
//   the Engine's gauges), a cancel after the restore bound (units held), and the no-progress heal of a
//   restored request.
//
// "Pins and leases at baseline" is witnessed through the public surface: the lane's occupied resources are
// zero, no Device copy of a cached block remains (every insert pins one until release), no snapshot is
// held, and a withdrawal that needed every pin dropped leaves the node and snapshot counts at their
// pre-request values.

#include "prefix_real_support.h"

#include <atomic>
#include <cstdio>
#include <string>
#include <utility>

namespace {

using namespace prefix_real;
namespace ops = infernix::ops;
namespace testing = infernix::runtime::testing;

constexpr std::uint32_t kChunk   = 128;
constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kOut     = 8;
constexpr auto kBf16 = infernix::KvCacheStorage::BFloat16;

struct Counts {
    std::uint32_t nodes = 0, snapshots = 0;
    std::uint64_t blocks = 0, image_writes = 0;
};
Counts counts(const g4::Program& program) {
    const auto s = program.hybrid_stats();
    return {s.nodes, s.snapshots, s.blocks_inserted, s.host_image_writes};
}

void decode_round(g4::Program& program, g4::SequenceHandle sequence, std::uint32_t remaining, rt::CommitDecision d) {
    const std::array sequences{sequence};
    const std::array budgets{rt::RoundBudget{remaining}};
    auto pending = program.decode(sequences, budgets);
    (void)commit_one(program, std::move(pending), d);
}

// The first token and `rounds` decode rounds committed: a sequence that is mid-decode.
g4::SequenceHandle mid_decode(g4::Program& program, const g4::Frontend& frontend, const Tokens& prompt, unsigned rounds) {
    const Started started = start(program, frontend, prompt, kOut, true);
    require(started.sequence.has_value(), "the binding did not publish");
    auto pending = first_token(program, *started.sequence);
    (void)commit_one(program, std::move(pending), decision());
    for (unsigned i = 0; i < rounds; ++i) { decode_round(program, *started.sequence, kOut - 1 - i, decision()); }
    return *started.sequence;
}

void program_recovery(const char* artifact) {
    std::printf("== Program path\n");
    Rig rig(artifact, kContext);
    const g4::Frontend& frontend = *rig.frontend;
    const Tokens p = tokens(1537, 17), q = tokens(700, 23), w = tokens(1400, 41), x = tokens(1000, 43),
                 r = tokens(2100, 53), s = tokens(900, 61), m = tokens(1700, 19),
                 e = concat(tokens(1537, 17), tokens(400, 99)); // p's prompt and 400 more tokens
    struct { Tokens p, q, w, x, r, s, m, e; } ref;
    {
        const auto cold = rig.program(kContext, kChunk, 0, kBf16, ops::Gemma4QkMode::BFloat16);
        const auto reference = [&](const Tokens& prompt) { return run(*cold, frontend, prompt, kOut, true).tokens; };
        ref = {reference(p), reference(q), reference(w), reference(x), reference(r), reference(s), reference(m), reference(e)};
    }
    const auto program_ptr = rig.program(kContext, kChunk, host_bytes_from_env(), kBf16, ops::Gemma4QkMode::BFloat16);
    g4::Program& program = *program_ptr;
    const auto same = [&](const Tokens& prompt, const Tokens& reference, const std::string& what, bool reuse) {
        const Outcome out = run(program, frontend, prompt, kOut, reuse);
        expect(out.tokens == reference, what + " equals a fresh Program's output (" + difference(out.tokens, reference) + ")");
        return out;
    };
    const auto released = [&](const std::string& what) { expect_released(program, frontend, what); };
    (void)same(p, ref.p, "the first request", true);
    (void)same(q, ref.q, "the second request", true);
    released("after two clean requests");

    std::printf("-- cancel mid-prefill\n");
    {
        const Started st = start(program, frontend, m, kOut, true); // a cold prompt: its calls publish as they go
        require(st.sequence.has_value(), "the binding did not publish");
        for (int i = 0; i < 4; ++i) { require(!program.advance_prefill(*st.sequence).complete, "the prompt completed too early"); }
        (void)program.abort(*st.sequence);
        released("after an abort between prefill calls");
        const Outcome cold = same(m, ref.m, "the cold request after the cancel", false);
        expect(cold.reused == 0, "an opted-out request restores nothing");
        const Outcome warm = same(m, ref.m, "the request resumed from what the cancelled one published", true);
        expect(warm.reused > 0 && warm.reused <= 512,
               "published state of a cancelled request stays valid and reusable (restored " + std::to_string(warm.reused) + ")");
    }

    std::printf("-- cancel mid-decode\n");
    {
        const auto sequence = mid_decode(program, frontend, q, 3);
        (void)program.abort(sequence);
        released("after an abort between decode rounds");
        (void)same(q, ref.q, "the request after a mid-decode abort", false);

        const auto staged = mid_decode(program, frontend, q, 2);
        const std::array sequences{staged};
        const std::array budgets{rt::RoundBudget{3}};
        auto pending = program.decode(sequences, budgets);
        const auto disposition = commit_one(program, std::move(pending), decision(false, true));
        expect(disposition == rt::CommitDisposition::CancelledReleased, "a cancelled commit releases the sequence");
        released("after a cancelled commit of a staged round");
        (void)same(q, ref.q, "the request after a cancelled staged round", true);

        const Started st = start(program, frontend, q, kOut, true);
        require(st.sequence.has_value(), "the binding did not publish");
        auto first = first_token(program, *st.sequence);
        expect(program.abort_pending(std::move(first)).status == rt::ConsumeStatus::Consumed &&
                   program.abort_pending(std::move(first)).status == rt::ConsumeStatus::InvariantMismatch,
               "abort_pending consumes the staged first token once");
        released("after abort_pending");
        (void)same(q, ref.q, "the request after abort_pending", true);
    }

    std::printf("-- non-finite withdrawal\n");
    {
        settle(program, frontend);
        const Counts before = counts(program);
        testing::set_nonfinite_fault(1); // the prompt's first token
        const Started st = start(program, frontend, w, kOut, true);
        require(st.sequence.has_value(), "the binding did not publish");
        auto pending = first_token(program, *st.sequence);
        testing::set_nonfinite_fault(0);
        expect(pending.row_failure(0) == rt::PendingRowFailure::NonFiniteLogits, "the first token reports non-finite logits");
        const Counts published = counts(program);
        expect(published.blocks > before.blocks && published.image_writes > before.image_writes,
               "the failing request had published blocks and a tap (positive control)");
        expect(commit_one(program, std::move(pending), decision(false, false, true)) == rt::CommitDisposition::FailedReleased,
               "the failed row is released");
        released("after a non-finite first token");
        const Counts after = counts(program);
        expect(after.nodes == before.nodes && after.snapshots == before.snapshots,
               "its blocks and snapshots are withdrawn (nodes " + std::to_string(after.nodes) + " of " +
                   std::to_string(before.nodes) + ", snapshots " + std::to_string(after.snapshots) + " of " +
                   std::to_string(before.snapshots) + ")");
        const Outcome retry = same(w, ref.w, "the retry", true);
        expect(retry.reused == 0, "the same prompt resumes from nothing (restored " + std::to_string(retry.reused) + ")");
        const Outcome again = same(w, ref.w, "the repeat of the retry", true);
        expect(again.reused > 0, "its repeat resumes from the retry's entries (restored " + std::to_string(again.reused) + ")");
        (void)same(p, ref.p, "an earlier request's entries after the withdrawal", true);

        settle(program, frontend);
        const Counts before_decode = counts(program);
        testing::set_nonfinite_fault(2); // first decode row, after the first output committed
        const Started sx = start(program, frontend, x, kOut, true);
        require(sx.sequence.has_value(), "the binding did not publish");
        auto first = first_token(program, *sx.sequence);
        (void)commit_one(program, std::move(first), decision());
        const std::array sequences{*sx.sequence};
        const std::array budgets{rt::RoundBudget{kOut - 1}};
        auto row = program.decode(sequences, budgets);
        testing::set_nonfinite_fault(0);
        expect(row.row_failure(0) == rt::PendingRowFailure::NonFiniteLogits, "a decode row reports non-finite logits");
        (void)commit_one(program, std::move(row), decision(false, false, true));
        released("after a non-finite decode row");
        const Counts after_decode = counts(program);
        expect(after_decode.nodes == before_decode.nodes && after_decode.snapshots == before_decode.snapshots,
               "the decode failure's entries are withdrawn and earlier ones stay");
        const Outcome resent = same(x, ref.x, "the resend after the decode failure", true);
        expect(resent.reused == 0, "it resumes from nothing (restored " + std::to_string(resent.reused) + ")");
    }

    std::printf("-- abort during restore\n");
    {
        const std::atomic<bool> stop{true};
        const Started cancelled = start(program, frontend, p, kOut, true, rt::CancellationFlagView{&stop});
        expect(!cancelled.sequence.has_value() && cancelled.reused == 0 && !program.has_context_transaction(),
               "a binding cancelled in its restore publishes nothing and restores nothing");
        released("after a cancelled restore");
        const Outcome source = same(p, ref.p, "the source of the cancelled restore", true);
        expect(source.reused > 0, "the source is intact (restored " + std::to_string(source.reused) + ")");
        const Started landed = start(program, frontend, p, kOut, true);
        require(landed.sequence.has_value(), "the binding did not publish");
        expect(landed.reused > 0, "the restore landed before the abort");
        (void)program.abort(*landed.sequence);
        released("after an abort at the restored frontier");
        (void)same(p, ref.p, "the request after an abort at the restored frontier", true);
    }

    std::printf("-- capture then cancel\n");
    {
        const std::uint64_t writes = program.hybrid_stats().host_image_writes;
        const Started st = start(program, frontend, r, kOut, true);
        require(st.sequence.has_value(), "the binding did not publish");
        bool captured = false;
        for (int call = 0; call < 20 && !captured; ++call) {
            require(!program.advance_prefill(*st.sequence).complete, "the prompt completed before a capture");
            captured = program.hybrid_stats().host_image_writes > writes;
        }
        expect(captured, "a tap capture was enqueued before the cancel (positive control)");
        (void)program.abort(*st.sequence);
        released("after an abort with a capture in flight");
        (void)same(s, ref.s, "an unrelated cold request overwriting pages and rings", false);
        const Outcome warm = same(r, ref.r, "the cancelled request's captured tap", true);
        expect(warm.reused > 0, "the capture survived the overwrite and restores (restored " + std::to_string(warm.reused) + ")");
        released("after the restore of the captured tap");
    }

    std::printf("-- memory fault after a restore\n");
    {
        const Started st = start(program, frontend, e, kOut, true); // resumes at p's tap, 400 tokens still to prefill
        require(st.sequence.has_value() && st.reused > 0, "the binding did not restore");
        infernix::testing::arm_memory_exhausted_here(100);
        bool failed = false;
        try { (void)program.advance_prefill(*st.sequence); }
        catch (const infernix::DeviceMemoryExhausted&) { failed = true; }
        infernix::testing::arm_memory_exhausted_here(0);
        expect(failed, "the forward after the restore hit the injected shortage");
        expect(program.recover_after_failure(), "the device recovered");
        const Counts after = counts(program);
        expect(after.nodes == 0 && after.snapshots == 0, "recovery rebuilt an empty cache");
        released("after recovery");
        const Outcome fresh = same(e, ref.e, "the next request", true);
        expect(fresh.reused == 0, "it starts cold (restored " + std::to_string(fresh.reused) + ")");
    }
}

// ---- the public Engine ---------------------------------------------------------------------------

void engine_recovery(const char* artifact) {
    std::printf("== Engine path\n");
    infernix::Engine engine = start_engine(engine_options(artifact, kChunk, host_bytes_from_env(), 2304));
    const Tokens p = tokens(1537, 17), w = tokens(1400, 41), cancelled = tokens(1800, 71), decoding = tokens(900, 73);
    const Served ref_p = generate(engine, p, kOut, false), ref_w = generate(engine, w, kOut, false);
    (void)generate(engine, p, kOut, true);
    settled(engine, "baseline");
    const auto next_equals = [&](const Tokens& prompt, const Served& reference, const std::string& what) {
        const Served cold = generate(engine, prompt, kOut, false);
        expect(cold.reused == 0 && cold.tokens == reference.tokens,
               what + ": the next cold request equals the reference (" + difference(cold.tokens, reference.tokens) + ")");
        const Served warm = generate(engine, prompt, kOut, true);
        expect(warm.tokens == reference.tokens, what + ": the next cache-enabled request equals the reference (" +
                                                    difference(warm.tokens, reference.tokens) + ")");
        settled(engine, what);
    };

    std::printf("-- cancel mid-prefill\n");
    {
        CancelSink sink;
        sink.prompt_threshold = 600;
        const auto result = generate_cancelled(engine, cancelled, kOut, sink);
        expect(result.finish_reason == infernix::FinishReason::Cancelled && result.generated_token_ids.empty(),
               "the request cancelled in its prefill ends cancelled with no output");
        settled(engine, "after the prefill cancel");
        next_equals(p, ref_p, "after the prefill cancel");
    }
    std::printf("-- cancel mid-decode\n");
    {
        CancelSink sink;
        sink.token_threshold = 3;
        const auto result = generate_cancelled(engine, decoding, 24, sink);
        expect(result.finish_reason == infernix::FinishReason::Cancelled && result.generated_token_ids.size() < 24,
               "the request cancelled in its decode stops short (" + std::to_string(result.generated_token_ids.size()) + " tokens)");
        settled(engine, "after the decode cancel");
        next_equals(p, ref_p, "after the decode cancel");
    }
    std::printf("-- non-finite first token\n");
    {
        const auto before = settled(engine, "before the fault");
        testing::set_nonfinite_fault(1);
        bool failed = false;
        std::string message;
        try { (void)generate(engine, w, kOut, true); }
        catch (const std::exception& error) { failed = true; message = error.what(); }
        testing::set_nonfinite_fault(0);
        expect(failed && message.find("non-finite") != std::string::npos, "the request fails with the non-finite error (" + message + ")");
        const auto after = stats_when(engine, [&](const infernix::RuntimeStats& s) {
            return s.hybrid_blocks_inserted > before.hybrid_blocks_inserted && s.hybrid_tree_blocks == before.hybrid_tree_blocks &&
                   s.hybrid_snapshots == before.hybrid_snapshots;
        });
        expect(after.hybrid_blocks_inserted > before.hybrid_blocks_inserted, "the failed request had published blocks (positive control)");
        expect(after.hybrid_tree_blocks == before.hybrid_tree_blocks && after.hybrid_snapshots == before.hybrid_snapshots,
               "its blocks and snapshots are withdrawn (blocks " + std::to_string(after.hybrid_tree_blocks) + " of " +
                   std::to_string(before.hybrid_tree_blocks) + ", snapshots " + std::to_string(after.hybrid_snapshots) + " of " +
                   std::to_string(before.hybrid_snapshots) + ")");
        settled(engine, "after the withdrawal");
        const Served retry = generate(engine, w, kOut, true);
        expect(retry.reused == 0 && retry.tokens == ref_w.tokens, "the retry resumes from nothing and equals the reference");
        const Served again = generate(engine, w, kOut, true);
        expect(again.reused > 0 && again.tokens == ref_w.tokens, "its repeat resumes from the retry's entries (restored " + std::to_string(again.reused) + ")");
        settled(engine, "after the retry");
    }
    std::printf("-- cancel after the restore bound\n");
    {
        const auto selections = engine.runtime_stats().prefix_selections;
        testing::hold_units(true);
        auto handle = engine.submit(engine.prepare_tokens(p), greedy(kOut, true));
        const auto bound = stats_when(engine, [&](const infernix::RuntimeStats& s) { return s.prefix_selections > selections; });
        expect(bound.prefix_selections > selections, "the request bound from a snapshot while its units were held");
        const std::atomic<bool> cancel{true};
        testing::hold_units(false);
        const auto result = handle.wait(nullptr, infernix::CancellationView([&cancel] { return cancel.load(); }));
        expect(result.finish_reason == infernix::FinishReason::Cancelled, "the restored, never-run request ends cancelled");
        settled(engine, "after the cancel at restore");
        next_equals(p, ref_p, "after the cancel at restore");
    }
    std::printf("-- no-progress heal of a restored request\n");
    {
        const auto selections = engine.runtime_stats().prefix_selections;
        const auto recoveries = engine.runtime_stats().no_progress_recoveries;
        testing::set_no_progress_bounds(500, 2000);
        testing::hold_units(true);
        auto handle = engine.submit(engine.prepare_tokens(p), greedy(kOut, true));
        const auto bound = stats_when(engine, [&](const infernix::RuntimeStats& s) { return s.prefix_selections > selections; });
        expect(bound.prefix_selections > selections, "the request bound from a snapshot before the stall");
        std::string message;
        try { (void)handle.wait(); }
        catch (const std::exception& error) { message = error.what(); }
        testing::hold_units(false);
        testing::set_no_progress_bounds(0, 0);
        expect(message.find("made no progress") != std::string::npos, "the stalled request fails with the no-progress error (" + message + ")");
        expect(engine.runtime_stats().no_progress_recoveries == recoveries + 1, "the Engine counted one heal");
        settled(engine, "after the heal");
        next_equals(p, ref_p, "after the heal");
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        program_recovery(artifact);
        engine_recovery(artifact);
    } catch (const std::exception& error) {
        infernix::testing::arm_memory_exhausted_here(0);
        testing::set_nonfinite_fault(0);
        testing::hold_units(false);
        testing::set_no_progress_bounds(0, 0);
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts(failures == 0 ? "all prefix recovery checks passed" : "prefix recovery checks failed");
    return failures == 0 ? 0 : 1;
}
