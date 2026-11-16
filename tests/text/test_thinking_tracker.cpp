// The model-neutral thinking control shared by every frontend that can open a reply inside its thinking
// (text::ThinkingTracker): the thinking budget's round license and control boundary, the output-capacity check,
// the preview/commit transaction, and the reasoning-loop guard's checks and actions. CPU only, no tokenizer: a
// token is a close flag and the bytes it decodes to.
#include "text/thinking_tracker.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using infernix::ReasoningLoopAction;
using infernix::ThinkingControlOptions;
using infernix::text::ThinkingTracker;
using Boundary = ThinkingTracker::Boundary;

int failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL " << label << '\n';
        ++failures;
    }
}

template <typename F>
bool throws(F&& call) {
    try { call(); } catch (const std::exception&) { return true; }
    return false;
}

constexpr std::size_t kControl = 25; // tokens of the control span in these tests

// One engine round of `n` plain thinking tokens.
Boundary round(ThinkingTracker& tracker, std::uint32_t n, std::uint32_t remaining_after = 1000, bool commit = true) {
    tracker.begin_preview();
    for (std::uint32_t i = 0; i < n; ++i) { tracker.model_token(false, "x"); }
    const Boundary boundary = tracker.end_of_round(remaining_after, kControl);
    if (commit) { tracker.commit_preview(); }
    return boundary;
}

// The control span: kControl - 1 text tokens, then the close.
void apply_control(ThinkingTracker& tracker) {
    tracker.begin_preview();
    for (std::size_t i = 0; i + 1 < kControl; ++i) { tracker.control_token(false); }
    tracker.control_token(true);
    tracker.control_applied(static_cast<std::uint32_t>(kControl));
    tracker.commit_preview();
}

void budget() {
    ThinkingTracker tracker(ThinkingControlOptions{.budget = 5}, true);
    expect(tracker.in_reasoning() && !tracker.control_pending(), "a reply that starts in thinking is in thinking");
    expect(tracker.model_token_budget_remaining(100) == 5 && tracker.model_token_budget_remaining(3) == 3,
           "the license is the smaller of the output left and the budget");
    expect(round(tracker, 3) == Boundary::Continue && tracker.model_token_budget_remaining(100) == 2,
           "three tokens leave two");
    // A previewed round that is never committed leaves the committed state alone.
    expect(round(tracker, 2, 1000, false) == Boundary::ApplyControl && tracker.model_token_budget_remaining(100) == 2 &&
               !tracker.control_pending() && tracker.stats().model_thinking_tokens == 3,
           "an uncommitted round changed the committed state");
    expect(round(tracker, 2) == Boundary::ApplyControl, "the cap asks for the control");
    expect(tracker.control_pending() && tracker.model_token_budget_remaining(100) == 0 && tracker.in_reasoning(),
           "no license while the control is due, and the thinking is still open");
    apply_control(tracker);
    const auto stats = tracker.stats();
    expect(!tracker.control_pending() && !tracker.in_reasoning() && stats.applied && stats.injected_tokens == kControl &&
               stats.model_thinking_tokens == 5 && stats.configured_budget == 5U,
           "the applied control is reported and the thinking is over");
    expect(tracker.model_token_budget_remaining(100) == 100, "after the control the model is licensed freely");
    // Tokens after the thinking are the answer: not counted.
    expect(round(tracker, 7) == Boundary::Continue && tracker.stats().model_thinking_tokens == 5,
           "answer tokens counted as thinking");
}

void overflow_and_natural_close() {
    ThinkingTracker tracker(ThinkingControlOptions{.budget = 3}, true);
    tracker.begin_preview();
    tracker.model_token(false, "a");
    tracker.model_token(false, "b");
    tracker.model_token(false, "c");
    expect(throws([&] { tracker.model_token(false, "d"); }), "a token past the budget was accepted");

    // The model's own close at the cap is the last thinking token: nothing is asked for.
    ThinkingTracker closed(ThinkingControlOptions{.budget = 3}, true);
    closed.begin_preview();
    closed.model_token(false, "a");
    closed.model_token(false, "b");
    closed.model_token(true, "</>");
    expect(!closed.preview_in_reasoning() && closed.end_of_round(10, kControl) == Boundary::Continue,
           "the model's close at the cap still asked for the control");
    closed.commit_preview();
    expect(!closed.in_reasoning() && !closed.control_pending() && !closed.stats().applied &&
               closed.stats().model_thinking_tokens == 3 && closed.model_token_budget_remaining(50) == 50,
           "the natural close left the budget armed");
}

void finished_replies_take_no_control() {
    ThinkingTracker tracker(ThinkingControlOptions{.budget = 2}, true);
    tracker.begin_preview();
    tracker.model_token(false, "a");
    tracker.model_token(false, "b");
    expect(tracker.end_of_round(10, kControl) == Boundary::ApplyControl, "the cap asked for the control");
    tracker.finished(); // a stop, the output limit or a cancel decided the round first
    tracker.commit_preview();
    expect(!tracker.control_pending(), "a finished reply kept its control pending");
}

void capacity() {
    ThinkingTracker tracker(ThinkingControlOptions{.budget = 10}, true);
    tracker.validate_generation_capacity(5, kControl);
    tracker.validate_generation_capacity(10, kControl); // the cap is never reached: no room needed
    expect(throws([&] { tracker.validate_generation_capacity(10 + kControl, kControl); }),
           "a limit without one answer token after the control was accepted");
    tracker.validate_generation_capacity(10 + kControl + 1, kControl);
    ThinkingTracker direct(ThinkingControlOptions{.budget = 10}, false);
    direct.validate_generation_capacity(11, kControl);
    expect(direct.model_token_budget_remaining(50) == 50, "a reply that does not start in thinking was capped");
    ThinkingTracker none(ThinkingControlOptions{}, true);
    none.validate_generation_capacity(11, kControl);
    expect(none.model_token_budget_remaining(50) == 50 && none.stats().model_thinking_tokens == 0 &&
               !none.stats().configured_budget,
           "a reply with no budget is capped");
    expect(throws([] { ThinkingTracker(ThinkingControlOptions{.budget = 0}, true); }), "a zero budget was accepted");
    // The thinking is counted without a budget.
    none.begin_preview();
    none.model_token(false, "a");
    none.model_token(true, "b");
    none.commit_preview();
    expect(none.stats().model_thinking_tokens == 2 && !none.in_reasoning(), "the thinking was not counted without a budget");
}

// Deterministic varied prose: a new combination of words in every sentence.
std::string varied(int sentences, unsigned seed = 7) {
    static const char* subjects[] = {"the parser", "this branch", "the cache", "our bound", "the loop", "a tile",
                                     "the queue", "that edge", "the schema", "each lane", "the budget"};
    static const char* verbs[]    = {"handles", "skips", "rounds", "checks", "splits", "merges", "drops",
                                     "keeps", "orders", "limits", "scales", "tracks", "maps"};
    static const char* objects[]  = {"empty input", "the last token", "two columns", "a ragged width",
                                     "every retry", "the null case", "large values", "unicode names",
                                     "negative offsets", "the first page", "stale entries", "short reads"};
    std::string text;
    unsigned state = seed;
    for (int s = 0; s < sentences; ++s) {
        state = state * 1103515245U + 12345U;
        text += "step " + std::to_string(s) + ": " + subjects[(state >> 8) % 11] + " " + verbs[(state >> 12) % 13] + " " +
                objects[(state >> 16) % 12] + " against bound " + std::to_string((state >> 4) % 811) + ". ";
    }
    return text;
}

struct Loop {
    Boundary boundary = Boundary::Continue;
    std::uint32_t committed = 0;
};

// Rounds of 200 tokens of `piece` bytes each until a boundary or the text ends.
Loop feed(ThinkingTracker& tracker, const std::string& text, std::uint32_t remaining, std::size_t piece = 4) {
    Loop out;
    std::size_t at = 0;
    while (at < text.size()) {
        tracker.begin_preview();
        std::uint32_t n = 0;
        for (; n < 200 && at < text.size(); ++n, at += piece) {
            tracker.model_token(false, std::string_view(text).substr(at, piece));
        }
        out.committed += n;
        out.boundary = tracker.end_of_round(remaining - out.committed, kControl);
        tracker.commit_preview();
        if (out.boundary != Boundary::Continue) { break; }
    }
    return out;
}

void loop_guard() {
    std::string looping = varied(260);
    for (int i = 0; i < 60; ++i) {
        looping += "Wait, let me double-check the case where the list is empty. Then the function returns "
                   "zero, which matches the expected output, so that case is fine. ";
    }
    {
        ThinkingTracker tracker(ThinkingControlOptions{.loop = ReasoningLoopAction::Stop}, true);
        const Loop run = feed(tracker, looping, 1'000'000);
        const auto stats = tracker.stats();
        expect(run.boundary == Boundary::Stop && stats.loop_detected && !stats.applied && stats.loop_coverage >= 0.25F &&
                   stats.loop_thinking_tokens >= 512 && !tracker.control_pending(),
               "Stop did not end a looping thought");
    }
    {
        ThinkingTracker tracker(ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude}, true);
        const Loop run = feed(tracker, looping, 1'000'000);
        expect(run.boundary == Boundary::ApplyControl && tracker.control_pending() && tracker.stats().loop_detected &&
                   tracker.model_token_budget_remaining(100) == 0,
               "Conclude did not ask for the control");
        apply_control(tracker);
        expect(tracker.stats().applied && tracker.stats().injected_tokens == kControl && !tracker.in_reasoning(),
               "Conclude's control was not applied");
    }
    {   // Conclude with no room for the control and one more token stops instead.
        ThinkingTracker probe(ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude}, true);
        const std::uint32_t fired_after = feed(probe, looping, 1'000'000).committed;
        ThinkingTracker tracker(ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude}, true);
        const Loop run = feed(tracker, looping, fired_after + 1);
        expect(run.boundary == Boundary::Stop && tracker.stats().loop_detected && !tracker.control_pending(),
               "Conclude without room did not stop");
    }
    for (const auto action : {ReasoningLoopAction::Off, ReasoningLoopAction::Conclude}) {
        // Varied thinking never fires; a guard that is off never fires on a loop.
        ThinkingTracker tracker(ThinkingControlOptions{.loop = action}, true);
        const std::string text = action == ReasoningLoopAction::Off ? looping : varied(260) + varied(300, 11);
        const Loop run = feed(tracker, text, 1'000'000);
        expect(run.boundary == Boundary::Continue && !tracker.stats().loop_detected,
               action == ReasoningLoopAction::Off ? "the guard fired while off" : "the guard fired on varied thinking");
    }
    {   // A loop written after the thinking closed is the answer's: the guard has stopped watching.
        ThinkingTracker tracker(ThinkingControlOptions{.loop = ReasoningLoopAction::Stop}, true);
        tracker.begin_preview();
        tracker.model_token(true, "</>");
        tracker.commit_preview();
        expect(!tracker.in_reasoning() && feed(tracker, looping, 1'000'000).boundary == Boundary::Continue &&
                   !tracker.stats().loop_detected,
               "the guard watched the answer");
    }
}
} // namespace

int main() {
    budget();
    overflow_and_natural_close();
    finished_replies_take_no_control();
    capacity();
    loop_guard();
    if (failures != 0) {
        std::cerr << failures << " thinking tracker check(s) failed\n";
        return 1;
    }
    std::cout << "all thinking tracker checks passed\n";
    return 0;
}
