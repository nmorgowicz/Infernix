#include "text/thinking_tracker.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infernix::text {

ThinkingTracker::ThinkingTracker(const ThinkingControlOptions& options, bool starts_in_reasoning) {
    if (options.budget && *options.budget == 0) {
        throw std::invalid_argument("thinking budget must be positive");
    }
    // The tracker counts the model's own thinking tokens for every reply that starts in thinking: the
    // request log reports that count, so it runs without a budget too (one token-id comparison per
    // generated token). The budget caps it and the loop guard reads it.
    committed_.budget       = options.budget;
    committed_.loop         = options.loop;
    committed_.in_reasoning = starts_in_reasoning;
    staged_                 = committed_;
}

std::uint32_t ThinkingTracker::model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept {
    const State& state = committed_;
    if (state.control_pending) { return 0; } // a budget's or the loop guard's
    if (!state.budget || !state.in_reasoning || state.applied) { return total_budget_remaining; }
    if (state.model_thinking_tokens >= *state.budget) { return 0; }
    return std::min(total_budget_remaining, *state.budget - state.model_thinking_tokens);
}

void ThinkingTracker::validate_generation_capacity(std::uint32_t effective_output_tokens,
                                                   std::size_t control_tokens) const {
    const State& state = committed_;
    if (!state.budget || !state.in_reasoning || effective_output_tokens <= *state.budget) { return; }
    const std::uint64_t remaining = static_cast<std::uint64_t>(effective_output_tokens) - *state.budget;
    const std::uint64_t required  = static_cast<std::uint64_t>(control_tokens) + 1U;
    if (remaining < required) {
        throw std::invalid_argument(
            "effective output capacity after the thinking budget must fit the complete control "
            "suffix and one post-close model token");
    }
}

ThinkingBudgetStats ThinkingTracker::stats() const noexcept {
    return ThinkingBudgetStats{
        .configured_budget     = committed_.budget,
        .model_thinking_tokens = committed_.model_thinking_tokens,
        .injected_tokens       = committed_.injected_tokens,
        .applied               = committed_.applied,
        .loop_detected         = committed_.loop_detected,
        .loop_thinking_tokens  = committed_.loop_thinking_tokens,
        .loop_coverage         = committed_.loop_coverage,
        .loop_run_on_chars     = committed_.loop_run_on_chars,
    };
}

void ThinkingTracker::begin_preview() { staged_ = committed_; }

bool ThinkingTracker::preview_reads_bytes() const noexcept {
    return staged_.in_reasoning && staged_.loop != ReasoningLoopAction::Off && !staged_.loop_detected;
}

void ThinkingTracker::model_token(bool closes, std::string_view bytes) {
    State& state = staged_;
    if (!state.in_reasoning) { return; }
    ++state.model_thinking_tokens;
    if (state.budget && state.model_thinking_tokens > *state.budget) {
        throw std::logic_error("model output exceeded the licensed thinking budget");
    }
    if (state.loop != ReasoningLoopAction::Off && !state.loop_detected) {
        append_words(bytes, state.loop_partial, state.loop_round);
        append_tail(bytes, state.loop_tail);
    }
    // The close token is the last thinking token; its spelling in ordinary tokens is not.
    if (closes) {
        state.in_reasoning    = false;
        state.control_pending = false;
    }
}

double ThinkingTracker::loop_coverage(const std::vector<LoopWord>& round) const {
    const std::size_t keep = kLoopHistoryWords > round.size() ? kLoopHistoryWords - round.size() : 0;
    std::vector<LoopWord> words(words_.end() - static_cast<std::ptrdiff_t>(std::min(keep, words_.size())),
                                words_.end());
    words.insert(words.end(), round.begin(), round.end());
    return loop_passage_coverage(words);
}

ThinkingTracker::Boundary ThinkingTracker::end_of_round(std::uint32_t remaining_after,
                                                        std::size_t control_tokens) {
    State& state = staged_;
    if (state.in_reasoning && state.budget && state.model_thinking_tokens == *state.budget) {
        state.control_pending = true;
        return Boundary::ApplyControl;
    }
    // Reasoning-loop guard: once per kLoopCheckTokens thinking tokens, at the end of a round.
    if (state.loop != ReasoningLoopAction::Off && state.in_reasoning && !state.applied && !state.loop_detected &&
        state.model_thinking_tokens >= state.loop_next_check) {
        state.loop_next_check = (state.model_thinking_tokens / kLoopCheckTokens + 1) * kLoopCheckTokens;
        // Strata reads a periodic tail as full coverage before its word measure; thinking that repeats
        // nothing can still be a hyphen-joined run-on (Strata #5218).
        const double coverage = periodic_tail(state.loop_tail) ? 1.0 : loop_coverage(state.loop_round);
        const std::size_t run_on = coverage >= kLoopCoverage ? 0 : longest_hyphen_chain(state.loop_tail);
        if (coverage >= kLoopCoverage || run_on >= kLoopRunOnChars) {
            state.loop_detected        = true;
            state.loop_thinking_tokens = state.model_thinking_tokens;
            state.loop_coverage        = static_cast<float>(coverage);
            state.loop_run_on_chars    = run_on >= kLoopRunOnChars ? static_cast<std::uint32_t>(run_on) : 0U;
            if (state.loop == ReasoningLoopAction::Conclude && control_tokens > 0 &&
                remaining_after >= control_tokens + 1U) {
                state.control_pending = true;
                return Boundary::ApplyControl;
            }
            return Boundary::Stop;
        }
    }
    return Boundary::Continue;
}

void ThinkingTracker::control_token(bool closes) noexcept {
    if (closes) { staged_.in_reasoning = false; }
}

void ThinkingTracker::control_applied(std::uint32_t injected_tokens) {
    if (staged_.in_reasoning) {
        throw std::logic_error("canonical thinking control did not close the thinking phase");
    }
    staged_.control_pending = false;
    staged_.applied         = true;
    staged_.injected_tokens = injected_tokens;
}

void ThinkingTracker::commit_preview() {
    std::swap(committed_, staged_);
    State& state = committed_;
    // Moves the committed round's words into the history (bounded); drops it once thinking ends.
    if (!state.in_reasoning || state.loop == ReasoningLoopAction::Off || state.loop_detected) {
        words_.clear();
        words_.shrink_to_fit();
        state.loop_round.clear();
        state.loop_tail.clear();
        return;
    }
    words_.insert(words_.end(), state.loop_round.begin(), state.loop_round.end());
    state.loop_round.clear();
    if (words_.size() > 2 * kLoopHistoryWords) {
        words_.erase(words_.begin(), words_.end() - static_cast<std::ptrdiff_t>(kLoopHistoryWords));
    }
}

} // namespace infernix::text
