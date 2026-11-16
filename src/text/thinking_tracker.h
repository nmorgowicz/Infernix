#pragma once

#include "infernix/types.h"
#include "text/reasoning_loop.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::text {

// The model-neutral half of a reply's thinking control, shared by every frontend that can open a
// reply inside its thinking: the thinking budget and the reasoning-loop guard. It counts the
// model's own tokens while the thinking is open, licenses the next round so that no round runs past
// the budget, and decides where the control span (a message and the model's thinking close) must be
// committed. A frontend owns the model-specific half: which token closes the thinking, the span's
// tokens and how they are presented.
//
// The tracker follows the output session's preview/commit transaction: begin_preview() starts from
// the committed state, the note calls advance the previewed copy, and commit_preview() makes it the
// committed state; a preview that is never committed leaves the committed state as it was.
class ThinkingTracker {
public:
    enum class Boundary : std::uint8_t {
        Continue,     // the thinking goes on, or is over
        ApplyControl, // the control span is due before the model samples again
        Stop,         // the reasoning-loop guard ends the reply (FinishReason::OutputLimit)
    };

    ThinkingTracker() = default;
    // `starts_in_reasoning`: the reply begins inside its thinking. A budget must be positive.
    ThinkingTracker(const ThinkingControlOptions& options, bool starts_in_reasoning);

    // ---- the committed state -------------------------------------------------------------------
    // Whether the committed output's thinking is open.
    [[nodiscard]] bool in_reasoning() const noexcept { return committed_.in_reasoning; }
    [[nodiscard]] bool control_pending() const noexcept { return committed_.control_pending; }
    // The tokens the model may sample in the next round: the total remaining, cut to the thinking
    // budget while it applies, and none while the control span is due.
    [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t total_budget_remaining) const noexcept;
    // A budget the output limit never reaches takes no effect and needs no room; otherwise what the
    // limit leaves past the budget must hold the whole control span and one more token. Throws
    // std::invalid_argument when it does not.
    void validate_generation_capacity(std::uint32_t effective_output_tokens, std::size_t control_tokens) const;
    [[nodiscard]] ThinkingBudgetStats stats() const noexcept;

    // ---- a preview ------------------------------------------------------------------------------
    void begin_preview();
    // Whether the previewed thinking is open.
    [[nodiscard]] bool preview_in_reasoning() const noexcept { return staged_.in_reasoning; }
    // Whether model_token() reads the token's bytes (the loop guard is watching open thinking), so a
    // caller that must decode them can skip it otherwise.
    [[nodiscard]] bool preview_reads_bytes() const noexcept;
    // One model-origin token. `closes`: it ends the thinking (the close token, or a token that
    // leaves it some other way). Counts while the previewed thinking is open; throws
    // std::logic_error when that passes the budget, which the round's license
    // (model_token_budget_remaining) rules out.
    void model_token(bool closes, std::string_view bytes);
    // After the round's last token, when the round did not finish the reply: whether the budget was
    // reached, or the loop guard fired. `remaining_after` is the output budget the round leaves;
    // `control_tokens` the control span's length. Call it once per round.
    [[nodiscard]] Boundary end_of_round(std::uint32_t remaining_after, std::size_t control_tokens);
    // The round, or a terminal decision, finished the reply: no control follows.
    void finished() noexcept { staged_.control_pending = false; }
    // One control token of the span being committed (`closes`: it ends the thinking), and the whole
    // span once it has been, `injected_tokens` long. control_applied throws std::logic_error when the
    // span did not close the thinking.
    void control_token(bool closes) noexcept;
    void control_applied(std::uint32_t injected_tokens);
    void commit_preview();

private:
    struct State {
        std::optional<std::uint32_t> budget;
        std::uint32_t model_thinking_tokens = 0;
        std::uint32_t injected_tokens       = 0;
        bool in_reasoning                   = false;
        bool control_pending                = false;
        bool applied                        = false;
        // Reasoning-loop guard: the words of the round being previewed (the committed history lives
        // in the tracker, outside the per-round copy), a word cut by the last token, the recent
        // reasoning the periodic-tail check reads, the next check and what fired.
        ReasoningLoopAction loop = ReasoningLoopAction::Off;
        std::string loop_partial;
        std::string loop_tail;
        std::vector<LoopWord> loop_round;
        std::uint32_t loop_next_check      = kLoopCheckTokens;
        bool loop_detected                 = false;
        std::uint32_t loop_thinking_tokens = 0;
        float loop_coverage                = 0.0F;
        std::uint32_t loop_run_on_chars    = 0;
    };

    // The repeated-passage coverage of the committed thinking plus the previewed round's words.
    [[nodiscard]] double loop_coverage(const std::vector<LoopWord>& round) const;

    State committed_;
    State staged_;
    std::vector<LoopWord> words_; // the committed thinking's words (reasoning-loop guard), bounded
};

} // namespace infernix::text
