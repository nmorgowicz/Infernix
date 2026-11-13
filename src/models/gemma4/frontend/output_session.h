#pragma once

#include "infernix/types.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "runtime/contract/request.h"
#include <array>
#include <exception>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::gemma4 {

class PublishedOutput {
public:
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] auto begin() noexcept { return values_.begin(); }
    [[nodiscard]] auto begin() const noexcept { return values_.begin(); }
    [[nodiscard]] auto end() noexcept { return values_.begin() + size_; }
    [[nodiscard]] auto end() const noexcept { return values_.begin() + size_; }
    [[nodiscard]] auto& tool_calls() noexcept { return tool_calls_; }
    [[nodiscard]] const auto& tool_calls() const noexcept { return tool_calls_; }
    void push_back(OutputDelta value) { values_.at(size_++) = std::move(value); }
private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
    std::vector<ToolCallDelta> tool_calls_;
};

// Gemma thinking budgets are accounting-only: no Qwen control suffix or stop-token ban.
// Grammar/tool requests are refused by Frontend before admission in P4.
class OutputSession {
public:
    OutputSession() noexcept = default;
    OutputSession(OutputSession&&) noexcept = default;
    OutputSession& operator=(OutputSession&&) noexcept = default;
    OutputSession(const OutputSession&) = delete;
    OutputSession& operator=(const OutputSession&) = delete;
    [[nodiscard]] runtime::OutputDecision preview_model(std::span<const TokenId> tokens,
        std::uint32_t total_budget_remaining, FinishReason limit_reason);
    [[nodiscard]] std::uint32_t model_token_budget_remaining(std::uint32_t remaining) const noexcept {
        return remaining;
    }
    [[nodiscard]] std::span<const TokenId> pending_control_tokens() const noexcept { return {}; }
    [[nodiscard]] runtime::OutputDecision preview_control(std::span<const TokenId>, std::uint32_t);
    void validate_generation_capacity(std::uint32_t) const noexcept {}
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview();
    void discard_preview() noexcept;
    [[nodiscard]] bool constrained() const noexcept { return false; }
    [[nodiscard]] runtime::ReasoningBan reasoning_ban() const noexcept { return {}; }
    void observe_constraint(bool, double) noexcept {}
    void constraint_uploaded(std::size_t) noexcept {}
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const { return {}; }
    [[nodiscard]] std::uint64_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>);
    [[nodiscard]] bool constraint_pending() const noexcept { return false; }
    [[nodiscard]] std::exception_ptr constraint_failure() const noexcept { return {}; }
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept { return {}; }
    [[nodiscard]] bool has_tool_calls() const noexcept { return false; }
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return state_.reasoning_tokens; }
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    enum class Mode : std::uint8_t { Content, ReasoningHeader, Reasoning };
    struct State {
        Mode mode = Mode::Content;
        std::string content;
        std::string reasoning;
        std::string header;
        std::uint32_t reasoning_tokens = 0;
        std::optional<std::size_t> matched_stop;
    };
    struct Setup {
        std::vector<TokenId> stop_ids;
        std::vector<StopString> stop_strings;
        bool publish_stop = false;
        bool starts_in_reasoning = false;
        std::optional<std::uint32_t> thinking_budget;
        OutputOptions output;
        TokenId channel_open = -1;
        TokenId channel_close = -1;
    };
    OutputSession(std::shared_ptr<const Tokenizer> tokenizer, Setup setup);
    [[nodiscard]] bool is_stop(TokenId token) const noexcept;
    void consume(State& state, TokenId token) const;
    static void append_text(State& state, std::string bytes);
    static void close_reasoning(State& state);
    [[nodiscard]] bool match_stop_string(State& state) const;
    [[nodiscard]] std::size_t stop_hold(const std::string& text, OutputChannel channel) const;
    std::shared_ptr<const Tokenizer> tokenizer_;
    Setup setup_;
    State state_;
    State staged_;
    std::uint32_t staged_tokens_ = 0;
    bool staged_terminal_ = false;
    bool preview_pending_ = false;
    bool terminal_ = false;
    friend class Frontend;
};

} // namespace infernix::models::gemma4
