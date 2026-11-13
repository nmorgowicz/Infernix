#pragma once

#include "infernix/types.h"
#include "models/gemma4/frontend/tokenizer.h"
#include "models/gemma4/frontend/tool_parser.h"
#include "runtime/contract/request.h"
#include "text/grammar.h"
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
    [[nodiscard]] OutputDelta* begin() noexcept { return overflow_.empty() ? values_.data() : overflow_.data(); }
    [[nodiscard]] const OutputDelta* begin() const noexcept { return overflow_.empty() ? values_.data() : overflow_.data(); }
    [[nodiscard]] OutputDelta* end() noexcept { return begin() + size_; }
    [[nodiscard]] const OutputDelta* end() const noexcept { return begin() + size_; }
    [[nodiscard]] auto& tool_calls() noexcept { return tool_calls_; }
    [[nodiscard]] const auto& tool_calls() const noexcept { return tool_calls_; }
    struct Publication { bool tool_call; std::size_t index; };
    // Empty for text-only output: the common case retains inline, allocation-free storage.
    [[nodiscard]] const auto& publication_order() const noexcept { return order_; }
    void push_back(OutputDelta value) {
        if (size_ < values_.size() && overflow_.empty()) { values_[size_] = std::move(value); }
        else {
            if (overflow_.empty()) {
                for (auto& item : values_) { overflow_.push_back(std::move(item)); }
            }
            overflow_.push_back(std::move(value));
        }
        if (!order_.empty()) { order_.push_back({false, size_}); }
        ++size_;
    }
    void push_tool_call(ToolCallDelta value) {
        if (order_.empty()) {
            for (std::size_t i = 0; i < size_; ++i) { order_.push_back({false, i}); }
        }
        order_.push_back({true, tool_calls_.size()});
        tool_calls_.push_back(std::move(value));
    }
private:
    std::array<OutputDelta, 2> values_{};
    std::vector<OutputDelta> overflow_;
    std::vector<Publication> order_;
    std::size_t size_ = 0;
    std::vector<ToolCallDelta> tool_calls_;
};

// Gemma thinking budgets are accounting-only: no Qwen control suffix or stop-token ban.
// Tool markers never reach normal text. Constrained calls stream canonical JSON argument prefixes;
// untrusted free calls publish only after parsing and declaration validation.
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
    [[nodiscard]] bool constrained() const noexcept { return setup_.grammar != nullptr; }
    [[nodiscard]] runtime::ReasoningBan reasoning_ban() const noexcept { return {}; }
    void observe_constraint(bool timings, double prepare) noexcept;
    void constraint_uploaded(std::size_t bytes) noexcept;
    [[nodiscard]] std::optional<ConstraintObservation> constraint_observation() const;
    [[nodiscard]] std::uint64_t grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>);
    [[nodiscard]] bool constraint_pending() const noexcept;
    [[nodiscard]] std::exception_ptr constraint_failure() const noexcept;
    // Calls move out once, at the end of the turn.
    [[nodiscard]] std::vector<GeneratedToolCall> take_tool_calls() noexcept {
        return std::move(state_.tool_calls);
    }
    [[nodiscard]] bool has_tool_calls() const noexcept {
        return state_.diagnostics.structured_call_count != 0;
    }
    [[nodiscard]] ToolCallParseDiagnostics tool_call_parse_diagnostics() const noexcept {
        return state_.diagnostics;
    }
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept { return state_.reasoning_tokens; }
    [[nodiscard]] ThinkingBudgetStats thinking_stats() const noexcept;
    [[nodiscard]] std::optional<std::string> matched_stop_string() const;

private:
    enum class Mode : std::uint8_t { Content, ReasoningHeader, Reasoning, ToolCall };
    struct State {
        Mode mode = Mode::Content;
        std::string content;
        std::string reasoning;
        std::string header;
        // The body of the call being generated, between `<|tool_call>` and `<tool_call|>`.
        std::string tool_text;
        // Every call completed so far, in order; take_tool_calls() moves them out at the end.
        std::vector<GeneratedToolCall> tool_calls;
        PublishedOutput ready;
        bool call_name_published = false;
        std::size_t arguments_published = 0;
        ToolCallStream tool_stream;
        ToolCallParseDiagnostics diagnostics;
        std::uint32_t reasoning_tokens = 0;
        std::optional<std::size_t> matched_stop;
        bool json_content = false;
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
        TokenId tool_call_open = -1;
        TokenId tool_call_close = -1;
        // "call:" or "call:<name>" when the prompt ends inside a forced tool call.
        std::optional<std::string> forced_call;
        // Tools the prompt declares; calls to other names are suppressed with diagnostics.
        std::vector<std::string> callable_tools;
        // A tool choice's allowed_names narrow callable_tools.
        std::optional<std::vector<std::string>> allowed_names;
        std::unique_ptr<text::GrammarSession> grammar;
        bool stream_tools = false;
        bool combined_json = false;
    };
    OutputSession(std::shared_ptr<const Tokenizer> tokenizer, Setup setup);
    [[nodiscard]] bool is_stop(TokenId token) const noexcept;
    void consume(State& state, TokenId token) const;
    static void append_text(State& state, std::string bytes);
    static void close_reasoning(State& state);
    void close_tool_call(State& state) const;
    void stream_tool_call(State& state, const GeneratedToolCall* complete = nullptr) const;
    void publish_text(State& state, bool final) const;
    static void flush_terminal(State& state);
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
