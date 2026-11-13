#include "models/gemma4/frontend/output_session.h"

#include "models/gemma4/frontend/tool_parser.h"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace infernix::models::gemma4 {
namespace {
constexpr std::string_view kThoughtHeader = "thought\n";
constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

std::size_t utf8_width(unsigned char lead) {
    if (lead < 0x80) { return 1; }
    if (lead >= 0xc2 && lead <= 0xdf) { return 2; }
    if (lead >= 0xe0 && lead <= 0xef) { return 3; }
    if (lead >= 0xf0 && lead <= 0xf4) { return 4; }
    return 0;
}

// Retain incomplete UTF-8 at chunk boundaries; replace invalid scalar encodings rather than
// publishing invalid UTF-8. Stop-string candidate suffixes are retained independently.
std::string drain(std::string& pending, std::size_t hold, bool final) {
    const auto available = pending.size() - hold;
    std::string out;
    std::size_t at = 0;
    while (at < available) {
        const auto lead = static_cast<unsigned char>(pending[at]);
        const auto width = utf8_width(lead);
        if (width == 0) { out += kReplacement; ++at; continue; }
        bool valid = true;
        const auto limit = std::min(width, available - at);
        for (std::size_t i = 1; i < limit; ++i) {
            const auto byte = static_cast<unsigned char>(pending[at + i]);
            valid = valid && (byte & 0xc0) == 0x80;
            if (i == 1) {
                valid = valid && !(lead == 0xe0 && byte < 0xa0) &&
                    !(lead == 0xed && byte >= 0xa0) && !(lead == 0xf0 && byte < 0x90) &&
                    !(lead == 0xf4 && byte >= 0x90);
            }
        }
        if (!valid) { out += kReplacement; ++at; continue; }
        if (width > available - at) {
            if (!final) { break; }
            out += kReplacement;
            ++at;
            continue;
        }
        out.append(pending, at, width);
        at += width;
    }
    pending.erase(0, at);
    return out;
}

std::size_t suffix_prefix(std::string_view text, std::string_view stop) {
    for (auto size = std::min(text.size(), stop.size() - 1); size > 0; --size) {
        if (text.ends_with(stop.substr(0, size))) { return size; }
    }
    return 0;
}
} // namespace

OutputSession::OutputSession(std::shared_ptr<const Tokenizer> tokenizer, Setup setup)
    : tokenizer_(std::move(tokenizer)), setup_(std::move(setup)) {
    for (const auto& stop : setup_.stop_strings) {
        if (stop.text.empty()) { throw std::invalid_argument("gemma4 stop strings must be non-empty"); }
    }
    state_.mode = !setup_.output.raw && setup_.starts_in_reasoning ? Mode::Reasoning : Mode::Content;
    if (setup_.forced_call && !setup_.output.raw) {
        // The prompt already holds `<|tool_call>` and the call's opening; the output continues it.
        state_.mode = Mode::ToolCall;
        state_.tool_text = *setup_.forced_call;
        state_.diagnostics.marker_seen = true;
        if (setup_.stream_tools) { state_.tool_stream.feed(*setup_.forced_call); }
    }
}

bool OutputSession::is_stop(TokenId token) const noexcept {
    return std::find(setup_.stop_ids.begin(), setup_.stop_ids.end(), token) != setup_.stop_ids.end();
}

runtime::OutputDecision OutputSession::preview_model(std::span<const TokenId> tokens,
    std::uint32_t remaining, FinishReason limit_reason) {
    if (preview_pending_ || terminal_ || tokens.empty() || remaining == 0) {
        throw std::logic_error("gemma4 output preview requires live tokens and no pending preview");
    }
    staged_ = state_;
    staged_tokens_ = 0;
    runtime::OutputDecision result;
    for (const auto token : tokens) {
        if (staged_tokens_ == remaining) { break; }
        ++staged_tokens_;
        if (setup_.grammar) { setup_.grammar->accept(token); }
        consume(staged_, token);
        if (is_stop(token)) { result.finish_reason = FinishReason::StopToken; break; }
        if (match_stop_string(staged_)) { result.finish_reason = FinishReason::StopString; break; }
    }
    if (!result.finished() && staged_tokens_ == remaining) { result.finish_reason = limit_reason; }
    result.accepted_tokens = staged_tokens_;
    staged_terminal_ = result.finished();
    preview_pending_ = true;
    return result;
}

runtime::OutputDecision OutputSession::preview_control(std::span<const TokenId>, std::uint32_t) {
    throw std::logic_error("gemma4 has no target-control suffix");
}
runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (preview_pending_ || terminal_ || reason == FinishReason::None) {
        throw std::logic_error("gemma4 terminal preview requires an unfinished session");
    }
    staged_ = state_;
    staged_tokens_ = 0;
    staged_terminal_ = true;
    preview_pending_ = true;
    runtime::OutputDecision result;
    result.finish_reason = reason;
    return result;
}

void OutputSession::append_text(State& state, std::string bytes) {
    if (state.mode == Mode::Content) { state.content += bytes; return; }
    if (state.mode == Mode::Reasoning) { state.reasoning += bytes; return; }
    state.header += bytes;
    if (state.header.size() < kThoughtHeader.size() && kThoughtHeader.starts_with(state.header)) {
        return;
    }
    state.mode = Mode::Reasoning;
    if (state.header.starts_with(kThoughtHeader)) {
        state.reasoning.append(state.header, kThoughtHeader.size());
    } else { state.reasoning += state.header; }
    state.header.clear();
}
void OutputSession::close_reasoning(State& state) {
    if (state.mode == Mode::ReasoningHeader) {
        if (!kThoughtHeader.starts_with(state.header)) { state.reasoning += state.header; }
        state.header.clear();
    }
    state.mode = Mode::Content;
}

// Invalid or undeclared calls are quarantined, not leaked as raw Gemma wire text.
void OutputSession::close_tool_call(State& state) const {
    auto call = parse_tool_call_body(state.tool_text);
    auto reason = ToolCallParseFallbackReason::MalformedStructure;
    if (call) {
        const auto& tools = setup_.callable_tools;
        const bool callable = std::find(tools.begin(), tools.end(), call->name) != tools.end() &&
            (!setup_.allowed_names || std::find(setup_.allowed_names->begin(), setup_.allowed_names->end(),
                call->name) != setup_.allowed_names->end());
        if (!callable) {
            call.reset();
            reason = ToolCallParseFallbackReason::UndeclaredTool;
        }
    }
    publish_text(state, true);
    if (call) {
        if (setup_.stream_tools) { stream_tool_call(state, &*call); }
        else {
            state.ready.push_tool_call(ToolCallDelta{
                .index = static_cast<std::uint32_t>(state.tool_calls.size()),
                .name = call->name, .arguments = call->arguments_json, .complete = true});
        }
        state.tool_calls.push_back(std::move(*call));
        ++state.diagnostics.structured_call_count;
    } else {
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = reason;
        }
    }
    state.tool_text.clear();
    state.call_name_published = false;
    state.arguments_published = 0;
    state.tool_stream = {};
    state.mode = Mode::Content;
}

void OutputSession::stream_tool_call(State& state, const GeneratedToolCall* complete) const {
    const auto& name = complete ? complete->name : state.tool_stream.name();
    const auto& arguments = complete ? complete->arguments_json : state.tool_stream.arguments();
    if (name.empty()) { return; }
    if (std::find(setup_.callable_tools.begin(), setup_.callable_tools.end(), name) ==
        setup_.callable_tools.end() || (setup_.allowed_names &&
        std::find(setup_.allowed_names->begin(), setup_.allowed_names->end(), name) ==
            setup_.allowed_names->end())) { return; }
    if (arguments.size() < state.arguments_published) {
        throw std::logic_error("Gemma argument prefix shrank");
    }
    auto next = arguments.substr(state.arguments_published);
    if (!state.call_name_published || !next.empty() || complete) {
        state.ready.push_tool_call(ToolCallDelta{
            .index = static_cast<std::uint32_t>(state.tool_calls.size()),
            .name = state.call_name_published ? std::string{} : name,
            .arguments = std::move(next), .complete = complete != nullptr});
        state.call_name_published = true;
        state.arguments_published = arguments.size();
    }
}

void OutputSession::publish_text(State& state, bool final) const {
    const auto publish = [&](std::string& bytes, OutputChannel channel) {
        auto value = drain(bytes, final ? 0 : stop_hold(bytes, channel), final);
        if (!value.empty()) { state.ready.push_back({.channel = channel, .text = std::move(value)}); }
    };
    publish(state.reasoning, OutputChannel::Reasoning);
    publish(state.content, OutputChannel::Content);
}

// A turn cut inside a call never completes it and never exposes its wire representation.
void OutputSession::flush_terminal(State& state) {
    if (state.mode == Mode::ToolCall) {
        state.tool_text.clear();
        if (state.diagnostics.fallback_reason == ToolCallParseFallbackReason::None) {
            state.diagnostics.fallback_reason = ToolCallParseFallbackReason::MalformedStructure;
        }
        state.mode = Mode::Content;
    } else if (state.mode == Mode::ReasoningHeader) {
        close_reasoning(state);
    }
}
void OutputSession::consume(State& state, TokenId token) const {
    if (is_stop(token)) {
        if (setup_.publish_stop) { state.content += tokenizer_->token_bytes(token); }
        return;
    }
    const bool in_reasoning = state.mode == Mode::Reasoning || state.mode == Mode::ReasoningHeader;
    if (in_reasoning) { ++state.reasoning_tokens; }
    if (setup_.output.raw) {
        state.content += tokenizer_->token_bytes(token);
        return;
    }
    // A JSON/grammar content branch is bytes, not Gemma protocol. An admitted marker token
    // inside a JSON string must not reopen reasoning or become a tool call.
    if (!in_reasoning && state.mode == Mode::Content && constrained() &&
        (!setup_.stream_tools || state.json_content ||
            (setup_.combined_json && token != setup_.tool_call_open))) {
        state.json_content = true;
        state.content += tokenizer_->token_bytes(token);
        return;
    }
    if (state.mode == Mode::ToolCall) {
        if (token == setup_.tool_call_close) { close_tool_call(state); }
        else {
            const auto bytes = tokenizer_->token_bytes(token);
            state.tool_text += bytes;
            if (setup_.stream_tools) { state.tool_stream.feed(bytes); }
        }
        return;
    }
    if (token == setup_.tool_call_open) {
        if (in_reasoning) { close_reasoning(state); }
        publish_text(state, true);
        state.diagnostics.marker_seen = true;
        state.mode = Mode::ToolCall;
        state.tool_text.clear();
        return;
    }
    if (token == setup_.channel_open && !in_reasoning) {
        state.mode = Mode::ReasoningHeader;
        ++state.reasoning_tokens;
        if (setup_.output.preserve_special_tokens) { state.reasoning += tokenizer_->token_bytes(token); }
        return;
    }
    if (token == setup_.channel_close) {
        if (in_reasoning) {
            if (setup_.output.preserve_special_tokens) { state.reasoning += tokenizer_->token_bytes(token); }
            close_reasoning(state);
        } else if (setup_.output.preserve_special_tokens) { state.content += tokenizer_->token_bytes(token); }
        return;
    }
    if (tokenizer_->is_special(token) && !setup_.output.preserve_special_tokens &&
        !(constrained() && tokenizer_->token_bytes(token) == "<|\"|>")) { return; }
    append_text(state, tokenizer_->token_bytes(token));
}

bool OutputSession::match_stop_string(State& state) const {
    std::optional<std::size_t> best;
    std::size_t best_at = 0;
    for (std::size_t i = 0; i < setup_.stop_strings.size(); ++i) {
        const auto& stop = setup_.stop_strings[i];
        const auto& text = stop.channel == OutputChannel::Content ? state.content : state.reasoning;
        const auto at = text.find(stop.text);
        if (at == std::string::npos) { continue; }
        if (!best || (setup_.stop_strings[*best].channel == stop.channel && at < best_at)) {
            best = i;
            best_at = at;
        }
    }
    if (!best) { return false; }
    const auto& stop = setup_.stop_strings[*best];
    auto& text = stop.channel == OutputChannel::Content ? state.content : state.reasoning;
    text.resize(best_at);
    if (stop.include_in_output) { text += stop.text; }
    state.matched_stop = *best;
    return true;
}
std::size_t OutputSession::stop_hold(const std::string& text, OutputChannel channel) const {
    std::size_t hold = 0;
    for (const auto& stop : setup_.stop_strings) {
        if (stop.channel == channel) { hold = std::max(hold, suffix_prefix(text, stop.text)); }
    }
    return hold;
}

PublishedOutput OutputSession::commit_preview() {
    if (!preview_pending_) { throw std::logic_error("gemma4 output has no preview to commit"); }
    state_ = std::move(staged_);
    const bool final = staged_terminal_;
    if (final) { flush_terminal(state_); }
    publish_text(state_, final);
    if (!final && setup_.stream_tools && state_.mode == Mode::ToolCall) { stream_tool_call(state_); }
    PublishedOutput output = std::move(state_.ready);
    state_.ready = {};
    if (setup_.grammar) { setup_.grammar->confirm(); }
    terminal_ = final;
    discard_preview();
    return output;
}
void OutputSession::discard_preview() noexcept {
    if (setup_.grammar) { setup_.grammar->discard(); }
    staged_ = {};
    staged_tokens_ = 0;
    staged_terminal_ = false;
    preview_pending_ = false;
}
std::uint64_t OutputSession::grammar_masks(std::span<const TokenId> drafts, std::span<std::uint32_t> words) {
    if (!constrained()) { throw std::logic_error("mask requested for unconstrained Gemma output"); }
    return setup_.grammar->masks(drafts, words);
}
bool OutputSession::constraint_pending() const noexcept {
    return constrained() && setup_.grammar->pending();
}
std::exception_ptr OutputSession::constraint_failure() const noexcept {
    return constrained() ? setup_.grammar->failure() : std::exception_ptr{};
}
void OutputSession::observe_constraint(bool timings, double prepare) noexcept {
    if (constrained()) { setup_.grammar->observe(timings, prepare); }
}
void OutputSession::constraint_uploaded(std::size_t bytes) noexcept {
    if (constrained()) { setup_.grammar->uploaded(bytes); }
}
std::optional<ConstraintObservation> OutputSession::constraint_observation() const {
    if (!constrained()) { return {}; }
    auto result = setup_.grammar->observation();
    result.branch = state_.diagnostics.marker_seen ? ConstraintOutputBranch::Tools
                                                  : ConstraintOutputBranch::Content;
    return result;
}
ThinkingBudgetStats OutputSession::thinking_stats() const noexcept {
    ThinkingBudgetStats stats;
    stats.configured_budget = setup_.thinking_budget;
    if (setup_.thinking_budget) { stats.model_thinking_tokens = state_.reasoning_tokens; }
    return stats;
}
std::optional<std::string> OutputSession::matched_stop_string() const {
    if (!state_.matched_stop) { return {}; }
    return setup_.stop_strings.at(*state_.matched_stop).text;
}

} // namespace infernix::models::gemma4
