#include "models/gemma4/frontend/output_session.h"

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
void OutputSession::consume(State& state, TokenId token) const {
    if (is_stop(token)) {
        if (setup_.publish_stop) { state.content += tokenizer_->token_bytes(token); }
        return;
    }
    const bool in_reasoning = state.mode != Mode::Content;
    if (in_reasoning) { ++state.reasoning_tokens; }
    if (setup_.output.raw) {
        state.content += tokenizer_->token_bytes(token);
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
    if (tokenizer_->is_special(token) && !setup_.output.preserve_special_tokens) { return; }
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
    if (final && state_.mode == Mode::ReasoningHeader) { close_reasoning(state_); }
    PublishedOutput output;
    const auto publish = [&](std::string& bytes, OutputChannel channel) {
        auto value = drain(bytes, final ? 0 : stop_hold(bytes, channel), final);
        if (!value.empty()) { output.push_back(OutputDelta{.channel = channel, .text = std::move(value)}); }
    };
    publish(state_.reasoning, OutputChannel::Reasoning);
    publish(state_.content, OutputChannel::Content);
    terminal_ = final;
    discard_preview();
    return output;
}
void OutputSession::discard_preview() noexcept {
    staged_ = {};
    staged_tokens_ = 0;
    staged_terminal_ = false;
    preview_pending_ = false;
}
std::uint64_t OutputSession::grammar_masks(std::span<const TokenId>, std::span<std::uint32_t>) {
    throw std::logic_error("gemma4 P4 has no grammar mask session");
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
