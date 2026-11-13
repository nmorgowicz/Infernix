#pragma once

#include "infernix/types.h"
#include "models/gemma4/ngram.h"
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::models::gemma4 {

class PreparedPrompt {
public:
    PreparedPrompt() noexcept = default;
    PreparedPrompt(PreparedPrompt&&) noexcept = default;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept = default;
    PreparedPrompt(const PreparedPrompt&) = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const noexcept { return summary_; }
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept { return stats_; }
    [[nodiscard]] explicit operator bool() const noexcept { return !tokens_.empty(); }
    [[nodiscard]] const std::vector<TokenId>& tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const TokenId> token_ids() const noexcept { return tokens_; }
    [[nodiscard]] std::unique_ptr<NgramArchive::Request>
    bind_ngram(NgramArchive&, const NgramSessionHints& hints) {
        if (!hints.key.empty() || hints.reset || !hints.parent.empty() || hints.parent_generation) {
            throw std::invalid_argument("gemma4 P4 does not support ngram archives");
        }
        return {};
    }

private:
    PreparedPrompt(std::vector<TokenId> tokens, PromptSummary summary)
        : tokens_(std::move(tokens)), summary_(summary) {}

    // A declared tool's name ("" for none) and the tokens of "call:<name>", which follow
    // `<|tool_call>` when a tool choice forces the call.
    struct ToolOpener {
        std::string name;
        std::vector<TokenId> tokens;
    };

    std::vector<TokenId> tokens_;
    PromptSummary summary_{};
    PromptPreparationStats stats_{};
    std::vector<ToolOpener> tool_openers_;
    std::vector<std::string> tool_jsons_;
    std::string continuation_content_;
    TokenId tool_call_open_ = -1;
    TokenId channel_close_  = -1;
    // "call:" or "call:<name>" once a tool choice has forced the call into the prompt.
    std::optional<std::string> forced_call_;
    friend class Frontend;
    friend class Program;
};

} // namespace infernix::models::gemma4
