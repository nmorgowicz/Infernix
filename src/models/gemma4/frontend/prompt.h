#pragma once

#include "infernix/types.h"
#include "models/gemma4/ngram.h"
#include <memory>
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
    std::vector<TokenId> tokens_;
    PromptSummary summary_{};
    PromptPreparationStats stats_{};
    friend class Frontend;
    friend class Program;
};

} // namespace infernix::models::gemma4
