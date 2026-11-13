#pragma once

#include "models/gemma4/frontend/output_session.h"
#include "models/gemma4/frontend/prompt.h"
#include "models/gemma4/model.h"
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::gemma4 {

struct FrontendOptions {
    std::uint32_t max_context = 2048;
    std::filesystem::path chat_template_path;
    std::size_t grammar_cache_bytes = 256ULL * 1024 * 1024;
};

class Frontend {
public:
    Frontend(const TextResources& resources, FrontendOptions options);
    Frontend(const Frontend&) noexcept = default;
    Frontend& operator=(const Frontend&) noexcept = default;
    Frontend(Frontend&&) noexcept = default;
    Frontend& operator=(Frontend&&) noexcept = default;
    ~Frontend();
    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
        const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
        const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
        bool allow_prefix_identity = true) const;
    [[nodiscard]] std::vector<TokenId> tokenize_text(std::string_view text) const;
    [[nodiscard]] std::string detokenize(std::span<const TokenId> token_ids) const;
    [[nodiscard]] std::string detokenize_bytes(std::span<const TokenId> token_ids) const;
    [[nodiscard]] std::string render(const PromptInput& input) const;
    [[nodiscard]] ChatTokens tokenize_chat(PromptInput input) const;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const noexcept { return {}; }
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
        const StopPolicy& caller_stop, const OutputOptions& output = {},
        const ThinkingControlOptions& thinking = {},
        const std::optional<OutputConstraint>& constraint = {},
        const ToolChoice& tool_choice = {}) const;
    void warm_constraints() const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    [[nodiscard]] const ModelSamplingDefaults& sampling_defaults() const noexcept;
    [[nodiscard]] ChatTemplateRoleSupport chat_template_role_support() const noexcept;
    static void apply_tool_choice(const PreparedPrompt&, const ToolChoice& tool_choice);
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
};

} // namespace infernix::models::gemma4
