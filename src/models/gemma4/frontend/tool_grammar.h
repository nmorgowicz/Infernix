#pragma once

#include "text/grammar.h"
#include <string>
#include <vector>

namespace infernix::models::gemma4 {

// A selected declaration. Non-strict tools constrain the wire format, not their schema.
struct GrammarTool {
    std::string name;
    std::string schema;
    bool strict = false;
};

[[nodiscard]] std::unique_ptr<text::GrammarSession> compile_tool_grammar(
    text::GrammarCompiler& compiler, const std::vector<GrammarTool>& tools,
    const ToolChoice& choice, TokenId open, TokenId close,
    const text::ReasoningFraming& reasoning, std::span<const TokenId> continuation,
    const std::optional<OutputConstraint>& body);

} // namespace infernix::models::gemma4
