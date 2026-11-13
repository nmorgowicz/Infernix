#pragma once

#include "infernix/types.h"
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace infernix::models::gemma4 {

// Parse one tool-call body, the text between `<|tool_call>` and `<tool_call|>`:
// `call:NAME{key:value,...}`. Keys are bare words, strings are wrapped in `<|"|>`, and values
// may nest objects and arrays (bounded by text::kMaxJsonNesting). Returns nullopt when the body
// is malformed, has trailing text, or repeats a key.
[[nodiscard]] std::optional<GeneratedToolCall> parse_tool_call_body(std::string_view body);

// Incremental decoder for grammar-validated Gemma arguments. It scans each byte once and buffers
// only the current key/string/scalar; emitted bytes are a stable prefix of the final canonical JSON.
class ToolCallStream {
public:
    void feed(std::string_view bytes);
    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::string& arguments() const noexcept { return arguments_; }
private:
    enum class Phase { Header, Key, Value, QuoteOpen, String, Scalar, AfterValue };
    Phase phase_ = Phase::Header;
    std::vector<char> containers_;
    std::string pending_;
    std::string name_;
    std::string arguments_;
};

// Parse every `<|tool_call>...<tool_call|>` block in text. Throws on a malformed block.
[[nodiscard]] std::vector<GeneratedToolCall> parse_tool_calls(std::string_view text);

} // namespace infernix::models::gemma4
