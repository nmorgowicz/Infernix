#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace infernix { struct PreparationControl; }

namespace infernix::models::gemma4 {

// Gemma 4 BPE tokenizer read from tokenizer.json. Added tokens are matched first and are never
// normalized; the text between them has spaces replaced by U+2581 and is merged as one BPE word,
// with byte fallback for characters outside the vocabulary.
class Tokenizer {
public:
    explicit Tokenizer(std::string_view tokenizer_json);
    ~Tokenizer();

    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    [[nodiscard]] std::vector<int> encode(std::string_view text,
        const PreparationControl* control = nullptr) const;

    [[nodiscard]] int vocab_size() const noexcept;
    [[nodiscard]] bool is_special(int id) const noexcept;
    // Raw bytes one token contributes to decoded text. A byte-fallback token yields one byte,
    // which may be part of a multi-byte UTF-8 sequence. Special tokens yield their content.
    [[nodiscard]] std::string token_bytes(int id) const;
    [[nodiscard]] std::string decode(std::span<const int> ids, bool skip_special) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace infernix::models::gemma4
