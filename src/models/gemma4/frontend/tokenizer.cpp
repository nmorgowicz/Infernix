#include "models/gemma4/frontend/tokenizer.h"
#include "infernix/types.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace infernix::models::gemma4 {
namespace {

constexpr std::string_view kSpace = "\u2581";

void preparation_checkpoint(const PreparationControl* control) {
    if (!control) { return; }
    if (control->cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "gemma4 tokenization cancelled");
    }
    if (control->deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control->deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "gemma4 tokenization expired");
    }
}

std::size_t skip_ws(std::string_view text, std::size_t index) {
    while (index < text.size() &&
           (text[index] == ' ' || text[index] == '\n' || text[index] == '\r' || text[index] == '\t')) {
        ++index;
    }
    return index;
}

int hex_value(char digit) {
    if (digit >= '0' && digit <= '9') { return digit - '0'; }
    if (digit >= 'a' && digit <= 'f') { return digit - 'a' + 10; }
    if (digit >= 'A' && digit <= 'F') { return digit - 'A' + 10; }
    return -1;
}

void append_utf8(std::string& out, unsigned code) {
    if (code < 0x80) {
        out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
        out.push_back(static_cast<char>(0xc0 | (code >> 6)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code < 0x10000) {
        out.push_back(static_cast<char>(0xe0 | (code >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (code >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
}

unsigned parse_hex4(std::string_view text, std::size_t& index) {
    unsigned code = 0;
    for (int nibble = 0; nibble < 4; ++nibble) {
        if (index >= text.size()) { throw std::invalid_argument("gemma4 tokenizer: bad unicode escape"); }
        const int value = hex_value(text[index++]);
        if (value < 0) { throw std::invalid_argument("gemma4 tokenizer: bad unicode escape"); }
        code = (code << 4) | static_cast<unsigned>(value);
    }
    return code;
}

std::string parse_string(std::string_view text, std::size_t& index) {
    if (index >= text.size() || text[index] != '"') {
        throw std::invalid_argument("gemma4 tokenizer: expected a string");
    }
    ++index;
    std::string out;
    while (index < text.size()) {
        const char ch = text[index++];
        if (ch == '"') { return out; }
        if (ch != '\\') {
            out.push_back(ch);
            continue;
        }
        if (index >= text.size()) { break; }
        const char esc = text[index++];
        switch (esc) {
        case '"':
        case '\\':
        case '/': out.push_back(esc); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
            unsigned code = parse_hex4(text, index);
            if (code >= 0xd800 && code < 0xdc00 && index + 1 < text.size() && text[index] == '\\' &&
                text[index + 1] == 'u') {
                index += 2;
                const unsigned low = parse_hex4(text, index);
                code               = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
            }
            append_utf8(out, code);
            break;
        }
        default: throw std::invalid_argument("gemma4 tokenizer: bad string escape");
        }
    }
    throw std::invalid_argument("gemma4 tokenizer: unterminated string");
}

std::int64_t parse_integer(std::string_view text, std::size_t& index) {
    std::int64_t sign = 1;
    if (index < text.size() && text[index] == '-') {
        sign = -1;
        ++index;
    }
    if (index >= text.size() || text[index] < '0' || text[index] > '9') {
        throw std::invalid_argument("gemma4 tokenizer: expected an integer");
    }
    std::int64_t value = 0;
    while (index < text.size() && text[index] >= '0' && text[index] <= '9') {
        value = value * 10 + (text[index++] - '0');
    }
    return sign * value;
}

void skip_value(std::string_view text, std::size_t& index) {
    index = skip_ws(text, index);
    if (index >= text.size()) { throw std::invalid_argument("gemma4 tokenizer: truncated json"); }
    const char ch = text[index];
    if (ch == '"') {
        (void)parse_string(text, index);
        return;
    }
    if (ch == '{' || ch == '[') {
        const char close = ch == '{' ? '}' : ']';
        ++index;
        while (true) {
            index = skip_ws(text, index);
            if (index < text.size() && text[index] == close) {
                ++index;
                return;
            }
            if (index < text.size() && (text[index] == ',' || text[index] == ':')) {
                ++index;
                continue;
            }
            skip_value(text, index);
        }
    }
    while (index < text.size() && text[index] != ',' && text[index] != '}' && text[index] != ']') {
        ++index;
    }
}

// Position just after `"key":` for the first occurrence of the key at or after `from`.
std::size_t value_of(std::string_view text, std::string_view key, std::size_t from) {
    const auto needle = std::string("\"") + std::string(key) + "\"";
    const auto at     = text.find(needle, from);
    if (at == std::string_view::npos) {
        throw std::invalid_argument("gemma4 tokenizer: " + std::string(key) + " is missing");
    }
    auto index = skip_ws(text, at + needle.size());
    if (index >= text.size() || text[index] != ':') {
        throw std::invalid_argument("gemma4 tokenizer: " + std::string(key) + " is malformed");
    }
    return skip_ws(text, index + 1);
}

template <class Entry>
void for_each_entry(std::string_view text, std::size_t& index, char open, Entry&& entry) {
    const char close = open == '{' ? '}' : ']';
    if (index >= text.size() || text[index] != open) {
        throw std::invalid_argument("gemma4 tokenizer: unexpected json shape");
    }
    ++index;
    while (true) {
        index = skip_ws(text, index);
        if (index < text.size() && text[index] == close) {
            ++index;
            return;
        }
        if (index < text.size() && text[index] == ',') {
            ++index;
            continue;
        }
        entry();
    }
}

std::uint64_t pair_key(int left, int right) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left)) << 32U) |
           static_cast<std::uint32_t>(right);
}

std::size_t utf8_width(unsigned char lead) {
    if ((lead & 0xe0U) == 0xc0U) { return 2; }
    if ((lead & 0xf0U) == 0xe0U) { return 3; }
    if ((lead & 0xf8U) == 0xf0U) { return 4; }
    return 1;
}

struct Merge {
    int rank   = 0;
    int result = 0;
};

struct Symbol {
    int id           = -1;
    std::size_t begin = 0;
    std::size_t end   = 0;
    int prev         = -1;
    int next         = -1;
};

struct Candidate {
    int rank  = 0;
    int pos   = 0;
    int left  = 0;
    int right = 0;
    bool operator>(const Candidate& other) const noexcept {
        return rank != other.rank ? rank > other.rank : pos > other.pos;
    }
};

} // namespace

struct Tokenizer::Impl {
    std::unordered_map<std::string, int> vocab;
    std::vector<std::string> pieces;
    std::vector<bool> special;
    std::unordered_map<std::uint64_t, Merge> merges;
    std::vector<std::pair<std::string, int>> added;
    std::array<int, 256> byte_ids{};

    void load(std::string_view text) {
        std::size_t index = value_of(text, "vocab", text.find("\"model\""));
        for_each_entry(text, index, '{', [&] {
            auto token = parse_string(text, index);
            index      = skip_ws(text, index);
            if (index >= text.size() || text[index] != ':') {
                throw std::invalid_argument("gemma4 tokenizer: vocab entry is malformed");
            }
            index         = skip_ws(text, index + 1);
            const auto id = parse_integer(text, index);
            if (id < 0 || id > (1 << 24)) {
                throw std::invalid_argument("gemma4 tokenizer: vocab id is out of range");
            }
            vocab.emplace(std::move(token), static_cast<int>(id));
        });
        int max_id = -1;
        for (const auto& [token, id] : vocab) { max_id = std::max(max_id, id); }
        pieces.assign(static_cast<std::size_t>(max_id) + 1, std::string());
        for (const auto& [token, id] : vocab) { pieces[static_cast<std::size_t>(id)] = token; }
        special.assign(pieces.size(), false);

        index    = value_of(text, "merges", index);
        int rank = 0;
        for_each_entry(text, index, '[', [&] {
            std::string left;
            std::string right;
            if (index < text.size() && text[index] == '"') {
                const auto joined = parse_string(text, index);
                const auto space  = joined.find(' ');
                if (space == std::string::npos) {
                    throw std::invalid_argument("gemma4 tokenizer: merge string has no space");
                }
                left  = joined.substr(0, space);
                right = joined.substr(space + 1);
            } else {
                std::vector<std::string> pair;
                for_each_entry(text, index, '[', [&] { pair.push_back(parse_string(text, index)); });
                if (pair.size() != 2) { throw std::invalid_argument("gemma4 tokenizer: merge must be a pair"); }
                left  = std::move(pair[0]);
                right = std::move(pair[1]);
            }
            const auto l = vocab.find(left);
            const auto r = vocab.find(right);
            const auto m = vocab.find(left + right);
            if (l != vocab.end() && r != vocab.end() && m != vocab.end()) {
                merges.emplace(pair_key(l->second, r->second), Merge{rank, m->second});
            }
            ++rank;
        });

        index = value_of(text, "added_tokens", 0);
        for_each_entry(text, index, '[', [&] {
            std::int64_t id = -1;
            std::string content;
            bool is_special = false;
            for_each_entry(text, index, '{', [&] {
                const auto key = parse_string(text, index);
                index          = skip_ws(text, index);
                if (index >= text.size() || text[index] != ':') {
                    throw std::invalid_argument("gemma4 tokenizer: added token is malformed");
                }
                index = skip_ws(text, index + 1);
                if (key == "id") {
                    id = parse_integer(text, index);
                } else if (key == "content") {
                    content = parse_string(text, index);
                } else if (key == "special") {
                    is_special = text.substr(index, 4) == "true";
                    skip_value(text, index);
                } else {
                    skip_value(text, index);
                }
            });
            if (id < 0 || content.empty()) {
                throw std::invalid_argument("gemma4 tokenizer: added token needs id and content");
            }
            const auto at = static_cast<std::size_t>(id);
            if (at >= pieces.size()) {
                pieces.resize(at + 1);
                special.resize(at + 1, false);
            }
            pieces[at]  = content;
            special[at] = is_special;
            added.emplace_back(std::move(content), static_cast<int>(id));
        });
        std::sort(added.begin(), added.end(), [](const auto& a, const auto& b) {
            return a.first.size() > b.first.size();
        });

        constexpr char hex[] = "0123456789ABCDEF";
        for (int byte = 0; byte < 256; ++byte) {
            std::string token = "<0x00>";
            token[3]          = hex[byte >> 4];
            token[4]          = hex[byte & 0xf];
            const auto found  = vocab.find(token);
            if (found == vocab.end()) {
                throw std::invalid_argument("gemma4 tokenizer: missing byte fallback token");
            }
            byte_ids[static_cast<std::size_t>(byte)] = found->second;
        }
    }

    void encode_word(std::string_view raw, std::vector<int>& ids,
                     const PreparationControl* control) const {
        if (raw.empty()) { return; }
        std::string text;
        text.reserve(raw.size() + raw.size() / 4);
        std::size_t work = 0;
        for (const char ch : raw) {
            if ((work++ & 1023U) == 0) { preparation_checkpoint(control); }
            if (ch == ' ') {
                text += kSpace;
            } else {
                text.push_back(ch);
            }
        }
        std::vector<Symbol> symbols;
        symbols.reserve(text.size());
        for (std::size_t index = 0; index < text.size();) {
            if ((work++ & 1023U) == 0) { preparation_checkpoint(control); }
            std::size_t width = utf8_width(static_cast<unsigned char>(text[index]));
            if (index + width > text.size()) { width = 1; }
            Symbol symbol;
            symbol.begin     = index;
            symbol.end       = index + width;
            const auto found = vocab.find(text.substr(index, width));
            symbol.id        = found == vocab.end() ? -1 : found->second;
            symbol.prev      = static_cast<int>(symbols.size()) - 1;
            symbol.next      = static_cast<int>(symbols.size()) + 1;
            symbols.push_back(symbol);
            index += width;
        }
        symbols.back().next = -1;

        std::priority_queue<Candidate, std::vector<Candidate>, std::greater<>> queue;
        const auto offer = [&](int pos) {
            if (pos < 0) { return; }
            const int next = symbols[static_cast<std::size_t>(pos)].next;
            if (next < 0) { return; }
            const int left  = symbols[static_cast<std::size_t>(pos)].id;
            const int right = symbols[static_cast<std::size_t>(next)].id;
            if (left < 0 || right < 0) { return; }
            const auto found = merges.find(pair_key(left, right));
            if (found == merges.end()) { return; }
            queue.push({found->second.rank, pos, left, right});
        };
        for (int pos = 0; pos + 1 < static_cast<int>(symbols.size()); ++pos) {
            if ((work++ & 1023U) == 0) { preparation_checkpoint(control); }
            offer(pos);
        }
        while (!queue.empty()) {
            if ((work++ & 1023U) == 0) { preparation_checkpoint(control); }
            const Candidate top = queue.top();
            queue.pop();
            Symbol& symbol = symbols[static_cast<std::size_t>(top.pos)];
            if (symbol.end == symbol.begin || symbol.id != top.left || symbol.next < 0) { continue; }
            Symbol& right = symbols[static_cast<std::size_t>(symbol.next)];
            if (right.id != top.right) { continue; }
            symbol.id  = merges.at(pair_key(top.left, top.right)).result;
            symbol.end = right.end;
            const int after = right.next;
            right.begin = right.end;
            right.id    = -2;
            symbol.next = after;
            if (after >= 0) { symbols[static_cast<std::size_t>(after)].prev = top.pos; }
            offer(symbol.prev);
            offer(top.pos);
        }

        for (int pos = 0; pos >= 0; pos = symbols[static_cast<std::size_t>(pos)].next) {
            if ((work++ & 1023U) == 0) { preparation_checkpoint(control); }
            const Symbol& symbol = symbols[static_cast<std::size_t>(pos)];
            if (symbol.id >= 0) {
                ids.push_back(symbol.id);
                continue;
            }
            for (std::size_t index = symbol.begin; index < symbol.end; ++index) {
                ids.push_back(byte_ids[static_cast<unsigned char>(text[index])]);
            }
        }
    }
};

Tokenizer::Tokenizer(std::string_view tokenizer_json) : impl_(std::make_unique<Impl>()) {
    if (tokenizer_json.find("\"BPE\"") == std::string_view::npos) {
        throw std::invalid_argument("gemma4 tokenizer: expected a BPE tokenizer json");
    }
    impl_->vocab.reserve(300000);
    impl_->merges.reserve(600000);
    impl_->load(tokenizer_json);
    if (impl_->vocab.empty() || impl_->merges.empty() || impl_->added.empty()) {
        throw std::invalid_argument("gemma4 tokenizer: empty vocabulary");
    }
}

Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

std::vector<int> Tokenizer::encode(std::string_view text, const PreparationControl* control) const {
    preparation_checkpoint(control);
    std::vector<int> ids;
    ids.reserve(text.size() / 3 + 4);
    std::size_t start = 0;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        preparation_checkpoint(control);
        const auto open = text.find('<', cursor);
        if (open == std::string_view::npos) { break; }
        int matched = -1;
        std::size_t width = 0;
        for (const auto& [content, id] : impl_->added) {
            if (text.compare(open, content.size(), content) == 0) {
                matched = id;
                width   = content.size();
                break;
            }
        }
        if (matched < 0) {
            cursor = open + 1;
            continue;
        }
        impl_->encode_word(text.substr(start, open - start), ids, control);
        ids.push_back(matched);
        start  = open + width;
        cursor = start;
    }
    impl_->encode_word(text.substr(start), ids, control);
    preparation_checkpoint(control);
    return ids;
}

int Tokenizer::vocab_size() const noexcept { return static_cast<int>(impl_->pieces.size()); }

bool Tokenizer::is_special(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < impl_->special.size() &&
           impl_->special[static_cast<std::size_t>(id)];
}

std::string Tokenizer::token_bytes(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= impl_->pieces.size()) {
        throw std::out_of_range("gemma4 tokenizer: token id is outside the vocabulary");
    }
    const std::string& piece = impl_->pieces[static_cast<std::size_t>(id)];
    if (is_special(id)) { return piece; }
    if (piece.size() == 6 && piece.starts_with("<0x") && piece.back() == '>') {
        const int high = hex_value(piece[3]);
        const int low  = hex_value(piece[4]);
        if (high >= 0 && low >= 0) { return std::string(1, static_cast<char>(high * 16 + low)); }
    }
    std::string out;
    out.reserve(piece.size());
    for (std::size_t index = 0; index < piece.size();) {
        if (piece.compare(index, kSpace.size(), kSpace) == 0) {
            out.push_back(' ');
            index += kSpace.size();
        } else {
            out.push_back(piece[index++]);
        }
    }
    return out;
}

std::string Tokenizer::decode(std::span<const int> ids, bool skip_special) const {
    std::string out;
    for (const int id : ids) {
        if (skip_special && is_special(id)) { continue; }
        out += token_bytes(id);
    }
    return out;
}

} // namespace infernix::models::gemma4
