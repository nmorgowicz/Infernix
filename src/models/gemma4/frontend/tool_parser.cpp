#include "models/gemma4/frontend/tool_parser.h"

#include "text/json_input.h"
#include <nlohmann/json.hpp>
#include <cctype>
#include <stdexcept>
#include <string>

namespace infernix::models::gemma4 {
namespace {
using Json = nlohmann::ordered_json;

constexpr std::string_view kOpen  = "<|tool_call>";
constexpr std::string_view kClose = "<tool_call|>";
constexpr std::string_view kQuote = "<|\"|>";

struct Malformed {};

void skip_ws(std::string_view text, std::size_t& index) {
    while (index < text.size() && std::isspace(static_cast<unsigned char>(text[index]))) { ++index; }
}

void expect(std::string_view text, std::size_t& index, char ch) {
    skip_ws(text, index);
    if (index >= text.size() || text[index] != ch) { throw Malformed{}; }
    ++index;
}

Json parse_value(std::string_view text, std::size_t& index, std::size_t depth);

std::string parse_key(std::string_view text, std::size_t& index) {
    skip_ws(text, index);
    if (text.substr(index, kQuote.size()) == kQuote) {
        index += kQuote.size();
        const auto end = text.find(kQuote, index);
        if (end == std::string_view::npos) { throw Malformed{}; }
        std::string key(text.substr(index, end - index));
        index = end + kQuote.size();
        return key;
    }
    const auto begin = index;
    while (index < text.size() && text[index] != ':' && text[index] != ',' && text[index] != '}' &&
           !std::isspace(static_cast<unsigned char>(text[index]))) {
        ++index;
    }
    if (begin == index) { throw Malformed{}; }
    return std::string(text.substr(begin, index - begin));
}

Json parse_object(std::string_view text, std::size_t& index, std::size_t depth) {
    if (depth > text::kMaxJsonNesting) { throw Malformed{}; }
    expect(text, index, '{');
    Json object = Json::object();
    skip_ws(text, index);
    if (index < text.size() && text[index] == '}') {
        ++index;
        return object;
    }
    while (true) {
        std::string key = parse_key(text, index);
        expect(text, index, ':');
        if (object.contains(key)) { throw Malformed{}; }
        object[key] = parse_value(text, index, depth);
        skip_ws(text, index);
        if (index < text.size() && text[index] == ',') {
            ++index;
            continue;
        }
        expect(text, index, '}');
        return object;
    }
}

Json parse_array(std::string_view text, std::size_t& index, std::size_t depth) {
    if (depth > text::kMaxJsonNesting) { throw Malformed{}; }
    expect(text, index, '[');
    Json array = Json::array();
    skip_ws(text, index);
    if (index < text.size() && text[index] == ']') {
        ++index;
        return array;
    }
    while (true) {
        array.push_back(parse_value(text, index, depth));
        skip_ws(text, index);
        if (index < text.size() && text[index] == ',') {
            ++index;
            continue;
        }
        expect(text, index, ']');
        return array;
    }
}

// depth counts the containers enclosing this value; the call's argument object is depth 1.
Json parse_value(std::string_view text, std::size_t& index, std::size_t depth) {
    skip_ws(text, index);
    if (index >= text.size()) { throw Malformed{}; }
    if (text.substr(index, kQuote.size()) == kQuote) {
        index += kQuote.size();
        const auto end = text.find(kQuote, index);
        if (end == std::string_view::npos) { throw Malformed{}; }
        Json value = std::string(text.substr(index, end - index));
        index      = end + kQuote.size();
        return value;
    }
    if (text[index] == '{') { return parse_object(text, index, depth + 1); }
    if (text[index] == '[') { return parse_array(text, index, depth + 1); }
    const auto begin = index;
    while (index < text.size() && text[index] != ',' && text[index] != '}' && text[index] != ']') {
        ++index;
    }
    std::string_view token = text.substr(begin, index - begin);
    while (!token.empty() && std::isspace(static_cast<unsigned char>(token.back()))) {
        token.remove_suffix(1);
    }
    if (token.empty()) { throw Malformed{}; }
    if (token == "true") { return true; }
    if (token == "false") { return false; }
    if (token == "null") { return nullptr; }
    const Json number = Json::parse(token, nullptr, false);
    if (!number.is_discarded() && number.is_number()) { return number; }
    return std::string(token);
}
} // namespace

std::optional<GeneratedToolCall> parse_tool_call_body(std::string_view body) {
    constexpr std::string_view kCall = "call:";
    std::size_t index = 0;
    skip_ws(body, index);
    if (body.substr(index, kCall.size()) != kCall) { return std::nullopt; }
    index += kCall.size();
    const auto brace = body.find('{', index);
    if (brace == std::string_view::npos || brace == index) { return std::nullopt; }
    GeneratedToolCall call;
    call.name = std::string(body.substr(index, brace - index));
    for (const char ch : call.name) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '_' && ch != '-' && ch != '.') {
            return std::nullopt;
        }
    }
    index = brace;
    try {
        const Json arguments = parse_object(body, index, 1);
        skip_ws(body, index);
        if (index != body.size()) { return std::nullopt; }
        call.arguments_json = arguments.dump(-1, ' ', false, Json::error_handler_t::replace);
    } catch (const Malformed&) {
        return std::nullopt;
    }
    return call;
}

void ToolCallStream::feed(std::string_view bytes) {
    const auto dump = [](const Json& value) {
        return value.dump(-1, ' ', false, Json::error_handler_t::replace);
    };
    for (const char ch : bytes) {
        if (phase_ == Phase::Header) {
            if (ch != '{') { pending_ += ch; continue; }
            if (!pending_.starts_with("call:") || pending_.size() == 5) {
                throw std::logic_error("invalid constrained Gemma call header");
            }
            name_ = pending_.substr(5);
            pending_.clear();
            arguments_ += '{';
            containers_.push_back('{');
            phase_ = Phase::Key;
            continue;
        }
        if (phase_ == Phase::QuoteOpen) {
            pending_ += ch;
            if (!kQuote.starts_with(pending_)) { throw std::logic_error("invalid Gemma string opener"); }
            if (pending_ == kQuote) { pending_.clear(); phase_ = Phase::String; }
            continue;
        }
        if (phase_ == Phase::String) {
            pending_ += ch;
            if (pending_.ends_with(kQuote)) {
                pending_.resize(pending_.size() - kQuote.size());
                arguments_ += dump(pending_);
                pending_.clear();
                phase_ = Phase::AfterValue;
            }
            continue;
        }
        if (phase_ == Phase::Scalar) {
            if (ch != ',' && ch != '}' && ch != ']') { pending_ += ch; continue; }
            const auto value = Json::parse(pending_, nullptr, false);
            if (value.is_discarded()) { throw std::logic_error("invalid constrained Gemma scalar"); }
            arguments_ += dump(value);
            pending_.clear();
            phase_ = Phase::AfterValue;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) { continue; }
        if (phase_ == Phase::Key) {
            if (ch == ':' && !pending_.empty()) {
                arguments_ += dump(pending_);
                arguments_ += ':';
                pending_.clear();
                phase_ = Phase::Value;
            } else if (ch == '}' && pending_.empty()) {
                arguments_ += ch;
                containers_.pop_back();
                phase_ = Phase::AfterValue;
            } else { pending_ += ch; }
        } else if (phase_ == Phase::Value) {
            if (ch == '{' || ch == '[') {
                arguments_ += ch;
                containers_.push_back(ch);
                phase_ = ch == '{' ? Phase::Key : Phase::Value;
            } else if (ch == '<') { pending_ = ch; phase_ = Phase::QuoteOpen; }
            else if (ch == ']') {
                arguments_ += ch;
                containers_.pop_back();
                phase_ = Phase::AfterValue;
            } else { pending_ = ch; phase_ = Phase::Scalar; }
        } else if (phase_ == Phase::AfterValue) {
            if (ch == ',' && !containers_.empty()) {
                arguments_ += ch;
                phase_ = containers_.back() == '{' ? Phase::Key : Phase::Value;
            } else if ((ch == '}' || ch == ']') && !containers_.empty()) {
                arguments_ += ch;
                containers_.pop_back();
            } else { throw std::logic_error("invalid constrained Gemma argument separator"); }
        }
    }
}

std::vector<GeneratedToolCall> parse_tool_calls(std::string_view text) {
    std::vector<GeneratedToolCall> calls;
    std::size_t cursor = 0;
    while (true) {
        const auto open = text.find(kOpen, cursor);
        if (open == std::string_view::npos) { return calls; }
        const auto begin = open + kOpen.size();
        const auto close = text.find(kClose, begin);
        if (close == std::string_view::npos) {
            throw std::invalid_argument("gemma4 tool call: missing close");
        }
        auto call = parse_tool_call_body(text.substr(begin, close - begin));
        if (!call) { throw std::invalid_argument("gemma4 tool call: malformed body"); }
        calls.push_back(std::move(*call));
        cursor = close + kClose.size();
    }
}

} // namespace infernix::models::gemma4
