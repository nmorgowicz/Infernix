// CPU-only behavioral constraint tests against the complete real Gemma 4 vocabulary.
// INFERNIX_GEMMA4_HF supplies tokenizer.json and chat_template.jinja; no downloads or GPU.
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/tokenizer.h"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace infernix;
namespace g4 = models::gemma4;
using Json = nlohmann::ordered_json;
constexpr std::size_t kWords = 262144 / 32;

void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
std::string file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}
PromptInput input(bool thinking = false, bool tools = false) {
    PromptInput result;
    ChatMessage user;
    user.role = ChatRole::User;
    MessagePart part;
    part.kind = MessagePartKind::Text;
    part.text = "Answer or use a tool.";
    user.parts.push_back(std::move(part));
    result.messages.push_back(std::move(user));
    result.options.enable_thinking = thinking;
    if (tools) {
        result.options.tool_jsons = {
            R"({"type":"function","function":{"name":"weather","strict":true,"parameters":{"type":"object","properties":{"city":{"type":"string","enum":["Paris"]},"flags":{"type":"array","items":{"type":"boolean"}}},"required":["city","flags"],"additionalProperties":false}}})",
            R"({"type":"function","function":{"name":"clock","strict":true,"parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"],"additionalProperties":false}}})"};
    }
    return result;
}
bool allows(const std::vector<std::uint32_t>& mask, TokenId token, std::size_t row = 0) {
    return (mask.at(row * kWords + static_cast<std::size_t>(token) / 32) &
        (1u << (static_cast<unsigned>(token) % 32))) != 0;
}
std::vector<std::uint32_t> masks(g4::OutputSession& session, std::span<const TokenId> drafts = {}) {
    std::vector<std::uint32_t> result((drafts.size() + 1) * kWords);
    (void)session.grammar_masks(drafts, result);
    require(!session.constraint_failure(), "constraint compile failed");
    return result;
}
bool admits(g4::OutputSession& session, const std::vector<TokenId>& tokens) {
    require(tokens.size() < 64, "test draft is too long");
    const auto mask = masks(session, tokens);
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (!allows(mask, tokens[i], i)) { return false; }
    }
    return true;
}
struct Stream {
    std::string content;
    std::string reasoning;
    std::vector<ToolCallDelta> calls;
    std::vector<char> order;
    FinishReason finish = FinishReason::None;
    void collect(const g4::PublishedOutput& published) {
        const auto text = [&](const OutputDelta& delta) {
            if (delta.channel == OutputChannel::Content) { content += delta.text; order.push_back('C'); }
            else { reasoning += delta.text; order.push_back('R'); }
        };
        if (published.publication_order().empty()) {
            for (const auto& delta : published) { text(delta); }
        } else {
            for (const auto& item : published.publication_order()) {
                if (item.tool_call) { calls.push_back(published.tool_calls()[item.index]); order.push_back('T'); }
                else { text(published.begin()[item.index]); }
            }
        }
    }
};
void feed(g4::OutputSession& session, Stream& out, const std::vector<TokenId>& tokens, std::size_t step) {
    for (std::size_t at = 0; at < tokens.size();) {
        const auto count = std::min({step, tokens.size() - at, std::size_t{48}});
        const auto batch = std::span<const TokenId>(tokens).subspan(at, count);
        // Exercise speculative mask rollback, then real acceptance of the same independently
        // specified wire text. This checks every selected token against the actual 262k mask.
        const auto mask = masks(session, batch);
        for (std::size_t i = 0; i < count; ++i) { require(allows(mask, batch[i], i), "expected wire token was masked"); }
        const auto decision = session.preview_model(batch, 10000, FinishReason::OutputLimit);
        require(decision.accepted_tokens == count, "unexpected constrained acceptance frontier");
        out.finish = decision.finish_reason;
        out.collect(session.commit_preview());
        at += count;
        if (decision.finished()) { require(at == tokens.size(), "unexpected early finish"); break; }
    }
}
std::vector<TokenId> ending(const g4::Frontend& frontend, std::string_view wire, TokenId eos) {
    auto tokens = frontend.tokenize_text(wire);
    tokens.push_back(eos);
    return tokens;
}
void check_calls(const Stream& stream, const std::vector<GeneratedToolCall>& calls) {
    struct Call { std::string name; std::string arguments; unsigned completed = 0; };
    std::map<std::uint32_t, Call> reconstructed;
    for (const auto& delta : stream.calls) {
        auto& call = reconstructed[delta.index];
        require(call.completed == 0, "delta after a completed identity");
        if (!delta.name.empty()) { require(call.name.empty(), "duplicate function-name delta"); call.name = delta.name; }
        call.arguments += delta.arguments;
        call.completed += delta.complete;
    }
    require(reconstructed.size() == calls.size(), "missing or extra streamed identity");
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const auto& call = reconstructed.at(static_cast<std::uint32_t>(i));
        require(call.completed == 1 && call.name == calls[i].name && call.arguments == calls[i].arguments_json,
            "tool delta reconstruction differs from completed result");
        require(Json::parse(call.arguments).is_object(), "tool arguments are not JSON objects");
    }
}
void general_constraints(const g4::Frontend& frontend) {
    const auto schema = OutputConstraint::json_schema(
        R"({"type":"object","properties":{"ok":{"const":true},"n":{"const":2}},"required":["ok","n"],"additionalProperties":false})");
    auto prompt = frontend.prepare(input());
    // All three model EOS IDs are allowed only after a complete JSON document.
    for (const TokenId eos : {TokenId{1}, TokenId{106}, TokenId{50}}) {
        auto session = frontend.make_output_session(prompt, {}, {}, {}, schema);
        require(session.constrained(), "JSON schema did not activate constraints");
        const auto initial = masks(session);
        require(!allows(initial, eos) && !admits(session, frontend.tokenize_text("oops")), "schema admits invalid initial output");
        Stream stream;
        feed(session, stream, frontend.tokenize_text(R"({"ok":true,"n":2})"), 1);
        const auto complete = masks(session);
        for (const TokenId stop : {TokenId{1}, TokenId{106}, TokenId{50}}) {
            require(allows(complete, stop), "complete JSON does not admit every configured EOS");
        }
        feed(session, stream, {eos}, 1);
        require(stream.finish == FinishReason::StopToken && stream.content == R"({"ok":true,"n":2})",
            "schema streaming or multi-EOS output differs");
        require(session.constraint_observation()->complete, "completed schema matcher did not complete");
    }
    // A schema's false branch is rejected, not merely parsed after the fact.
    auto negative = frontend.make_output_session(prompt, {}, {}, {}, schema);
    require(!admits(negative, frontend.tokenize_text(R"({"ok":false})")), "schema admits a false const");
    const std::vector<std::pair<OutputConstraint, std::string>> cases = {
        {OutputConstraint::grammar("root ::= \"yes\"\n"), "yes"},
        {OutputConstraint::regex("yes|no"), "no"},
        {OutputConstraint::choice({"red", "blue"}), "blue"},
        {OutputConstraint::json_object(), R"({"nested":[1,true,null]})"},
        {OutputConstraint::json_object(), R"({"literal":"<|tool_call><tool_call|><channel|>"})"}};
    for (const auto& [constraint, answer] : cases) {
        auto session = frontend.make_output_session(prompt, {}, {}, {}, constraint);
        require(!admits(session, frontend.tokenize_text("garbage")), "constraint admits unrelated text");
        const auto before = masks(session);
        auto first = frontend.tokenize_text(answer);
        (void)session.preview_model(std::span<const TokenId>(first).first(1), 100, FinishReason::OutputLimit);
        session.discard_preview();
        require(before == masks(session), "discard changed committed grammar state");
        Stream stream;
        feed(session, stream, ending(frontend, answer, 106), 2);
        require(stream.content == answer && stream.finish == FinishReason::StopToken,
            "GBNF/regex/choice/JSON output differs");
    }
    auto thinking_prompt = frontend.prepare(input(true));
    ThinkingControlOptions thinking;
    thinking.budget = 1;
    auto thinking_session = frontend.make_output_session(thinking_prompt, {}, {}, thinking, schema);
    const auto close = frontend.tokenize_text("<channel|>").at(0);
    require(allows(masks(thinking_session), close), "Gemma reasoning close is not allowed");
    Stream thought;
    feed(thinking_session, thought, ending(frontend, "long plan<channel|>{\"ok\":true,\"n\":2}", 50), 1);
    require(thought.reasoning == "long plan" && thought.content == R"({"ok":true,"n":2})" &&
        thinking_session.pending_control_tokens().empty() && thinking_session.thinking_stats().model_thinking_tokens > 1,
        "Gemma reasoning framing or accounting-only budget changed");
    bool invalid = false;
    try { (void)frontend.make_output_session(prompt, {}, {}, {}, OutputConstraint::grammar("root ::= (")); }
    catch (const RequestError& error) { invalid = error.kind() == RequestErrorKind::InvalidGrammar; }
    require(invalid, "invalid grammar did not fail admission");
    std::cout << "general constraints: schema, GBNF, regex, choice, JSON, reasoning, rollback, three EOS\n";
}
void tool_constraints(const g4::Frontend& frontend) {
    constexpr std::string_view weather = "<|tool_call>call:weather{city:<|\"|>Paris<|\"|>,flags:[true,false]}<tool_call|>";
    constexpr std::string_view clock = "<|tool_call>call:clock{zone:<|\"|>U\"T\nC<|\"|>}<tool_call|>";
    auto prompt = frontend.prepare(input(true, true));
    for (const std::size_t step : {std::size_t{1}, std::size_t{48}}) {
        auto session = frontend.make_output_session(prompt, {});
        const auto open = frontend.tokenize_text("<|tool_call>").at(0);
        require(!allows(masks(session), open), "tool opener is allowed inside reasoning");
        Stream stream;
        feed(session, stream, ending(frontend, "plan<channel|>Before" + std::string(weather) + std::string(clock), 50), step);
        const auto calls = session.take_tool_calls();
        require(calls.size() == 2 && calls[0].name == "weather" && calls[1].name == "clock" &&
            calls[0].arguments_json == R"({"city":"Paris","flags":[true,false]})" &&
            calls[1].arguments_json == "{\"zone\":\"U\\\"T\\nC\"}", "Gemma style nested/escaped arguments differ");
        check_calls(stream, calls);
        require(stream.reasoning == "plan" && stream.content == "Before", "tool framing leaked into text");
        if (step == 1) { require(stream.calls.size() > 4, "tool argument deltas were buffered whole"); }
        const auto first_tool = std::find(stream.order.begin(), stream.order.end(), 'T');
        require(std::find(first_tool, stream.order.end(), 'C') == stream.order.end(), "content followed constrained tools");
    }
    // Required and named choices continue the prompt's forced opener; non-parallel masks reject
    // another call and any trailing text, while every model EOS can finish the completed call.
    for (const bool named : {false, true}) {
        auto forced = frontend.prepare(input(false, true));
        ToolChoice choice;
        choice.mode = ToolChoiceMode::Required;
        choice.parallel = false;
        if (named) { choice.allowed_names = std::vector<std::string>{"clock"}; }
        g4::Frontend::apply_tool_choice(forced, choice);
        auto session = frontend.make_output_session(forced, {}, {}, {}, {}, choice);
        require(!allows(masks(session), 106), "required choice admits EOS before a call");
        const auto wrong = frontend.tokenize_text(named ? "{zone:1}" : "missing{}");
        require(!admits(session, wrong), "named/schema/undeclared constraint admits wrong call");
        Stream stream;
        feed(session, stream, frontend.tokenize_text(named ? "{zone:<|\"|>UTC<|\"|>}<tool_call|>"
            : "clock{zone:<|\"|>UTC<|\"|>}<tool_call|>"), 1);
        const auto done = masks(session);
        require(!allows(done, frontend.tokenize_text("<|tool_call>").at(0)) &&
            !admits(session, frontend.tokenize_text("After")), "non-parallel call admits another call or text");
        feed(session, stream, {106}, 1);
        check_calls(stream, session.take_tool_calls());
    }
    // A cut call can stream a prefix but never publishes a completed identity or raw markers.
    auto partial = frontend.prepare(input(false, true));
    ToolChoice required;
    required.mode = ToolChoiceMode::Required;
    g4::Frontend::apply_tool_choice(partial, required);
    auto cut = frontend.make_output_session(partial, {}, {}, {}, {}, required);
    Stream cut_stream;
    feed(cut, cut_stream, frontend.tokenize_text("clock{zone:<|\"|>UT"), 1);
    (void)cut.preview_terminal(FinishReason::OutputLimit);
    cut_stream.collect(cut.commit_preview());
    require(cut.take_tool_calls().empty() && cut_stream.content.empty() && !cut_stream.calls.empty() &&
        std::none_of(cut_stream.calls.begin(), cut_stream.calls.end(), [](const auto& call) { return call.complete; }),
        "cut constrained call leaked or completed");
    // Combined output uses the JSON branch or tools, not a free-text escape hatch.
    auto combined_prompt = frontend.prepare(input(false, true));
    const auto json = OutputConstraint::json_schema(R"({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false})");
    auto combined = frontend.make_output_session(combined_prompt, {}, {}, {}, json);
    require(!admits(combined, frontend.tokenize_text("free text")), "combined constraint admits free text");
    Stream answer;
    feed(combined, answer, ending(frontend, R"({"ok":true})", 1), 1);
    require(answer.content == R"({"ok":true})" && answer.calls.empty(), "combined JSON branch was parsed as tools");
    auto combined_tools = frontend.make_output_session(combined_prompt, {}, {}, {}, json);
    Stream tool_answer;
    feed(combined_tools, tool_answer, ending(frontend, clock, 106), 2);
    check_calls(tool_answer, combined_tools.take_tool_calls());
    require(tool_answer.content.empty(), "combined tool branch leaked wire text");
    std::cout << "tool constraints: Gemma style, nested arguments, split streams, required/named/parallel, cut, combined JSON\n";
}
} // namespace

int main() {
    const char* hf = std::getenv("INFERNIX_GEMMA4_HF");
    if (!hf || !*hf) { std::cout << "SKIP: INFERNIX_GEMMA4_HF is not set\n"; return 77; }
    try {
        const std::filesystem::path root(hf);
        const auto tokenizer_json = file(root / "tokenizer.json");
        const auto chat = file(root / "chat_template.jinja");
        const g4::Tokenizer tokenizer(tokenizer_json);
        require(tokenizer.vocab_size() == 262144, "test requires real Gemma 4 262144-token vocabulary");
        g4::TextResources resources;
        resources.tokenizer_json = tokenizer_json;
        resources.chat_template_jinja = chat;
        resources.generation_config_json = R"({"eos_token_id":[1,106,50]})";
        g4::FrontendOptions options;
        options.max_context = 262144;
        options.chat_template_path = std::filesystem::path(INFERNIX_SOURCE_DIR) / "tools/chat_templates/gemma4_fixed.jinja";
        const g4::Frontend frontend(resources, options);
        general_constraints(frontend);
        tool_constraints(frontend);
        std::cout << "OK gemma4 real-vocabulary grammar behavior\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
