// Gemma 4 frontend against Hugging Face fixtures, CPU only.
//
//   INFERNIX_GEMMA4_HF        a Hugging Face Gemma 4 directory holding tokenizer.json and
//                             chat_template.jinja (and generation_config.json when present);
//                             absent means skip (77).
//   INFERNIX_GEMMA4_FIXTURES  optional fixture directory; default tests/models/gemma4/fixtures.
//
// The fixture directory holds tok_fixtures.json (text, ids, decoded) and chat_fixtures20.json
// (messages, tools, enable_thinking, continue_final, text, ids). As in the fork's fixture harness,
// chats use their reference HF chat_template.jinja. The separate behavioral checks use the fixed override.
// The remaining checks cover tool calls through the output session, required tool choice, EOS arrays,
// history-argument refusal and the thinking precedence rules.

#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
namespace g4 = infernix::models::gemma4;
namespace fs = std::filesystem;

int failures = 0;

void check(bool ok, const std::string& label) {
    if (ok) { return; }
    std::cerr << "FAIL " << label << '\n';
    ++failures;
}

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

infernix::ChatRole role_of(const std::string& name) {
    if (name == "system") { return infernix::ChatRole::System; }
    if (name == "developer") { return infernix::ChatRole::Developer; }
    if (name == "user") { return infernix::ChatRole::User; }
    if (name == "assistant") { return infernix::ChatRole::Assistant; }
    if (name == "tool") { return infernix::ChatRole::Tool; }
    throw std::runtime_error("unknown role " + name);
}

infernix::ChatMessage message(infernix::ChatRole role, std::string text) {
    infernix::ChatMessage result;
    result.role = role;
    result.parts.push_back({.kind = infernix::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return result;
}

infernix::ToolCall call(std::string id, std::string name, std::string arguments_json) {
    return {.id = std::move(id), .name = std::move(name), .arguments_json = std::move(arguments_json)};
}

std::string declaration(const char* name) {
    return Json{{"type", "function"},
                {"function",
                 {{"name", name}, {"parameters", {{"type", "object"}, {"properties", Json::object()}}}}}}
        .dump();
}

infernix::PromptInput prompt_of(const Json& item) {
    infernix::PromptInput input;
    for (const Json& source : item.at("messages")) {
        auto chat = message(role_of(source.at("role").get<std::string>()), source.value("content", std::string()));
        chat.reasoning_content = source.value("reasoning_content", std::string());
        chat.tool_call_id = source.value("tool_call_id", std::string());
        if (source.contains("tool_calls")) {
            for (const Json& entry : source.at("tool_calls")) {
                chat.tool_calls.push_back(call(entry.value("id", std::string()),
                    entry.at("function").at("name").get<std::string>(),
                    entry.at("function").at("arguments").dump()));
            }
        }
        input.messages.push_back(std::move(chat));
    }
    if (item.contains("tools")) {
        for (const Json& tool : item.at("tools")) { input.options.tool_jsons.push_back(tool.dump()); }
    }
    input.options.enable_thinking = item.value("enable_thinking", false);
    if (item.value("continue_final", false)) {
        input.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
    }
    return input;
}

g4::Frontend make_frontend(const std::string& tokenizer_json, const std::string& embedded_template,
    const std::string& generation_config, std::uint32_t max_context, const fs::path& template_path) {
    g4::TextResources resources;
    resources.tokenizer_json = tokenizer_json;
    resources.chat_template_jinja = embedded_template;
    resources.generation_config_json = generation_config;
    g4::FrontendOptions options;
    options.max_context = max_context;
    options.chat_template_path = template_path;
    return g4::Frontend(resources, options);
}

struct Run {
    std::string content;
    std::string reasoning;
    infernix::FinishReason finish = infernix::FinishReason::None;
    std::uint32_t accepted = 0;
    std::vector<infernix::ToolCallDelta> deltas;
    std::vector<std::string> order; // "content", "reasoning" and "tool" in publication order
};

void collect(Run& run, const g4::PublishedOutput& published) {
    const auto text = [&](const infernix::OutputDelta& delta) {
        if (delta.channel == infernix::OutputChannel::Content) {
            run.content += delta.text;
            run.order.emplace_back("content");
        } else {
            run.reasoning += delta.text;
            run.order.emplace_back("reasoning");
        }
    };
    if (published.publication_order().empty()) {
        for (const auto& delta : published) { text(delta); }
    } else {
        for (const auto& item : published.publication_order()) {
            if (item.tool_call) {
                run.deltas.push_back(published.tool_calls()[item.index]);
                run.order.emplace_back("tool");
            } else { text(published.begin()[item.index]); }
        }
    }
}

infernix::ToolChoice free_choice() {
    infernix::ToolChoice choice;
    choice.constraints = infernix::ToolConstraintMode::Automatic;
    return choice;
}

// Feeds `tokens` in previews of `step` tokens; a turn the tokens do not finish ends at the output limit.
Run feed(g4::OutputSession& session, const std::vector<infernix::TokenId>& tokens, std::size_t step) {
    Run run;
    for (std::size_t at = 0; at < tokens.size() && run.finish == infernix::FinishReason::None; at += step) {
        const std::size_t count = std::min(step, tokens.size() - at);
        const auto decision = session.preview_model(
            std::span<const infernix::TokenId>(tokens).subspan(at, count), 1000,
            infernix::FinishReason::OutputLimit);
        run.finish = decision.finish_reason;
        run.accepted += decision.accepted_tokens;
        collect(run, session.commit_preview());
    }
    if (run.finish == infernix::FinishReason::None) {
        run.finish = session.preview_terminal(infernix::FinishReason::OutputLimit).finish_reason;
        collect(run, session.commit_preview());
    }
    return run;
}

std::string first_difference(const std::string& got, const std::string& want) {
    std::size_t at = 0;
    while (at < got.size() && at < want.size() && got[at] == want[at]) { ++at; }
    const std::size_t from = at > 40 ? at - 40 : 0;
    return "at byte " + std::to_string(at) + "\n got:  " + got.substr(from, 120) +
           "\n want: " + want.substr(from, 120);
}

std::size_t count_of(const std::string& text, const std::string& part) {
    std::size_t count = 0;
    for (auto at = text.find(part); at != std::string::npos; at = text.find(part, at + part.size())) { ++count; }
    return count;
}

bool contains(const std::string& text, const std::string& part) { return text.find(part) != std::string::npos; }

bool throws_invalid(const g4::Frontend& frontend, const infernix::PromptInput& input) {
    try { (void)frontend.render(input); }
    catch (const infernix::RequestError&) { return false; }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

void tool_call_checks(const g4::Frontend& frontend) {
    const auto ids = [&](std::string_view text) { return frontend.tokenize_text(text); };
    const auto with = [](std::vector<infernix::TokenId> tokens, std::initializer_list<infernix::TokenId> tail) {
        tokens.insert(tokens.end(), tail);
        return tokens;
    };
    const auto marker = [&](std::string_view text) {
        const auto tokens = ids(text);
        check(tokens.size() == 1, "marker is one token: " + std::string(text));
        return tokens.empty() ? infernix::TokenId{-1} : tokens.front();
    };
    const infernix::TokenId open = marker("<|tool_call>"), close = marker("<tool_call|>"),
                            turn_end = marker("<turn|>"), response = marker("<|tool_response>"),
                            channel_open = marker("<|channel>"), channel_close = marker("<channel|>");
    const auto tools_input = [] {
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "What time is it?"));
        input.options.enable_thinking = false;
        for (const char* name : {"get_weather", "get_time"}) { input.options.tool_jsons.push_back(declaration(name)); }
        return input;
    };

    {   // A named tool choice is part of the prompt: `<|tool_call>call:get_time` ends it.
        auto prompt = frontend.prepare(tools_input(), {});
        const auto plain = prompt.tokens();
        infernix::ToolChoice choice;
        choice.mode = infernix::ToolChoiceMode::Required;
        choice.allowed_names = std::vector<std::string>{"get_time"};
        g4::Frontend::apply_tool_choice(prompt, choice);
        auto expect = plain;
        expect.push_back(open);
        for (const auto id : ids("call:get_time")) { expect.push_back(id); }
        check(prompt.tokens() == expect && prompt.summary().prompt_tokens == expect.size(),
              "named tool choice appends <|tool_call>call:get_time");
        g4::Frontend::apply_tool_choice(prompt, choice);
        check(prompt.tokens() == expect, "applying a tool choice twice changes nothing");
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, choice);
        const Run run = feed(session, with(ids("{zone:<|\"|>Berlin<|\"|>}"), {close, turn_end}), 3);
        const auto calls = session.take_tool_calls();
        check(run.finish == infernix::FinishReason::StopToken && run.content.empty() && calls.size() == 1 &&
                  calls[0].name == "get_time" && calls[0].arguments_json == "{\"zone\":\"Berlin\"}",
              "named tool choice parses the forced call");
        std::string arguments;
        std::size_t completed = 0;
        for (const auto& delta : run.deltas) {
            check(delta.index == 0, "forced call index stays zero");
            arguments += delta.arguments;
            completed += delta.complete;
        }
        check(!run.deltas.empty() && run.deltas.front().name == "get_time" &&
                  arguments == "{\"zone\":\"Berlin\"}" && completed == 1,
              "forced call argument deltas reconstruct exactly one completed call");
    }
    {   // Required without a single allowed name opens `call:`; the model names the tool.
        auto prompt = frontend.prepare(tools_input(), {});
        infernix::ToolChoice choice;
        choice.mode = infernix::ToolChoiceMode::Required;
        g4::Frontend::apply_tool_choice(prompt, choice);
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, choice);
        (void)feed(session, with(ids("get_weather{city:<|\"|>Paris<|\"|>}"), {close, turn_end}), 1);
        const auto calls = session.take_tool_calls();
        check(calls.size() == 1 && calls[0].name == "get_weather" &&
                  calls[0].arguments_json == "{\"city\":\"Paris\"}",
              "required tool choice opens the call without a name");
    }
    {   // A thought left open when the choice is applied is closed first.
        auto input = tools_input();
        input.options.enable_thinking = true;
        auto prompt = frontend.prepare(std::move(input), {});
        check(prompt.summary().starts_in_reasoning, "a thinking prompt starts in reasoning");
        const auto plain = prompt.tokens();
        infernix::ToolChoice choice;
        choice.mode = infernix::ToolChoiceMode::Required;
        g4::Frontend::apply_tool_choice(prompt, choice);
        const auto& tokens = prompt.tokens();
        check(!prompt.summary().starts_in_reasoning && tokens.size() == plain.size() + 2 + ids("call:").size() &&
                  tokens[plain.size()] == channel_close && tokens[plain.size() + 1] == open,
              "a required tool choice closes the open thought first");
    }
    {   // Required needs declared tools, and a named tool declared; make_output_session needs it applied.
        infernix::PromptInput bare;
        bare.messages.push_back(message(infernix::ChatRole::User, "hi"));
        auto prompt = frontend.prepare(std::move(bare), {});
        infernix::ToolChoice choice;
        choice.mode = infernix::ToolChoiceMode::Required;
        bool threw = false;
        try { g4::Frontend::apply_tool_choice(prompt, choice); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "required tool choice without tools throws");

        auto other = frontend.prepare(tools_input(), {});
        infernix::ToolChoice missing;
        missing.mode = infernix::ToolChoiceMode::Required;
        missing.allowed_names = std::vector<std::string>{"send_mail"};
        threw = false;
        try { g4::Frontend::apply_tool_choice(other, missing); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "a named tool choice outside the declared tools throws");

        threw = false;
        try { (void)frontend.make_output_session(other, {}, {}, {}, std::nullopt, missing); }
        catch (const std::invalid_argument&) { threw = true; }
        check(threw, "an unapplied required tool choice is refused");
    }
    // Constrained allowed_names rejection is exercised through real masks in the grammar test;
    // feeding an illegal sampled token would violate the output-session contract.
    {   // Two calls in one turn: stable indexes, each published once and complete, no marker leaks.
        const auto turn = [&] {
            std::vector<infernix::TokenId> tokens{open};
            for (const auto id : ids("call:get_weather{city:<|\"|>Paris<|\"|>}")) { tokens.push_back(id); }
            tokens.push_back(close);
            tokens.push_back(open);
            for (const auto id : ids("call:get_time{zone:<|\"|>UTC<|\"|>}")) { tokens.push_back(id); }
            tokens.push_back(close);
            tokens.push_back(response);
            return tokens;
        }();
        for (const std::size_t step : {std::size_t{1}, std::size_t{4}, turn.size()}) {
            auto prompt = frontend.prepare(tools_input(), {});
            auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, free_choice());
            const Run run = feed(session, turn, step);
            const auto calls = session.take_tool_calls();
            const std::string label = " (step " + std::to_string(step) + ")";
            check(run.finish == infernix::FinishReason::StopToken && run.content.empty() && run.reasoning.empty(),
                  "parallel calls leave no text" + label);
            check(calls.size() == 2 && calls[0].name == "get_weather" && calls[1].name == "get_time" &&
                      calls[1].arguments_json == "{\"zone\":\"UTC\"}",
                  "parallel calls parse in order" + label);
            check(run.deltas.size() == 2 && run.deltas[0].index == 0 && run.deltas[1].index == 1 &&
                      run.deltas[0].complete && run.deltas[1].complete &&
                      run.deltas[0].name == "get_weather" && run.deltas[1].name == "get_time" &&
                      run.deltas[0].arguments == calls[0].arguments_json &&
                      run.deltas[1].arguments == calls[1].arguments_json,
                  "each call publishes exactly one complete delta matching its result" + label);
        }
    }
    {   // Text before a call is published before it; reasoning before both.
        std::vector<infernix::TokenId> tokens{channel_open};
        for (const auto id : ids("thought\nplan")) { tokens.push_back(id); }
        tokens.push_back(channel_close);
        for (const auto id : ids("Sure.")) { tokens.push_back(id); }
        tokens.push_back(open);
        for (const auto id : ids("call:get_time{}")) { tokens.push_back(id); }
        tokens.push_back(close);
        tokens.push_back(response);
        auto prompt = frontend.prepare(tools_input(), {});
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, free_choice());
        const Run run = feed(session, tokens, 2);
        const auto first_tool = std::find(run.order.begin(), run.order.end(), "tool");
        const auto last_text = std::find_if(run.order.rbegin(), run.order.rend(),
            [](const std::string& kind) { return kind != "tool"; });
        check(run.reasoning == "plan" && run.content == "Sure." && run.deltas.size() == 1 &&
                  first_tool != run.order.end() && last_text.base() <= first_tool,
              "reasoning and text precede the call");
        check(session.has_tool_calls() && session.tool_call_parse_diagnostics().structured_call_count == 1 &&
                  session.tool_call_parse_diagnostics().marker_seen,
              "diagnostics count the structured call");
    }
    {   // The publication container and Engine preserve text after a free call in the same batch.
        auto prompt = frontend.prepare(tools_input(), {});
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, free_choice());
        auto tokens = ids("Before<|tool_call>call:get_time{}<tool_call|>After");
        tokens.push_back(response);
        const auto run = feed(session, tokens, tokens.size());
        check(run.content == "BeforeAfter" && run.order ==
            std::vector<std::string>{"content", "tool", "content"},
            "same-batch free text/tool/text ordering");
    }
    {   // Invalid free calls are quarantined without leaking raw Gemma wire text.
        struct Case { const char* label; std::string body; infernix::ToolCallParseFallbackReason reason; };
        const std::vector<Case> cases = {
            {"undeclared tool", "call:send_mail{to:1}", infernix::ToolCallParseFallbackReason::UndeclaredTool},
            {"malformed body", "call:get_time{zone:}x", infernix::ToolCallParseFallbackReason::MalformedStructure},
            {"no call prefix", "get_time{}", infernix::ToolCallParseFallbackReason::MalformedStructure},
        };
        for (const auto& item : cases) {
            auto prompt = frontend.prepare(tools_input(), {});
            auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, free_choice());
            std::vector<infernix::TokenId> tokens{open};
            for (const auto id : ids(item.body)) { tokens.push_back(id); }
            tokens.push_back(close);
            tokens.push_back(turn_end);
            const Run run = feed(session, tokens, 3);
            check(run.deltas.empty() && session.take_tool_calls().empty() &&
                      run.content.empty() &&
                      session.tool_call_parse_diagnostics().fallback_reason == item.reason,
                  std::string("quarantined invalid call: ") + item.label);
        }
        // A turn cut inside a call never completes it.
        auto prompt = frontend.prepare(tools_input(), {});
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, free_choice());
        std::vector<infernix::TokenId> tokens{open};
        for (const auto id : ids("call:get_ti")) { tokens.push_back(id); }
        const Run run = feed(session, tokens, 2);
        check(run.finish == infernix::FinishReason::OutputLimit && run.deltas.empty() &&
                  session.take_tool_calls().empty() && run.content.empty() &&
                  session.tool_call_parse_diagnostics().fallback_reason ==
                      infernix::ToolCallParseFallbackReason::MalformedStructure,
              "a cut call stays out of calls and text");

        // tool_choice none: nothing is callable.
        auto none_prompt = frontend.prepare(tools_input(), {});
        infernix::ToolChoice none;
        none.mode = infernix::ToolChoiceMode::None;
        auto none_session = frontend.make_output_session(none_prompt, {}, {}, {}, std::nullopt, none);
        (void)feed(none_session, with(ids("call:get_time{}"), {close, turn_end}), 2);
        check(none_session.take_tool_calls().empty(), "tool_choice none makes no call");
    }
    for (const std::size_t step : {std::size_t{1}, std::size_t{8}}) {
        auto prompt = frontend.prepare(tools_input(), {});
        infernix::StopPolicy stop;
        stop.strings.push_back({.text = "lo wor", .channel = infernix::OutputChannel::Content,
                                .include_in_output = false});
        auto session = frontend.make_output_session(prompt, stop, {}, {}, std::nullopt, free_choice());
        const auto tokens = ids("Hello world, again");
        const Run run = feed(session, tokens, step);
        check(tokens.size() >= 3 && run.finish == infernix::FinishReason::StopString && run.content == "Hel" &&
                  session.matched_stop_string() == std::optional<std::string>("lo wor"),
              "a stop string split across tokens cuts the content (step " + std::to_string(step) + ", got \"" +
                  run.content + "\")");
    }
}

// generation_config.json's eos_token_id array reaches the stop policy and ends a streamed turn.
void multi_eos_checks(const std::string& tokenizer_json, const std::string& embedded_template,
    const fs::path& template_path) {
    const auto build = [&](const std::string& generation) {
        return make_frontend(tokenizer_json, embedded_template, generation, 4096, template_path);
    };
    const auto frontend = build(R"({"eos_token_id":[1,106,50]})");
    check(frontend.default_stop_policy().token_ids == std::vector<infernix::TokenId>({1, 106, 50}),
          "an eos array keeps its ids in order");
    check(build(R"({"eos_token_id":[1,106,106,50,1]})").default_stop_policy().token_ids ==
              std::vector<infernix::TokenId>({1, 106, 50}),
          "an eos array drops repeats");
    check(build(R"({"eos_token_id":106})").default_stop_policy().token_ids ==
              std::vector<infernix::TokenId>({106}),
          "a scalar eos id is one stop token");
    bool threw = false;
    try { (void)build(R"({"eos_token_id":[]})"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an empty eos array is refused");
    threw = false;
    try { (void)build(R"({"eos_token_id":[1,999999999]})"); } catch (const std::invalid_argument&) { threw = true; }
    check(threw, "an eos id outside the vocabulary is refused");

    infernix::PromptInput input;
    input.messages.push_back(message(infernix::ChatRole::User, "hi"));
    const auto prompt = frontend.prepare(std::move(input), {});
    for (const infernix::TokenId stop : {infernix::TokenId{1}, infernix::TokenId{106}, infernix::TokenId{50}}) {
        auto session = frontend.make_output_session(prompt, {}, {}, {}, std::nullopt, {});
        auto tokens = frontend.tokenize_text("Hello");
        const std::size_t before = tokens.size();
        tokens.push_back(stop);
        for (const auto id : frontend.tokenize_text("ignored")) { tokens.push_back(id); }
        const Run run = feed(session, tokens, tokens.size());
        check(run.finish == infernix::FinishReason::StopToken && run.accepted == before + 1 &&
                  run.content == "Hello",
              "a streamed turn ends on eos id " + std::to_string(stop) + " (got \"" + run.content + "\")");
    }
    // include_model_defaults off leaves the stop to the caller.
    {
        auto session = frontend.make_output_session(prompt, {.token_ids = {106}, .include_model_defaults = false},
            {}, {}, std::nullopt, {});
        auto tokens = frontend.tokenize_text("Hi");
        tokens.push_back(50);
        for (const auto id : frontend.tokenize_text("there")) { tokens.push_back(id); }
        tokens.push_back(106);
        const Run run = feed(session, tokens, 3);
        check(run.finish == infernix::FinishReason::StopToken && run.accepted == tokens.size(),
              "without model defaults only the caller's stop token ends the turn");
    }
}

void history_checks(const g4::Frontend& frontend) {
    const auto with_arguments = [](std::string arguments) {
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "go"));
        auto assistant = message(infernix::ChatRole::Assistant, "");
        assistant.tool_calls.push_back(call("c1", "get_time", std::move(arguments)));
        input.messages.push_back(std::move(assistant));
        input.messages.push_back(message(infernix::ChatRole::Tool, "12:00"));
        input.messages.back().tool_call_id = "c1";
        input.options.tool_jsons.push_back(declaration("get_time"));
        return input;
    };
    check(!throws_invalid(frontend, with_arguments("{\"zone\":\"UTC\"}")), "object arguments render");
    check(!throws_invalid(frontend, with_arguments("")), "empty arguments render as none");
    check(throws_invalid(frontend, with_arguments("[1,2]")), "array arguments are invalid_prompt input");
    check(throws_invalid(frontend, with_arguments("\"{\\\"zone\\\":1}\"")), "string arguments are invalid_prompt input");
    check(throws_invalid(frontend, with_arguments("7")), "scalar arguments are invalid_prompt input");
    check(throws_invalid(frontend, with_arguments("{\"zone\":")), "arguments that are not JSON are invalid_prompt input");
    check(throws_invalid(frontend, with_arguments(std::string(300, '[') + std::string(300, ']'))),
          "arguments nested past the bound are invalid_prompt input");

    // The extras no Gemma template reads stay refused.
    auto deferred = with_arguments("{}");
    deferred.options.deferred_tool_jsons.push_back(declaration("later"));
    check(throws_invalid(frontend, deferred), "deferred tools are refused");
}

// Thinking precedence, tool/system blocks, prior reasoning, parallel tool results and continued turns, as
// properties of the rendered prompt.
void precedence_checks(const g4::Frontend& frontend) {
    const auto base = [] {
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "hi"));
        return input;
    };
    const auto thinking = [&](infernix::PromptInput input) { return contains(frontend.render(input), "<|think|>"); };

    {   // enable_thinking flag > chat_template_kwargs > reasoning_effort
        auto input = base();
        input.options.enable_thinking = false;
        input.options.chat_template_kwargs_json = R"({"enable_thinking":true})";
        input.options.reasoning_effort = infernix::ReasoningEffort::High;
        check(!thinking(input), "the typed flag off beats kwargs on and a high effort");

        input = base();
        input.options.enable_thinking = true;
        input.options.chat_template_kwargs_json = R"({"enable_thinking":false})";
        check(thinking(input), "the typed flag on beats kwargs off");

        input = base();
        input.options.chat_template_kwargs_json = R"({"enable_thinking":false})";
        input.options.reasoning_effort = infernix::ReasoningEffort::High;
        check(!thinking(input), "kwargs off beats a high effort");

        input = base();
        input.options.chat_template_kwargs_json = R"({"enable_thinking":true})";
        input.options.reasoning_effort = infernix::ReasoningEffort::None;
        check(thinking(input), "kwargs on beats effort none");

        input = base();
        input.options.reasoning_effort = infernix::ReasoningEffort::High;
        check(thinking(input), "a high effort turns thinking on");
        input.options.reasoning_effort = infernix::ReasoningEffort::None;
        check(!thinking(input), "effort none leaves thinking off");
        check(!thinking(base()), "nothing set leaves thinking off");

        input = base();
        input.options.chat_template_kwargs_json = R"({"tools":[]})";
        check(throws_invalid(frontend, input), "kwargs cannot override tools");
    }
    {   // Tools and a system message with thinking disabled.
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::System, "Be brief."));
        input.messages.push_back(message(infernix::ChatRole::User, "hi"));
        input.options.tool_jsons.push_back(declaration("get_time"));
        input.options.enable_thinking = false;
        const std::string off = frontend.render(input);
        check(off.starts_with("<bos><|turn>system\nBe brief.<|tool>") && contains(off, "<tool|><turn|>\n<|turn>user\nhi<turn|>\n") &&
                  !contains(off, "<|think|>") && off.ends_with("<|turn>model\n<|channel>thought\n<channel|>"),
              "tools and a system message with thinking off\n" + off);
        input.options.enable_thinking = true;
        const std::string on = frontend.render(input);
        check(on.starts_with("<bos><|turn>system\n<|think|>\nBe brief.<|tool>") && on.ends_with("<|turn>model\n<|channel>thought\n"),
              "tools and a system message with thinking on\n" + on);
    }
    {   // Prior reasoning: kept only before the last user turn when preserve_thinking asks and the turn called
        // tools; never with drop_prior_thinking; always for the reply being continued.
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "u1"));
        auto assistant = message(infernix::ChatRole::Assistant, "");
        assistant.reasoning_content = "PRIOR";
        assistant.tool_calls.push_back(call("c1", "get_time", "{}"));
        input.messages.push_back(std::move(assistant));
        auto result = message(infernix::ChatRole::Tool, "12:00");
        result.tool_call_id = "c1";
        input.messages.push_back(std::move(result));
        input.messages.push_back(message(infernix::ChatRole::Assistant, "It is noon."));
        input.messages.push_back(message(infernix::ChatRole::User, "u2"));
        input.options.tool_jsons.push_back(declaration("get_time"));
        input.options.enable_thinking = false;
        check(!contains(frontend.render(input), "PRIOR"), "prior reasoning is dropped by default");
        input.options.preserve_thinking = true;
        check(contains(frontend.render(input), "<|channel>thought\nPRIOR\n<channel|>"),
              "preserve_thinking keeps the prior reasoning of a tool-calling turn");
        input.options.chat_template_kwargs_json = R"({"drop_prior_thinking":true})";
        check(!contains(frontend.render(input), "PRIOR"), "drop_prior_thinking beats preserve_thinking");

        infernix::PromptInput current;
        current.messages.push_back(message(infernix::ChatRole::User, "u"));
        auto reply = message(infernix::ChatRole::Assistant, "x");
        reply.reasoning_content = "CURRENT";
        current.messages.push_back(std::move(reply));
        current.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        const std::string text = frontend.render(current);
        check(contains(text, "<|channel>thought\nCURRENT\n<channel|>") && !text.ends_with("<turn|>\n"),
              "the reply being continued keeps its reasoning and stays open");
    }
    {   // Parallel tool results carry their call's name by id, whatever their order.
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "both"));
        auto assistant = message(infernix::ChatRole::Assistant, "");
        assistant.tool_calls.push_back(call("call_a", "get_weather", "{\"city\":\"Paris\"}"));
        assistant.tool_calls.push_back(call("call_b", "get_time", "{\"zone\":\"UTC\"}"));
        input.messages.push_back(std::move(assistant));
        auto second = message(infernix::ChatRole::Tool, "13:00");
        second.tool_call_id = "call_b";
        auto first = message(infernix::ChatRole::Tool, "sunny");
        first.tool_call_id = "call_a";
        input.messages.push_back(std::move(second));
        input.messages.push_back(std::move(first));
        input.options.tool_jsons.push_back(declaration("get_weather"));
        input.options.tool_jsons.push_back(declaration("get_time"));
        const std::string text = frontend.render(input);
        const auto time_at = text.find("response:get_time{");
        const auto weather_at = text.find("response:get_weather{");
        check(time_at != std::string::npos && weather_at != std::string::npos && time_at < weather_at &&
                  !contains(text, "response:unknown{"),
              "parallel tool results resolve their names by id\n" + text);
    }
    {   // A continued reply stays in its model turn; new turns open a model turn.
        infernix::PromptInput input;
        input.messages.push_back(message(infernix::ChatRole::User, "q"));
        input.messages.push_back(message(infernix::ChatRole::Assistant, "Par"));
        input.options.continuation = infernix::PromptContinuationMode::ContinueFinalAssistant;
        const std::string continued = frontend.render(input);
        check(continued.ends_with("<|turn>model\nPar") && count_of(continued, "<|turn>model") == 1,
              "a continued reply ends open inside its model turn\n" + continued);

        input.options.continuation = infernix::PromptContinuationMode::NewAssistantTurn;
        input.messages.push_back(message(infernix::ChatRole::User, "more"));
        const std::string fresh = frontend.render(input);
        check(fresh.ends_with("<|turn>model\n<|channel>thought\n<channel|>") && count_of(fresh, "<|turn>model") == 2,
              "a new turn opens a model turn\n" + fresh);

        infernix::PromptInput doubled;
        doubled.messages.push_back(message(infernix::ChatRole::User, "q"));
        doubled.messages.push_back(message(infernix::ChatRole::Assistant, "a"));
        doubled.messages.push_back(message(infernix::ChatRole::Assistant, "b"));
        check(contains(frontend.render(doubled), "<|turn>model\nab<turn|>\n"),
              "consecutive assistant messages share one model turn");
    }
}

} // namespace

int main() {
    const char* hf = std::getenv("INFERNIX_GEMMA4_HF");
    if (hf == nullptr || *hf == '\0') {
        std::cout << "SKIP: INFERNIX_GEMMA4_HF is not set\n";
        return 77;
    }
    const char* override_dir = std::getenv("INFERNIX_GEMMA4_FIXTURES");
    const fs::path fixtures = override_dir != nullptr && *override_dir != '\0'
        ? fs::path(override_dir) : fs::path(INFERNIX_GEMMA4_FIXTURE_DIR);
    const fs::path model = hf;
    const fs::path fixed_template = fs::path(INFERNIX_SOURCE_DIR) / "tools" / "chat_templates" / "gemma4_fixed.jinja";
    try {
        const std::string tokenizer_json = read_file(model / "tokenizer.json");
        const std::string embedded_template = read_file(model / "chat_template.jinja");
        const std::string generation = fs::exists(model / "generation_config.json")
            ? read_file(model / "generation_config.json") : std::string(R"({"eos_token_id":[1,106,50]})");
        const g4::Frontend frontend = make_frontend(tokenizer_json, embedded_template, generation, 262144, fixed_template);
        const g4::Frontend reference = make_frontend(tokenizer_json, embedded_template, generation, 262144, {});

        const g4::Tokenizer tokenizer(tokenizer_json);
        const Json strings = Json::parse(read_file(fixtures / "tok_fixtures.json"));
        int token_failures = 0;
        for (std::size_t index = 0; index < strings.size(); ++index) {
            const Json& record = strings[index];
            const auto text = record.at("text").get<std::string>();
            const auto want = record.at("ids").get<std::vector<int>>();
            const bool ids_ok = tokenizer.encode(text) == want;
            const bool text_ok = tokenizer.decode(want, false) == record.at("decoded").get<std::string>();
            if (!ids_ok || !text_ok) {
                if (++token_failures <= 5) {
                    std::cerr << "FAIL string " << index << (ids_ok ? "" : " ids") << (text_ok ? "" : " decode")
                              << ": " << Json(text).dump() << '\n';
                }
            }
        }
        check(token_failures == 0, "tokenizer fixtures");
        std::cout << "tokenizer: " << strings.size() - static_cast<std::size_t>(token_failures) << '/'
                  << strings.size() << " strings match\n";

        const Json chats = Json::parse(read_file(fixtures / "chat_fixtures20.json"));
        int chat_failures = 0;
        for (const Json& item : chats) {
            const std::string name = item.at("name").get<std::string>();
            const std::string want = item.at("text").get<std::string>();
            try {
                const std::string got = reference.render(prompt_of(item));
                if (got != want) {
                    ++chat_failures;
                    std::cerr << "FAIL " << name << " text " << first_difference(got, want) << '\n';
                    continue;
                }
                const auto prepared = reference.prepare(prompt_of(item), {});
                if (prepared.tokens() != item.at("ids").get<std::vector<infernix::TokenId>>()) {
                    ++chat_failures;
                    std::cerr << "FAIL " << name << " ids\n";
                }
            } catch (const std::exception& error) {
                ++chat_failures;
                std::cerr << "FAIL " << name << " threw: " << error.what() << '\n';
            }
        }
        check(chat_failures == 0, "chat fixtures");
        std::cout << "chat template: " << chats.size() - static_cast<std::size_t>(chat_failures) << '/'
                  << chats.size() << " conversations match\n";

        tool_call_checks(frontend);
        multi_eos_checks(tokenizer_json, embedded_template, fixed_template);
        history_checks(frontend);
        precedence_checks(frontend);

        // A prompt past max_context is a request error, which serve answers with HTTP 400.
        const auto tight = make_frontend(tokenizer_json, embedded_template, generation, 16, fixed_template);
        std::string words;
        for (int index = 0; index < 40; ++index) { words += "word "; }
        infernix::PromptInput long_input;
        long_input.messages.push_back(message(infernix::ChatRole::User, words));
        bool length_ok = false;
        try { (void)tight.prepare(std::move(long_input), {}); }
        catch (const infernix::RequestError& error) {
            length_ok = error.kind() == infernix::RequestErrorKind::ContextLengthExceeded;
        }
        check(length_ok, "a prompt past max_context is a ContextLengthExceeded request error");

        // Gemma 4's recommended sampling: temperature 1.0, top_p 0.95, top_k 64, thinking or not, over the
        // sampler's whole 64-candidate range.
        const infernix::ModelSamplingDefaults defaults = frontend.sampling_defaults();
        const auto same_preset = [](const infernix::SamplingPreset& preset) {
            return preset.temperature == 1.0F && preset.top_k == 64 && preset.top_p == 0.95F && preset.min_p == 0.0F &&
                   preset.presence_penalty == 0.0F && preset.frequency_penalty == 0.0F;
        };
        check(same_preset(defaults.thinking) && same_preset(defaults.non_thinking) && defaults.max_top_k == 64,
              "sampling defaults are top_k 64 for thinking and non-thinking");

        std::cout << (failures == 0 ? "OK" : "FAIL") << " gemma4 frontend fixtures\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "EXC " << error.what() << '\n';
        return 1;
    }
}
