#include "models/gemma4/frontend/frontend.h"

#include "models/gemma4/frontend/tool_grammar.h"
#include "text/grammar.h"
#include "text/jinja.h"
#include "text/json_input.h"
#include "text/unicode.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace infernix::models::gemma4 {
namespace {
using Json = nlohmann::ordered_json;
constexpr std::string_view kReasoningStart = "<|channel>thought\n";
constexpr std::string_view kCallPrefix = "call:";
constexpr std::string_view kTurnEnd = "<turn|>\n";
// Gemma 4 serves the whole 64-candidate sampler range; Qwen stays 20.
constexpr std::int32_t kMaxTopK = 64;

void checkpoint(const PreparationControl& control) {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled, "gemma4 prompt preparation cancelled");
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout, "gemma4 prompt preparation expired");
    }
}

Json parse_object(std::string_view source, const char* name) {
    if (text::json_text_nesting(source) > text::kMaxJsonNesting) { throw text::JsonNestingError(); }
    Json value = Json::parse(source);
    if (!value.is_object()) { throw std::invalid_argument(std::string(name) + " must be an object"); }
    return value;
}

TokenId single_token(const Tokenizer& tokenizer, std::string_view spelling) {
    const auto tokens = tokenizer.encode(spelling);
    if (tokens.size() != 1 || tokenizer.token_bytes(tokens[0]) != spelling) {
        throw std::invalid_argument("gemma4 tokenizer is missing control token " + std::string(spelling));
    }
    return tokens[0];
}

// A history tool call's arguments as the template reads them: an object (empty arguments are
// none). A client's string that is not one fails the request as invalid input (HTTP 400
// invalid_prompt) rather than escaping as a parse or template error. The nesting is bounded before
// the parse, which and the template's walk recurse once per level.
Json history_arguments(const ToolCall& call) {
    if (call.arguments_json.empty()) { return Json::object(); }
    const std::string subject = "the arguments of history tool call '" + call.name + "'";
    if (text::json_text_nesting(call.arguments_json) > text::kMaxJsonNesting) {
        throw std::invalid_argument(subject + " nest arrays or objects deeper than " +
            std::to_string(text::kMaxJsonNesting) + " levels");
    }
    Json arguments;
    try { arguments = Json::parse(call.arguments_json); }
    catch (const Json::parse_error& error) {
        throw std::invalid_argument(subject + " are not valid JSON (byte " + std::to_string(error.byte) + ")");
    }
    if (!arguments.is_object()) { throw std::invalid_argument(subject + " are not a JSON object"); }
    return arguments;
}

Json history_tool_call(const ToolCall& call) {
    return Json{{"id", call.id}, {"type", "function"},
        {"function", {{"name", call.name}, {"arguments", history_arguments(call)}}}};
}

// A declared tool in the OpenAI shape the template reads: {"type":"function","function":{...}}.
Json tool_declaration(const std::string& source) {
    Json tool = parse_object(source, "tool");
    if (tool.contains("function")) { return tool; }
    return Json{{"type", "function"}, {"function", std::move(tool)}};
}

// The function name of each declared tool, in declaration order.
std::vector<std::string> tool_names(const std::vector<std::string>& tool_jsons) {
    std::vector<std::string> names;
    for (const auto& source : tool_jsons) {
        const Json tool = tool_declaration(source);
        const Json& function = tool.at("function");
        if (function.is_object() && function.contains("name") && function.at("name").is_string()) {
            names.push_back(function.at("name").get<std::string>());
        }
    }
    return names;
}

const char* role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::Developer: return "developer";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: return "tool";
    }
    throw std::invalid_argument("gemma4 chat role is unknown");
}

Json template_context(const PromptInput& input, ChatTemplateRoleSupport support) {
    const auto& options = input.options;
    if (!options.deferred_tool_jsons.empty() || options.open_tool_names ||
        options.declared_parameter_names) {
        throw std::invalid_argument(
            "gemma4 does not support deferred tools, open tool names or declared parameter names");
    }
    if (options.add_vision_id) { throw std::invalid_argument("gemma4 P4 is text-only"); }
    if (options.continuation == PromptContinuationMode::NextTurnHistory) {
        throw std::invalid_argument("gemma4 P4 does not support next-turn cache preparation");
    }
    if (input.messages.empty()) { throw std::invalid_argument("gemma4 prompt has no messages"); }
    if (options.continuation == PromptContinuationMode::ContinueFinalAssistant &&
        input.messages.back().role != ChatRole::Assistant) {
        throw std::invalid_argument("continuing a reply needs a final assistant message");
    }
    Json context = options.chat_template_kwargs_json.empty() ? Json::object()
        : parse_object(options.chat_template_kwargs_json, "chat_template_kwargs");
    for (const char* reserved : {"messages", "tools", "add_generation_prompt", "bos_token"}) {
        if (context.contains(reserved)) {
            throw std::invalid_argument(std::string("chat_template_kwargs cannot override ") + reserved);
        }
    }
    // Typed enable_thinking wins over kwargs; kwargs wins over reasoning_effort. An absent
    // setting stays absent, preserving the embedded/override template's own default.
    if (options.enable_thinking) { context["enable_thinking"] = *options.enable_thinking; }
    else if (options.reasoning_effort && !context.contains("enable_thinking")) {
        context["enable_thinking"] = *options.reasoning_effort != ReasoningEffort::None;
    }
    if (options.preserve_thinking) { context["preserve_thinking"] = *options.preserve_thinking; }
    Json messages = Json::array();
    for (std::size_t i = 0; i < input.messages.size(); ++i) {
        const auto& message = input.messages[i];
        ChatRole role = message.role;
        if (role == ChatRole::Developer) {
            if (i == 0 && !support.leading_developer) {
                role = support.leading_system ? ChatRole::System : ChatRole::User;
            } else if (i != 0 && !support.late_developer) { role = ChatRole::User; }
        } else if (i != 0 && role == ChatRole::System && support.lowers_late_system()) {
            role = ChatRole::User;
        }
        std::string content;
        for (const auto& part : message.parts) {
            if (part.kind != MessagePartKind::Text) {
                throw std::invalid_argument("gemma4 P4 does not support image, video or audio input");
            }
            content += part.text;
        }
        Json item{{"role", role_name(role)}, {"content", std::move(content)}};
        if (!message.reasoning_content.empty()) { item["reasoning_content"] = message.reasoning_content; }
        if (!message.tool_calls.empty()) {
            Json calls = Json::array();
            for (const auto& call : message.tool_calls) { calls.push_back(history_tool_call(call)); }
            item["tool_calls"] = std::move(calls);
        }
        if (!message.tool_call_id.empty()) { item["tool_call_id"] = message.tool_call_id; }
        messages.push_back(std::move(item));
    }
    context["messages"] = std::move(messages);
    context["bos_token"] = "<bos>";
    context["add_generation_prompt"] = options.continuation == PromptContinuationMode::NewAssistantTurn;
    if (!options.tool_jsons.empty()) {
        Json tools = Json::array();
        for (const auto& source : options.tool_jsons) { tools.push_back(tool_declaration(source)); }
        context["tools"] = std::move(tools);
    }
    return context;
}

ChatTemplateRoleSupport probe_roles(const text::JinjaTemplate& chat) {
    const auto renders = [&](std::initializer_list<const char*> roles) {
        Json context{{"bos_token", "<bos>"}, {"add_generation_prompt", true}};
        context["messages"] = Json::array();
        for (const char* role : roles) {
            context["messages"].push_back(Json{{"role", role}, {"content", "probe"}});
        }
        try { (void)chat.render(context); return true; }
        catch (const std::exception&) { return false; }
    };
    ChatTemplateRoleSupport support;
    if (!renders({"user"})) { return support; }
    support.leading_system = renders({"system", "user"});
    support.late_system = renders({"user", "assistant", "system"});
    support.leading_developer = renders({"developer", "user"});
    support.late_developer = renders({"user", "assistant", "developer"});
    return support;
}

std::string template_source(const TextResources& resources, const FrontendOptions& options) {
    if (options.chat_template_path.empty()) { return std::string(resources.chat_template_jinja); }
    std::ifstream file(options.chat_template_path, std::ios::binary);
    if (!file) { throw std::invalid_argument("cannot open gemma4 chat template override"); }
    std::string source{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    if (file.bad()) { throw std::runtime_error("cannot read gemma4 chat template override"); }
    return source;
}

SamplingPreset sampling_preset(const Json& values, SamplingPreset preset) {
    for (const char* key : {"temperature", "top_p", "min_p", "presence_penalty", "frequency_penalty"}) {
        if (!values.contains(key)) { continue; }
        const auto value = values.at(key).get<float>();
        if (!std::isfinite(value)) { throw std::invalid_argument("gemma4 sampling value is not finite"); }
        if (std::string_view(key) == "temperature") { preset.temperature = value; }
        else if (std::string_view(key) == "top_p") { preset.top_p = value; }
        else if (std::string_view(key) == "min_p") { preset.min_p = value; }
        else if (std::string_view(key) == "presence_penalty") { preset.presence_penalty = value; }
        else { preset.frequency_penalty = value; }
    }
    if (values.contains("top_k")) {
        const auto value = values.at("top_k").get<std::int32_t>();
        if (value < 0) { throw std::invalid_argument("gemma4 sampling top_k is negative"); }
        // The checkpoint's profile is top 64, the model's whole candidate range.
        preset.top_k = std::min<std::int32_t>(value, kMaxTopK);
    }
    if (preset.temperature < 0 || preset.top_p <= 0 || preset.top_p > 1 ||
        preset.min_p < 0 || preset.min_p > 1 || std::abs(preset.presence_penalty) > 2 ||
        std::abs(preset.frequency_penalty) > 2) {
        throw std::invalid_argument("gemma4 sampling preset is outside supported bounds");
    }
    return preset;
}
} // namespace

struct Frontend::Impl {
    std::shared_ptr<const Tokenizer> tokenizer;
    text::JinjaTemplate chat;
    std::uint32_t max_context;
    StopPolicy stop;
    ModelSamplingDefaults sampling;
    ChatTemplateRoleSupport roles;
    TokenId channel_open;
    TokenId channel_close;
    TokenId tool_call_open;
    TokenId tool_call_close;
    TokenId turn_open;
    TokenId turn_close;
    std::size_t grammar_cache_bytes;
    mutable std::once_flag grammar_once;
    mutable std::unique_ptr<text::GrammarCompiler> grammar_compiler;

    text::GrammarCompiler& grammars() const {
        std::call_once(grammar_once, [&] {
            std::vector<std::string> vocabulary(tokenizer->vocab_size());
            for (TokenId id = 0; id < tokenizer->vocab_size(); ++id) {
                // xgrammar classifies empty entries as unconditionally forbidden special IDs,
                // even for Token terminals. Keep real bytes for the model-owned framing tokens.
                if (!tokenizer->is_special(id) || id == channel_close || id == tool_call_open ||
                    id == tool_call_close || tokenizer->token_bytes(id) == "<|\"|>") {
                    vocabulary[id] = tokenizer->token_bytes(id);
                }
            }
            grammar_compiler = std::make_unique<text::GrammarCompiler>(
                std::move(vocabulary), stop.token_ids, grammar_cache_bytes);
        });
        return *grammar_compiler;
    }

    Impl(const TextResources& resources, const FrontendOptions& options)
        : tokenizer(std::make_shared<const Tokenizer>(resources.tokenizer_json)),
          chat(template_source(resources, options), "gemma4-chat"), max_context(options.max_context),
          roles(probe_roles(chat)), channel_open(single_token(*tokenizer, "<|channel>")),
          channel_close(single_token(*tokenizer, "<channel|>")),
          tool_call_open(single_token(*tokenizer, "<|tool_call>")),
          tool_call_close(single_token(*tokenizer, "<tool_call|>")),
          turn_open(single_token(*tokenizer, "<|turn>")), turn_close(single_token(*tokenizer, "<turn|>")),
          grammar_cache_bytes(options.grammar_cache_bytes) {
        if (max_context == 0) { throw std::invalid_argument("gemma4 max_context must be positive"); }
        const Json generation = resources.generation_config_json.empty() ? Json::object()
            : parse_object(resources.generation_config_json, "generation_config.json");
        Json eos;
        if (generation.contains("eos_token_id")) { eos = generation.at("eos_token_id"); }
        else if (!resources.tokenizer_config_json.empty()) {
            const Json config = parse_object(resources.tokenizer_config_json, "tokenizer_config.json");
            if (config.contains("eos_token")) {
                const auto& token = config.at("eos_token");
                const std::string spelling = token.is_string() ? token.get<std::string>()
                    : token.at("content").get<std::string>();
                eos = single_token(*tokenizer, spelling);
            }
        }
        if (eos.is_null()) { throw std::invalid_argument("gemma4 resources do not declare EOS"); }
        const auto add_eos = [&](const Json& value) {
            if (!value.is_number_integer()) { throw std::invalid_argument("gemma4 EOS must be an integer"); }
            const auto id = value.get<std::int64_t>();
            if (id < 0 || id >= tokenizer->vocab_size()) { throw std::invalid_argument("gemma4 EOS outside vocabulary"); }
            const auto token = static_cast<TokenId>(id);
            if (std::find(stop.token_ids.begin(), stop.token_ids.end(), token) == stop.token_ids.end()) {
                stop.token_ids.push_back(token);
            }
        };
        if (eos.is_array()) { for (const auto& id : eos) { add_eos(id); } }
        else { add_eos(eos); }
        if (stop.token_ids.empty()) { throw std::invalid_argument("gemma4 EOS array is empty"); }
        // Gemma 4's recommended sampling: temperature 1.0, top_p 0.95, top_k 64, in thinking and not.
        SamplingPreset preset{.temperature = 1.0F, .top_k = kMaxTopK, .top_p = 0.95F};
        preset = sampling_preset(generation, preset);
        sampling.thinking = sampling.non_thinking = preset;
        sampling.max_top_k = kMaxTopK;
        if (!resources.sampling_json.empty()) {
            const Json presets = parse_object(resources.sampling_json, "sampling.json");
            sampling.thinking = sampling_preset(presets.at("thinking"), preset);
            sampling.non_thinking = sampling_preset(presets.at("non_thinking"), preset);
        }
    }

    void check_length(std::size_t size) const {
        if (size > max_context) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                "gemma4 prepared prompt has " + std::to_string(size) +
                " tokens, beyond Engine max_context " + std::to_string(max_context));
        }
    }

    std::string render(const PromptInput& input, const PreparationControl& control) const {
        checkpoint(control);
        const Json context = template_context(input, roles);
        std::string rendered = chat.render(context, {.checkpoint = [&] { checkpoint(control); }}).text;
        if (input.options.continuation == PromptContinuationMode::ContinueFinalAssistant &&
            rendered.ends_with(kTurnEnd)) { rendered.resize(rendered.size() - kTurnEnd.size()); }
        checkpoint(control);
        return rendered;
    }
};

Frontend::Frontend(const TextResources& resources, FrontendOptions options)
    : impl_(std::make_shared<const Impl>(resources, options)) {}
Frontend::~Frontend() = default;

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids, bool) const {
    if (token_ids.empty()) { throw std::invalid_argument("gemma4 prompt is empty"); }
    impl_->check_length(token_ids.size());
    for (const TokenId token : token_ids) {
        if (token < 0 || token >= impl_->tokenizer->vocab_size()) {
            throw std::invalid_argument("gemma4 prompt token outside vocabulary");
        }
    }
    PromptSummary summary;
    summary.prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    return PreparedPrompt(std::move(token_ids), summary);
}

PreparedPrompt Frontend::prepare(PromptInput input, const PreparationControl& control) const {
    const auto rendered = impl_->render(input, control);
    auto tokens = impl_->tokenizer->encode(rendered, &control);
    checkpoint(control);
    auto prompt = prepare_tokens(std::move(tokens));
    prompt.summary_.starts_in_reasoning = rendered.ends_with(kReasoningStart);
    if (input.options.continuation == PromptContinuationMode::ContinueFinalAssistant) {
        for (const auto& part : input.messages.back().parts) { prompt.continuation_content_ += part.text; }
    }
    prompt.tool_jsons_ = input.options.tool_jsons;
    if (!input.options.tool_jsons.empty()) {
        prompt.tool_call_open_ = impl_->tool_call_open;
        prompt.channel_close_ = impl_->channel_close;
        prompt.tool_openers_.push_back({"", impl_->tokenizer->encode(kCallPrefix)});
        for (auto& name : tool_names(input.options.tool_jsons)) {
            auto opener = impl_->tokenizer->encode(std::string(kCallPrefix) + name);
            prompt.tool_openers_.push_back({std::move(name), std::move(opener)});
        }
    }
    return prompt;
}
std::uint32_t Frontend::count_tokens(PromptInput input, const PreparationControl& control) const {
    return prepare(std::move(input), control).summary().prompt_tokens;
}
std::vector<TokenId> Frontend::tokenize_text(std::string_view text) const {
    return impl_->tokenizer->encode(text);
}
std::string Frontend::detokenize_bytes(std::span<const TokenId> token_ids) const {
    return impl_->tokenizer->decode(token_ids, false);
}
std::string Frontend::detokenize(std::span<const TokenId> token_ids) const {
    auto bytes = detokenize_bytes(token_ids);
    for (std::size_t at = 0; at < bytes.size();) {
        at += text::unicode_internal::utf8_codepoint_at(bytes, at, "gemma4 detokenize").length;
    }
    return bytes;
}
std::string Frontend::render(const PromptInput& input) const { return impl_->render(input, {}); }

ChatTokens Frontend::tokenize_chat(PromptInput input) const {
    ChatTokens result;
    result.ids = tokenize_text(render(input));
    result.generated.assign(result.ids.size(), 0);
    // Gemma turns are <|turn>ROLE\nBODY<turn|>. Only a completed model turn's body and
    // closing token are generated; the appended next-assistant opener is prompt text.
    for (std::size_t i = 0; i < result.ids.size();) {
        if (result.ids[i++] != impl_->turn_open) { continue; }
        std::string header;
        std::size_t body = i;
        for (; body < result.ids.size(); ++body) {
            header += impl_->tokenizer->token_bytes(result.ids[body]);
            if (header.find('\n') != std::string::npos) { ++body; break; }
            if (header.size() > 64) { break; }
        }
        std::size_t end = body;
        while (end < result.ids.size() && result.ids[end] != impl_->turn_close &&
               result.ids[end] != impl_->turn_open) { ++end; }
        if (header == "model\n" && end < result.ids.size() && result.ids[end] == impl_->turn_close) {
            std::fill(result.generated.begin() + body, result.generated.begin() + end + 1, 1);
        }
        i = end;
    }
    return result;
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt, const StopPolicy& caller,
    const OutputOptions& output, const ThinkingControlOptions& thinking,
    const std::optional<OutputConstraint>& constraint, const ToolChoice& choice) const {
    if (!prompt) { throw std::invalid_argument("gemma4 output needs a prepared prompt"); }
    if (choice.mode == ToolChoiceMode::Required && !prompt.forced_call_) {
        throw std::invalid_argument("gemma4 required tool choice was not applied to the prompt");
    }
    if (thinking.budget && *thinking.budget == 0) {
        throw std::invalid_argument("gemma4 thinking budget must be positive");
    }
    if (thinking.loop != ReasoningLoopAction::Off) {
        throw std::invalid_argument("gemma4 does not support the reasoning-loop guard");
    }
    if (output.tolerant_tool_calls) {
        throw std::invalid_argument("gemma4 does not support tolerant tool calls");
    }
    OutputSession::Setup setup;
    setup.stop_ids = caller.token_ids;
    if (caller.include_model_defaults) {
        for (const auto token : impl_->stop.token_ids) {
            if (std::find(setup.stop_ids.begin(), setup.stop_ids.end(), token) == setup.stop_ids.end()) {
                setup.stop_ids.push_back(token);
            }
        }
    }
    for (const auto token : setup.stop_ids) {
        if (token < 0 || token >= impl_->tokenizer->vocab_size()) {
            throw std::invalid_argument("gemma4 stop token outside vocabulary");
        }
    }
    setup.stop_strings = caller.strings;
    setup.publish_stop = caller.publish_stop_token;
    setup.starts_in_reasoning = prompt.summary().starts_in_reasoning;
    setup.thinking_budget = thinking.budget;
    setup.output = output;
    setup.channel_open = impl_->channel_open;
    setup.channel_close = impl_->channel_close;
    setup.tool_call_open = impl_->tool_call_open;
    setup.tool_call_close = impl_->tool_call_close;
    setup.forced_call = prompt.forced_call_;
    if (choice.mode != ToolChoiceMode::None) {
        for (const auto& opener : prompt.tool_openers_) {
            if (!opener.name.empty()) { setup.callable_tools.push_back(opener.name); }
        }
        setup.allowed_names = choice.allowed_names;
    }
    std::vector<GrammarTool> tools;
    bool strict = false;
    if (choice.allowed_names) {
        for (const auto& name : *choice.allowed_names) {
            if (std::find(setup.callable_tools.begin(), setup.callable_tools.end(), name) ==
                setup.callable_tools.end()) {
                throw RequestError(RequestErrorKind::InvalidToolConstraint,
                    "Gemma tool choice names an undeclared tool: " + name);
            }
        }
    }
    if (choice.mode != ToolChoiceMode::None) {
        for (const auto& source : prompt.tool_jsons_) {
            const auto function = tool_declaration(source).at("function");
            const auto name = function.at("name").get<std::string>();
            if (choice.allowed_names && std::find(choice.allowed_names->begin(),
                choice.allowed_names->end(), name) == choice.allowed_names->end()) { continue; }
            if (name.empty() || name.size() > output.tool_name_max_length ||
                !std::all_of(name.begin(), name.end(), [](unsigned char c) {
                    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
                })) {
                throw RequestError(RequestErrorKind::InvalidToolConstraint, "invalid Gemma tool name");
            }
            if (std::any_of(tools.begin(), tools.end(),
                [&](const GrammarTool& tool) { return tool.name == name; })) {
                throw RequestError(RequestErrorKind::InvalidToolConstraint, "duplicate Gemma tool name: " + name);
            }
            const bool tool_strict = function.value("strict", false);
            strict |= tool_strict;
            tools.push_back({name, function.value("parameters", Json{{"type", "object"}}).dump(), tool_strict});
        }
    }
    const bool tool_constraint = !tools.empty() && (constraint || strict ||
        choice.constraints == ToolConstraintMode::Basic || choice.mode == ToolChoiceMode::Required ||
        choice.allowed_names || !choice.parallel);
    if (choice.mode == ToolChoiceMode::Required && tools.empty()) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint, "required Gemma tool choice has no tools");
    }
    if (constraint || tool_constraint) {
        if (!caller.token_ids.empty() || !caller.strings.empty() || !caller.include_model_defaults ||
            caller.publish_stop_token || output.raw || output.preserve_special_tokens) {
            throw RequestError(tool_constraint ? RequestErrorKind::InvalidToolConstraint
                : text::constraint_error_kind(constraint->kind),
                "constraints require default EOS, text output and no custom stops");
        }
        const text::ReasoningFraming framing{
            .close_token = setup.starts_in_reasoning ? std::optional<TokenId>(impl_->channel_close) : std::nullopt,
            .separator = "", .withheld_token = impl_->tool_call_open};
        if (tool_constraint) {
            std::vector<TokenId> continuation;
            if (prompt.forced_call_) {
                continuation.push_back(impl_->tool_call_open);
                const auto prefix = impl_->tokenizer->encode(*prompt.forced_call_);
                continuation.insert(continuation.end(), prefix.begin(), prefix.end());
            } else if (!prompt.continuation_content_.empty()) {
                continuation = impl_->tokenizer->encode(prompt.continuation_content_);
            }
            setup.grammar = compile_tool_grammar(impl_->grammars(), tools, choice,
                impl_->tool_call_open, impl_->tool_call_close, framing, continuation, constraint);
        } else {
            setup.grammar = impl_->grammars().compile(*constraint, framing,
                prompt.continuation_content_, text::GrammarCompiler::Compile::Deferred);
        }
        setup.stream_tools = tool_constraint;
        setup.combined_json = tool_constraint && constraint && choice.mode != ToolChoiceMode::Required;
    }
    return OutputSession(impl_->tokenizer, std::move(setup));
}
// Required tool choice opens the call in the prompt: `<|tool_call>call:` (plus the name when one
// tool is allowed), closing an open thought first. The model continues inside the call.
void Frontend::apply_tool_choice(PreparedPrompt& prompt, const ToolChoice& choice) {
    if (choice.mode != ToolChoiceMode::Required || prompt.forced_call_) { return; }
    if (prompt.tool_openers_.empty()) {
        throw std::invalid_argument("gemma4 required tool choice needs declared tools");
    }
    std::string name;
    if (choice.allowed_names && choice.allowed_names->size() == 1) { name = choice.allowed_names->front(); }
    const auto opener = std::find_if(prompt.tool_openers_.begin(), prompt.tool_openers_.end(),
        [&](const PreparedPrompt::ToolOpener& item) { return item.name == name; });
    if (opener == prompt.tool_openers_.end()) {
        throw std::invalid_argument("gemma4 tool choice names an undeclared tool: " + name);
    }
    if (prompt.summary_.starts_in_reasoning) {
        prompt.tokens_.push_back(prompt.channel_close_);
        prompt.summary_.starts_in_reasoning = false;
    }
    prompt.tokens_.push_back(prompt.tool_call_open_);
    prompt.tokens_.insert(prompt.tokens_.end(), opener->tokens.begin(), opener->tokens.end());
    prompt.summary_.prompt_tokens = static_cast<std::uint32_t>(prompt.tokens_.size());
    prompt.forced_call_ = std::string(kCallPrefix) + name;
}
void Frontend::warm_constraints() const { (void)impl_->grammars(); }
const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->stop; }
const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept { return impl_->sampling; }
ChatTemplateRoleSupport Frontend::chat_template_role_support() const noexcept { return impl_->roles; }

} // namespace infernix::models::gemma4
