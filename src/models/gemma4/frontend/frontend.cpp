#include "models/gemma4/frontend/frontend.h"

#include "text/jinja.h"
#include "text/json_input.h"
#include "text/unicode.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <utility>

namespace infernix::models::gemma4 {
namespace {
using Json = nlohmann::ordered_json;
constexpr std::string_view kTurnEnd = "<turn|>\n";
constexpr std::string_view kReasoningStart = "<|channel>thought\n";

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

const char* role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System: return "system";
    case ChatRole::Developer: return "developer";
    case ChatRole::User: return "user";
    case ChatRole::Assistant: return "assistant";
    case ChatRole::Tool: break;
    }
    throw std::invalid_argument("gemma4 P4 does not support tool messages");
}

Json template_context(const PromptInput& input, ChatTemplateRoleSupport support) {
    const auto& options = input.options;
    if (!options.tool_jsons.empty() || !options.deferred_tool_jsons.empty() ||
        options.open_tool_names || options.declared_parameter_names) {
        throw std::invalid_argument("gemma4 P4 does not support tools");
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
        if (!message.tool_calls.empty() || !message.tool_call_id.empty() || message.role == ChatRole::Tool) {
            throw std::invalid_argument("gemma4 P4 does not support tool history");
        }
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
        messages.push_back(std::move(item));
    }
    context["messages"] = std::move(messages);
    context["bos_token"] = "<bos>";
    context["add_generation_prompt"] = options.continuation == PromptContinuationMode::NewAssistantTurn;
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
        // P4 uses the existing shared sampler. The checkpoint's top64 profile waits for P6.
        preset.top_k = std::min<std::int32_t>(value, 20);
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
    TokenId turn_open;
    TokenId turn_close;

    Impl(const TextResources& resources, const FrontendOptions& options)
        : tokenizer(std::make_shared<const Tokenizer>(resources.tokenizer_json)),
          chat(template_source(resources, options), "gemma4-chat"), max_context(options.max_context),
          roles(probe_roles(chat)), channel_open(single_token(*tokenizer, "<|channel>")),
          channel_close(single_token(*tokenizer, "<channel|>")),
          turn_open(single_token(*tokenizer, "<|turn>")), turn_close(single_token(*tokenizer, "<turn|>")) {
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
        SamplingPreset preset{.temperature = 1.0F, .top_k = 20, .top_p = 0.95F};
        preset = sampling_preset(generation, preset);
        sampling.thinking = sampling.non_thinking = preset;
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
    if (constraint) { throw RequestError(RequestErrorKind::InvalidGrammar,
        "gemma4 P4 does not support grammar constraints"); }
    apply_tool_choice(prompt, choice);
    if (thinking.budget && *thinking.budget == 0) {
        throw std::invalid_argument("gemma4 thinking budget must be positive");
    }
    if (thinking.loop != ReasoningLoopAction::Off) {
        throw std::invalid_argument("gemma4 P4 does not support the reasoning-loop guard");
    }
    if (output.tolerant_tool_calls) { throw std::invalid_argument("gemma4 P4 does not support tool parsing"); }
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
    return OutputSession(impl_->tokenizer, std::move(setup));
}
void Frontend::apply_tool_choice(const PreparedPrompt&, const ToolChoice& choice) {
    if (choice.mode == ToolChoiceMode::Required || choice.allowed_names) {
        throw std::invalid_argument("gemma4 P4 does not support tool choice");
    }
}
void Frontend::warm_constraints() const {
    throw std::invalid_argument("gemma4 P4 does not support grammar constraints");
}
const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->stop; }
const ModelSamplingDefaults& Frontend::sampling_defaults() const noexcept { return impl_->sampling; }
ChatTemplateRoleSupport Frontend::chat_template_role_support() const noexcept { return impl_->roles; }

} // namespace infernix::models::gemma4
