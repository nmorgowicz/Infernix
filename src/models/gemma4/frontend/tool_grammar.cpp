#include "models/gemma4/frontend/tool_grammar.h"
#include "text/json_schema.h"

#include "grammar_builder.h"
#include "grammar_functor.h"
#include "json_schema_converter.h"
#include <nlohmann/json.hpp>

namespace infernix::models::gemma4 {
namespace {
using Json = nlohmann::ordered_json;

xgrammar::Grammar build(const std::vector<GrammarTool>& tools, const ToolChoice& choice,
                       TokenId open, TokenId close, const std::optional<OutputConstraint>& body) {
    xgrammar::GrammarBuilder builder;
    std::vector<std::int32_t> alternatives;
    for (const auto& tool : tools) {
        // This is the vendored style="gemma" converter, including nested values and <|"|>
        // strings. compile_model deliberately bypasses public GBNF extension validation:
        // the converter uses TagDispatch to delimit strings.
        const auto source = tool.strict ? text::prepare_json_schema(tool.schema)
            : std::string(R"({"type":"object"})");
        const auto arguments = xgrammar::SubGrammarAdder::Apply(&builder,
            xgrammar::JSONSchemaToGrammar(source, false, std::nullopt,
                std::pair<std::string, std::string>{",", ":"}, tool.strict,
                std::nullopt, false, xgrammar::JSONFormat::kGemma, {"<tool_call|>"}));
        alternatives.push_back(builder.AddSequence({builder.AddByteString("call:" + tool.name),
            builder.AddRuleRef(arguments), builder.AddTokenSet({close})}));
    }
    const auto call = builder.AddRuleWithHint("call", builder.AddChoices(alternatives));
    const auto next = builder.AddSequence({builder.AddTokenSet({open}), builder.AddRuleRef(call)});
    const auto tail = choice.parallel ? builder.AddRepeatFromExpr("calls", next, 0, -1)
                                      : builder.AddEmptyStr();
    const auto calls = builder.AddRuleWithHint("calls",
        builder.AddSequence({builder.AddRuleRef(call), tail}));
    const bool required = choice.mode == ToolChoiceMode::Required;
    const auto root = required || body
        ? builder.AddSequence({builder.AddTokenSet({open}), builder.AddRuleRef(calls)})
        : builder.AddTokenTagDispatch({{{open, calls}}, false, {close}});
    auto grammar = xgrammar::GrammarNormalizer::Apply(
        builder.Get(builder.AddRuleWithHint("root", root)));
    if (body && !required) { grammar = xgrammar::Grammar::Union({text::build_json_grammar(*body), grammar}); }
    return grammar;
}
} // namespace

std::unique_ptr<text::GrammarSession> compile_tool_grammar(
    text::GrammarCompiler& compiler, const std::vector<GrammarTool>& tools,
    const ToolChoice& choice, TokenId open, TokenId close,
    const text::ReasoningFraming& reasoning, std::span<const TokenId> continuation,
    const std::optional<OutputConstraint>& body) {
    Json identity{{"gemma_tools_v1", Json::array()}, {"required", choice.mode == ToolChoiceMode::Required},
        {"parallel", choice.parallel}, {"open", open}, {"close", close}};
    for (const auto& tool : tools) {
        identity["gemma_tools_v1"].push_back({{"name", tool.name}, {"strict", tool.strict},
            {"schema", tool.strict ? tool.schema : ""}});
    }
    if (body) {
        if (body->kind != OutputConstraintKind::JsonObject && body->kind != OutputConstraintKind::JsonSchema) {
            throw RequestError(text::constraint_error_kind(body->kind),
                "active Gemma tools can be combined with JSON output constraints");
        }
        identity["body"] = {{"kind", static_cast<int>(body->kind)}, {"source", body->source}};
    }
    auto framing = reasoning;
    framing.withheld_token = open;
    try {
        return compiler.compile_model(identity.dump(), [&] { return build(tools, choice, open, close, body); },
            framing, continuation, {}, text::GrammarCompiler::Compile::Deferred);
    } catch (const RequestError&) { throw; }
    catch (const xgrammar::JSONSchemaCompileError& error) {
        const auto kind = error.kind == xgrammar::SchemaErrorType::kUnsupportedSchema
            ? RequestErrorKind::UnsupportedJsonSchema
            : error.kind == xgrammar::SchemaErrorType::kUnsatisfiableSchema
                ? RequestErrorKind::UnsatisfiableJsonSchema : RequestErrorKind::InvalidJsonSchema;
        throw RequestError(kind, error.what(), error.pointer, RequestErrorSource::Tools);
    } catch (const xgrammar::LogFatalError& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint, error.what());
    } catch (const std::invalid_argument& error) {
        throw RequestError(RequestErrorKind::InvalidToolConstraint, error.what());
    }
}
} // namespace infernix::models::gemma4
