// Gemma 4 tool-call body grammar: `call:NAME{key:value,...}` with `<|"|>` strings. CPU only, no model.

#include "models/gemma4/frontend/tool_parser.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char* label, const std::string& detail = {}) {
    if (ok) { return; }
    std::cerr << "FAIL " << label << ' ' << detail << '\n';
    ++failures;
}

} // namespace

int main() {
    using infernix::models::gemma4::parse_tool_call_body;
    using infernix::models::gemma4::parse_tool_calls;

    const auto calls = parse_tool_calls(
        "prefix <|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>,days:2}<tool_call|> tail");
    check(calls.size() == 1 && calls[0].name == "get_weather" &&
              calls[0].arguments_json == "{\"city\":\"Paris\",\"days\":2}",
          "flat", calls.empty() ? "" : calls[0].arguments_json);

    const auto empty = parse_tool_calls("<|tool_call>call:ping{}<tool_call|>");
    check(empty.size() == 1 && empty[0].name == "ping" && empty[0].arguments_json == "{}", "empty");

    const auto nested = parse_tool_call_body(
        "call:edit{path:<|\"|>a \"b\"\n{c}<|\"|>,opts:{deep:true,list:[1,-2.5,<|\"|>x<|\"|>,null]}}");
    check(nested && nested->name == "edit" &&
              nested->arguments_json ==
                  "{\"path\":\"a \\\"b\\\"\\n{c}\",\"opts\":{\"deep\":true,\"list\":[1,-2.5,\"x\",null]}}",
          "nested", nested ? nested->arguments_json : "nullopt");

    const auto two = parse_tool_calls("<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{}"
                                      "<tool_call|>");
    check(two.size() == 2 && two[0].name == "a" && two[1].name == "b", "two calls");

    check(!parse_tool_call_body("call:bad{x:<|\"|>open}"), "unterminated string");
    check(!parse_tool_call_body("call:bad{x:1} trailing"), "trailing text");
    check(!parse_tool_call_body("call:dup{x:1,x:2}"), "duplicate key");
    check(!parse_tool_call_body("nocall{}"), "missing call prefix");
    check(!parse_tool_call_body("call:{x:1}"), "empty name");
    check(!parse_tool_call_body("call:bad name{x:1}"), "name with a space");

    // A model's own output nests at most to the JSON bound; deeper is malformed, not a stack overflow.
    std::string deep = "call:deep{x:";
    for (int level = 0; level < 300; ++level) { deep += '['; }
    for (int level = 0; level < 300; ++level) { deep += ']'; }
    deep += '}';
    check(!parse_tool_call_body(deep), "nesting past the bound");

    // Each byte boundary, including nested values, escaped JSON and the Gemma delimiter.
    const std::string wire =
        "call:edit{path:<|\"|>a \"b\"\n{c}<|\"|>,opts:{deep:true,list:[1,-2.5,<|\"|>x<|\"|>,null]}}";
    infernix::models::gemma4::ToolCallStream stream;
    std::string previous;
    for (const char byte : wire) {
        stream.feed(std::string_view(&byte, 1));
        check(stream.arguments().starts_with(previous), "argument prefix never rewrites bytes");
        previous = stream.arguments();
    }
    check(nested && stream.name() == nested->name && stream.arguments() == nested->arguments_json,
          "incremental canonical JSON matches complete parser");
    bool threw = false;
    try { (void)parse_tool_calls("<|tool_call>call:a{x:1}"); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "missing close throws");
    threw = false;
    try { (void)parse_tool_calls("<|tool_call>call:a{x:<tool_call|>"); }
    catch (const std::invalid_argument&) { threw = true; }
    check(threw, "malformed block throws");

    std::cout << (failures ? "FAIL" : "OK") << " gemma4 tool parser\n";
    return failures ? 1 : 0;
}
