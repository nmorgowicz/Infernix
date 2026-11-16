// CPU-only behavior of Gemma 4's thinking control against the complete real vocabulary: the thinking
// budget (the round license, the control span committed at the cap, the answer after it), the budget
// message, the output-capacity check, the reasoning-loop guard, and that a request whose budget is not
// reached is unchanged. The sessions are driven the way the Engine drives them: a round of at most the
// licensed width, preview, commit, and at the cap the control span before the model samples again.
// INFERNIX_GEMMA4_HF supplies tokenizer.json and chat_template.jinja; no downloads or GPU.
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace infernix;
namespace g4 = models::gemma4;
using Tokens = std::vector<TokenId>;
constexpr TokenId kEos = 106;       // <turn|>
constexpr std::uint32_t kPlenty = 1'000'000;
constexpr std::string_view kDefaultControl =
    "\n\nConsidering the limited thinking time, I have to give my answer now, based on my thinking so far.\n<channel|>";

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

std::string file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) { throw std::runtime_error("cannot read " + path.string()); }
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

PromptInput input(bool thinking, bool tools = false) {
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
            R"({"type":"function","function":{"name":"clock","strict":true,"parameters":{"type":"object","properties":{"zone":{"type":"string"}},"required":["zone"],"additionalProperties":false}}})"};
    }
    return result;
}

// What a client sees of a reply.
struct Stream {
    std::string content;
    std::string reasoning;
    std::vector<ToolCallDelta> calls;
    void collect(const g4::PublishedOutput& published) {
        const auto text = [&](const OutputDelta& delta) {
            (delta.channel == OutputChannel::Content ? content : reasoning) += delta.text;
        };
        if (published.publication_order().empty()) {
            for (const auto& delta : published) { text(delta); }
        } else {
            for (const auto& item : published.publication_order()) {
                if (item.tool_call) { calls.push_back(published.tool_calls()[item.index]); }
                else { text(published.begin()[item.index]); }
            }
        }
    }
    friend bool operator==(const Stream& a, const Stream& b) {
        return a.content == b.content && a.reasoning == b.reasoning && a.calls.size() == b.calls.size();
    }
};

struct Run {
    Stream stream;
    FinishReason finish = FinishReason::None;
    std::uint32_t committed = 0;       // every committed output token, control included
    std::uint32_t controls = 0;        // control spans committed
    std::uint32_t widest_license = 0;
    std::vector<std::uint32_t> licenses;
    Tokens control;
};

// The Engine's loop for one request: each round the Program samples at most `width` tokens (a draft
// round's width), within the license the session gives it and the output budget; the control span is
// committed instead of a round while one is pending. `thought` is what the model writes while it is
// thinking, `answer` what it writes after its thinking closed (by itself or by the control).
Run drive(g4::OutputSession& session, const Tokens& thought, const Tokens& answer, std::size_t width,
          std::uint32_t limit) {
    Run run;
    std::size_t thought_at = 0, answer_at = 0;
    bool closed = false; // the control closed the thinking: the model writes its answer now
    while (run.finish == FinishReason::None) {
        const std::uint32_t remaining = limit - run.committed;
        require(remaining > 0, "the loop ran past the output limit");
        if (const auto span = session.pending_control_tokens(); !span.empty()) {
            run.control.assign(span.begin(), span.end());
            const auto decision = session.preview_control(run.control, remaining);
            require(decision.accepted_tokens == run.control.size() && !decision.finished() &&
                        decision.continuation == runtime::ContinuationAction::Decode,
                    "the control span was not accepted whole");
            run.stream.collect(session.commit_preview());
            run.committed += decision.accepted_tokens;
            ++run.controls;
            closed = true;
            continue;
        }
        const std::uint32_t license = session.model_token_budget_remaining(remaining);
        require(license > 0 && license <= remaining, "the session licensed no tokens, or more than are left");
        run.licenses.push_back(license);
        run.widest_license = std::max(run.widest_license, license);
        // Where the model is: still in its thinking stream, or past the control, in its answer.
        const Tokens& source = closed ? answer : thought;
        std::size_t& at = closed ? answer_at : thought_at;
        require(at < source.size(), "the model's stream ran out");
        const auto n = std::min<std::size_t>({width, license, source.size() - at});
        const auto decision = session.preview_model(std::span<const TokenId>(source).subspan(at, n), remaining,
                                                    FinishReason::OutputLimit);
        require(decision.accepted_tokens <= n, "more tokens accepted than were sampled");
        run.stream.collect(session.commit_preview());
        at += decision.accepted_tokens;
        run.committed += decision.accepted_tokens;
        run.finish = decision.finish_reason;
    }
    return run;
}

Tokens concat(Tokens a, const Tokens& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

// Distinct varied prose, a different combination of words in every sentence.
std::string varied(int sentences, unsigned seed = 7) {
    static const char* subjects[] = {"the parser", "this branch", "the cache", "our bound", "the loop", "a tile",
                                     "the queue", "that edge", "the schema", "each lane", "the budget"};
    static const char* verbs[] = {"handles", "skips", "rounds", "checks", "splits", "merges", "drops",
                                  "keeps", "orders", "limits", "scales", "tracks", "maps"};
    static const char* objects[] = {"empty input", "the last token", "two columns", "a ragged width",
                                    "every retry", "the null case", "large values", "unicode names",
                                    "negative offsets", "the first page", "stale entries", "short reads"};
    std::string text;
    unsigned state = seed;
    for (int s = 0; s < sentences; ++s) {
        state = state * 1103515245U + 12345U;
        text += "step " + std::to_string(s) + ": " + subjects[(state >> 8) % 11] + " " + verbs[(state >> 12) % 13] + " " +
                objects[(state >> 16) % 12] + " against bound " + std::to_string((state >> 4) % 811) + ". ";
    }
    return text;
}

struct Rig {
    g4::Frontend frontend;
    Tokens text(std::string_view value) const { return frontend.tokenize_text(value); }
    Tokens close() const { return frontend.tokenize_text("<channel|>"); }
};

// The budget stops the thinking at its cap: the control span is committed, the model answers.
void budget_ends_thinking(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    require(prompt.summary().starts_in_reasoning, "the thinking prompt does not open the thought channel");
    const Tokens thought = rig.text(varied(60));
    const Tokens answer = concat(rig.text("The answer is here."), {kEos});
    constexpr std::uint32_t budget = 24;

    for (const std::size_t width : {1U, 2U, 3U, 5U, 8U, 64U}) {
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = budget});
        const auto run = drive(session, thought, answer, width, 4096);
        const auto stats = session.thinking_stats();
        // No round ever carries a token past the cap (a draft round's width never exceeds the license).
        require(run.controls == 1 && run.finish == FinishReason::StopToken,
                "width " + std::to_string(width) + ": the budget did not end the thinking once and the reply after it");
        require(stats.configured_budget == budget && stats.model_thinking_tokens == budget && stats.applied &&
                    stats.injected_tokens == run.control.size(),
                "width " + std::to_string(width) + ": thinking accounting after the control is wrong");
        require(run.committed == budget + run.control.size() + answer.size(),
                "width " + std::to_string(width) + ": committed tokens are not the thinking, the control and the answer");
        require(run.stream.content == "The answer is here.",
                "width " + std::to_string(width) + ": the answer did not follow the control in the content channel: '" +
                    run.stream.content + "'");
        // The reasoning is the first `budget` tokens of the thought, then the message (the close token is framing).
        const std::string expected_thought = frontend.detokenize_bytes(std::span<const TokenId>(thought).first(budget));
        const std::string message(kDefaultControl.substr(0, kDefaultControl.size() - std::string_view("<channel|>").size()));
        require(run.stream.reasoning == expected_thought + message,
                "width " + std::to_string(width) + ": the reasoning is not the capped thought and the message: '" +
                    run.stream.reasoning + "'");
        require(frontend.detokenize_bytes(run.control) == kDefaultControl,
                "width " + std::to_string(width) + ": the default control span is not the documented message and close");
        require(run.control.back() == rig.close().at(0), "the control span does not end with the thought close");
        require(session.model_token_budget_remaining(500) == 500 && session.pending_control_tokens().empty(),
                "after the control the model is licensed freely again");
    }
    std::cout << "budget: control committed at the cap for every round width; the answer follows\n";
}

// The license: the model never samples past the cap, whatever the round width and output limit.
void license_follows_budget(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    const Tokens thought = rig.text(varied(40));
    constexpr std::uint32_t budget = 10;
    auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = budget});
    require(session.model_token_budget_remaining(100) == budget && session.model_token_budget_remaining(4) == 4 &&
                session.model_token_budget_remaining(1) == 1,
            "the license is not the smaller of the budget and the output left");

    // A round of the licensed width carries the count down to the cap.
    (void)session.preview_model(std::span<const TokenId>(thought).first(4), 100, FinishReason::OutputLimit);
    (void)session.commit_preview();
    require(session.model_token_budget_remaining(100) == budget - 4 && session.pending_control_tokens().empty(),
            "the budget left is not the budget less the thinking written");

    // A discarded round leaves everything as it was.
    const auto before = session.model_token_budget_remaining(100);
    (void)session.preview_model(std::span<const TokenId>(thought).subspan(4, 6), 100, FinishReason::OutputLimit);
    session.discard_preview();
    require(session.model_token_budget_remaining(100) == before && session.pending_control_tokens().empty() &&
                session.thinking_stats().model_thinking_tokens == 4,
            "a discarded round changed the thinking state");

    // A round past the license is a defect of the caller, never accepted silently.
    bool refused = false;
    try {
        (void)session.preview_model(std::span<const TokenId>(thought).subspan(4, 7), 100, FinishReason::OutputLimit);
    } catch (const std::logic_error&) { refused = true; }
    session.discard_preview();
    require(refused, "a round past the thinking budget was accepted");

    // At the cap the next round waits for the control: no model token may advance, and only the exact span is taken.
    const auto at_cap = session.preview_model(std::span<const TokenId>(thought).subspan(4, 6), 100,
                                              FinishReason::OutputLimit);
    require(at_cap.accepted_tokens == 6 && !at_cap.finished() &&
                at_cap.continuation == runtime::ContinuationAction::ApplyTargetControl,
            "the cap did not ask for the control span");
    (void)session.commit_preview();
    require(session.model_token_budget_remaining(100) == 0 && !session.pending_control_tokens().empty(),
            "the license is not closed while the control is pending");
    bool advanced = false;
    try { (void)session.preview_model(std::span<const TokenId>(thought).subspan(10, 1), 100, FinishReason::OutputLimit); }
    catch (const std::logic_error&) { advanced = true; }
    require(advanced, "the model advanced while the control was pending");
    const Tokens wrong = rig.text("Not the control.");
    bool wrong_span = false;
    try { (void)session.preview_control(wrong, 100); } catch (const std::invalid_argument&) { wrong_span = true; }
    require(wrong_span, "a span other than the pending control was accepted");
    bool no_room = false;
    const Tokens control(session.pending_control_tokens().begin(), session.pending_control_tokens().end());
    try { (void)session.preview_control(control, static_cast<std::uint32_t>(control.size()) - 1U); }
    catch (const std::invalid_argument&) { no_room = true; }
    require(no_room, "a control span longer than the output left was accepted");
    std::cout << "license: the round never passes the cap; the control waits for the exact span\n";
}

// A model that closes its own thinking, stops, or meets the output limit never gets the control.
void priorities(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    const Tokens thought = rig.text("one two three four five six");
    const Tokens close = rig.close();

    {   // The model's own close at the cap: the thinking is over, nothing is injected.
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 3});
        Tokens round = rig.text("one two");
        round.push_back(close.at(0));
        require(round.size() == 3, "test round is not three tokens");
        const auto decision = session.preview_model(round, 100, FinishReason::OutputLimit);
        Stream stream;
        stream.collect(session.commit_preview());
        require(!decision.finished() && decision.continuation == runtime::ContinuationAction::Decode &&
                    session.pending_control_tokens().empty() && !session.thinking_stats().applied &&
                    session.thinking_stats().model_thinking_tokens == 3,
                "the model's own close at the cap still asked for the control");
        require(stream.reasoning == "one two" && stream.content.empty(), "the natural close did not split the channels");
        // Tokens after the close are the answer: they are not thinking and take no budget.
        const auto after = session.preview_model(rig.text("hello again friend"), 100, FinishReason::OutputLimit);
        require(!after.finished() && session.thinking_stats().model_thinking_tokens == 3, "the answer counted as thinking");
        (void)session.commit_preview();
    }
    {   // A stop token at the cap ends the reply.
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 2});
        const auto decision = session.preview_model(Tokens{thought.at(0), kEos}, 100, FinishReason::OutputLimit);
        (void)session.commit_preview();
        require(decision.finish_reason == FinishReason::StopToken &&
                    decision.continuation == runtime::ContinuationAction::Decode &&
                    session.pending_control_tokens().empty(),
                "a stop token at the cap did not take priority");
    }
    {   // The output limit at the cap ends the reply; there is no room to commit the control into.
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 2});
        const auto decision = session.preview_model(std::span<const TokenId>(thought).first(2), 2,
                                                    FinishReason::ContextCapacity);
        (void)session.commit_preview();
        require(decision.finish_reason == FinishReason::ContextCapacity && session.pending_control_tokens().empty(),
                "the output limit at the cap did not take priority");
    }
    {   // A request that does not start in thinking has nothing to cap.
        const auto direct = frontend.prepare(input(false));
        require(!direct.summary().starts_in_reasoning, "the non-thinking prompt opens the thought channel");
        auto session = frontend.make_output_session(direct, {}, {}, ThinkingControlOptions{.budget = 2});
        require(session.model_token_budget_remaining(50) == 50, "a non-thinking reply was capped");
        const auto decision = session.preview_model(std::span<const TokenId>(thought).first(4), 50,
                                                    FinishReason::OutputLimit);
        require(!decision.finished() && decision.continuation == runtime::ContinuationAction::Decode,
                "a non-thinking reply asked for the control");
    }
    {   // A zero budget is a request error, as on every model.
        bool refused = false;
        try { (void)frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 0}); }
        catch (const std::invalid_argument&) { refused = true; }
        require(refused, "a zero thinking budget was accepted");
    }
    std::cout << "priorities: the model's close, a stop and the output limit all precede the control\n";
}

// The output limit has to leave room past the cap for the whole control span and one token of answer.
void capacity(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 8});
    // A limit the cap never reaches takes no effect and needs no room.
    session.validate_generation_capacity(1);
    session.validate_generation_capacity(8);
    const Tokens thought = rig.text(varied(10));
    (void)session.preview_model(std::span<const TokenId>(thought).first(8), 1000, FinishReason::OutputLimit);
    (void)session.commit_preview();
    const auto control = session.pending_control_tokens().size();
    require(control > 1, "the control span is a single token");
    // The same check as the Engine makes before it takes a request, on a fresh session.
    auto fresh = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 8});
    bool short_by_one = false;
    try { fresh.validate_generation_capacity(static_cast<std::uint32_t>(8 + control)); }
    catch (const std::invalid_argument&) { short_by_one = true; }
    require(short_by_one, "an output limit without room for one answer token after the control was accepted");
    fresh.validate_generation_capacity(static_cast<std::uint32_t>(8 + control + 1));
    auto direct = frontend.make_output_session(frontend.prepare(input(false)), {}, {}, ThinkingControlOptions{.budget = 8});
    direct.validate_generation_capacity(9);
    std::cout << "capacity: cap + control span + one token is the least a capped thinking request needs\n";
}

// The budget message replaces the built-in text before the close; the close is appended when missing.
void message(const std::string& tokenizer_json, const std::string& chat, const Rig& rig) {
    const auto build = [&](const std::string& text) {
        g4::TextResources resources;
        resources.tokenizer_json = tokenizer_json;
        resources.chat_template_jinja = chat;
        resources.generation_config_json = R"({"eos_token_id":[1,106,50]})";
        g4::FrontendOptions options;
        options.max_context = 262144;
        options.chat_template_path = std::filesystem::path(INFERNIX_SOURCE_DIR) / "tools/chat_templates/gemma4_fixed.jinja";
        options.thinking_budget_message = text;
        return g4::Frontend(resources, options);
    };
    const auto control_of = [&](const g4::Frontend& frontend) {
        auto session = frontend.make_output_session(frontend.prepare(input(true)), {}, {}, ThinkingControlOptions{.budget = 2});
        (void)session.preview_model(Tokens{rig.text("a").at(0), rig.text("b").at(0)}, 50, FinishReason::OutputLimit);
        (void)session.commit_preview();
        const auto span = session.pending_control_tokens();
        return frontend.detokenize_bytes(span);
    };
    require(control_of(build("Time to act now:")) == "Time to act now:\n<channel|>",
            "a message without the close did not get the canonical close");
    require(control_of(build("Time to act now:\n<channel|>")) == "Time to act now:\n<channel|>",
            "the canonical close was appended to a message that already had it");

    // The message passes through the session like any thinking: it publishes in the reasoning channel.
    {
        const auto frontend = build("Time to act now:");
        auto session = frontend.make_output_session(frontend.prepare(input(true)), {}, {}, ThinkingControlOptions{.budget = 2});
        const Tokens thought{rig.text("x").at(0), rig.text("y").at(0)};
        const Tokens answer = concat(rig.text("Done."), {kEos});
        const auto run = drive(session, thought, answer, 2, 100);
        require(run.stream.reasoning == "xyTime to act now:\n" && run.stream.content == "Done." &&
                    session.thinking_stats().injected_tokens == run.control.size(),
                "a custom message was not published as reasoning before the answer: '" + run.stream.reasoning + "'");
    }
    // What would end the reply or leave the thinking some other way is refused at startup.
    for (const char* bad : {"stop<eos>", "stop<turn|>", "<|channel>thought", "text <channel|> more",
                            "call <|tool_call>", "x<tool_call|>", "<|turn>model"}) {
        bool refused = false;
        try { (void)build(bad); } catch (const std::invalid_argument&) { refused = true; }
        require(refused, std::string("the thinking budget message '") + bad + "' was accepted");
    }
    std::cout << "message: replaces the built-in text, takes the canonical close, refuses control tokens\n";
}

// The guard watches the thinking for repeated passages and, at a check, ends the reply or concludes.
void loop_guard(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    std::string looping = varied(260);
    for (int i = 0; i < 60; ++i) {
        looping += "Wait, let me double-check the case where the list is empty. Then the function returns "
                   "zero, which matches the expected output, so that case is fine. ";
    }
    const Tokens loop_tokens = rig.text(looping);
    const Tokens varied_tokens = rig.text(varied(260) + varied(260, 11));
    const Tokens answer = concat(rig.text("Concluded."), {kEos});
    constexpr std::size_t kRound = 200;

    {   // Stop ends the reply where the guard fires.
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.loop = ReasoningLoopAction::Stop});
        const auto run = drive(session, loop_tokens, answer, kRound, kPlenty);
        const auto stats = session.thinking_stats();
        require(run.finish == FinishReason::OutputLimit && stats.loop_detected && !stats.applied &&
                    stats.loop_coverage >= 0.25F && stats.loop_thinking_tokens >= 512,
                "Stop did not end a looping thought");
    }
    {   // Conclude closes the thinking with the same control a budget commits, and the model answers.
        auto session = frontend.make_output_session(prompt, {}, {},
                                                    ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude});
        const auto run = drive(session, loop_tokens, answer, kRound, kPlenty);
        const auto stats = session.thinking_stats();
        require(run.controls == 1 && run.finish == FinishReason::StopToken && stats.loop_detected && stats.applied &&
                    stats.injected_tokens == run.control.size() && run.stream.content == "Concluded.",
                "Conclude did not close the looping thought and let the model answer");
        require(frontend.detokenize_bytes(run.control) == kDefaultControl, "Conclude committed another control span");
    }
    {   // Varied thinking, and a guard that is off, never fire.
        for (const bool on : {true, false}) {
            auto session = frontend.make_output_session(
                prompt, {}, {}, ThinkingControlOptions{.loop = on ? ReasoningLoopAction::Conclude : ReasoningLoopAction::Off});
            const Tokens& stream = on ? varied_tokens : loop_tokens;
            std::size_t at = 0;
            std::uint32_t budget = kPlenty;
            while (at < stream.size()) {
                const auto n = std::min(kRound, stream.size() - at);
                const auto decision = session.preview_model(std::span<const TokenId>(stream).subspan(at, n), budget,
                                                            FinishReason::OutputLimit);
                (void)session.commit_preview();
                at += n;
                budget -= decision.accepted_tokens;
                require(!decision.finished() && decision.continuation == runtime::ContinuationAction::Decode,
                        on ? "the guard fired on varied thinking" : "the guard fired while off");
            }
            require(!session.thinking_stats().loop_detected, "the guard recorded a loop it did not act on");
        }
    }
    {   // Without room for the control and one token, Conclude stops instead.
        auto probe = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude});
        std::uint32_t fired_after = 0;
        for (std::size_t at = 0; at < loop_tokens.size(); at += kRound) {
            const auto n = std::min(kRound, loop_tokens.size() - at);
            const auto d = probe.preview_model(std::span<const TokenId>(loop_tokens).subspan(at, n), kPlenty,
                                               FinishReason::OutputLimit);
            (void)probe.commit_preview();
            fired_after += d.accepted_tokens;
            if (d.continuation == runtime::ContinuationAction::ApplyTargetControl) { break; }
        }
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude});
        const auto run = drive(session, loop_tokens, answer, kRound, fired_after + 1U);
        require(run.finish == FinishReason::OutputLimit && session.thinking_stats().loop_detected &&
                    !session.thinking_stats().applied && run.controls == 0,
                "Conclude without room for the control did not stop");
    }
    std::cout << "loop guard: stop, conclude, varied and off, and conclude without room\n";
}

// A request whose budget is not reached runs exactly as one with no budget at all.
void unreached_budget_is_unchanged(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    // Thinking, the close, a reply, a tool-call-free ending: the streams, the decisions and the accounting agree.
    const Tokens thought = concat(rig.text(varied(30)), rig.close());
    const Tokens answer = concat(rig.text("A short reply."), {kEos});
    const Tokens stream = concat(thought, answer);
    struct Seen { Stream out; std::vector<std::uint32_t> accepted; FinishReason finish; std::uint32_t thinking; };
    const auto play = [&](ThinkingControlOptions options, std::size_t width) {
        auto session = frontend.make_output_session(prompt, {}, {}, options);
        Seen seen{{}, {}, FinishReason::None, 0};
        std::size_t at = 0;
        std::uint32_t remaining = 4096;
        while (seen.finish == FinishReason::None) {
            const auto n = std::min({width, static_cast<std::size_t>(session.model_token_budget_remaining(remaining)),
                                     stream.size() - at});
            const auto decision = session.preview_model(std::span<const TokenId>(stream).subspan(at, n), remaining,
                                                        FinishReason::OutputLimit);
            seen.out.collect(session.commit_preview());
            seen.accepted.push_back(decision.accepted_tokens);
            at += decision.accepted_tokens;
            remaining -= decision.accepted_tokens;
            seen.finish = decision.finish_reason;
            require(decision.continuation == runtime::ContinuationAction::Decode && session.pending_control_tokens().empty(),
                    "a control was requested although the budget cannot be reached");
        }
        seen.thinking = session.thinking_stats().model_thinking_tokens;
        return seen;
    };
    for (const std::size_t width : {1U, 4U, 7U, 4096U}) {
        const Seen none = play({}, width);
        const Seen big = play(ThinkingControlOptions{.budget = 100000}, width);
        const Seen guarded = play(ThinkingControlOptions{.loop = ReasoningLoopAction::Conclude}, width);
        require(none.out == big.out && none.out == guarded.out && none.accepted == big.accepted &&
                    none.accepted == guarded.accepted && none.finish == FinishReason::StopToken &&
                    none.finish == big.finish && none.finish == guarded.finish,
                "width " + std::to_string(width) + ": an unreached budget or guard changed the reply or its rounds");
        require(none.out.reasoning == rig.frontend.detokenize_bytes(rig.text(varied(30))) &&
                    none.out.content == "A short reply.",
                "the reply is not the thought and the answer");
        // The model's thinking is counted whether or not a budget is set (the request log reports it).
        require(none.thinking == big.thinking && none.thinking == thought.size(),
                "the thinking tokens were not counted the same with and without a budget");
    }
    std::cout << "unchanged: an unreached budget or guard leaves the reply, the rounds and the count as they were\n";
}

// The raw presentation keeps the control's bytes as the model's own text.
void raw_output(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto prompt = frontend.prepare(input(true));
    auto session = frontend.make_output_session(prompt, {}, OutputOptions{.raw = true}, ThinkingControlOptions{.budget = 2});
    const auto decision = session.preview_model(Tokens{rig.text("a").at(0), rig.text("b").at(0)}, 50,
                                                FinishReason::OutputLimit);
    Stream stream;
    stream.collect(session.commit_preview());
    require(decision.continuation == runtime::ContinuationAction::ApplyTargetControl,
            "raw presentation disabled the thinking budget");
    const Tokens control(session.pending_control_tokens().begin(), session.pending_control_tokens().end());
    (void)session.preview_control(control, 100);
    stream.collect(session.commit_preview());
    require(stream.content == "ab" + std::string(kDefaultControl) && stream.reasoning.empty(),
            "raw output did not keep the inserted control as text: '" + stream.content + "'");
    std::cout << "raw: the control stays in the text\n";
}

// A grammar-constrained reply takes the control span too: its framing accepts the message and the close,
// and the constrained answer follows; a tool call after it is parsed as usual.
void constrained(const Rig& rig) {
    const auto& frontend = rig.frontend;
    const auto schema = OutputConstraint::json_schema(
        R"({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false})");
    {
        const auto prompt = frontend.prepare(input(true));
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 3}, schema);
        const Tokens thought = rig.text("long plan that goes on and on");
        const Tokens answer = concat(rig.text(R"({"ok":true})"), {kEos});
        const auto run = drive(session, thought, answer, 2, 500);
        require(run.controls == 1 && run.finish == FinishReason::StopToken && run.stream.content == R"({"ok":true})" &&
                    session.thinking_stats().applied && !run.stream.reasoning.empty(),
                "a constrained reply did not take the control span and answer in its grammar: '" + run.stream.content + "'");
    }
    {
        const auto prompt = frontend.prepare(input(true, true));
        auto session = frontend.make_output_session(prompt, {}, {}, ThinkingControlOptions{.budget = 3});
        const Tokens thought = rig.text("long plan that goes on and on");
        const Tokens answer = concat(
            rig.text("<|tool_call>call:clock{zone:<|\"|>UTC<|\"|>}<tool_call|>"), {kEos});
        const auto run = drive(session, thought, answer, 3, 500);
        const auto calls = session.take_tool_calls();
        require(run.controls == 1 && calls.size() == 1 && calls[0].name == "clock" && run.stream.content.empty(),
                "a tool call after the control span was not parsed");
    }
    std::cout << "constrained: the grammar accepts the control; its answer and tool calls follow\n";
}
} // namespace

int main() {
    const char* hf = std::getenv("INFERNIX_GEMMA4_HF");
    if (!hf || !*hf) { std::cout << "SKIP: INFERNIX_GEMMA4_HF is not set\n"; return 77; }
    try {
        const std::filesystem::path root(hf);
        const auto tokenizer_json = file(root / "tokenizer.json");
        const auto chat = file(root / "chat_template.jinja");
        g4::TextResources resources;
        resources.tokenizer_json = tokenizer_json;
        resources.chat_template_jinja = chat;
        resources.generation_config_json = R"({"eos_token_id":[1,106,50]})";
        g4::FrontendOptions options;
        options.max_context = 262144;
        options.chat_template_path = std::filesystem::path(INFERNIX_SOURCE_DIR) / "tools/chat_templates/gemma4_fixed.jinja";
        const Rig rig{g4::Frontend(resources, options)};
        budget_ends_thinking(rig);
        license_follows_budget(rig);
        priorities(rig);
        capacity(rig);
        message(tokenizer_json, chat, rig);
        loop_guard(rig);
        unreached_budget_is_unchanged(rig);
        raw_output(rig);
        constrained(rig);
        std::cout << "OK gemma4 thinking control\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << "FAIL " << error.what() << '\n'; return 1; }
}
