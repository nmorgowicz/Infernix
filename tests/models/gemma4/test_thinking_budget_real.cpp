// The thinking budget through the public Engine on a real Gemma 4 artifact (control span committed to the resident
// sequence at the cap). INFERNIX_GEMMA4_ARTIFACT selects an artifact with the `mtp` component and the infernix chat
// template (thinking on opens the thought channel in the prompt); absent means skip (77).
// INFERNIX_GEMMA4_KV=fp8 / INFERNIX_GEMMA4_QK=fp8 select the FP8 routes.
//
// 1. A reply whose thinking reaches its budget gets the control span committed at that token: exactly `budget`
//    model tokens, the control, then an answer in the content channel. With MTP at 3 and 7 drafts (the cap lands inside
//    a draft round's width) every generated token is accounted for by the rounds, the plain steps, the first token
//    and the control: no round ran past the cap and no accepted count is off around the injection.
// 2. The budget message replaces the built-in text through the Engine option.
// 3. A request whose limit leaves too little room for the control is refused (HTTP 400's error kind); one the cap never
//    reaches runs to its limit; and a budget that is not reached changes nothing (token-identical to no budget).
// 4. A turn that continues a capped reply resumes from the state the capped reply left: on the free route it
//    reuses the history through the control span; on the reproducible route it equals a cache-off turn bit for bit.
#include "prefix_real_support.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace prefix_real;

constexpr std::uint32_t kBudget = 24;
constexpr std::string_view kDefaultControl =
    "\n\nConsidering the limited thinking time, I have to give my answer now, based on my thinking so far.\n<channel|>";

struct Setup {
    const char* artifact = nullptr;
    infernix::KvCacheStorage kv = infernix::KvCacheStorage::BFloat16;
    infernix::ops::Gemma4QkMode qk = infernix::ops::Gemma4QkMode::BFloat16;
};

// A few hundred tokens of notes, so that a prompt reaches the chunk grid a reproducible resume restores at.
std::string question() {
    std::string notes = "Describe a quiet harbour town at dawn in three sentences, using these notes:\n";
    for (int i = 1; i <= 40; ++i) {
        notes += "Note " + std::to_string(i) + ": boat " + std::to_string(i * 7) + " is moored by pier " +
                 std::to_string(i % 5 + 1) + " and its owner mends nets.\n";
    }
    return notes;
}

// `free_route`: free-cache-resume (the serving default: decode-computed state is published and resumed at any
// frontier); off, only full prefill calls on the chunk grid publish, and a resume equals a cold prefill bit for bit.
infernix::EngineOptions options_for(const Setup& setup, std::uint32_t drafts, std::size_t host_bytes, bool free_route,
                                    std::string message = {}) {
    auto options = engine_options(setup.artifact, 128, host_bytes, 4096);
    options.kv_cache = setup.kv;
    options.gemma4_qk = setup.qk;
    if (drafts != 0) {
        options.speculative.backend = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = drafts;
    }
    if (free_route) { options.fast_path_exclusions.clear(); }
    options.thinking_budget_message = std::move(message);
    return options;
}

infernix::PromptInput thinking_chat(const std::string& user) {
    infernix::PromptInput input;
    infernix::ChatMessage message;
    message.role = infernix::ChatRole::User;
    message.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = user});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = true;
    return input;
}

infernix::RequestOptions budgeted(std::uint32_t outputs, std::optional<std::uint32_t> budget, bool reuse) {
    auto options = greedy(outputs, reuse, /*model_stops=*/true);
    options.execution.thinking.budget = budget;
    return options;
}

infernix::GenerationResult run(infernix::Engine& engine, std::uint32_t outputs, std::optional<std::uint32_t> budget,
                               bool reuse = false) {
    const infernix::PreparedPrompt prompt = engine.prepare(thinking_chat(question()));
    require(prompt.summary().starts_in_reasoning,
            "the artifact's chat template does not open the thought channel for a thinking request");
    return engine.generate(engine.prepare(thinking_chat(question())), budgeted(outputs, budget, reuse));
}

// The control span a request committed: the tokens after its first `kBudget`.
Tokens control_of(const infernix::GenerationResult& result) {
    require(result.generated_token_ids.size() >= kBudget + result.thinking.injected_tokens,
            "the reply is shorter than its thinking budget and control");
    const auto first = result.generated_token_ids.begin() + kBudget;
    return Tokens(first, first + result.thinking.injected_tokens);
}

// The control span's length, read from a request that reaches its budget.
std::uint32_t control_size_of(infernix::Engine& engine, const std::string& label) {
    const auto probe = run(engine, kBudget + 200, kBudget);
    require(probe.thinking.applied && probe.thinking.injected_tokens > 1, label + ": the budget committed no control span");
    return probe.thinking.injected_tokens;
}

// A reply capped at `kBudget` thinking tokens, cut by its output limit shortly into the answer (so that no round
// of the reply is a truncated one and the speculative counters account for every token).
void capped_reply(infernix::Engine& engine, const std::string& label, std::uint32_t drafts, std::string_view control_text) {
    const std::uint32_t control_size = control_size_of(engine, label);
    const std::uint32_t outputs = kBudget + control_size + 40;
    const auto result = run(engine, outputs, kBudget);
    const auto& stats = result.thinking;
    require(stats.model_thinking_tokens == kBudget,
            label + ": the model's thinking is not capped at the budget (" + std::to_string(stats.model_thinking_tokens) +
                "); the prompt needs a longer thought");
    expect(stats.configured_budget == kBudget && stats.applied && stats.injected_tokens == control_size,
           label + ": thinking accounting (budget, applied, control tokens)");
    const Tokens control = control_of(result);
    expect(engine.detokenize_bytes(control) == control_text, label + ": the committed span is the control text");
    expect(std::search(result.generated_token_ids.begin() + kBudget + control.size(), result.generated_token_ids.end(),
                       control.begin(), control.end()) == result.generated_token_ids.end(),
           label + ": the control span is committed exactly once");
    expect(result.generated_token_ids.size() == outputs && result.finish_reason == infernix::FinishReason::OutputLimit,
           label + ": the reply ended by its limit (" + std::to_string(result.generated_token_ids.size()) + " tokens)");
    expect(result.reasoning_tokens == kBudget + control_size,
           label + ": the reasoning channel holds the capped thinking and the control (" +
               std::to_string(result.reasoning_tokens) + ")");
    const std::string message(control_text.substr(0, control_text.size() - std::string_view("<channel|>").size()));
    expect(result.reasoning.size() > message.size() && result.reasoning.ends_with(message),
           label + ": the reasoning stream ends with the control's message");
    expect(!result.content.empty() && result.content.find("<channel|>") == std::string::npos,
           label + ": the answer follows in the content channel, without the close marker");
    // The same request without a budget thinks past the cap: the cap is what ended it. A plain decoder's first
    // tokens do not depend on the cap (an MTP round's width does, near a tie, so only the plain run is compared).
    const auto uncapped = run(engine, kBudget + 1, std::nullopt);
    expect(uncapped.thinking.model_thinking_tokens == kBudget + 1 && !uncapped.thinking.applied,
           label + ": without a budget the thinking runs on");
    if (drafts == 0) {
        expect(std::equal(result.generated_token_ids.begin(), result.generated_token_ids.begin() + kBudget,
                          uncapped.generated_token_ids.begin()),
               label + ": the capped reply's thinking is the uncapped reply's first tokens");
    } else {
        const auto& s = result.speculative;
        require(s.enabled && s.backend == infernix::SpeculativeBackend::Mtp && s.draft_window == drafts,
                label + ": MTP identity lost");
        expect(s.accepted_tokens <= s.drafted_tokens && s.round_steps.size() == s.rounds,
               label + ": speculative counters are inconsistent");
        std::uint64_t round_tokens = 0, proposed = 0, accepted = 0;
        for (const auto& step : s.round_steps) {
            round_tokens += static_cast<std::uint64_t>(step.accepted) + 1U;
            proposed += step.drafted;
            accepted += step.accepted;
            expect(step.drafted >= 1 && step.drafted <= drafts && step.accepted <= step.drafted && !step.ngram,
                   label + ": a round's drafted/accepted counts are out of range");
        }
        expect(proposed == s.drafted_tokens && accepted == s.accepted_tokens,
               label + ": the round history disagrees with the totals");
        // The first token comes from the prefill, every other model token from a round (its accepted drafts and its
        // bonus token) or a plain step; the control span is committed without sampling.
        const std::uint64_t accounted = 1ULL + s.fallback_steps + round_tokens + control_size;
        expect(accounted == result.generated_token_ids.size(),
               label + ": rounds, plain steps and control account for " + std::to_string(accounted) + " of " +
                   std::to_string(result.generated_token_ids.size()) + " generated tokens");
        expect(s.rounds > 0, label + ": no draft round ran");
        std::printf("%s: %llu rounds, %llu/%llu drafts accepted, %llu plain steps\n", label.c_str(),
                    static_cast<unsigned long long>(s.rounds), static_cast<unsigned long long>(s.accepted_tokens),
                    static_cast<unsigned long long>(s.drafted_tokens), static_cast<unsigned long long>(s.fallback_steps));
    }
    std::printf("%s: %u thinking + %u control + %zu answer tokens, content %zu bytes\n", label.c_str(), kBudget,
                control_size, result.generated_token_ids.size() - kBudget - control_size, result.content.size());
}

void capacity_and_unreached(infernix::Engine& engine, const std::string& label) {
    const std::uint32_t control_size = control_size_of(engine, label);
    // The limit leaves less than the control and one answer token past the cap: refused, as an invalid request.
    for (const std::uint32_t outputs : {kBudget + 1U, kBudget + control_size}) {
        bool refused = false;
        try { (void)run(engine, outputs, kBudget); }
        catch (const infernix::RequestError& error) {
            refused = error.kind() == infernix::RequestErrorKind::ThinkingBudgetCapacityInsufficient;
        }
        expect(refused, label + ": a limit of " + std::to_string(outputs) + " tokens was not refused as capacity-insufficient");
    }
    {   // Exactly enough: the control and one answer token.
        const auto result = run(engine, kBudget + control_size + 1, kBudget);
        expect(result.thinking.applied && result.generated_token_ids.size() == kBudget + control_size + 1 &&
                   result.finish_reason == infernix::FinishReason::OutputLimit,
               label + ": the smallest limit that fits the control did not run it and one more token");
    }
    {   // A limit the cap never reaches: no control, the reply ends by its limit.
        const auto result = run(engine, kBudget, kBudget);
        expect(!result.thinking.applied && result.generated_token_ids.size() == kBudget &&
                   result.finish_reason == infernix::FinishReason::OutputLimit,
               label + ": a limit at the budget committed a control span");
    }
    {   // A budget that is not reached changes nothing: the same tokens as with no budget.
        const auto none = run(engine, 160, std::nullopt);
        const auto huge = run(engine, 160, 1'000'000);
        expect(none.generated_token_ids == huge.generated_token_ids && !huge.thinking.applied &&
                   none.thinking.model_thinking_tokens == huge.thinking.model_thinking_tokens,
               label + ": an unreached budget changed the reply (" + difference(none.generated_token_ids, huge.generated_token_ids) + ")");
        expect(none.thinking.model_thinking_tokens == none.generated_token_ids.size(),
               label + ": the thinking is counted without a budget");
    }
}

// What the first turn leaves and the second asks: the history (prompt, the capped reply, a new tail), what the
// cached follow-up generated and how much of the history it restored.
struct Resume {
    Tokens next;
    Tokens cached;
    std::uint32_t reused = 0;
    std::uint32_t through_control = 0;
};

Resume cached_follow_up(infernix::Engine& engine, const std::string& label) {
    const infernix::PreparedPrompt prepared = engine.prepare(thinking_chat(question()));
    const auto ids = prepared.token_ids();
    const Tokens prompt(ids.begin(), ids.end());
    const auto first = run(engine, kBudget + 120, kBudget, true);
    require(first.thinking.applied, label + ": the first turn did not apply its budget");
    Tokens history = concat(prompt, first.generated_token_ids);
    const infernix::TokenId end_of_turn = engine.tokenize_text("<turn|>").at(0);
    while (!history.empty() && history.back() == end_of_turn) { history.pop_back(); }
    Resume out;
    out.next = concat(history, engine.tokenize_text("\nGo on."));
    out.through_control = static_cast<std::uint32_t>(prompt.size()) + kBudget + first.thinking.injected_tokens;
    // Raw-token history tests use fixed output lengths, not the model's turn-stop policy.
    const auto resumed = engine.generate(engine.prepare_tokens(out.next), greedy(24, true));
    expect(resumed.finish_reason == infernix::FinishReason::OutputLimit && resumed.generated_token_ids.size() == 24,
           label + ": post-close continuation completes its requested output");
    // A different control message must not reuse state past the changed token. This checks that
    // forced tokens participate in the same content-addressed history as sampled tokens.
    Tokens changed = out.next;
    const auto changed_at = prompt.size() + kBudget;
    changed.at(changed_at) = engine.tokenize_text("Different").at(0);
    require(changed[changed_at] != out.next[changed_at], "the history mutation did not change a control token");
    const auto branched = engine.generate(engine.prepare_tokens(changed), greedy(8, true));
    expect(branched.reused_prompt_tokens <= changed_at && branched.generated_token_ids.size() == 8,
           label + ": changing the control message invalidates reuse past its first changed token");
    // An unrelated request displaces every live prefix; returning must restore a Host snapshot.
    (void)engine.generate(engine.prepare(thinking_chat("Reply with one word: fine.")), greedy(8, false));
    const auto before = engine.runtime_stats();
    auto restored = engine.generate(engine.prepare_tokens(out.next), greedy(24, true));
    const auto after = engine.runtime_stats();
    expect(restored.reused_prompt_tokens > 0 && restored.generated_token_ids.size() == 24 &&
               after.hybrid_host_image_restores > before.hybrid_host_image_restores,
           label + ": displaced history restores a Host snapshot and continues (reused " +
               std::to_string(restored.reused_prompt_tokens) + ", image restores " +
               std::to_string(after.hybrid_host_image_restores - before.hybrid_host_image_restores) + ")");
    // The cache-off oracle below must compare the actual Host-restored response, not only
    // the earlier live-lane continuation.
    out.cached = std::move(restored.generated_token_ids);
    out.reused = restored.reused_prompt_tokens;
    return out;
}

void check_follow_up(const Resume& resume, const infernix::GenerationResult& cold, const std::string& label, bool free_route) {
    if (free_route) {
        expect(resume.reused >= resume.through_control,
               label + ": the follow-up resumes through the control span (reused " + std::to_string(resume.reused) +
                   ", the control ends at " + std::to_string(resume.through_control) + ")");
        // Free-cache-resume is intentionally not cold-prefill-identical: decode and prefill have
        // different arithmetic partitions (test_prefix_scenarios_real.cpp uses the same contract).
        // Keep the comparison visible as evidence, not as an unsupported exact-output oracle.
        std::printf("%s: free-resume vs cold diagnostic: %s\n", label.c_str(),
                    difference(resume.cached, cold.generated_token_ids).c_str());
    } else {
        expect(resume.reused > 0 && resume.reused % 128 == 0,
               label + ": the follow-up resumes on the chunk grid (reused " + std::to_string(resume.reused) + ")");
        expect(resume.cached == cold.generated_token_ids,
               label + ": a restored follow-up equals the cache-off turn (" +
                   difference(resume.cached, cold.generated_token_ids) + ")");
    }
}
} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("skip: INFERNIX_GEMMA4_ARTIFACT is not set");
        return 77;
    }
    Setup setup{.artifact = artifact};
    const char* kv = std::getenv("INFERNIX_GEMMA4_KV");
    if (kv != nullptr && std::string(kv) == "fp8") { setup.kv = infernix::KvCacheStorage::Fp8E4M3Row256; }
    const char* qk = std::getenv("INFERNIX_GEMMA4_QK");
    if (qk != nullptr && std::string(qk) == "fp8") { setup.qk = infernix::ops::Gemma4QkMode::Fp8; }
    constexpr std::size_t kHost = 8192ULL << 20;
    try {
        for (const bool free_route : {true, false}) {
            // MTP: 3 drafts on the free route (the cap inside a round, capacity, the unreached budget), 7 on the
            // reproducible route (a wider round); each ends with the follow-up turn resumed from the capped reply.
            const std::uint32_t drafts = free_route ? 3 : 7;
            const std::string label = std::string("mtp d") + std::to_string(drafts) + (free_route ? " free" : " reproducible");
            Resume resume;
            {
                infernix::Engine engine = start_engine(options_for(setup, drafts, kHost, free_route));
                capped_reply(engine, label, drafts, kDefaultControl);
                if (free_route) { capacity_and_unreached(engine, label); }
                resume = cached_follow_up(engine, label);
            }
            infernix::Engine cold = start_engine(options_for(setup, drafts, 0, free_route));
            check_follow_up(resume, cold.generate(cold.prepare_tokens(resume.next), greedy(24, false)), label, free_route);
        }
        {   // Plain decoding and a custom message.
            infernix::Engine engine = start_engine(options_for(setup, 0, 0, true, "Time to act now:"));
            capped_reply(engine, "plain + message", 0, "Time to act now:\n<channel|>");
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d thinking-budget check(s) failed\n", failures);
        return 1;
    }
    std::puts("all thinking-budget checks passed");
    return 0;
}
