// Gemma4 prefix-cache scenarios on a real artifact, through the public Program and Engine paths.
// INFERNIX_GEMMA4_ARTIFACT selects the artifact (absent: skip, 77); INFERNIX_GEMMA4_HOST_MIB the Host
// tier (default: prefix_real_support.h). Greedy throughout.
//
// The cache follows the grid-aligned policy (plan §8.4, §17.6): with prefill chunk C every prefill call runs
// on the absolute chunk grid, only the prefix full grid calls computed is published, and a snapshot exists
// only at a multiple of C. A cached request therefore runs the very schedule a cache-off prefill of its full
// prompt runs: it must generate exactly the cache-off tokens, and restore exactly the deepest grid snapshot
// the earlier requests left on its prefix.
//
// Every Program scenario is a script of requests: a cache-off lane (host bytes 0, which also proves it never
// reuses even when asked) generates each step's reference first, later prompts embedding earlier replies;
// the cached Program then replays the script. Every cache-on prompt of L tokens offers its deepest grid state
// as a tap: D(L) = floor(L / C) * C, captured at the start of its final partial call or, when L is a multiple
// of C, right after its final full call (then a raw prompt's flexible prompt-tail tap also sits at L - C). A
// later prompt restores a snapshot only below its own last token, so a repeat of a grid-length prompt
// restores L - C. No request leaves an endpoint: a decoding request's committed frontier is past its exact
// prefix, and the prompt end of a grid-length prompt already holds a tap. Every step expected to resume is
// preceded by a scrub (an unrelated 1,600-token request with prefix reuse off) that overwrites the Program's
// KV pages and rings, so a restore must really bring the state back from the Host.
//
//   frontiers      One cache per first-prompt length L: an extension embedding its reply resumes at D(L).
//                  L = 769/831/832 restore 768 (final partial calls of 1, 63 and 64 tokens), 896 restores
//                  896 (ends on the grid), 897 restores 896, 1153/1215 restore 1152 (the sliding ring's 18
//                  pages = 1,152 tokens at chunk 128), 1280/1281 restore 1280 and 1471 restores 1408. For
//                  the grid-length 896 and 1280 a repeat of the first prompt restores 768 and 1152.
//   conversations  A (900 tokens: D = 896), the unrelated B, A's turn 2 (1,140 tokens, displaced: resumes at
//                  896, its own D = 1024), a branch diverging after A's reply (the superseded 896 still
//                  serves), A's turn 3 (resumes at 1024) and a repeat of A's first prompt (896).
//   identities     cold / extension (896) / its continuation (1024) / repeat (896) under BF16 KV, FP8 KV
//                  with BF16 QK and FP8 KV with FP8 QK: each layout identity reuses its own state. Refusal of
//                  a *foreign* identity cannot be driven here (a Program's cache is private; no API hands one
//                  Program's image to another or alters a header): test_prefix_store.cpp proves it on the
//                  header bytes, both directions, for both geometries.
//   saturated      A full Host tier either publishes the new prompt's grid tap (the extension resumes at
//                  896) or reports it skipped (the extension starts at the root); outputs exact either way.
//   chat           Voice-session turns through the public Engine. A chat prompt's snapshots are its deepest
//                  grid state D(P) and the Frontend's turn hints (the structural end of the system turn, the
//                  generation opener; the opener covers the prompt tail), each hint realized at the first
//                  chunk boundary at or past it and only before the prompt's final call. Every expected
//                  restore is the deepest such snapshot of the earlier turns on the new prompt's shared
//                  prefix and below its last token, computed from the rendered ids; every turn generates
//                  exactly what the same prompt with prefix reuse off generates. Engine statistics return to
//                  baseline, and an Engine with the cache off (--host-context-mib 0) never reuses.

#include "prefix_real_support.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <utility>

namespace {

using namespace prefix_real;
namespace ops = infernix::ops;

constexpr std::uint32_t kChunk   = 128;
constexpr std::uint32_t kContext = 4096;
constexpr std::uint32_t kReply   = 66;
constexpr auto kBf16 = infernix::KvCacheStorage::BFloat16;
constexpr auto kFp8  = infernix::KvCacheStorage::Fp8E4M3Row256;

// The chunk grid point at or below `x`, and at or above it.
constexpr std::uint32_t grid_down(std::uint64_t x) { return static_cast<std::uint32_t>(x / kChunk * kChunk); }
constexpr std::uint32_t grid_up(std::uint64_t x) { return static_cast<std::uint32_t>((x + kChunk - 1) / kChunk * kChunk); }
// The deepest grid state a cache-on prompt of `length` tokens offers as a tap.
constexpr std::uint32_t grid_tap(std::size_t length) { return grid_down(length); }

// ---- scripts -----------------------------------------------------------------------------------

struct Step {
    std::string name;
    Tokens prompt;
    std::uint32_t outputs = 0;
    std::uint32_t restore = 0; // the grid frontier the step must restore (0: none)
    Tokens reference;
    unsigned group = 0; // steps of one group share a cache; a new group starts on an empty one
};

// Runs each step on the cache-off lane as it is defined so later prompts can embed earlier replies.
struct Builder {
    g4::Program& cold;
    const g4::Frontend& frontend;
    std::vector<Step> steps;
    unsigned group = 0;

    Tokens add(std::string name, Tokens prompt, std::uint32_t outputs, std::uint32_t restore) {
        require(restore % kChunk == 0 && restore < prompt.size(),
                name + ": a scripted restore lies on the chunk grid below the last prompt token (restore " +
                    std::to_string(restore) + ", prompt " + std::to_string(prompt.size()) + ")");
        const Outcome out = run(cold, frontend, prompt, outputs, true);
        expect(out.reused == 0 && out.quoted == 0, name + ": a cache-off lane never reuses (restored " +
                                                       std::to_string(out.reused) + ", quoted " +
                                                       std::to_string(out.quoted) + ")");
        steps.push_back({std::move(name), std::move(prompt), outputs, restore, out.tokens, group});
        return out.tokens;
    }
};

// Every first-prompt length runs on a cache of its own: images of earlier cases would fill the Host tier,
// and a saturated tier legitimately declines a tap (see saturated_tier).
void frontier_script(Builder& b) {
    for (const std::uint32_t length : {769U, 831U, 832U, 896U, 897U, 1153U, 1215U, 1280U, 1281U, 1471U}) {
        ++b.group;
        const std::string at = "L=" + std::to_string(length);
        const Tokens a       = tokens(length, 100 + length);
        const Tokens reply   = b.add(at + " cold prompt", a, kReply, 0);
        b.add(at + " extension resumes at the grid tap " + std::to_string(grid_tap(length)),
              concat(concat(a, reply), tokens(70, 900 + length)), 8, grid_tap(length));
        // A grid-length prompt's tap at L lies at its own last token: its repeat restores the prompt-tail tap
        // at L - C.
        if (length % kChunk == 0) {
            b.add(at + " repeat resumes at the prompt-tail tap " + std::to_string(length - kChunk), a, kReply,
                  length - kChunk);
        }
    }
}

void conversation_script(Builder& b) {
    const Tokens a1 = tokens(900, 301); // grid tap 896
    const Tokens r1 = b.add("A turn 1 is cold", a1, 40, 0);
    b.add("B (another conversation) is cold and displaces the lane", tokens(1300, 302), 40, 0);
    const Tokens a2 = concat(concat(a1, r1), tokens(200, 303)); // 1,140 tokens: its grid tap 1024
    const Tokens r2 = b.add("A turn 2, displaced, resumes at turn 1's grid tap 896", a2, 40, grid_tap(a1.size()));
    b.add("a branch diverging after A's reply (before turn 2's tap) resumes at the superseded 896",
          concat(concat(a1, r1), tokens(55, 304)), 24, grid_tap(a1.size()));
    b.add("A turn 3 resumes at turn 2's grid tap 1024", concat(concat(a2, r2), tokens(40, 305)), 24, grid_tap(a2.size()));
    b.add("a repeat of A's first prompt resumes at its grid tap 896", a1, 40, grid_tap(a1.size()));
}

void identity_script(Builder& b) {
    const Tokens a     = tokens(1000, 701); // grid tap 896
    const Tokens reply = b.add("cold", a, kReply, 0);
    const Tokens ext   = concat(concat(a, reply), tokens(70, 702)); // 1,136 tokens: its grid tap 1024
    const Tokens ext_reply = b.add("the extension resumes at the grid tap 896", ext, 8, grid_tap(a.size()));
    b.add("the lineage continues from the extension's grid tap 1024", concat(concat(ext, ext_reply), tokens(70, 703)),
          8, grid_tap(ext.size()));
    b.add("a repeat of the first prompt resumes at its grid tap 896", a, kReply, grid_tap(a.size()));
}

// One pass over the script. A cache lives for one group of steps: a new Program with an empty Host tier
// starts each group. Every step generates its cache-off reference exactly and restores its scripted frontier.
void pass(Rig& rig, const std::string& title, const std::vector<Step>& steps, infernix::KvCacheStorage kv,
          ops::Gemma4QkMode qk) {
    std::unique_ptr<g4::Program> cache;
    std::uint64_t restored = 0;
    const Step* first = nullptr;
    unsigned scrubs = 0;
    const auto close = [&] {
        if (!cache) { return; }
        const auto stats = cache->hybrid_stats();
        // No step leaves an endpoint: each decodes past its exact prefix, and a grid-length prompt's end already
        // holds its grid tap.
        expect(stats.reused_tokens == restored && stats.blocks_inserted > 0 && stats.taps_created > 0 &&
                   stats.endpoints_created == 0 && stats.host_block_restores > 0 && stats.host_image_restores > 0,
               title + ": counters agree with the requests (restored " + std::to_string(stats.reused_tokens) + " of " +
                   std::to_string(restored) + ", blocks inserted " + std::to_string(stats.blocks_inserted) +
                   ", taps created " + std::to_string(stats.taps_created) + ", endpoints " +
                   std::to_string(stats.endpoints_created) + " of the expected 0, Host block restores " +
                   std::to_string(stats.host_block_restores) + ", image restores " +
                   std::to_string(stats.host_image_restores) + ")");
        const Outcome off = run(*cache, *rig.frontend, first->prompt, first->outputs, false);
        expect(off.reused == 0 && off.tokens == first->reference,
               title + ": a request that opts out never reuses, however full the cache (restored " +
                   std::to_string(off.reused) + ", " + difference(off.tokens, first->reference) + ")");
        expect_released(*cache, *rig.frontend, title + ": after the opt-out");
        cache.reset();
        restored = 0;
    };
    for (const Step& step : steps) {
        if (!cache || step.group != first->group) {
            close();
            cache = rig.program(kContext, kChunk, host_bytes_from_env(), kv, qk);
            first = &step;
        }
        // A step that must resume is preceded by a scrub: an unrelated 1,600-token request (prefix reuse off,
        // so it publishes and reuses nothing) that overwrites the Program's global pages and every ring page.
        // Without it the lane would still hold the very data a restore should bring back (the step's own
        // earlier prompt), and a broken or no-op restore would pass unnoticed.
        if (step.restore > 0) {
            const Outcome scrub = run(*cache, *rig.frontend, tokens(1600, 9000 + scrubs++), 2, false);
            expect(scrub.reused == 0, title + ": " + step.name + ": the scrub restored nothing (restored " +
                                          std::to_string(scrub.reused) + ")");
        }
        // The cache and the sources the Program offers just before the step, for a mismatch message.
        const std::string before = describe_stats(cache->hybrid_stats()) + "; " +
                                   describe_sources(*cache, *rig.frontend, step.prompt);
        const Outcome out = run(*cache, *rig.frontend, step.prompt, step.outputs, true);
        restored += out.reused;
        expect(out.tokens == step.reference, title + ": " + step.name +
                                                 ": generates exactly what the cache-off prefill of the same prompt does (" +
                                                 difference(out.tokens, step.reference) + ")");
        expect(out.reused % kChunk == 0 && out.reused < step.prompt.size(),
               title + ": " + step.name + ": the restore lands on the chunk grid below the last prompt token (restored " +
                   std::to_string(out.reused) + ", prompt " + std::to_string(step.prompt.size()) + ", chunk " +
                   std::to_string(kChunk) + ")");
        const bool exact = out.reused == step.restore;
        expect(exact, title + ": " + step.name + ": restored " + std::to_string(out.reused) + " of the expected " +
                          std::to_string(step.restore) + " (source quoted " + std::to_string(out.quoted) + ")" +
                          (exact ? std::string()
                                 : " | before the step: " + before + " | after: " + describe_stats(cache->hybrid_stats())));
        expect_released(*cache, *rig.frontend, title + ": " + step.name);
    }
    close();
}

void verify(Rig& rig, const std::string& title, void (*script)(Builder&), infernix::KvCacheStorage kv,
            ops::Gemma4QkMode qk) {
    std::printf("== %s\n", title.c_str());
    std::vector<Step> steps;
    {
        const auto cold = rig.program(kContext, kChunk, 0, kv, qk);
        Builder builder{*cold, *rig.frontend, {}};
        script(builder);
        steps = std::move(builder.steps);
        const auto stats = cold->hybrid_stats();
        expect(stats.nodes == 0 && stats.snapshots == 0 && stats.host_slabs == 0 && stats.reused_tokens == 0,
               title + ": a cache-off lane holds and reports nothing (nodes " + std::to_string(stats.nodes) +
                   ", snapshots " + std::to_string(stats.snapshots) + ", Host slabs " + std::to_string(stats.host_slabs) +
                   ", restored " + std::to_string(stats.reused_tokens) + ")");
    }
    pass(rig, title, steps, kv, qk);
}

// The prompts that fill the tier.
Tokens fill_prompt(int index) { return tokens(900 + 5 * index, 1000 + index); }

// A saturated Host tier, pinned honestly. Distinct prompts fill the tier until it evicts a snapshot; the
// cache then values a new tap against what it would displace, so one that saves little may be declined (the
// tier's value policy working as designed). Either the new prompt's grid tap is published and the extension
// restores exactly it, or the tap is reported skipped and the extension starts at the root; in both cases the
// outputs equal the cache-off prefill exactly and the lane and counters are at baseline.
void saturated_tier(Rig& rig) {
    const std::string title = "saturated Host tier";
    std::printf("== %s\n", title.c_str());
    const Tokens a          = tokens(1000, 801);
    const std::uint32_t tap = grid_tap(a.size()); // 896
    Tokens reply, b, ref_b;
    {
        const auto cold = rig.program(kContext, kChunk, 0, kBf16, ops::Gemma4QkMode::BFloat16);
        reply = run(*cold, *rig.frontend, a, kReply, true).tokens;
        b     = concat(concat(a, reply), tokens(70, 802));
        ref_b = run(*cold, *rig.frontend, b, 8, true).tokens;
    }
    auto cache = rig.program(kContext, kChunk, host_bytes_from_env(), kBf16, ops::Gemma4QkMode::BFloat16);
    int fills = 0;
    while (cache->hybrid_stats().host_snapshot_evictions == 0 && fills < 60) {
        (void)run(*cache, *rig.frontend, fill_prompt(fills), 8, true);
        ++fills;
    }
    expect(cache->hybrid_stats().host_snapshot_evictions > 0,
           title + ": distinct prompts filled the tier until it evicted (" + std::to_string(fills) + " prompts; " +
               describe_stats(cache->hybrid_stats()) + ")");
    expect_released(*cache, *rig.frontend, title + ": after filling");
    const auto before = cache->hybrid_stats();
    const Outcome first = run(*cache, *rig.frontend, a, kReply, true);
    expect(first.reused == 0 && first.tokens == reply,
           title + ": the new prompt restores nothing and generates exactly what the cache-off prefill does (restored " +
               std::to_string(first.reused) + ", " + difference(first.tokens, reply) + ")");
    expect_released(*cache, *rig.frontend, title + ": after the new prompt");
    const auto published = cache->hybrid_stats();
    const Outcome scrub  = run(*cache, *rig.frontend, tokens(1600, 9900), 2, false);
    expect(scrub.reused == 0, title + ": the scrub restored nothing (restored " + std::to_string(scrub.reused) + ")");
    const Outcome second = run(*cache, *rig.frontend, b, 8, true);
    const auto after     = cache->hybrid_stats();
    const bool skipped   = published.taps_skipped > before.taps_skipped;
    expect(second.reused == tap || (skipped && second.reused == 0),
           title + ": the extension resumes at the new prompt's grid tap, or at the root when the tap was reported "
                   "skipped (restored " + std::to_string(second.reused) + " of " + std::to_string(tap) + ", taps skipped " +
               std::to_string(before.taps_skipped) + " -> " + std::to_string(published.taps_skipped) + ")");
    expect(second.reused % kChunk == 0 && second.reused < b.size(),
           title + ": the restore lands on the chunk grid below the last prompt token (restored " +
               std::to_string(second.reused) + ", prompt " + std::to_string(b.size()) + ")");
    expect(second.tokens == ref_b, title + ": the extension generates exactly what the cache-off prefill does (" +
                                       difference(second.tokens, ref_b) + ")");
    expect(after.reused_tokens - before.reused_tokens == first.reused + second.reused &&
               after.host_free_slabs <= after.host_slabs,
           title + ": counters stay consistent (restored " + std::to_string(after.reused_tokens - before.reused_tokens) +
               " of " + std::to_string(first.reused + second.reused) + "; " + describe_stats(after) + ")");
    expect_released(*cache, *rig.frontend, title + ": after the extension");
}

// Returns the Host slab size of a BF16 cache (the Engine's KV format), for the Engine's capacity check.
std::uint64_t program_scenarios(const char* artifact) {
    Rig rig(artifact, kContext);
    verify(rig, "frontiers on the chunk grid around pages and the ring wrap (a fresh cache per first prompt)",
           frontier_script, kBf16, ops::Gemma4QkMode::BFloat16);
    verify(rig, "conversations (displacement, branch, return)", conversation_script, kBf16, ops::Gemma4QkMode::BFloat16);
    saturated_tier(rig);
    verify(rig, "identity BF16 KV", identity_script, kBf16, ops::Gemma4QkMode::BFloat16);
    verify(rig, "identity FP8 KV / BF16 QK", identity_script, kFp8, ops::Gemma4QkMode::BFloat16);
    verify(rig, "identity FP8 KV / FP8 QK", identity_script, kFp8, ops::Gemma4QkMode::Fp8);
    // The Host tier is the whole slabs that fit the budget.
    const std::uint64_t host = host_bytes_from_env();
    const auto stats = rig.program(kContext, kChunk, host, kBf16, ops::Gemma4QkMode::BFloat16)->hybrid_stats();
    const std::uint64_t capacity = stats.host_slab_bytes * stats.host_slabs;
    expect(stats.host_slab_bytes > 0 && capacity <= host && host - capacity < stats.host_slab_bytes,
           "a Program's Host tier is the whole slabs its budget holds (capacity " + std::to_string(capacity) + " of " +
               std::to_string(host) + " bytes, " + std::to_string(stats.host_slabs) + " slabs of " +
               std::to_string(stats.host_slab_bytes) + ")");
    return stats.host_slab_bytes;
}

// ---- chat through the Engine -------------------------------------------------------------------

void add_message(infernix::PromptInput& input, infernix::ChatRole role, std::string text) {
    infernix::ChatMessage message;
    message.role = role;
    message.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
}

std::string numbered(const std::string& topic, int lines, int salt) {
    std::string text;
    for (int line = 0; line < lines; ++line) {
        text += topic + " note " + std::to_string(salt * 1000 + line) +
                ": keep answers short, friendly and spoken; never read out lists longer than three items.\n";
    }
    return text;
}

// A long system turn with tools (about 2,000 tokens: past the 1,152-token ring) and a user turn.
infernix::PromptInput voice(const std::string& user, const std::string& assistant = {}, const std::string& next = {}) {
    infernix::PromptInput input;
    add_message(input, infernix::ChatRole::System, "You are Juno, a voice assistant. Reply in one spoken sentence.\n" + numbered("Style", 70, 1));
    add_message(input, infernix::ChatRole::User, user);
    if (!assistant.empty()) {
        add_message(input, infernix::ChatRole::Assistant, assistant);
        add_message(input, infernix::ChatRole::User, next);
    }
    input.options.enable_thinking = false;
    input.options.tool_jsons      = {
        R"({"type":"function","function":{"name":"get_time","description":"Current local time.","parameters":{"type":"object","properties":{}}}})"};
    return input;
}

infernix::PromptInput side(const std::string& user) {
    infernix::PromptInput input;
    add_message(input, infernix::ChatRole::System, "You summarize documents in one sentence.");
    add_message(input, infernix::ChatRole::User, user);
    input.options.enable_thinking = false;
    return input;
}

Served chat(infernix::Engine& engine, const infernix::PromptInput& input, std::uint32_t outputs, bool reuse) {
    return serve(engine.generate(engine.prepare(input), greedy(outputs, reuse, true)));
}

std::size_t common_prefix(const Tokens& a, const Tokens& b) {
    std::size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) { ++n; }
    return n;
}

std::string listed(const std::vector<std::uint32_t>& values) {
    std::string out;
    for (const std::uint32_t value : values) { out += (out.empty() ? "" : ",") + std::to_string(value); }
    return out.empty() ? "none" : out;
}

// A tap hint of a chat prompt: its position and whether it carries the generation opener.
struct Tap {
    std::uint32_t position = 0;
    bool opener = false;
};

// The grid snapshots a chat request of `prompt` tokens leaves when it prefilled from the grid frontier
// `base`: its deepest grid state D(prompt) whenever that lies past the base, and its hint taps, each
// realized at the first prefill call boundary at or past it and only before the prompt's final call (at most
// prompt - 1). A hint at or before the base serves nothing new; one less than 64 tokens
// (kMinimumTapSeparation) past it is dropped unless it carries the opener, which needs 16
// (kMinimumForwardTokens).
std::vector<std::uint32_t> grid_snapshots(std::size_t prompt, std::uint32_t base, const std::vector<Tap>& taps) {
    std::vector<std::uint32_t> out;
    if (grid_tap(prompt) > base) { out.push_back(grid_tap(prompt)); }
    for (const Tap& tap : taps) {
        const std::uint32_t at = grid_up(tap.position);
        if (tap.position <= base || tap.position - base < (tap.opener ? 16U : 64U) || at + 1U > prompt) { continue; }
        if (std::find(out.begin(), out.end(), at) == out.end()) { out.push_back(at); }
    }
    return out;
}

// An earlier request's rendered ids and the grid snapshots it left.
struct Lineage {
    std::string name;
    Tokens ids;
    std::vector<std::uint32_t> snapshots;
};

// The deepest snapshot of the earlier lineages that a prompt can restore: on its shared prefix and below its
// last token.
std::uint32_t deepest_restore(const std::vector<Lineage>& lineages, const Tokens& ids) {
    std::uint32_t best = 0;
    for (const Lineage& lineage : lineages) {
        const std::size_t reach = std::min(common_prefix(lineage.ids, ids), ids.size() - 1);
        for (const std::uint32_t frontier : lineage.snapshots) {
            if (frontier <= reach) { best = std::max(best, frontier); }
        }
    }
    return best;
}

std::string describe_lineages(const std::vector<Lineage>& lineages, const Tokens& ids) {
    std::string out;
    for (const Lineage& lineage : lineages) {
        out += (out.empty() ? "" : "; ") + lineage.name + ": shares " + std::to_string(common_prefix(lineage.ids, ids)) +
               " tokens, snapshots " + listed(lineage.snapshots);
    }
    return out.empty() ? "no earlier lineage" : out;
}

void chat_scenarios(const char* artifact, std::uint64_t slab_bytes) {
    std::printf("== chat through the Engine\n");
    const std::uint64_t host = host_bytes_from_env();
    infernix::Engine engine = start_engine(engine_options(artifact, kChunk, host, 3072));
    const auto baseline = engine.runtime_stats();
    const Tokens turn_open = engine.tokenize_text("<|turn>");
    require(turn_open.size() == 1, "the chat template's turn opener is one token (" + std::to_string(turn_open.size()) + ")");
    const auto ids_of = [&](const infernix::PromptInput& input) {
        const infernix::PreparedPrompt prepared = engine.prepare(input);
        const auto ids = prepared.token_ids();
        return Tokens(ids.begin(), ids.end());
    };
    const auto starts_of = [&](const Tokens& ids) {
        std::vector<std::uint32_t> starts;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == turn_open[0]) { starts.push_back(static_cast<std::uint32_t>(i)); }
        }
        return starts;
    };
    std::uint64_t served_reused = 0;
    std::vector<Lineage> lineages;
    // The cache-off reference of a prompt on the same Engine (same chunk, KV and QK).
    const auto cold = [&](const std::string& what, const infernix::PromptInput& input) {
        const Served out = chat(engine, input, 32, false);
        served_reused += out.reused;
        expect(out.reused == 0, what + ": with prefix reuse off nothing is restored (restored " + std::to_string(out.reused) + ")");
        return out;
    };
    // A cached turn generates exactly its cache-off reference and restores exactly `expected`.
    const auto check = [&](const std::string& what, const Served& s, const Tokens& ids, const Served& reference,
                           std::uint32_t expected) {
        served_reused += s.reused;
        expect(s.tokens == reference.tokens,
               what + ": generates exactly what the cache-off prefill does (" + difference(s.tokens, reference.tokens) + ")");
        expect(s.prompt == ids.size() && s.reused % kChunk == 0 && s.reused < ids.size(),
               what + ": the restore lands on the chunk grid below the last prompt token (restored " +
                   std::to_string(s.reused) + ", prompt " + std::to_string(s.prompt) + ", rendered " +
                   std::to_string(ids.size()) + ")");
        expect(s.reused == expected, what + ": restored " + std::to_string(s.reused) + " of the expected " +
                                         std::to_string(expected) + " (" + describe_lineages(lineages, ids) + ")");
    };

    // The user turn opens with long notes (about 270 tokens), so the structural tap's grid point (the first
    // chunk boundary past the system turn) lies inside them. The edited question diverges past it, usually
    // before the first turn's D(P): without the notes the edit would usually share no grid snapshot and restore
    // 0. The notes also keep a repeat positive when the first turn ends on the grid (its D(P) = P then lies at
    // its own last token).
    const std::string notes = numbered("Pantry", 12, 3);
    const auto user = [&](const std::string& question) { return "Here is what I have at home:\n" + notes + question; };

    const infernix::PromptInput first_input = voice(user("What should I cook tonight?"));
    const Tokens ids1 = ids_of(first_input);
    const auto s1     = starts_of(ids1);
    require(s1.size() == 3 && s1[2] - s1[1] >= 64 && ids1.size() - 1 - s1[2] < 64,
            "the first turn renders system, user and opener turns, the user turn 64 tokens or longer and the opener "
            "within 64 of the end (turn starts " + listed(s1) + ", prompt " + std::to_string(ids1.size()) + ")");
    const Served v1      = chat(engine, first_input, 32, true);
    const Served v1_cold = cold("the first turn", first_input);
    check("the cold first turn", v1, ids1, v1_cold, 0);
    require(!v1.content.empty(), "the first turn answered nothing; the later turns echo its reply");
    lineages.push_back({"first turn", ids1, grid_snapshots(ids1.size(), 0, {{s1[1], false}, {s1[2], true}})});

    const std::uint32_t again_restore = deepest_restore(lineages, ids1);
    require(again_restore > 0, "the first turn left a grid snapshot (" + describe_lineages(lineages, ids1) + ")");
    check("a repeat of the first turn", chat(engine, first_input, 32, true), ids1, v1_cold, again_restore);
    settled(engine, "after the repeat");

    const infernix::PromptInput side_input = side(numbered("Report", 40, 2));
    const Tokens ids_side = ids_of(side_input);
    const auto s_side     = starts_of(ids_side);
    require(s_side.size() == 3 && s_side[2] - s_side[1] >= 64 && ids_side.size() - 1 - s_side[2] < 64,
            "the side request renders system, user and opener turns (turn starts " + listed(s_side) + ", prompt " +
                std::to_string(ids_side.size()) + ")");
    const Served summary      = chat(engine, side_input, 32, true);
    const Served summary_cold = cold("the side request", side_input);
    check("an unrelated side request", summary, ids_side, summary_cold, deepest_restore(lineages, ids_side));
    lineages.push_back({"side request", ids_side,
                        grid_snapshots(ids_side.size(), summary.reused, {{s_side[1], false}, {s_side[2], true}})});

    // The follow-up turn: its reply was decoded, never published, so it shares the first turn's snapshots only.
    const infernix::PromptInput second = voice(user("What should I cook tonight?"), v1.content, "Something quicker, please.");
    const Tokens ids2 = ids_of(second);
    const auto s2     = starts_of(ids2);
    require(s2.size() == 5 && s2[3] - s2[1] >= 64 && s2[4] - s2[3] < 64 && ids2.size() - 1 - s2[3] < 64,
            "the follow-up renders five turns, its short last user turn and opener one tap (turn starts " + listed(s2) +
                ", prompt " + std::to_string(ids2.size()) + ")");
    const std::uint32_t v2_restore = deepest_restore(lineages, ids2);
    require(v2_restore > 0, "the follow-up shares a grid snapshot of the first turn (" + describe_lineages(lineages, ids2) + ")");
    const Served v2      = chat(engine, second, 32, true);
    const Served v2_cold = cold("the follow-up turn", second);
    check("the follow-up turn, displaced by the side request", v2, ids2, v2_cold, v2_restore);
    // The last user turn and the opener are one tap cluster (less than 64 tokens apart), kept at the user turn.
    lineages.push_back({"follow-up turn", ids2, grid_snapshots(ids2.size(), v2.reused, {{s2[1], false}, {s2[3], true}})});

    check("the repeated side request", chat(engine, side_input, 32, true), ids_side, summary_cold,
          deepest_restore(lineages, ids_side));

    const infernix::PromptInput changed = voice(user("What could I bake instead? I have flour, butter, eggs and sugar."));
    const Tokens ids3 = ids_of(changed);
    const auto s3     = starts_of(ids3);
    require(s3.size() == 3 && s3[2] - s3[1] >= 64 && ids3.size() - 1 - s3[2] < 64,
            "the edited turn renders system, user and opener turns (turn starts " + listed(s3) + ", prompt " +
                std::to_string(ids3.size()) + ")");
    const std::uint32_t v3_restore = deepest_restore(lineages, ids3);
    require(v3_restore > 0, "the edited turn shares the first turn's grid snapshot inside the notes (" +
                                describe_lineages(lineages, ids3) + ")");
    const Served v3      = chat(engine, changed, 32, true);
    const Served v3_cold = cold("the edited turn", changed);
    check("an edited user question", v3, ids3, v3_cold, v3_restore);
    lineages.push_back({"edited turn", ids3, grid_snapshots(ids3.size(), v3.reused, {{s3[1], false}, {s3[2], true}})});

    check("return to the follow-up turn's branch", chat(engine, second, 32, true), ids2, v2_cold,
          deepest_restore(lineages, ids2));
    settled(engine, "after the conversations");

    const auto stats = settled(engine, "at the end");
    expect(stats.reused_prompt_tokens - baseline.reused_prompt_tokens == served_reused &&
               stats.hybrid_reused_tokens - baseline.hybrid_reused_tokens == served_reused,
           "Engine and cache counters agree with the requests (Engine " +
               std::to_string(stats.reused_prompt_tokens - baseline.reused_prompt_tokens) + ", cache " +
               std::to_string(stats.hybrid_reused_tokens - baseline.hybrid_reused_tokens) + ", requests " +
               std::to_string(served_reused) + " tokens)");
    const std::uint64_t capacity = stats.hybrid_host_capacity_bytes;
    expect(stats.hybrid_blocks_inserted > baseline.hybrid_blocks_inserted && stats.hybrid_snapshots > 0 &&
               stats.hybrid_host_used_bytes > 0 && capacity <= host && host - capacity < slab_bytes,
           "the cache reports its blocks, snapshots and Host tier in whole slabs (blocks inserted " +
               std::to_string(stats.hybrid_blocks_inserted - baseline.hybrid_blocks_inserted) + ", snapshots " +
               std::to_string(stats.hybrid_snapshots) + ", Host used " + std::to_string(stats.hybrid_host_used_bytes) +
               ", capacity " + std::to_string(capacity) + " of the budget " + std::to_string(host) + ", slab " +
               std::to_string(slab_bytes) + " bytes)");
}

void disabled_engine(const char* artifact) {
    std::printf("== cache off (--host-context-mib 0)\n");
    infernix::Engine engine = start_engine(engine_options(artifact, kChunk, 0, 2048));
    expect(engine.options().context_cache.host_capacity_bytes.value_or(1) == 0, "the Engine runs with no Host tier");
    const Tokens prompt = tokens(1500, 55);
    const Served first  = generate(engine, prompt, 16, true);
    const Served repeat = generate(engine, prompt, 16, true);
    expect(first.reused == 0 && repeat.reused == 0 && repeat.tokens == first.tokens,
           "a repeat on a cache-off Engine reuses nothing and generates the same (restored " + std::to_string(first.reused) +
               " and " + std::to_string(repeat.reused) + ", " + difference(repeat.tokens, first.tokens) + ")");
    const auto stats = settled(engine, "cache-off Engine");
    expect(stats.hybrid_tree_blocks == 0 && stats.hybrid_snapshots == 0 && stats.hybrid_blocks_inserted == 0 &&
               stats.hybrid_host_capacity_bytes == 0 && stats.reused_prompt_tokens == 0,
           "a cache-off Engine caches and reports nothing (tree blocks " + std::to_string(stats.hybrid_tree_blocks) +
               ", snapshots " + std::to_string(stats.hybrid_snapshots) + ", blocks inserted " +
               std::to_string(stats.hybrid_blocks_inserted) + ", Host capacity " +
               std::to_string(stats.hybrid_host_capacity_bytes) + ", reused " + std::to_string(stats.reused_prompt_tokens) + ")");
}

} // namespace

int main() {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set");
        return 77;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        const std::uint64_t slab_bytes = program_scenarios(artifact);
        chat_scenarios(artifact, slab_bytes);
        disabled_engine(artifact);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts(failures == 0 ? "all prefix scenarios passed" : "prefix scenario checks failed");
    return failures == 0 ? 0 : 1;
}
