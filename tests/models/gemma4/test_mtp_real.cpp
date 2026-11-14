// Gemma 4 MTP on a real artifact through the native Program and public Engine.
// INFERNIX_GEMMA4_ARTIFACT selects an artifact with the `mtp` component; absent means skip (77).
// INFERNIX_GEMMA4_KV=fp8 runs FP8 global KV (default bf16); INFERNIX_GEMMA4_QK=fp8 adds FP8 QK.
//
// 1. Greedy: MTP at 1 and 5 drafts gives plain decode's tokens, except where the two schedules round a
//    near tie differently (the verify forward is K+1 columns wide, plain decode one): a divergence
//    passes only when the ORIGINAL plain reference's candidate gap is <=0.125; no scoring replay.
// 2. Sampled: the first three generated tokens have plain decode's distribution (300 seeds each,
//    temperature 1.5, top_k 20): the three most common openings match within four standard errors.
//    A seed repeats its tokens, another seed changes them, and drafts are accepted at temperature 1.
// 3. Prefix cache: a follow-up turn restored from the Host tier decodes with MTP exactly as the same
//    turn on a cache-off Engine does (decoded tokens are never cached; a restore is the cold state).
#include "prefix_real_support.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace prefix_real;

constexpr float kMtpTie = 0.125F;

struct Setup {
    const char* artifact = nullptr;
    infernix::KvCacheStorage kv = infernix::KvCacheStorage::BFloat16;
    infernix::ops::Gemma4QkMode qk = infernix::ops::Gemma4QkMode::BFloat16;
};

// Chunk 1024 for decode work; the prefix case uses 128 so its short turns reach the restore grid
// (restores happen only at whole prefill chunks, plan §17.6).
infernix::EngineOptions options_for(const Setup& setup, std::uint32_t drafts, std::size_t host_bytes,
                                    std::uint32_t chunk = 1024) {
    auto options = engine_options(setup.artifact, chunk, host_bytes, 4096);
    options.kv_cache = setup.kv;
    options.gemma4_qk = setup.qk;
    if (drafts != 0) {
        options.speculative.backend = infernix::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = drafts;
    }
    return options;
}

void add_message(infernix::PromptInput& input, infernix::ChatRole role, std::string text) {
    infernix::ChatMessage message;
    message.role = role;
    message.parts.push_back(infernix::MessagePart{.kind = infernix::MessagePartKind::Text, .text = std::move(text)});
    input.messages.push_back(std::move(message));
}

infernix::PromptInput chat(const std::string& user) {
    infernix::PromptInput input;
    add_message(input, infernix::ChatRole::User, user);
    input.options.enable_thinking = false;
    return input;
}

std::size_t common_prefix(const Tokens& a, const Tokens& b) {
    std::size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) { ++n; }
    return n;
}

struct GreedyTrace {
    Tokens tokens;
    std::vector<std::uint16_t> logits; // licensed columns only, in output-token order
    infernix::SpeculativeStats stats;
};

float bf16_value(std::uint16_t value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16);
}

void greedy_equality(const Setup& setup) {
    const std::vector<std::string> prompts{
        "Explain how a refrigerator keeps food cold, in two short paragraphs.",
        "Write a Python function that returns the n-th Fibonacci number iteratively.",
        "Translate 'Good morning, how are you today?' into French, German and Spanish.",
        "A train travels 180 km in 2.5 hours. What is its average speed? Show the steps.",
        "List five tips for writing clear commit messages."};
    std::vector<Tokens> prompt_ids;
    {
        infernix::Engine engine = start_engine(options_for(setup, 0, 0));
        for (const auto& prompt : prompts) {
            const infernix::PreparedPrompt prepared = engine.prepare(chat(prompt));
            const auto ids = prepared.token_ids();
            prompt_ids.emplace_back(ids.begin(), ids.end());
        }
    }
    std::vector<GreedyTrace> plain;
    // Observe selecting logits from the ORIGINAL graph/eager calls, never a fresh scoring replay:
    // a replay has another prefill schedule and cannot adjudicate a decode near tie.
    for (const std::uint32_t drafts : {0U, 1U, 5U}) {
        infernix::DeviceContext device(0);
        infernix::models::LoadOptions load;
        load.speculative = drafts ? infernix::SpeculativeBackend::Mtp : infernix::SpeculativeBackend::None;
        auto model = g4::load_model(setup.artifact, load, device);
        g4::execution::Parameters parameters(*model, device.stream);
        g4::Frontend frontend(model->resources(), {.max_context = 4096});
        std::vector<std::uint16_t> selecting;
        std::uint32_t predictor = 0, observed_width = 0;
        g4::ProgramOptions options;
        options.max_context = 4096; options.prefill_chunk = 1024;
        options.kv_cache = setup.kv; options.qk_mode = setup.qk; options.draft_tokens = drafts;
        options.logits_observer = [&](std::uint32_t first, std::uint32_t width,
                                       std::span<const std::uint16_t> rows) {
            predictor = first; observed_width = width;
            selecting.assign(rows.begin(), rows.end());
        };
        g4::Program program(parameters, device, options);
        const std::size_t vocab = model->config().text.vocab_size;
        std::uint64_t rounds = 0, accepted = 0;
        for (std::size_t index = 0; index < prompts.size(); ++index) {
            auto started = start(program, frontend, prompt_ids[index], 128, false);
            require(started.sequence.has_value(), "greedy trace did not bind");
            const auto sequence = *started.sequence;
            GreedyTrace trace;
            trace.logits.reserve(vocab * 128);
            while (trace.tokens.size() < 128) {
                const std::array sequences{sequence};
                const std::array budgets{rt::RoundBudget{128U - static_cast<std::uint32_t>(trace.tokens.size())}};
                auto batch = trace.tokens.empty() ? first_token(program, sequence) : program.decode(sequences, budgets);
                require(batch.row_failure(0) == rt::PendingRowFailure::None, "greedy trace logits failed");
                const auto count = batch.row_counts().empty() ? 1U : static_cast<std::uint32_t>(batch.row_counts()[0]);
                require(count <= observed_width && selecting.size() == observed_width * vocab,
                        "selecting-logit observer lost a licensed column");
                require(predictor + 1 == prompt_ids[index].size() + trace.tokens.size(),
                        "selecting-logit observer position mismatch");
                for (std::uint32_t c = 0; c < count; ++c) {
                    const auto row = std::span<const std::uint16_t>(selecting).subspan(c * vocab, vocab);
                    std::size_t winner = 0;
                    for (std::size_t v = 1; v < vocab; ++v) {
                        if (bf16_value(row[v]) > bf16_value(row[winner])) { winner = v; }
                    }
                    require(batch.tokens()[c] == static_cast<infernix::TokenId>(winner),
                            "greedy acceptance did not select its actual column's argmax");
                }
                trace.tokens.insert(trace.tokens.end(), batch.tokens().begin(), batch.tokens().begin() + count);
                trace.logits.insert(trace.logits.end(), selecting.begin(), selecting.begin() + count * vocab);
                rt::CommitDecision decision;
                decision.accepted_tokens = count; decision.terminal = trace.tokens.size() == 128;
                const std::array decisions{decision};
                (void)program.commit(std::move(batch), decisions);
            }
            auto finished = program.finish(sequence);
            require(finished.status == rt::ConsumeStatus::Consumed, "greedy trace did not finish");
            trace.stats = std::move(finished.speculative);
            rounds += trace.stats.rounds; accepted += trace.stats.accepted_tokens;
            if (!drafts) { plain.push_back(std::move(trace)); continue; }
            const auto& reference = plain[index];
            const auto same = common_prefix(trace.tokens, reference.tokens);
            const auto name = "d" + std::to_string(drafts) + " prompt " + std::to_string(index);
            if (trace.tokens == reference.tokens) {
                expect(true, name + " matches plain greedy");
            } else if (same < trace.tokens.size() && same < reference.tokens.size()) {
                const auto a = reference.tokens[same], b = trace.tokens[same];
                const auto offset = same * vocab;
                const float pa = bf16_value(reference.logits[offset + a]), pb = bf16_value(reference.logits[offset + b]);
                const float ma = bf16_value(trace.logits[offset + a]), mb = bf16_value(trace.logits[offset + b]);
                std::printf("%s: first divergence output %zu: plain (%d %.6f, %d %.6f), MTP (%.6f, %.6f)\n",
                            name.c_str(), same, a, pa, b, pb, ma, mb);
                expect(std::fabs(pa - pb) <= kMtpTie,
                       name + " selects within the original plain reference's near-tie envelope");
            } else { expect(false, name + " has a different output length"); }
        }
        if (drafts) { expect(rounds > 0 && accepted > 0, "d" + std::to_string(drafts) + " drafts are accepted"); }
    }
}

infernix::RequestOptions sampled(std::uint32_t outputs, std::uint64_t seed, float temperature, float top_p) {
    infernix::RequestOptions request;
    request.execution.requested_output_tokens = outputs;
    request.execution.sampling.temperature = temperature;
    request.execution.sampling.top_k = 20;
    request.execution.sampling.top_p = top_p;
    request.execution.sampling.seed = seed;
    request.execution.allow_prefix_reuse = false;
    request.stop.include_model_defaults = false;
    return request;
}

void sampled_distribution(const Setup& setup) {
    constexpr std::uint64_t kSeeds = 300;
    using Opening = std::array<infernix::TokenId, 3>;
    std::map<Opening, std::array<int, 2>> joined;
    int totals[2] = {0, 0};
    const infernix::PromptInput prompt = chat("Write the opening sentence of a fantasy novel.");
    for (const int side : {0, 1}) {
        infernix::Engine engine = start_engine(options_for(setup, side == 0 ? 0U : 5U, 0));
        for (std::uint64_t seed = 0; seed < kSeeds; ++seed) {
            const auto result = engine.generate(engine.prepare(prompt), sampled(4, seed, 1.5F, 1.0F));
            const auto& ids = result.generated_token_ids;
            if (ids.size() < 3) { continue; }
            ++joined[{ids[0], ids[1], ids[2]}][side];
            ++totals[side];
        }
        if (side == 0) { continue; }
        const infernix::PromptInput story = chat("Tell a short story about a lighthouse keeper.");
        const auto first = engine.generate(engine.prepare(story), sampled(96, 7, 1.0F, 0.95F));
        const auto again = engine.generate(engine.prepare(story), sampled(96, 7, 1.0F, 0.95F));
        const auto other = engine.generate(engine.prepare(story), sampled(96, 8, 1.0F, 0.95F));
        std::printf("sampled story: rounds %llu, drafted %llu, accepted %llu\n",
                    static_cast<unsigned long long>(first.speculative.rounds),
                    static_cast<unsigned long long>(first.speculative.drafted_tokens),
                    static_cast<unsigned long long>(first.speculative.accepted_tokens));
        expect(first.generated_token_ids == again.generated_token_ids, "mtp sampling repeats with the same seed");
        expect(first.generated_token_ids != other.generated_token_ids, "mtp sampling differs with another seed");
        expect(first.speculative.rounds > 0 && first.speculative.accepted_tokens > 0,
               "mtp drafts are accepted at temperature 1");
    }
    expect(totals[0] > 0 && totals[1] > 0 && joined.size() >= 8, "sampled runs produce varied openings");
    std::vector<std::pair<int, std::array<int, 2>>> common;
    for (const auto& [key, counts] : joined) { common.push_back({counts[0] + counts[1], counts}); }
    std::sort(common.begin(), common.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t index = 0; index < std::min<std::size_t>(3, common.size()); ++index) {
        const double plain = static_cast<double>(common[index].second[0]) / totals[0];
        const double mtp = static_cast<double>(common[index].second[1]) / totals[1];
        const double pool = (plain + mtp) / 2.0;
        const double error = std::sqrt(std::max(pool * (1.0 - pool), 1.0e-4) * (1.0 / totals[0] + 1.0 / totals[1]));
        std::printf("opening %zu: plain %.4f, mtp %.4f\n", index, plain, mtp);
        expect(std::fabs(plain - mtp) <= 4.0 * error, "mtp matches plain frequency of opening " + std::to_string(index));
    }
}

void constrained_fallback(const Setup& setup) {
    infernix::Engine engine = start_engine(options_for(setup, 7, 0));
    const std::string forced = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu";
    auto request = sampled(128, 93, 1.0F, 0.95F);
    request.execution.sampling.top_k = 64;
    request.constraint = infernix::OutputConstraint::choice({forced});
    request.stop.include_model_defaults = true;
    const auto result = engine.generate(engine.prepare(chat("Continue with the constrained output.")), request);
    expect(result.content == forced, "MTP fallback obeys the output grammar");
    expect(result.speculative.enabled && result.speculative.rounds == 0 && result.speculative.drafted_tokens == 0 &&
               result.speculative.fallback_steps > 0,
           "constrained MTP uses plain decode and reports fallback steps");
}

void prefix_resume(const Setup& setup) {
    // A first turn of several hundred tokens, so its prefill publishes grid snapshots the follow-up restores.
    std::string notes = "Describe a quiet harbour town at dawn in three sentences, using these notes:\n";
    for (int i = 1; i <= 40; ++i) {
        notes += "Note " + std::to_string(i) + ": boat " + std::to_string(i * 7) + " is moored by pier " +
                 std::to_string(i % 5 + 1) + " and its owner mends nets.\n";
    }
    const infernix::PromptInput first = chat(notes);
    Tokens cached_turn, cold_turn;
    std::uint32_t reused = 0;
    {
        infernix::Engine engine = start_engine(options_for(setup, 1, 8192ULL << 20, 128));
        const auto reply = engine.generate(engine.prepare(first), greedy(48, true, true));
        infernix::PromptInput second = first;
        add_message(second, infernix::ChatRole::Assistant, reply.content);
        add_message(second, infernix::ChatRole::User, "Now describe the same town at midnight.");
        const auto result = engine.generate(engine.prepare(second), greedy(64, true, true));
        cached_turn = result.generated_token_ids;
        reused = result.reused_prompt_tokens;
        const infernix::PreparedPrompt prepared = engine.prepare(second);
        const auto ids = prepared.token_ids();
        cold_turn.assign(ids.begin(), ids.end());
    }
    expect(reused > 0 && reused % 128 == 0,
           "the follow-up turn restores from the Host tier on the chunk grid (reused " + std::to_string(reused) + ")");
    infernix::Engine engine = start_engine(options_for(setup, 1, 0, 128));
    const auto cold = engine.generate(engine.prepare_tokens(cold_turn), greedy(64, false, true));
    expect(cold.generated_token_ids == cached_turn, "a restored MTP turn equals the cache-off MTP turn");
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
    try {
        greedy_equality(setup);
        sampled_distribution(setup);
        constrained_fallback(setup);
        prefix_resume(setup);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    if (failures != 0) {
        std::printf("%d MTP check(s) failed\n", failures);
        return 1;
    }
    std::puts("all MTP checks passed");
    return 0;
}
