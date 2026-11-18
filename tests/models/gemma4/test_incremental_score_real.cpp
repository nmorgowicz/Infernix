// Explicit operator-owned native Q5 artifact; no downloads. Match the public generation
// path, then isolate decode arithmetic with an identical all-A16 prefix across compute modes.
#include "infernix/engine.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace infernix;
void require(bool ok, const char* message) {
    if (!ok) { throw std::runtime_error(message); }
}
std::vector<TokenId> tokens(std::size_t n) {
    std::vector<TokenId> out(n);
    for (std::size_t i = 0; i < n; ++i) { out[i] = static_cast<TokenId>(1000 + (i * 997) % 30000); }
    return out;
}
EngineOptions options(const char* artifact, Gemma4Q5Prefill mode) {
    EngineOptions out;
    out.artifact_path = artifact;
    out.max_context = 512;
    out.prefill_chunk = 128;
    out.context_cache.enabled = false;
    out.kv_cache = KvCacheStorage::Fp8E4M3Row256;
    out.logits_dtype = LogitsDtype::BF16;
    out.gemma4_q5_prefill = mode;
    out.speculative.backend_auto = false;
    out.speculative.backend = SpeculativeBackend::None;
    return out;
}
}
int main() {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (!artifact) { return 77; }
    try {
        std::vector<float> a16_logits;
        for (const auto mode : {Gemma4Q5Prefill::A16, Gemma4Q5Prefill::Fp8,
                                Gemma4Q5Prefill::Int8G32, Gemma4Q5Prefill::Int8G64}) {
            auto config = options(artifact, mode);
            auto prefix = tokens(129);
            std::vector<TokenId> generated;
            {
                Engine generation(config);
                RequestOptions request;
                request.execution.requested_output_tokens = 8;
                request.execution.sampling.temperature = 0;
                request.execution.sampling.top_k = 20;
                request.execution.allow_prefix_reuse = false;
                request.stop.include_model_defaults = false;
                generated = generation.generate(generation.prepare_tokens(prefix), request).generated_token_ids;
                require(generated.size() == 8, "generation did not complete eight tokens");
            }
            config.purpose = EnginePurpose::CausalScoring;
            Engine scorer(config);
            prefix.insert(prefix.end(), generated.begin(), generated.end());
            ScoreOptions selection;
            selection.top_k = 1;
            selection.incremental = true;
            selection.logits_positions = {0, 1, 7};
            const auto scored = scorer.score_tokens(prefix, 129, selection);
            require(scored.top_ids == generated, "incremental teacher forcing differs from public greedy decode");
            const auto repeat = scorer.score_tokens(prefix, 129, selection);
            require(scored.logits == repeat.logits && scored.logprobs == repeat.logprobs,
                    "incremental scoring leaked KV state between independent windows");
            // The first output is genuinely the prefill distribution, not an extra decode.
            const auto first = scorer.score_tokens(std::vector<TokenId>(prefix.begin(), prefix.begin() + 130), 129,
                                                   {.logits_positions = {0}});
            require(std::equal(first.logits.begin(), first.logits.end(), scored.logits.begin()),
                    "incremental position zero is not the unchanged final prefill distribution");
            // 31-column prefix, then single-token forwards: none can select a wide lossy route.
            // Identical tokens and identical prefix/KV state isolate the decode arithmetic itself.
            auto common = tokens(39);
            const auto narrow = scorer.score_tokens(common, 31, selection);
            if (mode == Gemma4Q5Prefill::A16) { a16_logits = narrow.logits; }
            else { require(narrow.logits == a16_logits, "compute mode changed all-A16 incremental logits"); }
            std::cout << "PASS " << gemma4_q5_prefill_name(mode)
                      << " generation/incremental alignment, reset, first-prefill state and shared-prefix A16 decode\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
