#include "runtime/engine/gemma4_instance.h"
#include "runtime/engine/model_instance.h"

#include <cstdint>
#include <exception>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

using infernix::EngineOptions;
using infernix::SpeculativeBackend;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

EngineOptions mtp_options() {
    EngineOptions options;
    options.logits_dtype = infernix::LogitsDtype::BF16;
    options.speculative.backend = SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 1;
    return options;
}

// Follow the Engine's artifact-independent Gemma option path. No model, device or artifact is opened.
EngineOptions normalize(EngineOptions options) {
    options = infernix::runtime::normalize_engine_options(
        std::move(options), infernix::models::Architecture::Gemma4);
    infernix::runtime::validate_gemma4_options(options);
    return options;
}

int refuses(EngineOptions options, const char* message) {
    try {
        (void)normalize(std::move(options));
    } catch (const std::invalid_argument&) {
        return 0;
    }
    return check(false, message);
}

int run() {
    int failures = 0;

    for (std::uint32_t width = 1; width <= 7; ++width) {
        auto options = mtp_options();
        options.speculative.draft_tokens = width;
        const auto out = normalize(options);
        failures += check(out.speculative.backend == SpeculativeBackend::Mtp &&
                              out.speculative.draft_tokens == width && !out.speculative.draft_tokens_auto,
                          "fixed Gemma4 MTP width was changed");
    }
    {
        auto options = mtp_options();
        options.speculative.backend = SpeculativeBackend::None;
        options.speculative.draft_tokens = 0;
        const auto out = normalize(options);
        failures += check(out.speculative.backend == SpeculativeBackend::None && out.speculative.draft_tokens == 0,
                          "plain Gemma4 decoding was not preserved");
    }
    {
        auto options = mtp_options();
        options.speculative.draft_tokens = 0;
        options.speculative.draft_tokens_auto = true;
        const auto out = normalize(options);
        failures += check(out.speculative.backend == SpeculativeBackend::Mtp &&
                              out.speculative.draft_tokens == 6 && !out.speculative.draft_tokens_auto,
                          "Gemma4 automatic MTP length did not resolve to fixed six");
        for (std::uint32_t ceiling = 1; ceiling <= 7; ++ceiling) {
            options.speculative.draft_tokens = ceiling;
            const auto capped = normalize(options);
            failures += check(capped.speculative.draft_tokens == ceiling && !capped.speculative.draft_tokens_auto,
                              "Gemma4 automatic MTP lost an explicit ceiling");
        }
    }
    for (const auto width : {0U, 8U}) {
        auto options = mtp_options();
        options.speculative.draft_tokens = width;
        failures += refuses(options, "Gemma4 accepted an invalid fixed MTP width");
    }
    {
        auto options = mtp_options();
        options.speculative.draft_tokens = 8;
        options.speculative.draft_tokens_auto = true;
        failures += refuses(options, "Gemma4 accepted an invalid automatic MTP ceiling");
    }
    for (const auto backend : {SpeculativeBackend::DFlash, SpeculativeBackend::DFlash2}) {
        auto options = mtp_options();
        options.speculative.backend = backend;
        failures += refuses(options, "Gemma4 accepted an unsupported backend");
    }
    {
        auto options = mtp_options();
        options.speculative.backend_auto = true;
        failures += refuses(options, "normalization accepted an unresolved automatic backend");
    }
    {
        auto options = mtp_options();
        options.speculative.proposal_head = infernix::ProposalHead::Optimized;
        failures += refuses(options, "Gemma4 accepted an unsupported proposal head");
    }
    // Even an explicitly zero confidence threshold is unsupported, not silently ignored.
    for (const float confidence : {0.0F, 0.5F}) {
        auto options = mtp_options();
        options.speculative.draft_min_p = confidence;
        failures += refuses(options, "Gemma4 accepted a draft-confidence cut");
    }
    {
        auto options = mtp_options();
        options.speculative.draft_tree_auto = true;
        failures += refuses(options, "Gemma4 accepted automatic draft trees");
        options = mtp_options();
        options.speculative.draft_tree_nodes.front() = 8;
        failures += refuses(options, "Gemma4 accepted explicit draft trees");
        options = mtp_options();
        options.speculative.draft_tree_nodes.back() = 8;
        failures += refuses(options, "Gemma4 ignored a draft tree for an inactive batch size");
    }
    {
        auto options = mtp_options();
        options.speculative.ngram_draft_tokens = 4;
        failures += refuses(options, "Gemma4 accepted ngram drafting");
        options = mtp_options();
        options.speculative.ngram_draft_tokens_auto = true;
        failures += refuses(options, "Gemma4 accepted automatic ngram drafting");
        options = mtp_options();
        options.speculative.draft_model = "unsupported-companion.safetensors";
        failures += refuses(options, "Gemma4 accepted a companion drafter");
    }
    {
        auto options = mtp_options();
        options.max_concurrency = 2;
        failures += refuses(options, "Gemma4 MTP accepted multiple lanes");
    }
    {
        auto options = mtp_options();
        options.fast_paths = false;
        failures += refuses(options, "Gemma4 MTP accepted reproducible mode");
        options = mtp_options();
        options.lossless = true;
        failures += refuses(options, "Gemma4 MTP accepted lossless mode");
        options = mtp_options();
        options.reference = true;
        failures += refuses(options, "Gemma4 MTP accepted reference mode");
    }
    return failures == 0 ? 0 : 1;
}

} // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 MTP option test failed: " << error.what() << '\n';
        return 1;
    }
}
