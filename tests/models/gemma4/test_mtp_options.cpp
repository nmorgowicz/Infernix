#include "core/temporary_switches.h"
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

    failures += check(!EngineOptions{}.max_context_auto && !EngineOptions{}.gemma4_key_only_kv,
                      "Gemma4 memory choices must remain opt-in");
    {
        bool removed = false;
        try { (void)infernix::temporary_switches::flag("INFERNIX_TMP_GEMMA4_KEY_ONLY_KV"); }
        catch (const std::logic_error&) { removed = true; }
        failures += check(removed, "the obsolete key-only environment switch is still registered");
    }
    {
        auto options = mtp_options();
        options.gemma4_key_only_kv = true;
        failures += refuses(options, "key-only KV accepted BF16 globals");
        options.kv_cache = infernix::KvCacheStorage::Fp8E4M3Row256;
        failures += check(normalize(options).gemma4_key_only_kv, "normalization dropped explicit key-only KV");
        options.max_context_auto = true;
        options.max_context = 131072;
        failures += refuses(options, "automatic context accepted an explicit KV capacity");
        options.kv_capacity = infernix::KvCapacityPolicy::automatic();
        failures += check(normalize(options).max_context_auto, "normalization dropped automatic context");
        options.vram_past_budget = true;
        failures += refuses(options, "automatic context allowed pageable past-budget startup");
    }
    for (const auto architecture : {infernix::models::Architecture::Qwen3_5,
                                    infernix::models::Architecture::Qwen4Exp}) {
        for (const bool automatic : {false, true}) {
            auto options = mtp_options();
            options.artifact_path = "unused.infernix";
            options.max_context_auto = automatic;
            options.gemma4_key_only_kv = !automatic;
            bool refused = false;
            try { infernix::runtime::validate_engine_options(options, architecture); }
            catch (const std::invalid_argument&) { refused = true; }
            const bool expected = !automatic || architecture != infernix::models::Architecture::Qwen3_5;
            failures += check(refused == expected, "automatic Qwen context or Gemma-only key storage was misclassified");
        }
    }
    for (const auto architecture : {infernix::models::Architecture::Qwen3_5,
                                    infernix::models::Architecture::Qwen4Exp}) {
        auto options = mtp_options();
        options.artifact_path = "unused.infernix";
        options.allow_shared_memory = true;
        bool refused = false;
        try { infernix::runtime::validate_engine_options(options, architecture); }
        catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, "a non-Gemma family accepted shared GPU memory spill");
    }
    {
        auto options = mtp_options();
        options.ablate_sidecar_path = "sidecar.json";
        failures += refuses(options, "Gemma silently accepted a Qwen ablation sidecar");
        options.ablate_sidecar_path.clear();
        options.lora_path = "adapter";
        failures += refuses(options, "Gemma silently accepted a dense Qwen LoRA adapter");
    }
    // Non-affine context-dependent reservation: workspace/table steps belong to each candidate,
    // not to one high-ceiling curve reused with a smaller KV count.
    const auto reservation = [](std::uint32_t context) {
        return std::size_t{2000} + context * 3ULL + ((context + 4095U) / 4096U) * 700ULL;
    };
    for (const auto boundary : {9216U, 12288U, 12352U, 32768U, 131072U}) {
        std::uint32_t calls = 0;
        const auto resolved = infernix::runtime::resolve_gemma4_auto_context(
            262144, reservation(boundary) + 1000, 1000,
            [&](std::uint32_t candidate) {
                ++calls;
                failures += check(candidate >= 9216 && candidate <= 131072 && candidate % 64 == 0,
                                  "auto context planned outside the page-aligned usable range");
                return reservation(candidate);
            });
        failures += check(resolved == boundary && calls <= 15,
                          "auto context did not find the exact bounded-search resident fit");
    }
    failures += check(infernix::runtime::resolve_gemma4_auto_context(
                          12345, reservation(131072), 0, reservation) == 12288,
                      "native/explicit upper ceiling must round down to a page");
    for (const auto available : {std::size_t{999}, static_cast<std::size_t>(reservation(9216) + 999)}) {
        bool refused = false;
        try { (void)infernix::runtime::resolve_gemma4_auto_context(131072, available, 1000, reservation); }
        catch (const std::runtime_error&) { refused = true; }
        failures += check(refused, "auto context shrank below 9216 or consumed required headroom");
    }
    {
        bool refused = false;
        try { (void)infernix::runtime::resolve_gemma4_auto_context(9215, reservation(131072), 0, reservation); }
        catch (const std::runtime_error&) { refused = true; }
        failures += check(refused, "auto context accepted an artifact ceiling below the usable minimum");
    }

    // --allow-shared-memory: a start that cannot fit resident takes the usable minimum instead of refusing,
    // while anything that does fit keeps its exact resident size and a ceiling below the minimum still refuses.
    for (const auto available : {std::size_t{999}, static_cast<std::size_t>(reservation(9216) + 999)}) {
        failures += check(infernix::runtime::resolve_gemma4_auto_context(131072, available, 1000, reservation, true) == 9216,
                          "shared-memory auto context did not fall back to the usable minimum");
    }
    failures += check(infernix::runtime::resolve_gemma4_auto_context(
                          262144, reservation(32768) + 1000, 1000, reservation, true) == 32768,
                      "shared-memory auto context shrank a context that fits resident");
    {
        bool refused = false;
        try { (void)infernix::runtime::resolve_gemma4_auto_context(9215, 0, 0, reservation, true); }
        catch (const std::runtime_error&) { refused = true; }
        failures += check(refused, "shared-memory auto context accepted a ceiling below the usable minimum");
    }
    {
        infernix::runtime::SequenceCapacityCurve curve;
        curve.main_page_tokens = 64;
        curve.minimum_main_page_groups = 1;
        curve.maximum_main_page_groups = 4096;
        curve.minimum_device_reservation_bytes = 5000;
        curve.bytes_per_additional_main_page_group = 100;
        const auto needed = curve.reservation_bytes(144) + 77;
        failures += check(infernix::runtime::gemma4_shared_memory_room(curve, 10, 9216, 77) == needed,
                          "shared-memory room did not count the reservation of the context plus headroom");
        failures += check(infernix::runtime::gemma4_shared_memory_room(curve, needed + 1, 9216, 77) == needed + 1,
                          "shared-memory room reduced free memory that already fits");
        failures += check(infernix::runtime::gemma4_shared_memory_room(curve, 0, 1U << 30, 0) ==
                              curve.reservation_bytes(4096),
                          "shared-memory room did not clamp to the curve's maximum");
    }

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
