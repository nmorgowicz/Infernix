#include "runtime/contract/sampling.h"

#include "models/qwen3_5/frontend/frontend.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool same_preset(const infernix::SamplingPreset& actual, const infernix::SamplingPreset& expected) {
    return actual.temperature == expected.temperature && actual.top_k == expected.top_k &&
           actual.top_p == expected.top_p && actual.min_p == expected.min_p &&
           actual.presence_penalty == expected.presence_penalty &&
           actual.frequency_penalty == expected.frequency_penalty;
}

bool throws_invalid(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}


} // namespace

int main() {
    using infernix::models::Architecture;
    using infernix::models::qwen3_5::default_sampling;
    int failures     = 0;
    const auto dense = default_sampling(Architecture::Qwen3_5);
    const auto moe   = default_sampling(Architecture::Qwen3_5Moe);

    const infernix::SamplingPreset dense_thinking{
        .temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .min_p = 0.0F};
    const infernix::SamplingPreset dense_non_thinking{
        .temperature      = 0.7F,
        .top_k            = 20,
        .top_p            = 0.8F,
        .min_p            = 0.0F,
        .presence_penalty = 1.5F,
    };
    const infernix::SamplingPreset moe_thinking{
        .temperature      = 1.0F,
        .top_k            = 20,
        .top_p            = 0.95F,
        .min_p            = 0.0F,
        .presence_penalty = 1.5F,
    };

    failures +=
        check(same_preset(dense.thinking, dense_thinking), "Dense thinking defaults mismatch");
    failures += check(same_preset(dense.non_thinking, dense_non_thinking),
                      "Dense non-thinking defaults mismatch");
    failures += check(same_preset(moe.thinking, moe_thinking) &&
                          same_preset(moe.non_thinking, dense_non_thinking),
                      "MoE mode defaults mismatch");

    const infernix::ResolvedSamplingParameters thinking = infernix::runtime::resolve_sampling(
        dense, infernix::SamplingMode::Thinking, infernix::SamplingOverrides{}, false);
    const infernix::ResolvedSamplingParameters non_thinking = infernix::runtime::resolve_sampling(
        dense, infernix::SamplingMode::NonThinking, infernix::SamplingOverrides{}, false);
    failures += check(thinking.temperature == 1.0F && thinking.top_p == 0.95F &&
                          thinking.presence_penalty == 0.0F && thinking.seed == 0,
                      "omitted overrides did not select Dense thinking defaults");
    failures += check(non_thinking.temperature == 0.7F && non_thinking.top_p == 0.8F &&
                          non_thinking.presence_penalty == 1.5F,
                      "omitted overrides did not select Dense non-thinking defaults");

    // A content constraint drops the preset penalties (they would steer sampling to legal tokens
    // that alter the content); every other preset field and an explicit penalty still apply.
    const infernix::ResolvedSamplingParameters constrained = infernix::runtime::resolve_sampling(
        moe, infernix::SamplingMode::Thinking, infernix::SamplingOverrides{}, true);
    failures += check(constrained.temperature == 1.0F && constrained.top_p == 0.95F &&
                          constrained.presence_penalty == 0.0F && constrained.frequency_penalty == 0.0F,
                      "a content constraint kept the preset penalties");
    infernix::SamplingOverrides penalized;
    penalized.presence_penalty = 1.25F;
    failures += check(infernix::runtime::resolve_sampling(dense, infernix::SamplingMode::NonThinking, penalized,
                                                          true)
                              .presence_penalty == 1.25F,
                      "a content constraint dropped an explicit presence penalty");

    infernix::SamplingOverrides overrides;
    overrides.temperature       = 0.0F;
    overrides.top_k             = 0;
    overrides.top_p             = 0.0F;
    overrides.min_p             = 0.0F;
    overrides.presence_penalty  = 0.0F;
    overrides.frequency_penalty = -1.0F;
    overrides.seed              = 123;
    const infernix::ResolvedSamplingParameters overridden =
        infernix::runtime::resolve_sampling(dense, infernix::SamplingMode::NonThinking, overrides, false);
    failures += check(overridden.temperature == 0.0F && overridden.top_k == 20 &&
                          overridden.top_p == 0.0F && overridden.presence_penalty == 0.0F &&
                          overridden.frequency_penalty == -1.0F && overridden.seed == 123,
                      "explicit zero sampling overrides were not normalized to the target cap");

    // A top_k above the model's ceiling is clamped to it, never refused; an in-range value and the
    // zero default are untouched; a negative value is still refused.
    infernix::ModelSamplingDefaults wide;
    wide.thinking = wide.non_thinking = {.temperature = 1.0F, .top_k = 64, .top_p = 0.95F};
    wide.max_top_k                    = 64;
    const auto resolved_top_k         = [](const infernix::ModelSamplingDefaults& model, int top_k) {
        infernix::SamplingOverrides request;
        request.top_k = top_k;
        return infernix::runtime::resolve_sampling(model, infernix::SamplingMode::Thinking, request, false)
            .top_k;
    };
    failures += check(resolved_top_k(dense, 1) == 1 && resolved_top_k(dense, 20) == 20 &&
                          resolved_top_k(dense, 21) == 20 && resolved_top_k(dense, 100) == 20,
                      "a Qwen top_k was not passed through in range and clamped to 20 above it");
    failures += check(resolved_top_k(wide, 1) == 1 && resolved_top_k(wide, 40) == 40 &&
                          resolved_top_k(wide, 64) == 64 && resolved_top_k(wide, 65) == 64 &&
                          resolved_top_k(wide, 100) == 64 && resolved_top_k(wide, 500) == 64 &&
                          resolved_top_k(wide, std::numeric_limits<std::int32_t>::max()) == 64,
                      "a Gemma top_k was not passed through in range and clamped to 64 above it");
    failures += check(resolved_top_k(wide, 0) == 64 && resolved_top_k(dense, 0) == 20,
                      "top_k 0 no longer resolves to the model's default candidate width");
    failures += check(throws_invalid([&] { (void)resolved_top_k(wide, -1); }),
                      "a negative top_k was accepted");
    overrides.top_k = 0;

    // Temperature and the presence/frequency penalties accept their documented bounds exactly and
    // refuse anything beyond them; the sampler takes the same values for every model.
    const auto resolved_with = [&](infernix::SamplingOverrides request) {
        return infernix::runtime::resolve_sampling(wide, infernix::SamplingMode::Thinking, request, false);
    };
    for (const float temperature : {0.0F, 1.0F, 2.0F, 2.5F, 5.0F, infernix::kMaxSamplingTemperature}) {
        infernix::SamplingOverrides request;
        request.temperature = temperature;
        failures += check(resolved_with(request).temperature == temperature,
                          "a temperature inside [0, kMaxSamplingTemperature] was not preserved");
    }
    for (const float temperature : {-0.01F, 10.01F, 100.0F}) {
        infernix::SamplingOverrides request;
        request.temperature = temperature;
        failures += check(throws_invalid([&] { (void)resolved_with(request); }),
                          "a temperature outside [0, kMaxSamplingTemperature] was accepted");
    }
    for (const float penalty : {-infernix::kMaxSamplingPenalty, -2.5F, 0.0F, 3.0F, infernix::kMaxSamplingPenalty}) {
        infernix::SamplingOverrides request;
        request.presence_penalty  = penalty;
        request.frequency_penalty = -penalty;
        const auto resolved       = resolved_with(request);
        failures += check(resolved.presence_penalty == penalty && resolved.frequency_penalty == -penalty,
                          "a penalty inside +-kMaxSamplingPenalty was not preserved");
    }
    for (const float penalty : {-5.01F, 5.01F, 50.0F}) {
        infernix::SamplingOverrides presence;
        presence.presence_penalty = penalty;
        infernix::SamplingOverrides frequency;
        frequency.frequency_penalty = penalty;
        failures += check(throws_invalid([&] { (void)resolved_with(presence); }) &&
                              throws_invalid([&] { (void)resolved_with(frequency); }),
                          "a penalty outside +-kMaxSamplingPenalty was accepted");
    }

    overrides.temperature = std::numeric_limits<float>::quiet_NaN();
    failures += check(throws_invalid([&] {
                          (void)infernix::runtime::resolve_sampling(
                              dense, infernix::SamplingMode::Thinking, overrides, false);
                      }),
                      "non-finite sampling override was accepted");

    // An artifact's sampling.json (the converter's --resource sampling.json=PATH, e.g. Ornith-1.5-122B's
    // card set for both modes) replaces the architecture's presets; a malformed one is refused.
    {
        using infernix::models::qwen3_5::sampling_presets;
        const char* card = R"({"source": "card",
            "thinking": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                         "presence_penalty": 1.5, "frequency_penalty": 0.0, "repetition_penalty": 1.0},
            "non_thinking": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                             "presence_penalty": 0.0, "frequency_penalty": 0.0}})";
        const auto presets = sampling_presets(card);
        failures += check(same_preset(presets.thinking, moe_thinking), "sampling.json thinking preset mismatch");
        failures += check(same_preset(presets.non_thinking, {.temperature = 0.6F, .top_k = 20, .top_p = 0.95F}),
                          "sampling.json non-thinking preset mismatch");
        const auto resolved = infernix::runtime::resolve_sampling(presets, infernix::SamplingMode::NonThinking,
                                                                  infernix::SamplingOverrides{}, false);
        failures += check(resolved.temperature == 0.6F && resolved.presence_penalty == 0.0F,
                          "sampling.json presets did not resolve as request defaults");
        const char* missing    = R"({"thinking": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                                     "presence_penalty": 1.5, "frequency_penalty": 0.0}})";
        const char* wide_k     = R"({"thinking": {"temperature": 1.0, "top_k": 21, "top_p": 0.95, "min_p": 0.0,
                                     "presence_penalty": 1.5, "frequency_penalty": 0.0},
                                     "non_thinking": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                                     "presence_penalty": 1.5, "frequency_penalty": 0.0}})";
        const char* repetition = R"({"thinking": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                                     "presence_penalty": 1.5, "frequency_penalty": 0.0, "repetition_penalty": 1.1},
                                     "non_thinking": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
                                     "presence_penalty": 1.5, "frequency_penalty": 0.0}})";
        const char* unknown    = R"({"thinking": {}, "non_thinking": {}, "greedy": true})";
        for (const char* bad : {missing, wide_k, repetition, unknown}) {
            failures += check(throws_invalid([&] { (void)sampling_presets(bad); }),
                              "a malformed sampling.json was accepted");
        }
    }

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
