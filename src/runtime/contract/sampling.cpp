#include "runtime/contract/sampling.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::runtime {
namespace {

// The fast sampling pipeline's candidate width: what top_k 0 means for a model that serves no wider one.
constexpr std::int32_t kFastTopK = 20;

void validate(const ResolvedSamplingParameters& sampling, std::int32_t max_top_k) {
    if (!std::isfinite(sampling.temperature) || !std::isfinite(sampling.top_p) ||
        !std::isfinite(sampling.min_p) || !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty)) {
        throw std::invalid_argument("sampling parameters must be finite");
    }
    if (sampling.temperature < 0.0F || sampling.temperature > kMaxSamplingTemperature) {
        throw std::invalid_argument("temperature must be in [0,10]");
    }
    if (sampling.top_k < 1 || sampling.top_k > max_top_k) {
        throw std::invalid_argument("resolved top_k must be in [1," + std::to_string(max_top_k) + "]");
    }
    if (sampling.top_p < 0.0F || sampling.top_p > 1.0F) {
        throw std::invalid_argument("top_p must be in [0,1]");
    }
    if (sampling.min_p < 0.0F || sampling.min_p > 1.0F) {
        throw std::invalid_argument("min_p must be in [0,1]");
    }
    if (std::fabs(sampling.presence_penalty) > kMaxSamplingPenalty) {
        throw std::invalid_argument("presence_penalty must be in [-5,5]");
    }
    if (std::fabs(sampling.frequency_penalty) > kMaxSamplingPenalty) {
        throw std::invalid_argument("frequency_penalty must be in [-5,5]");
    }
}

} // namespace

ResolvedSamplingParameters resolve_sampling(const ModelSamplingDefaults& defaults,
                                            SamplingMode mode, const SamplingOverrides& overrides,
                                            bool constrained_output) {
    SamplingPreset preset = defaults.for_mode(mode);
    if (constrained_output) {
        preset.presence_penalty  = 0.0F;
        preset.frequency_penalty = 0.0F;
    }
    ResolvedSamplingParameters resolved{
        .temperature       = overrides.temperature.value_or(preset.temperature),
        .top_k             = overrides.top_k.value_or(preset.top_k),
        .top_p             = overrides.top_p.value_or(preset.top_p),
        .min_p             = overrides.min_p.value_or(preset.min_p),
        .presence_penalty  = overrides.presence_penalty.value_or(preset.presence_penalty),
        .frequency_penalty = overrides.frequency_penalty.value_or(preset.frequency_penalty),
        .seed              = overrides.seed.value_or(0),
    };
    // Zero keeps its public meaning (use the model's default candidate width), as does -1, the
    // "disabled" value of vLLM and SillyTavern's text-completion slider. Then runtimes see
    // only concrete values: the 20-wide pipeline's cap, or for a model with a wider ceiling the
    // preset's top_k (64 for Gemma 4) when it carries one. A top_k above the model's ceiling is
    // clamped to it: front ends send sliders that reach 100 and beyond, and the ceiling is the
    // sampler's candidate width, which no request can widen.
    if (resolved.top_k == 0 || resolved.top_k == -1) {
        resolved.top_k = defaults.max_top_k > kFastTopK && preset.top_k > 0 ? preset.top_k : kFastTopK;
    }
    resolved.top_k = std::min(resolved.top_k, defaults.max_top_k);
    validate(resolved, defaults.max_top_k);
    return resolved;
}

} // namespace infernix::runtime
