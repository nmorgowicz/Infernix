// Engine option combinations the Engine rejects before it reads any artifact.
#include "infernix/engine.h"
#include "models/gemma4/program/kv_layout.h"
#include "runtime/contract/sampling.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The exception an Engine over a nonexistent artifact raises for these options.
std::string construction_error(infernix::EngineOptions options, bool& invalid_argument,
                               const char* artifact = "absent-artifact-for-option-validation.ninfer") {
    options.artifact_path = artifact;
    try {
        infernix::Engine engine(std::move(options));
    } catch (const std::invalid_argument& error) {
        invalid_argument = true;
        return error.what();
    } catch (const std::exception& error) {
        invalid_argument = false;
        return error.what();
    }
    invalid_argument = false;
    return {};
}

} // namespace

int main() {
    int failures = 0;
    for (const std::int32_t cap : {20, 64}) {
        infernix::ModelSamplingDefaults defaults;
        defaults.max_top_k = cap;
        defaults.thinking.top_k = cap;
        defaults.non_thinking.top_k = cap;
        for (const auto mode : {infernix::SamplingMode::Thinking, infernix::SamplingMode::NonThinking}) {
            for (const std::int32_t top_k : {0, cap}) {
                const auto resolved = infernix::runtime::resolve_sampling(
                    defaults, mode, infernix::SamplingOverrides{.top_k = top_k}, false);
                failures += check(resolved.top_k == cap, "model-specific top-k cap was not retained");
            }
            bool refused = false;
            try {
                (void)infernix::runtime::resolve_sampling(
                    defaults, mode, infernix::SamplingOverrides{.top_k = cap + 1}, false);
            } catch (const std::invalid_argument&) { refused = true; }
            failures += check(refused, "runtime accepted top-k above the loaded model's ceiling");
        }
    }
    // CPU-only sizing: every width gets its own reservation for the complete rounded
    // context, and equality includes headroom. No candidate may trade away context.
    for (const std::uint32_t expected : {2048U, 1024U, 512U, 256U, 128U}) {
        std::uint32_t previous = 4096;
        const auto chunk = infernix::models::gemma4::auto_prefill_chunk(
            257, 1000 + 320 + expected, 1000,
            [&](std::uint32_t candidate, std::uint32_t capacity) {
                failures += check(candidate < previous && capacity == 320,
                                  "auto prefill must plan descending widths with the whole rounded context");
                previous = candidate;
                return static_cast<std::size_t>(capacity + candidate);
            });
        failures += check(chunk == expected, "auto prefill did not choose the largest fitting chunk");
    }
    for (const std::size_t available : {1447U, 999U}) {
        bool refused = false;
        try {
            (void)infernix::models::gemma4::auto_prefill_chunk(
                257, available, 1000,
                [](std::uint32_t chunk, std::uint32_t capacity) {
                    return static_cast<std::size_t>(capacity + chunk);
                });
        } catch (const std::runtime_error&) { refused = true; }
        failures += check(refused, "auto prefill must refuse rather than shrink context or headroom");
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "SKIP: no usable CUDA device\n";
        return failures == 0 ? 77 : 1;
    }
    infernix::EngineOptions options;
    failures += check(!options.prefill_chunk_auto && options.prefill_chunk == 0,
                      "unset Engine prefill chunk must not enable auto");
    options.prefill_chunk_auto = true;
    bool auto_invalid = false;
    const auto auto_error = construction_error(options, auto_invalid);
    failures += check(auto_invalid && auto_error.find("only for Gemma4") != std::string::npos,
                      "non-Gemma Engine accepted automatic prefill chunks");
    options.prefill_chunk_auto = false;
    // Both artifact extensions name the same container; any other is rejected before reading.
    for (const char* artifact : {"absent-artifact.infernix", "absent-artifact.ninfer"}) {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, artifact);
        failures += check(what.find("accepts only") == std::string::npos,
                          "an .infernix or .ninfer artifact path was rejected by its extension");
    }
    {
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument, "absent-artifact.gguf");
        failures += check(invalid_argument &&
                              what.find("accepts only .infernix and .ninfer") != std::string::npos,
                          "an artifact path with another extension was accepted");
    }
    // Without the opt-outs every KV format passes this check; INT8 and NVFP4 then take the fast
    // kernels.
    for (const auto storage :
         {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
          infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures += check(what.find("INT8 KV") == std::string::npos &&
                              what.find("NVFP4 KV") == std::string::npos,
                          "the default prefill kernel choice was rejected");
    }
    options.original_int8_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Fp8E4M3Row256,
                               infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool invalid_argument  = false;
        const std::string what = construction_error(options, invalid_argument);
        failures +=
            check(invalid_argument && what.find("INT8 KV") != std::string::npos,
                  "the original INT8 prefill kernel was accepted without the INT8 KV cache");
    }
    // With INT8 KV the options pass, so construction proceeds to (and fails on) the artifact.
    options.kv_cache       = infernix::KvCacheStorage::Int8Group64;
    bool invalid_argument  = false;
    const std::string what = construction_error(options, invalid_argument);
    failures += check(what.find("INT8 KV") == std::string::npos,
                      "the original INT8 prefill kernel was rejected with the INT8 KV cache");
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
                               infernix::KvCacheStorage::Fp8E4M3Row256}) {
        options.kv_cache       = storage;
        bool nvfp4_invalid     = false;
        const std::string text = construction_error(options, nvfp4_invalid);
        failures +=
            check(nvfp4_invalid && text.find("NVFP4 KV") != std::string::npos,
                  "the original NVFP4 prefill kernel was accepted without the NVFP4 KV cache");
    }
    options.kv_cache             = infernix::KvCacheStorage::Nvfp4Group16;
    bool nvfp4_invalid           = false;
    const std::string nvfp4_what = construction_error(options, nvfp4_invalid);
    failures += check(nvfp4_what.find("NVFP4 KV") == std::string::npos,
                      "the original NVFP4 prefill kernel was rejected with the NVFP4 KV cache");
    // The prompt split workspace is bounded and applies to every KV cache.
    options.original_int8_prefill_kernel  = false;
    options.original_nvfp4_prefill_kernel = false;
    options.kv_cache                      = infernix::KvCacheStorage::BFloat16;
    for (const std::uint32_t mib : {0U, infernix::kMaximumPrefillSplitWorkspaceMiB}) {
        options.prefill_split_workspace_mib = mib;
        bool split_invalid                  = false;
        const std::string text              = construction_error(options, split_invalid);
        failures += check(text.find("prefill_split_workspace_mib") == std::string::npos,
                          "a prompt split workspace within [0,16384] MiB was rejected");
    }
    options.prefill_split_workspace_mib = infernix::kMaximumPrefillSplitWorkspaceMiB + 1;
    bool split_invalid                  = false;
    const std::string split_text        = construction_error(options, split_invalid);
    failures += check(split_invalid && split_text.find("prefill_split_workspace_mib") !=
                                           std::string::npos,
                      "a prompt split workspace above 16384 MiB was accepted");
    // Fast paths are on by default and accepted with every KV cache, but the lossless route (the
    // quality reference's) refuses fast paths and qualification candidates.
    options.prefill_split_workspace_mib = infernix::kDefaultPrefillSplitWorkspaceMiB;
    failures += check(infernix::EngineOptions{}.fast_paths, "fast paths must default to on");
    options.fast_paths = true;
    for (const auto storage : {infernix::KvCacheStorage::BFloat16, infernix::KvCacheStorage::Int8Group64,
                               infernix::KvCacheStorage::Nvfp4Group16}) {
        options.kv_cache       = storage;
        bool fast_invalid      = false;
        const std::string text = construction_error(options, fast_invalid);
        failures += check(text.find("fast paths") == std::string::npos,
                          "fast paths were rejected without the lossless route");
    }
    options.lossless                = true;
    bool lossless_invalid           = false;
    const std::string lossless_text = construction_error(options, lossless_invalid);
    failures += check(lossless_invalid && lossless_text.find("fast paths") != std::string::npos,
                      "fast paths were accepted with the lossless route");
    options.fast_paths           = false;
    options.fast_path_candidates = {infernix::FastPath::DecodeSplit64};
    bool candidate_invalid       = false;
    const std::string candidate_text = construction_error(options, candidate_invalid);
    failures += check(candidate_invalid && candidate_text.find("fast paths") != std::string::npos,
                      "a fast-path candidate was accepted with the lossless route");
    options.fast_path_candidates.clear();
    // The 8-bit activation routes (--a8-activations) are approximate: the lossless route refuses them.
    options.a8_activations = true;
    bool a8_invalid        = false;
    const std::string a8_text = construction_error(options, a8_invalid);
    failures += check(a8_invalid && a8_text.find("fast paths") != std::string::npos,
                      "--a8-activations was accepted with the lossless route");
    options.a8_activations = false;
    bool exact_invalid     = false;
    const std::string exact_text = construction_error(options, exact_invalid);
    failures += check(exact_text.find("fast paths") == std::string::npos,
                      "the reproducible route was rejected with the lossless route");
    options.lossless = false;
    // --no-fast-path leaves a path out of the default route's selection: the reproducible route refuses it.
    options.fast_path_exclusions = {infernix::FastPath::PromptPv8};
    bool excluded_invalid        = false;
    const std::string excluded_text = construction_error(options, excluded_invalid);
    failures += check(excluded_invalid && excluded_text.find("--no-fast-path") != std::string::npos,
                      "a fast-path exclusion was accepted on the reproducible route");
    options.fast_path_exclusions.clear();
    // Every fast path round-trips through its name.
    for (const infernix::FastPath path : infernix::kFastPaths) {
        failures += check(infernix::parse_fast_path(infernix::fast_path_name(path)) == path,
                          "a fast path name does not round-trip");
    }
    failures += check(!infernix::parse_fast_path("unknown").has_value(),
                      "an unknown fast path name parsed");
    std::cout << (failures == 0 ? "PASS" : "FAIL") << " Engine option validation\n";
    return failures == 0 ? 0 : 1;
}
