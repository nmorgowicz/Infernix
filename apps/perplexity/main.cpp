#include "infernix_build_id.h"
#include "corpus.h"
#include "distribution.h"
#include "evaluation.h"
#include "kl.h"

#include "product/logits_options.h"
#include "product/rope_yarn_options.h"
#include "infernix/engine.h"
#include "product/logging/logging.h"
#include "product/logging/pretty_format.h"
#include "product/logging/engine_diagnostics.h"
#include "product/logging/startup_log.h"

#include <nlohmann/json.hpp>
#include <spdlog/logger.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using json  = nlohmann::json;
using infernix::perplexity::CorpusSelection;
using infernix::perplexity::DivergenceAggregate;
using infernix::perplexity::ScoreAggregate;
using infernix::perplexity::WindowPlan;

struct Options {
    bool help_requested = false;
    std::filesystem::path artifact;
    std::optional<std::filesystem::path> corpus;
    std::optional<std::filesystem::path> text;
    std::optional<std::filesystem::path> output;
    std::optional<std::filesystem::path> save_top_tokens;
    std::optional<std::filesystem::path> save_tokens;
    std::optional<std::filesystem::path> kl_reference;
    std::optional<std::filesystem::path> full_vocab_reference_write;
    std::optional<std::filesystem::path> full_vocab_reference;
    std::optional<std::string> full_vocab_reference_kv;
    std::optional<infernix::ops::Gemma4QkMode> full_vocab_reference_qk;
    std::vector<infernix::TokenId> full_vocab_prefix;
    std::uint64_t full_vocab_seed = 20261105;
    std::uint32_t full_vocab_resamples = 1000;
    std::filesystem::path expert_stripe; // offloaded experts' record tails on a second drive
    std::filesystem::path lora;          // --lora: EngineOptions::lora_path
    std::filesystem::path ablate;        // --ablate: EngineOptions::ablate_sidecar_path
    float rope_yarn_factor              = 1.0F;
    std::uint32_t context               = 4096;
    std::uint32_t stride                = 2048;
    std::uint32_t score_tail = 0;
    bool incremental = false;
    std::optional<std::uint32_t> prefill_chunk;
    int device                          = 0;
    infernix::KvCacheStorage kv           = infernix::KvCacheStorage::Fp8E4M3Row256;
    infernix::ops::Gemma4QkMode gemma4_qk = infernix::ops::Gemma4QkMode::BFloat16;
    bool gemma4_key_only_kv = false;
    infernix::KvCacheStorage gemma4_sliding_kv = infernix::KvCacheStorage::BFloat16;
    infernix::Gemma4Q5Prefill gemma4_q5_prefill = infernix::Gemma4Q5Prefill::A16;
    infernix::LogitsDtype logits_dtype    = infernix::LogitsDtype::FP32;
    bool quick                          = false;
    bool original_int8_prefill_kernel   = false;
    bool original_nvfp4_prefill_kernel  = false;
    bool lossless                       = false;
    bool reference                      = false; // the design reference (with lossless)
    bool expert_a4_activations          = false;
    bool expert_a16_activations         = false;
    bool a8_activations                 = false; // EngineOptions::a8_activations
    bool fast_paths                     = true;
    std::vector<infernix::FastPath> fast_path_candidates;
    std::vector<infernix::FastPath> fast_path_exclusions;
    std::optional<std::filesystem::path> ngram_volume;
    infernix::product::LogLevel log_level = infernix::product::LogLevel::Info;
};

std::string usage_text() {
    return "usage: infernix-perplexity <model.infernix> "
           "(--corpus <manifest.json> [--quick] | --text <utf8-file>)\n"
           "       [--context N] [--stride N] [--device N]\n"
           "       [--score-window-tail N] (score only the last N targets per unchanged window;\n"
           "        0 scores all targets, default; bounded references are a distinct protocol)\n"
           "       [--incremental] (Gemma4 only: prefill through the first target's predictor,\n"
           "        then teacher-force one token per forward with retained KV; first score is prefill)\n"
           "       [--prefill-chunk N] (tokens per scoring prefill pass, a multiple of 128;\n"
           "        default the Engine's 1024)\n"
           "       [--rope-yarn-factor F] (startup-fixed, finite [1,4], default 1; ceiling only)\n"
           "       [--kv-dtype bf16|bf16p|int8|fp8|nvfp4|k8v4|vq2|k4v2] (default fp8)\n"
           "       [--gemma4-qk bf16|fp8] (Gemma4 global-layer QK precision; fp8 needs --kv-dtype fp8)\n"
           "       [--gemma4-key-only-kv] (derive global V from FP8 K; opt-in lossy cache)\n"
           "       [--gemma4-sliding-kv bf16|fp8] (Gemma4 sliding ring K/V format; fp8 is an opt-in lossy\n"
           "        cache with one scale per row; needs --kv-dtype fp8)\n"
           "       [--gemma4-q5-prefill a16|fp8|int8-g32|int8-g64] (Gemma4 native Q5 trunk projections of 64+\n"
           "        columns; fp8 row-E4M3 bake, int8-g* exact codes with groupwise INT8 activations; lossy opt-ins)\n"
           "       [--logits-dtype fp32|bf16] (the head's logits scoring reads; default fp32, bf16\n"
           "        rounds each once, as Transformers returns them, for comparisons)\n"
           "       [--use-original-int8-prefill-kernel (int8 only; default fast kernel)]\n"
           "       [--use-original-nvfp4-prefill-kernel (nvfp4 only; default fast kernel)]\n"
           "       [--lora PATH] (Qwen3.5 dense: a LoRA adapter applied at runtime, a Hugging Face PEFT\n"
           "        adapter directory or its adapter_model.safetensors with adapter_config.json beside it)\n"
           "       [--ablate FILE] (an ablation sidecar applied at runtime, every model: its\n"
           "        sidecar.json with the direction file beside it)\n"
           "       [--lossless] (Qwen3.5/3.6/3.8 and Qwen3.8-Flash-Next: the quality reference's\n"
           "        route, every operation of the model correctly rounded; no fast path, one\n"
           "        pass per window: give --prefill-chunk the context; logits as --logits-dtype)\n"
           "       [--reference] (the design reference: --lossless with the configuration's declared\n"
           "        steps applied exactly, the activation quantization its artifact permits and\n"
           "        the --kv-dtype's codec; implies --lossless)\n"
           "       [--expert-a4-activations] (nvidia/Qwen3.8-Flash-Next-FP4-Local only: routed\n"
           "        experts with 4-bit activations, using the activation scales NVIDIA\n"
           "        calibrated for them; default 16-bit)\n"
           "       [--a16-activations] (16-bit activations where the checkpoint specifies fewer\n"
           "        bits: Flash-Next's W4A4 NVFP4 experts, NVIDIA's NVFP4 27B's NVFP4 and FP8 layers)\n"
           "       [--a8-activations] (8-bit activations where infernix-serve's option takes them;\n"
           "        not on the reproducible or lossless route)\n"
           "       [--reproducible] (score the reproducible route; by default the qualified fast\n"
           "        paths run, as in infernix-serve)\n"
           "       [--fast-path NAME]... (qualification: also run this fast path, qualified or\n"
           "        not: prompt-pv8, decode-split64)\n"
           "       [--no-fast-path NAME]... (leave-one-out: score the default route without this\n"
           "        selected fast path)\n"
           "       [--ngram-volume <file>] (Qwen3.8-Flash-Next n-gram volume; default\n"
           "        <artifact>.ngram)\n"
           "       [--output <directory>]\n"
           "       [--save-top-tokens <file>] (record each position's 64 most probable tokens)\n"
           "       [--save-tokens <file>] (record each window's input tokens as JSON, for an external\n"
           "        evaluator that writes a --kl-reference of the same windows)\n"
           "       [--kl-reference <file>] (KL divergence from a --save-top-tokens run of the\n"
           "        same corpus, context and stride, by context length)\n"
           "       [--full-vocab-reference-write <cache.nklref> | --full-vocab-reference <cache.nklref>]\n"
           "       [--full-vocab-seed N] [--full-vocab-resamples N] (exact full-vocabulary KL;\n"
           "        requires --logits-dtype bf16; default seed 20261105, resamples 1000)\n"
           "       [--full-vocab-reference-kv-dtype bf16|bf16p|int8|fp8|nvfp4|k8v4|vq2|k4v2]\n"
           "       [--full-vocab-reference-gemma4-qk bf16|fp8] (expected reference configuration;\n"
           "        defaults to this run; token/window protocol must still match exactly)\n"
           "       [--full-vocab-prefix-ids id,id,...] (prepend unscored tokens to every window;\n"
           "        supported for NLL, top-token KL and full-vocabulary KL)\n"
           "       [--expert-stripe <file>] (offloaded experts, Qwen3.8-Flash-Next and Qwen3.5-122B:\n"
           "        the expert record tails on a second drive, read beside the artifact by the SSD\n"
           "        expert tier)\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n";
}

[[noreturn]] void usage_error(std::string_view message) {
    throw std::invalid_argument(std::string(message));
}

template <class Integer>
Integer parse_integer(std::string_view text, const char* label) {
    Integer value{};
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || end != text.data() + text.size()) {
        usage_error(std::string("invalid ") + label + ": " + std::string(text));
    }
    return value;
}

Options parse_options(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        return Options{.help_requested = true};
    }
    if (argc < 2 || std::string_view(argv[1]).starts_with("--")) {
        usage_error("artifact path is required");
    }
    Options out;
    out.artifact = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string_view option = argv[i];
        const auto value              = [&](const char* label) -> std::string_view {
            if (++i >= argc) { usage_error(std::string(label) + " requires a value"); }
            return argv[i];
        };
        if (option == "--corpus") {
            out.corpus = std::filesystem::path(value("--corpus"));
        } else if (option == "--text") {
            out.text = std::filesystem::path(value("--text"));
        } else if (option == "--quick") {
            out.quick = true;
        } else if (option == "--rope-yarn-factor") {
            out.rope_yarn_factor = infernix::product::parse_rope_yarn_factor(value("--rope-yarn-factor"));
        } else if (option == "--context") {
            out.context = parse_integer<std::uint32_t>(value("--context"), "context");
        } else if (option == "--stride") {
            out.stride = parse_integer<std::uint32_t>(value("--stride"), "stride");
        } else if (option == "--score-window-tail") {
            out.score_tail = parse_integer<std::uint32_t>(value("--score-window-tail"), "score-window-tail");
        } else if (option == "--prefill-chunk") {
            out.prefill_chunk = parse_integer<std::uint32_t>(value("--prefill-chunk"), "prefill-chunk");
        } else if (option == "--device") {
            out.device = parse_integer<int>(value("--device"), "device");
        } else if (option == "--use-original-int8-prefill-kernel") {
            out.original_int8_prefill_kernel = true;
        } else if (option == "--use-original-nvfp4-prefill-kernel") {
            out.original_nvfp4_prefill_kernel = true;
        } else if (option == "--lossless") {
            out.lossless = true;
        } else if (option == "--reference") {
            out.lossless  = true;
            out.reference = true;
        } else if (option == "--expert-a4-activations") {
            out.expert_a4_activations = true;
        } else if (option == "--a16-activations") {
            out.expert_a16_activations = true;
        } else if (option == "--a8-activations") {
            out.a8_activations = true;
        } else if (option == "--ngram-volume") {
            out.ngram_volume = std::filesystem::path(value("--ngram-volume"));
        } else if (option == "--lora") {
            out.lora = std::filesystem::path(value("--lora"));
        } else if (option == "--ablate") {
            out.ablate = std::filesystem::path(value("--ablate"));
        } else if (option == "--reproducible") {
            out.fast_paths = false;
        } else if (option == "--incremental") {
            out.incremental = true;
        } else if (option == "--fast-path") {
            const std::string_view name = value("--fast-path");
            const std::optional<infernix::FastPath> path = infernix::parse_fast_path(name);
            if (!path) { usage_error("unknown fast path: " + std::string(name)); }
            out.fast_path_candidates.push_back(*path);
        } else if (option == "--no-fast-path") {
            const std::string_view name = value("--no-fast-path");
            const std::optional<infernix::FastPath> path = infernix::parse_fast_path(name);
            if (!path) { usage_error("unknown fast path: " + std::string(name)); }
            out.fast_path_exclusions.push_back(*path);
        } else if (option == "--kv-dtype") {
            const std::string_view dtype = value("--kv-dtype");
            if (dtype == "bf16") {
                out.kv = infernix::KvCacheStorage::BFloat16;
            } else if (dtype == "int8") {
                out.kv = infernix::KvCacheStorage::Int8Group64;
            } else if (dtype == "fp8") {
                out.kv = infernix::KvCacheStorage::Fp8E4M3Row256;
            } else if (dtype == "nvfp4") {
                out.kv = infernix::KvCacheStorage::Nvfp4Group16;
            } else if (dtype == "k8v4") {
                out.kv = infernix::KvCacheStorage::Fp8KeyNvfp4Value;
            } else if (dtype == "vq2") {
                out.kv = infernix::KvCacheStorage::Vq2;
            } else if (dtype == "k4v2") {
                out.kv = infernix::KvCacheStorage::Q4KeyVq2Value;
            } else if (dtype == "bf16p") {
                out.kv = infernix::KvCacheStorage::Bf16Packed;
            } else {
                usage_error("--kv-dtype must be bf16, bf16p, int8, fp8, nvfp4, k8v4, vq2, or k4v2");
            }
        } else if (option == "--gemma4-qk") {
            try {
                out.gemma4_qk = infernix::parse_gemma4_qk(value("--gemma4-qk"));
            } catch (const std::invalid_argument& error) {
                usage_error(error.what());
            }
        } else if (option == "--gemma4-key-only-kv") {
            out.gemma4_key_only_kv = true;
        } else if (option == "--gemma4-sliding-kv") {
            try {
                out.gemma4_sliding_kv = infernix::parse_gemma4_sliding_kv(value("--gemma4-sliding-kv"));
            } catch (const std::invalid_argument& error) {
                usage_error(error.what());
            }
        } else if (option == "--gemma4-q5-prefill") {
            try {
                out.gemma4_q5_prefill = infernix::parse_gemma4_q5_prefill(value("--gemma4-q5-prefill"));
            } catch (const std::invalid_argument& error) {
                usage_error(error.what());
            }
        } else if (option == "--logits-dtype") {
            try {
                out.logits_dtype = infernix::product::parse_logits_dtype(value("--logits-dtype"));
            } catch (const std::invalid_argument& error) { usage_error(error.what()); }
        } else if (option == "--output") {
            out.output = std::filesystem::path(value("--output"));
        } else if (option == "--save-top-tokens") {
            out.save_top_tokens = std::filesystem::path(value("--save-top-tokens"));
        } else if (option == "--save-tokens") {
            out.save_tokens = std::filesystem::path(value("--save-tokens"));
        } else if (option == "--kl-reference") {
            out.kl_reference = std::filesystem::path(value("--kl-reference"));
        } else if (option == "--full-vocab-reference-write") {
            out.full_vocab_reference_write = std::filesystem::path(value("--full-vocab-reference-write"));
        } else if (option == "--full-vocab-reference") {
            out.full_vocab_reference = std::filesystem::path(value("--full-vocab-reference"));
        } else if (option == "--full-vocab-seed") {
            out.full_vocab_seed = parse_integer<std::uint64_t>(value("--full-vocab-seed"), "full-vocab seed");
        } else if (option == "--full-vocab-resamples") {
            out.full_vocab_resamples = parse_integer<std::uint32_t>(value("--full-vocab-resamples"), "full-vocab resamples");
        } else if (option == "--full-vocab-reference-kv-dtype") {
            const std::string_view dtype = value("--full-vocab-reference-kv-dtype");
            if (dtype == "int8") { out.full_vocab_reference_kv = "int8-g64"; }
            else if (dtype == "fp8") { out.full_vocab_reference_kv = "fp8-e4m3-r256"; }
            else if (dtype == "bf16" || dtype == "bf16p" || dtype == "nvfp4" ||
                     dtype == "k8v4" || dtype == "vq2" || dtype == "k4v2") {
                out.full_vocab_reference_kv = std::string(dtype);
            } else { usage_error("invalid --full-vocab-reference-kv-dtype"); }
        } else if (option == "--full-vocab-reference-gemma4-qk") {
            out.full_vocab_reference_qk = infernix::parse_gemma4_qk(value("--full-vocab-reference-gemma4-qk"));
        } else if (option == "--full-vocab-prefix-ids") {
            std::string_view list = value("--full-vocab-prefix-ids");
            out.full_vocab_prefix.clear();
            while (!list.empty()) {
                const auto comma = list.find(',');
                const auto id = parse_integer<infernix::TokenId>(list.substr(0, comma), "full-vocab prefix id");
                if (id < 0) { usage_error("--full-vocab-prefix-ids requires nonnegative token IDs"); }
                out.full_vocab_prefix.push_back(id);
                if (comma != std::string_view::npos && comma + 1 == list.size()) {
                    usage_error("--full-vocab-prefix-ids cannot end with a comma");
                }
                list = comma == std::string_view::npos ? std::string_view{} : list.substr(comma + 1);
            }
            if (out.full_vocab_prefix.empty()) { usage_error("--full-vocab-prefix-ids requires token IDs"); }
        } else if (option == "--expert-stripe") {
            out.expert_stripe = std::filesystem::path(value("--expert-stripe"));
        } else if (option == "--log-level") {
            out.log_level = infernix::product::parse_log_level(value("--log-level"));
        } else {
            usage_error("unknown option: " + std::string(option));
        }
    }
    if (out.corpus.has_value() == out.text.has_value()) {
        usage_error("exactly one of --corpus and --text is required");
    }
    if (out.quick && !out.corpus) { usage_error("--quick requires --corpus"); }
    if (out.incremental && (out.kl_reference || out.save_top_tokens)) {
        usage_error("--incremental requires NLL or full-vocabulary mode; top-token caches do not bind its schedule");
    }
    if (out.full_vocab_reference_write && out.full_vocab_reference) {
        usage_error("--full-vocab-reference-write and --full-vocab-reference are mutually exclusive");
    }
    if (out.full_vocab_reference_write || out.full_vocab_reference) {
        if (out.kl_reference || out.save_top_tokens) {
            usage_error("full-vocabulary modes cannot be combined with --kl-reference or --save-top-tokens");
        }
        if (out.logits_dtype != infernix::LogitsDtype::BF16) {
            usage_error("full-vocabulary modes require --logits-dtype bf16");
        }
    }
    if ((out.full_vocab_reference_kv || out.full_vocab_reference_qk) && !out.full_vocab_reference) {
        usage_error("reference configuration overrides require --full-vocab-reference");
    }
    if (out.full_vocab_resamples == 0) { usage_error("--full-vocab-resamples must be positive"); }
    if (out.full_vocab_prefix.size() > std::numeric_limits<std::uint32_t>::max() - out.context) {
        usage_error("context plus full-vocabulary prefix is too large");
    }
    if (out.context < 2 || out.stride == 0 || out.stride >= out.context) {
        usage_error("context/stride must satisfy context>=2 and 1<=stride<context");
    }
    return out;
}

std::string kv_name(infernix::KvCacheStorage value) {
    switch (value) {
    case infernix::KvCacheStorage::BFloat16:
        return "bf16";
    case infernix::KvCacheStorage::Int8Group64:
        return "int8-g64";
    case infernix::KvCacheStorage::Fp8E4M3Row256:
        return "fp8-e4m3-r256";
    case infernix::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case infernix::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    case infernix::KvCacheStorage::Vq2:
        return "vq2";
    case infernix::KvCacheStorage::Q4KeyVq2Value:
        return "k4v2";
    case infernix::KvCacheStorage::Bf16Packed:
        return "bf16p";
    }
    throw std::logic_error("unknown KV dtype");
}

std::string safe_component(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char c : value) {
        out.push_back(std::isalnum(c) || c == '-' || c == '_' || c == '.' ? static_cast<char>(c)
                                                                          : '-');
    }
    return out.empty() ? "unknown" : out;
}

std::string timestamp() {
    const std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream out;
    out << std::put_time(&utc, "%Y%m%d-%H%M%S");
    return out.str();
}

std::filesystem::path prepare_output_directory(const Options& options,
                                               const infernix::LoadSummary& load,
                                               const CorpusSelection& corpus) {
    std::filesystem::path output = options.output.value_or(
        std::filesystem::path("profiles/perplexity") / safe_component(load.model_name) /
        safe_component(load.prefill_signature) / kv_name(options.kv) /
        safe_component(corpus.corpus_id) / safe_component(corpus.mode) / timestamp());
    if (std::filesystem::exists(output)) {
        if (!std::filesystem::is_directory(output) ||
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator()) {
            throw std::runtime_error("output directory exists and is not empty: " +
                                     output.string());
        }
    } else if (!std::filesystem::create_directories(output)) {
        throw std::runtime_error("cannot create output directory: " + output.string());
    }
    return std::filesystem::absolute(output).lexically_normal();
}

double seconds_since(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

json aggregate_json(const ScoreAggregate& value) {
    return json{{"scored_tokens", value.scored_tokens},
                {"total_nll", value.total_nll},
                {"mean_nll", value.mean_nll()},
                {"perplexity", value.ppl()}};
}

struct EvaluationStream {
    infernix::perplexity::CorpusStream source;
    std::vector<infernix::TokenId> tokens;
    std::vector<WindowPlan> windows;
};

namespace kl = infernix::perplexity::kl;

// Exact paired full-vocabulary KL (NKLREF01 cache format and statistics). Receives widened BF16 logits, turns
// them into F32 log-probabilities and either appends them to the reference cache or compares them
// with the cached reference rows. The cache stores no logits and is bound to the run protocol.
class KlRun {
public:
    KlRun(const Options& options, const infernix::LoadSummary& load, const CorpusSelection& corpus,
          const std::vector<EvaluationStream>& streams, const infernix::EngineOptions& engine,
          std::uint64_t total_positions)
        : options_(options), writing_(options.full_vocab_reference_write.has_value()),
          total_(total_positions) {
        kl::Fnv64 hash;
        json stream_ids = json::array();
        for (const auto& stream : streams) {
            stream_ids.push_back({{"id", stream.source.id},
                                  {"domain", stream.source.domain},
                                  {"tokens", stream.tokens.size()}});
            hash.add(stream.tokens.data(), stream.tokens.size() * sizeof(infernix::TokenId));
            for (const WindowPlan& window : stream.windows) {
                hash.add_value(window.input_begin);
                hash.add_value(window.input_end);
                hash.add_value(window.target_begin);
                hash.add_value(window.target_end);
            }
        }
        if (!options.full_vocab_prefix.empty()) {
            hash.add(options.full_vocab_prefix.data(),
                     options.full_vocab_prefix.size() * sizeof(infernix::TokenId));
        }
        std::ostringstream hex;
        hex << std::hex << std::setw(16) << std::setfill('0') << hash.value();
        protocol_ = {{"schema", "ninfer-kl-protocol/1"},
                     {"corpus_id", corpus.corpus_id},
                     {"corpus_mode", corpus.mode},
                     {"streams", std::move(stream_ids)},
                     {"token_window_hash_fnv64", hex.str()},
                     {"context_tokens", options.context},
                     {"stride_tokens", options.stride},
                     {"prefill_chunk_tokens", engine.prefill_chunk},
                     {"prefix_ids", options.full_vocab_prefix},
                     {"kv_dtype", kv_name(options.kv)},
                     {"gemma4_qk", infernix::gemma4_qk_name(engine.gemma4_qk)},
                     {"positions", total_positions}};
        if (options.score_tail != 0) { protocol_["score_window_tail"] = options.score_tail; }
        if (options.incremental) { protocol_["scoring_schedule"] = "prefill-then-one-token/1"; }
        std::error_code size_error;
        const auto bytes = std::filesystem::file_size(options.artifact, size_error);
        reference_ = {{"artifact",
                       std::filesystem::absolute(options.artifact).lexically_normal().string()},
                      {"artifact_file_bytes", size_error ? json(nullptr) : json(bytes)},
                      {"name", load.model_name},
                      {"architecture", load.architecture},
                      {"prefill_signature", load.prefill_signature},
                      {"formats", load.weight_formats}};
        stats_.kl.assign(total_, 0.0F);
        stats_.ref_lp.assign(total_, 0.0F);
        stats_.cand_lp.assign(total_, 0.0F);
        stats_.top1_agree.assign(total_, 1);
        if (!writing_) {
            if (options.full_vocab_reference_kv) {
                protocol_["kv_dtype"] = *options.full_vocab_reference_kv;
            }
            if (options.full_vocab_reference_qk) {
                protocol_["gemma4_qk"] = infernix::gemma4_qk_name(*options.full_vocab_reference_qk);
            }
        }
        candidate_configuration_ = {{"kv_dtype", kv_name(options.kv)},
                                    {"gemma4_qk", infernix::gemma4_qk_name(engine.gemma4_qk)},
                                    {"logits_dtype", "bf16"},
                                    {"rope_yarn_factor", options.rope_yarn_factor},
                                    {"lossless", options.lossless}, {"reference", options.reference},
                                    {"expert_a4_activations", options.expert_a4_activations},
                                    {"expert_a16_activations", options.expert_a16_activations},
                                    {"fast_paths", load.fast_paths},
                                    {"fast_path_configuration", load.fast_path_configuration}};
    }

    // The public API returns at most 256 full-logit rows. Replay an identical independent
    // window for further selections; never shorten it or change its prefill schedule.
    [[nodiscard]] infernix::ScoreResult score(infernix::Engine& engine,
                                               const std::vector<infernix::TokenId>& input,
                                               std::uint32_t first_target) {
        infernix::ScoreResult result;
        for (std::size_t begin = 0; begin < targets_.size();) {
            const auto columns = static_cast<std::uint32_t>(std::min<std::size_t>(
                targets_.size() - begin, infernix::kMaximumScoreLogitPositions));
            infernix::ScoreOptions selection;
            selection.incremental = options_.incremental;
            selection.logits_positions.resize(columns);
            std::iota(selection.logits_positions.begin(), selection.logits_positions.end(),
                      static_cast<std::uint32_t>(begin));
            auto scored = engine.score_tokens(input, first_target, std::move(selection));
            if (scored.logprobs.size() != targets_.size() || scored.logits.empty() ||
                scored.logits.size() % columns != 0 ||
                scored.logits.size() / columns > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::runtime_error("full-vocabulary scoring returned an invalid logit shape");
            }
            if (begin != 0 &&
                std::memcmp(scored.logprobs.data(), result.logprobs.data(),
                            scored.logprobs.size() * sizeof(float)) != 0) {
                throw std::runtime_error(
                    "full-vocabulary window replay changed target log-probability bits; "
                    "refusing to assemble distributions from different forward passes");
            }
            const auto vocab = static_cast<std::int32_t>(scored.logits.size() / columns);
            tile(scored.logits, vocab, static_cast<std::int32_t>(columns));
            // Check every selected row against its own scoring call, including replay batches.
            for (std::size_t i = 0; i < columns; ++i) {
                const double diff = std::abs(static_cast<double>(scored.logprobs[begin + i]) -
                    static_cast<double>(stats_.cand_lp[window_begin_ + begin + i]));
                if (!std::isfinite(diff)) { throw std::runtime_error("non-finite full-vocabulary target log-probability"); }
                max_target_diff_ = std::max(max_target_diff_, diff);
            }
            if (begin == 0) { result.logprobs = std::move(scored.logprobs); }
            begin += columns;
        }
        return result;
    }

    void begin_window(std::span<const infernix::TokenId> targets, const std::string& stream,
                      const std::string& domain) {
        targets_       = targets;
        stream_        = stream;
        domain_        = domain;
        window_cursor_ = 0;
        window_begin_  = position_;
        window_logits_hash_ = kl::Fnv64{};
    }

    void end_window(std::span<const float> engine_logprobs) {
        if (window_cursor_ != targets_.size() || engine_logprobs.size() != targets_.size()) {
            throw std::runtime_error("KL observer saw " + std::to_string(window_cursor_) +
                                     " columns for a window of " + std::to_string(targets_.size()) +
                                     " targets");
        }
        for (std::size_t i = 0; i < targets_.size(); ++i) {
            const double diff = std::abs(static_cast<double>(engine_logprobs[i]) -
                                         static_cast<double>(stats_.cand_lp[window_begin_ + i]));
            max_target_diff_  = std::max(max_target_diff_, diff);
        }
        // The host log-softmax of the BF16 logits must agree with the engine's own target
        // log-probability; a larger gap means the observer columns are misaligned with targets.
        if (max_target_diff_ > 0.05) {
            throw std::runtime_error("KL observer target log-probability disagrees with the "
                                     "engine by " + std::to_string(max_target_diff_));
        }
        windows_.push_back(
            kl::WindowRecord{.stream = stream_, .domain = domain_, .begin = window_begin_,
                             .count = targets_.size()});
        std::ostringstream digest;
        digest << std::hex << std::setw(16) << std::setfill('0') << window_logits_hash_.value();
        logit_hashes_.push_back(digest.str());
    }

    [[nodiscard]] const json& logit_hashes() const noexcept { return logit_hashes_; }

    [[nodiscard]] json finish(const std::filesystem::path& output_directory) {
        if (position_ != total_) {
            throw std::runtime_error("KL run scored " + std::to_string(position_) + " of " +
                                     std::to_string(total_) + " positions");
        }
        json out{{"mode", writing_ ? "reference_write" : "compare"},
                 {"protocol", protocol_},
                 {"target_logprob_max_abs_diff", max_target_diff_},
                 {"dtype", "f32"},
                 {"candidate_configuration", candidate_configuration_}};
        if (writing_) {
            writer_->finish();
            out["cache"]     = std::filesystem::absolute(*options_.full_vocab_reference_write).string();
            out["positions"] = writer_->positions();
            return out;
        }
        if (reader_->consumed() != reader_->positions()) {
            throw std::runtime_error("KL reference cache holds " +
                                     std::to_string(reader_->positions()) +
                                     " positions but this run consumed " +
                                     std::to_string(reader_->consumed()));
        }
        out["reference_cache"]    = std::filesystem::absolute(*options_.full_vocab_reference).string();
        out["reference_artifact"] = reader_->reference();
        out["candidate_artifact"] = reference_;
        out["statistics"]         = kl::summarize(stats_, windows_, options_.full_vocab_resamples,
                                                  options_.full_vocab_seed);
        const std::filesystem::path sidecar = output_directory / "kl_positions.f32";
        std::ofstream file(sidecar, std::ios::binary | std::ios::trunc);
        for (std::size_t i = 0; i < total_; ++i) {
            const float record[4] = {stats_.kl[i], stats_.ref_lp[i], stats_.cand_lp[i],
                                     static_cast<float>(stats_.top1_agree[i])};
            file.write(reinterpret_cast<const char*>(record), sizeof(record));
        }
        if (!file) { throw std::runtime_error("cannot write " + sidecar.string()); }
        out["per_position_file"] = {{"path", sidecar.string()},
                                    {"record", "f32 x4: kl, ref_target_lp, cand_target_lp, top1_agree"}};
        return out;
    }

private:
    void open(std::int32_t vocab) {
        vocab_             = vocab;
        protocol_["vocab"] = vocab;
        if (writing_) {
            writer_ = std::make_unique<kl::CacheWriter>(*options_.full_vocab_reference_write, vocab,
                                                        protocol_, reference_);
        } else {
            reader_ = std::make_unique<kl::CacheReader>(*options_.full_vocab_reference, vocab, protocol_);
        }
    }

    void tile(std::span<const float> logits, std::int32_t vocab, std::int32_t columns) {
        if (vocab_ == 0) { open(vocab); }
        if (vocab != vocab_ || logits.size() != static_cast<std::size_t>(vocab) * columns) {
            throw std::runtime_error("KL observer tile has an inconsistent shape");
        }
        if (window_cursor_ + static_cast<std::size_t>(columns) > targets_.size() ||
            position_ + static_cast<std::uint64_t>(columns) > total_) {
            throw std::runtime_error("KL observer received more columns than scored positions");
        }
        kl::hash_bf16_logits(window_logits_hash_, logits);
        const std::size_t cells = static_cast<std::size_t>(vocab) * columns;
        cand_.resize(cells);
        kl::parallel_columns(columns, [&](std::int32_t begin, std::int32_t end) {
            for (std::int32_t c = begin; c < end; ++c) {
                kl::log_softmax_f32(logits.data() + static_cast<std::size_t>(c) * vocab, vocab,
                                     cand_.data() + static_cast<std::size_t>(c) * vocab);
            }
        });
        if (writing_) {
            writer_->append(cand_, columns);
        } else {
            ref_.resize(cells);
            reader_->read(columns, ref_);
        }
        for (std::int32_t c = 0; c < columns; ++c) {
            const auto target = targets_[window_cursor_ + static_cast<std::size_t>(c)];
            if (target < 0 || target >= vocab) {
                throw std::runtime_error("KL observer target id is outside the vocab");
            }
        }
        kl::parallel_columns(columns, [&](std::int32_t begin, std::int32_t end) {
            for (std::int32_t c = begin; c < end; ++c) {
                const float* q = cand_.data() + static_cast<std::size_t>(c) * vocab;
                const auto t   = targets_[window_cursor_ + static_cast<std::size_t>(c)];
                const std::size_t at = position_ + static_cast<std::size_t>(c);
                stats_.cand_lp[at]   = q[t];
                if (!writing_) {
                    const float* r      = ref_.data() + static_cast<std::size_t>(c) * vocab;
                    stats_.ref_lp[at]   = r[t];
                    stats_.kl[at]       = static_cast<float>(kl::kl_divergence(r, q, vocab));
                    stats_.top1_agree[at] = kl::argmax(r, vocab) == kl::argmax(q, vocab) ? 1 : 0;
                }
            }
        });
        position_ += static_cast<std::uint64_t>(columns);
        window_cursor_ += static_cast<std::size_t>(columns);
    }

    const Options& options_;
    bool writing_;
    std::uint64_t total_;
    json protocol_;
    json reference_;
    json candidate_configuration_;
    kl::Fnv64 window_logits_hash_;
    json logit_hashes_ = json::array();
    std::unique_ptr<kl::CacheWriter> writer_;
    std::unique_ptr<kl::CacheReader> reader_;
    std::int32_t vocab_ = 0;
    std::vector<float> cand_;
    std::vector<float> ref_;
    kl::PositionStats stats_;
    std::vector<kl::WindowRecord> windows_;
    std::span<const infernix::TokenId> targets_;
    std::string stream_;
    std::string domain_;
    std::size_t window_cursor_ = 0;
    std::uint64_t window_begin_ = 0;
    std::uint64_t position_     = 0;
    double max_target_diff_     = 0.0;
};

int run(const Options& options, const std::shared_ptr<spdlog::logger>& logger,
        infernix::product::StartupLogRenderer& startup_log,
        const std::shared_ptr<infernix::product::TerminalProgress>& progress) {
    const Clock::time_point total_started = Clock::now();
    infernix::EngineOptions engine_options;
    engine_options.artifact_path    = options.artifact;
    engine_options.lora_path        = options.lora;
    engine_options.ablate_sidecar_path = options.ablate;
    engine_options.purpose          = infernix::EnginePurpose::CausalScoring;
    engine_options.device           = options.device;
    engine_options.max_context      = options.context + static_cast<std::uint32_t>(options.full_vocab_prefix.size());
    engine_options.rope_yarn_factor  = options.rope_yarn_factor;
    if (options.prefill_chunk) { engine_options.prefill_chunk = *options.prefill_chunk; }
    engine_options.kv_cache         = options.kv;
    engine_options.gemma4_qk        = options.gemma4_qk;
    engine_options.gemma4_key_only_kv = options.gemma4_key_only_kv;
    engine_options.gemma4_sliding_kv = options.gemma4_sliding_kv;
    engine_options.gemma4_q5_prefill = options.gemma4_q5_prefill;
    engine_options.logits_dtype     = options.logits_dtype;
    engine_options.original_int8_prefill_kernel = options.original_int8_prefill_kernel;
    engine_options.original_nvfp4_prefill_kernel = options.original_nvfp4_prefill_kernel;
    engine_options.lossless                      = options.lossless;
    engine_options.reference                     = options.reference;
    engine_options.expert_a4_activations         = options.expert_a4_activations;
    engine_options.expert_a16_activations        = options.expert_a16_activations;
    engine_options.a8_activations                = options.a8_activations;
    // The lossless route runs no fast path.
    engine_options.fast_paths           = options.fast_paths && !options.lossless;
    engine_options.fast_path_candidates = options.fast_path_candidates;
    engine_options.fast_path_exclusions = options.fast_path_exclusions;
    if (options.ngram_volume) { engine_options.ngram_volume_path = *options.ngram_volume; }
    engine_options.expert_stripe_path   = options.expert_stripe;
    engine_options.startup_observer = startup_log.observer();
    engine_options.diagnostic_observer = infernix::product::engine_diagnostic_observer(logger);
    infernix::Engine engine(std::move(engine_options));
    const infernix::LoadSummary load = engine.load_summary();
    startup_log.engine_ready(load);

    const Clock::time_point preflight_started = Clock::now();
    logger->info("preparing corpus");
    CorpusSelection corpus = options.corpus
                                 ? infernix::perplexity::load_corpus(*options.corpus, options.quick)
                                 : infernix::perplexity::load_custom_text(*options.text);
    std::vector<EvaluationStream> streams;
    streams.reserve(corpus.streams.size());
    std::uint64_t total_scored_tokens = 0;
    std::uint64_t total_input_tokens  = 0;
    std::uint64_t total_windows       = 0;
    for (auto& source : corpus.streams) {
        std::vector<infernix::TokenId> tokens = engine.tokenize_text(source.text);
        if (tokens.size() < 2) {
            throw std::runtime_error("stream tokenized to fewer than two tokens: " + source.id);
        }
        std::vector<WindowPlan> windows =
            infernix::perplexity::plan_windows(tokens.size(), options.context, options.stride, options.score_tail);
        total_input_tokens += static_cast<std::uint64_t>(tokens.size());
        for (const auto& window : windows) {
            total_scored_tokens += static_cast<std::uint64_t>(window.target_end - window.target_begin);
        }
        total_windows += static_cast<std::uint64_t>(windows.size());
        streams.push_back(EvaluationStream{.source  = std::move(source),
                                           .tokens  = std::move(tokens),
                                           .windows = std::move(windows)});
    }
    const double preflight_seconds = seconds_since(preflight_started);
    logger->info("corpus ready | {} streams | {} input tokens | {} scored tokens | {} windows | {}",
                 infernix::product::format_pretty_count(streams.size()),
                 infernix::product::format_pretty_count(total_input_tokens),
                 infernix::product::format_pretty_count(total_scored_tokens),
                 infernix::product::format_pretty_count(total_windows),
                 infernix::product::format_pretty_duration(preflight_seconds));

    const std::filesystem::path output_directory = prepare_output_directory(options, load, corpus);
    std::unique_ptr<KlRun> kl_run;
    if (options.full_vocab_reference_write || options.full_vocab_reference) {
        kl_run = std::make_unique<KlRun>(options, load, corpus, streams, engine.options(),
                                        total_scored_tokens);
    }
    const infernix::perplexity::ReferenceHeader reference_header{.corpus_id = corpus.corpus_id,
                                                                .mode      = corpus.mode,
                                                                .context   = options.context,
                                                                .stride    = options.stride};
    std::optional<infernix::perplexity::ReferenceWriter> top_tokens;
    if (options.save_top_tokens) { top_tokens.emplace(*options.save_top_tokens, reference_header); }
    std::optional<infernix::perplexity::ReferenceReader> kl_reference;
    if (options.kl_reference) {
        kl_reference.emplace(*options.kl_reference);
        const infernix::perplexity::ReferenceHeader& saved = kl_reference->header();
        if (saved.corpus_id != reference_header.corpus_id || saved.mode != reference_header.mode ||
            saved.context != reference_header.context || saved.stride != reference_header.stride) {
            throw std::runtime_error("--kl-reference was recorded for corpus " + saved.corpus_id +
                                     " / " + saved.mode + " with context/stride " +
                                     std::to_string(saved.context) + "/" +
                                     std::to_string(saved.stride));
        }
    }
    // The evaluated run's own most probable token, for top-1 agreement.
    const std::uint32_t own_top_tokens =
        top_tokens ? infernix::perplexity::kReferenceTopTokens : (kl_reference ? 1U : 0U);
    DivergenceAggregate overall_divergence;
    // --save-tokens: every window's input tokens and its targets, in scoring order.
    json saved_windows                           = json::array();
    const Clock::time_point scoring_started      = Clock::now();
    logger->info("scoring | {} streams | {} tokens | {} windows",
                 infernix::product::format_pretty_count(streams.size()),
                 infernix::product::format_pretty_count(total_scored_tokens),
                 infernix::product::format_pretty_count(total_windows));
    Clock::time_point next_progress = scoring_started + std::chrono::seconds(10);
    ScoreAggregate overall;
    std::map<std::string, ScoreAggregate> domains;
    json stream_reports             = json::array();
    std::uint64_t completed_windows = 0;

    for (std::size_t stream_index = 0; stream_index < streams.size(); ++stream_index) {
        EvaluationStream& stream = streams[stream_index];
        std::ostringstream stream_status;
        stream_status << "  scoring [" << stream_index + 1 << '/' << streams.size() << "] "
                      << infernix::product::format_pretty_text(stream.source.id) << " | "
                      << infernix::product::format_pretty_count(stream.tokens.size()) << " tokens | "
                      << infernix::product::format_pretty_count(stream.windows.size()) << " windows";
        if (progress->enabled()) {
            progress->update(stream_status.str());
        } else {
            logger->debug("{}", stream_status.str());
        }
        const Clock::time_point stream_started = Clock::now();
        ScoreAggregate stream_score;
        DivergenceAggregate stream_divergence;
        json window_reports = json::array();
        for (std::size_t window_index = 0; window_index < stream.windows.size(); ++window_index) {
            const WindowPlan& window = stream.windows[window_index];
            std::vector<infernix::TokenId> input = options.full_vocab_prefix;
            input.insert(input.end(),
                         stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_begin),
                         stream.tokens.begin() + static_cast<std::ptrdiff_t>(window.input_end));
            const auto first_target = window.first_target +
                                      static_cast<std::uint32_t>(options.full_vocab_prefix.size());
            const Clock::time_point window_started = Clock::now();
            const std::uint64_t input_hash = infernix::perplexity::token_fingerprint(input);
            if (options.save_tokens) {
                saved_windows.push_back({{"stream_index", stream_index},
                                         {"target_begin", window.target_begin},
                                         {"target_end", window.target_end},
                                         {"first_target", first_target},
                                         {"input_hash", input_hash},
                                         {"tokens", input}});
            }
            infernix::ScoreOptions score_options{.top_k = own_top_tokens};
            score_options.incremental = options.incremental;
            std::optional<infernix::perplexity::ReferenceWindow> reference;
            if (kl_reference) {
                reference = kl_reference->next();
                if (reference->stream_index != stream_index ||
                    reference->target_begin != window.target_begin ||
                    reference->target_end != window.target_end ||
                    reference->input_hash != input_hash) {
                    throw std::runtime_error("--kl-reference window " +
                                             std::to_string(window_index) + " of " +
                                             stream.source.id + " scored different tokens");
                }
                score_options.candidates_per_position = kl_reference->header().top_k;
                score_options.candidates              = reference->top_ids;
            }
            infernix::ScoreResult scored;
            try {
                if (kl_run) {
                    const std::size_t count = window.target_end - window.target_begin;
                    kl_run->begin_window(std::span<const infernix::TokenId>(stream.tokens)
                                             .subspan(window.target_begin, count),
                                         stream.source.id, stream.source.domain);
                    scored = kl_run->score(engine, input, first_target);
                    kl_run->end_window(scored.logprobs);
                } else {
                    scored = engine.score_tokens(std::move(input), first_target,
                                                 std::move(score_options));
                }
            } catch (const std::exception& error) {
                throw std::runtime_error("scoring " + stream.source.id + " window " +
                                         std::to_string(window_index) + " failed: " + error.what());
            }
            const std::vector<float>& logprobs = scored.logprobs;
            const std::size_t expected         = window.target_end - window.target_begin;
            if (logprobs.size() != expected) {
                throw std::runtime_error("scoring returned an invalid target count for " +
                                         stream.source.id);
            }
            if (top_tokens) {
                top_tokens->write({.stream_index    = static_cast<std::uint32_t>(stream_index),
                                   .target_begin    = window.target_begin,
                                   .target_end      = window.target_end,
                                   .input_hash      = input_hash,
                                   .target_logprobs = logprobs,
                                   .top_ids         = scored.top_ids,
                                   .top_logprobs    = scored.top_logprobs});
            }
            DivergenceAggregate window_divergence;
            if (reference) {
                const std::size_t k = kl_reference->header().top_k;
                for (std::size_t position = 0; position < expected; ++position) {
                    const std::span<const float> reference_top(
                        reference->top_logprobs.data() + position * k, k);
                    const std::span<const float> evaluated_top(
                        scored.candidate_logprobs.data() + position * k, k);
                    const bool top1_match =
                        scored.top_ids[position * own_top_tokens] == reference->top_ids[position * k];
                    // The predicted token at window input index first_target + position follows
                    // that many tokens of context.
                    window_divergence.add(first_target + position,
                                          infernix::perplexity::position_divergence(
                                              reference_top, evaluated_top, top1_match,
                                              reference->target_logprobs[position],
                                              logprobs[position]));
                }
                stream_divergence.add(window_divergence);
            }
            ScoreAggregate window_score;
            window_score.add(logprobs);
            stream_score.add(window_score);
            overall.add(window_score);
            domains[stream.source.domain].add(window_score);
            ++completed_windows;
            json window_report            = aggregate_json(window_score);
            window_report["index"]        = window_index;
            window_report["input_begin"]  = window.input_begin;
            window_report["input_end"]    = window.input_end;
            window_report["target_begin"] = window.target_begin;
            window_report["target_end"]   = window.target_end;
            window_report["first_target"] = first_target;
            window_report["seconds"]      = seconds_since(window_started);
            // Windows are independent samples, which a comparison pairs between runs.
            if (reference) { window_report["kl_divergence"] = window_divergence.summary_json(); }
            window_reports.push_back(std::move(window_report));

            if (Clock::now() >= next_progress) {
                const double elapsed = seconds_since(scoring_started);
                const double rate    = static_cast<double>(overall.scored_tokens) / elapsed;
                const std::uint64_t remaining = total_scored_tokens - overall.scored_tokens;
                const double eta = rate > 0 ? static_cast<double>(remaining) / rate : 0.0;
                std::ostringstream line;
                line << "scoring | " << infernix::product::format_pretty_count(overall.scored_tokens)
                     << '/' << infernix::product::format_pretty_count(total_scored_tokens)
                     << " tokens | " << completed_windows << '/' << total_windows
                     << " windows | PPL " << std::fixed << std::setprecision(4) << overall.ppl();
                if (kl_reference && stream_divergence.positions() > 0) {
                    line << " | stream KL " << std::scientific << std::setprecision(3)
                         << stream_divergence.mean_kl() << std::fixed;
                }
                line
                     << " | " << infernix::product::format_pretty_rate(rate, "tok") << " | elapsed "
                     << infernix::product::format_pretty_duration(elapsed) << " | ETA "
                     << infernix::product::format_pretty_duration(eta);
                if (progress->enabled()) {
                    progress->update("  " + line.str());
                } else {
                    logger->info("{}", line.str());
                }
                next_progress = Clock::now() + std::chrono::seconds(10);
            }
        }
        const double stream_seconds = seconds_since(stream_started);
        progress->clear();
        logger->info("[{}/{}] {} | {} scored tokens | PPL {:.6g} | {}", stream_index + 1,
                     streams.size(), infernix::product::format_pretty_text(stream.source.id),
                     infernix::product::format_pretty_count(stream_score.scored_tokens),
                     stream_score.ppl(), infernix::product::format_pretty_duration(stream_seconds));
        json stream_report               = aggregate_json(stream_score);
        if (kl_reference) {
            logger->info("[{}/{}] {} | KL {:.4e} | top-1 agreement {:.4f} | by context {}",
                         stream_index + 1, streams.size(),
                         infernix::product::format_pretty_text(stream.source.id),
                         stream_divergence.mean_kl(), stream_divergence.top1_agreement(),
                         stream_divergence.bucket_summary());
            stream_report["kl_divergence"] = stream_divergence.to_json();
            overall_divergence.add(stream_divergence);
        }
        stream_report["id"]              = stream.source.id;
        stream_report["domain"]          = stream.source.domain;
        stream_report["path"]            = stream.source.path.string();
        stream_report["input_tokens"]    = stream.tokens.size();
        stream_report["unscored_tokens"] = stream.tokens.size() - stream_score.scored_tokens;
        stream_report["seconds"]         = stream_seconds;
        stream_report["windows"]         = std::move(window_reports);
        stream_reports.push_back(std::move(stream_report));
    }

    const double scoring_seconds = seconds_since(scoring_started);
    progress->clear();
    if (top_tokens) { top_tokens->finish(); }
    if (kl_reference) { kl_reference->finish(); }
    if (options.save_tokens) {
        std::ofstream file(*options.save_tokens);
        file << json{{"corpus_id", corpus.corpus_id},
                     {"mode", corpus.mode},
                     {"context", options.context},
                     {"stride", options.stride},
                     {"prefix_ids", options.full_vocab_prefix},
                     {"windows", std::move(saved_windows)}}
                    .dump();
        if (!file) { throw std::runtime_error("cannot write " + options.save_tokens->string()); }
    }
    logger->info("scoring complete | {} tokens | {} windows | PPL {:.6g} | {} | {}",
                 infernix::product::format_pretty_count(overall.scored_tokens), completed_windows,
                 overall.ppl(), infernix::product::format_pretty_duration(scoring_seconds),
                 infernix::product::format_pretty_rate(
                     static_cast<double>(overall.scored_tokens) / scoring_seconds, "tok"));
    json domain_reports = json::array();
    for (const auto& [domain, aggregate] : domains) {
        json item      = aggregate_json(aggregate);
        item["domain"] = domain;
        domain_reports.push_back(std::move(item));
    }

    json report{
        {"schema_version", 3},
        {"metric",
         {{"name", "fixed-window truncated-context causal perplexity"}, {"log_base", "natural"}}},
        {"artifact",
         {{"path", std::filesystem::absolute(options.artifact).lexically_normal().string()},
          {"architecture", load.architecture},
          {"name", load.model_name},
          {"prefill_signature", load.prefill_signature},
          {"formats", load.weight_formats}}},
        {"corpus",
         {{"id", corpus.corpus_id},
          {"mode", corpus.mode},
          {"source", corpus.source.string()},
          {"stream_count", streams.size()}}},
        {"execution",
         {{"purpose", "causal_scoring"},
          {"device", options.device},
          {"context_tokens", options.context},
          {"prefix_ids", options.full_vocab_prefix},
          {"score_window_tail", options.score_tail},
          {"rope_yarn_factor", options.rope_yarn_factor},
          {"incremental", options.incremental},
          {"original_int8_prefill_kernel", options.original_int8_prefill_kernel},
          {"original_nvfp4_prefill_kernel", options.original_nvfp4_prefill_kernel},
          {"lossless", options.lossless},
          {"reference", options.reference},
          {"expert_a4_activations", options.expert_a4_activations},
          {"expert_a16_activations", options.expert_a16_activations},
          {"a8_activations", options.a8_activations},
          {"lora_adapter", load.lora_adapter.empty() ? nlohmann::json(nullptr) : nlohmann::json(load.lora_adapter)},
          {"ablate_sidecar",
           load.ablate_sidecar.empty() ? nlohmann::json(nullptr) : nlohmann::json(load.ablate_sidecar)},
          {"fast_paths", load.fast_paths},
          {"fast_path_configuration", load.fast_path_configuration},
          {"stride_tokens", options.stride},
          {"prefill_chunk_tokens", engine.options().prefill_chunk},
          {"score_tile_tokens", 1024},
          {"kv_dtype", kv_name(options.kv)},
          {"gemma4_qk", infernix::gemma4_qk_name(options.gemma4_qk)},
          {"gemma4_key_only_kv", options.gemma4_key_only_kv},
          {"gemma4_sliding_kv", options.gemma4_sliding_kv == infernix::KvCacheStorage::Fp8E4M3Row256 ? "fp8" : "bf16"},
          {"gemma4_q5_prefill", infernix::gemma4_q5_prefill_name(options.gemma4_q5_prefill)},
          {"logits_dtype", infernix::product::logits_dtype_name(options.logits_dtype)}}},
        {"timing",
         {{"load_seconds", load.load_seconds},
          {"read_and_tokenize_seconds", preflight_seconds},
          {"score_seconds", scoring_seconds},
          {"total_seconds", seconds_since(total_started)},
          {"scored_tokens_per_second",
           static_cast<double>(overall.scored_tokens) / scoring_seconds}}},
        {"streams", std::move(stream_reports)},
        {"domains", std::move(domain_reports)},
        {"overall", aggregate_json(overall)},
    };
    if (top_tokens) {
        report["top_tokens"] = {{"path", std::filesystem::absolute(*options.save_top_tokens).string()},
                                {"per_position", infernix::perplexity::kReferenceTopTokens}};
    }
    if (kl_reference) {
        json divergence         = overall_divergence.to_json();
        divergence["reference"] = std::filesystem::absolute(*options.kl_reference).string();
        divergence["reference_top_tokens"] = kl_reference->header().top_k;
        divergence["definition"] =
            "KL(reference || this run) over the reference's top tokens plus one bucket for all "
            "others (a lower bound); context is the number of tokens before the predicted one";
        report["kl_divergence"] = std::move(divergence);
    }

    if (kl_run) {
        report["kl"] = kl_run->finish(output_directory);
        report["logit_window_hashes_fnv64"] = kl_run->logit_hashes();
    }

    const std::filesystem::path temporary = output_directory / "report.json.tmp";
    const std::filesystem::path final     = output_directory / "report.json";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) { throw std::runtime_error("cannot create report: " + temporary.string()); }
        output << std::setw(2) << report << '\n';
        output.flush();
        if (!output) { throw std::runtime_error("cannot write report: " + temporary.string()); }
    }
    std::filesystem::rename(temporary, final);

    std::cout << "Perplexity result\n"
              << "artifact: " << load.model_name << '\n'
              << "kv: " << kv_name(options.kv) << ", corpus: " << corpus.corpus_id << " / "
              << corpus.mode << ", context/stride: " << options.context << '/' << options.stride
              << "\n\n";
    std::cout << std::left << std::setw(24) << "domain" << std::right << std::setw(16) << "tokens"
              << std::setw(16) << "mean_nll" << std::setw(16) << "ppl" << '\n';
    for (const auto& [domain, aggregate] : domains) {
        std::cout << std::left << std::setw(24) << domain << std::right << std::setw(16)
                  << aggregate.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
                  << aggregate.mean_nll() << std::setw(16) << aggregate.ppl() << '\n';
    }
    std::cout << std::left << std::setw(24) << "overall" << std::right << std::setw(16)
              << overall.scored_tokens << std::setw(16) << std::fixed << std::setprecision(6)
              << overall.mean_nll() << std::setw(16) << overall.ppl() << "\n\n";
    if (kl_reference) {
        std::cout << "KL divergence from reference: mean " << std::scientific
                  << std::setprecision(4) << overall_divergence.mean_kl() << ", top-1 agreement "
                  << std::fixed << std::setprecision(4) << overall_divergence.top1_agreement()
                  << "\nby context: " << overall_divergence.bucket_summary() << "\n\n";
    }
    std::cout << "score rate: " << std::fixed << std::setprecision(1)
              << static_cast<double>(overall.scored_tokens) / scoring_seconds << " tok/s\n"
              << "report: " << final << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    Options options;
    try {
        options = parse_options(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "infernix-perplexity: " << error.what() << '\n';
        std::cerr << usage_text();
        return 1;
    }
    if (options.help_requested) {
        std::cout << usage_text();
        return 0;
    }

    infernix::product::LoggingRuntime logging(
        {.logger_name  = "infernix-perplexity",
         .level        = options.log_level,
         .presentation = infernix::product::LogPresentation::Tool});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
#if defined(INFERNIX_BUILD_ID) && defined(INFERNIX_CUDA_COMPILER_VERSION)
    logger->info("build {} (CUDA {})", INFERNIX_BUILD_ID, INFERNIX_CUDA_COMPILER_VERSION);
#elif defined(INFERNIX_BUILD_ID)
    logger->info("build {}", INFERNIX_BUILD_ID);
#endif
    infernix::product::StartupLogRenderer startup_log(logging);
    try {
        return run(options, logger, startup_log, logging.terminal_progress());
    } catch (const std::exception& error) {
        logging.terminal_progress()->clear();
        logger->error("{}", infernix::product::format_pretty_text(error.what()));
        return 1;
    }
}
