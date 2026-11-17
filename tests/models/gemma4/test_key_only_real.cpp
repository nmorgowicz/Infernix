// The experimental key-only global KV cache (KvStorageModel::KeyOnly) on a real artifact; no
// downloads. INFERNIX_GEMMA4_ARTIFACT (with its MTP assistant unless --mtp 0) selects it, missing
// skips (77); INFERNIX_GEMMA4_HOST_MIB sizes the Host tier. Optional: --chunk 128|2048 --graphs 0|1
// --mtp 0|1 (default: every combination).
//
// Checked: refusals; the loaded key-norm weights are the recorded constants; direct native calls
// write byte-identical K planes (and earlier layers) to the K==V FP8 cache; Program accounting is
// the K planes alone; generation is deterministic, a root replay and a Host snapshot restore equal
// the cold key-only run, and MTP drafts read the derived values. Measured (METRIC lines, not gates):
// the first global layer's derived V and stored FP8 V against its BF16 V, final-position logits of
// key-only and stored FP8 against BF16 K/V, token agreement with stored FP8, causal-score deltas.
#include "prefix_real_support.h"
#include "models/gemma4/program/kv_layout.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <utility>

namespace {
using namespace prefix_real;
using namespace infernix;
constexpr std::uint32_t kContext = 4352;
constexpr std::uint32_t kOutputs = 16;
constexpr auto kFp8 = KvCacheStorage::Fp8E4M3Row256;

float bf16_value(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }
float half_value(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16, exponent = (bits >> 10) & 31U, mantissa = bits & 1023U;
    if (exponent == 0) { return (sign ? -1.0f : 1.0f) * std::ldexp(float(mantissa), -24); }
    if (exponent == 31) { return std::numeric_limits<float>::quiet_NaN(); }
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23) | (mantissa << 13));
}
float code_value(std::uint8_t code) {
    const int exp = (code >> 3) & 15, mantissa = code & 7;
    if (exp == 15 && mantissa == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const float value = exp == 0 ? std::ldexp(float(mantissa), -9) : std::ldexp(1.0f + float(mantissa) / 8.0f, exp - 7);
    return code & 128 ? -value : value;
}

// One native cache set: contiguous pages for every layer (sliding layers too), so every byte is
// inspectable; key-only global layers get derived-value descriptors.
struct Caches {
    std::vector<DeviceBuffer> keys, values, key_scales, value_scales;
    std::vector<Tensor> key_views, value_views, key_scale_views, value_scale_views;
    std::vector<g4::Gemma4CacheLayer> layers;
    Caches(const g4::execution::Parameters& p, int pages, KvCacheStorage kv, g4::KvStorageModel storage,
           const Tensor& table) {
        const auto count = p.layers.size();
        for (auto* v : {&keys, &values, &key_scales, &value_scales}) { v->reserve(count); }
        for (auto* v : {&key_views, &value_views, &key_scale_views, &value_scale_views}) { v->reserve(count); }
        layers.reserve(count);
        const auto& config = p.model.config().text;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& layer = p.layers[i];
            const auto layout = g4::kv_layer_layout(kv, layer.global_layer, layer.head_dim, layer.kv_heads, storage);
            const std::initializer_list<std::int32_t> shape{layer.head_dim, 64, layer.kv_heads, pages};
            const std::initializer_list<std::int32_t> scale_shape{layout.scale_groups, 64, layer.kv_heads, pages};
            keys.emplace_back(layout.code_page_bytes * pages); keys.back().fill(0);
            key_views.emplace_back(keys.back().p, layout.code_dtype, shape);
            key_scales.emplace_back(); key_scale_views.emplace_back();
            values.emplace_back(); value_views.emplace_back(); value_scales.emplace_back(); value_scale_views.emplace_back();
            if (layout.scale_groups) {
                key_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); key_scales.back().fill(0);
                key_scale_views.back() = Tensor(key_scales.back().p, DType::FP16, scale_shape);
            }
            if (layout.stores_value()) {
                values.back() = DeviceBuffer(layout.code_page_bytes * pages); values.back().fill(0);
                value_views.back() = Tensor(values.back().p, layout.code_dtype, shape);
                if (layout.scale_groups) {
                    value_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); value_scales.back().fill(0);
                    value_scale_views.back() = Tensor(value_scales.back().p, DType::FP16, scale_shape);
                }
            }
            g4::Gemma4CacheLayer view{layout.format, &key_views.back(),
                                      layout.stores_value() ? &value_views.back() : nullptr,
                                      layout.scale_groups ? &key_scale_views.back() : nullptr,
                                      layout.scale_groups && layout.stores_value() ? &value_scale_views.back() : nullptr,
                                      &table};
            if (!layout.stores_value()) {
                view.value_source = ops::Gemma4ValueSource::Key;
                view.value_scale = 1.0f / config.global_k_norm_scalar.at(static_cast<std::uint32_t>(i));
                view.rope_theta = layer.rope_theta;
                view.rotary_dim = layer.rotary_dim;
            }
            layers.push_back(view);
        }
    }
    template <class T>
    static std::vector<T> host(const DeviceBuffer& buffer) {
        std::vector<T> out(buffer.bytes / sizeof(T));
        if (!out.empty()) { buffer.copy_to_host(out.data(), buffer.bytes); }
        return out;
    }
};

// Softmax KL(P || Q) of two BF16 logit rows, and whether their argmaxes agree.
std::pair<double, bool> kl(const std::vector<std::uint16_t>& p, const std::vector<std::uint16_t>& q) {
    double pm = -1e300, qm = -1e300;
    for (std::size_t i = 0; i < p.size(); ++i) { pm = std::max<double>(pm, bf16_value(p[i])); qm = std::max<double>(qm, bf16_value(q[i])); }
    double ps = 0, qs = 0;
    for (std::size_t i = 0; i < p.size(); ++i) { ps += std::exp(bf16_value(p[i]) - pm); qs += std::exp(bf16_value(q[i]) - qm); }
    double out = 0;
    std::size_t pa = 0, qa = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double lp = bf16_value(p[i]) - pm - std::log(ps), lq = bf16_value(q[i]) - qm - std::log(qs);
        out += std::exp(lp) * (lp - lq);
        if (p[i] != p[pa] && bf16_value(p[i]) > bf16_value(p[pa])) { pa = i; }
        if (q[i] != q[qa] && bf16_value(q[i]) > bf16_value(q[qa])) { qa = i; }
    }
    return {out, pa == qa};
}

// Direct native forwards over BF16 K==V, FP8 K==V and FP8 key-only caches of one prompt.
void execution(const g4::execution::Parameters& p, DeviceContext& device, std::uint32_t chunk) {
    const auto& config = p.model.config().text;
    const int pages = static_cast<int>((chunk + 1 + 63) / 64);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    DeviceBuffer table_memory(table.size() * sizeof(std::int32_t));
    table_memory.copy_from_host(table.data(), table_memory.bytes);
    Tensor table_view(table_memory.p, DType::I32, {pages});
    Caches bf16(p, pages, KvCacheStorage::BFloat16, g4::KvStorageModel::KeyEqualsValue, table_view);
    Caches stored(p, pages, kFp8, g4::KvStorageModel::KeyEqualsValue, table_view);
    Caches derived(p, pages, kFp8, g4::KvStorageModel::KeyOnly, table_view);
    std::size_t workspace_bytes = g4::gemma4_logits_workspace_bytes(p.output_head, 1);
    for (const auto& layer : p.layers) { workspace_bytes = std::max(workspace_bytes, g4::gemma4_layer_workspace_bytes(layer, chunk)); }
    WorkspaceArena workspace(workspace_bytes);
    ops::Fp8RowCublasContext gemm(device.stream);
    DeviceBuffer input_memory(4ULL * chunk), position_memory(4ULL * chunk);
    DeviceBuffer hidden_memory(2ULL * config.hidden_size * chunk), logits_memory(2ULL * config.vocab_size);
    const auto prompt = tokens(chunk + 1, 101);
    std::vector<std::vector<std::uint16_t>> logits;
    for (Caches* caches : {&bf16, &stored, &derived}) {
        for (int step = 0; step < 2; ++step) {
            const int width = step == 0 ? static_cast<int>(chunk) : 1;
            const int first = step == 0 ? 0 : static_cast<int>(chunk);
            std::vector<std::int32_t> positions(width);
            std::iota(positions.begin(), positions.end(), first);
            input_memory.copy_from_host(prompt.data() + first, 4ULL * width);
            position_memory.copy_from_host(positions.data(), 4ULL * width);
            Tensor input(input_memory.p, DType::I32, {width});
            Tensor pos(position_memory.p, DType::I32, {width});
            Tensor hidden(hidden_memory.p, DType::BF16, {static_cast<int>(config.hidden_size), width});
            g4::gemma4_layers(input, hidden, p.embedding, p.layers, caches->layers, pos, gemm, workspace, device.stream);
            if (step == 1) {
                Tensor out(logits_memory.p, DType::BF16, {static_cast<int>(config.vocab_size), 1});
                g4::gemma4_logits(hidden, out, p.final_norm, p.output_head, config.final_logit_softcapping,
                                 config.rms_norm_eps, gemm, workspace, device.stream);
            }
        }
        device.synchronize();
        logits.push_back(Caches::host<std::uint16_t>(logits_memory));
    }
    // Layers up to and including the first global layer see identical inputs: their K bytes (and the
    // sliding layers' V) must equal the K==V FP8 cache's.
    std::size_t g = 0;
    while (!p.layers[g].global_layer) { ++g; }
    for (std::size_t layer = 0; layer <= g; ++layer) {
        require(Caches::host<std::uint8_t>(derived.keys[layer]) == Caches::host<std::uint8_t>(stored.keys[layer]),
                "key-only K codes differ from the K==V FP8 cache at layer " + std::to_string(layer));
        require(Caches::host<std::uint8_t>(derived.key_scales[layer]) == Caches::host<std::uint8_t>(stored.key_scales[layer]),
                "key-only K scales differ from the K==V FP8 cache at layer " + std::to_string(layer));
        if (layer < g) {
            require(Caches::host<std::uint8_t>(derived.values[layer]) == Caches::host<std::uint8_t>(stored.values[layer]),
                    "a sliding layer's V differs under the key-only layout at layer " + std::to_string(layer));
        }
    }
    require(derived.values[g].bytes == 0 && derived.value_scales[g].bytes == 0, "a key-only global layer allocated V");
    // The first global layer's V: stored FP8, and derived from FP8 K, against its BF16 V.
    const auto& layer = p.layers[g];
    const auto value_bf16 = Caches::host<std::uint16_t>(bf16.values[g]);
    const auto value_codes = Caches::host<std::uint8_t>(stored.values[g]);
    const auto value_scales = Caches::host<std::uint16_t>(stored.value_scales[g]);
    const auto key_codes = Caches::host<std::uint8_t>(derived.keys[g]);
    const auto key_scales = Caches::host<std::uint16_t>(derived.key_scales[g]);
    const double value_scale = 1.0 / config.global_k_norm_scalar.at(static_cast<std::uint32_t>(g));
    const int dim = layer.head_dim, half = layer.rotary_dim / 2;
    double reference = 0, stored_error = 0, derived_error = 0;
    for (int page = 0; page < pages; ++page) {
        for (int head = 0; head < layer.kv_heads; ++head) {
            for (int slot = 0; slot < 64; ++slot) {
                const int position = page * 64 + slot;
                if (position > static_cast<int>(chunk)) { continue; }
                const std::size_t row = (static_cast<std::size_t>(page) * layer.kv_heads + head) * 64 + slot;
                std::vector<double> k(dim);
                for (int d = 0; d < dim; ++d) { k[d] = code_value(key_codes[row * dim + d]) * half_value(key_scales[row * 2 + d / 256]); }
                for (int d = 0; d < dim; ++d) {
                    double v = k[d];
                    if (d < 2 * half) {
                        const int pair = d % half;
                        const float angle = static_cast<float>(position) *
                                            std::pow(layer.rope_theta, -2.0f * float(pair) / float(layer.rotary_dim));
                        const double s = std::sin(double(angle)), c = std::cos(double(angle));
                        v = d < half ? k[pair] * c + k[pair + half] * s : k[pair + half] * c - k[pair] * s;
                    }
                    v *= value_scale;
                    const double exact = bf16_value(value_bf16[row * dim + d]);
                    const double quantized = code_value(value_codes[row * dim + d]) * half_value(value_scales[row * 2 + d / 256]);
                    reference += exact * exact;
                    stored_error += (quantized - exact) * (quantized - exact);
                    derived_error += (v - exact) * (v - exact);
                }
            }
        }
    }
    std::printf("METRIC first-global-layer=%zu chunk=%u V rel-RMS vs BF16: stored-fp8 %.6f key-derived %.6f\n", g, chunk,
                std::sqrt(stored_error / reference), std::sqrt(derived_error / reference));
    const auto [stored_kl, stored_top] = kl(logits[0], logits[1]);
    const auto [derived_kl, derived_top] = kl(logits[0], logits[2]);
    const auto [between_kl, between_top] = kl(logits[1], logits[2]);
    std::printf("METRIC chunk=%u final-position KL vs BF16 K/V: stored-fp8 %.6g (top1 %d) key-only %.6g (top1 %d); "
                "KL(stored||key-only) %.6g (top1 %d)\n",
                chunk, stored_kl, stored_top ? 1 : 0, derived_kl, derived_top ? 1 : 0, between_kl, between_top ? 1 : 0);
}

g4::ProgramOptions program_options(std::uint32_t chunk, bool graphs, bool mtp, g4::KvStorageModel storage) {
    g4::ProgramOptions options;
    options.max_context = kContext; options.prefill_chunk = chunk; options.use_cuda_graph = graphs;
    options.kv_cache = kFp8; options.qk_mode = ops::Gemma4QkMode::Fp8; options.global_kv_storage = storage;
    options.draft_tokens = mtp ? 3 : 0;
    options.prefix_cache = true; options.host_bytes = host_bytes_from_env();
    return options;
}

struct Trace {
    Tokens output;
    std::vector<std::uint64_t> counters;
};

Trace finish_trace(g4::Program& program, g4::SequenceHandle sequence) {
    Trace trace;
    while (trace.output.size() < kOutputs) {
        auto pending = [&]() -> g4::PendingBatch {
            if (!trace.output.empty()) {
                const std::array sequences{sequence};
                const std::array budgets{rt::RoundBudget{kOutputs - static_cast<std::uint32_t>(trace.output.size())}};
                return program.decode(sequences, budgets);
            }
            for (;;) {
                auto progress = program.advance_prefill(sequence);
                if (progress.pending) { return std::move(*progress.pending); }
                require(!progress.complete, "prefill completed without a selecting distribution");
            }
        }();
        require(pending.row_failure(0) == rt::PendingRowFailure::None, "non-finite selecting logits");
        const auto n = pending.row_counts().empty() ? 1U : static_cast<std::uint32_t>(pending.row_counts()[0]);
        require(n > 0 && n <= kOutputs - trace.output.size(), "invalid licensed token count");
        trace.output.insert(trace.output.end(), pending.tokens().begin(), pending.tokens().begin() + n);
        rt::CommitDecision decision;
        decision.accepted_tokens = n; decision.terminal = trace.output.size() == kOutputs;
        const std::array decisions{decision};
        const auto committed = program.commit(std::move(pending), decisions);
        const auto& c = committed.rows[0].speculative_counters;
        trace.counters.insert(trace.counters.end(), {c.rounds, c.drafted_tokens, c.accepted_tokens, c.fallback_steps});
    }
    const auto finished = program.finish(sequence);
    require(finished.status == rt::ConsumeStatus::Consumed, "sequence did not finish");
    const auto& s = finished.speculative;
    trace.counters.insert(trace.counters.end(), {s.rounds, s.drafted_tokens, s.accepted_tokens, s.fallback_steps});
    require(!s.enabled || (s.rounds > 0 && s.drafted_tokens > 0), "MTP did not exercise any draft rounds");
    return trace;
}

g4::SequenceHandle bind(g4::Program& program, const g4::RequestBasePlan& plan, g4::ResumeState* resume = nullptr) {
    require(static_cast<bool>(program.start_binding(plan, rt::LaneId{0}, {}, resume)), "binding refused");
    auto progress = program.poll_context({});
    require(progress.published && progress.sequence.has_value(), "binding did not publish");
    if (progress.replaying) {
        while (!program.advance_replay(*progress.sequence).complete) {}
    }
    return *progress.sequence;
}

std::uint64_t global_payload(const g4::execution::Parameters& p, g4::KvStorageModel storage, std::uint32_t pages) {
    std::uint64_t bytes = 0;
    for (const auto& layer : p.layers) {
        if (layer.global_layer) {
            bytes += g4::kv_layer_layout(kFp8, true, layer.head_dim, layer.kv_heads, storage).page_bytes() * pages;
        }
    }
    return bytes;
}

void generation(const g4::execution::Parameters& parameters, DeviceContext& device, const g4::Frontend& frontend,
                std::uint32_t chunk, bool graphs, bool mtp) {
    const auto prompt = tokens(std::max(2101U, 2 * chunk + 3), 17);
    const auto label = std::string(mtp ? "mtp" : "plain") + " chunk=" + std::to_string(chunk) + " graphs=" +
                       std::to_string(graphs ? 1 : 0);
    Tokens stored_tokens;
    std::uint64_t stored_kv = 0;
    {
        g4::Program program(parameters, device, program_options(chunk, graphs, mtp, g4::KvStorageModel::KeyEqualsValue));
        auto plan = program.plan_request(frontend.prepare_tokens(prompt), greedy_execution(kOutputs, false));
        stored_tokens = finish_trace(program, bind(program, plan)).output;
        stored_kv = program.device_plan().kv;
    }
    device.synchronize();
    g4::Program program(parameters, device, program_options(chunk, graphs, mtp, g4::KvStorageModel::KeyOnly));
    const auto memory = program.memory_summary();
    const auto pages = program.device_plan().kv_pages;
    require(memory.gemma4_kv && memory.gemma4_kv->global_key_only, "memory summary does not report the key-only layout");
    require(memory.gemma4_kv->global_payload_bytes == global_payload(parameters, g4::KvStorageModel::KeyOnly, pages) &&
                2 * memory.gemma4_kv->global_payload_bytes == global_payload(parameters, g4::KvStorageModel::KeyEqualsValue, pages),
            "key-only global payload is not the K planes alone");
    require(memory.kv_payload_bytes + 8ULL * pages == program.device_plan().kv, "planner and constructor disagree on key-only KV bytes");
    std::printf("METRIC %s plan kv bytes: K==V %llu key-only %llu (global payload %llu -> %llu)\n", label.c_str(),
                static_cast<unsigned long long>(stored_kv), static_cast<unsigned long long>(program.device_plan().kv),
                static_cast<unsigned long long>(global_payload(parameters, g4::KvStorageModel::KeyEqualsValue, pages)),
                static_cast<unsigned long long>(memory.gemma4_kv->global_payload_bytes));
    auto execution = greedy_execution(kOutputs, false);
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), execution);
    const auto reference = finish_trace(program, bind(program, plan));
    const auto repeat = finish_trace(program, bind(program, plan));
    require(reference.output == repeat.output && reference.counters == repeat.counters,
            "key-only generation is not deterministic (" + label + ")");
    std::size_t agree = 0;
    while (agree < kOutputs && reference.output[agree] == stored_tokens[agree]) { ++agree; }
    std::printf("METRIC %s greedy tokens equal to K==V FP8 before the first difference: %zu of %u; counters", label.c_str(),
                agree, kOutputs);
    for (std::size_t i = reference.counters.size() - 4; i < reference.counters.size(); ++i) {
        std::printf(" %llu", static_cast<unsigned long long>(reference.counters[i]));
    }
    std::printf("\n");

    // Pause after two full chunks with no source: the binding replays from root.
    auto sequence = bind(program, plan);
    for (int call = 0; call < 2; ++call) {
        const auto partial = program.advance_prefill(sequence);
        require(!partial.complete && !partial.pending && partial.processed_prompt_tokens == chunk,
                "pause was not after an intermediate chunk");
    }
    require(program.start_pause(sequence), "pause refused");
    auto paused = program.poll_context({});
    require(paused.paused && paused.paused->frontier() == 2 * chunk, "pause lost its frontier");
    const auto replayed = finish_trace(program, bind(program, plan, &*paused.paused));
    require(replayed.output == reference.output && replayed.counters == reference.counters,
            "key-only root replay changed tokens or MTP counters (" + label + ")");

    // A reusable plan captures a Host snapshot between prompt chunks; after another request
    // displaces the lane, restoring it reproduces the cold key-only run.
    execution.allow_prefix_reuse = true;
    auto reusable = program.plan_request(frontend.prepare_tokens(prompt), execution);
    sequence = bind(program, reusable);
    const auto writes = program.hybrid_stats().host_image_writes;
    std::uint32_t frontier = 0;
    while (program.hybrid_stats().host_image_writes == writes) {
        const auto progress = program.advance_prefill(sequence);
        require(!progress.complete && !progress.pending, "no snapshot captured before the final prompt chunk");
        frontier += progress.processed_prompt_tokens;
    }
    require(program.abort(sequence).status == rt::ConsumeStatus::Consumed, "capture abort did not consume sequence");
    const auto other = start(program, frontend, tokens(chunk + 1, 99), 1, false);
    require(other.sequence.has_value(), "overwrite did not bind");
    (void)complete(program, *other.sequence, 1);
    program.maintain();
    bool restored = false;
    for (const auto& source : program.hybrid_sources(reusable, nullptr)) {
        if (source.reused_tokens == 0) { continue; }
        require(source.reused_tokens <= frontier, "snapshot passed the interrupted frontier");
        if (!program.start_binding(reusable, rt::LaneId{0}, source)) { continue; }
        auto progress = program.poll_context({});
        require(progress.published && progress.sequence, "snapshot restore did not publish");
        require(progress.operations.state_restores == 1 && !progress.replaying, "snapshot binding did not restore");
        const auto resumed = finish_trace(program, *progress.sequence);
        require(resumed.output == reference.output && resumed.counters == reference.counters,
                "key-only Host snapshot restore changed tokens or MTP counters (" + label + ")");
        std::printf("METRIC %s Host restore reused %u of %zu prompt tokens (block restores %llu, image restores %llu)\n",
                    label.c_str(), source.reused_tokens, prompt.size(),
                    static_cast<unsigned long long>(program.hybrid_stats().host_block_restores),
                    static_cast<unsigned long long>(program.hybrid_stats().host_image_restores));
        restored = true;
        break;
    }
    require(restored, "no intermediate captured snapshot was restored (" + label + ")");
}

void scoring(const g4::execution::Parameters& parameters, DeviceContext& device, const g4::Frontend& frontend,
             std::uint32_t chunk) {
    const auto prompt = tokens(2 * chunk + 3, 91);
    const std::uint32_t first_target = chunk + 2;
    ScoreOptions score;
    score.top_k = 1;
    std::vector<float> logprobs[2];
    for (const auto storage : {g4::KvStorageModel::KeyEqualsValue, g4::KvStorageModel::KeyOnly}) {
        auto options = program_options(chunk, false, false, storage);
        options.causal_scoring = true; options.prefix_cache = false; options.host_bytes = 0;
        g4::Program program(parameters, device, options);
        const auto result = program.causal_score(frontend.prepare_tokens(prompt), first_target, score);
        require(std::all_of(result.logprobs.begin(), result.logprobs.end(), [](float v) { return std::isfinite(v) && v <= 0; }),
                "invalid key-only teacher-forced logprob");
        logprobs[storage == g4::KvStorageModel::KeyOnly ? 1 : 0] = result.logprobs;
        device.synchronize();
    }
    double sum = 0, worst = 0, mean_stored = 0, mean_derived = 0;
    for (std::size_t i = 0; i < logprobs[0].size(); ++i) {
        const double delta = std::fabs(double(logprobs[1][i]) - logprobs[0][i]);
        sum += delta; worst = std::max(worst, delta);
        mean_stored -= logprobs[0][i]; mean_derived -= logprobs[1][i];
    }
    const auto n = static_cast<double>(logprobs[0].size());
    std::printf("METRIC chunk=%u causal score over %zu targets: mean NLL K==V %.6f key-only %.6f, |delta| mean %.6f max %.6f\n",
                chunk, logprobs[0].size(), mean_stored / n, mean_derived / n, sum / n, worst);
}
} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') { std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set"); return 77; }
    try {
        std::uint32_t selected_chunk = 0;
        int selected_graphs = -1, selected_mtp = -1;
        for (int i = 1; i < argc; i += 2) {
            require(i + 1 < argc, "expected --chunk 128|2048, --graphs 0|1 or --mtp 0|1");
            const std::string option(argv[i]), value(argv[i + 1]);
            if (option == "--chunk" && (value == "128" || value == "2048")) { selected_chunk = value == "128" ? 128 : 2048; }
            else if (option == "--graphs" && (value == "0" || value == "1")) { selected_graphs = value == "1"; }
            else if (option == "--mtp" && (value == "0" || value == "1")) { selected_mtp = value == "1"; }
            else { throw std::runtime_error("expected --chunk 128|2048, --graphs 0|1 or --mtp 0|1"); }
        }
        DeviceContext device(0);
        for (const bool mtp : {false, true}) {
            if (selected_mtp >= 0 && selected_mtp != static_cast<int>(mtp)) { continue; }
            models::LoadOptions load;
            load.speculative = mtp ? SpeculativeBackend::Mtp : SpeculativeBackend::None;
            auto model = g4::load_model(artifact, load, device);
            g4::execution::Parameters parameters(*model, device.stream);
            require(!mtp || parameters.draft.has_value(), "artifact must include the MTP assistant");
            g4::Frontend frontend(model->resources(), {.max_context = kContext});
            if (!mtp) {
                // Refusals: BF16 globals, the unimplemented value-only layout.
                auto refused = [&](g4::KvStorageModel storage, KvCacheStorage kv) {
                    auto options = program_options(128, false, false, storage);
                    options.kv_cache = kv; options.qk_mode = ops::Gemma4QkMode::BFloat16;
                    try { (void)g4::Program::plan_device(*model, options, model->config().text.vocab_size); }
                    catch (const std::invalid_argument& error) { std::printf("refused: %s\n", error.what()); return true; }
                    return false;
                };
                require(refused(g4::KvStorageModel::KeyOnly, KvCacheStorage::BFloat16), "key-only BF16 KV was accepted");
                require(refused(g4::KvStorageModel::ValueOnly, kFp8), "value-only KV was accepted");
                std::printf("global key-norm constants:");
                for (const auto& [layer, value] : model->config().text.global_k_norm_scalar) { std::printf(" %u=%.10g", layer, value); }
                std::printf("\n");
            }
            for (const std::uint32_t chunk : {128U, 2048U}) {
                if (selected_chunk != 0 && selected_chunk != chunk) { continue; }
                if (!mtp) {
                    execution(parameters, device, chunk);
                    scoring(parameters, device, frontend, chunk);
                }
                for (const bool graphs : {false, true}) {
                    if (selected_graphs >= 0 && selected_graphs != static_cast<int>(graphs)) { continue; }
                    generation(parameters, device, frontend, chunk, graphs, mtp);
                }
            }
            device.synchronize();
        }
        std::puts("PASS key-only global KV invariants");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
