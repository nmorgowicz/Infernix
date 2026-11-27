// The opt-in FP8 sliding ring (KvCacheLayout::sliding = Fp8E4M3Row256: FP8 E4M3 K and V with one FP16
// scale per D256 row) on a real artifact; no downloads. INFERNIX_GEMMA4_ARTIFACT (with its MTP assistant
// unless --mtp 0) selects it, missing skips (77); INFERNIX_GEMMA4_HOST_MIB sizes the Host tier. Optional:
// --chunk 128|2048 --graphs 0|1 --mtp 0|1 (default: every combination).
//
// Checked: refusals (BF16 globals); a direct native forward writes the first sliding layer's FP8 ring rows as
// exactly the row codec of the BF16 ring's rows; Program accounting (the sliding payload is the FP8 ring's
// codes and row scales, planner and constructor agree); generation is deterministic, and a root replay and a
// Host snapshot restore equal the cold FP8-ring run, with stored and key-only globals, MTP off and on (the
// drafter reads the same ring). Measured (METRIC lines, not gates): every sliding layer's K/V quantization
// error at 256, 128 and 64 columns per scale on the BF16 ring's rows; final-position logits KL of the FP8
// ring against the BF16 ring; ring and plan bytes; greedy agreement; causal-score deltas.
#include "prefix_real_support.h"
#include "models/gemma4/program/kv_layout.h"

#include <algorithm>
#include <array>
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
constexpr auto kBf16 = KvCacheStorage::BFloat16;

float bf16_value(std::uint16_t bits) { return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16); }
float half_value(std::uint16_t bits) {
    const std::uint32_t sign = (bits & 0x8000U) << 16, exponent = (bits >> 10) & 31U, mantissa = bits & 1023U;
    if (exponent == 0) { return (sign ? -1.0f : 1.0f) * std::ldexp(float(mantissa), -24); }
    if (exponent == 31) { return std::numeric_limits<float>::quiet_NaN(); }
    return std::bit_cast<float>(sign | ((exponent + 112U) << 23) | (mantissa << 13));
}
std::uint16_t half_bits(float x) {
    // Round to nearest even FP16 (finite, normal or subnormal; x is a bounded positive scale here).
    if (x == 0.0f) { return 0; }
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(x);
    const int exponent = static_cast<int>((bits >> 23) & 255U) - 127;
    if (exponent < -14) {
        const float scaled = std::ldexp(x, 24);
        return static_cast<std::uint16_t>(std::nearbyint(scaled));
    }
    std::uint32_t mantissa = bits & 0x7FFFFFU;
    std::uint32_t half = static_cast<std::uint32_t>(exponent + 15) << 10 | (mantissa >> 13);
    const std::uint32_t rest = mantissa & 0x1FFFU;
    if (rest > 0x1000U || (rest == 0x1000U && (half & 1U))) { ++half; }
    return static_cast<std::uint16_t>(half);
}
float code_value(std::uint8_t code) {
    const int exp = (code >> 3) & 15, mantissa = code & 7;
    if (exp == 15 && mantissa == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const float value = exp == 0 ? std::ldexp(float(mantissa), -9) : std::ldexp(1.0f + float(mantissa) / 8.0f, exp - 7);
    return code & 128 ? -value : value;
}
// E4M3 round-to-nearest-even with saturation at 448: the nearest of the 127 nonnegative codes, ties to the
// even code, as the device encoder (__NV_SATFINITE) rounds.
std::uint8_t encode(float x) {
    static const auto table = [] {
        std::array<float, 127> out{};
        for (int code = 0; code < 127; ++code) { out[code] = code_value(static_cast<std::uint8_t>(code)); }
        return out;
    }();
    const float magnitude = std::fabs(x);
    const auto above = std::lower_bound(table.begin(), table.end(), magnitude);
    int code = 126;
    if (above != table.end()) {
        code = static_cast<int>(above - table.begin());
        if (code > 0) {
            const float low = magnitude - table[code - 1], high = table[code] - magnitude;
            if (low < high || (low == high && ((code - 1) & 1) == 0)) { --code; }
        }
    }
    return static_cast<std::uint8_t>(code | (std::signbit(x) ? 128 : 0));
}
// The ring codec over `columns`-wide groups of a BF16 row: FP16-represented bounded amax / 448 scales, codes of
// x / scale; nonfinite groups poison (NaN codes, unit scale). Returns the represented values.
void quantize_row(const std::uint16_t* x, int width, int columns, std::uint8_t* codes, std::uint16_t* scales,
                  float* represented) {
    for (int group = 0; group < width / columns; ++group) {
        const std::uint16_t* g = x + group * columns;
        float maximum = 0.0f;
        bool poison = false;
        for (int i = 0; i < columns; ++i) {
            const float v = bf16_value(g[i]);
            poison |= !std::isfinite(v);
            maximum = std::fmax(maximum, std::fabs(v));
        }
        const std::uint16_t scale = half_bits(poison ? 1.0f : maximum == 0.0f ? 0.0f :
                                              std::fmin(65504.0f, std::fmax(0x1p-24f, maximum / 448.0f)));
        scales[group] = scale;
        const float inverse = maximum == 0.0f ? 0.0f : 1.0f / half_value(scale);
        for (int i = 0; i < columns; ++i) {
            const std::uint8_t code = poison ? 0x7f : maximum == 0.0f ? 0 : encode(bf16_value(g[i]) * inverse);
            codes[group * columns + i] = code;
            if (represented) { represented[group * columns + i] = code_value(code) * half_value(scale); }
        }
    }
}

// One native cache set: contiguous pages for every layer, so every byte is inspectable.
struct Caches {
    std::vector<DeviceBuffer> keys, values, key_scales, value_scales;
    std::vector<Tensor> key_views, value_views, key_scale_views, value_scale_views;
    std::vector<g4::Gemma4CacheLayer> layers;
    Caches(const g4::execution::Parameters& p, int pages, const g4::KvCacheLayout& request, const Tensor& table) {
        const auto count = p.layers.size();
        for (auto* v : {&keys, &values, &key_scales, &value_scales}) { v->reserve(count); }
        for (auto* v : {&key_views, &value_views, &key_scale_views, &value_scale_views}) { v->reserve(count); }
        layers.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto& layer = p.layers[i];
            const auto layout = g4::kv_layer_layout(request, layer.global_layer, layer.head_dim, layer.kv_heads);
            const std::initializer_list<std::int32_t> shape{layer.head_dim, 64, layer.kv_heads, pages};
            const std::initializer_list<std::int32_t> scale_shape{layout.scale_groups, 64, layer.kv_heads, pages};
            keys.emplace_back(layout.code_page_bytes * pages); keys.back().fill(0);
            values.emplace_back(layout.code_page_bytes * pages); values.back().fill(0);
            key_views.emplace_back(keys.back().p, layout.code_dtype, shape);
            value_views.emplace_back(values.back().p, layout.code_dtype, shape);
            key_scales.emplace_back(); value_scales.emplace_back(); key_scale_views.emplace_back(); value_scale_views.emplace_back();
            if (layout.scale_groups) {
                key_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); key_scales.back().fill(0);
                value_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); value_scales.back().fill(0);
                key_scale_views.back() = Tensor(key_scales.back().p, DType::FP16, scale_shape);
                value_scale_views.back() = Tensor(value_scales.back().p, DType::FP16, scale_shape);
            }
            layers.push_back({layout.format, &key_views.back(), &value_views.back(),
                              layout.scale_groups ? &key_scale_views.back() : nullptr,
                              layout.scale_groups ? &value_scale_views.back() : nullptr, &table});
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
        if (bf16_value(p[i]) > bf16_value(p[pa])) { pa = i; }
        if (bf16_value(q[i]) > bf16_value(q[qa])) { qa = i; }
    }
    return {out, pa == qa};
}

// Direct native forwards of one prompt (a chunk, then one decode step) over FP8 globals with the BF16 ring and
// with the FP8 ring.
void execution(const g4::execution::Parameters& p, DeviceContext& device, std::uint32_t chunk) {
    const auto& config = p.model.config().text;
    const int pages = static_cast<int>((chunk + 1 + 63) / 64);
    std::vector<std::int32_t> table(pages);
    std::iota(table.begin(), table.end(), 0);
    DeviceBuffer table_memory(table.size() * sizeof(std::int32_t));
    table_memory.copy_from_host(table.data(), table_memory.bytes);
    Tensor table_view(table_memory.p, DType::I32, {pages});
    Caches bf16(p, pages, {kFp8, g4::KvStorageModel::KeyEqualsValue, kBf16}, table_view);
    Caches ring(p, pages, {kFp8, g4::KvStorageModel::KeyEqualsValue, kFp8}, table_view);
    std::size_t workspace_bytes = g4::gemma4_logits_workspace_bytes(p.output_head, 1);
    for (const auto& layer : p.layers) { workspace_bytes = std::max(workspace_bytes, g4::gemma4_layer_workspace_bytes(layer, chunk)); }
    WorkspaceArena workspace(workspace_bytes);
    ops::Fp8RowCublasContext gemm(device.stream);
    DeviceBuffer input_memory(4ULL * chunk), position_memory(4ULL * chunk);
    DeviceBuffer hidden_memory(2ULL * config.hidden_size * chunk), logits_memory(2ULL * config.vocab_size);
    const auto prompt = tokens(chunk + 1, 101);
    std::vector<std::vector<std::uint16_t>> logits;
    for (Caches* caches : {&bf16, &ring}) {
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
    // The first sliding layer sees identical inputs in both runs: its FP8 ring rows are exactly the row codec of
    // the BF16 ring's rows (unwritten rows are zero in both).
    std::size_t s = 0;
    while (p.layers[s].global_layer) { ++s; }
    {
        const int dim = p.layers[s].head_dim;
        for (const bool value : {false, true}) {
            const auto reference = Caches::host<std::uint16_t>(value ? bf16.values[s] : bf16.keys[s]);
            const auto codes = Caches::host<std::uint8_t>(value ? ring.values[s] : ring.keys[s]);
            const auto scales = Caches::host<std::uint16_t>(value ? ring.value_scales[s] : ring.key_scales[s]);
            std::vector<std::uint8_t> want_codes(codes.size());
            std::vector<std::uint16_t> want_scales(scales.size());
            for (std::size_t row = 0; row < scales.size(); ++row) {
                quantize_row(reference.data() + row * dim, dim, dim, want_codes.data() + row * dim, want_scales.data() + row,
                             nullptr);
            }
            require(codes == want_codes && scales == want_scales,
                    std::string("the FP8 ring's ") + (value ? "V" : "K") + " rows of sliding layer " + std::to_string(s) +
                        " are not the row codec of the BF16 ring's rows");
        }
    }
    // Scale granularity on real rows (chunk 128 only; offline): every sliding layer's K and V rows of the BF16 run,
    // quantized with one scale per 256 (the ring's codec), 128 and 64 columns.
    if (chunk == 128) {
        constexpr std::array<int, 3> kColumns{256, 128, 64};
        std::array<double, 3> error{}, norm{}, worst_layer{};
        double row_peak = 0.0;
        for (std::size_t layer = 0; layer < p.layers.size(); ++layer) {
            if (p.layers[layer].global_layer) { continue; }
            const int dim = p.layers[layer].head_dim;
            const int heads = p.layers[layer].kv_heads;
            std::array<double, 3> layer_error{}, layer_norm{};
            for (const bool value : {false, true}) {
                const auto rows = Caches::host<std::uint16_t>(value ? bf16.values[layer] : bf16.keys[layer]);
                std::vector<std::uint8_t> codes(dim);
                std::vector<std::uint16_t> scales(dim / 64);
                std::vector<float> represented(dim);
                for (int page = 0; page < pages; ++page) {
                    for (int head = 0; head < heads; ++head) {
                        for (int slot = 0; slot < 64; ++slot) {
                            if (page * 64 + slot > static_cast<int>(chunk)) { continue; }
                            const std::uint16_t* row = rows.data() + ((static_cast<std::size_t>(page) * heads + head) * 64 + slot) * dim;
                            double sum = 0.0, peak = 0.0;
                            for (int d = 0; d < dim; ++d) { sum += double(bf16_value(row[d])) * bf16_value(row[d]); peak = std::max(peak, std::fabs(double(bf16_value(row[d])))); }
                            if (sum > 0.0) { row_peak = std::max(row_peak, peak / std::sqrt(sum / dim)); }
                            for (std::size_t g = 0; g < kColumns.size(); ++g) {
                                quantize_row(row, dim, kColumns[g], codes.data(), scales.data(), represented.data());
                                for (int d = 0; d < dim; ++d) {
                                    const double e = double(represented[d]) - bf16_value(row[d]);
                                    layer_error[g] += e * e;
                                    layer_norm[g] += double(bf16_value(row[d])) * bf16_value(row[d]);
                                }
                            }
                        }
                    }
                }
            }
            for (std::size_t g = 0; g < kColumns.size(); ++g) {
                error[g] += layer_error[g]; norm[g] += layer_norm[g];
                worst_layer[g] = std::max(worst_layer[g], std::sqrt(layer_error[g] / layer_norm[g]));
            }
        }
        std::printf("METRIC scale granularity on every sliding layer's K/V rows (positions 0..%u): rel-RMS error "
                    "256 cols/scale %.6f (worst layer %.6f), 128 %.6f (%.6f), 64 %.6f (%.6f); max row peak/RMS %.2f\n",
                    chunk, std::sqrt(error[0] / norm[0]), worst_layer[0], std::sqrt(error[1] / norm[1]), worst_layer[1],
                    std::sqrt(error[2] / norm[2]), worst_layer[2], row_peak);
    }
    const auto [ring_kl, ring_top] = kl(logits[0], logits[1]);
    std::printf("METRIC chunk=%u final-position KL(BF16 ring || FP8 ring) %.6g (top1 %d)\n", chunk, ring_kl, ring_top ? 1 : 0);
}

g4::ProgramOptions program_options(std::uint32_t chunk, bool graphs, bool mtp, g4::KvStorageModel storage,
                                   KvCacheStorage sliding) {
    g4::ProgramOptions options;
    options.max_context = kContext; options.prefill_chunk = chunk; options.use_cuda_graph = graphs;
    options.kv_cache = kFp8; options.qk_mode = ops::Gemma4QkMode::Fp8; options.global_kv_storage = storage;
    options.sliding_kv_cache = sliding;
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

// The sliding payload: every sliding layer's `ring_pages` pages, plus the shared staging ring (one layer's pages) when staged.
std::uint64_t sliding_payload(const g4::execution::Parameters& p, KvCacheStorage sliding, std::uint32_t ring_pages,
                              std::uint32_t staging_pages = 0) {
    std::uint64_t bytes = 0;
    bool staging_added = staging_pages == 0;
    for (const auto& layer : p.layers) {
        if (!layer.global_layer) {
            const auto page = g4::kv_layer_layout({kFp8, g4::KvStorageModel::KeyEqualsValue, sliding}, false, layer.head_dim,
                                                  layer.kv_heads).page_bytes();
            bytes += page * ring_pages;
            if (!staging_added) { bytes += page * staging_pages; staging_added = true; }
        }
    }
    return bytes;
}

void generation(const g4::execution::Parameters& parameters, DeviceContext& device, const g4::Frontend& frontend,
                std::uint32_t chunk, bool graphs, bool mtp, g4::KvStorageModel storage) {
    const auto prompt = tokens(std::max(2101U, 2 * chunk + 3), 17);
    const auto label = std::string(mtp ? "mtp" : "plain") + " chunk=" + std::to_string(chunk) + " graphs=" +
                       std::to_string(graphs ? 1 : 0) + (storage == g4::KvStorageModel::KeyOnly ? " key-only" : " stored");
    Tokens bf16_tokens;
    std::uint64_t bf16_kv = 0, bf16_sliding = 0;
    {
        g4::Program program(parameters, device, program_options(chunk, graphs, mtp, storage, kBf16));
        auto plan = program.plan_request(frontend.prepare_tokens(prompt), greedy_execution(kOutputs, false));
        bf16_tokens = finish_trace(program, bind(program, plan)).output;
        bf16_kv = program.device_plan().kv;
        bf16_sliding = program.memory_summary().gemma4_kv->sliding_payload_bytes;
    }
    device.synchronize();
    g4::Program program(parameters, device, program_options(chunk, graphs, mtp, storage, kFp8));
    const auto memory = program.memory_summary();
    const auto pages = program.device_plan().kv_pages;
    const auto& text = parameters.model.config().text;
    const auto ring_pages = std::min(pages, (chunk + text.sliding_window - 1 + 63) / 64);
    require(memory.gemma4_kv && memory.gemma4_kv->sliding == kFp8 && memory.gemma4_kv->global == kFp8 &&
                memory.gemma4_kv->global_key_only == (storage == g4::KvStorageModel::KeyOnly),
            "memory summary does not report the FP8 ring");
    const auto& plan_pages = program.device_plan();
    require(memory.gemma4_kv->sliding_payload_bytes == sliding_payload(parameters, kFp8, plan_pages.ring_pages, plan_pages.staging_pages) &&
                bf16_sliding == sliding_payload(parameters, kBf16, plan_pages.ring_pages, plan_pages.staging_pages),
            "the sliding payload is not the rings' codes and row scales");
    require(memory.kv_payload_bytes + (plan_pages.staging_pages != 0 ? 12ULL : 8ULL) * pages == program.device_plan().kv,
            "planner and constructor disagree on FP8-ring KV bytes");
    require(bf16_kv - program.device_plan().kv == bf16_sliding - memory.gemma4_kv->sliding_payload_bytes,
            "the plan's KV saving is not the ring saving");
    std::printf("METRIC %s ring pages %u: sliding payload BF16 %llu FP8 %llu bytes (%.4f); plan kv BF16 %llu FP8 %llu\n",
                label.c_str(), ring_pages, static_cast<unsigned long long>(bf16_sliding),
                static_cast<unsigned long long>(memory.gemma4_kv->sliding_payload_bytes),
                double(memory.gemma4_kv->sliding_payload_bytes) / double(bf16_sliding),
                static_cast<unsigned long long>(bf16_kv), static_cast<unsigned long long>(program.device_plan().kv));
    auto execution = greedy_execution(kOutputs, false);
    auto plan = program.plan_request(frontend.prepare_tokens(prompt), execution);
    const auto reference = finish_trace(program, bind(program, plan));
    const auto repeat = finish_trace(program, bind(program, plan));
    require(reference.output == repeat.output && reference.counters == repeat.counters,
            "FP8-ring generation is not deterministic (" + label + ")");
    std::size_t agree = 0;
    while (agree < kOutputs && reference.output[agree] == bf16_tokens[agree]) { ++agree; }
    std::printf("METRIC %s greedy tokens equal to the BF16 ring before the first difference: %zu of %u; counters", label.c_str(),
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
            "FP8-ring root replay changed tokens or MTP counters (" + label + ")");

    // A reusable plan captures a Host snapshot (the FP8 ring's window) between prompt chunks; after another request
    // displaces the lane, restoring it reproduces the cold FP8-ring run.
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
                "FP8-ring Host snapshot restore changed tokens or MTP counters (" + label + ")");
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
    for (const auto sliding : {kBf16, kFp8}) {
        auto options = program_options(chunk, false, false, g4::KvStorageModel::KeyEqualsValue, sliding);
        options.causal_scoring = true; options.prefix_cache = false; options.host_bytes = 0;
        g4::Program program(parameters, device, options);
        const auto result = program.causal_score(frontend.prepare_tokens(prompt), first_target, score);
        require(std::all_of(result.logprobs.begin(), result.logprobs.end(), [](float v) { return std::isfinite(v) && v <= 0; }),
                "invalid FP8-ring teacher-forced logprob");
        logprobs[sliding == kFp8 ? 1 : 0] = result.logprobs;
        device.synchronize();
    }
    double sum = 0, worst = 0, mean_bf16 = 0, mean_fp8 = 0;
    for (std::size_t i = 0; i < logprobs[0].size(); ++i) {
        const double delta = std::fabs(double(logprobs[1][i]) - logprobs[0][i]);
        sum += delta; worst = std::max(worst, delta);
        mean_bf16 -= logprobs[0][i]; mean_fp8 -= logprobs[1][i];
    }
    const auto n = static_cast<double>(logprobs[0].size());
    std::printf("METRIC chunk=%u causal score over %zu targets: mean NLL BF16 ring %.6f FP8 ring %.6f, |delta| mean %.6f max %.6f\n",
                chunk, logprobs[0].size(), mean_bf16 / n, mean_fp8 / n, sum / n, worst);
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
                auto options = program_options(128, false, false, g4::KvStorageModel::KeyEqualsValue, kFp8);
                options.kv_cache = kBf16; options.qk_mode = ops::Gemma4QkMode::BFloat16;
                bool refused = false;
                try { (void)g4::Program::plan_device(*model, options, model->config().text.vocab_size); }
                catch (const std::invalid_argument& error) { std::printf("refused: %s\n", error.what()); refused = true; }
                require(refused, "the FP8 ring with BF16 globals was accepted");
            }
            for (const std::uint32_t chunk : {128U, 2048U}) {
                if (selected_chunk != 0 && selected_chunk != chunk) { continue; }
                if (!mtp) {
                    execution(parameters, device, chunk);
                    scoring(parameters, device, frontend, chunk);
                }
                for (const bool graphs : {false, true}) {
                    if (selected_graphs >= 0 && selected_graphs != static_cast<int>(graphs)) { continue; }
                    generation(parameters, device, frontend, chunk, graphs, mtp, g4::KvStorageModel::KeyEqualsValue);
                    // The most aggressive stack: key-only globals, FP8 QK and the FP8 ring together.
                    if (graphs) { generation(parameters, device, frontend, chunk, graphs, mtp, g4::KvStorageModel::KeyOnly); }
                }
            }
            device.synchronize();
        }
        std::puts("PASS FP8 sliding ring invariants");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
