// Gemma 4 FP8 sliding ring oracles: D256 windowed caches stored as FP8 E4M3 codes with one FP16 scale
// per row (ops::Gemma4KvFormat::Fp8E4m3Row256 on a sliding layer). Checked: cache validation (only
// D256 windowed caches with BF16 QK, stored values and [1,64,Hkv,pages] scales); the fused writer
// (gemma4_qkv_prep, D256 with its own V rows) stores exactly the row codec of the BF16 ring's values,
// nonfinite rows included, and nothing else; attention on the decode (T < 64) and tensor-core
// (T >= 64) kernels over a wrapped ring with an unmapped and an out-of-range page and windows that
// start mid-page, equal bit for bit to BF16 attention over the ring expanded on the host (FP32
// product rounded once to BF16) and within the BF16 attention tolerance of an FP64 oracle on those
// represented values, never reading the never-written rows; graph replay; and, from one writer
// input, FP8-ring attention against the BF16 ring (measurement).
#include "core/temporary_switches.h"
#include "infernix/ops/gemma4_attention.h"
#include "infernix/ops/gemma4_qkv_prep.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {
constexpr int D = 256;
constexpr int Heads = 16;  // Gemma 4 31B sliding layers: 16 KV heads, 32 query heads, window 1024
constexpr int QHeads = 32;
constexpr int Window = 1024;
constexpr float kTheta = 10000.0f;

std::uint16_t half_bits(float x) {
    const __half h = __float2half_rn(x);
    return reinterpret_cast<const __half_raw&>(h).x;
}
float half_value(std::uint16_t bits) {
    __half_raw raw;
    raw.x = bits;
    return __half2float(__half(raw));
}
float code_value(std::uint8_t code) {
    const int exp = (code >> 3) & 15;
    const int mantissa = code & 7;
    if (exp == 15 && mantissa == 7) { return std::numeric_limits<float>::quiet_NaN(); }
    const float value = exp == 0 ? std::ldexp(float(mantissa), -9) :
                                  std::ldexp(1.0f + float(mantissa) / 8.0f, exp - 7);
    return code & 128 ? -value : value;
}
std::uint8_t encode(float x) {
    const float magnitude = std::fabs(x);
    int best = 0;
    float error = std::numeric_limits<float>::infinity();
    for (int code = 0; code <= 126; ++code) {
        const float next = std::fabs(magnitude - code_value(static_cast<std::uint8_t>(code)));
        if (next < error || (next == error && !(code & 1))) { best = code; error = next; }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(x) ? 128 : 0));
}
// The row codec: one FP16-represented bounded scale amax / 448 per 256-element row, RNE-saturating
// E4M3 codes of x / scale; a row with any nonfinite value is canonical NaN codes with unit scale.
void quantize_row(const std::uint16_t* x, std::uint8_t* codes, std::uint16_t& scale) {
    float maximum = 0.0f;
    bool poison = false;
    for (int i = 0; i < D; ++i) {
        const float v = bf16_to_f32(x[i]);
        poison |= !std::isfinite(v);
        maximum = std::fmax(maximum, std::fabs(v));
    }
    scale = half_bits(poison ? 1.0f : maximum == 0.0f ? 0.0f : std::fmin(65504.0f, std::fmax(0x1p-24f, maximum / 448.0f)));
    const float inverse = maximum == 0.0f ? 0.0f : 1.0f / half_value(scale);
    for (int i = 0; i < D; ++i) {
        codes[i] = poison ? 0x7f : maximum == 0.0f ? 0 : encode(bf16_to_f32(x[i]) * inverse);
    }
}
// The value every FP8 reader consumes: FP32 code x FP32 scale, rounded once to BF16.
std::uint16_t represented(std::uint8_t code, std::uint16_t scale) {
    return f32_to_bf16(code_value(code) * half_value(scale));
}
// A (code, scale) whose FP32 product rounds to BF16 differently than its FP16 product does.
std::pair<std::uint8_t, std::uint16_t> expansion_boundary() {
    for (int scale = 0x1001; scale < 0x3c00; ++scale) {
        for (int code = 1; code <= 126; ++code) {
            const float product = code_value(static_cast<std::uint8_t>(code)) * half_value(static_cast<std::uint16_t>(scale));
            if (f32_to_bf16(product) != f32_to_bf16(half_value(half_bits(product)))) {
                return {static_cast<std::uint8_t>(code), static_cast<std::uint16_t>(scale)};
            }
        }
    }
    return {0, 0};
}
std::int64_t row_of(int physical, int head, int slot) { return ops::detail::gemma4_kv_row(physical, Heads, head, slot); }

int validation_test() {
    // Host-only validation never reads these deliberately opaque addresses.
    int storage = 0;
    constexpr int Pages = 4;
    Tensor k(&storage, DType::U8, {D, 64, Heads, Pages}), v = k;
    Tensor ks(&storage, DType::FP16, {1, 64, Heads, Pages}), vs = ks;
    Tensor table(&storage, DType::I32, {Pages});
    const ops::Gemma4CacheView valid{ops::Gemma4KvFormat::Fp8E4m3Row256, &k, &v, &ks, &vs, &table};
    ops::detail::validate_gemma4_cache(valid, D, false, "test");
    int failures = 0;
    const auto rejects = [&](const ops::Gemma4CacheView& cache, int dim, bool global, const char* what) {
        try { ops::detail::validate_gemma4_cache(cache, dim, global, "test"); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "invalid FP8 ring accepted: " << what << '\n'; ++failures;
    };
    rejects(valid, D, true, "D256 global (window 0)");
    rejects(valid, 512, false, "D512 sliding");
    auto bad = valid;
    bad.qk_mode = ops::Gemma4QkMode::Fp8;
    rejects(bad, D, false, "FP8 QK on the ring");
    bad = valid;
    bad.value_source = ops::Gemma4ValueSource::Key;
    bad.v_pages = nullptr; bad.v_scales = nullptr;
    bad.value_scale = 2.0f; bad.rope_theta = kTheta; bad.rotary_dim = ops::kGemma4KeyValueRotaryDim;
    rejects(bad, D, false, "key-derived values on the ring");
    bad = valid; bad.v_scales = nullptr; rejects(bad, D, false, "missing V scales");
    bad = valid; bad.k_scales = nullptr; rejects(bad, D, false, "missing K scales");
    Tensor wide(&storage, DType::FP16, {2, 64, Heads, Pages});
    bad = valid; bad.k_scales = &wide; rejects(bad, D, false, "two scale groups per D256 row");
    Tensor bf16(&storage, DType::BF16, {D, 64, Heads, Pages});
    bad = valid; bad.k_pages = &bf16; rejects(bad, D, false, "BF16 pages on an FP8 descriptor");
    bad = valid; bad.format = ops::Gemma4KvFormat::BFloat16; bad.k_pages = bad.v_pages = &bf16;
    rejects(bad, D, false, "scale planes on a BF16 ring");
    bad.k_scales = bad.v_scales = nullptr;
    ops::detail::validate_gemma4_cache(bad, D, false, "test");
    return failures;
}

// One prompt through gemma4_qkv_prep into a BF16 ring and an FP8 ring (same positions, an unmapped
// and an out-of-range page, a negative position, nonfinite K and V rows): Q is unchanged, every
// written FP8 row is the row codec of the BF16 ring's row, and no other byte changes.
int qkv_test() {
    constexpr int Pages = 6;
    constexpr int Rows = 64 * Heads * Pages;
    const std::vector<int> positions{-1, 0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 300, 383, 384, 449};
    const std::vector<int> table{3, 0, -1, 5, Pages, 1, 2, 4};
    const int tokens = static_cast<int>(positions.size());
    const int width = (QHeads + 2 * Heads) * D;
    std::vector<float> input(static_cast<std::size_t>(width) * tokens), q_weights(D), k_weights(D);
    fill_uniform(input, 7101, -3.0f, 3.0f);
    fill_uniform(q_weights, 7102, 0.4f, 1.7f);
    fill_uniform(k_weights, 7103, 0.4f, 1.7f);
    // Token 3's K head 5 (position 63) and token 10's V head 11 (position 192) carry an infinity and a
    // NaN: their rows poison.
    input[static_cast<std::size_t>(3) * width + (QHeads + 5) * D + 17] = std::numeric_limits<float>::infinity();
    input[static_cast<std::size_t>(10) * width + (QHeads + Heads + 11) * D + 200] = std::numeric_limits<float>::quiet_NaN();
    auto dx = to_device_bf16(input), dqw = to_device_bf16(q_weights), dkw = to_device_bf16(k_weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    GuardedDeviceBuffer q8(static_cast<std::size_t>(QHeads) * D * tokens * 2), qb(q8.bytes());
    GuardedDeviceBuffer k8(Rows * D), v8(Rows * D), ks8(Rows * 2), vs8(Rows * 2), kb(Rows * D * 2), vb(Rows * D * 2);
    const std::vector<std::uint8_t> code_background(Rows * D, 0xa5);
    const std::vector<std::uint16_t> scale_background(Rows, 0x3555), bf_background(Rows * D, 0x3f80);
    k8.copy_from_host(code_background.data(), k8.bytes()); v8.copy_from_host(code_background.data(), v8.bytes());
    ks8.copy_from_host(scale_background.data(), ks8.bytes()); vs8.copy_from_host(scale_background.data(), vs8.bytes());
    kb.copy_from_host(bf_background.data(), kb.bytes()); vb.copy_from_host(bf_background.data(), vb.bytes());
    Tensor x(dx.p, DType::BF16, {width, tokens}), qw(dqw.p, DType::BF16, {D}), kw(dkw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {static_cast<int>(table.size())});
    Tensor q8t(q8.data(), DType::BF16, {D, QHeads, tokens}), qbt(qb.data(), DType::BF16, {D, QHeads, tokens});
    Tensor k8t(k8.data(), DType::U8, {D, 64, Heads, Pages}), v8t(v8.data(), DType::U8, {D, 64, Heads, Pages});
    Tensor ks8t(ks8.data(), DType::FP16, {1, 64, Heads, Pages}), vs8t(vs8.data(), DType::FP16, {1, 64, Heads, Pages});
    Tensor kbt(kb.data(), DType::BF16, {D, 64, Heads, Pages}), vbt(vb.data(), DType::BF16, {D, 64, Heads, Pages});
    const ops::Gemma4CacheView ring{ops::Gemma4KvFormat::Fp8E4m3Row256, &k8t, &v8t, &ks8t, &vs8t, &t};
    const ops::Gemma4CacheView reference{ops::Gemma4KvFormat::BFloat16, &kbt, &vbt, nullptr, nullptr, &t};
    ops::gemma4_qkv_prep(x, qw, kw, p, D, kTheta, 1e-6f, false, q8t, ring, nullptr);
    ops::gemma4_qkv_prep(x, qw, kw, p, D, kTheta, 1e-6f, false, qbt, reference, nullptr);
    cuda_synchronize();
    int failures = verify_exact("FP8 ring writer Q", from_device<std::uint16_t>(q8.data(), q8.bytes() / 2),
                                from_device<std::uint16_t>(qb.data(), qb.bytes() / 2));
    const auto kbits = from_device<std::uint16_t>(kb.data(), static_cast<std::size_t>(Rows) * D);
    const auto vbits = from_device<std::uint16_t>(vb.data(), static_cast<std::size_t>(Rows) * D);
    auto ek = code_background, ev = code_background;
    auto eks = scale_background, evs = scale_background;
    int poisoned = 0;
    for (const int pos : positions) {
        if (pos < 0 || pos / 64 >= static_cast<int>(table.size())) { continue; }
        const int physical = table[pos / 64];
        if (physical < 0 || physical >= Pages) { continue; }
        for (int h = 0; h < Heads; ++h) {
            const auto row = row_of(physical, h, pos % 64);
            quantize_row(kbits.data() + row * D, ek.data() + row * D, eks[row]);
            quantize_row(vbits.data() + row * D, ev.data() + row * D, evs[row]);
            poisoned += (ek[row * D] == 0x7f && eks[row] == half_bits(1.0f)) + (ev[row * D] == 0x7f && evs[row] == half_bits(1.0f));
        }
    }
    if (poisoned < 2) { std::cerr << "FP8 ring writer fixture lost its nonfinite rows\n"; ++failures; }
    failures += verify_exact("FP8 ring K codes after norm/rope", from_device<std::uint8_t>(k8.data(), ek.size()), ek);
    failures += verify_exact("FP8 ring V codes after norm", from_device<std::uint8_t>(v8.data(), ev.size()), ev);
    failures += verify_exact("FP8 ring K row scales", from_device<std::uint16_t>(ks8.data(), eks.size()), eks);
    failures += verify_exact("FP8 ring V row scales", from_device<std::uint16_t>(vs8.data(), evs.size()), evs);
    for (auto* buffer : {&q8, &k8, &v8, &ks8, &vs8}) { failures += buffer->verify_guards("FP8 ring writer"); }
    return failures;
}

struct RingFixture {
    int ring_pages = 0, logical_pages = 0;
    std::vector<int> positions, table;
    std::vector<std::uint8_t> k_codes, v_codes;
    std::vector<std::uint16_t> k_scales, v_scales, k_values, v_values; // values: host expansion
};

// A ring of pages_for(T + W - 1) pages as the Program sizes it, logical page l on physical page
// l % ring (one page unmapped and one out of range when `holes`). Rows of positions the ring holds
// get codes and scales (V rows with the FP32-vs-FP16 rounding witness); every other row is a
// never-written row poisoned with NaN codes and scales, which no visible key may read.
RingFixture make_ring(int first, int tokens, bool holes, std::uint32_t seed) {
    RingFixture f;
    f.ring_pages = (tokens + Window - 1 + 63) / 64;
    const int last = first + tokens - 1;
    f.logical_pages = last / 64 + 1;
    f.positions.resize(tokens);
    for (int i = 0; i < tokens; ++i) { f.positions[i] = first + i; }
    f.table.resize(f.logical_pages);
    for (int l = 0; l < f.logical_pages; ++l) { f.table[l] = l % f.ring_pages; }
    if (holes) {
        // A page inside every query's window, and the page holding the first query's window start.
        f.table[last / 64 - 2 >= 0 ? last / 64 - 2 : 0] = -1;
        const int start_page = std::max(0, first - Window + 1) / 64;
        if (start_page != last / 64 - 2 && start_page < last / 64) { f.table[start_page] = f.ring_pages; }
    }
    const std::size_t rows = static_cast<std::size_t>(f.ring_pages) * 64 * Heads;
    f.k_codes.assign(rows * D, 0x7f); f.v_codes.assign(rows * D, 0x7f);
    f.k_scales.assign(rows, 0x7e00); f.v_scales.assign(rows, 0x7e00);
    static const auto boundary = expansion_boundary();
    std::uint32_t state = seed;
    const auto next = [&] { state = state * 1664525u + 1013904223u; return state >> 8; };
    for (int pos = std::max(0, last - f.ring_pages * 64 + 1); pos <= last; ++pos) {
        const int physical = f.table[pos / 64];
        if (physical < 0 || physical >= f.ring_pages) { continue; }
        for (int h = 0; h < Heads; ++h) {
            const auto row = static_cast<std::size_t>(row_of(physical, h, pos % 64));
            f.k_scales[row] = half_bits(0.0015f + float(next() % 1000) * 0.0000025f);
            f.v_scales[row] = row % 3 == 0 ? boundary.second : half_bits(0.002f + float(next() % 1000) * 0.000003f);
            for (int d = 0; d < D; ++d) {
                const auto code = static_cast<std::uint8_t>(next() % 0x7f) | static_cast<std::uint8_t>(next() & 128);
                f.k_codes[row * D + d] = code == 0x7f || code == 0xff ? 0x3c : code;
                const auto vcode = static_cast<std::uint8_t>(next() % 0x7f) | static_cast<std::uint8_t>(next() & 128);
                f.v_codes[row * D + d] = row % 3 == 0 && d % 7 == 0 ? boundary.first :
                                         vcode == 0x7f || vcode == 0xff ? 0x41 : vcode;
            }
        }
    }
    f.k_values.resize(rows * D); f.v_values.resize(rows * D);
    for (std::size_t i = 0; i < rows * D; ++i) {
        f.k_values[i] = represented(f.k_codes[i], f.k_scales[i / D]);
        f.v_values[i] = represented(f.v_codes[i], f.v_scales[i / D]);
    }
    return f;
}

// FP64 softmax attention over the represented BF16 values of the visible mapped keys, for the
// query tokens in `sampled` (all heads).
std::vector<double> ring_reference(const RingFixture& f, const std::vector<std::uint16_t>& query,
                                   const std::vector<int>& sampled) {
    std::vector<double> out;
    out.reserve(sampled.size() * QHeads * D);
    std::vector<double> logits, value(D);
    for (const int token : sampled) {
        const int position = f.positions[token];
        const int low = std::max(0, position - Window + 1);
        for (int head = 0; head < QHeads; ++head) {
            const int kvh = head / (QHeads / Heads);
            const std::uint16_t* q = query.data() + (static_cast<std::size_t>(token) * QHeads + head) * D;
            logits.assign(position - low + 1, -std::numeric_limits<double>::infinity());
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = low; key <= position; ++key) {
                const int physical = f.table[key / 64];
                if (physical < 0 || physical >= f.ring_pages) { continue; }
                const auto row = static_cast<std::size_t>(row_of(physical, kvh, key % 64));
                double score = 0.0;
                for (int d = 0; d < D; ++d) { score += double(bf16_to_f32(q[d])) * bf16_to_f32(f.k_values[row * D + d]); }
                logits[key - low] = score;
                maximum = std::max(maximum, score);
            }
            std::fill(value.begin(), value.end(), 0.0);
            if (std::isfinite(maximum)) {
                double sum = 0.0;
                for (auto& logit : logits) { logit = std::exp(logit - maximum); sum += logit; }
                for (int key = low; key <= position; ++key) {
                    if (logits[key - low] == 0.0) { continue; }
                    const auto row = static_cast<std::size_t>(row_of(f.table[key / 64], kvh, key % 64));
                    const double weight = logits[key - low] / sum;
                    for (int d = 0; d < D; ++d) { value[d] += weight * bf16_to_f32(f.v_values[row * D + d]); }
                }
            }
            out.insert(out.end(), value.begin(), value.end());
        }
    }
    return out;
}

int attention_test(int first, int tokens, bool holes) {
    const auto f = make_ring(first, tokens, holes, 9001u + static_cast<std::uint32_t>(first * 7 + tokens));
    std::vector<float> query(static_cast<std::size_t>(D) * QHeads * tokens);
    fill_uniform(query, 9101 + tokens, -0.25f, 0.25f);
    auto dq = to_device_bf16(query), dp = to_device_i32(f.positions), dt = to_device_i32(f.table);
    auto dkc = to_device(f.k_codes), dvc = to_device(f.v_codes), dks = to_device(f.k_scales), dvs = to_device(f.v_scales);
    auto dkb = to_device(f.k_values), dvb = to_device(f.v_values);
    const std::size_t elements = query.size();
    GuardedDeviceBuffer out(elements * 2), reference(elements * 2);
    Tensor qt(dq.p, DType::BF16, {D, QHeads, tokens}), pt(dp.p, DType::I32, {tokens});
    Tensor tt(dt.p, DType::I32, {f.logical_pages});
    Tensor kc(dkc.p, DType::U8, {D, 64, Heads, f.ring_pages}), vc(dvc.p, DType::U8, {D, 64, Heads, f.ring_pages});
    Tensor ks(dks.p, DType::FP16, {1, 64, Heads, f.ring_pages}), vs(dvs.p, DType::FP16, {1, 64, Heads, f.ring_pages});
    Tensor kb(dkb.p, DType::BF16, {D, 64, Heads, f.ring_pages}), vb(dvb.p, DType::BF16, {D, 64, Heads, f.ring_pages});
    Tensor ot(out.data(), DType::BF16, {D, QHeads, tokens}), rt(reference.data(), DType::BF16, {D, QHeads, tokens});
    const ops::Gemma4CacheView ring{ops::Gemma4KvFormat::Fp8E4m3Row256, &kc, &vc, &ks, &vs, &tt};
    const ops::Gemma4CacheView expanded{ops::Gemma4KvFormat::BFloat16, &kb, &vb, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, Heads, tokens)));
    ops::gemma4_attention(qt, ring, pt, Window, 1.0f, ot, workspace, nullptr);
    ops::gemma4_attention(qt, expanded, pt, Window, 1.0f, rt, workspace, nullptr);
    cuda_synchronize();
    const std::string label = "FP8 ring attention first=" + std::to_string(first) + " T=" + std::to_string(tokens) +
                              (holes ? " holes" : "") + (tokens >= 64 ? " tensor-core" : " decode");
    const auto eager = from_device<std::uint16_t>(out.data(), elements);
    int failures = verify_exact((label + " equals BF16 attention over the expanded ring").c_str(), eager,
                                from_device<std::uint16_t>(reference.data(), elements));
    std::vector<int> sampled;
    // Every query of a decode width; every 13th of a prefill chunk (and its last), all heads.
    const int step = tokens >= 64 ? 13 : 1;
    for (int token = 0; token < tokens; token += step) { sampled.push_back(token); }
    if (sampled.back() != tokens - 1) { sampled.push_back(tokens - 1); }
    const auto expected = ring_reference(f, from_device<std::uint16_t>(dq, elements), sampled);
    const auto all = from_device_bf16(out.data(), elements);
    std::vector<double> got;
    got.reserve(expected.size());
    for (const int token : sampled) {
        const auto begin = all.begin() + static_cast<std::ptrdiff_t>(token) * QHeads * D;
        got.insert(got.end(), begin, begin + QHeads * D);
    }
    // The BF16 attention criterion of the FP8 global tests, unrelaxed.
    failures += verify_pointwise(label + " versus FP64", got, expected, PointwiseCriterion{2.0e-5, 8.0e-3});
    for (const auto bits : eager) {
        if (!std::isfinite(bf16_to_f32(bits))) { std::cerr << label << ": consumed a never-written row\n"; ++failures; break; }
    }
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    ops::gemma4_attention(qt, ring, pt, Window, 1.0f, ot, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    cuda_check(cudaMemsetAsync(out.data(), 0, elements * 2, stream), "clear output");
    cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
    cuda_synchronize(stream);
    failures += verify_exact((label + " eager versus graph replay").c_str(),
                             from_device<std::uint16_t>(out.data(), elements), eager);
    cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return failures + out.verify_guards("FP8 ring attention") + reference.verify_guards("expanded ring attention");
}

double relative_rms(const std::vector<double>& got, const std::vector<double>& reference) {
    double error = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        error += (got[i] - reference[i]) * (got[i] - reference[i]);
        norm += reference[i] * reference[i];
    }
    return std::sqrt(error / norm);
}

// One writer input (a full window plus `queries`, from `first`) through the BF16 and the FP8 ring,
// then attention of the last `queries` positions on each: the FP8 ring's error against the BF16 ring
// is reported (measurement, not an adoption gate). Key-norm weights of `key_weight` scale K: unit weights
// make rmsnorm'd Q.K scores of standard deviation ~16 over 256 dimensions, a near-argmax softmax that
// amplifies any K perturbation; 0.05-0.2 gives moderate scores.
void ring_comparison_test(int first, int queries, float key_weight_low, float key_weight_high) {
    const int tokens = Window + queries;
    const int ring_pages = (queries + Window - 1 + 63) / 64;
    const int logical_pages = (first + tokens - 1) / 64 + 1;
    std::vector<int> positions(tokens), table(logical_pages);
    for (int i = 0; i < tokens; ++i) { positions[i] = first + i; }
    for (int l = 0; l < logical_pages; ++l) { table[l] = l % ring_pages; }
    const int width = (QHeads + 2 * Heads) * D;
    std::vector<float> input(static_cast<std::size_t>(width) * tokens), q_weights(D), k_weights(D);
    fill_uniform(input, 7201 + first, -3.0f, 3.0f);
    fill_uniform(q_weights, 7202, 0.4f, 1.7f);
    fill_uniform(k_weights, 7203, key_weight_low, key_weight_high);
    auto dx = to_device_bf16(input), dqw = to_device_bf16(q_weights), dkw = to_device_bf16(k_weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    const std::size_t rows = static_cast<std::size_t>(ring_pages) * 64 * Heads;
    DeviceBuffer q(static_cast<std::size_t>(QHeads) * D * tokens * 2);
    DeviceBuffer k8(rows * D), v8(rows * D), ks8(rows * 2), vs8(rows * 2), kb(rows * D * 2), vb(rows * D * 2);
    for (auto* buffer : {&k8, &v8, &ks8, &vs8, &kb, &vb}) { buffer->fill(0); }
    Tensor x(dx.p, DType::BF16, {width, tokens}), qw(dqw.p, DType::BF16, {D}), kw(dkw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {logical_pages});
    Tensor qo(q.p, DType::BF16, {D, QHeads, tokens});
    Tensor k8t(k8.p, DType::U8, {D, 64, Heads, ring_pages}), v8t(v8.p, DType::U8, {D, 64, Heads, ring_pages});
    Tensor ks8t(ks8.p, DType::FP16, {1, 64, Heads, ring_pages}), vs8t(vs8.p, DType::FP16, {1, 64, Heads, ring_pages});
    Tensor kbt(kb.p, DType::BF16, {D, 64, Heads, ring_pages}), vbt(vb.p, DType::BF16, {D, 64, Heads, ring_pages});
    const ops::Gemma4CacheView ring{ops::Gemma4KvFormat::Fp8E4m3Row256, &k8t, &v8t, &ks8t, &vs8t, &t};
    const ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &kbt, &vbt, nullptr, nullptr, &t};
    // The window, then the queries: two calls, as consecutive prefill chunks write a ring.
    std::vector<double> results[2];
    const std::size_t elements = static_cast<std::size_t>(queries) * QHeads * D;
    DeviceBuffer out(elements * 2);
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, Heads, queries)));
    for (int arm = 0; arm < 2; ++arm) {
        const auto& cache = arm == 0 ? bf16 : ring;
        for (const auto& [begin, count] : {std::pair{0, Window}, std::pair{Window, queries}}) {
            Tensor xs(static_cast<std::uint16_t*>(dx.p) + static_cast<std::size_t>(begin) * width, DType::BF16, {width, count});
            Tensor ps(static_cast<std::int32_t*>(dp.p) + begin, DType::I32, {count});
            Tensor qs(static_cast<std::uint16_t*>(q.p) + static_cast<std::size_t>(begin) * QHeads * D, DType::BF16,
                      {D, QHeads, count});
            ops::gemma4_qkv_prep(xs, qw, kw, ps, D, kTheta, 1e-6f, false, qs, cache, nullptr);
        }
        Tensor qt(static_cast<std::uint16_t*>(q.p) + static_cast<std::size_t>(Window) * QHeads * D, DType::BF16,
                  {D, QHeads, queries});
        Tensor pt(static_cast<std::int32_t*>(dp.p) + Window, DType::I32, {queries});
        Tensor ot(out.p, DType::BF16, {D, QHeads, queries});
        ops::gemma4_attention(qt, cache, pt, Window, 1.0f, ot, workspace, nullptr);
        cuda_synchronize();
        results[arm] = from_device_bf16(out.p, elements);
    }
    const double error = relative_rms(results[1], results[0]);
    std::cout << "GEMMA4_FP8_RING_VS_BF16 first=" << first << " queries=" << queries << " window=" << Window
              << " key_norm_weights=" << key_weight_low << ".." << key_weight_high
              << " fp8_ring_rel_rms_vs_bf16_ring=" << error << '\n';
    // Measurement only. The initial experiment's 0.10 cutoff failed (up to 0.139); replacing it with
    // 0.25 did not establish acceptance. Exact codec and represented-value FP64 checks above carry
    // correctness; model-level quality and adoption are reported separately.
}

// Measurement only (`--time`): event-timed launches of the tensor-core kernel on a 2,048-token chunk
// with a full window, the FP8 instance and the BF16 instance over the same ring (median of 50 after
// 10 warm-up launches, microseconds), as the prompt of a long request runs it on every sliding layer.
void timing_test(int first, int tokens) {
    const auto f = make_ring(first, tokens, false, 9001u + static_cast<std::uint32_t>(first * 7 + tokens));
    std::vector<float> query(static_cast<std::size_t>(D) * QHeads * tokens);
    fill_uniform(query, 9101 + tokens, -0.25f, 0.25f);
    auto dq = to_device_bf16(query), dp = to_device_i32(f.positions), dt = to_device_i32(f.table);
    auto dkc = to_device(f.k_codes), dvc = to_device(f.v_codes), dks = to_device(f.k_scales), dvs = to_device(f.v_scales);
    auto dkb = to_device(f.k_values), dvb = to_device(f.v_values);
    const std::size_t elements = query.size();
    DeviceBuffer out(elements * 2);
    Tensor qt(dq.p, DType::BF16, {D, QHeads, tokens}), pt(dp.p, DType::I32, {tokens});
    Tensor tt(dt.p, DType::I32, {f.logical_pages});
    Tensor kc(dkc.p, DType::U8, {D, 64, Heads, f.ring_pages}), vc(dvc.p, DType::U8, {D, 64, Heads, f.ring_pages});
    Tensor ks(dks.p, DType::FP16, {1, 64, Heads, f.ring_pages}), vs(dvs.p, DType::FP16, {1, 64, Heads, f.ring_pages});
    Tensor kb(dkb.p, DType::BF16, {D, 64, Heads, f.ring_pages}), vb(dvb.p, DType::BF16, {D, 64, Heads, f.ring_pages});
    Tensor ot(out.p, DType::BF16, {D, QHeads, tokens});
    const ops::Gemma4CacheView ring{ops::Gemma4KvFormat::Fp8E4m3Row256, &kc, &vc, &ks, &vs, &tt};
    const ops::Gemma4CacheView expanded{ops::Gemma4KvFormat::BFloat16, &kb, &vb, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, Heads, tokens)));
    cudaEvent_t begin, end;
    cuda_check(cudaEventCreate(&begin), "cudaEventCreate");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate");
    const auto median_us = [&](const ops::Gemma4CacheView& cache) {
        std::vector<float> micros;
        for (int launch = 0; launch < 60; ++launch) {
            cuda_check(cudaEventRecord(begin, nullptr), "cudaEventRecord");
            ops::gemma4_attention(qt, cache, pt, Window, 1.0f, ot, workspace, nullptr);
            cuda_check(cudaEventRecord(end, nullptr), "cudaEventRecord");
            cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize");
            float milliseconds = 0.0f;
            cuda_check(cudaEventElapsedTime(&milliseconds, begin, end), "cudaEventElapsedTime");
            if (launch >= 10) { micros.push_back(milliseconds * 1000.0f); }
        }
        std::sort(micros.begin(), micros.end());
        return micros[micros.size() / 2];
    };
    const float bf16_us = median_us(expanded);
    const float fp8_us = median_us(ring);
    std::cout << "GEMMA4_FP8_RING_TIME first=" << first << " tokens=" << tokens << " window=" << Window
              << " bf16_instance_us=" << bf16_us << " fp8_instance_us=" << fp8_us << " ratio=" << fp8_us / bf16_us << '\n';
    cuda_check(cudaEventDestroy(begin), "cudaEventDestroy");
    cuda_check(cudaEventDestroy(end), "cudaEventDestroy");
}
} // namespace

INFERNIX_OP_TEST(gemma4_fp8_ring) {
    bool timing = false;
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--time") {
            timing = true;
            continue;
        }
        if (key != "--split" || i + 1 >= argc || (std::string_view(argv[i + 1]) != "0" && std::string_view(argv[i + 1]) != "1")) {
            std::cerr << "usage: gemma4_fp8_ring [--split 0|1] [--time]\n";
            return 2;
        }
        const std::string setting = std::string_view(argv[++i]) == "0" ? "1" : "0";
#ifdef _WIN32
        if (_putenv_s("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str()) != 0) { return 2; }
#else
        if (setenv("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str(), 1) != 0) { return 2; }
#endif
    }
    std::cout << "temporary switches | " << temporary_switches::describe(temporary_switches::Scope::Gemma4) << '\n';
    int failures = validation_test();
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return failures ? 1 : 77; }
    if (timing) {
        timing_test(3 * 3072 + 101, 2048);
        return failures ? 1 : 0;
    }
    failures += qkv_test();
    // Decode and MTP verify widths, the widest decode width, then tensor-core chunks (partial and full
    // tiles, the chunk-128 and auto-chunk widths): before the window fills, mid-page window starts,
    // and a ring wrapped several times.
    for (const int first : {0, 70, 1000, 5 * 1152 + 37, 3 * 3072 + 101}) {
        for (const int tokens : {1, 2, 4, 8, 63, 64, 65, 128, 129, 2048}) {
            failures += attention_test(first, tokens, false);
        }
    }
    for (const int tokens : {1, 8, 64, 129, 2048}) { failures += attention_test(5 * 1152 + 37, tokens, true); }
    for (const int first : {0, 9000}) {
        for (const int queries : {1, 64, 128}) {
            ring_comparison_test(first, queries, 0.4f, 1.7f);
            ring_comparison_test(first, queries, 0.05f, 0.2f);
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_fp8_ring\n";
    return failures ? 1 : 0;
}
