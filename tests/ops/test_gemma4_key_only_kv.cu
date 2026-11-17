// Gemma 4 key-only global KV (ops::Gemma4ValueSource::Key) oracles: cache validation; the writer's K
// planes equal the K==V writer's while no V exists; the forward convention (unrope(K) / c recovers
// the BF16 V gemma4_qkv_prep stores, and a wrong rotation direction does not); attention against an
// FP64 oracle on the D512 decode and tensor-core kernels (BF16 and FP8 QK, split context, graph
// replay) at positions up to 262,143; and the attention difference of key-only against stored FP8
// V, both measured against BF16 K/V from the same writer.
#include "core/temporary_switches.h"
#include "infernix/ops/gemma4_attention.h"
#include "infernix/ops/gemma4_qkv_prep.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {
constexpr int D = 512;
constexpr int Heads = 2;
constexpr int QHeads = 4;
constexpr int Pages = 16;
constexpr int Rows = 64 * Heads * Pages;
constexpr int Rotary = ops::kGemma4KeyValueRotaryDim;
constexpr int LogicalPages = 4096; // positions up to 262,143
constexpr float kTheta = 31.62277603149414f;  // prope_theta_eff of the Gemma 4 31B artifacts
constexpr float kKeyNorm = 0.0615234375f;     // a recorded global key-norm constant
const float kValueScale = 1.0f / kKeyNorm;

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
// The FP8 QK query codec (finite rows only), as the tensor-core kernel quantizes queries.
void quantize(const std::uint16_t* x, std::uint8_t* codes, std::uint16_t* scales) {
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        for (int i = 0; i < 256; ++i) { maximum = std::fmax(maximum, std::fabs(bf16_to_f32(x[group * 256 + i]))); }
        scales[group] = half_bits(maximum == 0.0f ? 0.0f : std::fmin(65504.0f, std::fmax(0x1p-24f, maximum / 448.0f)));
        const float inverse = maximum == 0.0f ? 0.0f : 1.0f / half_value(scales[group]);
        for (int i = 0; i < 256; ++i) {
            codes[group * 256 + i] = maximum == 0.0f ? 0 : encode(bf16_to_f32(x[group * 256 + i]) * inverse);
        }
    }
}

__global__ void frequency_kernel(float theta, float* out) {
    if (threadIdx.x < Rotary / 2) {
        out[threadIdx.x] = ops::detail::gemma4_rope_frequency(static_cast<int>(threadIdx.x), Rotary, theta);
    }
}
// The pair frequencies the writer and the readers form on the device.
std::vector<float> device_frequencies() {
    GuardedDeviceBuffer out(Rotary / 2 * sizeof(float));
    frequency_kernel<<<1, 64>>>(kTheta, static_cast<float*>(out.data()));
    cuda_check_last_launch("frequency_kernel");
    cuda_synchronize();
    return from_device<float>(out.data(), Rotary / 2);
}

// The FP32 angle the writer used for pair i at `position` (one IEEE product), in FP64 sine/cosine.
void rotation(int position, float frequency, double& s, double& c) {
    const float angle = static_cast<float>(position) * frequency;
    s = std::sin(static_cast<double>(angle));
    c = std::cos(static_cast<double>(angle));
}

// V of one cached K row in FP64: unrope(code x scale) x value_scale; `forward` turns the wrong way
// (the discrimination control).
std::array<double, D> derived_value(const std::uint8_t* codes, const std::uint16_t* scales, int position,
                                    const std::vector<float>& frequency, bool forward = false) {
    std::array<double, D> k{};
    for (int d = 0; d < D; ++d) { k[d] = double(code_value(codes[d]) * half_value(scales[d / 256])); }
    std::array<double, D> v{};
    for (int d = 0; d < D; ++d) { v[d] = k[d]; }
    for (int i = 0; i < Rotary / 2; ++i) {
        double s, c;
        rotation(position, frequency[i], s, c);
        if (forward) { s = -s; }
        v[i] = k[i] * c + k[i + Rotary / 2] * s;
        v[i + Rotary / 2] = k[i + Rotary / 2] * c - k[i] * s;
    }
    for (auto& x : v) { x *= double(kValueScale); }
    return v;
}

// Each listed position's logical page maps to the next free physical page.
std::vector<int> sparse_table(const std::vector<int>& positions) {
    std::vector<int> table(LogicalPages, -1);
    int next = 0;
    for (int position : positions) {
        if (position < 0 || table[position / 64] >= 0) { continue; }
        table[position / 64] = next++;
    }
    if (next > Pages) { throw std::logic_error("sparse_table: too many pages"); }
    return table;
}

// Physical page -> logical page of a table (-1: unmapped).
std::vector<int> logical_of(const std::vector<int>& table) {
    std::vector<int> out(Pages, -1);
    for (std::size_t page = 0; page < table.size(); ++page) {
        if (table[page] >= 0 && table[page] < Pages) { out[table[page]] = static_cast<int>(page); }
    }
    return out;
}

Tensor pages_view(void* data, DType dtype, int rows) { return Tensor(data, dtype, {rows, 64, Heads, Pages}); }

ops::Gemma4CacheView key_only_view(Tensor* k, Tensor* ks, const Tensor* table) {
    ops::Gemma4CacheView view{ops::Gemma4KvFormat::Fp8E4m3Row256, k, nullptr, ks, nullptr, table};
    view.value_source = ops::Gemma4ValueSource::Key;
    view.value_scale = kValueScale;
    view.rope_theta = kTheta;
    view.rotary_dim = Rotary;
    return view;
}

int validation_test() {
    int storage = 0;
    Tensor k(&storage, DType::U8, {D, 64, Heads, Pages}), v = k;
    Tensor ks(&storage, DType::FP16, {2, 64, Heads, Pages}), vs = ks;
    Tensor table(&storage, DType::I32, {Pages});
    const auto valid = key_only_view(&k, &ks, &table);
    ops::detail::validate_gemma4_cache(valid, D, true, "test");
    auto fp8_qk = valid;
    fp8_qk.qk_mode = ops::Gemma4QkMode::Fp8;
    ops::detail::validate_gemma4_cache(fp8_qk, D, true, "test");
    int failures = 0;
    const auto rejects = [&](const ops::Gemma4CacheView& cache, int dim = D, bool global = true) {
        try { ops::detail::validate_gemma4_cache(cache, dim, global, "test"); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "invalid key-only cache accepted\n"; ++failures;
    };
    auto bad = valid; bad.v_pages = &v; rejects(bad);
    bad = valid; bad.v_scales = &vs; rejects(bad);
    bad = valid; bad.k_scales = nullptr; rejects(bad);
    bad = valid; bad.rotary_dim = 64; rejects(bad);
    bad = valid; bad.rope_theta = 0.0f; rejects(bad);
    bad = valid; bad.rope_theta = std::numeric_limits<float>::quiet_NaN(); rejects(bad);
    bad = valid; bad.value_scale = 0.0f; rejects(bad);
    bad = valid; bad.value_scale = std::numeric_limits<float>::infinity(); rejects(bad);
    bad = valid; bad.value_source = static_cast<ops::Gemma4ValueSource>(7); rejects(bad);
    rejects(valid, 256); rejects(valid, D, false);
    Tensor bf(&storage, DType::BF16, {D, 64, Heads, Pages});
    bad = valid; bad.format = ops::Gemma4KvFormat::BFloat16; bad.k_pages = &bf; bad.k_scales = nullptr; rejects(bad);
    // Stored values take no derivation parameters.
    ops::Gemma4CacheView stored{ops::Gemma4KvFormat::Fp8E4m3Row256, &k, &v, &ks, &vs, &table};
    ops::detail::validate_gemma4_cache(stored, D, true, "test");
    bad = stored; bad.value_scale = kValueScale; rejects(bad);
    bad = stored; bad.rope_theta = kTheta; rejects(bad);
    bad = stored; bad.rotary_dim = Rotary; rejects(bad);
    return failures;
}

// The key-only writer stores exactly the K==V writer's K planes and nothing else, and the K==V
// writer's BF16 planes satisfy V = unrope(K) / c under the readers' convention.
int qkv_test(const std::vector<float>& frequency) {
    const std::vector<int> positions{-1, 0, 1, 63, 64, 100, 1000, 4095, 65537, 131071, 200001, 262143};
    const auto table = sparse_table(positions);
    const int tokens = static_cast<int>(positions.size());
    std::vector<float> input((QHeads + Heads) * D * tokens), q_weights(D);
    fill_uniform(input, 4101, -3.0f, 3.0f);
    fill_uniform(q_weights, 4102, 0.4f, 1.7f);
    const std::vector<float> k_weights(D, kKeyNorm);
    auto dx = to_device_bf16(input), dqw = to_device_bf16(q_weights), dkw = to_device_bf16(k_weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    GuardedDeviceBuffer q(QHeads * D * tokens * 2), q2(q.bytes()), qr(q.bytes());
    GuardedDeviceBuffer k(Rows * D), v(Rows * D), ks(Rows * 4), vs(Rows * 4);
    GuardedDeviceBuffer k2(Rows * D), ks2(Rows * 4);
    GuardedDeviceBuffer kr(Rows * D * 2), vr(Rows * D * 2);
    for (auto* buffer : {&k, &v, &ks, &vs, &k2, &ks2, &kr, &vr}) { buffer->fill(0x3c); }
    Tensor x(dx.p, DType::BF16, {(QHeads + Heads) * D, tokens});
    Tensor qw(dqw.p, DType::BF16, {D}), kw(dkw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {LogicalPages});
    Tensor qo(q.data(), DType::BF16, {D, QHeads, tokens}), qo2(q2.data(), DType::BF16, {D, QHeads, tokens});
    Tensor qro(qr.data(), DType::BF16, {D, QHeads, tokens});
    Tensor kt = pages_view(k.data(), DType::U8, D), vt = pages_view(v.data(), DType::U8, D);
    Tensor kst = pages_view(ks.data(), DType::FP16, 2), vst = pages_view(vs.data(), DType::FP16, 2);
    Tensor k2t = pages_view(k2.data(), DType::U8, D), ks2t = pages_view(ks2.data(), DType::FP16, 2);
    Tensor krt = pages_view(kr.data(), DType::BF16, D), vrt = pages_view(vr.data(), DType::BF16, D);
    const ops::Gemma4CacheView stored{ops::Gemma4KvFormat::Fp8E4m3Row256, &kt, &vt, &kst, &vst, &t};
    const auto derived = key_only_view(&k2t, &ks2t, &t);
    const ops::Gemma4CacheView reference{ops::Gemma4KvFormat::BFloat16, &krt, &vrt, nullptr, nullptr, &t};
    ops::gemma4_qkv_prep(x, qw, kw, p, Rotary, kTheta, 1e-6f, true, qo, stored, nullptr);
    ops::gemma4_qkv_prep(x, qw, kw, p, Rotary, kTheta, 1e-6f, true, qo2, derived, nullptr);
    ops::gemma4_qkv_prep(x, qw, kw, p, Rotary, kTheta, 1e-6f, true, qro, reference, nullptr);
    cuda_synchronize();
    int failures = verify_exact("key-only writer Q", from_device<std::uint16_t>(q2.data(), q2.bytes() / 2),
                                from_device<std::uint16_t>(q.data(), q.bytes() / 2));
    failures += verify_exact("key-only writer K codes equal the K==V writer's",
                             from_device<std::uint8_t>(k2.data(), k2.bytes()), from_device<std::uint8_t>(k.data(), k.bytes()));
    failures += verify_exact("key-only writer K scales equal the K==V writer's",
                             from_device<std::uint16_t>(ks2.data(), ks2.bytes() / 2),
                             from_device<std::uint16_t>(ks.data(), ks.bytes() / 2));
    for (auto* buffer : {&q, &q2, &qr, &k, &v, &ks, &vs, &k2, &ks2, &kr, &vr}) {
        failures += buffer->verify_guards("key-only writer");
    }
    // The convention: unrope(K_bf16) / c against the stored BF16 V of every written row, and the
    // same with the rotation reversed, which must fail at nonzero positions.
    const auto kb = from_device<std::uint16_t>(kr.data(), Rows * D);
    const auto vb = from_device<std::uint16_t>(vr.data(), Rows * D);
    double worst = 0.0, worst_wrong = 0.0;
    for (int position : positions) {
        if (position < 0) { continue; }
        for (int h = 0; h < Heads; ++h) {
            const auto row = ops::detail::gemma4_kv_row(table[position / 64], Heads, h, position % 64);
            double amax = 0.0, error = 0.0, wrong = 0.0;
            for (int i = 0; i < D; ++i) { amax = std::max(amax, std::fabs(double(bf16_to_f32(vb[row * D + i])))); }
            std::array<double, D> k_row{};
            for (int i = 0; i < D; ++i) { k_row[i] = bf16_to_f32(kb[row * D + i]); }
            for (int i = 0; i < D; ++i) {
                double back = k_row[i], turned = k_row[i];
                if (i < Rotary) {
                    const int pair = i % (Rotary / 2);
                    double s, c;
                    rotation(position, frequency[pair], s, c);
                    const double a = k_row[pair], b = k_row[pair + Rotary / 2];
                    back = i < Rotary / 2 ? a * c + b * s : b * c - a * s;
                    turned = i < Rotary / 2 ? a * c - b * s : b * c + a * s;
                }
                const double value = bf16_to_f32(vb[row * D + i]);
                error = std::max(error, std::fabs(back * kValueScale - value));
                wrong = std::max(wrong, std::fabs(turned * kValueScale - value));
            }
            worst = std::max(worst, error / amax);
            if (position > 0) { worst_wrong = std::max(worst_wrong, wrong / amax); }
        }
    }
    std::cout << "GEMMA4_KEY_ONLY_CONVENTION max_error_over_row_absmax=" << worst
              << " reversed_rotation=" << worst_wrong << '\n';
    // Three BF16 roundings (K before and after RoPE, V) bound the error near 3 x 2^-9 of the row.
    if (!(worst <= 0x1p-6)) { std::cerr << "unrope(K) / c does not recover V: " << worst << '\n'; ++failures; }
    if (!(worst_wrong > 0.05)) { std::cerr << "the convention check cannot tell rotation directions apart\n"; ++failures; }
    // The writer refuses a key-only cache that is not a K==V layer or uses another RoPE.
    const auto refuses = [&](auto&& work, const char* what) {
        try { work(); } catch (const std::invalid_argument&) { return; }
        std::cerr << "key-only writer accepted " << what << '\n'; ++failures;
    };
    Tensor wide(dx.p, DType::BF16, {(QHeads + 2 * Heads) * D, tokens});
    refuses([&] { ops::gemma4_qkv_prep(wide, qw, kw, p, Rotary, kTheta, 1e-6f, false, qo2, derived, nullptr); },
            "a layer with its own V rows");
    refuses([&] { ops::gemma4_qkv_prep(x, qw, kw, p, Rotary, 10000.0f, 1e-6f, true, qo2, derived, nullptr); },
            "another RoPE base");
    return failures;
}

// FP64 attention over a key-only cache: BF16 QK on represented K (or FP8 QK codes), V from
// derived_value unrounded. The kernels round each rebuilt V element once to BF16 from FP32
// arithmetic, so outputs may differ by up to half a BF16 ulp of the largest visible V
// (`value_max`), which the callers' absolute tolerance allows.
std::vector<double> attention_reference(const std::vector<std::uint16_t>& query, const std::vector<std::uint8_t>& codes,
                                        const std::vector<std::uint16_t>& scales, const std::vector<int>& positions,
                                        const std::vector<int>& table, const std::vector<float>& frequency,
                                        bool fp8_qk, double& value_max) {
    const auto logical = logical_of(table);
    std::vector<double> values(static_cast<std::size_t>(Rows) * D, 0.0);
    value_max = 0.0;
    for (int physical = 0; physical < Pages; ++physical) {
        if (logical[physical] < 0) { continue; }
        for (int h = 0; h < Heads; ++h) {
            for (int slot = 0; slot < 64; ++slot) {
                const auto row = ops::detail::gemma4_kv_row(physical, Heads, h, slot);
                const int position = logical[physical] * 64 + slot;
                const auto v = derived_value(codes.data() + row * D, scales.data() + row * 2, position, frequency);
                for (int d = 0; d < D; ++d) {
                    values[row * D + d] = v[d];
                    if (position <= positions.back() && std::isfinite(v[d])) { value_max = std::max(value_max, std::fabs(v[d])); }
                }
            }
        }
    }
    std::vector<std::uint8_t> qc(query.size());
    std::vector<std::uint16_t> qs(query.size() / 256);
    if (fp8_qk) {
        for (std::size_t row = 0; row < query.size() / D; ++row) {
            quantize(query.data() + row * D, qc.data() + row * D, qs.data() + row * 2);
        }
    }
    std::vector<double> expected(query.size(), 0.0);
    for (std::size_t token = 0; token < positions.size(); ++token) {
        for (int head = 0; head < QHeads; ++head) {
            const auto qr = token * QHeads + head;
            std::vector<std::pair<std::int64_t, double>> keys;
            double maximum = -std::numeric_limits<double>::infinity();
            for (int page = 0; page <= positions[token] / 64; ++page) {
                const int physical = table[page];
                if (physical < 0 || physical >= Pages) { continue; }
                for (int slot = 0; slot < 64 && page * 64 + slot <= positions[token]; ++slot) {
                    const auto kr = ops::detail::gemma4_kv_row(physical, Heads, head / (QHeads / Heads), slot);
                    double score = 0.0;
                    for (int group = 0; group < 2; ++group) {
                        double partial = 0.0;
                        for (int d = group * 256; d < (group + 1) * 256; ++d) {
                            if (fp8_qk) {
                                partial += double(code_value(qc[qr * D + d])) * code_value(codes[kr * D + d]);
                            } else {
                                const float represented = bf16_to_f32(f32_to_bf16(
                                    code_value(codes[kr * D + d]) * half_value(scales[kr * 2 + group])));
                                partial += double(bf16_to_f32(query[qr * D + d])) * represented;
                            }
                        }
                        score += fp8_qk ? partial * double(half_value(qs[qr * 2 + group])) *
                                                       half_value(scales[kr * 2 + group]) : partial;
                    }
                    keys.emplace_back(kr, score);
                    maximum = std::max(maximum, score);
                }
            }
            if (keys.empty()) { continue; }
            double sum = 0.0;
            for (auto& [row, score] : keys) { score = std::exp(score - maximum); sum += score; }
            for (const auto& [row, weight] : keys) {
                for (int d = 0; d < D; ++d) { expected[qr * D + d] += weight / sum * values[row * D + d]; }
            }
        }
    }
    return expected;
}

int attention_test(const std::vector<float>& frequency, int tokens, bool fp8_qk, bool far) {
    // Queries end at `last`; the keys are page 0 and the pages holding the 130 positions before the
    // first query and every query position, so long-angle reconstruction is exercised.
    const int last = far ? 262143 : 64 + tokens;
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = last - tokens + 1 + i; }
    std::vector<int> mapped{0};
    for (int position = std::max(0, positions.front() - 130); position <= last; position += 64) { mapped.push_back(position); }
    mapped.push_back(last);
    const auto table = sparse_table(mapped);
    std::vector<float> query(D * QHeads * tokens);
    fill_uniform(query, 5301 + tokens, -0.05f, 0.05f);
    if (fp8_qk) {
        for (std::size_t i = 0; i < query.size(); ++i) { query[i] *= i % D < 256 ? 0.37f : 7.1f; }
    }
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(Rows) * D);
    std::vector<std::uint16_t> scales(static_cast<std::size_t>(Rows) * 2);
    const auto logical = logical_of(table);
    for (int row = 0; row < Rows; ++row) {
        const int physical = row / (64 * Heads);
        const int position = logical[physical] < 0 ? -1 : logical[physical] * 64 + row % 64;
        const bool unwritten = position > last;
        scales[row * 2] = half_bits(0.00314159f + (row % 7) * 0.001f);
        scales[row * 2 + 1] = half_bits(0.0253f + (row % 5) * 0.01f);
        if (unwritten) { scales[row * 2] = scales[row * 2 + 1] = 0x7e00; }
        for (int d = 0; d < D; ++d) {
            codes[row * D + d] = unwritten ? std::uint8_t{0x7f} :
                static_cast<std::uint8_t>(((d * 17 + row * 13) % 120) | ((d + row) & 1 ? 128 : 0));
        }
    }
    auto dq = to_device_bf16(query), dp = to_device_i32(positions), dt = to_device_i32(table);
    auto dc = to_device(codes), ds = to_device(scales);
    GuardedDeviceBuffer out(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, QHeads, tokens}), pt(dp.p, DType::I32, {tokens});
    Tensor tt(dt.p, DType::I32, {LogicalPages});
    Tensor ct = pages_view(dc.p, DType::U8, D), st = pages_view(ds.p, DType::FP16, 2);
    Tensor ot(out.data(), DType::BF16, {D, QHeads, tokens});
    auto cache = key_only_view(&ct, &st, &tt);
    if (fp8_qk) { cache.qk_mode = ops::Gemma4QkMode::Fp8; }
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, Heads, tokens)));
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, nullptr);
    cuda_synchronize();
    const auto eager = from_device<std::uint16_t>(out.data(), query.size());
    const bool tensor_fp8_qk = fp8_qk && tokens >= 64;
    double value_max = 0.0;
    const auto expected = attention_reference(from_device<std::uint16_t>(dq, query.size()), codes, scales, positions,
                                              table, frequency, tensor_fp8_qk, value_max);
    const std::string label = "key-only attention T=" + std::to_string(tokens) + (fp8_qk ? " fp8-qk" : " bf16-qk") +
                              (far ? " far" : " near") + " versus FP64";
    // BF16 QK: the rebuilt V's BF16 rounding (half an ulp is at most 2^-8 of the largest V) beside the
    // stored-V oracle's relative tolerance. FP8 QK also moves the softmax weights; its bound is
    // 2^-6 of the largest V. A wrong pair, direction or angle errs by O(1) of V (the convention
    // check's reversed rotation: 2.3 of the row maximum).
    int failures = verify_pointwise(label, from_device_bf16(out.data(), query.size()), expected,
        tensor_fp8_qk ? PointwiseCriterion{value_max * 0x1p-6, 0.006} : PointwiseCriterion{value_max * 0x1p-8, 8.0e-3});
    for (auto bits : eager) {
        if (!std::isfinite(bf16_to_f32(bits))) { std::cerr << label << ": consumed a poisoned row\n"; ++failures; break; }
    }
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "cudaStreamBeginCapture");
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, stream);
    cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
    cuda_synchronize(stream);
    failures += verify_exact((label + " eager versus graph replay").c_str(),
                             from_device<std::uint16_t>(out.data(), query.size()), eager);
    cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return failures + out.verify_guards("key-only attention");
}

double relative_rms(const std::vector<double>& got, const std::vector<double>& reference) {
    double error = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        error += (got[i] - reference[i]) * (got[i] - reference[i]);
        norm += reference[i] * reference[i];
    }
    return std::sqrt(error / norm);
}

// One writer input through BF16 K==V, FP8 K==V and FP8 key-only caches, then attention of the last
// `queries` positions (64: tensor-core, 1: decode) over all of them: the key-only and stored FP8
// errors against BF16 K/V are reported side by side (measurement, not an adoption gate).
int stored_comparison_test(int first, int queries, bool fp8_qk) {
    constexpr int tokens = 512;
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = first + i; }
    const auto table = sparse_table(positions);
    std::vector<float> input((QHeads + Heads) * D * tokens), q_weights(D);
    fill_uniform(input, 6101 + first, -3.0f, 3.0f);
    fill_uniform(q_weights, 6102, 0.4f, 1.7f);
    const std::vector<float> k_weights(D, kKeyNorm);
    auto dx = to_device_bf16(input), dqw = to_device_bf16(q_weights), dkw = to_device_bf16(k_weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    DeviceBuffer q(static_cast<std::size_t>(QHeads) * D * tokens * 2);
    DeviceBuffer k8(Rows * D), v8(Rows * D), ks8(Rows * 4), vs8(Rows * 4), kd(Rows * D), ksd(Rows * 4);
    DeviceBuffer kb(Rows * D * 2), vb(Rows * D * 2);
    for (auto* buffer : {&k8, &v8, &ks8, &vs8, &kd, &ksd, &kb, &vb}) { buffer->fill(0); }
    Tensor x(dx.p, DType::BF16, {(QHeads + Heads) * D, tokens});
    Tensor qw(dqw.p, DType::BF16, {D}), kw(dkw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {LogicalPages});
    Tensor qo(q.p, DType::BF16, {D, QHeads, tokens});
    Tensor k8t = pages_view(k8.p, DType::U8, D), v8t = pages_view(v8.p, DType::U8, D);
    Tensor ks8t = pages_view(ks8.p, DType::FP16, 2), vs8t = pages_view(vs8.p, DType::FP16, 2);
    Tensor kdt = pages_view(kd.p, DType::U8, D), ksdt = pages_view(ksd.p, DType::FP16, 2);
    Tensor kbt = pages_view(kb.p, DType::BF16, D), vbt = pages_view(vb.p, DType::BF16, D);
    ops::Gemma4CacheView stored{ops::Gemma4KvFormat::Fp8E4m3Row256, &k8t, &v8t, &ks8t, &vs8t, &t};
    auto derived = key_only_view(&kdt, &ksdt, &t);
    ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &kbt, &vbt, nullptr, nullptr, &t};
    for (const auto* cache : {&stored, &derived, &bf16}) {
        ops::gemma4_qkv_prep(x, qw, kw, p, Rotary, kTheta, 1e-6f, true, qo, *cache, nullptr);
    }
    if (fp8_qk) { stored.qk_mode = derived.qk_mode = ops::Gemma4QkMode::Fp8; }
    const auto offset = static_cast<std::size_t>(tokens - queries) * QHeads * D;
    Tensor qt(static_cast<std::uint16_t*>(q.p) + offset, DType::BF16, {D, QHeads, queries});
    Tensor pt(static_cast<std::int32_t*>(dp.p) + tokens - queries, DType::I32, {queries});
    const std::size_t elements = static_cast<std::size_t>(queries) * QHeads * D;
    DeviceBuffer out(elements * 2);
    Tensor ot(out.p, DType::BF16, {D, QHeads, queries});
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, Heads, queries)));
    std::vector<std::vector<double>> results;
    for (const auto* cache : {&bf16, &stored, &derived}) {
        ops::gemma4_attention(qt, *cache, pt, 0, 1.0f, ot, workspace, nullptr);
        cuda_synchronize();
        results.push_back(from_device_bf16(out.p, elements));
    }
    const double stored_error = relative_rms(results[1], results[0]);
    const double derived_error = relative_rms(results[2], results[0]);
    const double between = relative_rms(results[2], results[1]);
    std::cout << "GEMMA4_KEY_ONLY_VS_STORED first=" << first << " queries=" << queries
              << " qk=" << (fp8_qk ? "fp8" : "bf16") << " keys=" << tokens
              << " stored_fp8_rel_rms_vs_bf16=" << stored_error << " key_only_rel_rms_vs_bf16=" << derived_error
              << " key_only_vs_stored_rel_rms=" << between << '\n';
    // Sanity, not quality: both FP8 caches stay within a few percent of BF16 K/V on this fixture.
    if (!(derived_error < 0.1 && stored_error < 0.1)) {
        std::cerr << "key-only or stored FP8 attention far from BF16: " << derived_error << ", " << stored_error << '\n';
        return 1;
    }
    return 0;
}
} // namespace

INFERNIX_OP_TEST(gemma4_key_only_kv) {
    for (int i = 1; i < argc; i += 2) {
        const std::string_view key(argv[i]);
        if (key != "--split" || i + 1 >= argc || (std::string_view(argv[i + 1]) != "0" && std::string_view(argv[i + 1]) != "1")) {
            std::cerr << "usage: gemma4_key_only_kv [--split 0|1]\n";
            return 2;
        }
        const std::string setting = std::string_view(argv[i + 1]) == "0" ? "1" : "0";
#ifdef _WIN32
        if (_putenv_s("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str()) != 0) { return 2; }
#else
        if (setenv("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str(), 1) != 0) { return 2; }
#endif
    }
    std::cout << "temporary switches | " << temporary_switches::describe(temporary_switches::Scope::Gemma4) << '\n';
    int failures = validation_test();
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return failures ? 1 : 77; }
    const auto frequency = device_frequencies();
    failures += qkv_test(frequency);
    for (const bool far : {false, true}) {
        for (int tokens : {1, 2, 5, 8, 9, 16, 63}) { failures += attention_test(frequency, tokens, false, far); }
        for (int tokens : {64, 65, 129}) {
            failures += attention_test(frequency, tokens, false, far);
            failures += attention_test(frequency, tokens, true, far);
        }
    }
    for (const int first : {0, 131072, 261632}) {
        for (const int queries : {1, 64}) {
            failures += stored_comparison_test(first, queries, false);
        }
        failures += stored_comparison_test(first, 64, true);
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_key_only_kv\n";
    return failures ? 1 : 0;
}
