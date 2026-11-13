// Adapted from ninfer.spo b330aaf (Gemma 4 FP8 KV/QK oracles and event benchmark).
#include "core/temporary_switches.h"
#include "infernix/ops/gemma4_attention.h"
#include "infernix/ops/gemma4_kv_append.h"
#include "infernix/ops/gemma4_qkv_prep.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/launcher/gemma4_attention.h"
#include "ops/launcher/gemma4_attention_fp8_qk.cuh"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <vector>
#include <utility>

using namespace infernix;
using namespace infernix::test;

namespace {
constexpr int D = 512;
constexpr int Heads = 2;
constexpr int Pages = 4;
constexpr int Rows = 64 * Heads * Pages;

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
void quantize(const std::uint16_t* x, std::uint8_t* codes, std::uint16_t* scales) {
    for (int group = 0; group < 2; ++group) {
        float maximum = 0.0f;
        bool poison = false;
        for (int i = 0; i < 256; ++i) {
            const float v = bf16_to_f32(x[group * 256 + i]);
            poison |= !std::isfinite(v);
            maximum = std::fmax(maximum, std::fabs(v));
        }
        const float raw = maximum / 448.0f;
        scales[group] = half_bits(poison ? 1.0f : maximum == 0.0f ? 0.0f :
                                  std::fmin(65504.0f, std::fmax(0x1p-24f, raw)));
        const float inverse = maximum == 0.0f ? 0.0f : 1.0f / half_value(scales[group]);
        for (int i = 0; i < 256; ++i) {
            codes[group * 256 + i] = poison ? 0x7f : maximum == 0.0f ? 0 :
                encode(bf16_to_f32(x[group * 256 + i]) * inverse);
        }
    }
}

__global__ void codec_kernel(const __nv_bfloat162* x, std::uint8_t* codes, __half* scales) {
    __nv_bfloat162 values[8];
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    for (int s = 0; s < 8; ++s) { values[s] = x[row * 256 + lane + 32 * s]; }
    ops::detail::gemma4_fp8_store(values, codes, scales, row, lane);
}
__global__ void expand_kernel(const std::uint8_t* codes, const __half* scales,
                              __nv_bfloat16* out, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { out[i] = ops::detail::gemma4_fp8_expand(codes[i], scales[i / 256]); }
}

int codec_test() {
    constexpr int count = 10 * D;
    std::vector<float> values(count, 0.0f);
    for (int row = 1; row < 10; ++row) {
        for (int i = 0; i < D; ++i) {
            const float multiplier = i < 256 ? 1.0f : 17.0f;
            values[row * D + i] = multiplier * float((i * 19) % 101 - 50) / 37.0f;
        }
    }
    for (int i = D; i < 2 * D; ++i) { values[i] *= 0x1p-100f; }
    for (int i = 2 * D; i < 3 * D; ++i) { values[i] *= 0x1p30f; }
    values[3 * D + 17] = std::numeric_limits<float>::quiet_NaN();
    values[4 * D + 301] = std::numeric_limits<float>::infinity();
    values[5 * D + 31] = -std::numeric_limits<float>::infinity();
    for (int i = 6 * D; i < 7 * D; ++i) { values[i] = std::numeric_limits<float>::quiet_NaN(); }
    // Half-scale rounding and row-half boundaries use deliberately different amplitudes.
    values[7 * D + 255] = 0.252f;
    values[7 * D + 256] = -15.125f;
    values[8 * D + 0] = -0.0f;
    auto input = to_device_bf16(values);
    const auto input_bits = from_device<std::uint16_t>(input, count);
    GuardedDeviceBuffer codes(count), scales(10 * 2 * 2), expanded(count * 2);
    codec_kernel<<<10, 32>>>(static_cast<const __nv_bfloat162*>(input.p),
        static_cast<std::uint8_t*>(codes.data()), static_cast<__half*>(scales.data()));
    cuda_check_last_launch("codec_kernel");
    expand_kernel<<<(count + 127) / 128, 128>>>(static_cast<const std::uint8_t*>(codes.data()),
        static_cast<const __half*>(scales.data()), static_cast<__nv_bfloat16*>(expanded.data()), count);
    cuda_synchronize();
    std::vector<std::uint8_t> expected_codes(count);
    std::vector<std::uint16_t> expected_scales(20);
    for (int row = 0; row < 10; ++row) {
        quantize(input_bits.data() + row * D, expected_codes.data() + row * D,
                 expected_scales.data() + row * 2);
    }
    int failures = verify_exact("codec represented scales", from_device<std::uint16_t>(scales.data(), 20), expected_scales);
    failures += verify_exact("codec codes/nonfinite policy", from_device<std::uint8_t>(codes.data(), count), expected_codes);
    const auto got = from_device<std::uint16_t>(expanded.data(), count);
    for (int i = 0; i < count; ++i) {
        const float reference = code_value(expected_codes[i]) * half_value(expected_scales[i / 256]);
        if (std::isnan(reference) ? !std::isnan(bf16_to_f32(got[i])) : got[i] != f32_to_bf16(reference)) {
            std::cerr << "codec FP32-product/BF16 expansion mismatch at " << i << '\n';
            ++failures; break;
        }
    }
    failures += codes.verify_guards("codes") + scales.verify_guards("scales") + expanded.verify_guards("expanded");
    return failures;
}

std::pair<std::uint8_t, std::uint16_t> expansion_boundary() {
    // FP32 product -> BF16 must differ from FP16 product -> BF16.
    std::uint16_t chosen_scale = 0;
    std::uint8_t chosen_code = 0;
    for (int scale = 0x1001; scale < 0x3c00 && !chosen_scale; ++scale) {
        for (int code = 1; code <= 126; ++code) {
            const float product = code_value(static_cast<std::uint8_t>(code)) *
                                  half_value(static_cast<std::uint16_t>(scale));
            if (f32_to_bf16(product) != f32_to_bf16(half_value(half_bits(product)))) {
                chosen_scale = static_cast<std::uint16_t>(scale);
                chosen_code = static_cast<std::uint8_t>(code);
                break;
            }
        }
    }
    return {chosen_code, chosen_scale};
}

int expansion_boundary_test() {
    const auto [chosen_code, chosen_scale] = expansion_boundary();
    if (!chosen_scale) { std::cerr << "no expansion boundary found\n"; return 1; }
    auto codes = to_device(std::vector<std::uint8_t>(256, chosen_code));
    auto scales = to_device(std::vector<std::uint16_t>{chosen_scale});
    GuardedDeviceBuffer expanded(256 * 2);
    expand_kernel<<<2, 128>>>(static_cast<const std::uint8_t*>(codes.p),
        static_cast<const __half*>(scales.p), static_cast<__nv_bfloat16*>(expanded.data()), 256);
    cuda_synchronize();
    const auto expected = f32_to_bf16(code_value(chosen_code) * half_value(chosen_scale));
    return verify_exact("FP32 not FP16 dequant product",
        from_device<std::uint16_t>(expanded.data(), 256), std::vector<std::uint16_t>(256, expected)) +
        expanded.verify_guards("expansion boundary");
}

// Exercise the exact private primitives used by the attention kernel, including
// both copies of the QK state retained by paired output-half warps.
__global__ void qk_probe_kernel(const __nv_bfloat16* q, const std::uint8_t* k,
                                const __half* ks, int physical, int last,
                                std::uint8_t* qc, __half* qs, float* result) {
    __shared__ __align__(16) std::uint8_t query[64 * D];
    __shared__ __align__(16) std::uint8_t keys[16 * D];
    __shared__ __align__(4) __half query_scales[64 * 2];
    __shared__ __align__(4) __half key_scales[16 * 2];
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    ops::detail::gemma4_fp8_qk_queries<64, 8>(q, D, 63, query, query_scales, warp, lane);
    ops::detail::gemma4_fp8_qk_keys<16, 256>(keys, key_scales, k, ks, Heads, 1,
                                          physical, 48, last, tid);
    ops::cp_commit();
    ops::cp_wait<0>();
    __syncthreads();
    for (int i = tid; i < 64 * D; i += 256) {
        qc[i] = query[ops::detail::gemma4_fp8_qk_offset(i / D, i % D)];
    }
    if (tid < 128) { qs[tid] = query_scales[tid]; }
    float score[2][4];
    const int base = (warp & 3) * 16;
    ops::detail::gemma4_fp8_qk_scores<16>(score, query, keys, query_scales, key_scales, base, lane);
    const int row = base + (lane >> 2);
    const int column = 2 * (lane & 3);
    const int copy = warp >> 2;
    for (int nt = 0; nt < 2; ++nt) {
        result[(copy * 64 + row) * 16 + nt * 8 + column] = score[nt][0];
        result[(copy * 64 + row) * 16 + nt * 8 + column + 1] = score[nt][1];
        result[(copy * 64 + row + 8) * 16 + nt * 8 + column] = score[nt][2];
        result[(copy * 64 + row + 8) * 16 + nt * 8 + column + 1] = score[nt][3];
    }
}

int qk_probe_test(int physical, int last, bool nonfinite_keys = false) {
    std::vector<float> query(64 * D);
    fill_uniform(query, 721, -0.5f, 0.5f);
    for (int row = 0; row < 64; ++row) {
        for (int d = 0; d < D; ++d) {
            query[row * D + d] *= d < 256 ? 0.31f + row * 0.01f : 7.3f + row * 0.1f;
        }
    }
    std::fill(query.begin(), query.begin() + D, 0.0f);
    for (int d = 0; d < D; ++d) { query[D + d] *= 0x1p-100f; }
    query[2 * D + 255] = 0x1p30f; // upper scale clamp and saturated outlier
    query[3 * D + 256] = 0.252f;  // lower/upper half boundary
    query[4 * D + 17] = std::numeric_limits<float>::quiet_NaN();
    query[5 * D + 301] = std::numeric_limits<float>::infinity();
    query[6 * D + 31] = -std::numeric_limits<float>::infinity();
    std::fill(query.begin() + 63 * D, query.end(), std::numeric_limits<float>::quiet_NaN());
    auto dq = to_device_bf16(query);
    auto qb = from_device<std::uint16_t>(dq, query.size());
    // The final query row is out of bounds, despite poisoned backing input.
    std::fill(qb.begin() + 63 * D, qb.end(), std::uint16_t{0});
    std::vector<std::uint8_t> expected_codes(query.size());
    std::vector<std::uint16_t> expected_scales(128);
    for (int row = 0; row < 64; ++row) {
        quantize(qb.data() + row * D, expected_codes.data() + row * D, expected_scales.data() + row * 2);
    }
    std::vector<std::uint8_t> keys(Rows * D);
    std::vector<std::uint16_t> scales(Rows * 2);
    for (int row = 0; row < Rows; ++row) {
        scales[row * 2] = half_bits(0.00314159f + (row % 11) * 0.001f);
        scales[row * 2 + 1] = half_bits(0.07123f + (row % 7) * 0.01f);
        for (int d = 0; d < D; ++d) {
            keys[row * D + d] = static_cast<std::uint8_t>((8 + (d * 17 + row * 13) % 110) |
                                                          ((d + row) & 1 ? 128 : 0));
        }
    }
    if (nonfinite_keys && physical >= 0) {
        for (int group = 0; group < 2; ++group) {
            const auto row = ops::detail::gemma4_kv_row(physical, Heads, 1, 50 + group);
            std::fill(keys.begin() + row * D + group * 256,
                      keys.begin() + row * D + (group + 1) * 256, std::uint8_t{0x7f});
            scales[row * 2 + group] = half_bits(1.0f);
        }
    }
    // Poison the unwritten tail: staging must not read its codes or scales.
    if (physical >= 0) {
        for (int slot = last + 1; slot < 64; ++slot) {
            const auto row = ops::detail::gemma4_kv_row(physical, Heads, 1, slot);
            std::fill(keys.begin() + row * D, keys.begin() + (row + 1) * D, std::uint8_t{0x7f});
            scales[row * 2] = scales[row * 2 + 1] = 0x7e00;
        }
    }
    auto dk = to_device(keys), ds = to_device(scales);
    GuardedDeviceBuffer qc(query.size()), qs(256), score(2 * 64 * 16 * sizeof(float));
    qk_probe_kernel<<<1, 256>>>(static_cast<const __nv_bfloat16*>(dq.p),
        static_cast<const std::uint8_t*>(dk.p), static_cast<const __half*>(ds.p), physical, last,
        static_cast<std::uint8_t*>(qc.data()), static_cast<__half*>(qs.data()), static_cast<float*>(score.data()));
    cuda_check_last_launch("qk_probe_kernel");
    cuda_synchronize();
    int failures = verify_exact("transient Q codes", from_device<std::uint8_t>(qc.data(), query.size()), expected_codes);
    failures += verify_exact("transient Q represented scales", from_device<std::uint16_t>(qs.data(), 128), expected_scales);
    const auto got = from_device<float>(score.data(), 2 * 64 * 16);
    for (int row = 0; row < 64; ++row) {
        for (int key = 0; key < 16; ++key) {
            float expected = 0.0f;
            for (int group = 0; group < 2; ++group) {
                float partial = 0.0f;
                const bool mapped = physical >= 0 && 48 + key <= last;
                const auto kr = mapped ? ops::detail::gemma4_kv_row(physical, Heads, 1, 48 + key) : 0;
                for (int d = group * 256; d < (group + 1) * 256; ++d) {
                    const float kval = mapped ? code_value(keys[kr * D + d]) : 0.0f;
                    partial += code_value(expected_codes[row * D + d]) * kval;
                }
                const float product = half_value(expected_scales[row * 2 + group]) *
                                      (mapped ? half_value(scales[kr * 2 + group]) : 0.0f);
                expected += partial * product;
            }
            for (int copy = 0; copy < 2; ++copy) {
                const float actual = got[(copy * 64 + row) * 16 + key];
                if (std::isnan(expected) ? !std::isnan(actual) :
                    !std::isfinite(actual) || std::fabs(actual - expected) > 1e-4f + 3e-6f * std::fabs(expected)) {
                    std::cerr << "FP8 QK represented score mismatch row=" << row << " key=" << key
                              << " copy=" << copy << " got=" << actual << " ref=" << expected << '\n';
                    return failures + 1;
                }
            }
        }
    }
    return failures + qc.verify_guards("Q codes") + qs.verify_guards("Q scales") + score.verify_guards("QK scores");
}

int qkv_test() {
    const std::vector<int> positions{-1, 0, 1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256};
    const std::vector<int> table{2, 0, -1, Pages};
    const int tokens = static_cast<int>(positions.size());
    std::vector<float> input((4 + Heads) * D * tokens), weights(D);
    fill_uniform(input, 101, -3.0f, 3.0f);
    fill_uniform(weights, 102, 0.4f, 1.7f);
    auto dx = to_device_bf16(input), dw = to_device_bf16(weights);
    auto dp = to_device_i32(positions), dt = to_device_i32(table);
    GuardedDeviceBuffer q(4 * D * tokens * 2), qr(q.bytes());
    GuardedDeviceBuffer k(Rows * D), v(Rows * D), ks(Rows * 4), vs(Rows * 4);
    GuardedDeviceBuffer kr(Rows * D * 2), vr(Rows * D * 2);
    std::vector<std::uint8_t> background(Rows * D, 0xa5);
    std::vector<std::uint16_t> scale_background(Rows * 2, 0x3555), bf_background(Rows * D, 0x3f80);
    k.copy_from_host(background.data(), k.bytes()); v.copy_from_host(background.data(), v.bytes());
    ks.copy_from_host(scale_background.data(), ks.bytes()); vs.copy_from_host(scale_background.data(), vs.bytes());
    kr.copy_from_host(bf_background.data(), kr.bytes()); vr.copy_from_host(bf_background.data(), vr.bytes());
    Tensor x(dx.p, DType::BF16, {(4 + Heads) * D, tokens}), w(dw.p, DType::BF16, {D});
    Tensor p(dp.p, DType::I32, {tokens}), t(dt.p, DType::I32, {Pages});
    Tensor qo(q.data(), DType::BF16, {D, 4, tokens}), qro(qr.data(), DType::BF16, {D, 4, tokens});
    Tensor kt(k.data(), DType::U8, {D, 64, Heads, Pages}), vt(v.data(), DType::U8, {D, 64, Heads, Pages});
    Tensor kst(ks.data(), DType::FP16, {2, 64, Heads, Pages}), vst(vs.data(), DType::FP16, {2, 64, Heads, Pages});
    Tensor krt(kr.data(), DType::BF16, {D, 64, Heads, Pages}), vrt(vr.data(), DType::BF16, {D, 64, Heads, Pages});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Fp8E4m3Row256, &kt, &vt, &kst, &vst, &t};
    ops::Gemma4CacheView reference{ops::Gemma4KvFormat::BFloat16, &krt, &vrt, nullptr, nullptr, &t};
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qo, cache, nullptr);
    ops::gemma4_qkv_prep(x, w, w, p, 128, 1000000.0f, 1e-6f, true, qro, reference, nullptr);
    cuda_synchronize();
    int failures = verify_exact("qkv Q unchanged", from_device<std::uint16_t>(q.data(), q.bytes() / 2),
                                 from_device<std::uint16_t>(qr.data(), qr.bytes() / 2));
    const auto kb = from_device<std::uint16_t>(kr.data(), Rows * D);
    const auto vb = from_device<std::uint16_t>(vr.data(), Rows * D);
    auto ek = background, ev = background;
    auto eks = scale_background, evs = scale_background;
    for (int pos : positions) {
        if (pos < 0 || pos / 64 >= Pages) { continue; }
        const int physical = table[pos / 64];
        if (physical < 0 || physical >= Pages) { continue; }
        for (int h = 0; h < Heads; ++h) {
            const auto row = ops::detail::gemma4_kv_row(physical, Heads, h, pos % 64);
            quantize(kb.data() + row * D, ek.data() + row * D, eks.data() + row * 2);
            quantize(vb.data() + row * D, ev.data() + row * D, evs.data() + row * 2);
        }
    }
    failures += verify_exact("qkv K codes after norm/rope", from_device<std::uint8_t>(k.data(), ek.size()), ek);
    failures += verify_exact("qkv V codes from raw K", from_device<std::uint8_t>(v.data(), ev.size()), ev);
    failures += verify_exact("qkv K scales", from_device<std::uint16_t>(ks.data(), eks.size()), eks);
    failures += verify_exact("qkv V scales", from_device<std::uint16_t>(vs.data(), evs.size()), evs);
    for (auto* buffer : {&q, &k, &v, &ks, &vs}) { failures += buffer->verify_guards("qkv FP8"); }
    return failures;
}

// Independent FP64 softmax/PV reference. BF16 QK uses the represented K values;
// opt-in FP8 QK contracts codes/scales without an intermediate BF16 rounding.
// Both modes consume V rounded once from the FP32 product to BF16.
std::vector<double> fp8_attention_reference(
    const std::vector<std::uint16_t>& query, const std::vector<std::uint8_t>& codes,
    const std::vector<std::uint16_t>& scales, const std::vector<std::uint16_t>& values,
    const std::vector<int>& positions, const std::vector<int>& table, bool fp8_qk) {
    std::vector<std::uint8_t> qc(query.size());
    std::vector<std::uint16_t> qs(query.size() / 256);
    if (fp8_qk) {
        for (std::size_t row = 0; row < query.size() / D; ++row) {
            quantize(query.data() + row * D, qc.data() + row * D, qs.data() + row * 2);
        }
    }
    std::vector<double> expected(query.size(), 0.0);
    for (std::size_t token = 0; token < positions.size(); ++token) {
        for (int head = 0; head < 4; ++head) {
            const auto qr = token * 4 + head;
            std::vector<double> logits(positions[token] + 1, -std::numeric_limits<double>::infinity());
            double maximum = -std::numeric_limits<double>::infinity();
            for (int key = 0; key <= positions[token]; ++key) {
                if (key / 64 >= static_cast<int>(table.size())) { continue; }
                const int physical = table[key / 64];
                if (physical < 0 || physical >= Pages) { continue; }
                const auto kr = ops::detail::gemma4_kv_row(physical, Heads, head / 2, key % 64);
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
                logits[key] = score;
                maximum = std::max(maximum, score);
            }
            if (!std::isfinite(maximum)) { continue; }
            double sum = 0.0;
            for (auto& logit : logits) { logit = std::exp(logit - maximum); sum += logit; }
            for (int key = 0; key <= positions[token]; ++key) {
                if (logits[key] == 0.0) { continue; }
                const auto kr = ops::detail::gemma4_kv_row(table[key / 64], Heads, head / 2, key % 64);
                const double weight = logits[key] / sum;
                for (int d = 0; d < D; ++d) {
                    expected[qr * D + d] += weight * bf16_to_f32(values[kr * D + d]);
                }
            }
        }
    }
    return expected;
}

int attention_test(int tokens, bool invalid_high_page = true, bool fp8_qk = false,
                   bool split_context = false, bool empty = false) {
    // Extra unmapped logical pages request multiple segments even for an early query.
    // Some segments and some per-row causal ranges are empty; no poisoned tail may leak.
    std::vector<int> table{2, -1, 0, invalid_high_page ? Pages : 1};
    if (split_context) { table.resize(16, -1); }
    if (empty) { std::fill(table.begin(), table.end(), -1); }
    std::vector<int> positions(tokens);
    for (int i = 0; i < tokens; ++i) { positions[i] = 65 + i; }
    std::vector<float> query(D * 4 * tokens);
    fill_uniform(query, 301, -0.05f, 0.05f);
    if (fp8_qk) {
        for (std::size_t i = 0; i < query.size(); ++i) {
            query[i] *= i % D < 256 ? 0.37f + (i / D % 5) * 0.1f : 7.1f + (i / D % 7) * 0.2f;
        }
        std::fill(query.begin(), query.begin() + D, 0.0f);
        for (int d = 0; d < D; ++d) { query[D + d] *= 0x1p-100f; }
        if (tokens > 1) { query[2 * D + 255] = 0.252f; query[3 * D + 256] = -1.125f; }
    }
    auto dq = to_device_bf16(query), dp = to_device_i32(positions), dt = to_device_i32(table);
    std::vector<std::uint8_t> codes(Rows * D);
    std::vector<std::uint16_t> scales(Rows * 2), expanded(Rows * D);
    for (int row = 0; row < Rows; ++row) {
        const int last = positions.back();
        const bool unwritten = table[last / 64] == row / (64 * Heads) &&
                               row % 64 > last % 64;
        scales[row * 2] = half_bits(0.00314159f + (row % 7) * 0.001f);
        scales[row * 2 + 1] = half_bits(0.0253f + (row % 5) * 0.01f);
        if (unwritten) { scales[row * 2] = scales[row * 2 + 1] = 0x7e00; }
        for (int d = 0; d < D; ++d) {
            codes[row * D + d] = static_cast<std::uint8_t>(((d * 17 + row * 13) % 120) | ((d + row) & 1 ? 128 : 0));
            if (unwritten) { codes[row * D + d] = 0x7f; }
            expanded[row * D + d] = f32_to_bf16(code_value(codes[row * D + d]) * half_value(scales[row * 2 + d / 256]));
        }
    }
    // Keep V distinct from K and inject the FP32-vs-FP16 double-rounding witness
    // into every live row. The oracle consumes that once-rounded BF16 value.
    static const auto boundary = expansion_boundary();
    auto value_codes = codes;
    auto value_scales = scales;
    std::vector<std::uint16_t> value_expanded(Rows * D);
    for (int row = 0; row < Rows; ++row) {
        const bool unwritten = std::isnan(half_value(scales[row * 2]));
        if (!unwritten) { value_scales[row * 2] = boundary.second; }
        for (int d = 0; d < D; ++d) {
            if (!unwritten && d < 256 && d % 7 == 0) { value_codes[row * D + d] = boundary.first; }
            value_expanded[row * D + d] = f32_to_bf16(
                code_value(value_codes[row * D + d]) * half_value(value_scales[row * 2 + d / 256]));
        }
    }
    auto dc = to_device(codes), ds = to_device(scales), de = to_device(expanded);
    auto dvc = to_device(value_codes), dvs = to_device(value_scales), dve = to_device(value_expanded);
    GuardedDeviceBuffer out(query.size() * 2), reference(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, 4, tokens}), pt(dp.p, DType::I32, {tokens}),
        tt(dt.p, DType::I32, {static_cast<std::int32_t>(table.size())});
    Tensor ct(dc.p, DType::U8, {D, 64, Heads, Pages}), st(ds.p, DType::FP16, {2, 64, Heads, Pages});
    Tensor et(de.p, DType::BF16, {D, 64, Heads, Pages});
    Tensor vct(dvc.p, DType::U8, {D, 64, Heads, Pages});
    Tensor vst(dvs.p, DType::FP16, {2, 64, Heads, Pages});
    Tensor vet(dve.p, DType::BF16, {D, 64, Heads, Pages});
    Tensor ot(out.data(), DType::BF16, {D, 4, tokens}), rt(reference.data(), DType::BF16, {D, 4, tokens});
    ops::Gemma4CacheView cache{ops::Gemma4KvFormat::Fp8E4m3Row256, &ct, &vct, &st, &vst, &tt};
    if (fp8_qk) { cache.qk_mode = ops::Gemma4QkMode::Fp8; }
    ops::Gemma4CacheView bf16{ops::Gemma4KvFormat::BFloat16, &et, &vet, nullptr, nullptr, &tt};
    WorkspaceArena workspace(std::max<std::size_t>(256, ops::gemma4_attention_workspace_capacity_bytes(D, 4, Heads, tokens)));
    ops::gemma4_attention(qt, cache, pt, 0, 1.0f, ot, workspace, nullptr);
    ops::gemma4_attention(qt, bf16, pt, 0, 1.0f, rt, workspace, nullptr);
    cuda_synchronize();
    const auto eager = from_device<std::uint16_t>(out.data(), query.size());
    int failures = 0;
    const bool tensor_fp8_qk = fp8_qk && tokens >= 64;
    const auto expected = fp8_attention_reference(from_device<std::uint16_t>(dq, query.size()),
        codes, scales, value_expanded, positions, table, tensor_fp8_qk);
    // Existing tolerances: do not relax either the BF16 or FP8-QK oracle.
    failures += verify_pointwise("represented attention versus FP64 CPU",
        from_device_bf16(out.data(), query.size()), expected,
        tensor_fp8_qk ? PointwiseCriterion{0.002, 0.006} : PointwiseCriterion{2.0e-5, 8.0e-3});
    if (!tensor_fp8_qk) {
        failures += verify_exact("attention same represented BF16 values", eager,
                                  from_device<std::uint16_t>(reference.data(), query.size()));
    }
    for (auto bits : from_device<std::uint16_t>(out.data(), query.size())) {
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "attention consumed poisoned unused tail\n"; ++failures; break;
        }
    }
    // Capture the descriptor dispatch and replay without allocations or descriptor changes.
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
    failures += verify_exact("FP8 eager versus graph replay",
                              from_device<std::uint16_t>(out.data(), query.size()), eager);
    if (split_context) {
        // Same captured grid/workspace, new device positions: every segment is now empty.
        const std::vector<int> negative_positions(tokens, -1);
        cuda_check(cudaMemcpy(dp.p, negative_positions.data(), tokens * sizeof(int),
                              cudaMemcpyHostToDevice), "update captured positions");
        cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch empty partitions");
        cuda_synchronize(stream);
        failures += verify_exact("captured split context reads current positions",
            from_device<std::uint16_t>(out.data(), query.size()),
            std::vector<std::uint16_t>(query.size(), 0));
    }
    cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
    cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    failures += out.verify_guards("FP8 attention") + reference.verify_guards("BF16 attention");
    return failures;
}

int validation_test() {
    // Validation is host-only and never reads these deliberately opaque addresses.
    int storage = 0;
    Tensor k(&storage, DType::U8, {D, 64, Heads, Pages}), v = k;
    Tensor ks(&storage, DType::FP16, {2, 64, Heads, Pages}), vs = ks;
    Tensor table(&storage, DType::I32, {Pages});
    const ops::Gemma4CacheView valid{ops::Gemma4KvFormat::Fp8E4m3Row256, &k, &v, &ks, &vs, &table};
    ops::detail::validate_gemma4_cache(valid, D, true, "test");
    int failures = 0;
    const auto rejects = [&](const ops::Gemma4CacheView& cache, int dim = D, bool global = true) {
        try { ops::detail::validate_gemma4_cache(cache, dim, global, "test"); }
        catch (const std::invalid_argument&) { return; }
        std::cerr << "invalid cache accepted\n"; ++failures;
    };
    auto bad = valid; bad.k_scales = nullptr; rejects(bad);
    bad = valid; bad.v_scales = nullptr; rejects(bad);
    bad = valid; bad.k_pages = nullptr; rejects(bad);
    bad = valid; bad.v_pages = nullptr; rejects(bad);
    bad = valid; bad.block_table = nullptr; rejects(bad);
    bad = valid; bad.format = static_cast<ops::Gemma4KvFormat>(99); rejects(bad);
    rejects(valid, 256); rejects(valid, D, false);
    bad = valid; bad.qk_mode = static_cast<ops::Gemma4QkMode>(99); rejects(bad);
    bad = valid; bad.qk_mode = ops::Gemma4QkMode::Fp8;
    ops::detail::validate_gemma4_cache(bad, D, true, "test");
    rejects(bad, 256); rejects(bad, D, false);
    for (Tensor* plane : {&ks, &vs}) {
        const Tensor saved = *plane;
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        plane->nb[1] += 2; rejects(valid); *plane = saved;
        for (int axis = 0; axis < 4; ++axis) {
            ++plane->ne[axis]; rejects(valid); *plane = saved;
        }
    }
    for (Tensor* plane : {&k, &v}) {
        const Tensor saved = *plane;
        plane->dtype = DType::BF16; rejects(valid); *plane = saved;
        plane->data = nullptr; rejects(valid); *plane = saved;
        plane->nb[1] += 1; rejects(valid); *plane = saved;
        ++plane->ne[3]; rejects(valid); *plane = saved;
    }
    Tensor bfk(&storage, DType::BF16, {D, 64, Heads, Pages});
    bad = valid; bad.format = ops::Gemma4KvFormat::BFloat16; bad.k_pages = &bfk; bad.v_pages = &bfk;
    rejects(bad); // stale FP8 scale planes are not admitted on BF16 descriptors.
    bad.k_scales = bad.v_scales = nullptr;
    ops::detail::validate_gemma4_cache(bad, D, true, "test");
    bad.qk_mode = ops::Gemma4QkMode::Fp8; rejects(bad);
    bad.qk_mode = static_cast<ops::Gemma4QkMode>(99); rejects(bad);
    Tensor sliding(&storage, DType::BF16, {256, 64, Heads, Pages});
    bad.qk_mode = ops::Gemma4QkMode::BFloat16;
    bad.k_pages = bad.v_pages = &sliding;
    ops::detail::validate_gemma4_cache(bad, 256, false, "test");
    bad.qk_mode = ops::Gemma4QkMode::Fp8; rejects(bad, 256, false);
    return failures;
}

int append_test() {
    std::vector<float> values(D * Heads, 0.5f);
    auto input = to_device_bf16(values);
    auto positions = to_device_i32({63});
    auto table = to_device_i32({0});
    GuardedDeviceBuffer kp(D * 64 * Heads * 2), vp(kp.bytes());
    Tensor x(input.p, DType::BF16, {D, Heads, 1}), p(positions.p, DType::I32, {1}), t(table.p, DType::I32, {1});
    Tensor k(kp.data(), DType::BF16, {D, 64, Heads, 1}), v(vp.data(), DType::BF16, {D, 64, Heads, 1});
    ops::gemma4_kv_append(x, x, p, k, v, t, nullptr);
    cuda_synchronize();
    int failures = 0;
    const auto got = from_device<std::uint16_t>(kp.data(), kp.bytes() / 2);
    for (int h = 0; h < Heads; ++h) {
        for (int d = 0; d < D; ++d) {
            if (got[(h * 64 + 63) * D + d] != f32_to_bf16(0.5f)) { ++failures; break; }
        }
    }
    for (Tensor* plane : {&k, &v}) {
        plane->dtype = DType::U8;
        try { ops::gemma4_kv_append(x, x, p, k, v, t, nullptr); ++failures; }
        catch (const std::invalid_argument&) {}
        plane->dtype = DType::BF16;
    }
    return failures + kp.verify_guards("external append K") + vp.verify_guards("external append V");
}

// Synthetic data is generated once, outside all event intervals. Power-of-two
// row scales make FP8 code*scale exactly BF16 representable for a fair storage
// comparison; separate K/V allocations preserve the actual read-traffic shape.
__global__ void benchmark_cache_kernel(std::uint8_t* k, std::uint8_t* v,
                                        __half* ks, __half* vs, __nv_bfloat16* kb,
                                        __nv_bfloat16* vb, std::int64_t elements) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < elements; i += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
        const auto row = i / D;
        const int d = static_cast<int>(i % D);
        const auto kc = static_cast<std::uint8_t>((16 + (i * 17 + row * 13) % 96) |
                                                  ((i + row) & 1 ? 128 : 0));
        const auto vc = static_cast<std::uint8_t>((8 + (i * 23 + row * 7) % 104) |
                                                  ((i / 3 + row) & 1 ? 128 : 0));
        const __half kscale = __float2half_rn(d < 256 ? 0x1p-6f : 0x1p-4f);
        const __half vscale = __float2half_rn(d < 256 ? 0x1p-5f : 0x1p-3f);
        k[i] = kc; v[i] = vc;
        kb[i] = ops::detail::gemma4_fp8_expand(kc, kscale);
        vb[i] = ops::detail::gemma4_fp8_expand(vc, vscale);
        if (d % 256 == 0) { ks[row * 2 + d / 256] = kscale; vs[row * 2 + d / 256] = vscale; }
    }
}

// Bounded independent long-context oracle: two query/head rows and four output
// columns, but every visible key and all 512 QK dimensions. Query width stays 128
// as history grows; no full T*Hq*L*D host reference allocation is needed.
std::vector<double> benchmark_sample_reference(
    const std::vector<float>& query, int history, bool fp8_qk) {
    constexpr int columns[]{0, 255, 256, 511};
    std::vector<double> result;
    for (int sample = 0; sample < 2; ++sample) {
        const int token = sample == 0 ? 0 : 127;
        const int head = sample == 0 ? 0 : 31;
        std::uint16_t qb[D], qs[2]{};
        std::uint8_t qc[D]{};
        for (int d = 0; d < D; ++d) { qb[d] = f32_to_bf16(query[(token * 32 + head) * D + d]); }
        if (fp8_qk) { quantize(qb, qc, qs); }
        std::vector<double> logits(history + token + 1);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int key = 0; key <= history + token; ++key) {
            const auto row = (static_cast<std::int64_t>(key / 64) * 4 + head / 8) * 64 + key % 64;
            double score = 0.0;
            for (int group = 0; group < 2; ++group) {
                double dot = 0.0;
                const double ks = group == 0 ? 0x1p-6 : 0x1p-4;
                for (int d = group * 256; d < (group + 1) * 256; ++d) {
                    const auto i = row * D + d;
                    const auto code = static_cast<std::uint8_t>((16 + (i * 17 + row * 13) % 96) |
                                                               ((i + row) & 1 ? 128 : 0));
                    dot += fp8_qk ? double(code_value(qc[d])) * code_value(code) :
                                    double(bf16_to_f32(qb[d])) * code_value(code) * ks;
                }
                score += fp8_qk ? dot * half_value(qs[group]) * ks : dot;
            }
            logits[key] = score;
            maximum = std::max(maximum, score);
        }
        double sum = 0.0;
        for (auto& score : logits) { score = std::exp(score - maximum); sum += score; }
        for (int d : columns) {
            double value = 0.0;
            for (int key = 0; key <= history + token; ++key) {
                const auto row = (static_cast<std::int64_t>(key / 64) * 4 + head / 8) * 64 + key % 64;
                const auto i = row * D + d;
                const auto code = static_cast<std::uint8_t>((8 + (i * 23 + row * 7) % 104) |
                                                           ((i / 3 + row) & 1 ? 128 : 0));
                const double vs = d < 256 ? 0x1p-5 : 0x1p-3;
                value += logits[key] * code_value(code) * vs;
            }
            result.push_back(value / sum);
        }
    }
    return result;
}

int attention_benchmark(int history, int Chunk) {
    constexpr int QHeads = 32;
    constexpr int KvHeads = 4;
    constexpr int Warmups = 3;
    constexpr int Samples = 7;
    constexpr int Repetitions = 3;
    const int pages = (history + Chunk + 63) / 64;
    const std::int64_t elements = static_cast<std::int64_t>(pages) * 64 * KvHeads * D;
    const std::size_t code_bytes = static_cast<std::size_t>(elements);
    DeviceBuffer k(code_bytes), v(code_bytes), ks(code_bytes / 128), vs(code_bytes / 128);
    DeviceBuffer kb(code_bytes * 2), vb(code_bytes * 2);
    benchmark_cache_kernel<<<4096, 256>>>(static_cast<std::uint8_t*>(k.p), static_cast<std::uint8_t*>(v.p),
        static_cast<__half*>(ks.p), static_cast<__half*>(vs.p),
        static_cast<__nv_bfloat16*>(kb.p), static_cast<__nv_bfloat16*>(vb.p), elements);
    cuda_check_last_launch("benchmark_cache_kernel");
    std::vector<float> query(D * QHeads * Chunk);
    fill_uniform(query, 20261104, -0.15f, 0.15f);
    std::vector<int> positions(Chunk), table(pages);
    fill_iota_i32(positions, history);
    fill_iota_i32(table);
    auto dq = to_device_bf16(query), dp = to_device_i32(positions), dt = to_device_i32(table);
    DeviceBuffer output(query.size() * 2);
    Tensor qt(dq.p, DType::BF16, {D, QHeads, Chunk}), pt(dp.p, DType::I32, {Chunk});
    Tensor tt(dt.p, DType::I32, {pages}), ot(output.p, DType::BF16, {D, QHeads, Chunk});
    Tensor kt(k.p, DType::U8, {D, 64, KvHeads, pages}), vt(v.p, DType::U8, {D, 64, KvHeads, pages});
    Tensor kst(ks.p, DType::FP16, {2, 64, KvHeads, pages}), vst(vs.p, DType::FP16, {2, 64, KvHeads, pages});
    Tensor kbt(kb.p, DType::BF16, {D, 64, KvHeads, pages}), vbt(vb.p, DType::BF16, {D, 64, KvHeads, pages});
    ops::Gemma4CacheView modes[]{
        {ops::Gemma4KvFormat::BFloat16, &kbt, &vbt, nullptr, nullptr, &tt},
        {ops::Gemma4KvFormat::Fp8E4m3Row256, &kt, &vt, &kst, &vst, &tt},
        {ops::Gemma4KvFormat::Fp8E4m3Row256, &kt, &vt, &kst, &vst, &tt}};
    modes[2].qk_mode = ops::Gemma4QkMode::Fp8;
    constexpr int ModeCount = 3;
    const char* labels[]{"bf16", "fp8_phase1", "fp8_phase2"};
    WorkspaceArena workspace(std::max<std::size_t>(256,
        ops::gemma4_attention_workspace_capacity_bytes(D, QHeads, KvHeads, Chunk)));
    cudaStream_t stream;
    cudaEvent_t begin, end;
    cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
    cuda_check(cudaEventCreate(&begin), "cudaEventCreate");
    cuda_check(cudaEventCreate(&end), "cudaEventCreate");
    const auto launch = [&](int mode) {
        ops::gemma4_attention(qt, modes[mode], pt, 0, 1.0f, ot, workspace, stream);
    };
    cuda_synchronize();
    for (int warmup = 0; warmup < Warmups; ++warmup) {
        for (int mode = 0; mode < ModeCount; ++mode) { launch(mode); }
    }
    cuda_synchronize(stream);
    for (int mode = 0; mode < ModeCount && Chunk >= 64; ++mode) {
        const auto resources = ops::detail::gemma4_attention_mma_d512_resources(modes[mode]);
        int device = 0, max_threads = 0;
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerMultiProcessor, device),
                   "cudaDeviceGetAttribute max threads");
        std::cout << "GEMMA4_D512_RESOURCES history=" << history << " chunk=" << Chunk
                  << " mode=" << labels[mode] << " registers_per_thread=" << resources.attributes.numRegs
                  << " static_shared_bytes=" << resources.attributes.sharedSizeBytes
                  << " dynamic_shared_bytes=" << resources.dynamic_shared_bytes
                  << " local_bytes_per_thread=" << resources.attributes.localSizeBytes
                  << " threads=" << resources.threads
                  << " api_max_active_blocks_per_sm=" << resources.active_blocks_per_sm
                  << " api_max_active_warps_per_sm=" << resources.active_blocks_per_sm * resources.threads / 32
                  << " api_occupancy_pct="
                  << 100.0 * resources.active_blocks_per_sm * resources.threads / max_threads
                  << " occupancy_source=cuda_runtime_limit_not_measured\n";
    }
    std::vector<float> timings[ModeCount];
    for (int sample = 0; sample < Samples && Chunk >= 64; ++sample) {
        for (int order = 0; order < ModeCount; ++order) {
            const int mode = (sample + order) % ModeCount;
            cuda_check(cudaEventRecord(begin, stream), "cudaEventRecord");
            for (int repeat = 0; repeat < Repetitions; ++repeat) { launch(mode); }
            cuda_check(cudaEventRecord(end, stream), "cudaEventRecord");
            cuda_check(cudaEventSynchronize(end), "cudaEventSynchronize");
            float ms = 0.0f;
            cuda_check(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
            timings[mode].push_back(ms / Repetitions);
            std::cout << "GEMMA4_QK_EVENT_SAMPLE history=" << history << " chunk=" << Chunk
                      << " mode=" << labels[mode] << " sample=" << sample
                      << " repetitions=" << Repetitions << " ms_per_op=" << ms / Repetitions << '\n';
        }
    }
    for (int mode = 0; mode < ModeCount && Chunk >= 64; ++mode) {
        std::sort(timings[mode].begin(), timings[mode].end());
        std::cout << "GEMMA4_QK_EVENT_MEDIAN history=" << history << " chunk=" << Chunk
                  << " q_heads=" << QHeads << " kv_heads=" << KvHeads << " mode=" << labels[mode]
                  << " warmups=" << Warmups << " samples=" << Samples << " repetitions=" << Repetitions
                  << " median_ms=" << timings[mode][Samples / 2]
                  << " min_ms=" << timings[mode].front() << " max_ms=" << timings[mode].back()
                  << " execution=eager data=synthetic_represented_equal\n";
    }
    int failures = 0;
    if (Chunk == 128) {
        const auto expected_bf16 = benchmark_sample_reference(query, history, false);
        const auto expected_fp8 = benchmark_sample_reference(query, history, true);
        constexpr int columns[]{0, 255, 256, 511};
        for (int mode = 0; mode < ModeCount; ++mode) {
            launch(mode);
            cuda_synchronize(stream);
            const auto got = from_device<std::uint16_t>(output.p, query.size());
            std::vector<double> sampled;
            for (int sample = 0; sample < 2; ++sample) {
                const int token = sample == 0 ? 0 : 127;
                const int head = sample == 0 ? 0 : 31;
                for (int d : columns) { sampled.push_back(bf16_to_f32(got[(token * 32 + head) * D + d])); }
            }
            const bool fp8_qk = mode == 2;
            const auto& expected = fp8_qk ? expected_fp8 : expected_bf16;
            double max_abs = 0.0, squared = 0.0;
            for (std::size_t i = 0; i < sampled.size(); ++i) {
                const double error = double(sampled[i]) - expected[i];
                max_abs = std::max(max_abs, std::fabs(error));
                squared += error * error;
            }
            failures += verify_pointwise("long fixed-width sampled FP64 CPU", sampled, expected,
                fp8_qk ? PointwiseCriterion{0.002, 0.006} : PointwiseCriterion{2.0e-5, 8.0e-3});
            std::cout << "GEMMA4_LONG_ERROR history=" << history << " chunk=" << Chunk
                      << " mode=" << labels[mode] << " sampled_rows=2 sampled_columns=4"
                      << " all_visible_keys=true qk_dimensions=512 max_abs_error=" << max_abs
                      << " rms_error=" << std::sqrt(squared / sampled.size()) << '\n';
        }
    }
    cuda_check(cudaEventDestroy(begin), "cudaEventDestroy");
    cuda_check(cudaEventDestroy(end), "cudaEventDestroy");
    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return failures;
}

// Comma-separated positive integers; empty or malformed lists are rejected.
std::vector<int> parse_list(std::string_view text) {
    std::vector<int> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        const std::string item(text.substr(0, comma));
        char* end = nullptr;
        const long value = std::strtol(item.c_str(), &end, 10);
        if (item.empty() || *end != '\0' || value <= 0 || value > (1 << 22)) { return {}; }
        values.push_back(static_cast<int>(value));
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    }
    return values;
}
} // namespace

INFERNIX_OP_TEST(gemma4_fp8_kv) {
    std::vector<int> chunks{128, 512, 640, 1024};
    std::vector<int> histories{8192, 32768, 131072};
    bool benchmark = false, valid = true;
    int segments = -1;
    for (int i = 1; valid && i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--benchmark" || key == "--bench") { benchmark = true; continue; }
        if (++i >= argc) { valid = false; break; }
        const std::string_view text(argv[i]);
        if (key == "--split" && segments < 0 && (text == "0" || text == "1")) {
            segments = text == "0" ? 1 : 0;
        } else if (key == "--segments" && segments < 0) {
            for (int n = 1; n <= 13; ++n) {
                if (text == std::to_string(n)) { segments = n; break; }
            }
            valid = segments >= 1;
        } else if (key == "--chunks" || key == "--histories") {
            auto values = parse_list(text);
            if (values.empty()) { valid = false; }
            else if (key == "--chunks") { chunks = std::move(values); }
            else { histories = std::move(values); }
        } else { valid = false; }
    }
    valid = valid && std::all_of(chunks.begin(), chunks.end(), [](int n) { return n >= 64; });
    if (!valid) {
        std::cerr << "usage: gemma4_fp8_kv [--split 0|1 | --segments 1..13] "
                     "[--benchmark [--chunks N,... (>=64)] [--histories N,...]]\n";
        return 2;
    }
    // All command-line overrides precede the process-wide immutable switch snapshot.
    if (segments >= 0) {
        const std::string setting = std::to_string(segments);
#ifdef _WIN32
        const int status = _putenv_s("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str());
#else
        const int status = setenv("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str(), 1);
#endif
        if (status != 0) { std::cerr << "cannot set attention split override\n"; return 2; }
    }
    std::cout << "temporary switches | "
              << temporary_switches::describe(temporary_switches::Scope::Gemma4) << '\n';
    if (benchmark) {
        if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return 77; }
        int device = 0;
        cudaDeviceProp properties{};
        cuda_check(cudaGetDevice(&device), "cudaGetDevice");
        cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
        std::cout << "GEMMA4_QK_BENCHMARK gpu=" << properties.name
                  << " sm=" << properties.major << properties.minor
                  << " timing=cuda_events order=rotating_paired\n";
        int failures = 0;
        for (int chunk : chunks) {
            for (int history : histories) { failures += attention_benchmark(history, chunk); }
        }
        return failures ? 1 : 0; // Numerical assertions only, never performance thresholds.
    }
    int failures = validation_test();
    if (cuda_unavailable()) { std::cout << "SKIP: no usable CUDA device\n"; return failures ? 1 : 77; }
    failures += codec_test() + expansion_boundary_test() + qkv_test() + append_test();
    failures += qk_probe_test(2, 63) + qk_probe_test(0, 52) + qk_probe_test(-1, 63);
    failures += qk_probe_test(2, 63, true);
    // Shared-KV decode is the production default; no obsolete decode or raw-V arms.
    for (int tokens = 1; tokens <= 16; ++tokens) { failures += attention_test(tokens); }
    failures += attention_test(16, false);
    failures += attention_test(63, false);
    for (int tokens : {1, 5, 16}) {
        failures += attention_test(tokens, false, false, true);
        failures += attention_test(tokens, true, false, true, true);
    }
    for (int tokens : {64, 65, 129}) { failures += attention_test(tokens); }
    failures += attention_test(129, false);
    for (int tokens : {1, 63, 64, 65, 127, 128, 129}) {
        failures += attention_test(tokens, true, true);
    }
    failures += attention_test(129, false, true);
    for (bool fp8_qk : {false, true}) {
        for (int tokens : {64, 65, 128, 129}) {
            failures += attention_test(tokens, false, fp8_qk, true);
        }
        failures += attention_test(65, true, fp8_qk, true, true);
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_fp8_kv\n";
    return failures ? 1 : 0;
}
