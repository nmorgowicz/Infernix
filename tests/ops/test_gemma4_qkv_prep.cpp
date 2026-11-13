#include "core/device.h"
#include "infernix/ops/gemma4_qkv_prep.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kPage  = 64;
constexpr float kEps = 1.0e-6f;
// Two BF16 roundings (after the norm and after the rotation) of values up to about 6.
constexpr PointwiseCriterion kCriterion{1.0e-2, 1.6e-2};

struct Case {
    const char* label;
    int head_dim;
    int q_heads;
    int kv_heads;
    int rotary_dim;
    float theta;
    bool v_from_k;
    int tokens;
    int first_position;
};

std::vector<std::uint16_t> to_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

// FP64 norm then rotation of one head, from the represented BF16 inputs.
std::vector<double> oracle_head(const float* x, const float* weight, int d, int rotary_dim,
                                float theta, int position, bool rotate) {
    double squares = 0.0;
    for (int i = 0; i < d; ++i) { squares += static_cast<double>(x[i]) * x[i]; }
    const double inv = 1.0 / std::sqrt(squares / d + static_cast<double>(kEps));
    std::vector<double> normed(static_cast<std::size_t>(d));
    for (int i = 0; i < d; ++i) {
        normed[static_cast<std::size_t>(i)] = x[i] * inv * (weight != nullptr ? weight[i] : 1.0);
    }
    if (!rotate) { return normed; }
    std::vector<double> out = normed;
    const int half          = rotary_dim / 2;
    for (int i = 0; i < half; ++i) {
        const double angle = position * std::pow(static_cast<double>(theta), -2.0 * i / rotary_dim);
        const double c     = std::cos(angle);
        const double s     = std::sin(angle);
        out[static_cast<std::size_t>(i)]        = normed[i] * c - normed[i + half] * s;
        out[static_cast<std::size_t>(i + half)] = normed[i + half] * c + normed[i] * s;
    }
    return out;
}

std::size_t page_index(const Case& c, int pages, int head, int position, int dim) {
    const int physical = pages - 1 - position / kPage;
    return static_cast<std::size_t>(c.head_dim) * kPage *
               (static_cast<std::size_t>(head) + static_cast<std::size_t>(c.kv_heads) * physical) +
           static_cast<std::size_t>(c.head_dim) * (position % kPage) + dim;
}

int run_case(const Case& c) {
    const int d                  = c.head_dim;
    const int q_rows             = c.q_heads * d;
    const int k_rows             = c.kv_heads * d;
    const int rows               = q_rows + (c.v_from_k ? 1 : 2) * k_rows;
    const int last               = c.first_position + c.tokens;
    const int pages              = (last + kPage - 1) / kPage + 1;
    const std::size_t page_count = static_cast<std::size_t>(d) * kPage * c.kv_heads * pages;

    std::vector<float> qkv(static_cast<std::size_t>(rows) * c.tokens);
    std::vector<float> q_norm(static_cast<std::size_t>(d)), k_norm(static_cast<std::size_t>(d));
    std::vector<float> background(page_count);
    fill_uniform(qkv, 11u + static_cast<std::uint32_t>(d), -3.0f, 3.0f);
    fill_uniform(q_norm, 12u, 0.5f, 1.5f);
    fill_uniform(k_norm, 13u, 0.5f, 1.5f);
    fill_uniform(background, 14u, -1.0f, 1.0f);
    round_to_bf16(qkv);
    round_to_bf16(q_norm);
    round_to_bf16(k_norm);
    round_to_bf16(background);

    std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
    for (int page = 0; page < pages; ++page) {
        table[static_cast<std::size_t>(page)] = pages - 1 - page;
    }
    std::vector<std::int32_t> positions(static_cast<std::size_t>(c.tokens));
    for (int t = 0; t < c.tokens; ++t) {
        positions[static_cast<std::size_t>(t)] = c.first_position + t;
    }

    // Split Q, K, V columns for the oracle.
    std::vector<float> q_split(static_cast<std::size_t>(q_rows) * c.tokens);
    std::vector<float> k_split(static_cast<std::size_t>(k_rows) * c.tokens);
    std::vector<float> v_split(k_split.size());
    for (int t = 0; t < c.tokens; ++t) {
        const float* column = qkv.data() + static_cast<std::size_t>(t) * rows;
        for (int i = 0; i < q_rows; ++i) {
            q_split[static_cast<std::size_t>(t) * q_rows + i] = column[i];
        }
        for (int i = 0; i < k_rows; ++i) {
            k_split[static_cast<std::size_t>(t) * k_rows + i] = column[q_rows + i];
            v_split[static_cast<std::size_t>(t) * k_rows + i] =
                column[(c.v_from_k ? q_rows : q_rows + k_rows) + i];
        }
    }

    const auto qkv_bits        = to_bits(qkv);
    const auto background_bits = to_bits(background);
    GuardedDeviceBuffer dqkv(qkv_bits.size() * 2), dqn(static_cast<std::size_t>(d) * 2),
        dkn(static_cast<std::size_t>(d) * 2), dpos(positions.size() * 4),
        dtable(table.size() * 4), dq_out(q_split.size() * 2), dk_pages(page_count * 2),
        dv_pages(page_count * 2);
    dqkv.copy_from_host(qkv_bits.data(), dqkv.bytes());
    dqn.copy_from_host(to_bits(q_norm).data(), dqn.bytes());
    dkn.copy_from_host(to_bits(k_norm).data(), dkn.bytes());
    dpos.copy_from_host(positions.data(), dpos.bytes());
    dtable.copy_from_host(table.data(), dtable.bytes());
    for (GuardedDeviceBuffer* pages_buffer : {&dk_pages, &dv_pages}) {
        pages_buffer->copy_from_host(background_bits.data(), pages_buffer->bytes());
    }

    Tensor qkv_tensor(dqkv.data(), DType::BF16, {rows, c.tokens});
    Tensor q_norm_tensor(dqn.data(), DType::BF16, {d});
    Tensor k_norm_tensor(dkn.data(), DType::BF16, {d});
    Tensor pos_tensor(dpos.data(), DType::I32, {c.tokens});
    Tensor table_tensor(dtable.data(), DType::I32, {pages});
    Tensor q_out(dq_out.data(), DType::BF16, {d, c.q_heads, c.tokens});
    Tensor k_pages(dk_pages.data(), DType::BF16, {d, kPage, c.kv_heads, pages});
    Tensor v_pages(dv_pages.data(), DType::BF16, {d, kPage, c.kv_heads, pages});

    const ops::Gemma4CacheView cache{ops::Gemma4KvFormat::BFloat16, &k_pages, &v_pages,
                                    nullptr, nullptr, &table_tensor};
    ops::gemma4_qkv_prep(qkv_tensor, q_norm_tensor, k_norm_tensor, pos_tensor, c.rotary_dim,
                         c.theta, kEps, c.v_from_k, q_out, cache, nullptr);
    cuda_synchronize();

    int failures       = 0;
    const auto bits_of = [](const GuardedDeviceBuffer& buffer) {
        std::vector<std::uint16_t> out(buffer.bytes() / 2);
        buffer.copy_to_host(out.data(), buffer.bytes());
        return out;
    };
    const auto q_bits = bits_of(dq_out);
    const auto k_bits = bits_of(dk_pages);
    const auto v_bits = bits_of(dv_pages);

    std::vector<double> got, expected;
    for (int t = 0; t < c.tokens; ++t) {
        const int position = positions[static_cast<std::size_t>(t)];
        for (int h = 0; h < c.q_heads; ++h) {
            const float* x = q_split.data() + (static_cast<std::size_t>(t) * c.q_heads + h) * d;
            const auto ref =
                oracle_head(x, q_norm.data(), d, c.rotary_dim, c.theta, position, true);
            for (int i = 0; i < d; ++i) {
                expected.push_back(ref[static_cast<std::size_t>(i)]);
                got.push_back(
                    bf16_to_f32(q_bits[(static_cast<std::size_t>(t) * c.q_heads + h) * d + i]));
            }
        }
        for (int h = 0; h < c.kv_heads; ++h) {
            const std::size_t offset = (static_cast<std::size_t>(t) * c.kv_heads + h) * d;
            const auto k_expect      = oracle_head(k_split.data() + offset, k_norm.data(), d,
                                                   c.rotary_dim, c.theta, position, true);
            const auto v_expect = oracle_head(v_split.data() + offset, nullptr, d, c.rotary_dim,
                                              c.theta, position, false);
            for (int i = 0; i < d; ++i) {
                const std::size_t at = page_index(c, pages, h, position, i);
                expected.push_back(k_expect[static_cast<std::size_t>(i)]);
                got.push_back(bf16_to_f32(k_bits[at]));
                expected.push_back(v_expect[static_cast<std::size_t>(i)]);
                got.push_back(bf16_to_f32(v_bits[at]));
            }
        }
    }
    failures += verify_pointwise(c.label, got, expected, kCriterion);
    failures += dq_out.verify_guards("q_out");
    failures += dk_pages.verify_guards("k_pages");
    failures += dv_pages.verify_guards("v_pages");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gemma4_qkv_prep) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const Case cases[] = {
        {"sliding decode", 256, 32, 16, 256, 10000.0f, false, 1, 0},
        {"sliding decode late", 256, 32, 16, 256, 10000.0f, false, 1, 4099},
        {"sliding chunk across pages", 256, 32, 16, 256, 10000.0f, false, 150, 30},
        {"sliding verify", 256, 32, 16, 256, 10000.0f, false, 6, 1100},
        {"global decode", 512, 32, 4, 128, 1000000.0f, true, 1, 0},
        {"global decode late", 512, 32, 4, 128, 1000000.0f, true, 1, 4099},
        {"global chunk across pages", 512, 32, 4, 128, 1000000.0f, true, 130, 100},
        {"global with V rows", 512, 8, 4, 128, 1000000.0f, false, 3, 63},
    };
    int failures = 0;
    for (const Case& c : cases) { failures += run_case(c); }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_qkv_prep\n";
    return failures ? 1 : 0;
}
