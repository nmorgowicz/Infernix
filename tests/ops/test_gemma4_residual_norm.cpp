#include "infernix/ops/gemma4_residual_norm.h"
#include "infernix/ops/residual_add.h"
#include "infernix/ops/residual_scale.h"
#include "infernix/ops/rmsnorm.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr float kEps = 1.0e-6f;
// BF16 roundings of the normed projection (|v| <= ~6) and of the residual sum (|x| <= ~10).
constexpr PointwiseCriterion kCriterion{2.5e-2, 1.0e-2};

std::vector<std::uint16_t> to_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
    return out;
}

std::vector<std::uint16_t> read_bits(const GuardedDeviceBuffer& buffer) {
    std::vector<std::uint16_t> out(buffer.bytes() / 2);
    buffer.copy_to_host(out.data(), buffer.bytes());
    return out;
}

// `exact` also runs rmsnorm, residual_add, residual_scale and rmsnorm and requires equal bits.
int run_case(int hidden, int tokens, float scale, bool next, bool exact) {
    const std::string label = "residual_norm H=" + std::to_string(hidden) +
                              " T=" + std::to_string(tokens) + " scale=" + std::to_string(scale) +
                              (next ? " next" : "");
    const std::size_t count = static_cast<std::size_t>(hidden) * tokens;
    std::vector<float> projected(count), x(count), post(static_cast<std::size_t>(hidden)),
        next_w(static_cast<std::size_t>(hidden));
    fill_uniform(projected, 31u + static_cast<std::uint32_t>(tokens), -4.0f, 4.0f);
    fill_uniform(x, 32u + static_cast<std::uint32_t>(tokens), -6.0f, 6.0f);
    fill_uniform(post, 33u, 0.25f, 1.5f);
    fill_uniform(next_w, 34u, 0.25f, 1.5f);
    for (std::vector<float>* values : {&projected, &x, &post, &next_w}) { round_to_bf16(*values); }

    GuardedDeviceBuffer dproj(count * 2), dx(count * 2), dpost(post.size() * 2),
        dnext(next_w.size() * 2), dout(count * 2), dx_ref(count * 2), dnormed(count * 2),
        dout_ref(count * 2);
    dproj.copy_from_host(to_bits(projected).data(), dproj.bytes());
    dx.copy_from_host(to_bits(x).data(), dx.bytes());
    dx_ref.copy_from_host(to_bits(x).data(), dx_ref.bytes());
    dpost.copy_from_host(to_bits(post).data(), dpost.bytes());
    dnext.copy_from_host(to_bits(next_w).data(), dnext.bytes());
    Tensor proj_t(dproj.data(), DType::BF16, {hidden, tokens});
    Tensor x_t(dx.data(), DType::BF16, {hidden, tokens});
    Tensor post_t(dpost.data(), DType::BF16, {hidden});
    Tensor next_t(dnext.data(), DType::BF16, {hidden});
    Tensor out_t(dout.data(), DType::BF16, {hidden, tokens});
    ops::gemma4_residual_norm(proj_t, post_t, scale, next ? &next_t : nullptr, kEps, x_t,
                              next ? &out_t : nullptr, nullptr);
    if (exact) {
        Tensor x_ref(dx_ref.data(), DType::BF16, {hidden, tokens});
        Tensor normed(dnormed.data(), DType::BF16, {hidden, tokens});
        Tensor out_ref(dout_ref.data(), DType::BF16, {hidden, tokens});
        ops::rmsnorm(proj_t, post_t, kEps, false, normed, nullptr);
        ops::residual_add(normed, x_ref, nullptr);
        if (scale != 1.0f) { ops::residual_scale(x_ref, scale, nullptr); }
        if (next) { ops::rmsnorm(x_ref, next_t, kEps, false, out_ref, nullptr); }
    }
    cuda_synchronize();

    int failures        = 0;
    const auto x_bits   = read_bits(dx);
    const auto out_bits = read_bits(dout);
    if (exact && x_bits != read_bits(dx_ref)) {
        std::cerr << label << ": x differs from rmsnorm + residual_add + residual_scale\n";
        ++failures;
    }
    if (exact && next && out_bits != read_bits(dout_ref)) {
        std::cerr << label << ": next_out differs from rmsnorm\n";
        ++failures;
    }

    std::vector<double> got, expected;
    for (int t = 0; t < tokens; ++t) {
        const std::size_t base = static_cast<std::size_t>(t) * hidden;
        double squares         = 0.0;
        for (int i = 0; i < hidden; ++i) {
            squares += static_cast<double>(projected[base + i]) * projected[base + i];
        }
        const double inv = 1.0 / std::sqrt(squares / hidden + static_cast<double>(kEps));
        std::vector<double> sum(static_cast<std::size_t>(hidden));
        double next_squares = 0.0;
        for (int i = 0; i < hidden; ++i) {
            sum[static_cast<std::size_t>(i)] =
                (x[base + i] + projected[base + i] * inv * post[static_cast<std::size_t>(i)]) *
                scale;
            next_squares += sum[static_cast<std::size_t>(i)] * sum[static_cast<std::size_t>(i)];
            got.push_back(bf16_to_f32(x_bits[base + i]));
            expected.push_back(sum[static_cast<std::size_t>(i)]);
        }
        if (!next) { continue; }
        const double next_inv = 1.0 / std::sqrt(next_squares / hidden + static_cast<double>(kEps));
        for (int i = 0; i < hidden; ++i) {
            got.push_back(bf16_to_f32(out_bits[base + i]));
            expected.push_back(sum[static_cast<std::size_t>(i)] * next_inv *
                               next_w[static_cast<std::size_t>(i)]);
        }
    }
    failures += verify_pointwise(label, got, expected, kCriterion);
    failures += dx.verify_guards("x");
    failures += dout.verify_guards("next_out");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gemma4_residual_norm) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (int tokens : {1, 3, 2048}) {
        for (float scale : {1.0f, 0.734375f}) {
            for (bool next : {false, true}) {
                failures += run_case(5376, tokens, scale, next, true);
            }
        }
    }
    failures += run_case(256, 5, 0.5f, true, false);
    failures += run_case(256, 1, 1.0f, false, false);
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_residual_norm\n";
    return failures ? 1 : 0;
}
