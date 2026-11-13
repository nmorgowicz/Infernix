#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/q4/q4_g32_w4a8.cuh"
#include "ops/op_tester.h"
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace infernix::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }
int q4_g32_a16() {
    constexpr std::array kCalls{a16(1), a16(7), a16(16), a16(32)};
    int failures = 0;
    failures += run_shape("Q4_G32", ActivationCompute::A16, make_q4_g32_fp16_weight,
                          {128, 256, 41U, Comparison::Full, true, kCalls});
    failures += run_shape("Q4_G32", ActivationCompute::A16, make_q4_g32_fp16_weight,
                          {256, 5376, 43U, Comparison::Full, true, kCalls});
    constexpr std::array kShortK{a16(7)};
    failures += run_shape("Q4_G32", ActivationCompute::A16, make_q4_g32_fp16_weight,
                          {128, 8192, 53U, Comparison::Full, true, kShortK});
    // T=1 on every K of the unrolled DRAM GEMV, with one and four rows per warp.
    constexpr std::array kOne{a16(1)};
    for (const std::int32_t k : {8192, 16384, 21504}) {
        failures += run_shape("Q4_G32", ActivationCompute::A16, make_q4_g32_fp16_weight,
                              {64, k, 59U, Comparison::Full, true, kOne});
    }
    failures += run_shape("Q4_G32", ActivationCompute::A16, make_q4_g32_fp16_weight,
                          {8192, 5376, 61U, Comparison::Full, true, kOne});
    constexpr std::array kA8{Invocation{128, CallForm::Policy, infernix::ops::LinearPolicy::AllowA8, false}};
    failures += run_shape("Q4_G32_A8", ActivationCompute::A8, make_q4_g32_fp16_weight,
                          {128, 5376, 47U, Comparison::Full, true, kA8});
    return failures;
}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16;
    const int exponent       = (bits >> 10) & 0x1f;
    std::uint32_t mantissa   = bits & 0x3ffu;
    if (exponent == 0) {
        const float value = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -value : value;
    }
    if (exponent == 31) { return sign ? -INFINITY : INFINITY; }
    const std::uint32_t word = sign | (static_cast<std::uint32_t>(exponent - 15 + 127) << 23) | (mantissa << 13);
    float out;
    std::memcpy(&out, &word, 4);
    return out;
}

float bf16_round_up(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, 4);
    if ((word & 0xffffu) != 0) { word = (word & 0xffff0000u) + 0x10000u; }
    float out;
    std::memcpy(&out, &word, 4);
    return out;
}

double e4m3_value(int code) {
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (code & 0x80) ? -magnitude : magnitude;
}

// Round to nearest E4M3FN, ties to the even code, saturating at 448.
double e4m3_round(float value) {
    const double magnitude = std::fabs(static_cast<double>(value));
    int best               = 0;
    for (int code = 1; code <= 126; ++code) {
        const double candidate = e4m3_value(code);
        const double distance  = std::fabs(candidate - magnitude);
        const double current   = std::fabs(e4m3_value(best) - magnitude);
        if (distance < current || (distance == current && (code & 1) == 0)) { best = code; }
    }
    return value < 0 ? -e4m3_value(best) : e4m3_value(best);
}

// The q4 W4A8 prefill route rounds each q4 row to row-scaled E4M3 (q4_bake_e4m3_kernel). This
// oracle applies the same FP32 steps on the host, then checks the GEMM against FP64 with the A8
// activation allowance. Rows are sampled to keep the host product short.
int q4_g32_baked_a8() {
    using namespace infernix;
    int failures = 0;
    struct Case {
        std::int32_t n, k, t;
    };
    // 9 to 384 tokens take the fused route (32-token tile to 32, 64-token above), 512 the bake
    // route; the split-K cases differ from the bake route only in summation order.
    for (const Case c : {Case{5376, 8192, 128}, Case{16384, 5376, 200}, Case{5376, 21504, 512},
                         Case{5376, 21504, 17}, Case{43008, 5376, 64}, Case{18432, 5376, 33},
                         Case{16384, 5376, 32}, Case{5376, 16384, 9}, Case{18432, 5376, 384}}) {
        auto packed = make_q4_g32_fp16_weight(c.n, c.k, 61U + static_cast<std::uint32_t>(c.t));
        const std::int32_t groups = c.k / 32;
        const std::size_t code_bytes = static_cast<std::size_t>(c.n) * groups * 16;
        const std::size_t scale_offset = (code_bytes + 255) / 256 * 256;
        const std::uint8_t* codes = packed.payload.data();
        const std::uint8_t* scales = packed.payload.data() + scale_offset;
        DeviceBuffer device_weight(packed.payload.size());
        device_weight.copy_from_host(packed.payload.data(), device_weight.bytes);
        const Weight weight = packed.device_weight(device_weight.p);

        std::vector<std::uint16_t> activation(static_cast<std::size_t>(c.k) * c.t);
        std::vector<float> host_x(activation.size());
        std::uint32_t state = 0x9e3779b9u ^ static_cast<std::uint32_t>(c.n + c.k + c.t);
        for (std::size_t i = 0; i < activation.size(); ++i) {
            state = state * 1664525u + 1013904223u;
            const float value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 2.0f;
            activation[i]     = test::f32_to_bf16(value);
            const std::uint32_t word = static_cast<std::uint32_t>(activation[i]) << 16;
            std::memcpy(&host_x[i], &word, 4);
        }
        DeviceBuffer act(activation.size() * 2);
        act.copy_from_host(activation.data(), act.bytes);
        DeviceBuffer out(static_cast<std::size_t>(c.n) * c.t * 2);
        Tensor input(act.p, DType::BF16, {c.k, c.t});
        Tensor destination(out.p, DType::BF16, {c.n, c.t});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, c.n, c.k, ops::LinearPolicy::AllowA8, c.t, c.t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        ops::linear(input, weight, destination, ops::LinearPolicy::AllowA8, workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "baked q4 linear");
        if (workspace.peak_used() > capacity || workspace.used() != 0) {
            std::cerr << "Q4_G32 baked: workspace query/scope mismatch\n";
            ++failures;
        }
        std::vector<std::uint16_t> result(static_cast<std::size_t>(c.n) * c.t);
        out.copy_to_host(result.data(), result.size() * 2);
        Weight prepared = weight;
        DeviceBuffer row_storage;
        ops::linear_prepare_a8_rows(prepared, row_storage, nullptr);
        if (prepared.row_scales == nullptr) {
            std::cerr << "Q4_G32 baked A8: trunk shape got no stored row multipliers\n";
            ++failures;
        }
        ops::linear(input, prepared, destination, ops::LinearPolicy::AllowA8, workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "baked q4 linear, stored rows");
        std::vector<std::uint16_t> via_stored(result.size());
        out.copy_to_host(via_stored.data(), via_stored.size() * 2);
        if (via_stored != result) {
            std::cerr << "Q4_G32 baked A8: stored row multipliers change the output\n";
            ++failures;
        }
        DeviceArena bake_workspace(ops::detail::q4_g32_baked_workspace_bytes(c.n, c.k, c.t));
        ops::detail::q4_g32_w4a8_baked(ops::detail::q4_linear_operands(input, weight),
                                       static_cast<__nv_bfloat16*>(out.p), bake_workspace, nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "baked q4 linear, bake route");
        std::vector<std::uint16_t> via_bake(result.size());
        out.copy_to_host(via_bake.data(), via_bake.size() * 2);
        // Same E4M3 bytes and multipliers. Only the split-K summation order may differ between the
        // two routes' tile choices: one BF16 step, or a little more on results that cancel to near zero.
        std::size_t differing = 0, beyond_one_step = 0;
        for (std::size_t i = 0; i < result.size(); ++i) {
            const int step = std::abs(static_cast<int>(result[i]) - static_cast<int>(via_bake[i]));
            const std::uint32_t a_word = static_cast<std::uint32_t>(result[i]) << 16;
            const std::uint32_t b_word = static_cast<std::uint32_t>(via_bake[i]) << 16;
            float a, b;
            std::memcpy(&a, &a_word, 4);
            std::memcpy(&b, &b_word, 4);
            differing += step != 0;
            beyond_one_step += step > 1 && std::fabs(a - b) > 1e-3f;
        }
        std::cout << "Q4_G32 fused vs bake [" << c.n << "," << c.k << "] T=" << c.t << ": " << differing
                  << " outputs differ by the summation order\n";
        if (beyond_one_step != 0) {
            std::cerr << "Q4_G32 baked A8: " << beyond_one_step << " outputs differ between the fused and the bake route\n";
            ++failures;
        }

        double err_baked = 0, err_exact = 0, norm_baked = 0, weight_err = 0, weight_norm = 0;
        std::vector<double> baked(c.k), exact(c.k);
        for (std::int32_t sample = 0; sample < 48; ++sample) {
            const std::int32_t row = static_cast<std::int32_t>((static_cast<std::int64_t>(c.n - 1) * sample) / 47);
            const auto* row_scales = reinterpret_cast<const std::uint16_t*>(scales) + static_cast<std::size_t>(row) * groups;
            float largest = 0.0f;
            for (std::int32_t g = 0; g < groups; ++g) { largest = std::max(largest, std::fabs(half_to_float(row_scales[g]))); }
            const float m          = 8.0f * largest;
            const float multiplier = m > 0.0f ? bf16_round_up(m / 448.0f) : 1.0f;
            const float inverse    = 1.0f / multiplier;
            for (std::int32_t g = 0; g < groups; ++g) {
                const float scale = half_to_float(row_scales[g]);
                const float mult  = scale * inverse;
                for (int lane = 0; lane < 32; ++lane) {
                    const std::uint8_t byte = codes[(static_cast<std::size_t>(row) * groups + g) * 16 + lane / 2];
                    const int code = (((lane & 1) ? (byte >> 4) : (byte & 0xf)) ^ 8) - 8;
                    const float product = static_cast<float>(code) * mult;
                    const std::size_t column = static_cast<std::size_t>(g) * 32 + lane;
                    baked[column] = e4m3_round(product) * static_cast<double>(multiplier);
                    exact[column] = static_cast<double>(code) * static_cast<double>(scale);
                    weight_err += (baked[column] - exact[column]) * (baked[column] - exact[column]);
                    weight_norm += exact[column] * exact[column];
                }
            }
            for (std::int32_t token = 0; token < c.t; ++token) {
                double sum_baked = 0, sum_exact = 0;
                const float* xt = host_x.data() + static_cast<std::size_t>(token) * c.k;
                for (std::int32_t column = 0; column < c.k; ++column) {
                    sum_baked += baked[column] * xt[column];
                    sum_exact += exact[column] * xt[column];
                }
                const std::uint32_t word = static_cast<std::uint32_t>(result[static_cast<std::size_t>(token) * c.n + row]) << 16;
                float actual;
                std::memcpy(&actual, &word, 4);
                err_baked += (actual - sum_baked) * (actual - sum_baked);
                err_exact += (actual - sum_exact) * (actual - sum_exact);
                norm_baked += sum_baked * sum_baked;
            }
        }
        const double rel_baked = std::sqrt(err_baked / norm_baked);
        const double rel_exact = std::sqrt(err_exact / norm_baked);
        const double rel_weight = std::sqrt(weight_err / weight_norm);
        std::cout << "Q4_G32 baked A8 [" << c.n << "," << c.k << "] T=" << c.t << " rel_l2 vs baked "
                  << rel_baked << ", vs exact " << rel_exact << ", weight rounding " << rel_weight << '\n';
        if (!(rel_baked <= 0.04)) {
            std::cerr << "Q4_G32 baked A8: relative L2 above the A8 allowance\n";
            ++failures;
        }
    }
    return failures;
}

// The down projection takes the codes GeGLU wrote (bit-equal to the quantize kernel, test_geglu).
// With those codes it must give the bytes of the ordinary A8 call.
int q4_g32_ready_codes() {
    using namespace infernix;
    constexpr std::int32_t n = 5376, k = 21504;
    int failures = 0;
    auto packed  = make_q4_g32_fp16_weight(n, k, 29U);
    DeviceBuffer device_weight(packed.payload.size());
    device_weight.copy_from_host(packed.payload.data(), device_weight.bytes);
    const Weight weight = packed.device_weight(device_weight.p);
    for (const std::int32_t t : {9, 17, 128, 333, 400}) {
        if (!ops::linear_takes_a8_ready(weight, t, ops::LinearPolicy::AllowA8)) {
            std::cerr << "Q4_G32 ready: down projection does not take ready codes at T=" << t << '\n';
            ++failures;
            continue;
        }
        std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * t);
        std::uint32_t state = 0x2545f491u + static_cast<std::uint32_t>(t);
        for (auto& value : activation) {
            state = state * 1664525u + 1013904223u;
            value = test::f32_to_bf16((static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 3.0f);
        }
        DeviceBuffer act(activation.size() * 2);
        act.copy_from_host(activation.data(), act.bytes);
        DeviceBuffer codes(static_cast<std::size_t>(k) * t);
        DeviceBuffer scales(static_cast<std::size_t>(t) * sizeof(float));
        DeviceBuffer plain(static_cast<std::size_t>(n) * t * 2);
        DeviceBuffer ready(static_cast<std::size_t>(n) * t * 2);
        Tensor input(act.p, DType::BF16, {k, t});
        Tensor plain_out(plain.p, DType::BF16, {n, t});
        Tensor ready_out(ready.p, DType::BF16, {n, t});
        Weight shape;
        shape.k = k;
        ops::detail::launch_fp8_a8_quantize(
            input, shape,
            ops::detail::Fp8A8Workspace{static_cast<std::uint8_t*>(codes.p), static_cast<float*>(scales.p), nullptr},
            nullptr);
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, n, k, ops::LinearPolicy::AllowA8, t, t);
        DeviceArena workspace(capacity);
        ops::linear(input, weight, plain_out, ops::LinearPolicy::AllowA8, workspace, nullptr);
        ops::linear_a8_ready(input, weight, ready_out, ops::LinearPolicy::AllowA8,
                             static_cast<std::uint8_t*>(codes.p), static_cast<float*>(scales.p), workspace,
                             nullptr);
        test::cuda_check(cudaDeviceSynchronize(), "q4 ready codes");
        if (workspace.peak_used() > capacity || workspace.used() != 0) {
            std::cerr << "Q4_G32 ready: workspace query/scope mismatch\n";
            ++failures;
        }
        std::vector<std::uint16_t> a(static_cast<std::size_t>(n) * t), b(a.size());
        plain.copy_to_host(a.data(), a.size() * 2);
        ready.copy_to_host(b.data(), b.size() * 2);
        if (a != b) {
            std::cerr << "Q4_G32 ready: output differs from the ordinary A8 call at T=" << t << '\n';
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_q4_g32_a16) {
    try {
        if (!cuda_available()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        const int failures = q4_g32_a16() + q4_g32_baked_a8() + q4_g32_ready_codes();
        if (failures != 0) {
            std::cerr << "Q4_G32 test: " << failures << " failures\n";
            return 1;
        }
        std::cout << "OK Q4_G32 A16\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Q4_G32 test: " << error.what() << '\n';
        return 1;
    }
}
