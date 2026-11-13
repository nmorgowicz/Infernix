// q4_g128_fp16 / q4_g256_fp16 W4A8 linear (AllowA8). For every route (fused, baked, exact, and the
// non-TMA exact fallback, ready codes, stored row multipliers):
//  * outputs equal bit for bit those of the g32 kernels on the same weights expressed as a g32 tensor
//    (converter emulation: scales repeated G / 32 times);
//  * the fused route agrees with the bake route as it does for g32 (summation order only);
//  * accuracy against the FP64 oracle within the g32 test's bounds.
// Run once plain, and once with INFERNIX_TMP_Q4_EXACT=1 (the exact route on the trunk shapes, deterministic).
#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/linear.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/linear/q4/q4_g32_route.h"
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
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix::test::linear;
using infernix::QType;
namespace qw = infernix::test::quantized_weight;

struct Format {
    const char* name;
    QType qtype;
    std::int32_t group;
    std::int32_t shift;
    WeightGenerator make;
};

constexpr std::array kFormats{
    Format{"Q4_G128", QType::Q4_G128_FP16, 128, 2, make_q4_g128_fp16_weight},
    Format{"Q4_G256", QType::Q4_G256_FP16, 256, 3, make_q4_g256_fp16_weight},
};

struct Shape {
    std::int32_t n, k;
};
constexpr std::array<Shape, 6> kTrunk{
    {{16384, 5376}, {18432, 5376}, {43008, 5376}, {5376, 8192}, {5376, 16384}, {5376, 21504}}};
constexpr std::array<std::int32_t, 6> kTokens{9, 33, 128, 384, 385, 1024};

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
    const int exponent     = (code >> 3) & 0xf;
    const int mantissa     = code & 7;
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

struct Activation {
    std::vector<std::uint16_t> bits;
    std::vector<float> values;
};

Activation make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    Activation a;
    a.bits.resize(static_cast<std::size_t>(k) * t);
    a.values.resize(a.bits.size());
    std::uint32_t state = 0x9e3779b9u ^ seed;
    for (std::size_t i = 0; i < a.bits.size(); ++i) {
        state             = state * 1664525u + 1013904223u;
        const float value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 2.0f;
        a.bits[i]         = infernix::test::f32_to_bf16(value);
        const std::uint32_t word = static_cast<std::uint32_t>(a.bits[i]) << 16;
        std::memcpy(&a.values[i], &word, 4);
    }
    return a;
}

// A weight on the device with its payload.
struct DeviceWeight {
    infernix::DeviceBuffer buffer;
    infernix::Weight weight;
    DeviceWeight(const qw::PackedWeight& packed) : buffer(packed.payload.size()) {
        buffer.copy_from_host(packed.payload.data(), buffer.bytes);
        weight = packed.device_weight(buffer.p);
    }
};

// One AllowA8 call; the workspace must be exactly scoped and within the queried capacity.
std::vector<std::uint16_t> run_a8(const infernix::Weight& weight, const infernix::DeviceBuffer& activation,
                                  std::int32_t t, int& failures, const char* label) {
    using namespace infernix;
    const std::int32_t n = weight.n, k = weight.k;
    DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
    std::vector<std::uint8_t> poison(out.bytes, 0xff);
    out.copy_from_host(poison.data(), poison.size());
    Tensor input(activation.p, DType::BF16, {k, t});
    Tensor destination(out.p, DType::BF16, {n, t});
    const std::size_t capacity =
        ops::linear_workspace_capacity_bytes(weight.qtype, n, k, ops::LinearPolicy::AllowA8, t, t);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::linear(input, weight, destination, ops::LinearPolicy::AllowA8, workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "coarse A8 linear");
    if (workspace.peak_used() > capacity || workspace.used() != 0) {
        std::cerr << label << ": workspace query/scope mismatch\n";
        ++failures;
    }
    std::vector<std::uint16_t> result(static_cast<std::size_t>(n) * t);
    out.copy_to_host(result.data(), result.size() * 2);
    return result;
}

bool poisoned(const std::vector<std::uint16_t>& values) {
    return std::count(values.begin(), values.end(), std::uint16_t{0xffff}) != 0;
}

// Reference rows of the baked weights: e4m3_rn(code * scale / row) * row, as the bake kernel does.
struct BakedRows {
    std::vector<std::int32_t> rows;
    std::vector<std::vector<double>> baked;
    double weight_err = 0, weight_norm = 0;
};

BakedRows bake_reference(const qw::PackedWeight& packed, std::int32_t shift) {
    const std::int32_t n = packed.weight.n, k = packed.weight.k;
    const std::int32_t groups = k / 32, scale_groups = groups >> shift;
    BakedRows result;
    for (std::int32_t sample = 0; sample < 16; ++sample) {
        const std::int32_t row = static_cast<std::int32_t>((static_cast<std::int64_t>(n - 1) * sample) / 15);
        result.rows.push_back(row);
        std::vector<double> baked(k);
        float largest = 0.0f;
        std::vector<float> scale(scale_groups);
        for (std::int32_t g = 0; g < scale_groups; ++g) {
            const std::size_t at = packed.scale_plane_offset + (static_cast<std::size_t>(row) * scale_groups + g) * 2;
            scale[g] = half_to_float(static_cast<std::uint16_t>(packed.payload[at] | (packed.payload[at + 1] << 8)));
            largest  = std::max(largest, std::fabs(scale[g]));
        }
        const float m          = 8.0f * largest;
        const float multiplier = m > 0.0f ? bf16_round_up(m / 448.0f) : 1.0f;
        const float inverse    = 1.0f / multiplier;
        for (std::int32_t g = 0; g < groups; ++g) {
            const float mult = scale[g >> shift] * inverse;
            for (int lane = 0; lane < 32; ++lane) {
                const std::uint8_t byte = packed.payload[(static_cast<std::size_t>(row) * groups + g) * 16 + lane / 2];
                const int code          = (((lane & 1) ? (byte >> 4) : (byte & 0xf)) ^ 8) - 8;
                const std::size_t column = static_cast<std::size_t>(g) * 32 + lane;
                baked[column]            = e4m3_round(static_cast<float>(code) * mult) * static_cast<double>(multiplier);
                const double exact       = static_cast<double>(code) * static_cast<double>(scale[g >> shift]);
                result.weight_err += (baked[column] - exact) * (baked[column] - exact);
                result.weight_norm += exact * exact;
            }
        }
        result.baked.push_back(std::move(baked));
    }
    return result;
}

double relative_l2(const BakedRows& reference, const Activation& x, const std::vector<std::uint16_t>& result,
                   std::int32_t n, std::int32_t k, std::int32_t t) {
    double error = 0, norm = 0;
    for (std::size_t sample = 0; sample < reference.rows.size(); ++sample) {
        for (std::int32_t token = 0; token < t; ++token) {
            double sum          = 0;
            const float* xt     = x.values.data() + static_cast<std::size_t>(token) * k;
            const auto& weights = reference.baked[sample];
            for (std::int32_t column = 0; column < k; ++column) { sum += weights[column] * xt[column]; }
            const std::uint32_t word =
                static_cast<std::uint32_t>(result[static_cast<std::size_t>(token) * n + reference.rows[sample]]) << 16;
            float actual;
            std::memcpy(&actual, &word, 4);
            error += (actual - sum) * (actual - sum);
            norm += sum * sum;
        }
    }
    return std::sqrt(error / norm);
}

// The fused and the bake route use the same bytes; only the split-K summation order may differ.
std::size_t beyond_one_step(const std::vector<std::uint16_t>& a, const std::vector<std::uint16_t>& b) {
    std::size_t beyond = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const int step             = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        const std::uint32_t a_word = static_cast<std::uint32_t>(a[i]) << 16;
        const std::uint32_t b_word = static_cast<std::uint32_t>(b[i]) << 16;
        float fa, fb;
        std::memcpy(&fa, &a_word, 4);
        std::memcpy(&fb, &b_word, 4);
        beyond += step > 1 && std::fabs(fa - fb) > 1e-3f;
    }
    return beyond;
}

std::vector<std::uint16_t> run_bake(const infernix::Weight& weight, const infernix::DeviceBuffer& activation,
                                    std::int32_t t) {
    using namespace infernix;
    DeviceBuffer out(static_cast<std::size_t>(weight.n) * t * 2);
    Tensor input(activation.p, DType::BF16, {weight.k, t});
    DeviceArena workspace(ops::detail::q4_g32_baked_workspace_bytes(weight.n, weight.k, t));
    ops::detail::q4_g32_w4a8_baked(ops::detail::q4_linear_operands(input, weight),
                                   static_cast<__nv_bfloat16*>(out.p), workspace, nullptr);
    test::cuda_check(cudaDeviceSynchronize(), "coarse bake route");
    std::vector<std::uint16_t> result(static_cast<std::size_t>(weight.n) * t);
    out.copy_to_host(result.data(), result.size() * 2);
    return result;
}

// Trunk shapes: A8 outputs of the native coarse weight against its g32 emulation, route by route.
int trunk_a8(const Format& format) {
    using namespace infernix;
    const bool exact_mode = ops::detail::q4_exact_prefill();
    int failures          = 0;
    std::uint32_t seed    = 7000U + static_cast<std::uint32_t>(format.group);
    for (const Shape shape : kTrunk) {
        const auto coarse   = format.make(shape.n, shape.k, seed++);
        const auto emulated = qw::emulate_q4_g32_from_coarse(coarse);
        const DeviceWeight coarse_device(coarse), emulated_device(emulated);
        // Stored row multipliers (the loader's prepare step) must equal the g32 emulation's.
        Weight coarse_prepared = coarse_device.weight, emulated_prepared = emulated_device.weight;
        DeviceBuffer coarse_rows, emulated_rows;
        ops::linear_prepare_a8_rows(coarse_prepared, coarse_rows, nullptr);
        ops::linear_prepare_a8_rows(emulated_prepared, emulated_rows, nullptr);
        if (coarse_prepared.row_scales == nullptr || emulated_prepared.row_scales == nullptr) {
            std::cerr << format.name << " [" << shape.n << "," << shape.k << "]: no stored row multipliers\n";
            ++failures;
            continue;
        }
        std::vector<std::uint16_t> a(static_cast<std::size_t>(shape.n)), b(a.size());
        coarse_rows.copy_to_host(a.data(), a.size() * 2);
        emulated_rows.copy_to_host(b.data(), b.size() * 2);
        if (a != b) {
            std::cerr << format.name << " [" << shape.n << "," << shape.k
                      << "]: row multipliers differ from the g32 emulation\n";
            ++failures;
        }
        const BakedRows reference = bake_reference(coarse, format.shift);
        for (const std::int32_t t : kTokens) {
            const std::string label = std::string(format.name) + " A8 [" + std::to_string(shape.n) + "," +
                                      std::to_string(shape.k) + "] T=" + std::to_string(t);
            const Activation x = make_activation(shape.k, t, seed + static_cast<std::uint32_t>(t));
            DeviceBuffer activation(x.bits.size() * 2);
            activation.copy_from_host(x.bits.data(), activation.bytes);

            const auto from_coarse   = run_a8(coarse_device.weight, activation, t, failures, label.c_str());
            const auto from_emulated = run_a8(emulated_device.weight, activation, t, failures, label.c_str());
            if (poisoned(from_coarse)) {
                std::cerr << label << ": output retains poison\n";
                ++failures;
            }
            if (from_coarse != from_emulated) {
                std::cerr << label << ": native coarse output differs from the g32 emulation\n";
                ++failures;
            }
            // Repeated runs and stored row multipliers give the same bytes.
            if (run_a8(coarse_device.weight, activation, t, failures, label.c_str()) != from_coarse) {
                std::cerr << label << ": repeated run differs\n";
                ++failures;
            }
            if (run_a8(coarse_prepared, activation, t, failures, label.c_str()) != from_coarse) {
                std::cerr << label << ": stored row multipliers change the output\n";
                ++failures;
            }
            if (exact_mode) { continue; }
            const auto baked_coarse   = run_bake(coarse_device.weight, activation, t);
            const auto baked_emulated = run_bake(emulated_device.weight, activation, t);
            if (baked_coarse != baked_emulated) {
                std::cerr << label << ": bake route differs from the g32 emulation\n";
                ++failures;
            }
            const std::size_t beyond = beyond_one_step(from_coarse, baked_coarse);
            if (beyond != 0) {
                std::cerr << label << ": " << beyond << " outputs differ between the fused and the bake route\n";
                ++failures;
            }
            const double rel = relative_l2(reference, x, from_coarse, shape.n, shape.k, t);
            std::cout << label << " rel_l2 vs baked-weight oracle " << rel << ", weight rounding "
                      << std::sqrt(reference.weight_err / reference.weight_norm) << '\n';
            if (!(rel <= 0.04)) {
                std::cerr << label << ": relative L2 above the A8 allowance\n";
                ++failures;
            }
        }
    }
    return failures;
}

// The down projection takes the codes GeGLU wrote; the ready call must give the ordinary A8 bytes,
// and the same bytes as the g32 emulation.
int ready_codes(const Format& format) {
    using namespace infernix;
    constexpr std::int32_t n = 5376, k = 21504;
    int failures             = 0;
    const auto coarse        = format.make(n, k, 29U);
    const auto emulated      = qw::emulate_q4_g32_from_coarse(coarse);
    const DeviceWeight coarse_device(coarse), emulated_device(emulated);
    for (const std::int32_t t : {9, 17, 128, 333, 400, 1024}) {
        const std::string label = std::string(format.name) + " ready [" + std::to_string(n) + "," +
                                  std::to_string(k) + "] T=" + std::to_string(t);
        const bool takes = ops::linear_takes_a8_ready(coarse_device.weight, t, ops::LinearPolicy::AllowA8);
        if (takes == ops::detail::q4_exact_prefill()) {
            // INFERNIX_TMP_Q4_EXACT leaves the fused and baked routes, which are the ones taking ready codes.
            std::cerr << label << ": ready-codes eligibility is " << takes << " under INFERNIX_TMP_Q4_EXACT="
                      << ops::detail::q4_exact_prefill() << '\n';
            ++failures;
            continue;
        }
        if (!takes) { continue; }
        const Activation x = make_activation(k, t, 0x2545f491u + static_cast<std::uint32_t>(t));
        DeviceBuffer activation(x.bits.size() * 2);
        activation.copy_from_host(x.bits.data(), activation.bytes);
        DeviceBuffer codes(static_cast<std::size_t>(k) * t);
        DeviceBuffer scales(static_cast<std::size_t>(t) * sizeof(float));
        Tensor input(activation.p, DType::BF16, {k, t});
        Weight shape;
        shape.k = k;
        ops::detail::launch_fp8_a8_quantize(
            input, shape,
            ops::detail::Fp8A8Workspace{static_cast<std::uint8_t*>(codes.p), static_cast<float*>(scales.p), nullptr},
            nullptr);
        const auto ready = [&](const Weight& weight) {
            DeviceBuffer out(static_cast<std::size_t>(n) * t * 2);
            Tensor destination(out.p, DType::BF16, {n, t});
            const std::size_t capacity =
                ops::linear_workspace_capacity_bytes(weight.qtype, n, k, ops::LinearPolicy::AllowA8, t, t);
            DeviceArena workspace(capacity);
            ops::linear_a8_ready(input, weight, destination, ops::LinearPolicy::AllowA8,
                                 static_cast<std::uint8_t*>(codes.p), static_cast<float*>(scales.p), workspace,
                                 nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "coarse ready codes");
            if (workspace.peak_used() > capacity || workspace.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
            std::vector<std::uint16_t> result(static_cast<std::size_t>(n) * t);
            out.copy_to_host(result.data(), result.size() * 2);
            return result;
        };
        const auto plain       = run_a8(coarse_device.weight, activation, t, failures, label.c_str());
        const auto via_ready   = ready(coarse_device.weight);
        const auto via_emulated = ready(emulated_device.weight);
        if (plain != via_ready) {
            std::cerr << label << ": ready output differs from the ordinary A8 call\n";
            ++failures;
        }
        if (via_ready != via_emulated) {
            std::cerr << label << ": ready output differs from the g32 emulation\n";
            ++failures;
        }
    }
    return failures;
}

// Exact route and its non-TMA fallback (rows % 128 != 0), against the FP64 oracle and the emulation.
int exact_route(const Format& format) {
    using namespace infernix;
    int failures = 0;
    constexpr std::array kA8{Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8, false},
                             Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA8, false}};
    // TMA shapes (rows and K multiples of 128), then the shared-memory fallback (1000, 136 and 72 rows).
    failures += run_shape(std::string(format.name) + "_A8", ActivationCompute::A8, format.make,
                          {128, 5376, 47U, Comparison::Full, true, kA8});
    failures += run_shape(std::string(format.name) + "_A8", ActivationCompute::A8, format.make,
                          {1000, 5376, 53U, Comparison::Full, true, kA8});
    failures += run_shape(std::string(format.name) + "_A8", ActivationCompute::A8, format.make,
                          {136, 5120, 59U, Comparison::Full, true, kA8});
    failures += run_shape(std::string(format.name) + "_A8", ActivationCompute::A8, format.make,
                          {72, 6144, 61U, Comparison::Full, true, kA8});
    if (ops::detail::q4_exact_prefill()) {
        // INFERNIX_TMP_Q4_EXACT: the trunk shapes take the exact route from 128 tokens.
        std::uint32_t trunk_seed = 8300U + static_cast<std::uint32_t>(format.group);
        for (const Shape shape : kTrunk) {
            failures += run_shape(std::string(format.name) + "_A8_exact", ActivationCompute::A8, format.make,
                                  {shape.n, shape.k, trunk_seed++, Comparison::Sampled, true, kA8});
        }
    }
    std::uint32_t seed = 8100U + static_cast<std::uint32_t>(format.group);
    for (const Shape shape : {Shape{128, 5376}, Shape{1000, 5376}, Shape{136, 5120}, Shape{72, 6144}, Shape{256, 8192}}) {
        const auto coarse   = format.make(shape.n, shape.k, seed++);
        const auto emulated = qw::emulate_q4_g32_from_coarse(coarse);
        const DeviceWeight coarse_device(coarse), emulated_device(emulated);
        for (const std::int32_t t : {128, 200, 1024}) {
            const std::string label = std::string(format.name) + " exact [" + std::to_string(shape.n) + "," +
                                      std::to_string(shape.k) + "] T=" + std::to_string(t);
            const Activation x = make_activation(shape.k, t, seed + static_cast<std::uint32_t>(t));
            DeviceBuffer activation(x.bits.size() * 2);
            activation.copy_from_host(x.bits.data(), activation.bytes);
            const auto got  = run_a8(coarse_device.weight, activation, t, failures, label.c_str());
            const auto want = run_a8(emulated_device.weight, activation, t, failures, label.c_str());
            if (got != want || poisoned(got)) {
                std::cerr << label << ": native coarse output differs from the g32 emulation\n";
                ++failures;
            }
            if (run_a8(coarse_device.weight, activation, t, failures, label.c_str()) != got) {
                std::cerr << label << ": repeated run differs\n";
                ++failures;
            }
        }
    }
    return failures;
}

// A coarse weight with A8 allowed still takes the A16 routes below 9 tokens.
int decode_with_a8_allowed(const Format& format) {
    using namespace infernix;
    int failures        = 0;
    const auto coarse   = format.make(5376, 8192, 77U);
    const auto emulated = qw::emulate_q4_g32_from_coarse(coarse);
    const DeviceWeight coarse_device(coarse), emulated_device(emulated);
    for (const std::int32_t t : {1, 2, 7, 8}) {
        const std::string label = std::string(format.name) + " AllowA8 decode T=" + std::to_string(t);
        const Activation x      = make_activation(8192, t, 5U + static_cast<std::uint32_t>(t));
        DeviceBuffer activation(x.bits.size() * 2);
        activation.copy_from_host(x.bits.data(), activation.bytes);
        if (run_a8(coarse_device.weight, activation, t, failures, label.c_str()) !=
            run_a8(emulated_device.weight, activation, t, failures, label.c_str())) {
            std::cerr << label << ": differs from the g32 emulation\n";
            ++failures;
        }
    }
    return failures;
}
} // namespace

INFERNIX_OP_TEST(linear_q4_coarse_a8) {
    try {
        if (!cuda_available()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        std::cout << "INFERNIX_TMP_Q4_EXACT=" << (infernix::ops::detail::q4_exact_prefill() ? 1 : 0) << '\n';
        int failures = 0;
        for (const Format& format : kFormats) {
            failures += decode_with_a8_allowed(format);
            failures += trunk_a8(format);
            failures += ready_codes(format);
            failures += exact_route(format);
        }
        if (failures != 0) {
            std::cerr << "Q4 coarse A8 test: " << failures << " failures\n";
            return 1;
        }
        std::cout << "OK Q4 coarse A8\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Q4 coarse A8 test: " << error.what() << '\n';
        return 1;
    }
}
