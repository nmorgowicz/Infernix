// q4_g128_fp16 / q4_g256_fp16 A16 linear: numerics against the FP64 dequantized-weight oracle on the
// Gemma 4 31B trunk shapes and head, and bit-exact equality with the q4_g32 kernels on the same
// weights expressed as a g32 tensor (converter emulation: scales repeated G / 32 times). W4A8 is
// test_q4_coarse_a8.cpp.
#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/device.h"
#include "infernix/ops/linear.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix::test::linear;
using infernix::QType;

constexpr Invocation a16(std::int32_t t) { return {t}; }

struct Format {
    const char* name;
    QType qtype;
    std::int32_t group;
    WeightGenerator make;
};

constexpr std::array kFormats{
    Format{"Q4_G128", QType::Q4_G128_FP16, 128, make_q4_g128_fp16_weight},
    Format{"Q4_G256", QType::Q4_G256_FP16, 256, make_q4_g256_fp16_weight},
};

constexpr std::array kTrunkN{16384, 18432, 43008, 5376};
constexpr std::array kTrunkK{5376, 8192, 16384, 21504};

// T=1 (unrolled DRAM GEMV), 2..8 and 9..16 (tensor-core passes), 17 and 33 (wide).
constexpr std::array kTokens{a16(1),  a16(2),  a16(3),  a16(4),  a16(5),  a16(6),
                             a16(7),  a16(8),  a16(9),  a16(12), a16(16), a16(17),
                             a16(33)};
// Shapes the tensor-core and unrolled T=1 kernels reject land on the shared-tile GEMV (T=1..8 and 16)
// and the wide kernel (9, 17, 33).
constexpr std::array kFallbackTokens{a16(1), a16(2), a16(3), a16(4),  a16(5),  a16(6), a16(7),
                                     a16(8), a16(9), a16(16), a16(17), a16(33)};

int numerics(const Format& format) {
    int failures       = 0;
    std::uint32_t seed = 101U;
    for (const std::int32_t n : kTrunkN) {
        for (const std::int32_t k : kTrunkK) {
            failures += run_shape(format.name, ActivationCompute::A16, format.make,
                                  {n, k, seed++, Comparison::Sampled, true, kTokens});
        }
    }
    // The head: one 8-row-per-warp T=1 pass, and 64-row tensor-core tiles.
    constexpr std::array kHeadCalls{a16(1), a16(4)};
    failures += run_shape(format.name, ActivationCompute::A16, format.make,
                          {262144, 5376, seed++, Comparison::Sampled, true, kHeadCalls});
    // Rows or K the fast kernels do not tile: every row and token is compared.
    failures += run_shape(format.name, ActivationCompute::A16, format.make,
                          {136, 5376, seed++, Comparison::Full, true, kFallbackTokens});
    failures += run_shape(format.name, ActivationCompute::A16, format.make,
                          {256, 1792, seed++, Comparison::Full, true, kFallbackTokens});
    failures += run_shape(format.name, ActivationCompute::A16, format.make,
                          {128, 256, seed++, Comparison::Full, true, kTokens});
    if (format.group == 128) {
        // K a multiple of 128 but not 256: no tensor-core route.
        failures += run_shape(format.name, ActivationCompute::A16, format.make,
                              {72, 640, seed++, Comparison::Full, true, kFallbackTokens});
        failures += run_shape(format.name, ActivationCompute::A16, format.make,
                              {48, 128, seed++, Comparison::Full, true, kFallbackTokens});
    }
    return failures;
}

std::vector<std::uint16_t> make_bf16_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::uint32_t state = 0x9e3779b9u ^ seed;
    for (auto& word : bits) {
        state             = state * 1664525u + 1013904223u;
        const float value = (static_cast<float>((state >> 8) & 0xffffu) / 32768.0f - 1.0f) * 2.0f;
        word              = infernix::test::f32_to_bf16(value);
    }
    return bits;
}

// Same dequantized weights, two layouts: the coarse kernels must reproduce the g32 kernels' bits.
int equals_emulated_g32(const Format& format, std::int32_t n, std::int32_t k,
                        std::span<const Invocation> calls, std::uint32_t seed) {
    using namespace infernix;
    const auto coarse   = format.make(n, k, seed);
    const auto emulated = test::quantized_weight::emulate_q4_g32_from_coarse(coarse);
    DeviceBuffer coarse_device(coarse.payload.size());
    coarse_device.copy_from_host(coarse.payload.data(), coarse_device.bytes);
    DeviceBuffer emulated_device(emulated.payload.size());
    emulated_device.copy_from_host(emulated.payload.data(), emulated_device.bytes);
    const Weight coarse_weight   = coarse.device_weight(coarse_device.p);
    const Weight emulated_weight = emulated.device_weight(emulated_device.p);

    const std::int32_t max_t =
        std::max_element(calls.begin(), calls.end(), [](const Invocation& a, const Invocation& b) {
            return a.t < b.t;
        })->t;
    const auto activation = make_bf16_activation(k, max_t, seed + 7U);
    DeviceBuffer device_activation(activation.size() * 2);
    device_activation.copy_from_host(activation.data(), device_activation.bytes);

    int failures = 0;
    for (const Invocation& call : calls) {
        const std::string label = std::string(format.name) + " == emulated g32 [" + std::to_string(n) +
                                  "," + std::to_string(k) + "] T=" + std::to_string(call.t);
        const std::size_t words = static_cast<std::size_t>(n) * call.t;
        DeviceBuffer coarse_out(words * 2), emulated_out(words * 2);
        std::vector<std::uint8_t> poison(words * 2, 0xff);
        coarse_out.copy_from_host(poison.data(), poison.size());
        emulated_out.copy_from_host(poison.data(), poison.size());
        Tensor input(device_activation.p, DType::BF16, {k, call.t});
        Tensor coarse_dst(coarse_out.p, DType::BF16, {n, call.t});
        Tensor emulated_dst(emulated_out.p, DType::BF16, {n, call.t});
        try {
            ops::linear(input, coarse_weight, coarse_dst, nullptr);
            ops::linear(input, emulated_weight, emulated_dst, nullptr);
            infernix::test::cuda_check(cudaDeviceSynchronize(), "synchronize coarse equivalence");
        } catch (const std::exception& error) {
            std::cerr << label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }
        std::vector<std::uint16_t> got(words), want(words);
        coarse_out.copy_to_host(got.data(), got.size() * 2);
        emulated_out.copy_to_host(want.data(), want.size() * 2);
        if (got != want) {
            std::size_t first = 0;
            while (got[first] == want[first]) { ++first; }
            std::cerr << label << ": outputs differ from the g32 kernel on the emulated tensor, first at row "
                      << first % n << " token " << first / n << " (0x" << std::hex << got[first]
                      << " vs 0x" << want[first] << std::dec << ")\n";
            ++failures;
        }
        if (std::count(got.begin(), got.end(), std::uint16_t{0xffff}) != 0 ||
            std::count(got.begin(), got.end(), std::uint16_t{0}) == static_cast<std::ptrdiff_t>(got.size())) {
            std::cerr << label << ": output is poisoned or all zero\n";
            ++failures;
        }
    }
    return failures;
}

int equivalence() {
    int failures       = 0;
    std::uint32_t seed = 900U;
    constexpr std::array kBitTokens{a16(1), a16(2), a16(5), a16(7), a16(8), a16(9),
                                    a16(16), a16(17), a16(40)};
    for (const Format& format : kFormats) {
        for (const std::int32_t n : kTrunkN) {
            for (const std::int32_t k : kTrunkK) {
                failures += equals_emulated_g32(format, n, k, kBitTokens, seed++);
            }
        }
        failures += equals_emulated_g32(format, 262144, 5376, std::array{a16(1), a16(4)}, seed++);
        failures += equals_emulated_g32(format, 136, 5376, kFallbackTokens, seed++);
        failures += equals_emulated_g32(format, 256, 1792, kFallbackTokens, seed++);
        failures += equals_emulated_g32(format, 5376 + 16, 5376, kBitTokens, seed++);
        if (format.group == 128) {
            failures += equals_emulated_g32(format, 72, 640, kFallbackTokens, seed++);
        }
    }
    return failures;
}
} // namespace
INFERNIX_OP_TEST(linear_q4_coarse_a16) {
    try {
        if (!cuda_available()) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        int failures = 0;
        for (const Format& format : kFormats) { failures += numerics(format); }
        failures += equivalence();
        if (failures != 0) {
            std::cerr << "Q4 coarse test: " << failures << " failures\n";
            return 1;
        }
        std::cout << "OK Q4 coarse A16\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Q4 coarse test: " << error.what() << '\n';
        return 1;
    }
}
