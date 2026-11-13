// The A8 route of Gemma's 5376x21504 FP8 down projection with its activation codes and per-token scales
// already written by the producer (the GeGLU's FP8 output): the GEMM of detail::fp8_dispatch_ready
// against the FP64 oracle over independently decoded codes and scales. The activation operands are
// arbitrary valid E4M3 codes and positive scales, so the criterion is the reduction's rounding only.
#include "ops/op_family.h"
#include "infernix/ops/linear.h"
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kCriterion{kBf16UnitRoundoff, kBf16UnitRoundoff,
                                        2.0 * kBf16UnitRoundoff};

int run_ready() {
    constexpr std::int32_t n          = 5376;
    constexpr std::int32_t k          = 21504;
    constexpr std::int32_t max_tokens = 513;
    const auto host                   = make_fp8_weight(n, k, 931U);
    DeviceBuffer device_payload       = to_device(host.payload);
    const Weight weight               = host.device_weight(device_payload.p);

    // Every 97th row, and the last.
    std::vector<std::int32_t> rows;
    for (std::int32_t row = 0; row < n; row += 97) rows.push_back(row);
    rows.push_back(n - 1);
    const std::vector<float> dense = quantized_weight::materialize_rows_fp32(host, rows);

    std::mt19937 rng(932U);
    std::vector<std::uint8_t> codes(static_cast<std::size_t>(max_tokens) * k);
    for (auto& code : codes) {
        code = static_cast<std::uint8_t>(rng() & 0xFFU);
        if ((code & 0x7FU) == 0x7FU) code &= 0xFEU; // E4M3FN has no infinity; 0x7F is NaN
    }
    std::vector<float> scales(max_tokens);
    for (auto& scale : scales) scale = (1.0F + static_cast<float>(rng() % 1000U) / 1000.0F) / 1024.0F;
    std::vector<float> activation(codes.size());
    for (std::int32_t token = 0; token < max_tokens; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            const std::size_t index = static_cast<std::size_t>(token) * k + column;
            activation[index]       = static_cast<float>(
                quantized_weight::detail::decode_e4m3fn(codes[index]) *
                static_cast<double>(scales[static_cast<std::size_t>(token)]));
        }
    }
    const auto row_count = static_cast<std::int32_t>(rows.size());
    std::vector<double> reference(static_cast<std::size_t>(row_count) * max_tokens);
    cpu_linear_gemm_fp64(dense.data(), activation.data(), reference.data(), row_count, k, max_tokens);

    DeviceBuffer device_codes  = to_device(codes);
    DeviceBuffer device_scales = to_device(scales);
    DeviceBuffer device_x(static_cast<std::size_t>(max_tokens) * k * 2);
    DeviceBuffer device_out(static_cast<std::size_t>(max_tokens) * n * 2);
    int failures = 0;
    // 33: the 32x64 tile; 65: the 64-token tile split over K; 129: the 128-row tile; 513: the bulk tile.
    for (const std::int32_t tokens : {33, 65, 129, 513}) {
        Tensor x(device_x.p, DType::BF16, {k, tokens});
        Tensor out(device_out.p, DType::BF16, {n, tokens});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16, n, k, ops::LinearPolicy::AllowA8, tokens, tokens);
        DeviceArena workspace(capacity);
        cuda_check(cudaMemset(device_out.p, 0, device_out.bytes), "clear fp8 ready output");
        ops::detail::fp8_dispatch_ready(x, weight, out, ops::LinearPolicy::AllowA8,
                                        static_cast<std::uint8_t*>(device_codes.p),
                                        static_cast<float*>(device_scales.p), &workspace, nullptr);
        cuda_synchronize();

        const std::vector<double> all =
            from_device_bf16(device_out, static_cast<std::size_t>(tokens) * n);
        std::vector<double> actual;
        std::vector<double> expected;
        for (std::int32_t token = 0; token < tokens; ++token) {
            for (std::int32_t selected = 0; selected < row_count; ++selected) {
                actual.push_back(all[static_cast<std::size_t>(token) * n + rows[selected]]);
                expected.push_back(reference[static_cast<std::size_t>(token) * row_count + selected]);
            }
        }
        failures += verify_reduction("fp8_dispatch_ready N=5376 K=21504 T=" + std::to_string(tokens),
                                     actual, expected, kCriterion);
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_gemma4_fp8_ready) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_ready();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 FP8 ready-codes down projection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 FP8 ready-codes down projection: " << error.what() << '\n';
        return 1;
    }
}
