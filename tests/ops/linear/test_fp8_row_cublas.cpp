// ops::fp8_row_cublas against the FP64 oracle at the A16 linear criterion.
#include "ops/op_family.h"
#include "infernix/ops/fp8_row_cublas.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;
constexpr ReductionCriterion kCriterion{kBf16UnitRoundoff, kBf16UnitRoundoff,
                                        2.0 * kBf16UnitRoundoff};

int run_case(std::int32_t n, std::int32_t k, std::int32_t tokens, std::uint32_t seed) {
    const std::string label = "fp8_row_cublas N=" + std::to_string(n) + " K=" + std::to_string(k) +
                              " T=" + std::to_string(tokens);
    const auto host = linear::make_fp8_weight(n, k, seed);
    std::vector<float> x(static_cast<std::size_t>(k) * tokens);
    fill_uniform(x, seed + 1U, -2.0F, 2.0F);
    round_to_bf16(x);

    DeviceBuffer device_payload = to_device(host.payload);
    DeviceBuffer device_x       = to_device_bf16(x);
    DeviceBuffer device_out(static_cast<std::size_t>(n) * tokens * 2);
    const Weight w = host.device_weight(device_payload.p);
    Tensor x_t(device_x.p, DType::BF16, {k, tokens});
    Tensor out_t(device_out.p, DType::BF16, {n, tokens});
    WorkspaceArena workspace(ops::fp8_row_cublas_workspace_bytes(n, k, tokens));
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create fp8_row_cublas stream");
    {
        ops::Fp8RowCublasContext gemm(stream);
        ops::fp8_row_cublas(x_t, w, out_t, gemm, workspace, stream);
        cuda_check(cudaStreamSynchronize(stream), "fp8_row_cublas synchronize");
    }
    cuda_check(cudaStreamDestroy(stream), "destroy fp8_row_cublas stream");

    std::vector<std::int32_t> rows(static_cast<std::size_t>(n));
    std::iota(rows.begin(), rows.end(), 0);
    const std::vector<float> dense = quantized_weight::materialize_rows_fp32(host, rows);
    std::vector<double> reference(static_cast<std::size_t>(n) * tokens);
    linear::cpu_linear_gemm_fp64(dense.data(), x.data(), reference.data(), n, k, tokens);
    const std::vector<double> actual = from_device_bf16(device_out, reference.size());
    return verify_reduction(label, actual, reference, kCriterion);
}

} // namespace

INFERNIX_OP_TEST(linear_fp8_row_cublas) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        failures += run_case(2048, 5376, 96, 701U);
        failures += run_case(384, 21504, 33, 702U);
        failures += run_case(258, 512, 7, 703U);
        failures += run_case(5376, 8192, 1, 704U);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " fp8_row_cublas\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fp8_row_cublas: " << error.what() << '\n';
        return 1;
    }
}
