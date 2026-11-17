// Direct model project() qualification: native Q5 narrow/A16Only, and explicit AllowA8
// prefill through independently specified row-E4M3 weights and per-token E4M3 activations.
#include "models/gemma4/execution/projection.h"
#include "ops/linear/linear_test_common.h"
#include "ops/q5_bake_oracle.h"
#include "ops/op_tester.h"
#include "core/decode_graph.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>
#include <utility>

namespace {
using namespace infernix;
namespace g4 = infernix::models::gemma4;
namespace oracle = infernix::test::q5_bake;

constexpr std::array kTokens{1, 2, 8, 63, 64, 65, 127, 128, 129, 2048};
constexpr test::ReductionCriterion kA16{1.0 / 256, 1.0 / 256, 2.0 / 256};
// Whole-route allowance against stored Q5 W / BF16 X: the existing 0.04 FP8-activation allowance plus
// the independent row-E4M3 weight rounding of the transient bake (measured 0.037-0.041 relative L2,
// gross 0.055 of the maximum, on the patterned rows). This is a gross-error gate; model quality is
// qualified by native KL, not this unit threshold.
constexpr test::ReductionCriterion kA8{0.05, 1.0 / 256, 0.08};

int check_result(const test::GuardedDeviceBuffer& output, const std::vector<std::uint16_t>& activation,
                 const std::vector<int>& rows, const std::vector<float>& original_weight,
                 const std::vector<float>& baked_weight, int n, int k, int tokens, bool baked,
                 bool negative, const std::string& label) {
    const auto columns = oracle::sampled_tokens(tokens);
    std::vector<float> exact_x, represented_x;
    for (int column : columns) {
        auto exact = oracle::activation_column(activation, k, column, false, negative);
        exact_x.insert(exact_x.end(), exact.begin(), exact.end());
        auto represented = oracle::activation_column(activation, k, column, baked, negative);
        represented_x.insert(represented_x.end(), represented.begin(), represented.end());
    }
    const int nr = static_cast<int>(rows.size()), nt = static_cast<int>(columns.size());
    std::vector<double> original(static_cast<std::size_t>(nr) * nt), represented(original.size()), actual(original.size());
    test::linear::cpu_linear_gemm_fp64(original_weight.data(), exact_x.data(), original.data(), nr, k, nt);
    test::linear::cpu_linear_gemm_fp64(baked ? baked_weight.data() : original_weight.data(), represented_x.data(),
                                        represented.data(), nr, k, nt);
    // Scan every output, not only the oracle samples: no poison/non-finite or unwritten tail.
    int failures = output.verify_guards(label);
    std::vector<std::uint16_t> column(n);
    std::size_t selected = 0;
    bool nonfinite = false;
    for (int token = 0; token < tokens; ++token) {
        output.copy_to_host(column.data(), column.size() * 2, static_cast<std::size_t>(token) * n * 2);
        for (auto bits : column) { nonfinite = nonfinite || (bits & 0x7f80U) == 0x7f80U; }
        if (selected < columns.size() && token == columns[selected]) {
            for (int row = 0; row < nr; ++row) {
                actual[selected * nr + row] = oracle::bf16_value(column[rows[row]]);
            }
            ++selected;
        }
    }
    if (nonfinite) {
        std::cerr << label << ": output contains poison or non-finite values\n";
        ++failures;
    }
    // First isolate GPU arithmetic over independently encoded represented operands (every sampled
    // row, including the handcrafted tie/zero/extreme-scale rows). Then hold the whole lossy route
    // to the existing A8 criterion against stored Q5 W / BF16 X. The handcrafted rows (index < 9)
    // are dominated by a few 448-magnitude weights whose FP8 activation rounding noise is large by
    // construction, so only the patterned rows enter the stored-weight quality criterion.
    failures += test::verify_reduction(label + " represented FP64", actual, represented, kA16);
    if (!baked) {
        failures += test::verify_reduction(label + " stored Q5 FP64", actual, original, kA16);
        return failures;
    }
    std::vector<double> patterned_actual, patterned_original;
    for (std::size_t sample = 0; sample < columns.size(); ++sample) {
        for (int row = 0; row < nr; ++row) {
            if (rows[row] < 9) { continue; }
            patterned_actual.push_back(actual[sample * nr + row]);
            patterned_original.push_back(original[sample * nr + row]);
        }
    }
    failures += test::verify_reduction(label + " stored Q5 FP64 patterned rows", patterned_actual,
                                       patterned_original, kA8);
    return failures;
}

int run_shape(int n, int k, std::uint32_t seed) {
    auto host = oracle::make_weight(n, k, seed);
    const auto rows = oracle::sampled_rows(n);
    std::vector<float> original_weight, baked_weight;
    for (int row : rows) {
        const auto encoded = oracle::encode_row(host, row);
        original_weight.insert(original_weight.end(), encoded.original.begin(), encoded.original.end());
        baked_weight.insert(baked_weight.end(), encoded.represented.begin(), encoded.represented.end());
    }
    const auto activation = oracle::make_activation(k, kTokens.back(), seed + 1);
    DeviceBuffer device_weight(host.payload.size()), device_x(activation.size() * 2);
    device_weight.copy_from_host(host.payload.data(), device_weight.bytes);
    device_x.copy_from_host(activation.data(), device_x.bytes);
    const Weight weight = host.device_weight(device_weight.p);
    DeviceContext context;
    ops::Fp8RowCublasContext gemm(context.stream);
    int failures = 0;
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        const auto envelope = g4::projection_bytes(weight, policy, kTokens.back());
        for (int tokens : kTokens) {
            const bool baked = policy == ops::LinearPolicy::AllowA8 && tokens >= g4::kQ5DenseMinTokens;
            const auto capacity = g4::projection_bytes(weight, policy, tokens);
            const std::string label = "Gemma Q5 project [" + std::to_string(n) + ',' + std::to_string(k) +
                                      "] T=" + std::to_string(tokens) + (baked ? " baked A8" : " native A16") +
                                      (policy == ops::LinearPolicy::A16Only ? " A16Only" : " AllowA8");
            if (capacity > envelope || (!baked && capacity != 0) || (baked && capacity == 0)) {
                std::cerr << label << ": inconsistent planning envelope\n";
                ++failures;
            }
            DeviceArena workspace(std::max<std::size_t>(capacity, 256));
            test::GuardedDeviceBuffer output(static_cast<std::size_t>(n) * tokens * 2);
            Tensor x(device_x.p, DType::BF16, {k, tokens});
            Tensor out(output.data(), DType::BF16, {n, tokens});
            const auto launch = [&] { g4::project(x, weight, policy, out, gemm, workspace, context.stream); };
            output.fill(0xff);
            launch();
            CUDA_CHECK(cudaStreamSynchronize(context.stream));
            failures += check_result(output, activation, rows, original_weight, baked_weight, n, k, tokens,
                                     baked, false, label + " eager");
            // Every requested representative width also runs captured with changed X. The
            // graph must rebake/requantize current inputs, not capture host-derived values.
            if (tokens == 1 || tokens == 2 || tokens == 8 || tokens == 128 || tokens == 2048) {
                DecodeGraphDefinition definition;
                DecodeGraphExecutable graph;
                definition.capture(context.stream, launch);
                graph.instantiate(definition);
                auto negative = activation;
                for (auto& bits : negative) { bits ^= 0x8000U; }
                device_x.copy_from_host(negative.data(), device_x.bytes);
                output.fill(0xff);
                CUDA_CHECK(cudaDeviceSynchronize());
                graph.launch(context.stream);
                CUDA_CHECK(cudaStreamSynchronize(context.stream));
                failures += check_result(output, activation, rows, original_weight, baked_weight, n, k, tokens,
                                         baked, true, label + " graph -X");
                device_x.copy_from_host(activation.data(), device_x.bytes);
            }
            if (workspace.peak_used() > capacity || workspace.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch\n";
                ++failures;
            }
        }
    }
    std::vector<std::uint8_t> weight_after(host.payload.size());
    device_weight.copy_to_host(weight_after.data(), device_weight.bytes);
    if (weight_after != host.payload) {
        std::cerr << "Gemma Q5 project modified persistent weight\n";
        ++failures;
    }
    std::vector<std::uint16_t> x_after(activation.size());
    device_x.copy_to_host(x_after.data(), device_x.bytes);
    if (x_after != activation) {
        std::cerr << "Gemma Q5 project modified BF16 input\n";
        ++failures;
    }
    return failures;
}
} // namespace

int main() {
    if (infernix::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        std::uint32_t seed = 831;
        for (auto [n, k] : std::array{std::pair{16384, 5376}, std::pair{18432, 5376},
                                      std::pair{43008, 5376}, std::pair{5376, 8192},
                                      std::pair{5376, 16384}, std::pair{5376, 21504}}) {
            failures += run_shape(n, k, seed++);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma native Q5 projection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma native Q5 projection: " << error.what() << '\n';
        return 1;
    }
}
