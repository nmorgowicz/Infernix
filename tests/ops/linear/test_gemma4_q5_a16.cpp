#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

// Exactly the matrices Parameters::geometry prepares: sliding QKV = 8192 + 4096 + 4096,
// global QK = 16384 + 2048 (V is K), gate/up = 2 * 21504, and the three output projections.
constexpr std::array kShapes{std::pair{16384, 5376}, std::pair{18432, 5376},
                             std::pair{43008, 5376}, std::pair{5376, 8192},
                             std::pair{5376, 16384}, std::pair{5376, 21504}};

constexpr Invocation graph(int tokens) {
    return {tokens, CallForm::Policy, ops::LinearPolicy::A16Only, true};
}

// Both sides of every SIMT/sliced-K/MMA/ring boundary, including the engine's prefill
// chunk sizes and the decode/MTP verification widths. Captures replay twice, changing X.
constexpr std::array kCalls{Invocation{1}, Invocation{2}, Invocation{3}, Invocation{4},
                            Invocation{5}, Invocation{8}, Invocation{9}, Invocation{16},
                            Invocation{17}, Invocation{32}, Invocation{33}, Invocation{64},
                            Invocation{65}, Invocation{127}, Invocation{128}, Invocation{129},
                            Invocation{2048}, graph(1), graph(2), graph(8), graph(128), graph(2048),
                            Invocation{1, CallForm::A16Convenience}};
constexpr std::array kStressCalls{Invocation{1}, Invocation{2}, Invocation{8},
                                  Invocation{128}, Invocation{2048}};

int run_gemma4_q5() {
    int failures = 0;
    std::uint32_t seed = 5376;
    for (auto [n, k] : kShapes) {
        // The interval query is the model planner's route; these A16 launches need no scratch.
        if (ops::linear_workspace_capacity_bytes(QType::Q5_G64_FP16, n, k,
                                                 ops::LinearPolicy::A16Only, 1, 2048) != 0) {
            std::cerr << "Gemma4 native Q5: unexpected workspace for [" << n << ',' << k << "]\n";
            ++failures;
        }
        // The existing fixture independently decodes the stored five-bit planes and exact
        // binary16 scales to FP32, then sums BF16 X * decoded W in FP64. Sample 32 evenly
        // spaced rows/columns (including both ends), while checking every output for poison,
        // non-finites and guards; do not compare against another GPU kernel.
        failures += run_shape("Gemma4 Q5 A16", ActivationCompute::A16,
                              make_q5_g64_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, kCalls});
    }
    failures += run_shape("Gemma4 Q5 A16 cancellation", ActivationCompute::A16,
                          make_q5_g64_fp16_weight,
                          {5376, 21504, seed++, Comparison::Sampled, true, kStressCalls,
                           ActivationPattern::Cancellation});
    failures += run_shape("Gemma4 Q5 A16 final group", ActivationCompute::A16,
                          make_q5_g64_fp16_weight,
                          {43008, 5376, seed++, Comparison::Sampled, true, kStressCalls,
                           ActivationPattern::KTail});

    // A nearby unregistered shape must not silently acquire a runtime-shape fallback.
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::Q5_G64_FP16, 5376, 5376,
                                                   ops::LinearPolicy::A16Only, 1, 2048);
        std::cerr << "Gemma4 native Q5: admitted an unregistered shape\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_gemma4_q5_a16) {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_gemma4_q5();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 native Q5 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 native Q5 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
