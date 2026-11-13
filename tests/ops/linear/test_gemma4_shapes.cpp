// Gemma 4 31B FP8 (A16 and A8) and Q4 g32/g128/g256 (A16 and A8) linear shapes
// against the FP64 oracle.
#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;

constexpr std::array kTrunk{std::pair{16384, 5376}, std::pair{18432, 5376}, std::pair{43008, 5376},
                            std::pair{5376, 8192},  std::pair{5376, 16384}, std::pair{5376, 21504}};

std::vector<Invocation> a16_calls() {
    std::vector<Invocation> calls;
    for (int t : {1,  2,  3,  4,  5,  8,  9,  16,  17,  24,  25,   32,  33,
                  40, 48, 49, 56, 57, 64, 65, 96, 97, 127, 128, 129, 512, 1024, 1600})
        calls.push_back({t});
    for (int t : {1, 7, 33, 129})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    calls.push_back({1, CallForm::A16Convenience});
    return calls;
}

// AllowA8 up to 32 tokens keeps A16.
std::vector<Invocation> a8_small_calls() {
    std::vector<Invocation> calls;
    for (int t : {1, 4, 9, 17, 32})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    return calls;
}

// From 33 tokens: the short-band tiles below 128, then the 128-token and bulk tiles.
std::vector<Invocation> a8_calls() {
    std::vector<Invocation> calls;
    for (int t : {33,  40,  48,  63,  64,  65,  96,  127, 128,  129,  192,  255,
                  256, 257, 384, 511, 512, 513, 1023, 1024, 1025, 2048})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8});
    for (int t : {33, 65, 129, 1025})
        calls.push_back({t, CallForm::Policy, ops::LinearPolicy::AllowA8, true});
    return calls;
}

int run_gemma4_shapes() {
    int failures        = 0;
    std::uint32_t seed  = 901U;
    const auto a16      = a16_calls();
    const auto a8       = a8_calls();
    const auto a8_small = a8_small_calls();
    for (auto [n, k] : kTrunk) {
        failures += run_shape("Gemma4 FP8_A16", ActivationCompute::A16, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += run_shape("Gemma4 FP8_A16 (AllowA8)", ActivationCompute::A16, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, a8_small});
        failures += run_shape("Gemma4 FP8_A8", ActivationCompute::A8, make_fp8_weight,
                              {n, k, seed++, Comparison::Sampled, true, a8});
        failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, n, k);
        failures += run_shape("Gemma4 Q4_G32_A16", ActivationCompute::A16, make_q4_g32_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += run_shape("Gemma4 Q4_G128_A16", ActivationCompute::A16, make_q4_g128_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += run_shape("Gemma4 Q4_G256_A16", ActivationCompute::A16, make_q4_g256_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, a16});
        failures += run_shape("Gemma4 Q4_G32_A8", ActivationCompute::A8, make_q4_g32_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, a8});
        failures += verify_workspace_envelopes(QType::Q4_G32_FP16, n, k);
        failures += verify_workspace_envelopes(QType::Q4_G128_FP16, n, k);
        failures += verify_workspace_envelopes(QType::Q4_G256_FP16, n, k);
    }
    failures += run_shape("Gemma4 FP8_A16 head", ActivationCompute::A16, make_fp8_weight,
                          {262144, 5376, seed++, Comparison::Sampled, true, a16});
    failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, 262144, 5376);
    std::vector<Invocation> draft_head;
    for (int t : {1, 8, 16, 32, 33}) draft_head.push_back({t});
    draft_head.push_back({1, CallForm::A16Convenience});
    draft_head.push_back({1, CallForm::Policy, ops::LinearPolicy::A16Only, true});
    failures += run_shape("Gemma4 FP8_A16 draft head", ActivationCompute::A16, make_fp8_weight,
                          {262144, 1024, seed++, Comparison::Sampled, true, draft_head});
    failures += verify_workspace_envelopes(QType::FP8_E4M3FN_ROW_BF16, 262144, 1024);
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_gemma4_shapes) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_gemma4_shapes();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 linear shapes\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 linear shapes: " << error.what() << '\n';
        return 1;
    }
}
