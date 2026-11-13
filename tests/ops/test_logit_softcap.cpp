#include "infernix/ops/logit_softcap.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr PointwiseCriterion kCriterion{2.0e-6, 4.05e-3};

int run_case(const char* label, float cap, std::int32_t n, std::uint32_t seed) {
    std::vector<float> input(static_cast<std::size_t>(n));
    fill_uniform(input, seed, -80.0f, 80.0f);
    round_to_bf16(input);
    std::vector<double> expected(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) {
        expected[i] = static_cast<double>(cap) * std::tanh(input[i] / static_cast<double>(cap));
    }
    std::vector<std::uint16_t> bits(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) { bits[i] = f32_to_bf16(input[i]); }
    GuardedDeviceBuffer device(bits.size() * sizeof(std::uint16_t));
    device.copy_from_host(bits.data(), device.bytes());
    Tensor tensor(device.data(), DType::BF16, {n});
    ops::logit_softcap(tensor, cap, nullptr);
    cuda_synchronize();
    int failures =
        verify_pointwise(label, from_device_bf16(device.data(), input.size()), expected, kCriterion);
    failures += device.verify_guards("logit_softcap");
    return failures;
}

} // namespace

INFERNIX_OP_TEST(logit_softcap) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case("softcap 30", 30.0f, 4096, 30u);
    failures += run_case("softcap 30 odd", 30.0f, 3, 31u);
    std::cout << (failures ? "FAIL" : "OK") << " logit_softcap\n";
    return failures ? 1 : 0;
}
