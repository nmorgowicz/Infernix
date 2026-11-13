#include "infernix/ops/geglu.h"
#include "ops/linear/fp8/fp8_a8_plan.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr PointwiseCriterion kCriterion{2.0e-6, 4.05e-3};

double gelu_tanh(double z) {
    constexpr double c = 0.79788456080286535587989211986876;
    return 0.5 * z * (1.0 + std::tanh(c * (z + 0.044715 * z * z * z)));
}

int run_case(const char* label, std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -6.0f, 6.0f);
    fill_uniform(up, seed + 1, -6.0f, 6.0f);
    round_to_bf16(gate);
    round_to_bf16(up);
    std::vector<double> expected(count);
    for (std::size_t i = 0; i < count; ++i) { expected[i] = gelu_tanh(gate[i]) * up[i]; }

    std::vector<std::uint16_t> gate_bits(count), up_bits(count);
    for (std::size_t i = 0; i < count; ++i) {
        gate_bits[i] = f32_to_bf16(gate[i]);
        up_bits[i]   = f32_to_bf16(up[i]);
    }
    GuardedDeviceBuffer device_gate(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_up(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate.copy_from_host(gate_bits.data(), device_gate.bytes());
    device_up.copy_from_host(up_bits.data(), device_up.bytes());
    Tensor gate_tensor(device_gate.data(), DType::BF16, {rows, columns});
    Tensor up_tensor(device_up.data(), DType::BF16, {rows, columns});
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::geglu(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    kCriterion);
    failures += device_gate.verify_guards("geglu gate");
    failures += device_up.verify_guards("geglu up");
    failures += device_out.verify_guards("geglu out");
    return failures;
}

int run_strided_gate_up_case(const char* label, std::int32_t rows, std::int32_t columns,
                             std::uint32_t seed) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> gate(count), up(count);
    fill_uniform(gate, seed, -6.0f, 6.0f);
    fill_uniform(up, seed + 1, -6.0f, 6.0f);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<float> gate_up(2 * count);
    for (std::int32_t column = 0; column < columns; ++column) {
        const std::size_t source = static_cast<std::size_t>(column) * rows;
        const std::size_t target = static_cast<std::size_t>(column) * 2 * rows;
        std::copy_n(gate.data() + source, rows, gate_up.data() + target);
        std::copy_n(up.data() + source, rows, gate_up.data() + target + rows);
    }

    std::vector<double> expected(count);
    for (std::size_t i = 0; i < count; ++i) { expected[i] = gelu_tanh(gate[i]) * up[i]; }

    std::vector<std::uint16_t> gate_up_bits(gate_up.size());
    for (std::size_t i = 0; i < gate_up.size(); ++i) { gate_up_bits[i] = f32_to_bf16(gate_up[i]); }
    GuardedDeviceBuffer device_gate_up(gate_up_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    device_gate_up.copy_from_host(gate_up_bits.data(), device_gate_up.bytes());

    Tensor gate_up_tensor(device_gate_up.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = gate_up_tensor.slice(0, 0, rows);
    Tensor up_tensor   = gate_up_tensor.slice(0, rows, rows);
    Tensor out_tensor(device_out.data(), DType::BF16, {rows, columns});
    ops::geglu(gate_tensor, up_tensor, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), count), expected,
                                    kCriterion);
    failures += verify_exact("geglu strided inputs unchanged",
                             from_device<std::uint16_t>(device_gate_up.data(), gate_up_bits.size()),
                             gate_up_bits);
    failures += device_gate_up.verify_guards("geglu strided inputs");
    failures += device_out.verify_guards("geglu strided out");
    return failures;
}

} // namespace

int fp8_codes_match_quantize(std::int32_t rows, std::int32_t columns) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    const std::size_t wide  = static_cast<std::size_t>(rows) * 2 * columns;
    std::vector<float> gate(wide / 2), up(wide / 2);
    fill_uniform(gate, 7u, -4.0f, 4.0f);
    fill_uniform(up, 8u, -4.0f, 4.0f);
    round_to_bf16(gate);
    round_to_bf16(up);
    std::vector<std::uint16_t> bits(wide);
    for (std::size_t col = 0; col < static_cast<std::size_t>(columns); ++col) {
        for (std::int32_t row = 0; row < rows; ++row) {
            bits[col * rows * 2 + row]            = f32_to_bf16(gate[col * rows + row]);
            bits[col * rows * 2 + rows + row]     = f32_to_bf16(up[col * rows + row]);
        }
    }
    GuardedDeviceBuffer packed(bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer out_fused(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer out_plain(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer codes_fused(count);
    GuardedDeviceBuffer codes_ref(count);
    GuardedDeviceBuffer scales_fused(static_cast<std::size_t>(columns) * sizeof(float));
    GuardedDeviceBuffer scales_ref(static_cast<std::size_t>(columns) * sizeof(float));
    packed.copy_from_host(bits.data(), packed.bytes());
    Tensor wide_tensor(packed.data(), DType::BF16, {2 * rows, columns});
    Tensor gate_tensor = wide_tensor.slice(0, 0, rows);
    Tensor up_tensor   = wide_tensor.slice(0, rows, rows);
    Tensor fused(out_fused.data(), DType::BF16, {rows, columns});
    Tensor plain(out_plain.data(), DType::BF16, {rows, columns});
    ops::geglu(gate_tensor, up_tensor, fused, nullptr, static_cast<std::uint8_t*>(codes_fused.data()),
               static_cast<float*>(scales_fused.data()));
    ops::geglu(gate_tensor, up_tensor, plain, nullptr);
    Weight weight;
    weight.k = rows;
    infernix::ops::detail::Fp8A8Workspace scratch{
        static_cast<std::uint8_t*>(codes_ref.data()), static_cast<float*>(scales_ref.data()),
        nullptr};
    infernix::ops::detail::launch_fp8_a8_quantize(plain, weight, scratch, nullptr);
    cuda_synchronize();
    const auto fused_bf16 = from_device<std::uint16_t>(out_fused.data(), count);
    const auto plain_bf16 = from_device<std::uint16_t>(out_plain.data(), count);
    int failures          = fused_bf16 == plain_bf16 ? 0 : 1;
    if (failures != 0) { std::cout << "FAIL geglu fp8 changed the bf16 output\n"; }
    const auto a = from_device<std::uint8_t>(codes_fused.data(), count);
    const auto b = from_device<std::uint8_t>(codes_ref.data(), count);
    if (a != b) {
        std::cout << "FAIL geglu fp8 codes differ from the quantize kernel\n";
        failures += 1;
    }
    const auto sa = from_device<float>(scales_fused.data(), static_cast<std::size_t>(columns));
    const auto sb = from_device<float>(scales_ref.data(), static_cast<std::size_t>(columns));
    if (sa != sb) {
        std::cout << "FAIL geglu fp8 scales differ from the quantize kernel\n";
        failures += 1;
    }
    return failures;
}

INFERNIX_OP_TEST(geglu) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = run_case("geglu [21504,8]", 21504, 8, 41u);
    failures += run_case("geglu odd", 1, 7, 42u);
    failures += run_strided_gate_up_case("geglu strided [21504,8]", 21504, 8, 43u);
    failures += run_strided_gate_up_case("geglu strided odd", 7, 3, 44u);
    failures += fp8_codes_match_quantize(21504, 4);
    std::cout << (failures ? "FAIL" : "OK") << " geglu\n";
    return failures ? 1 : 0;
}
