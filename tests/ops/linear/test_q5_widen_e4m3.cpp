#include "ops/op_family.h"
#include "ops/q5_bake_oracle.h"
#include "ops/op_tester.h"
#include "infernix/ops/linear.h"
#include "core/decode_graph.h"

#include <array>
#include <exception>
#include <iostream>
#include <utility>

namespace {
using namespace infernix;
namespace oracle = infernix::test::q5_bake;

int run_bake(int n, int k, std::uint32_t seed) {
    auto host = oracle::make_weight(n, k, seed);
    DeviceBuffer device(host.payload.size());
    device.copy_from_host(host.payload.data(), device.bytes);
    const Weight weight = host.device_weight(device.p);
    test::GuardedDeviceBuffer output(static_cast<std::size_t>(n) * k);
    test::GuardedDeviceBuffer scales(static_cast<std::size_t>(n) * 2);
    Tensor codes(output.data(), DType::U8, {k, n});
    Tensor rows(scales.data(), DType::BF16, {n});
    DeviceContext context;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    const auto launch = [&] { ops::q5_widen_e4m3(weight, codes, rows, context.stream); };
    CUDA_CHECK(cudaDeviceSynchronize());
    definition.capture(context.stream, launch);
    graph.instantiate(definition);
    int failures = 0;
    const auto selected = oracle::sampled_rows(n);
    for (int replay = 0; replay < 3; ++replay) {
        if (replay == 2) {
            // Same captured addresses, different signed binary16 scales (including signed zero).
            const int groups = host.weight.padded_shape[1] / 64;
            for (int row : selected) {
                for (int g = 0; g < groups; ++g) {
                    host.payload[host.scale_plane_offset + (static_cast<std::size_t>(row) * groups + g) * 2 + 1] ^= 0x80;
                }
            }
            device.copy_from_host(host.payload.data(), device.bytes);
        }
        output.fill(0xff);
        scales.fill(0xff);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (replay == 0) { launch(); } else { graph.launch(context.stream); }
        CUDA_CHECK(cudaStreamSynchronize(context.stream));
        failures += output.verify_guards("Q5 bake codes");
        failures += scales.verify_guards("Q5 bake scales");
        std::vector<std::uint16_t> multipliers(n);
        scales.copy_to_host(multipliers.data(), multipliers.size() * 2);
        for (int row : selected) {
            const auto reference = oracle::encode_row(host, row);
            std::vector<std::uint8_t> actual(k);
            output.copy_to_host(actual.data(), actual.size(), static_cast<std::size_t>(row) * k);
            if (multipliers[row] != reference.scale || actual != reference.codes) {
                std::cerr << "Q5 bake [" << n << ',' << k << "] row=" << row
                          << " replay=" << replay << ": differs from independent exact byte oracle\n";
                ++failures;
            }
        }
    }
    std::vector<std::uint8_t> after(host.payload.size());
    device.copy_to_host(after.data(), device.bytes);
    if (after != host.payload) {
        std::cerr << "Q5 bake modified persistent weight\n";
        ++failures;
    }
    return failures;
}
} // namespace

INFERNIX_OP_TEST(linear_q5_widen_e4m3) {
    if (infernix::test::cuda_unavailable()) { return 77; }
    try {
        int failures = 0;
        std::uint32_t seed = 517;
        for (auto [n, k] : std::array{std::pair{16384, 5376}, std::pair{18432, 5376},
                                      std::pair{43008, 5376}, std::pair{5376, 8192},
                                      std::pair{5376, 16384}, std::pair{5376, 21504},
                                      // Codec-only case: logical groups end before the padded row stride.
                                      std::pair{19, 192}}) {
            failures += run_bake(n, k, seed++);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5 row E4M3 widening\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5 row E4M3 widening: " << error.what() << '\n';
        return 1;
    }
}
