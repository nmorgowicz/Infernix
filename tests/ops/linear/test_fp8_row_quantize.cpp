// ops::quantize_fp8_row_bf16: BF16 rows become row-scaled E4M3; an FP8 Weight is left alone.
#include "ops/op_family.h"
#include "infernix/ops/fp8_row_quantize.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;

int run_quantize() {
    constexpr std::int32_t n = 32;
    constexpr std::int32_t k = 64;
    auto host                = make_bf16_weight(n, k, 11U);
    std::memset(host.payload.data(), 0, static_cast<std::size_t>(k) * 2);

    DeviceBuffer device_src(host.payload.size());
    device_src.copy_from_host(host.payload.data(), device_src.bytes);
    Weight weight = host.device_weight(device_src.p);

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create fp8_row_quantize stream");
    DeviceBuffer storage;
    ops::quantize_fp8_row_bf16(weight, storage, stream);
    cuda_check(cudaStreamDestroy(stream), "destroy fp8_row_quantize stream");

    (void)ops::detail::validate_fp8_weight(weight, "fp8_row_quantize test");
    if (weight.n != n || weight.k != k) {
        std::cerr << "fp8_row_quantize: shape changed\n";
        return 1;
    }

    std::vector<std::uint8_t> payload(storage.bytes);
    storage.copy_to_host(payload.data(), payload.size());
    const auto geometry = ops::detail::fp8_row_geometry(n, k, "fp8_row_quantize test");

    int failures = 0;
    for (std::int32_t row = 0; row < n; ++row) {
        const std::uint16_t scale_bits = quantized_weight::detail::load_u16_le(
            payload, geometry.scale_plane_offset + static_cast<std::size_t>(row) * 2);
        const float scale = quantized_weight::detail::bf16_to_f32(scale_bits);
        float amax        = 0.0F;
        for (std::int32_t col = 0; col < k; ++col) {
            const auto offset = (static_cast<std::size_t>(row) * k + col) * 2;
            amax              = std::max(amax, std::fabs(quantized_weight::detail::bf16_to_f32(
                                      quantized_weight::detail::load_u16_le(host.payload, offset))));
        }
        if (row == 0) {
            bool nonzero_code = false;
            for (std::int32_t col = 0; col < k; ++col) {
                if (payload[static_cast<std::size_t>(col)] != 0) { nonzero_code = true; }
            }
            if (scale_bits != 0 || nonzero_code) {
                std::cerr << "fp8_row_quantize: zero row kept a nonzero scale or code\n";
                ++failures;
            }
            continue;
        }
        if (!(scale > 0.0F) || !std::isfinite(scale)) {
            std::cerr << "fp8_row_quantize: row " << row << " scale is not positive\n";
            ++failures;
            continue;
        }
        for (std::int32_t col = 0; col < k; ++col) {
            const auto offset = (static_cast<std::size_t>(row) * k + col) * 2;
            const float orig  = quantized_weight::detail::bf16_to_f32(
                quantized_weight::detail::load_u16_le(host.payload, offset));
            const std::uint8_t code =
                payload[static_cast<std::size_t>(row) * k + static_cast<std::size_t>(col)];
            const float recon =
                static_cast<float>(quantized_weight::detail::decode_e4m3fn(code)) * scale;
            const float limit = 0.15F * std::max(amax, 1.0e-6F) + 1.0e-6F;
            if (std::fabs(recon - orig) > limit) {
                std::cerr << "fp8_row_quantize: row " << row << " col " << col << " recon " << recon
                          << " vs " << orig << "\n";
                ++failures;
                break;
            }
        }
    }
    return failures;
}

int run_already_fp8() {
    const auto host = make_fp8_weight(32, 64, 13U);
    DeviceBuffer device(host.payload.size());
    device.copy_from_host(host.payload.data(), device.bytes);
    Weight weight           = host.device_weight(device.p);
    const void* qdata       = weight.qdata;
    cudaStream_t stream     = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create fp8_row_quantize no-op stream");
    DeviceBuffer storage;
    ops::quantize_fp8_row_bf16(weight, storage, stream);
    cuda_check(cudaStreamDestroy(stream), "destroy fp8_row_quantize no-op stream");
    if (weight.qtype != QType::FP8_E4M3FN_ROW_BF16 || weight.qdata != qdata ||
        storage.p != nullptr) {
        std::cerr << "fp8_row_quantize: FP8 weight was rewritten\n";
        return 1;
    }
    return 0;
}

} // namespace

INFERNIX_OP_TEST(linear_fp8_row_quantize) {
    if (!infernix::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_quantize() + run_already_fp8();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " fp8_row_quantize\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "fp8_row_quantize: " << error.what() << '\n';
        return 1;
    }
}
