#include "infernix/ops/argmax.h"
#include "infernix/ops/topk_columns.h"

#include "core/device.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

// Row a ranks before row b: a NaN in row 0 first, a NaN elsewhere after every number, then
// the larger value, then the lower row.
bool ranks_before(float a, std::int32_t row_a, float b, std::int32_t row_b) {
    const auto rank_class = [](float value, std::int32_t row) {
        return std::isnan(value) ? (row == 0 ? 2 : 0) : 1;
    };
    const int ca = rank_class(a, row_a);
    const int cb = rank_class(b, row_b);
    if (ca != cb) { return ca > cb; }
    if (ca != 1) { return row_a < row_b; }
    return a > b || (a == b && row_a < row_b);
}

std::vector<std::int32_t> topk_oracle(const std::vector<std::uint16_t>& logits,
                                      std::int32_t physical_rows, std::int32_t tokens,
                                      std::int32_t valid_rows, std::int32_t k) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(k) * tokens, -1);
    std::vector<std::int32_t> rows(static_cast<std::size_t>(valid_rows));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < valid_rows; ++row) { rows[static_cast<std::size_t>(row)] = row; }
        std::sort(rows.begin(), rows.end(), [&](std::int32_t a, std::int32_t b) {
            return ranks_before(bf16_to_f32(logits[base + a]), a, bf16_to_f32(logits[base + b]), b);
        });
        for (std::int32_t rank = 0; rank < std::min(k, valid_rows); ++rank) {
            expected[static_cast<std::size_t>(token) * k + rank] = rows[static_cast<std::size_t>(rank)];
        }
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -8.0f + static_cast<float>(mixed % 1024u) * (1.0f / 64.0f);
            logits[base + row] = f32_to_bf16(value);
        }
        // Two exact ties inside the valid rows, and a huge value past valid_rows.
        const std::int32_t first  = (17 + token * 97) % valid_rows;
        std::int32_t second       = (valid_rows - 1 - token * 13) % valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        logits[base + second] = logits[base + first];
        if (valid_rows < physical_rows) {
            logits[base + valid_rows] = f32_to_bf16(1.0e4f);
        }
    }
    return logits;
}

enum class NanCase { kNone, kRowZero, kOtherRows, kAllButZero };

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens,
             std::int32_t k, NanCase nan_case = NanCase::kNone) {
    auto logits = make_logits(physical_rows, tokens, valid_rows);
    constexpr std::uint16_t kNan = 0x7fc0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        if (nan_case == NanCase::kRowZero) { logits[base] = kNan; }
        if (nan_case == NanCase::kOtherRows) {
            for (std::int32_t row = 1 + token; row < valid_rows; row += 5) { logits[base + row] = kNan; }
        }
        if (nan_case == NanCase::kAllButZero) {
            for (std::int32_t row = 1; row < valid_rows; ++row) { logits[base + row] = kNan; }
        }
    }
    auto expected = topk_oracle(logits, physical_rows, tokens, valid_rows, k);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_ids(static_cast<std::size_t>(k) * tokens * sizeof(std::int32_t));
    GuardedDeviceBuffer device_argmax(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor ids_tensor(device_ids.data(), DType::I32, {k, tokens});
    Tensor argmax_tensor(device_argmax.data(), DType::I32, {tokens});
    ops::topk_columns(logits_tensor, valid_rows, k, ids_tensor, nullptr);
    ops::argmax(logits_tensor, argmax_tensor, valid_rows, nullptr);
    cuda_synchronize();

    const auto actual = from_device<std::int32_t>(device_ids.data(),
                                                  static_cast<std::size_t>(k) * tokens);
    const auto argmax_ids =
        from_device<std::int32_t>(device_argmax.data(), static_cast<std::size_t>(tokens));
    const std::string label = "topk rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens) + " k=" + std::to_string(k);
    int failures = verify_exact(label.c_str(), actual, expected);
    for (std::int32_t token = 0; token < tokens; ++token) {
        if (actual[static_cast<std::size_t>(token) * k] != argmax_ids[static_cast<std::size_t>(token)]) {
            std::cerr << label << " rank 0 token " << token << " is not argmax\n";
            ++failures;
        }
    }
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_ids.verify_guards((label + " ids").c_str());
    return failures;
}

} // namespace

INFERNIX_OP_TEST(topk_columns) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    for (const int k : {1, 4, 8}) {
        for (const int tokens : {1, 3, 6}) {
            failures += run_case(262144, 262144, tokens, k);
            failures += run_case(64, 50, tokens, k);
            failures += run_case(4096, 4096, tokens, k, NanCase::kRowZero);
            failures += run_case(4096, 4096, tokens, k, NanCase::kOtherRows);
        }
    }
    // Fewer valid rows than k: the missing ranks are -1.
    failures += run_case(8, 3, 2, 8);
    // Every row NaN but row 0: row 0 first, then the NaN rows by index.
    failures += run_case(16, 16, 1, 8, NanCase::kAllButZero);
    std::cout << (failures ? "FAIL" : "OK") << " topk_columns\n";
    return failures ? 1 : 0;
}
