#pragma once

// Test-owned codec/oracle: no CUDA intrinsics or production decoder/FP8 converter.
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace infernix::test::q5_bake {
namespace qw = quantized_weight;

inline float f32_product(float a, float b) {
    volatile float rounded = a * b;
    return rounded;
}
inline float f32_divide(float a, float b) {
    volatile float rounded = a / b;
    return rounded;
}
inline float bf16_value(std::uint16_t bits) {
    const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}
inline std::uint16_t bf16_bits(float value) {
    std::uint32_t word;
    std::memcpy(&word, &value, sizeof(word));
    return static_cast<std::uint16_t>((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
}
inline std::uint16_t bf16_up(float positive) {
    std::uint32_t word;
    std::memcpy(&word, &positive, sizeof(word));
    return static_cast<std::uint16_t>((word >> 16) + ((word & 0xffffU) != 0));
}
inline double e4m3_value(std::uint8_t code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (code & 128) ? -magnitude : magnitude;
}
inline std::uint8_t e4m3_bits(float value) {
    static const auto positive = [] {
        std::array<double, 127> result{};
        for (int code = 0; code <= 126; ++code) { result[code] = e4m3_value(static_cast<std::uint8_t>(code)); }
        return result;
    }();
    const double magnitude = std::fabs(static_cast<double>(value));
    const auto upper = std::lower_bound(positive.begin(), positive.end(), magnitude);
    int code = static_cast<int>(upper - positive.begin());
    if (code == 127) {
        code = 126;
    } else if (code > 0) {
        const double high_error = positive[code] - magnitude;
        const double low_error = magnitude - positive[code - 1];
        if (low_error < high_error || (low_error == high_error && ((code - 1) & 1) == 0)) { --code; }
    }
    return static_cast<std::uint8_t>(code | (std::signbit(value) ? 128 : 0));
}

// Sample both endpoints and the handcrafted rows, as well as the interior row strides.
inline std::vector<std::int32_t> sampled_rows(int extent) {
    std::vector<std::int32_t> rows;
    for (int row = 0; row < std::min(extent, 9); ++row) { rows.push_back(row); }
    for (int i = 0; i < 16; ++i) { rows.push_back(static_cast<int>(static_cast<std::int64_t>(extent - 1) * i / 15)); }
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
    return rows;
}
inline std::vector<std::int32_t> sampled_tokens(int extent) {
    std::vector<std::int32_t> tokens;
    for (int i = 0; i < std::min(extent, 32); ++i) {
        tokens.push_back(extent <= 32 ? i : static_cast<int>(static_cast<std::int64_t>(extent - 1) * i / 31));
    }
    return tokens;
}

inline void store_code(qw::PackedWeight& packed, int row, int column, int code) {
    const int groups = packed.weight.padded_shape[1] / 64;
    const auto group = static_cast<std::size_t>(row) * groups + column / 64;
    const int lane = column % 64;
    const auto low_at = group * 32 + lane / 2;
    const int shift = (lane & 1) * 4;
    const unsigned raw = static_cast<unsigned>(code) & 31U;
    packed.payload[low_at] = static_cast<std::uint8_t>((packed.payload[low_at] & ~(15U << shift)) |
                                                     ((raw & 15U) << shift));
    const auto high_at = packed.high_plane_offset + group * 8 + lane / 8;
    packed.payload[high_at] = static_cast<std::uint8_t>((packed.payload[high_at] & ~(1U << (lane % 8))) |
                                                       ((raw >> 4) << (lane % 8)));
}

inline qw::PackedWeight make_weight(int n, int k, std::uint32_t seed) {
    auto packed = qw::make_patterned_weight(QType::Q5_G64_FP16, n, k, seed,
                                            {qw::RowSplitScalePattern::Small, qw::RowSplitCodePattern::Hashed});
    const int groups = packed.weight.padded_shape[1] / 64;
    for (int row = 0; row < std::min(n, 9); ++row) {
        for (int group = 0; group < groups; ++group) {
            std::uint16_t scale = 0x2401; // Non-power-of-two binary16 multiplier.
            if (row == 1) { scale = 0; }
            if (row == 2) { scale = static_cast<std::uint16_t>(1 + group % 1023); }
            if (row == 3) { scale = static_cast<std::uint16_t>(0x8001U + group % 1023); }
            if (row == 4) { scale = group == 0 ? 0x3c01 : 0x1801; }
            if (row == 5) { scale = group == 0 ? 0x7bff : 0x1c01; }
            if (row == 6) { scale = group % 2 ? 0x0001 : 0x3c00; }
            // m=448 -> row multiplier=1. E4M3 halfway cases 8.5/9.5 and half-subnormal ties.
            if (row == 7) {
                constexpr std::uint16_t scales[]{0x5400, 0x4840, 0x48c0, 0x1400, 0x1a00};
                scale = scales[group % 5];
            }
            const auto at = packed.scale_plane_offset + (static_cast<std::size_t>(row) * groups + group) * 2;
            packed.payload[at] = static_cast<std::uint8_t>(scale);
            packed.payload[at + 1] = static_cast<std::uint8_t>(scale >> 8);
        }
        for (int column = 0; column < k; ++column) {
            int code = (column + row) % 32 - 16;
            if (row == 0) { code = 0; }
            if (row == 4) { code = column / 64 == 0 ? 1 : 15; }
            if (row == 5 && column < 64) { code = 0; }
            if (row == 7) { code = column / 64 % 5 == 0 ? 7 : (column % 2 ? -1 : 1); }
            if (row == 8) { code = column < k - 64 ? 0 : code; }
            store_code(packed, row, column, code);
        }
    }
    return packed;
}

struct Row {
    std::uint16_t scale;
    std::vector<std::uint8_t> codes;
    std::vector<float> original;
    std::vector<float> represented;
};
inline Row encode_row(const qw::PackedWeight& packed, int row) {
    Row out;
    const int k = packed.weight.k;
    out.original.resize(k);
    out.represented.resize(k);
    out.codes.resize(k);
    float maximum = 0;
    for (int column = 0; column < k; ++column) {
        // Test-owned persistent decoder reads low/high bits and exact binary16 scales.
        const float value = static_cast<float>(qw::logical_weight_fp64(packed, row, column));
        out.original[column] = value;
        maximum = std::max(maximum, std::fabs(value));
    }
    out.scale = maximum > 0 ? bf16_up(f32_divide(maximum, 448.0F)) : bf16_bits(1.0F);
    const float scale = bf16_value(out.scale);
    const float inverse = f32_divide(1.0F, scale);
    for (int column = 0; column < k; ++column) {
        out.codes[column] = e4m3_bits(f32_product(out.original[column], inverse));
        out.represented[column] = static_cast<float>(e4m3_value(out.codes[column]) * scale);
    }
    return out;
}

inline std::vector<std::uint16_t> make_activation(int k, int tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    std::uint32_t state = seed;
    for (auto& value : bits) {
        state = state * 1664525U + 1013904223U;
        value = bf16_bits(static_cast<float>(static_cast<int>((state >> 16) & 1023U) - 512) / 512.0F);
    }
    // Include cancellation, a zero column and an activation outlier in every run.
    for (int column = 0; column < k; ++column) { bits[column] = bf16_bits(column % 2 ? -0.25F : 0.25F); }
    if (tokens > 1) { std::fill(bits.begin() + k, bits.begin() + 2 * k, 0); }
    if (tokens > 2) { bits[static_cast<std::size_t>(2) * k + k - 1] = bf16_bits(16.0F); }
    return bits;
}

inline std::vector<float> activation_column(const std::vector<std::uint16_t>& bits, int k, int token,
                                             bool encoded, bool negative = false) {
    std::vector<float> values(k);
    float maximum = 0;
    for (int column = 0; column < k; ++column) {
        float value = bf16_value(bits[static_cast<std::size_t>(token) * k + column]);
        if (negative) { value = -value; }
        values[column] = value;
        maximum = std::max(maximum, std::fabs(value));
    }
    if (!encoded) { return values; }
    const float scale = maximum > 0 ? f32_divide(maximum, 448.0F) : 0;
    const float inverse = scale > 0 ? f32_divide(1.0F, scale) : 0;
    for (auto& value : values) {
        value = static_cast<float>(e4m3_value(e4m3_bits(f32_product(value, inverse))) * scale);
    }
    return values;
}

} // namespace infernix::test::q5_bake
