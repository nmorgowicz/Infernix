// Gemma 4 31B native Q5 group64 on the opt-in INT8-activation prefill route (LinearRoutes::Q5Int8G32 / Q5Int8G64).
//
// The route's semantics are an exact integer dot per activation group, FP32 scale products in ascending group
// order and one BF16 rounding. This test restates them independently on the host (it decodes the stored five-bit
// planes and FP16 scales itself, quantizes each sampled token by the documented rule, takes the dot in int64 and
// the chain in float with std::fmaf) and compares the BF16 output bit for bit on sampled rows and tokens, beside a
// binary64 ideal (exact stored weights x the quantized activations' dequantization is the chain; the BF16
// activations are only the quantizer's input). Every output element is checked for poison and guards, the
// workspace query and arena scope are checked, calls replay from a CUDA graph with a changed input, and widths
// below the route's threshold (and the exact route without the flag) are untouched.
//
// INFERNIX_Q5_INT8_DUMP_DIR: also write each case's sampled planes, activations and device output
// (manifest.jsonl + case<i>.bin) for the independent NumPy oracle (camp q5-int8-20261118/oracle_engine.py).

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "infernix/ops/linear.h"
#include "ops/linear/linear_test_common.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test::linear;
namespace qw = infernix::test::quantized_weight;

constexpr std::array kShapes{std::pair{16384, 5376}, std::pair{18432, 5376}, std::pair{43008, 5376},
                             std::pair{5376, 8192},  std::pair{5376, 16384}, std::pair{5376, 21504}};

ops::LinearRoutes routes_of(int group) { return group == 32 ? ops::LinearRoutes::Q5Int8G32 : ops::LinearRoutes::Q5Int8G64; }

std::uint32_t hash32(std::uint64_t v) { return static_cast<std::uint32_t>(qw::detail::mix64(v) >> 32); }

// BF16 activations [tokens][k], token-dependent magnitude, all-zero groups, rare outliers, both signs and values that
// land on round-half-even ties of the quantizer.
std::vector<std::uint16_t> make_activations(int k, int tokens, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (int t = 0; t < tokens; ++t) {
        const float magnitude = std::ldexp(1.0f, -6 + static_cast<int>(hash32(seed * 31ULL + t) % 11U));
        for (int c = 0; c < k; ++c) {
            const std::uint32_t h = hash32((static_cast<std::uint64_t>(seed) << 40) ^ (static_cast<std::uint64_t>(t) << 20) ^ c);
            float value           = static_cast<float>(static_cast<int>(h & 0xffU) - 128) / 256.0f * magnitude;
            if ((h >> 8) % 977U == 0U) { value *= 64.0f; }
            if ((static_cast<std::uint32_t>(c / 32) * 7U + static_cast<std::uint32_t>(t)) % 13U == 0U) { value = 0.0f; }
            bits[static_cast<std::size_t>(t) * k + c] = test::f32_to_bf16(value);
        }
    }
    return bits;
}

std::vector<int> sample_indices(int extent, int count) {
    std::vector<int> out;
    for (int i = 0; i < count; ++i) { out.push_back(static_cast<int>((static_cast<std::int64_t>(extent - 1) * i) / (count - 1))); }
    out.push_back(extent - 1);
    if (extent > 1) { out.push_back(extent - 2); }
    if (extent > 128) { out.push_back((extent - 1) / 128 * 128); }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// One stored row, decoded from the payload planes: K-linear signed codes and the FP16 group-64 scales.
struct DecodedRow {
    std::vector<int> q;
    std::vector<float> ws;
};

DecodedRow decode_row(const qw::PackedWeight& host, int row, int k) {
    const auto spec     = qw::detail::quant_spec(QType::Q5_G64_FP16);
    const int groups    = host.weight.padded_shape[1] / 64;
    DecodedRow out;
    out.q.resize(k);
    out.ws.resize(k / 64);
    for (int g = 0; g < k / 64; ++g) {
        const std::size_t group = static_cast<std::size_t>(row) * groups + g;
        const std::uint8_t* nibble = host.payload.data() + group * 32;
        const std::uint8_t* high   = host.payload.data() + host.high_plane_offset + group * 8;
        for (int i = 0; i < 64; ++i) { out.q[g * 64 + i] = qw::detail::unpack_lowbit_code(nibble, high, spec, i); }
        out.ws[g] = qw::detail::f16_to_f32(qw::detail::load_u16_le(host.payload, host.scale_plane_offset + group * 2));
    }
    return out;
}

// The documented activation rule: scale max|x| / 127 in FP32 (1 for an all-zero group), two low mantissa bits
// cleared, FP32 reciprocal multiply, round-half-even, clamp +-127.
void quantize_token(const float* x, int k, int group, std::vector<int>& codes, std::vector<float>& scale) {
    codes.assign(k, 0);
    scale.assign(k / group, 1.0f);
    for (int g = 0; g < k / group; ++g) {
        float ma = 0.0f;
        for (int i = 0; i < group; ++i) { ma = std::max(ma, std::fabs(x[g * group + i])); }
        float s = ma > 0.0f ? ma / 127.0f : 1.0f;
        std::uint32_t bits;
        std::memcpy(&bits, &s, 4);
        bits &= 0xFFFFFFFCu;
        std::memcpy(&s, &bits, 4);
        const float inv = 1.0f / s;
        for (int i = 0; i < group; ++i) {
            const float q          = std::nearbyintf(x[g * group + i] * inv);
            codes[g * group + i]   = static_cast<int>(std::min(127.0f, std::max(-127.0f, q)));
        }
        scale[g] = s;
    }
}

struct Expected {
    std::uint16_t bf16;
    double ideal; // exact stored weights x dequantized INT8 activations, binary64
};

Expected reference(const DecodedRow& row, const std::vector<int>& codes, const std::vector<float>& scale, int k, int group) {
    float run     = 0.0f;
    double ideal  = 0.0;
    for (int g = 0; g < k / group; ++g) {
        std::int64_t dot = 0;
        for (int i = 0; i < group; ++i) { dot += static_cast<std::int64_t>(row.q[g * group + i]) * codes[g * group + i]; }
        const float t  = static_cast<float>(static_cast<double>(dot) * static_cast<double>(scale[g]));
        const float ws = row.ws[(g * group) / 64];
        run            = std::fmaf(t, ws, run);
        ideal += static_cast<double>(dot) * scale[g] * ws;
    }
    return {test::f32_to_bf16(run), ideal};
}

std::string dump_dir() {
    const char* value = std::getenv("INFERNIX_Q5_INT8_DUMP_DIR");
    return value ? value : "";
}

struct Fixture {
    int n, k;
    qw::PackedWeight host;
    DeviceBuffer device;
    Weight weight;
    Fixture(int n_, int k_, std::uint32_t seed) : n(n_), k(k_), host(make_q5_g64_fp16_weight(n_, k_, seed)), device(host.payload.size()) {
        device.copy_from_host(host.payload.data(), device.bytes);
        weight = host.device_weight(device.p);
    }
};

int g_case = 0;

// One call of `tokens` columns on `group`'s route, optionally replayed from a CUDA graph with the input negated.
int run_call(Fixture& f, int tokens, int group, bool graph_replay, std::uint32_t seed) {
    const int n = f.n, k = f.k;
    const std::string label = "Q5 int8-g" + std::to_string(group) + " [" + std::to_string(n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(tokens) + (graph_replay ? " graph" : "");
    int failures = 0;
    auto x_bits  = make_activations(k, tokens, seed);
    DeviceBuffer x_device(x_bits.size() * 2);
    x_device.copy_from_host(x_bits.data(), x_device.bytes);
    test::GuardedDeviceBuffer output(static_cast<std::size_t>(n) * tokens * 2);
    Tensor input(x_device.p, DType::BF16, {k, tokens});
    Tensor destination(output.data(), DType::BF16, {n, tokens});
    const auto routes         = routes_of(group);
    const std::size_t capacity = ops::linear_q5_int8_prefill_workspace_bytes(n, k, tokens, routes);
    if (capacity == 0) {
        std::cerr << label << ": the workspace query returned zero for a selected call\n";
        return 1;
    }
    DeviceArena workspace(capacity);
    std::unique_ptr<DeviceContext> context;
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (graph_replay) { context = std::make_unique<DeviceContext>(); }
    const cudaStream_t stream = context ? context->stream : nullptr;
    const auto launch         = [&] { ops::linear(input, f.weight, destination, ops::LinearPolicy::A16Only, workspace, stream, routes); };
    if (!ops::linear_q5_int8_prefill_selected(f.weight, tokens, routes)) {
        std::cerr << label << ": the route was not selected\n";
        return 1;
    }
    const auto rows   = sample_indices(n, 33);
    const auto cols   = sample_indices(tokens, 33);
    std::vector<DecodedRow> decoded;
    for (int row : rows) { decoded.push_back(decode_row(f.host, row, k)); }

    try {
        if (graph_replay) {
            test::cuda_check(cudaDeviceSynchronize(), "finish input initialization before capture");
            definition.capture(stream, launch);
            executable.instantiate(definition);
        }
        for (int replay = 0; replay < (graph_replay ? 2 : 1); ++replay) {
            if (replay == 1) {
                for (auto& bits : x_bits) { bits ^= 0x8000; } // BF16 negation is exact
                x_device.copy_from_host(x_bits.data(), x_device.bytes);
            }
            output.fill(0xff);
            test::cuda_check(cudaDeviceSynchronize(), "finish poisoning before the call");
            if (graph_replay) {
                executable.launch(stream);
            } else {
                launch();
            }
            test::cuda_check(cudaStreamSynchronize(stream), "synchronize linear");
            if (workspace.peak_used() > capacity || workspace.used() != 0) {
                std::cerr << label << ": workspace query/scope mismatch (peak " << workspace.peak_used() << " of " << capacity << ")\n";
                ++failures;
            }
            failures += output.verify_guards(label);
            std::vector<std::uint16_t> out(static_cast<std::size_t>(n) * tokens);
            output.copy_to_host(out.data(), out.size() * 2);
            std::size_t unwritten = 0;
            for (std::uint16_t v : out) { unwritten += v == 0xffffU ? 1 : 0; }
            if (unwritten != 0) {
                std::cerr << label << ": " << unwritten << " output elements were not written\n";
                ++failures;
            }
            // The independent host reference on the sampled rows and columns.
            std::vector<float> activation(k);
            std::vector<int> codes;
            std::vector<float> scale;
            std::size_t mismatches = 0;
            double error = 0.0, norm = 0.0;
            for (int token : cols) {
                for (int c = 0; c < k; ++c) { activation[c] = test::bf16_to_f32(x_bits[static_cast<std::size_t>(token) * k + c]); }
                quantize_token(activation.data(), k, group, codes, scale);
                for (std::size_t r = 0; r < rows.size(); ++r) {
                    const Expected e = reference(decoded[r], codes, scale, k, group);
                    const std::uint16_t actual = out[static_cast<std::size_t>(token) * n + rows[r]];
                    if (actual != e.bf16) { ++mismatches; }
                    const double device_value = test::bf16_to_f32(actual);
                    error += (device_value - e.ideal) * (device_value - e.ideal);
                    norm += e.ideal * e.ideal;
                }
            }
            if (mismatches != 0) {
                std::cerr << label << (replay ? " replay" : "") << ": " << mismatches << " of " << rows.size() * cols.size()
                          << " sampled BF16 outputs differ from the independent chain\n";
                ++failures;
            }
            // The chain's only deviation from the binary64 ideal is FP32 association and the final BF16 rounding.
            if (norm > 0.0 && std::sqrt(error / norm) > 0.004) {
                std::cerr << label << ": relative L2 to the binary64 ideal " << std::sqrt(error / norm) << " exceeds 0.004\n";
                ++failures;
            }
            if (const std::string dir = dump_dir(); !dir.empty() && replay == 0) {
                std::filesystem::create_directories(dir);
                const std::string base = dir + "/case" + std::to_string(g_case);
                std::ofstream bin(base + ".bin", std::ios::binary);
                const int groups = f.host.weight.padded_shape[1] / 64;
                for (int row : rows) {
                    const std::size_t g0 = static_cast<std::size_t>(row) * groups;
                    bin.write(reinterpret_cast<const char*>(f.host.payload.data() + g0 * 32), static_cast<std::streamsize>(k / 2));
                }
                for (int row : rows) {
                    const std::size_t g0 = static_cast<std::size_t>(row) * groups;
                    bin.write(reinterpret_cast<const char*>(f.host.payload.data() + f.host.high_plane_offset + g0 * 8), static_cast<std::streamsize>(k / 8));
                }
                for (int row : rows) {
                    const std::size_t g0 = static_cast<std::size_t>(row) * groups;
                    bin.write(reinterpret_cast<const char*>(f.host.payload.data() + f.host.scale_plane_offset + g0 * 2), static_cast<std::streamsize>(k / 64 * 2));
                }
                for (int token : cols) {
                    bin.write(reinterpret_cast<const char*>(x_bits.data() + static_cast<std::size_t>(token) * k), static_cast<std::streamsize>(k * 2));
                }
                for (int token : cols) {
                    for (int row : rows) { bin.write(reinterpret_cast<const char*>(&out[static_cast<std::size_t>(token) * n + row]), 2); }
                }
                std::ofstream manifest(dir + "/manifest.jsonl", std::ios::app);
                manifest << "{\"case\":" << g_case << ",\"n\":" << n << ",\"k\":" << k << ",\"t\":" << tokens << ",\"group\":" << group
                         << ",\"rows\":[";
                for (std::size_t i = 0; i < rows.size(); ++i) { manifest << (i ? "," : "") << rows[i]; }
                manifest << "],\"tokens\":[";
                for (std::size_t i = 0; i < cols.size(); ++i) { manifest << (i ? "," : "") << cols[i]; }
                manifest << "]}\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << label << ": unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    ++g_case;
    return failures;
}

// Calls the route does not take run the exact route's bits, and the flag combinations are validated.
int run_boundaries(Fixture& f) {
    int failures = 0;
    const int k  = f.k;
    for (int group : {32, 64}) {
        for (int tokens : {1, 8, 63}) {
            const auto x_bits = make_activations(k, tokens, 77);
            DeviceBuffer x_device(x_bits.size() * 2);
            x_device.copy_from_host(x_bits.data(), x_device.bytes);
            DeviceBuffer exact(static_cast<std::size_t>(f.n) * tokens * 2), flagged(static_cast<std::size_t>(f.n) * tokens * 2);
            Tensor input(x_device.p, DType::BF16, {k, tokens});
            Tensor a(exact.p, DType::BF16, {f.n, tokens}), b(flagged.p, DType::BF16, {f.n, tokens});
            DeviceArena workspace(256);
            ops::linear(input, f.weight, a, ops::LinearPolicy::A16Only, workspace, nullptr, ops::LinearRoutes::Exact);
            ops::linear(input, f.weight, b, ops::LinearPolicy::A16Only, workspace, nullptr, routes_of(group));
            test::cuda_check(cudaDeviceSynchronize(), "boundary calls");
            std::vector<std::uint16_t> ha(static_cast<std::size_t>(f.n) * tokens), hb(ha.size());
            exact.copy_to_host(ha.data(), ha.size() * 2);
            flagged.copy_to_host(hb.data(), hb.size() * 2);
            if (ha != hb) {
                std::cerr << "Q5 int8-g" << group << " T=" << tokens << ": a call below 64 columns changed the exact route's bits\n";
                ++failures;
            }
            if (ops::linear_q5_int8_prefill_selected(f.weight, tokens, routes_of(group)) ||
                ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, tokens, routes_of(group)) != 0) {
                std::cerr << "Q5 int8-g" << group << " T=" << tokens << ": selected below 64 columns\n";
                ++failures;
            }
        }
    }
    if (ops::linear_q5_int8_prefill_selected(f.weight, 64, ops::LinearRoutes::Exact) ||
        ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, 2048, ops::LinearRoutes::Exact) != 0) {
        std::cerr << "Q5 int8: selected without a flag\n";
        ++failures;
    }
    std::size_t last = 0;
    for (int tokens : {64, 65, 128, 129, 2048, 4096}) {
        const std::size_t bytes = ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, tokens, ops::LinearRoutes::Q5Int8G32);
        if (bytes < last || bytes <= static_cast<std::size_t>(tokens) * k) {
            std::cerr << "Q5 int8: the workspace query is not monotonic or does not cover the activations at T=" << tokens << '\n';
            ++failures;
        }
        last = bytes;
    }
    if (ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, 2048, ops::LinearRoutes::Q5Int8G32) <=
        ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, 2048, ops::LinearRoutes::Q5Int8G64)) {
        std::cerr << "Q5 int8: per-32 scales must need more scratch than per-64\n";
        ++failures;
    }
    // Both group flags together name no route.
    try {
        const auto x_bits = make_activations(k, 128, 78);
        DeviceBuffer x_device(x_bits.size() * 2), out_device(static_cast<std::size_t>(f.n) * 128 * 2);
        x_device.copy_from_host(x_bits.data(), x_device.bytes);
        Tensor input(x_device.p, DType::BF16, {k, 128}), destination(out_device.p, DType::BF16, {f.n, 128});
        DeviceArena workspace(ops::linear_q5_int8_prefill_workspace_bytes(f.n, k, 128, ops::LinearRoutes::Q5Int8G32));
        ops::linear(input, f.weight, destination, ops::LinearPolicy::A16Only, workspace, nullptr,
                    ops::LinearRoutes::Q5Int8G32 | ops::LinearRoutes::Q5Int8G64);
        std::cerr << "Q5 int8: both group flags were accepted\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    return failures;
}

int run_all() {
    int failures = 0;
    std::uint32_t seed = 31;
    for (auto [n, k] : kShapes) {
        Fixture f(n, k, seed++);
        for (int group : {32, 64}) {
            for (int tokens : {64, 65, 128, 129, 193, 256}) { failures += run_call(f, tokens, group, false, seed++); }
            failures += run_call(f, 128, group, true, seed++);
        }
        if (n == 43008 || k == 21504) {
            for (int group : {32, 64}) {
                failures += run_call(f, 2048, group, false, seed++);
                failures += run_call(f, 2048, group, true, seed++);
            }
        }
        if (n == 16384) { failures += run_boundaries(f); }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(linear_gemma4_q5_int8_prefill) {
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_all();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 native Q5 INT8 prefill Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 native Q5 INT8 prefill Linear: " << error.what() << '\n';
        return 1;
    }
}
