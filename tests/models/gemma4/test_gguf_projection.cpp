// Gemma 4 projections stored as GGUF blocks (a llama.cpp GGUF import, tools/convert/gemma4_gguf.py) through
// the execution layer's project(): a fused projection's rows from several stored parents in different ggml
// formats into one output, and a global layer's output projection, whose columns stay in source order, reading
// the attention output (written in the converter's prope order) through global_output_input_perm.
//
// Weights are random block bytes with finite binary16 scales, decoded independently in binary64 by the op
// suites' transcription of llama.cpp's dequantize_row_* (ops/ggml_block_oracle.h); the oracle sums them with
// the activation in source order in binary64. Decode widths (the exact A16 route), a width on ggml_linear's
// BF16 tensor-core tiles and prefill widths (the dense route: blocks decoded to BF16, one cuBLAS GEMM) must agree
// within a BF16 rounding plus an FP32-sum and BF16-weight bound; a wrong permutation or segment offset is off by
// O(1). The dense decode itself (ops::ggml_dequantize_bf16) must equal the oracle's values rounded once to BF16,
// bit for bit, at the permuted columns. The permutation table is checked against the real Gemma 4 31B geometry
// (512-wide global heads, 64 rotated pairs).
#include "core/arena.h"
#include "core/device.h"
#include "core/ggml_format.h"
#include "models/gemma4/config.h"
#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/execution/projection.h"
#include "ops/ggml_block_oracle.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace infernix;
namespace g4 = infernix::models::gemma4;

void require(bool ok, const std::string& message) {
    if (!ok) { throw std::runtime_error(message); }
}

std::uint16_t bf16_bits(float v) {
    std::uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    return static_cast<std::uint16_t>((u + 0x7FFF + ((u >> 16) & 1)) >> 16);
}

double bf16_value(std::uint16_t b) {
    const std::uint32_t u = static_cast<std::uint32_t>(b) << 16;
    float v = 0;
    std::memcpy(&v, &u, 4);
    return v;
}

// Random rows of `q` blocks whose binary16 scales are finite and moderate (exponent 5..20).
std::vector<std::uint8_t> random_rows(QType q, int rows, int k, std::mt19937& rng) {
    const int bytes = static_cast<int>(ggml_block_bytes(q)), blocks = rows * (k / static_cast<int>(ggml_block_elems(q)));
    std::vector<std::uint8_t> data(static_cast<std::size_t>(blocks) * bytes);
    std::uniform_int_distribution<int> byte(0, 255), mantissa(0, 1023), exponent(5, 20);
    for (auto& b : data) { b = static_cast<std::uint8_t>(byte(rng)); }
    const std::vector<int> offsets = q == QType::GGML_Q6_K ? std::vector<int>{208}
                                     : (q == QType::GGML_Q4_K || q == QType::GGML_Q5_K) ? std::vector<int>{0, 2}
                                                                                          : std::vector<int>{0};
    for (int i = 0; i < blocks; ++i) {
        for (const int offset : offsets) {
            const auto w = static_cast<std::uint16_t>(mantissa(rng) | (exponent(rng) << 10));
            data[static_cast<std::size_t>(i) * bytes + offset]     = static_cast<std::uint8_t>(w & 0xFF);
            data[static_cast<std::size_t>(i) * bytes + offset + 1] = static_cast<std::uint8_t>(w >> 8);
        }
    }
    return data;
}

// Exact values [rows, k] of `rows` rows of q blocks.
std::vector<double> decode(QType q, const std::vector<std::uint8_t>& data, int rows, int k) {
    const int elems = static_cast<int>(ggml_block_elems(q)), bytes = static_cast<int>(ggml_block_bytes(q));
    std::vector<double> out(static_cast<std::size_t>(rows) * k);
    for (int b = 0; b < rows * (k / elems); ++b) {
        test::ggml_oracle::decode_block(q, data.data() + static_cast<std::size_t>(b) * bytes,
                                        out.data() + static_cast<std::size_t>(b) * elems);
    }
    return out;
}

struct Part {
    QType format;
    int rows;
};

std::uint16_t bf16_from_double(double v);

// project() over parts (one stored parent each, stacked rows) with x given in `x_order` (x_order[k] is the
// source index of element k), the input permutation `perm` (null: none), against the source-order oracle. From
// kGgmlDenseMinTokens columns the dense route multiplies BF16-rounded weights: the oracle rounds them the same way
// (ggml_dequantize_bf16's exactness is checked separately), so only the FP32-sum bound remains.
void run(const std::string& label, const std::vector<Part>& parts, int k, int columns,
         const std::vector<int>& x_order, const std::vector<std::int32_t>& perm, std::mt19937& rng) {
    int n = 0;
    std::vector<std::vector<std::uint8_t>> stored;
    std::vector<double> w; // [n, k] source order
    for (const auto& part : parts) {
        stored.push_back(random_rows(part.format, part.rows, k, rng));
        const auto values = decode(part.format, stored.back(), part.rows, k);
        w.insert(w.end(), values.begin(), values.end());
        n += part.rows;
    }
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<std::uint16_t> source(static_cast<std::size_t>(k) * columns), x(source.size());
    for (auto& v : source) { v = bf16_bits(normal(rng)); }
    for (int t = 0; t < columns; ++t) {
        for (int i = 0; i < k; ++i) {
            x[static_cast<std::size_t>(t) * k + i] = source[static_cast<std::size_t>(t) * k + x_order[i]];
        }
    }

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    std::vector<DeviceBuffer> device;
    g4::GgmlProjection projection;
    int row = 0;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        device.emplace_back(stored[i].size());
        device.back().copy_from_host(stored[i].data(), stored[i].size());
        projection.segments.push_back({parts[i].format, static_cast<const std::uint8_t*>(device.back().p), parts[i].rows,
                                       row, 0, ops::GgmlLayout::Rows});
        row += parts[i].rows;
    }
    DeviceBuffer dperm(std::max<std::size_t>(perm.size(), 1) * 4), dx(x.size() * 2),
        dout(static_cast<std::size_t>(n) * columns * 2);
    if (!perm.empty()) {
        dperm.copy_from_host(perm.data(), perm.size() * 4);
        projection.input_perm = static_cast<const std::int32_t*>(dperm.p);
        projection.perm_block = g4::execution::kGlobalOutputPermBlock;
    }
    dx.copy_from_host(x.data(), x.size() * 2);
    Weight descriptor;
    descriptor.n = n;
    descriptor.k = k;
    WorkspaceArena work(g4::projection_bytes(descriptor, projection, ops::LinearPolicy::A16Only, columns) + (1 << 20));
    ops::Fp8RowCublasContext gemm(stream);
    Tensor tx(dx.p, DType::BF16, {k, columns});
    Tensor tout(dout.p, DType::BF16, {n, columns});
    g4::project(tx, descriptor, projection, ops::LinearPolicy::A16Only, tout, gemm, work, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<std::uint16_t> got(static_cast<std::size_t>(n) * columns);
    dout.copy_to_host(got.data(), got.size() * 2);
    CUDA_CHECK(cudaStreamDestroy(stream));

    const bool dense = columns >= g4::kGgmlDenseMinTokens;
    if (dense) {
        for (auto& value : w) { value = bf16_value(bf16_from_double(value)); }
    }
    std::size_t bad = 0;
    for (int t = 0; t < columns; ++t) {
        for (int r = 0; r < n; ++r) {
            double sum = 0, magnitude = 0;
            for (int i = 0; i < k; ++i) {
                const double term = w[static_cast<std::size_t>(r) * k + i] *
                                    bf16_value(source[static_cast<std::size_t>(t) * k + i]);
                sum += term;
                magnitude += std::fabs(term);
            }
            const double actual = bf16_value(got[static_cast<std::size_t>(t) * n + r]);
            bad += std::fabs(actual - sum) > std::ldexp(std::fabs(sum), -8) + std::ldexp(magnitude, -14);
        }
    }
    require(bad == 0, label + ": " + std::to_string(bad) + " of " + std::to_string(got.size()) +
                          " outputs differ from the source-order oracle");
}

g4::TextConfig gemma4_31b_global(std::uint32_t heads) {
    g4::TextConfig c;
    c.global_head_dim     = 512;
    c.prope_pairs         = 64;
    c.num_attention_heads = heads;
    return c;
}

// x_order of the attention output: converted position p of a head holds source dim prope_perm[p].
std::vector<int> converted_order(int heads, int dim, int pairs) {
    std::vector<int> perm;
    for (int i = 0; i < pairs; ++i) { perm.push_back(i); }
    for (int i = 0; i < pairs; ++i) { perm.push_back(dim / 2 + i); }
    for (int i = 0; i < dim; ++i) {
        if (!(i < pairs || (i >= dim / 2 && i < dim / 2 + pairs))) { perm.push_back(i); }
    }
    std::vector<int> order;
    for (int h = 0; h < heads; ++h) {
        for (const int p : perm) { order.push_back(h * dim + p); }
    }
    return order;
}

void permutation_table() {
    const auto perm = g4::execution::global_output_input_perm(gemma4_31b_global(32));
    require(perm.size() == 32 * 8, "the permutation has one entry per 64-column block of 32 heads");
    const std::int32_t head[] = {0, 2, 3, 4, 1, 5, 6, 7};
    for (int h = 0; h < 32; ++h) {
        for (int b = 0; b < 8; ++b) {
            require(perm[static_cast<std::size_t>(h) * 8 + b] == h * 8 + head[b],
                    "source block " + std::to_string(b) + " of head " + std::to_string(h) + " reads the wrong block");
        }
    }
    auto split = gemma4_31b_global(1);
    split.prope_pairs = 16; // rotated runs of 16 columns: no whole 64-column blocks
    bool refused = false;
    try {
        (void)g4::execution::global_output_input_perm(split);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a permutation that splits a 64-column block was accepted");
}

// BF16 bits of a binary64 value, rounded once to nearest even (normal range).
std::uint16_t bf16_from_double(double v) {
    if (v == 0) { return std::signbit(v) ? 0x8000 : 0; }
    int e = 0;
    const double m = std::frexp(std::fabs(v), &e); // [0.5, 1)
    double r = std::nearbyint(std::ldexp(m, 8));    // 8 significant bits, ties to even
    const float f = static_cast<float>(std::copysign(std::ldexp(r, e - 8), v));
    std::uint32_t u = 0;
    std::memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>(u >> 16);
}

// ops::ggml_dequantize_bf16 against the oracle's exact values rounded once to BF16, at the permuted columns.
void dense_decode(QType q, int rows, int k, const std::vector<std::int32_t>& perm, std::mt19937& rng) {
    const auto stored = random_rows(q, rows, k, rng);
    const auto exact  = decode(q, stored, rows, k);
    DeviceBuffer dw(stored.size()), dout(static_cast<std::size_t>(rows) * k * 2), dperm(std::max<std::size_t>(perm.size(), 1) * 4);
    dw.copy_from_host(stored.data(), stored.size());
    if (!perm.empty()) { dperm.copy_from_host(perm.data(), perm.size() * 4); }
    const ops::GgmlSegment segment{q, static_cast<const std::uint8_t*>(dw.p), rows, 0, 0, ops::GgmlLayout::Rows};
    Tensor out(dout.p, DType::BF16, {k, rows});
    ops::ggml_dequantize_bf16(std::span<const ops::GgmlSegment>(&segment, 1), k, out,
                              perm.empty() ? nullptr : static_cast<const std::int32_t*>(dperm.p),
                              g4::execution::kGlobalOutputPermBlock, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint16_t> got(static_cast<std::size_t>(rows) * k);
    dout.copy_to_host(got.data(), got.size() * 2);
    const int block = g4::execution::kGlobalOutputPermBlock;
    std::size_t bad = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < k; ++c) {
            const int to = perm.empty() ? c : perm[static_cast<std::size_t>(c / block)] * block + c % block;
            bad += got[static_cast<std::size_t>(r) * k + to] != bf16_from_double(exact[static_cast<std::size_t>(r) * k + c]);
        }
    }
    require(bad == 0, "dense decode " + std::to_string(static_cast<int>(q)) + ": " + std::to_string(bad) + " of " +
                          std::to_string(got.size()) + " values differ from the oracle rounded once to BF16");
}

// E4M3 (FN, satfinite) of a binary32 value, rounded to nearest with ties to the even code: the nearest of the
// 127 finite magnitudes, independent of the device conversion.
std::uint8_t e4m3_from_float(float v) {
    static const std::vector<double> magnitudes = [] {
        std::vector<double> out;
        for (int code = 0; code < 0x7F; ++code) {
            const int e = code >> 3, m = code & 7;
            out.push_back(e == 0 ? std::ldexp(m, -9) : std::ldexp(8 + m, e - 10));
        }
        return out;
    }();
    const double a = std::fabs(static_cast<double>(v));
    int best = 0;
    for (int code = 1; code < 0x7F; ++code) {
        const double d = std::fabs(magnitudes[static_cast<std::size_t>(code)] - a), b = std::fabs(magnitudes[static_cast<std::size_t>(best)] - a);
        if (d < b || (d == b && (code & 1) == 0)) { best = code; }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(v) ? 0x80 : 0));
}

// ops::ggml_widen_e4m3 against the documented contract evaluated on the host from the oracle's values.
void widen(QType q, int rows, int k, const std::vector<std::int32_t>& perm, std::mt19937& rng) {
    const auto stored = random_rows(q, rows, k, rng);
    const auto exact  = decode(q, stored, rows, k);
    DeviceBuffer dw(stored.size()), dcodes(static_cast<std::size_t>(rows) * k), dscales(static_cast<std::size_t>(rows) * 2),
        dperm(std::max<std::size_t>(perm.size(), 1) * 4);
    dw.copy_from_host(stored.data(), stored.size());
    if (!perm.empty()) { dperm.copy_from_host(perm.data(), perm.size() * 4); }
    const ops::GgmlSegment segment{q, static_cast<const std::uint8_t*>(dw.p), rows, 0, 0, ops::GgmlLayout::Rows};
    Tensor codes(dcodes.p, DType::U8, {k, rows});
    Tensor scales(dscales.p, DType::BF16, {rows});
    ops::ggml_widen_e4m3(std::span<const ops::GgmlSegment>(&segment, 1), k, codes, scales,
                         perm.empty() ? nullptr : static_cast<const std::int32_t*>(dperm.p),
                         g4::execution::kGlobalOutputPermBlock, nullptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<std::uint8_t> got(static_cast<std::size_t>(rows) * k);
    std::vector<std::uint16_t> got_scales(static_cast<std::size_t>(rows));
    dcodes.copy_to_host(got.data(), got.size());
    dscales.copy_to_host(got_scales.data(), got_scales.size() * 2);
    const int block = g4::execution::kGlobalOutputPermBlock;
    std::size_t bad_codes = 0, bad_scales = 0;
    for (int r = 0; r < rows; ++r) {
        float largest = 0;
        for (int c = 0; c < k; ++c) { largest = std::max(largest, std::fabs(static_cast<float>(exact[static_cast<std::size_t>(r) * k + c]))); }
        std::uint16_t scale = 0x3F80; // 1.0
        if (largest > 0) {
            const float q448 = largest / 448.0F;
            std::uint32_t u  = 0;
            std::memcpy(&u, &q448, 4);
            scale = static_cast<std::uint16_t>((u >> 16) + ((u & 0xFFFF) != 0 ? 1 : 0)); // round up (positive)
        }
        bad_scales += got_scales[static_cast<std::size_t>(r)] != scale;
        const float inverse = 1.0F / static_cast<float>(bf16_value(scale));
        for (int c = 0; c < k; ++c) {
            const float v = static_cast<float>(exact[static_cast<std::size_t>(r) * k + c]);
            const int to = perm.empty() ? c : perm[static_cast<std::size_t>(c / block)] * block + c % block;
            bad_codes += got[static_cast<std::size_t>(r) * k + to] != e4m3_from_float(v * inverse);
        }
    }
    require(bad_scales == 0 && bad_codes == 0, "E4M3 widening " + std::to_string(static_cast<int>(q)) + ": " +
                                                   std::to_string(bad_scales) + " row multipliers and " +
                                                   std::to_string(bad_codes) + " codes differ from the contract");
}

// --bench: the dense route's two steps timed apart on the Gemma 4 31B trunk shapes (IQ3_S blocks, T columns).
void bench(int columns) {
    std::mt19937 rng(7);
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ops::Fp8RowCublasContext gemm(stream);
    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    const std::pair<int, int> shapes[] = {{16384, 5376}, {18432, 5376}, {5376, 8192}, {5376, 16384}, {43008, 5376},
                                          {5376, 21504}};
    for (const auto& [n, k] : shapes) {
        const auto stored = random_rows(QType::GGML_IQ3_S, n, k, rng);
        DeviceBuffer dw(stored.size()), dense(static_cast<std::size_t>(n) * k * 2), dx(static_cast<std::size_t>(k) * columns * 2),
            dp(static_cast<std::size_t>(n) * columns * 4);
        dw.copy_from_host(stored.data(), stored.size());
        dx.fill(0);
        const ops::GgmlSegment segment{QType::GGML_IQ3_S, static_cast<const std::uint8_t*>(dw.p), n, 0, 0, ops::GgmlLayout::Rows};
        Tensor tw(dense.p, DType::BF16, {k, n}), tx(dx.p, DType::BF16, {k, columns}), tp(dp.p, DType::FP32, {n, columns});
        float decode_ms = 0, gemm_ms = 0;
        for (int rep = 0; rep < 6; ++rep) {
            CUDA_CHECK(cudaEventRecord(e0, stream));
            ops::ggml_dequantize_bf16(std::span<const ops::GgmlSegment>(&segment, 1), k, tw, nullptr, 0, stream);
            CUDA_CHECK(cudaEventRecord(e1, stream));
            gemm.gemm_bf16_fp32(tx, tw, tp, stream);
            CUDA_CHECK(cudaEventRecord(e2, stream));
            CUDA_CHECK(cudaEventSynchronize(e2));
            float a = 0, b = 0;
            CUDA_CHECK(cudaEventElapsedTime(&a, e0, e1));
            CUDA_CHECK(cudaEventElapsedTime(&b, e1, e2));
            if (rep > 0) { decode_ms += a / 5; gemm_ms += b / 5; }
        }
        std::cout << n << "x" << k << " T=" << columns << ": decode " << decode_ms << " ms, gemm " << gemm_ms << " ms ("
                  << 2.0 * n * k * columns / (gemm_ms * 1e-3) / 1e12 << " TFLOP/s)\n";
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

} // namespace

int main(int argc, char** argv) {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device; skipped\n";
        return 77;
    }
    try {
        if (argc > 1 && std::string(argv[1]) == "--bench") {
            bench(argc > 2 ? std::atoi(argv[2]) : 1024);
            return 0;
        }
        permutation_table();
        std::mt19937 rng(20261115);
        const int heads = 2, dim = 512, k = heads * dim;
        const auto perm = g4::execution::global_output_input_perm(gemma4_31b_global(heads));
        const auto order = converted_order(heads, dim, 64);
        std::vector<int> identity(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) { identity[static_cast<std::size_t>(i)] = i; }
        for (const QType q : {QType::GGML_Q2_0, QType::GGML_Q4_K, QType::GGML_Q5_K, QType::GGML_Q6_K, QType::GGML_IQ2_S,
                              QType::GGML_IQ3_XXS, QType::GGML_IQ3_S, QType::GGML_IQ4_XS, QType::GGML_IQ4_NL}) {
            dense_decode(q, 24, k, {}, rng);
            dense_decode(q, 24, k, perm, rng);
            widen(q, 24, k, {}, rng);
            widen(q, 24, k, perm, rng);
        }
        for (const int columns : {1, 4, 40, 64, 300}) {
            const std::string width = " T=" + std::to_string(columns);
            run("global output IQ3_S" + width, {{QType::GGML_IQ3_S, 96}}, k, columns, order, perm, rng);
            run("global output Q6_K" + width, {{QType::GGML_Q6_K, 64}}, k, columns, order, perm, rng);
            run("fused IQ3_XXS+IQ2_S+Q6_K" + width,
                {{QType::GGML_IQ3_XXS, 48}, {QType::GGML_IQ2_S, 16}, {QType::GGML_Q6_K, 16}}, k, columns, identity, {},
                rng);
        }
        std::cout << "gemma4 GGUF projections (fused segments, global output permutation) passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
