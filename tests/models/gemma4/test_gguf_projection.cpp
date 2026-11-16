// Gemma 4 projections stored as GGUF blocks (a llama.cpp GGUF import, tools/convert/gemma4_gguf.py) through
// the execution layer's project(): a fused projection's rows from several stored parents in different ggml
// formats into one output, and a global layer's output projection, whose columns stay in source order, reading
// the attention output (written in the converter's prope order) through global_output_input_perm.
//
// Weights are random block bytes with finite binary16 scales, decoded independently in binary64 by the op
// suites' transcription of llama.cpp's dequantize_row_* (ops/ggml_block_oracle.h); the oracle sums them with
// the activation in source order in binary64. Decode widths (the exact A16 route) and a prefill width (the
// BF16 tensor-core route) must agree within a BF16 rounding plus an FP32-sum bound; a wrong permutation or
// segment offset is off by O(1). The permutation table itself is checked against the real Gemma 4 31B
// geometry (512-wide global heads, 64 rotated pairs).
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

// project() over parts (one stored parent each, stacked rows) with x given in `x_order` (x_order[k] is the
// source index of element k), the input permutation `perm` (null: none), against the source-order oracle.
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

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::cout << "no CUDA device; skipped\n";
        return 77;
    }
    try {
        permutation_table();
        std::mt19937 rng(20261115);
        const int heads = 2, dim = 512, k = heads * dim;
        const auto perm = g4::execution::global_output_input_perm(gemma4_31b_global(heads));
        const auto order = converted_order(heads, dim, 64);
        std::vector<int> identity(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) { identity[static_cast<std::size_t>(i)] = i; }
        for (const int columns : {1, 4, 40}) {
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
