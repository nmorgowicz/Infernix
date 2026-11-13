// Adapted from ninfer.spo b330aaf (Gemma 4 attention oracles).
#include "core/temporary_switches.h"
#include "infernix/ops/gemma4_attention.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kPage = 64;
// From this many query tokens on, attention runs on the tensor-core path.
constexpr int kMmaMinTokens = 64;
constexpr PointwiseCriterion kCriterion{2.0e-5, 8.0e-3};

// Configure before the immutable temporary-switch snapshot is first read.
bool configure_split(int argc, char** argv) {
    if (argc == 1) { return true; }
    if (argc != 3) { return false; }
    const std::string_view key(argv[1]), value(argv[2]);
    int segments = -1;
    if (key == "--split" && (value == "0" || value == "1")) {
        segments = value == "0" ? 1 : 0;
    } else if (key == "--segments") {
        for (int n = 1; n <= 13; ++n) {
            if (value == std::to_string(n)) { segments = n; break; }
        }
    }
    if (segments < 0) { return false; }
    const std::string setting = std::to_string(segments);
#ifdef _WIN32
    return _putenv_s("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str()) == 0;
#else
    return setenv("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", setting.c_str(), 1) == 0;
#endif
}

// Logical page p lives in physical page pages - 1 - p, so the block table is exercised.
std::size_t kv_index(int head_dim, int kv_heads, int pages, int head, int position, int dim) {
    const int slot     = position % kPage;
    const int physical = pages - 1 - position / kPage;
    return (static_cast<std::size_t>(head_dim) * kPage *
            (static_cast<std::size_t>(head) + static_cast<std::size_t>(kv_heads) * physical)) +
           static_cast<std::size_t>(head_dim) * slot + static_cast<std::size_t>(dim);
}

// Queries sit at the last `tokens` of `kv_len` cached positions.
int run_case(const char* base_label, int head_dim, int q_heads, int kv_heads, int tokens,
             int kv_len, int window, float scale = 1.0f) {
    const std::string name(base_label);
    const char* label = name.c_str();
    const int pages            = (kv_len + kPage - 1) / kPage;
    const int first_query      = kv_len - tokens;
    const std::size_t q_count  = static_cast<std::size_t>(head_dim) * q_heads * tokens;
    const std::size_t kv_count = static_cast<std::size_t>(head_dim) * kPage * kv_heads * pages;
    std::vector<float> q(q_count), k(kv_count, 0.0f), v(kv_count, 0.0f);
    fill_uniform(q, 70u, -1.0f, 1.0f);
    round_to_bf16(q);
    std::vector<float> k_logical(static_cast<std::size_t>(head_dim) * kv_heads * kv_len);
    std::vector<float> v_logical = k_logical;
    fill_uniform(k_logical, 71u, -1.0f, 1.0f);
    fill_uniform(v_logical, 72u, -1.0f, 1.0f);
    round_to_bf16(k_logical);
    round_to_bf16(v_logical);
    for (int position = 0; position < kv_len; ++position) {
        for (int head = 0; head < kv_heads; ++head) {
            for (int dim = 0; dim < head_dim; ++dim) {
                const std::size_t logical =
                    (static_cast<std::size_t>(position) * kv_heads + head) * head_dim + dim;
                k[kv_index(head_dim, kv_heads, pages, head, position, dim)] = k_logical[logical];
                v[kv_index(head_dim, kv_heads, pages, head, position, dim)] = v_logical[logical];
            }
        }
    }

    const int group = q_heads / kv_heads;
    // FP64 reference for the given per-token positions; a negative position sees no key and
    // yields a zero row.
    const auto oracle = [&](const std::vector<std::int32_t>& at) {
        std::vector<double> expected(q_count, 0.0);
        std::vector<double> scores;
        for (int token = 0; token < tokens; ++token) {
            const int position = at[static_cast<std::size_t>(token)];
            if (position < 0) { continue; }
            for (int qh = 0; qh < q_heads; ++qh) {
                const int kvh   = qh / group;
                const int first = window == 0 ? 0 : std::max(0, position - window + 1);
                scores.assign(static_cast<std::size_t>(position - first + 1), 0.0);
                double max_score = -1.0e300;
                for (int key = first; key <= position; ++key) {
                    double dot = 0.0;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        dot += static_cast<double>(
                                   q[(static_cast<std::size_t>(token) * q_heads + qh) * head_dim +
                                     dim]) *
                               k_logical[(static_cast<std::size_t>(key) * kv_heads + kvh) *
                                             head_dim +
                                         dim];
                    }
                    scores[static_cast<std::size_t>(key - first)] = dot * scale;
                    max_score = std::max(max_score, dot * scale);
                }
                double sum = 0.0;
                for (double& value : scores) {
                    value = std::exp(value - max_score);
                    sum += value;
                }
                double* row = &expected[(static_cast<std::size_t>(token) * q_heads + qh) * head_dim];
                for (int key = first; key <= position; ++key) {
                    const double weight = scores[static_cast<std::size_t>(key - first)] / sum;
                    for (int dim = 0; dim < head_dim; ++dim) {
                        row[dim] += weight *
                                    v_logical[(static_cast<std::size_t>(key) * kv_heads + kvh) *
                                                  head_dim +
                                              dim];
                    }
                }
            }
        }
        return expected;
    };
    std::vector<std::int32_t> positions(static_cast<std::size_t>(tokens));
    for (int i = 0; i < tokens; ++i) { positions[static_cast<std::size_t>(i)] = first_query + i; }
    const std::vector<double> expected = oracle(positions);

    auto bits = [](const std::vector<float>& values) {
        std::vector<std::uint16_t> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) { out[i] = f32_to_bf16(values[i]); }
        return out;
    };
    const auto q_bits = bits(q);
    const auto k_bits = bits(k);
    const auto v_bits = bits(v);
    GuardedDeviceBuffer dq(q_bits.size() * 2), dk(k_bits.size() * 2), dv(v_bits.size() * 2),
        dout(q_bits.size() * 2);
    std::vector<std::int32_t> table(static_cast<std::size_t>(pages));
    for (int page = 0; page < pages; ++page) {
        table[static_cast<std::size_t>(page)] = pages - 1 - page;
    }
    GuardedDeviceBuffer dtable(table.size() * sizeof(std::int32_t)),
        dpos(positions.size() * sizeof(std::int32_t));
    dq.copy_from_host(q_bits.data(), dq.bytes());
    dk.copy_from_host(k_bits.data(), dk.bytes());
    dv.copy_from_host(v_bits.data(), dv.bytes());
    dtable.copy_from_host(table.data(), dtable.bytes());
    dpos.copy_from_host(positions.data(), dpos.bytes());
    Tensor q_tensor(dq.data(), DType::BF16, {head_dim, q_heads, tokens});
    Tensor k_tensor(dk.data(), DType::BF16, {head_dim, kPage, kv_heads, pages});
    Tensor v_tensor(dv.data(), DType::BF16, {head_dim, kPage, kv_heads, pages});
    Tensor table_tensor(dtable.data(), DType::I32, {pages});
    Tensor pos_tensor(dpos.data(), DType::I32, {tokens});
    Tensor out_tensor(dout.data(), DType::BF16, {head_dim, q_heads, tokens});
    const std::size_t capacity =
        ops::gemma4_attention_workspace_capacity_bytes(head_dim, q_heads, kv_heads, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    const ops::Gemma4CacheView cache{ops::Gemma4KvFormat::BFloat16, &k_tensor, &v_tensor,
                                    nullptr, nullptr, &table_tensor};
    ops::gemma4_attention(q_tensor, cache, pos_tensor, window, scale,
                          out_tensor, workspace, nullptr);
    cuda_synchronize();
    std::vector<std::uint16_t> first_bits(q_count), second_bits(q_count);
    dout.copy_to_host(first_bits.data(), dout.bytes());
    int failures =
        verify_pointwise(label, from_device_bf16(dout.data(), q_count), expected, kCriterion);
    ops::gemma4_attention(q_tensor, cache, pos_tensor, window, scale,
                          out_tensor, workspace, nullptr);
    cuda_synchronize();
    dout.copy_to_host(second_bits.data(), dout.bytes());
    if (second_bits != first_bits) {
        std::cerr << label << ": a second run changed the output\n";
        ++failures;
    }
    if (tokens < kMmaMinTokens) {
        // The decode grid is fixed at capture and every key range comes from the device
        // positions: capture at early positions, replay at the real ones (the context grows
        // under the captured grid), then at the early ones again, then with no visible key.
        std::vector<std::int32_t> early(positions.size());
        for (std::size_t i = 0; i < early.size(); ++i) { early[i] = positions[i] / 3; }
        const std::vector<std::int32_t> none(positions.size(), -1);
        const auto upload = [&](const std::vector<std::int32_t>& at) {
            dpos.copy_from_host(at.data(), dpos.bytes());
        };
        cudaStream_t stream;
        cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");
        upload(early);
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                   "cudaStreamBeginCapture");
        ops::gemma4_attention(q_tensor, cache, pos_tensor, window, scale, out_tensor, workspace,
                              stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "cudaStreamEndCapture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0),
                   "cudaGraphInstantiate");
        upload(positions);
        cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
        cuda_synchronize(stream);
        std::vector<std::uint16_t> replay_bits(q_count);
        dout.copy_to_host(replay_bits.data(), dout.bytes());
        if (replay_bits != first_bits) {
            std::cerr << label << ": graph replay at grown positions differs from eager\n";
            ++failures;
        }
        upload(early);
        cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
        cuda_synchronize(stream);
        failures += verify_pointwise(
            (name + " graph replay at captured positions").c_str(),
            from_device_bf16(dout.data(), q_count), oracle(early), kCriterion);
        upload(none);
        cuda_check(cudaGraphLaunch(executable, stream), "cudaGraphLaunch");
        cuda_synchronize(stream);
        dout.copy_to_host(replay_bits.data(), dout.bytes());
        if (replay_bits != std::vector<std::uint16_t>(q_count, 0)) {
            std::cerr << label << ": graph replay with no visible key is not zero\n";
            ++failures;
        }
        upload(positions);
        cuda_check(cudaGraphExecDestroy(executable), "cudaGraphExecDestroy");
        cuda_check(cudaGraphDestroy(graph), "cudaGraphDestroy");
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }
    failures += dq.verify_guards("q");
    failures += dout.verify_guards("out");
    return failures;
}

int run_cases() {
    int failures = run_case("sliding window", 256, 32, 16, 6, 6, 1024);
    failures += run_case("sliding narrow", 256, 32, 16, 5, 5, 2);
    failures += run_case("global", 512, 32, 4, 4, 4, 0);
    failures += run_case("sliding pages", 256, 32, 16, 150, 150, 40);
    failures += run_case("global pages", 512, 32, 4, 130, 130, 0);
    failures += run_case("sliding decode", 256, 32, 16, 1, 300, 100);
    failures += run_case("global decode", 512, 32, 4, 1, 333, 0);
    failures += run_case("global chunk", 512, 32, 4, 20, 200, 0);
    failures += run_case("sliding decode past window", 256, 32, 16, 1, 1500, 1024);
    failures += run_case("global decode long", 512, 32, 4, 1, 4096, 0);
    failures += run_case("global decode short", 512, 32, 4, 1, 3, 0);
    failures += run_case("global verify tokens", 512, 32, 4, 5, 2000, 0);
    failures += run_case("sliding verify tokens", 256, 32, 16, 6, 1100, 1024);
    // Every decode/verify width 1..16 on the global layer: a context that is all causal
    // prefix (most chunks empty), one that ends mid-tile and mid-page, and a long one whose
    // last chunk is partial.
    for (int width = 1; width <= 16; ++width) {
        const std::string suffix = " width " + std::to_string(width);
        failures += run_case(("global causal" + suffix).c_str(), 512, 32, 4, width, width, 0);
        failures += run_case(("global partial" + suffix).c_str(), 512, 32, 4, width, 333 + width, 0);
        failures += run_case(("global long" + suffix).c_str(), 512, 32, 4, width, 1100 + 13 * width,
                             0);
    }
    failures += run_case("global verify 16 very long", 512, 32, 4, 16, 4100, 0);
    failures += run_case("global decode scaled", 512, 32, 4, 3, 777, 0, 0.125f);
    // 172 and 252 (token x KV head) CTAs exceed half a 170-SM wave (2 CTAs per SM), so the
    // shared kernel does not split the keys and writes BF16 directly.
    failures += run_case("global wide 43 tokens", 512, 32, 4, 43, 450, 0);
    failures += run_case("global wide 63 tokens", 512, 32, 4, 63, 700, 0);
    // Other group sizes: the shared kernel serves groups up to 8 with idle warps below 8 and
    // the original kernel keeps wider groups.
    failures += run_case("global group 1", 512, 4, 4, 2, 300, 0);
    failures += run_case("global group 2", 512, 8, 4, 3, 700, 0);
    failures += run_case("global group 8 two KV heads", 512, 16, 2, 16, 1500, 0);
    failures += run_case("global group 9", 512, 36, 4, 2, 200, 0);
    // From 64 query tokens: the tensor-core path.
    failures += run_case("sliding one tile", 256, 32, 16, 64, 64, 1024);
    failures += run_case("sliding chunk past window", 256, 32, 16, 300, 1400, 1024);
    failures += run_case("sliding chunk narrow window", 256, 32, 16, 129, 400, 70);
    failures += run_case("global one tile", 512, 32, 4, 64, 64, 0);
    failures += run_case("global chunk after prefix", 512, 32, 4, 193, 700, 0);
    failures += run_case("global split context scaled", 512, 4, 2, 65, 769, 0, 0.125f);
    failures += run_case("global split context sharp", 512, 4, 2, 64, 1025, 0);
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gemma4_attention) {
    if (!configure_split(argc, argv)) {
        std::cerr << "usage: gemma4_attention [--split 0|1 | --segments 1..13]\n";
        return 2;
    }
    std::cout << "temporary switches | "
              << temporary_switches::describe(temporary_switches::Scope::Gemma4) << '\n';
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    const int failures = run_cases();
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_attention\n";
    return failures ? 1 : 0;
}
