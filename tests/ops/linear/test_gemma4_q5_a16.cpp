#include "ops/op_family.h"
#include "ops/linear/linear_test_common.h"
#include "ops/linear/q5/q5_launch.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace infernix;
using namespace infernix::test;
using namespace infernix::test::linear;

// Exactly the matrices Parameters::geometry prepares: sliding QKV = 8192 + 4096 + 4096,
// global QK = 16384 + 2048 (V is K), gate/up = 2 * 21504, and the three output projections.
constexpr std::array kShapes{std::pair{16384, 5376}, std::pair{18432, 5376},
                             std::pair{43008, 5376}, std::pair{5376, 8192},
                             std::pair{5376, 16384}, std::pair{5376, 21504}};

constexpr Invocation graph(int tokens) {
    return {tokens, CallForm::Policy, ops::LinearPolicy::A16Only, true};
}

// Both sides of every SIMT/sliced-K/MMA/ring boundary, including the engine's prefill
// chunk sizes and the decode/MTP verification widths. Captures replay twice, changing X.
constexpr std::array kCalls{Invocation{1}, Invocation{2}, Invocation{3}, Invocation{4},
                            Invocation{5}, Invocation{8}, Invocation{9}, Invocation{16},
                            Invocation{17}, Invocation{32}, Invocation{33}, Invocation{64},
                            Invocation{65}, Invocation{127}, Invocation{128}, Invocation{129},
                            Invocation{2048}, graph(1), graph(2), graph(8), graph(128), graph(2048),
                            Invocation{1, CallForm::A16Convenience}};
constexpr std::array kStressCalls{Invocation{1}, Invocation{2}, Invocation{8},
                                  Invocation{128}, Invocation{2048}};

int run_gemma4_q5() {
    int failures = 0;
    std::uint32_t seed = 5376;
    for (auto [n, k] : kShapes) {
        // The interval query is the model planner's route; these A16 launches need no scratch.
        if (ops::linear_workspace_capacity_bytes(QType::Q5_G64_FP16, n, k,
                                                 ops::LinearPolicy::A16Only, 1, 2048) != 0) {
            std::cerr << "Gemma4 native Q5: unexpected workspace for [" << n << ',' << k << "]\n";
            ++failures;
        }
        // The existing fixture independently decodes the stored five-bit planes and exact
        // binary16 scales to FP32, then sums BF16 X * decoded W in FP64. Sample 32 evenly
        // spaced rows/columns (including both ends), while checking every output for poison,
        // non-finites and guards; do not compare against another GPU kernel.
        failures += run_shape("Gemma4 Q5 A16", ActivationCompute::A16,
                              make_q5_g64_fp16_weight,
                              {n, k, seed++, Comparison::Sampled, true, kCalls});
    }
    failures += run_shape("Gemma4 Q5 A16 cancellation", ActivationCompute::A16,
                          make_q5_g64_fp16_weight,
                          {5376, 21504, seed++, Comparison::Sampled, true, kStressCalls,
                           ActivationPattern::Cancellation});
    failures += run_shape("Gemma4 Q5 A16 final group", ActivationCompute::A16,
                          make_q5_g64_fp16_weight,
                          {43008, 5376, seed++, Comparison::Sampled, true, kStressCalls,
                           ActivationPattern::KTail});

    // A nearby unregistered shape must not silently acquire a runtime-shape fallback.
    try {
        (void)ops::linear_workspace_capacity_bytes(QType::Q5_G64_FP16, 5376, 5376,
                                                   ops::LinearPolicy::A16Only, 1, 2048);
        std::cerr << "Gemma4 native Q5: admitted an unregistered shape\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    return failures;
}

// Measurement only (`--time`): event-timed, L2-cold GEMV launches, median of 50 after 10 warm-ups.
// Three rotated copies of the weight payload, activation and output buffers defeat the 96 MiB L2
// (a 76 MiB down weight set would otherwise stay resident across the timing loop), so every timed
// launch streams its weights from DRAM the way one decode step does. Every instance of a cell runs
// on identical device data and its output is byte-compared against the cell's first instance: the
// scheduling-identity oracle of the Q5 decode work (A16 exactness makes any arithmetic or
// accumulation-order change visible as a byte difference).
class ColdL2Timer {
public:
    static constexpr int kCopies = 3;

    ColdL2Timer(quantized_weight::PackedWeight host, std::int32_t k, std::int32_t tokens)
        : payload_bytes_(host.payload.size()), n_(host.weight.n), k_(k), tokens_(tokens) {
        for (int copy = 0; copy < kCopies; ++copy) {
            payloads_.emplace_back(host.payload.size());
            payloads_.back().copy_from_host(host.payload.data(), host.payload.size());
            weights_.push_back(host.device_weight(payloads_.back().p));
        }
        std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens));
        for (std::size_t index = 0; index < bits.size(); ++index) {
            const float value   = static_cast<float>(static_cast<int>(index % 13U) - 6) * 0.125F;
            const auto raw      = static_cast<std::uint32_t>(*reinterpret_cast<const std::uint32_t*>(&value));
            bits[index]         = static_cast<std::uint16_t>(raw >> 16U);
        }
        for (int copy = 0; copy < kCopies; ++copy) {
            activations_.emplace_back(bits.size() * sizeof(std::uint16_t));
            activations_.back().copy_from_host(bits.data(), bits.size() * sizeof(std::uint16_t));
            outputs_.emplace_back(static_cast<std::size_t>(n_) * static_cast<std::size_t>(tokens) *
                                  sizeof(std::uint16_t));
            activations_view_.push_back(Tensor(activations_.back().p, DType::BF16, {k_, tokens_}));
            outputs_view_.push_back(Tensor(outputs_.back().p, DType::BF16, {n_, tokens_}));
        }
        cuda_check(cudaEventCreate(&begin_), "cudaEventCreate");
        cuda_check(cudaEventCreate(&end_), "cudaEventCreate");
    }

    ColdL2Timer(const ColdL2Timer&)            = delete;
    ColdL2Timer& operator=(const ColdL2Timer&) = delete;
    ~ColdL2Timer() {
        cudaEventDestroy(begin_);
        cudaEventDestroy(end_);
    }

    // `launch(copy)` must enqueue one kernel launch on the default stream reading copy `copy`.
    // Each event window times a batch of 8 rotating launches and reports the per-launch figure:
    // kernels queue back-to-back inside a window, so the per-launch sync gap (which idles the
    // device and sags the clocks the way no decode step does) is amortized over the batch.
    template <class Launch>
    float median_us(Launch launch) {
        constexpr int kBatch  = 8;
        constexpr int kWindows = 8;
        for (int warm = 0; warm < kBatch; ++warm) { launch(warm % kCopies); }
        cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        std::vector<float> micros;
        micros.reserve(kWindows);
        for (int window = 0; window < kWindows; ++window) {
            cuda_check(cudaEventRecord(begin_, nullptr), "cudaEventRecord");
            for (int item = 0; item < kBatch; ++item) {
                launch((window * kBatch + item) % kCopies);
            }
            cuda_check(cudaEventRecord(end_, nullptr), "cudaEventRecord");
            cuda_check(cudaEventSynchronize(end_), "cudaEventSynchronize");
            float milliseconds = 0.0F;
            cuda_check(cudaEventElapsedTime(&milliseconds, begin_, end_), "cudaEventElapsedTime");
            micros.push_back(milliseconds * 1000.0F / kBatch);
        }
        std::sort(micros.begin(), micros.end());
        return micros[micros.size() / 2];
    }

    const Weight& weight(int copy) const { return weights_[static_cast<std::size_t>(copy)]; }

    const Tensor& activation(int copy) const {
        return activations_view_[static_cast<std::size_t>(copy)];
    }

    Tensor& output(int copy) { return outputs_view_[static_cast<std::size_t>(copy)]; }

    // Exact bytes one launch moves: the three weight planes plus the activation read and the
    // output write (the Q5 payload is codes 32 B + high 8 B + scale 2 B per 64-value group).
    double bytes() const {
        return static_cast<double>(payload_bytes_) + static_cast<double>(k_) * tokens_ * 2.0 +
               static_cast<double>(n_) * tokens_ * 2.0;
    }

    std::vector<std::uint16_t> read_output(int copy) const {
        const auto words =
            static_cast<std::size_t>(n_) * static_cast<std::size_t>(tokens_);
        std::vector<std::uint16_t> host(words);
        outputs_[static_cast<std::size_t>(copy)].copy_to_host(host.data(), words * sizeof(std::uint16_t));
        return host;
    }

private:
    std::vector<DeviceBuffer> payloads_;
    std::vector<DeviceBuffer> activations_;
    std::vector<DeviceBuffer> outputs_;
    std::vector<Tensor> activations_view_;
    std::vector<Tensor> outputs_view_;
    std::vector<Weight> weights_;
    std::uint64_t payload_bytes_ = 0;
    std::int32_t n_              = 0;
    std::int32_t k_              = 0;
    std::int32_t tokens_         = 0;
    cudaEvent_t begin_           = nullptr;
    cudaEvent_t end_             = nullptr;
};

struct Q5TimeInstance {
    const char* name;
    ops::detail::Q5Launch launch;
};

// Times every (instance, tokens) cell on identical data; byte-compares each instance's output
// against the cell's first instance (skipped when the cell holds a single instance).
void time_q5_cells(std::string_view shape_label, std::int32_t n, std::int32_t k,
                   std::span<const Q5TimeInstance> instances,
                   std::span<const std::int32_t> token_list, std::uint32_t seed) {
    for (const std::int32_t tokens : token_list) {
        std::vector<std::vector<std::uint16_t>> reference;
        for (const Q5TimeInstance& instance : instances) {
            ColdL2Timer timer(make_q5_g64_fp16_weight(n, k, seed), k, tokens);
            const float micros = timer.median_us([&](int copy) {
                instance.launch(timer.activation(copy), timer.weight(copy), timer.output(copy), nullptr);
            });
            const double gb_s = timer.bytes() / static_cast<double>(micros) / 1000.0;
            std::cout << "Q5_TIME shape=" << shape_label << " n=" << n << " k=" << k
                      << " tokens=" << tokens << " instance=" << instance.name
                      << " us=" << micros << " gb_s=" << gb_s << '\n';
            instance.launch(timer.activation(0), timer.weight(0), timer.output(0), nullptr);
            cuda_check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
            std::vector<std::uint16_t> produced = timer.read_output(0);
            if (reference.empty()) {
                reference.push_back(std::move(produced));
            } else {
                const auto& expect = reference.front();
                if (produced.size() != expect.size() ||
                    std::memcmp(produced.data(), expect.data(), produced.size() * 2U) != 0) {
                    std::size_t first = produced.size() < expect.size() ? produced.size() : expect.size();
                    for (std::size_t index = 0; index < first; ++index) {
                        if (produced[index] != expect[index]) {
                            first = index;
                            break;
                        }
                    }
                    std::cout << "Q5_TIME_MISMATCH shape=" << shape_label << " tokens=" << tokens
                              << " instance=" << instance.name << " words=" << produced.size()
                              << " expect_words=" << expect.size();
                    if (first < produced.size() && first < expect.size()) {
                        std::cout << " first_word=" << first << " got=" << produced[first]
                                  << " expect=" << expect[first];
                    }
                    std::cout << '\n';
                } else {
                    std::cout << "Q5_TIME_IDENTICAL shape=" << shape_label << " tokens=" << tokens
                              << " instance=" << instance.name << '\n';
                }
            }
        }
    }
}

} // namespace

// The diagnostic cell lists of the Q5 decode work: the shipped trunk arm (DirectR2T4W2G8B4) on the
// three mixed-A decode shapes, the dispatch-tuned draft instances on their shape (static K, full
// unroll), the trunk instance again on the draft shape as a cross-check, and the bit-identical
// candidate schedules (static K per shape, optional rows-per-warp) beside the trunk arm.
void time_q5_decode() {
    const std::array down_instances{
        Q5TimeInstance{"r2t4w2g8b4", ops::detail::launch_q5_a16_direct_r2_t4_w2_g8_b4},
        Q5TimeInstance{"rows_m1_t4", ops::detail::launch_q5_a16_rows_m1_t4_k21504},
        Q5TimeInstance{"rows_m2_t4", ops::detail::launch_q5_a16_rows_m2_t4_k21504},
        Q5TimeInstance{"rows_m1_t1", ops::detail::launch_q5_a16_rows_m1_t1_k21504},
        Q5TimeInstance{"rows_m2_t1", ops::detail::launch_q5_a16_rows_m2_t1_k21504},
        Q5TimeInstance{"rows_r4_m2_t4", ops::detail::launch_q5_a16_rows_r4_m2_t4_k21504}};
    const std::array o_instances{
        Q5TimeInstance{"r2t4w2g8b4", ops::detail::launch_q5_a16_direct_r2_t4_w2_g8_b4},
        Q5TimeInstance{"rows_m1_t4", ops::detail::launch_q5_a16_rows_m1_t4_k8192},
        Q5TimeInstance{"rows_m2_t4", ops::detail::launch_q5_a16_rows_m2_t4_k8192},
        Q5TimeInstance{"rows_m2_t1", ops::detail::launch_q5_a16_rows_m2_t1_k8192}};
    const std::array global_o_instances{
        Q5TimeInstance{"r2t4w2g8b4", ops::detail::launch_q5_a16_direct_r2_t4_w2_g8_b4},
        Q5TimeInstance{"rows_m1_t4", ops::detail::launch_q5_a16_rows_m1_t4_k16384},
        Q5TimeInstance{"rows_m2_t4", ops::detail::launch_q5_a16_rows_m2_t4_k16384},
        Q5TimeInstance{"rows_m2_t1", ops::detail::launch_q5_a16_rows_m2_t1_k16384}};
    const std::array draft_t1{Q5TimeInstance{"r1t1w4k17408",
                                             ops::detail::launch_q5_a16_direct_r1_t1_w4_k17408}};
    const std::array draft_t2{Q5TimeInstance{"r1t2w2k17408",
                                             ops::detail::launch_q5_a16_direct_r1_t2_w2_k17408}};
    const std::array draft_t4{Q5TimeInstance{"r1t4w2k17408",
                                             ops::detail::launch_q5_a16_direct_r1_t4_w2_k17408}};
    const std::array cross_t1{
        Q5TimeInstance{"r2t4w2g8b4", ops::detail::launch_q5_a16_direct_r2_t4_w2_g8_b4},
        Q5TimeInstance{"r1t1w4k17408", ops::detail::launch_q5_a16_direct_r1_t1_w4_k17408}};
    constexpr std::array kTrunkTokens{1, 2, 4};
    constexpr std::array kOneToken{1};
    constexpr std::array kTwoTokens{2};
    constexpr std::array kFourTokens{4};

    time_q5_cells("down", 5376, 21504, down_instances, kTrunkTokens, 401u);
    time_q5_cells("o", 5376, 8192, o_instances, kTrunkTokens, 402u);
    time_q5_cells("global_o", 5376, 16384, global_o_instances, kTrunkTokens, 403u);
    time_q5_cells("draft17408_t1", 5120, 17408, draft_t1, kOneToken, 404u);
    time_q5_cells("draft17408_t2", 5120, 17408, draft_t2, kTwoTokens, 404u);
    time_q5_cells("draft17408_t4", 5120, 17408, draft_t4, kFourTokens, 404u);
    time_q5_cells("draft17408_cross_t1", 5120, 17408, cross_t1, kOneToken, 404u);

    // Q4 g32 control at the down shape: the profiled 96-98 %-of-ceiling kernel, timed under the
    // identical rotation harness through the public dispatch (q4g32_gemv_t1 at tokens=1).
    quantized_weight::PackedWeight host = make_q4_g32_fp16_weight(5376, 21504, 405u);
    ColdL2Timer timer(std::move(host), 21504, 1);
    DeviceArena workspace(std::max<std::size_t>(
        ops::linear_workspace_capacity_bytes(QType::Q4_G32_FP16, 5376, 21504,
                                             ops::LinearPolicy::A16Only, 1, 1),
        256));
    const float micros = timer.median_us([&](int copy) {
        ops::linear(timer.activation(copy), timer.weight(copy), timer.output(copy),
                    ops::LinearPolicy::A16Only, workspace, nullptr);
    });
    std::cout << "Q5_TIME shape=q4_control n=5376 k=21504 tokens=1 instance=q4g32_gemv_t1 us="
              << micros << " gb_s=" << timer.bytes() / static_cast<double>(micros) / 1000.0 << '\n';
}

INFERNIX_OP_TEST(linear_gemma4_q5_a16) {
    bool timing = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string_view(argv[i]) == "--time") {
            timing = true;
            continue;
        }
        std::cerr << "usage: linear_gemma4_q5_a16 [--time]\n";
        return 2;
    }
    if (!cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (timing) {
            time_q5_decode();
            return 0;
        }
        const int failures = run_gemma4_q5();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Gemma4 native Q5 A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Gemma4 native Q5 A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
