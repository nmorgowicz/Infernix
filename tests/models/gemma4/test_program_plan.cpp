// Program::plan_device's device reservation, on a geometry-only Gemma 4 model with the real FP8 head dimensions
// (sliding 256, global 512): the reservation the KV capacity is sized against holds the FP8 sliding ring's
// first-launch allowance on every FP8-ring route, with or without CUDA graphs, and nothing of it on a BF16 ring.
// CPU only: no device, no artifact.

#include "artifact/reader.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/program.h"
#include "tiny_model_fixture.h"

#include <cstdint>
#include <iostream>
#include <string>

namespace {

namespace g4 = infernix::models::gemma4;
using infernix::KvCacheStorage;
using infernix::test::gemma4_fixture::ModelFixture;

constexpr std::uint64_t kMiB = 1ULL << 20;
constexpr std::uint64_t kGraphs = 64 * kMiB;
constexpr std::uint64_t kRing = 128 * kMiB;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

g4::ProgramOptions route(bool key_only, KvCacheStorage ring, bool graphs) {
    g4::ProgramOptions options;
    options.max_context      = 64;
    options.prefill_chunk    = 16;
    options.kv_cache         = KvCacheStorage::Fp8E4M3Row256;
    options.sliding_kv_cache = ring;
    options.global_kv_storage = key_only ? g4::KvStorageModel::KeyOnly : g4::KvStorageModel::KeyEqualsValue;
    options.use_cuda_graph   = graphs;
    return options;
}

// What the plan reserves beyond the KV payload, workspace, state, I/O and prepared scales, read off the capacity
// curve (the reservation the KV sizing uses) and checked against the plan's own total.
std::uint64_t allowances(const g4::ProgramDevicePlan& plan) {
    const std::uint64_t payload = plan.kv + plan.workspace + plan.state + plan.io + plan.prepared_row_scales;
    check(plan.curve.minimum_device_reservation_bytes == plan.fixed_bytes(),
          "the capacity curve's reservation differs from the plan's fixed bytes");
    return plan.curve.minimum_device_reservation_bytes - payload;
}

void check_route(const g4::Model& model, bool key_only, KvCacheStorage ring) {
    const bool fp8_ring = ring == KvCacheStorage::Fp8E4M3Row256;
    const std::string name = std::string(key_only ? "key-only" : "stored") + (fp8_ring ? "/fp8-ring" : "/bf16-ring");
    const auto on  = g4::Program::plan_device(model, route(key_only, ring, true), model.config().text.vocab_size);
    const auto off = g4::Program::plan_device(model, route(key_only, ring, false), model.config().text.vocab_size);

    check(on.graph_bound > 0 && off.graph_bound == 0, name + ": graph executables are planned only with CUDA graphs");
    check(on.kv == off.kv && on.workspace == off.workspace && on.io == off.io,
          name + ": CUDA graphs change the payload the plan reserves");
    // With graphs the ring's allowance rides in the graph allowance (the warm launches fall in its window);
    // without graphs it stands alone. The reservation holds it either way, exactly once.
    check(on.graph_allowance == kGraphs + (fp8_ring ? kRing : 0) && on.ring_launch_allowance == 0,
          name + ": graphs on, allowance " + std::to_string(on.graph_allowance >> 20) + " MiB + ring " +
              std::to_string(on.ring_launch_allowance >> 20) + " MiB");
    check(off.graph_allowance == 0 && off.ring_launch_allowance == (fp8_ring ? kRing : 0),
          name + ": graphs off, allowance " + std::to_string(off.graph_allowance >> 20) + " MiB + ring " +
              std::to_string(off.ring_launch_allowance >> 20) + " MiB");
    check(allowances(on) == kGraphs + (fp8_ring ? kRing : 0),
          name + ": graphs-on reservation beyond the payload is " + std::to_string(allowances(on) >> 20) + " MiB");
    check(allowances(off) == (fp8_ring ? kRing : 0),
          name + ": graphs-off reservation beyond the payload is " + std::to_string(allowances(off) >> 20) + " MiB");
    check(on.curve.minimum_device_reservation_bytes - off.curve.minimum_device_reservation_bytes == kGraphs,
          name + ": the graphs add exactly the graph allowance to the reservation (the ring term is not counted twice)");
}

void run() {
    for (const bool key_only : {false, true}) {
        ModelFixture fixture(false, infernix::QType::BF16, 256, 512);
        if (key_only) {
            auto& config = fixture.file.root["components"]["text"]["config"];
            config["prope_pairs"]            = 64;
            config["prope_rotary_dim"]       = 128;
            config["global_k_norm_scalar"]   = {{"1", 0.5}};
        }
        fixture.file.write();
        infernix::artifact::Reader reader(fixture.file.entry);
        const auto model = g4::plan_model(g4::plan_load(reader));
        check_route(*model, key_only, KvCacheStorage::BFloat16);
        check_route(*model, key_only, KvCacheStorage::Fp8E4M3Row256);
    }
}

} // namespace

int main() {
    try {
        run();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4 program plan allowance\n";
    return failures ? 1 : 0;
}
