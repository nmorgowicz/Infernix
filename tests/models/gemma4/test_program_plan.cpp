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
constexpr std::uint64_t kRing = 32 * kMiB;

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

// The staged sliding ring (24 persistent pages per layer plus one shared staging ring, from a chunk of 1,024): its plan
// against the full-size ring's, exactly, and the unchanged plan below that chunk.
void check_staging(const g4::Model& model, KvCacheStorage ring) {
    const auto& c = model.config().text;
    std::uint32_t sliding_layers = 0;
    for (const auto kind : c.layer_types) { sliding_layers += kind == g4::LayerKind::Global ? 0U : 1U; }
    for (const std::uint32_t chunk : {512U, 1024U, 2048U}) {
        auto staged_options = route(false, ring, true);
        staged_options.max_context   = 4096;
        staged_options.prefill_chunk = chunk;
        auto full_options = staged_options;
        full_options.staged_sliding_ring = false;
        const auto staged = g4::Program::plan_device(model, staged_options, c.vocab_size);
        const auto full   = g4::Program::plan_device(model, full_options, c.vocab_size);
        const std::uint32_t full_pages = (chunk + c.sliding_window - 1 + 63) / 64;
        const std::string name = "staging/" + std::string(ring == KvCacheStorage::Fp8E4M3Row256 ? "fp8-ring" : "bf16-ring") +
                                 "/chunk " + std::to_string(chunk);
        check(full.staging_pages == 0 && full.ring_pages == full_pages, name + ": the reference plan has the full-size ring");
        if (chunk < 1024) {
            check(staged.staging_pages == 0 && staged.ring_pages == full_pages && staged.kv == full.kv &&
                      staged.fixed_bytes() == full.fixed_bytes(),
                  name + ": below a chunk of 1,024 the plan is unchanged");
            continue;
        }
        const auto layout = g4::kv_layer_layout(staged_options.kv_layout(), false, c.head_dim, c.num_key_value_heads);
        const std::int64_t saving =
            (static_cast<std::int64_t>(sliding_layers) * (full_pages - 24) - full_pages) *
                static_cast<std::int64_t>(layout.page_bytes()) -
            4LL * staged.kv_pages;
        check(staged.ring_pages == 24 && staged.staging_pages == full_pages, name + ": 24 + " + std::to_string(full_pages) + " pages");
        check(static_cast<std::int64_t>(full.kv) - static_cast<std::int64_t>(staged.kv) == saving,
              name + ": the planned KV saving is the arithmetic " + std::to_string(saving) + " B");
        check(staged.curve.minimum_device_reservation_bytes ==
                  staged.fixed_bytes() - staged.curve.bytes_per_additional_main_page_group *
                                             (staged.kv_pages - staged.curve.minimum_main_page_groups),
              name + ": the capacity curve's minimum reservation is the fixed bytes less the extra page groups");
        check(staged.curve.bytes_per_additional_main_page_group == full.curve.bytes_per_additional_main_page_group + 4,
              name + ": one more 4-byte table entry per page group");
    }
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
    // The staged ring needs a full-size ring above 24 pages: a fixture whose positions reach 4,096.
    ModelFixture long_fixture(false, infernix::QType::BF16, 256, 512);
    auto& long_config = long_fixture.file.root["components"]["text"]["config"];
    long_config["max_position_embeddings"] = 4096;
    long_config["sliding_window"] = 1024;
    long_fixture.file.write();
    infernix::artifact::Reader long_reader(long_fixture.file.entry);
    const auto long_model = g4::plan_model(g4::plan_load(long_reader));
    check_staging(*long_model, KvCacheStorage::BFloat16);
    check_staging(*long_model, KvCacheStorage::Fp8E4M3Row256);
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
