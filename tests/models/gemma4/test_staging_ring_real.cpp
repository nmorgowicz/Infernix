// The staged sliding ring (ProgramOptions::staged_sliding_ring: 24 pages per layer plus one shared staging ring that
// prompt calls wider than 513 tokens run on) against the full-size ring on a real artifact; no downloads.
// INFERNIX_GEMMA4_ARTIFACT (an MTP artifact when --phase mtp|all runs), missing skips (77); INFERNIX_GEMMA4_HOST_MIB
// sizes the Host tier. Options: --phase plan|native|plain|mtp|all (default all), --ring bf16|fp8|all (default all).
//
// plan   Program::plan_device for chunks 128 / 512 / 1,024 / 2,048 / 4,096: staging pages, persistent pages, KV bytes
//        and the exact saving against the full-size ring; chunks below 1,024 are unchanged. CPU only.
// native The prompt call by call through gemma4_layers on the full-size ring and on the staged ring (chunks 2,048 and
//        1,024; calls of 2,048/1,024, a ragged tail, single tokens, an 8-wide verify, a 500-wide call that fits the
//        staged ring directly and a 600-wide one that stages): after every call the hidden columns, the final logits,
//        every global layer's bytes and every sliding layer's resident rows, FP8 codes and scales included, identical.
// plain  One scripted conversation per configuration, run once on the staged Program and once on the full-size ring:
//        cold prompt, regenerate, edit near the end, unrelated request, return to the first prompt, branch far back,
//        and a cancelled prefill followed by the first prompt again. Every request's generated tokens and the hash of
//        every logits tile the head produced (the observer) must be bit-identical between the two Programs, request
//        by request, wherever the two took the same source (the snapshot restore path and a live resume inside the
//        staged ring's reach do; the live-resume reach of the two rings differs by design and the steps beyond the
//        staged reach are reported, not compared). On the reproducible route (no free-cache-resume) every restore
//        also equals the cold run of the same prompt on the same Program.
// mtp    Greedy MTP with 1, 3 and 7 drafts: tokens, speculative counters (rounds, drafted, accepted, history) and
//        the hash of every verification logits tile bit-identical between the staged and the full-size ring, over a
//        prompt whose prefill wraps the full-size ring, then a regenerate.
#include "prefix_real_support.h"
#include "models/gemma4/program/kv_layout.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <span>
#include <string>

namespace {
using namespace prefix_real;
using namespace infernix;

constexpr std::uint32_t kContext = 7168;
constexpr auto kFp8 = KvCacheStorage::Fp8E4M3Row256;
constexpr auto kBf16 = KvCacheStorage::BFloat16;

std::uint64_t fnv(std::uint64_t h, const void* data, std::size_t bytes) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) { h = (h ^ p[i]) * 1099511628211ULL; }
    return h;
}

struct Route {
    const char* name;
    KvCacheStorage global;
    KvCacheStorage sliding;
};
constexpr Route kBf16Ring{"bf16 ring", kBf16, kBf16};
constexpr Route kFp8Ring{"fp8 ring, fp8 kv", kFp8, kFp8};

struct StageRig {
    DeviceContext device{0};
    std::unique_ptr<g4::Model> model;
    std::unique_ptr<g4::execution::Parameters> parameters;
    std::unique_ptr<g4::Frontend> frontend;
    StageRig(const char* artifact, bool mtp) {
        models::LoadOptions load;
        if (mtp) { load.speculative = SpeculativeBackend::Mtp; }
        model = g4::load_model(artifact, load, device);
        parameters = std::make_unique<g4::execution::Parameters>(*model, device.stream);
        frontend = std::make_unique<g4::Frontend>(model->resources(), g4::FrontendOptions{.max_context = kContext});
    }
    g4::ProgramOptions options(const Route& route, std::uint32_t chunk, bool graphs, std::uint32_t drafts, bool staged,
                               bool free_resume, std::uint64_t host) const {
        g4::ProgramOptions o;
        o.max_context = kContext;
        o.prefill_chunk = chunk;
        o.kv_cache = route.global;
        o.sliding_kv_cache = route.sliding;
        o.draft_tokens = drafts;
        o.prefix_cache = host != 0;
        o.host_bytes = host;
        o.free_cache_resume = free_resume;
        o.use_cuda_graph = graphs;
        o.staged_sliding_ring = staged;
        return o;
    }
};

// ---- plan ----------------------------------------------------------------------------------------------------------

void plan_checks(StageRig& rig) {
    const auto& c = rig.model->config().text;
    std::uint32_t sliding_layers = 0;
    for (const auto kind : c.layer_types) { sliding_layers += kind == g4::LayerKind::Global ? 0U : 1U; }
    for (const Route* route : {&kBf16Ring, &kFp8Ring}) {
        for (const std::uint32_t chunk : {128U, 512U, 1024U, 2048U, 4096U}) {
            auto staged_options = rig.options(*route, chunk, true, 0, true, true, 0);
            auto full_options = rig.options(*route, chunk, true, 0, false, true, 0);
            const auto staged = g4::Program::plan_device(*rig.model, staged_options, c.vocab_size);
            const auto full = g4::Program::plan_device(*rig.model, full_options, c.vocab_size);
            const std::uint32_t full_pages = (chunk + c.sliding_window - 1 + 63) / 64;
            const auto layout = g4::kv_layer_layout(staged_options.kv_layout(), false, c.head_dim, c.num_key_value_heads);
            const std::uint64_t layer_page = layout.page_bytes();
            const std::string label = std::string(route->name) + " chunk " + std::to_string(chunk);
            if (chunk < 1024) {
                expect(staged.staging_pages == 0 && staged.ring_pages == full_pages && staged.kv == full.kv &&
                           staged.kv_max == full.kv_max && staged.fixed_bytes() == full.fixed_bytes(),
                       label + ": below chunk 1,024 the plan is the full-size ring's (pages " +
                           std::to_string(staged.ring_pages) + ")");
                continue;
            }
            const std::int64_t expected_saving =
                (static_cast<std::int64_t>(sliding_layers) * (full_pages - 24) - full_pages) *
                    static_cast<std::int64_t>(layer_page) -
                4LL * staged.kv_pages;
            const std::int64_t saving = static_cast<std::int64_t>(full.kv) - static_cast<std::int64_t>(staged.kv);
            expect(staged.ring_pages == 24 && staged.staging_pages == full_pages && full.staging_pages == 0 &&
                       full.ring_pages == full_pages,
                   label + ": 24 persistent pages and a staging ring of " + std::to_string(staged.staging_pages) +
                       " pages (full-size ring " + std::to_string(full.ring_pages) + ")");
            expect(saving == expected_saving,
                   label + ": planned KV saving " + std::to_string(saving) + " B (" +
                       std::to_string(static_cast<double>(saving) / 1048576.0) + " MiB) equals the arithmetic " +
                       std::to_string(expected_saving) + " B");
            expect(staged.fixed_bytes() + static_cast<std::uint64_t>(saving) == full.fixed_bytes(),
                   label + ": the whole device reservation moves by exactly the KV saving");
            expect(staged.curve.bytes_per_additional_main_page_group == full.curve.bytes_per_additional_main_page_group + 4,
                   label + ": the staging ring's page table is one more 4-byte entry per page group");
        }
    }
}

// ---- scripted conversation -----------------------------------------------------------------------------------------

struct Step {
    std::string name;
    Tokens tokens;
    std::uint64_t reused = 0;
    std::vector<std::uint64_t> logits; // hash of every observed tile, in order
    SpeculativeStats speculative;
};
using Transcript = std::vector<Step>;

struct Arm {
    StageRig& rig;
    std::vector<std::uint64_t> observed;
    std::unique_ptr<g4::Program> program;
    Arm(StageRig& r, g4::ProgramOptions options) : rig(r) {
        options.logits_observer = [this](std::uint32_t first, std::uint32_t width, std::span<const std::uint16_t> values) {
            std::uint64_t h = fnv(1469598103934665603ULL, &first, sizeof(first));
            h = fnv(h, &width, sizeof(width));
            observed.push_back(fnv(h, values.data(), values.size_bytes()));
        };
        program = std::make_unique<g4::Program>(*rig.parameters, rig.device, options);
    }

    // The generated tokens of one request (speculative rounds commit every licensed token), the way the Engine drives it.
    Step generate(const std::string& name, const Tokens& prompt, std::uint32_t outputs, bool reuse) {
        Step step;
        step.name = name;
        const std::size_t before = observed.size();
        auto started = start(*program, *rig.frontend, prompt, outputs, reuse);
        require(started.sequence.has_value(), name + ": the binding did not publish");
        step.reused = started.reused;
        const auto sequence = *started.sequence;
        auto commit = [&](g4::PendingBatch&& batch, bool first) {
            require(batch.row_failure(0) == rt::PendingRowFailure::None, name + ": logits failed");
            const auto n = batch.row_counts().empty() ? 1U : static_cast<std::uint32_t>(batch.row_counts()[0]);
            const auto take = std::min<std::uint32_t>(n, outputs - static_cast<std::uint32_t>(step.tokens.size()));
            step.tokens.insert(step.tokens.end(), batch.tokens().begin(), batch.tokens().begin() + take);
            const bool terminal = step.tokens.size() == outputs;
            rt::CommitDecision d;
            d.accepted_tokens = take;
            d.terminal = terminal;
            const std::array decisions{d};
            (void)program->commit(std::move(batch), decisions);
            (void)first;
            return terminal;
        };
        bool terminal = commit(first_token(*program, sequence), true);
        const std::array sequences{sequence};
        while (!terminal) {
            const std::array budgets{rt::RoundBudget{outputs - static_cast<std::uint32_t>(step.tokens.size())}};
            terminal = commit(program->decode(sequences, budgets), false);
        }
        auto finished = program->finish(sequence);
        require(finished.status == rt::ConsumeStatus::Consumed, name + ": terminal sequence did not finish");
        step.speculative = std::move(finished.speculative);
        step.logits.assign(observed.begin() + static_cast<std::ptrdiff_t>(before), observed.end());
        return step;
    }

    // A prefill that is started, advanced one call and abandoned; the lane must be free and the next request normal.
    void cancel_mid_prefill(const Tokens& prompt) {
        auto started = start(*program, *rig.frontend, prompt, 8, true);
        require(started.sequence.has_value(), "the cancelled request did not bind");
        auto progress = program->advance_prefill(*started.sequence);
        require(!progress.complete, "the cancelled request finished its prefill in one call");
        const auto aborted = program->abort(*started.sequence);
        require(aborted.status == rt::ConsumeStatus::Consumed && !aborted.device_error, "abort refused");
    }
};

bool same_speculative(const SpeculativeStats& a, const SpeculativeStats& b) {
    if (a.backend != b.backend || a.enabled != b.enabled || a.draft_window != b.draft_window || a.rounds != b.rounds ||
        a.drafted_tokens != b.drafted_tokens || a.accepted_tokens != b.accepted_tokens ||
        a.fallback_steps != b.fallback_steps || a.accepted_per_position != b.accepted_per_position ||
        a.rounds_by_draft_length != b.rounds_by_draft_length || a.round_steps.size() != b.round_steps.size()) {
        return false;
    }
    for (std::size_t i = 0; i < a.round_steps.size(); ++i) {
        if (a.round_steps[i].drafted != b.round_steps[i].drafted || a.round_steps[i].accepted != b.round_steps[i].accepted) {
            return false;
        }
    }
    return true;
}

std::string first_difference(const Step& a, const Step& b) {
    if (a.tokens != b.tokens) { return "tokens: " + difference(a.tokens, b.tokens); }
    if (a.logits.size() != b.logits.size()) {
        return "logits tiles " + std::to_string(a.logits.size()) + " vs " + std::to_string(b.logits.size());
    }
    for (std::size_t i = 0; i < a.logits.size(); ++i) {
        if (a.logits[i] != b.logits[i]) { return "logits tile " + std::to_string(i); }
    }
    return "identical";
}

void conversation(Arm& arm, Transcript& t, bool reproducible_equals) {
    const Tokens p1 = tokens(5200, 11);
    const Tokens edit_near = concat(Tokens(p1.begin(), p1.begin() + 5100), tokens(300, 7));
    const Tokens unrelated = tokens(1000, 3);
    const Tokens branch_far = concat(Tokens(p1.begin(), p1.begin() + 2600), tokens(1500, 5));
    t.push_back(arm.generate("cold", p1, 24, true));
    t.push_back(arm.generate("regenerate", p1, 24, true));
    t.push_back(arm.generate("edit near the end", edit_near, 24, true));
    t.push_back(arm.generate("unrelated", unrelated, 8, true));
    t.push_back(arm.generate("return to the first prompt", p1, 24, true));
    t.push_back(arm.generate("branch far back", branch_far, 16, true));
    arm.cancel_mid_prefill(tokens(6000, 31));
    t.push_back(arm.generate("after a cancelled prefill", p1, 24, true));
    if (reproducible_equals) {
        // Every restore equals the cold prompt (the reproducible route's contract) on this Program.
        for (std::size_t i : {1U, 4U, 6U}) {
            expect((t[i].tokens == t[0].tokens && t[i].logits == t[0].logits) || t[i].reused == 0,
                   "reproducible route: " + t[i].name + " (restored " + std::to_string(t[i].reused) +
                       " tokens) equals the cold run");
        }
    }
}

void plain_cell(StageRig& rig, const Route& route, std::uint32_t chunk, bool graphs, bool free_resume) {
    const std::string label = std::string(route.name) + ", chunk " + std::to_string(chunk) + ", graphs " +
                              (graphs ? "on" : "off") + ", " + (free_resume ? "free-cache-resume" : "reproducible");
    std::printf("== plain: %s\n", label.c_str());
    const auto host = host_bytes_from_env();
    Transcript staged, full;
    {
        Arm arm(rig, rig.options(route, chunk, graphs, 0, true, free_resume, host));
        expect(arm.program->device_plan().staging_pages != 0, label + ": the staged Program has a staging ring (" +
                   std::to_string(arm.program->device_plan().ring_pages) + " + " +
                   std::to_string(arm.program->device_plan().staging_pages) + " pages)");
        conversation(arm, staged, !free_resume);
    }
    {
        Arm arm(rig, rig.options(route, chunk, graphs, 0, false, free_resume, host));
        expect(arm.program->device_plan().staging_pages == 0, label + ": the reference Program has the full-size ring");
        conversation(arm, full, !free_resume);
    }
    for (std::size_t i = 0; i < staged.size(); ++i) {
        const bool same_source = staged[i].reused == full[i].reused;
        if (!same_source) {
            std::printf("  note  %s: %s took different sources (staged restored %llu, full-size ring %llu): not compared\n",
                        label.c_str(), staged[i].name.c_str(), static_cast<unsigned long long>(staged[i].reused),
                        static_cast<unsigned long long>(full[i].reused));
            continue;
        }
        const auto difference_text = first_difference(staged[i], full[i]);
        expect(difference_text == "identical",
               label + ": " + staged[i].name + " (restored " + std::to_string(staged[i].reused) + ", " +
                   std::to_string(staged[i].logits.size()) + " logits tiles) bit-identical to the full-size ring: " +
                   difference_text);
    }
}

void mtp_cell(StageRig& rig, const Route& route, std::uint32_t chunk, bool graphs, std::uint32_t drafts) {
    const std::string label = std::string(route.name) + ", chunk " + std::to_string(chunk) + ", graphs " +
                              (graphs ? "on" : "off") + ", " + std::to_string(drafts) + " drafts";
    std::printf("== mtp: %s\n", label.c_str());
    const auto host = host_bytes_from_env();
    Transcript staged, full;
    for (const bool use_staged : {true, false}) {
        Arm arm(rig, rig.options(route, chunk, graphs, drafts, use_staged, true, host));
        auto& t = use_staged ? staged : full;
        const Tokens p = tokens(4400, 41);
        t.push_back(arm.generate("cold", p, 64, true));
        t.push_back(arm.generate("regenerate", p, 64, true));
        t.push_back(arm.generate("cold, prefix reuse off", p, 32, false));
    }
    for (std::size_t i = 0; i < staged.size(); ++i) {
        const auto text = first_difference(staged[i], full[i]);
        expect(text == "identical" && same_speculative(staged[i].speculative, full[i].speculative) &&
                   staged[i].reused == full[i].reused,
               label + ": " + staged[i].name + " (rounds " + std::to_string(staged[i].speculative.rounds) + ", drafted " +
                   std::to_string(staged[i].speculative.drafted_tokens) + ", accepted " +
                   std::to_string(staged[i].speculative.accepted_tokens) + ", " +
                   std::to_string(staged[i].logits.size()) + " logits tiles) bit-identical to the full-size ring: " + text);
        expect(staged[i].speculative.accepted_tokens > 0, label + ": the run accepted drafts");
    }
}

// ---- native forward ------------------------------------------------------------------------------------------------

// One native cache set over the real layers: global layers on contiguous pages; sliding layers on a ring of `ring_pages`
// pages (block table logical page % ring_pages), and, when `staging_pages` is set, the shared staging ring the wide calls
// run on, exactly the Program's layout. Every byte stays inspectable.
struct CacheSet {
    std::uint32_t ring_pages, staging_pages, logical_pages;
    DeviceBuffer global_table_memory, ring_table_memory, staging_table_memory;
    Tensor global_table, ring_table, staging_table;
    std::vector<DeviceBuffer> keys, values, key_scales, value_scales;
    std::vector<Tensor> key_views, value_views, key_scale_views, value_scale_views;
    std::vector<g4::Gemma4CacheLayer> layers;
    DeviceBuffer staging_keys, staging_values, staging_key_scales, staging_value_scales;
    Tensor staging_key_view, staging_value_view, staging_key_scale_view, staging_value_scale_view;
    g4::Gemma4RingStage stage;

    CacheSet(const g4::execution::Parameters& p, const Route& route, std::uint32_t ring_pages_, std::uint32_t staging_pages_)
        : ring_pages(ring_pages_), staging_pages(staging_pages_), logical_pages(kContext / 64) {
        const auto upload = [&](DeviceBuffer& memory, Tensor& table, std::uint32_t modulo) {
            std::vector<std::int32_t> entries(logical_pages);
            std::iota(entries.begin(), entries.end(), 0);
            if (modulo != 0) { for (auto& e : entries) { e %= static_cast<std::int32_t>(modulo); } }
            memory = DeviceBuffer(entries.size() * sizeof(std::int32_t));
            memory.copy_from_host(entries.data(), memory.bytes);
            table = Tensor(memory.p, DType::I32, {static_cast<int>(logical_pages)});
        };
        upload(global_table_memory, global_table, 0);
        upload(ring_table_memory, ring_table, ring_pages);
        if (staging_pages != 0) { upload(staging_table_memory, staging_table, staging_pages); }
        const auto count = p.layers.size();
        for (auto* v : {&keys, &values, &key_scales, &value_scales}) { v->reserve(count); }
        for (auto* v : {&key_views, &value_views, &key_scale_views, &value_scale_views}) { v->reserve(count); }
        layers.reserve(count);
        const g4::KvCacheLayout request{route.global, g4::KvStorageModel::KeyEqualsValue, route.sliding};
        bool staging_made = false;
        for (std::size_t i = 0; i < count; ++i) {
            const auto& layer = p.layers[i];
            const auto layout = g4::kv_layer_layout(request, layer.global_layer, layer.head_dim, layer.kv_heads);
            const std::uint32_t pages = layer.global_layer ? logical_pages : ring_pages;
            const std::initializer_list<std::int32_t> shape{layer.head_dim, 64, layer.kv_heads, static_cast<int>(pages)};
            const std::initializer_list<std::int32_t> scale_shape{layout.scale_groups, 64, layer.kv_heads, static_cast<int>(pages)};
            keys.emplace_back(layout.code_page_bytes * pages); keys.back().fill(0);
            values.emplace_back(layout.code_page_bytes * pages); values.back().fill(0);
            key_views.emplace_back(keys.back().p, layout.code_dtype, shape);
            value_views.emplace_back(values.back().p, layout.code_dtype, shape);
            key_scales.emplace_back(); value_scales.emplace_back(); key_scale_views.emplace_back(); value_scale_views.emplace_back();
            if (layout.scale_groups != 0) {
                key_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); key_scales.back().fill(0);
                value_scales.back() = DeviceBuffer(layout.scale_page_bytes * pages); value_scales.back().fill(0);
                key_scale_views.back() = Tensor(key_scales.back().p, DType::FP16, scale_shape);
                value_scale_views.back() = Tensor(value_scales.back().p, DType::FP16, scale_shape);
            }
            const bool scaled = layout.scale_groups != 0;
            layers.push_back({.format = layout.format, .k_pages = &key_views.back(), .v_pages = &value_views.back(),
                              .k_scales = scaled ? &key_scale_views.back() : nullptr,
                              .v_scales = scaled ? &value_scale_views.back() : nullptr,
                              .block_table = layer.global_layer ? &global_table : &ring_table});
            if (!layer.global_layer && staging_pages != 0 && !staging_made) {
                staging_made = true;
                const std::initializer_list<std::int32_t> st_shape{layer.head_dim, 64, layer.kv_heads, static_cast<int>(staging_pages)};
                const std::initializer_list<std::int32_t> st_scale_shape{layout.scale_groups, 64, layer.kv_heads, static_cast<int>(staging_pages)};
                staging_keys = DeviceBuffer(layout.code_page_bytes * staging_pages); staging_keys.fill(0);
                staging_values = DeviceBuffer(layout.code_page_bytes * staging_pages); staging_values.fill(0);
                staging_key_view = Tensor(staging_keys.p, layout.code_dtype, st_shape);
                staging_value_view = Tensor(staging_values.p, layout.code_dtype, st_shape);
                if (scaled) {
                    staging_key_scales = DeviceBuffer(layout.scale_page_bytes * staging_pages); staging_key_scales.fill(0);
                    staging_value_scales = DeviceBuffer(layout.scale_page_bytes * staging_pages); staging_value_scales.fill(0);
                    staging_key_scale_view = Tensor(staging_key_scales.p, DType::FP16, st_scale_shape);
                    staging_value_scale_view = Tensor(staging_value_scales.p, DType::FP16, st_scale_shape);
                }
                stage.staging = {.format = layout.format, .k_pages = &staging_key_view, .v_pages = &staging_value_view,
                                 .k_scales = scaled ? &staging_key_scale_view : nullptr,
                                 .v_scales = scaled ? &staging_value_scale_view : nullptr, .block_table = &staging_table};
                stage.persistent_rows = static_cast<int>(ring_pages * 64);
            }
        }
    }
    const g4::Gemma4RingStage* staged() const { return staging_pages != 0 ? &stage : nullptr; }
};

// The rows [lo, hi) of one sliding layer's four planes, in position order: K codes, V codes, K scales, V scales.
std::vector<unsigned char> ring_rows(const CacheSet& set, std::size_t layer, const g4::execution::Parameters& p,
                                     std::uint32_t lo, std::uint32_t hi) {
    std::vector<unsigned char> out;
    const int heads = p.layers[layer].kv_heads;
    for (const DeviceBuffer* buffer : {&set.keys[layer], &set.values[layer], &set.key_scales[layer], &set.value_scales[layer]}) {
        if (buffer->bytes == 0) { continue; }
        std::vector<unsigned char> bytes(buffer->bytes);
        buffer->copy_to_host(bytes.data(), bytes.size());
        const std::size_t row_bytes = buffer->bytes / (static_cast<std::size_t>(set.ring_pages) * 64 * heads);
        for (std::uint32_t pos = lo; pos < hi; ++pos) {
            const std::size_t page = (pos / 64) % set.ring_pages;
            for (int h = 0; h < heads; ++h) {
                const std::size_t offset = row_bytes * (pos % 64 + 64 * (static_cast<std::size_t>(h) + heads * page));
                out.insert(out.end(), bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                           bytes.begin() + static_cast<std::ptrdiff_t>(offset + row_bytes));
            }
        }
    }
    return out;
}

// The same prompt call by call through gemma4_layers on the full-size ring and on the staged ring: after every call the
// hidden columns, every global layer's bytes and every sliding layer's resident rows (codes and scales) must be
// identical, and the final logits too.
void native_cell(StageRig& rig, const Route& route, std::uint32_t chunk) {
    const std::string label = std::string(route.name) + ", chunk " + std::to_string(chunk);
    std::printf("== native: %s\n", label.c_str());
    const auto& p = *rig.parameters;
    const auto& c = rig.model->config().text;
    const std::uint32_t full_pages = (chunk + c.sliding_window - 1 + 63) / 64;
    CacheSet full(p, route, full_pages, 0);
    CacheSet staged(p, route, 24, full_pages);
    std::size_t workspace_bytes = g4::gemma4_logits_workspace_bytes(p.output_head, 1);
    for (const auto& layer : p.layers) { workspace_bytes = std::max(workspace_bytes, g4::gemma4_layer_workspace_bytes(layer, chunk)); }
    WorkspaceArena workspace(workspace_bytes);
    ops::Fp8RowCublasContext gemm(rig.device.stream);
    DeviceBuffer input_memory(4ULL * chunk), position_memory(4ULL * chunk);
    DeviceBuffer hidden_memory[2] = {DeviceBuffer(2ULL * c.hidden_size * chunk), DeviceBuffer(2ULL * c.hidden_size * chunk)};
    DeviceBuffer logits_memory[2] = {DeviceBuffer(2ULL * c.vocab_size), DeviceBuffer(2ULL * c.vocab_size)};
    const Tokens prompt = tokens(kContext, 101);
    // Calls: the prompt in chunks, then single-token decode steps, an MTP-width verify, a narrow call that fits the
    // staged ring directly (500 <= 513) and a wide call that does not (600).
    std::vector<std::pair<std::uint32_t, std::uint32_t>> calls;
    std::uint32_t at = 0;
    for (; at + chunk <= 5200; at += chunk) { calls.push_back({at, chunk}); }
    if (at < 5200) { calls.push_back({at, 5200 - at}); at = 5200; }
    for (const std::uint32_t width : {1U, 1U, 8U, 500U, 600U, 1U}) { calls.push_back({at, width}); at += width; }
    std::size_t step = 0;
    for (const auto& [first, width] : calls) {
        std::vector<std::int32_t> positions(width);
        std::iota(positions.begin(), positions.end(), static_cast<std::int32_t>(first));
        input_memory.copy_from_host(prompt.data() + first, 4ULL * width);
        position_memory.copy_from_host(positions.data(), 4ULL * width);
        Tensor input(input_memory.p, DType::I32, {static_cast<int>(width)});
        Tensor pos(position_memory.p, DType::I32, {static_cast<int>(width)});
        std::vector<std::uint16_t> hidden[2], logits[2];
        for (int arm = 0; arm < 2; ++arm) {
            CacheSet& set = arm == 0 ? full : staged;
            Tensor hidden_view(hidden_memory[arm].p, DType::BF16, {static_cast<int>(c.hidden_size), static_cast<int>(width)});
            g4::gemma4_layers(input, hidden_view, p.embedding, p.layers, set.layers, pos, gemm, workspace, rig.device.stream,
                              true, set.staged());
            Tensor last = hidden_view.slice(1, static_cast<int>(width) - 1, 1);
            Tensor logit_view(logits_memory[arm].p, DType::BF16, {static_cast<int>(c.vocab_size), 1});
            g4::gemma4_logits(last, logit_view, p.final_norm, p.output_head, c.final_logit_softcapping, c.rms_norm_eps, gemm,
                              workspace, rig.device.stream);
            rig.device.synchronize();
            hidden[arm].resize(static_cast<std::size_t>(c.hidden_size) * width);
            hidden_memory[arm].copy_to_host(hidden[arm].data(), hidden[arm].size() * 2);
            logits[arm].resize(c.vocab_size);
            logits_memory[arm].copy_to_host(logits[arm].data(), logits[arm].size() * 2);
        }
        const std::uint32_t fed = first + width;
        const std::string where = label + ": call " + std::to_string(step++) + " [" + std::to_string(first) + ", " +
                                  std::to_string(fed) + ")";
        expect(hidden[0] == hidden[1], where + ": hidden columns of all " + std::to_string(width) + " tokens bit-identical");
        expect(logits[0] == logits[1], where + ": final-column logits bit-identical");
        bool global_ok = true, sliding_ok = true;
        std::uint64_t rows_checked = 0;
        const std::uint32_t lo = fed > 1536 ? fed - 1536 : 0;
        for (std::size_t layer = 0; layer < p.layers.size(); ++layer) {
            if (p.layers[layer].global_layer) {
                for (auto pair : {std::pair{&full.keys[layer], &staged.keys[layer]},
                                  std::pair{&full.values[layer], &staged.values[layer]},
                                  std::pair{&full.key_scales[layer], &staged.key_scales[layer]},
                                  std::pair{&full.value_scales[layer], &staged.value_scales[layer]}}) {
                    if (pair.first->bytes == 0) { continue; }
                    std::vector<unsigned char> a(pair.first->bytes), b(pair.second->bytes);
                    pair.first->copy_to_host(a.data(), a.size());
                    pair.second->copy_to_host(b.data(), b.size());
                    global_ok = global_ok && a == b;
                }
            } else {
                const auto a = ring_rows(full, layer, p, lo, fed);
                const auto b = ring_rows(staged, layer, p, lo, fed);
                sliding_ok = sliding_ok && a == b;
                rows_checked += a.size();
            }
        }
        expect(global_ok, where + ": every global layer's K/V bytes identical");
        expect(sliding_ok, where + ": every sliding layer's resident rows [" + std::to_string(lo) + ", " + std::to_string(fed) +
                               ") identical, codes and scales (" + std::to_string(rows_checked >> 20) + " MiB compared)");
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set");
        return 77;
    }
    std::string phase = "all", ring = "all";
    for (int i = 1; i + 1 < argc; i += 2) {
        if (std::string(argv[i]) == "--phase") { phase = argv[i + 1]; }
        else if (std::string(argv[i]) == "--ring") { ring = argv[i + 1]; }
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::vector<const Route*> routes;
    if (ring == "bf16" || ring == "all") { routes.push_back(&kBf16Ring); }
    if (ring == "fp8" || ring == "all") { routes.push_back(&kFp8Ring); }
    try {
        if (phase == "plan" || phase == "native" || phase == "all") {
            StageRig rig(artifact, false);
            if (phase != "native") { plan_checks(rig); }
            if (phase != "plan") {
                for (const Route* route : routes) {
                    native_cell(rig, *route, 2048);
                    native_cell(rig, *route, 1024);
                }
            }
        }
        if (phase == "plain" || phase == "all") {
            StageRig rig(artifact, false);
            for (const Route* route : routes) {
                plain_cell(rig, *route, 2048, false, true);
                plain_cell(rig, *route, 1024, true, true);
                if (route == &kBf16Ring) {
                    plain_cell(rig, *route, 2048, false, false);
                    plain_cell(rig, *route, 1024, true, false);
                }
            }
        }
        if (phase == "mtp" || phase == "all") {
            StageRig rig(artifact, true);
            for (const Route* route : routes) {
                for (const std::uint32_t drafts : {1U, 3U, 7U}) {
                    mtp_cell(rig, *route, 2048, false, drafts);
                    if (drafts == 3U) { mtp_cell(rig, *route, 1024, true, drafts); }
                }
            }
        }
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    std::puts(failures == 0 ? "all staging ring checks passed" : "staging ring checks failed");
    return failures == 0 ? 0 : 1;
}
