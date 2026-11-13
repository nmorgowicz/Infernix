// Real BF16- and FP8-KV Gemma transactions, including damage after a sliding-ring wrap. No downloads.
// INFERNIX_GEMMA4_ARTIFACT selects an operator-owned artifact; absent means skip (77).
#include "core/device.h"
#include "infernix/engine.h"
#include "models/gemma4/execution/parameters.h"
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/load.h"
#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/program.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using namespace infernix;
namespace g4 = models::gemma4;
using Tokens = std::vector<TokenId>;
void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}
Tokens tokens(std::size_t count, std::uint32_t seed) {
    Tokens result;
    for (std::size_t i = 0; i < count; ++i) {
        seed = seed * 1664525U + 1013904223U;
        result.push_back(static_cast<TokenId>(1000U + (seed >> 8U) % 30000U));
    }
    return result;
}
g4::SequenceHandle bind(g4::Program& program, const g4::RequestBasePlan& plan,
                        g4::ResumeState* resume = nullptr) {
    require(static_cast<bool>(program.start_binding(plan, runtime::LaneId{0}, {}, resume)), "binding refused");
    auto progress = program.poll_context({});
    require(progress.published && progress.sequence.has_value(), "binding did not publish");
    return *progress.sequence;
}
g4::PendingBatch prefill(g4::Program& program, g4::SequenceHandle sequence) {
    for (;;) {
        auto progress = program.advance_prefill(sequence);
        if (progress.pending) { return std::move(*progress.pending); }
        require(!progress.complete, "prefill completed without its first-token transaction");
    }
}
Tokens complete(g4::Program& program, g4::SequenceHandle sequence, std::uint32_t outputs) {
    Tokens result;
    const std::array sequences{sequence};
    for (std::uint32_t i = 0; i < outputs; ++i) {
        const std::array budgets{runtime::RoundBudget{outputs - i}};
        auto pending = i == 0 ? prefill(program, sequence) : program.decode(sequences, budgets);
        require(pending.row_failure(0) == runtime::PendingRowFailure::None, "reference logits failed");
        result.push_back(pending.tokens()[0]);
        runtime::CommitDecision decision;
        decision.accepted_tokens = 1;
        decision.terminal = i + 1 == outputs;
        const std::array decisions{decision};
        (void)program.commit(std::move(pending), decisions);
    }
    require(program.finish(sequence).status == runtime::ConsumeStatus::Consumed, "terminal sequence did not finish");
    return result;
}
void native_transactions(const char* artifact) {
    DeviceContext device(0);
    auto model = g4::load_model(artifact, {}, device);
    const g4::execution::Parameters parameters(*model, device.stream);
    g4::Frontend frontend(model->resources(), {.max_context = 2048});
    {
        // Eager execution lets the fault land inside layers rather than count a single Graph replay.
        g4::Program program(parameters, device,
            {.max_context = 2048, .prefill_chunk = 128, .use_cuda_graph = false});
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.sampling.temperature = 0;
        options.sampling.top_k = 20;
        options.sampling.top_p = 1;
        options.sampling.min_p = 0;
        options.sampling.seed = 7;
        options.allow_prefix_reuse = false;
        auto plan = program.plan_request(frontend.prepare_tokens(tokens(1537, 17)), options);
        const auto reference = complete(program, bind(program, plan), 8);

        // Abort an output that was sampled but never licensed; neither its ledger nor KV may leak.
        const auto abandoned = bind(program, plan);
        auto pending = prefill(program, abandoned);
        require(program.abort_pending(std::move(pending)).status == runtime::ConsumeStatus::Consumed,
                "pending abort did not consume");
        require(program.abort_pending(std::move(pending)).status == runtime::ConsumeStatus::InvariantMismatch,
                "a consumed pending capability was reusable");
        require(complete(program, bind(program, plan), 8) == reference, "aborted output changed the next request");
        bool stale_rejected = false;
        try { (void)program.advance_prefill(abandoned); }
        catch (const std::logic_error&) { stale_rejected = true; }
        require(stale_rejected, "an old sequence epoch was reusable");

        // Preserve a committed 1024-token frontier without claiming a cache snapshot. Resume replays it.
        auto sequence = bind(program, plan);
        for (int i = 0; i < 8; ++i) {
            require(!program.advance_prefill(sequence).complete, "pause prompt completed too early");
        }
        require(program.start_pause(sequence), "pause refused");
        auto paused = program.poll_context({});
        require(paused.paused && paused.paused->frontier() == 1024, "pause lost its committed frontier");
        sequence = bind(program, plan, &*paused.paused);
        while (!program.advance_replay(sequence).complete) {}
        require(complete(program, sequence, 8) == reference, "root replay changed the continuation");

        // At 1152 tokens the next chunk wraps the 18-page ring. Inject inside real layer work.
        sequence = bind(program, plan);
        for (int i = 0; i < 9; ++i) { (void)program.advance_prefill(sequence); }
        testing::arm_memory_exhausted_here(100);
        bool failed = false;
        try { (void)program.advance_prefill(sequence); }
        catch (const DeviceMemoryExhausted&) { failed = true; }
        testing::arm_memory_exhausted_here(0);
        require(failed, "the partial-forward memory fault did not fire");
        require(program.recover_after_failure(), "the real device did not recover after partial forward");
        require(complete(program, bind(program, plan), 8) == reference,
                "partially overwritten rings or global KV changed a fresh request");
        std::puts("PASS native pending-abort, stale epoch, root replay and partial-ring-write recovery");
    }
    // Scoring has no generation worker's recovery wrapper; its own failure must settle real writes.
    g4::Program scorer(parameters, device,
        {.max_context = 2048, .prefill_chunk = 128, .causal_scoring = true, .use_cuda_graph = false});
    const auto score_tokens = tokens(257, 91);
    ScoreOptions score_options;
    score_options.top_k = 3;
    score_options.candidates_per_position = 2;
    score_options.logits_positions = {0, 127};
    for (std::size_t i = 129; i < score_tokens.size(); ++i) {
        score_options.candidates.push_back(score_tokens[i]);
        score_options.candidates.push_back(7);
    }
    const auto scored = scorer.causal_score(frontend.prepare_tokens(score_tokens), 129, score_options);
    for (std::size_t i = 0; i < scored.logprobs.size(); ++i) {
        require(std::abs(scored.logprobs[i] - scored.candidate_logprobs[2 * i]) < 1e-5F,
                "candidate columns are not aligned with teacher-forced targets");
    }
    testing::arm_memory_exhausted_here(100);
    bool scoring_failed = false;
    try { (void)scorer.causal_score(frontend.prepare_tokens(score_tokens), 129, score_options); }
    catch (const DeviceMemoryExhausted&) { scoring_failed = true; }
    testing::arm_memory_exhausted_here(0);
    require(scoring_failed, "the scoring memory fault did not fire");
    const auto after = scorer.causal_score(frontend.prepare_tokens(score_tokens), 129, score_options);
    require(after.logprobs == scored.logprobs && after.logits == scored.logits &&
            after.top_ids == scored.top_ids && after.candidate_logprobs == scored.candidate_logprobs,
            "a failed scoring call contaminated subsequent public distributions");
    std::puts("PASS scoring alignment and partial-forward recovery");
}
RequestOptions request() {
    RequestOptions out;
    out.execution.requested_output_tokens = 8;
    out.execution.sampling.temperature = 0.0F;
    out.execution.sampling.top_k = 20;
    out.execution.allow_prefix_reuse = false;
    out.stop.include_model_defaults = false;
    return out;
}
void public_recovery(const char* artifact) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 2048;
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.context_cache.enabled = false;
    options.kv_cache = KvCacheStorage::BFloat16;
    options.logits_dtype = LogitsDtype::BF16;
    options.speculative.backend_auto = false;
    options.speculative.backend = SpeculativeBackend::None;
    Engine engine(options);
    const auto prompt = tokens(257, 21);
    const auto other = tokens(129, 22);
    const auto reference = engine.generate(engine.prepare_tokens(prompt), request()).generated_token_ids;
    const auto queued_reference = engine.generate(engine.prepare_tokens(other), request()).generated_token_ids;
    const auto recoveries = engine.runtime_stats().device_memory_recoveries;
    runtime::testing::hold_units(true);
    auto first = engine.submit(engine.prepare_tokens(prompt), request());
    auto second = engine.submit(engine.prepare_tokens(other), request());
    runtime::testing::request_memory_fault(1);
    runtime::testing::hold_units(false);
    bool failed = false;
    try { (void)first.wait(); }
    catch (const std::exception& error) {
        failed = std::string(error.what()).find("out of device memory") != std::string::npos;
    }
    runtime::testing::request_memory_fault(0);
    require(failed, "the public Engine did not surface the retryable shortage");
    require(second.wait().generated_token_ids == queued_reference, "recovery lost or changed the queued request");
    require(engine.runtime_stats().device_memory_recoveries == recoveries + 1, "Engine did not count its recovery");
    require(engine.generate(engine.prepare_tokens(prompt), request()).generated_token_ids == reference,
            "Engine recovery changed the next request");
    runtime::testing::set_nonfinite_fault(2); // first decode, after the first output was committed
    failed = false;
    try { (void)engine.generate(engine.prepare_tokens(prompt), request()); }
    catch (const std::exception&) { failed = true; }
    runtime::testing::set_nonfinite_fault(0);
    require(failed, "non-finite decode output was published as success");
    require(engine.generate(engine.prepare_tokens(prompt), request()).generated_token_ids == reference,
            "a failed decode contaminated the next request");
    std::puts("PASS public Engine queued OOM recovery and failed-decode withdrawal");
}
runtime::ResolvedExecutionOptions greedy_execution() {
    runtime::ResolvedExecutionOptions options;
    options.requested_output_tokens = 8;
    options.sampling.temperature = 0;
    options.sampling.top_k = 20;
    options.sampling.top_p = 1;
    options.sampling.min_p = 0;
    options.sampling.seed = 7;
    options.allow_prefix_reuse = false;
    return options;
}
bool refused(const g4::Model& model, const g4::ProgramOptions& options) {
    try { (void)g4::Program::plan_device(model, options, model.config().text.vocab_size); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}
std::uint64_t expected_global_payload(const g4::execution::Parameters& parameters, KvCacheStorage storage,
                                      std::uint32_t pages) {
    std::uint64_t bytes = 0;
    for (const auto& layer : parameters.layers) {
        if (layer.global_layer) {
            bytes += g4::kv_layer_layout(storage, true, layer.head_dim, layer.kv_heads).page_bytes() * pages;
        }
    }
    return bytes;
}
// One FP8-KV Program: accounting must match the real allocation, and an eager forward that dies
// midway must leave no NaN/stale state that changes the next request (unwritten rows are zero).
Tokens fp8_run(const g4::execution::Parameters& parameters, DeviceContext& device, g4::Frontend& frontend,
               ops::Gemma4QkMode qk, bool graphs) {
    constexpr auto fp8 = KvCacheStorage::Fp8E4M3Row256;
    g4::Program program(parameters, device,
        {.max_context = 2048, .prefill_chunk = 128, .kv_cache = fp8, .qk_mode = qk, .use_cuda_graph = graphs});
    const auto memory = program.memory_summary();
    require(memory.kv_cache == fp8 && memory.gemma4_kv.has_value(), "FP8 memory summary lost the KV format");
    const auto& kv = *memory.gemma4_kv;
    require(kv.requested == fp8 && kv.global == fp8 && kv.sliding == KvCacheStorage::BFloat16 && kv.qk_mode == qk,
            "FP8 memory summary misreports the global/sliding formats or QK mode");
    const auto pages = program.device_plan().kv_pages;
    require(kv.global_payload_bytes == expected_global_payload(parameters, fp8, pages),
            "global FP8 payload does not include the code and scale planes");
    require(memory.kv_payload_bytes == kv.global_payload_bytes + kv.sliding_payload_bytes,
            "kv_payload_bytes disagrees with the Gemma payload split");
    require(memory.kv_payload_bytes + 8ULL * pages == program.device_plan().kv,
            "planner and constructor disagree on FP8 KV bytes");
    require(memory.cuda_graph_measured_bytes <= memory.cuda_graph_allowance_bytes,
            "CUDA graph preparation exceeded its allowance");
    auto plan = program.plan_request(frontend.prepare_tokens(tokens(1537, 17)), greedy_execution());
    const auto reference = complete(program, bind(program, plan), 8);
    if (!graphs) {
        auto sequence = bind(program, plan);
        for (int i = 0; i < 9; ++i) { (void)program.advance_prefill(sequence); }
        testing::arm_memory_exhausted_here(100);
        bool failed = false;
        try { (void)program.advance_prefill(sequence); }
        catch (const DeviceMemoryExhausted&) { failed = true; }
        testing::arm_memory_exhausted_here(0);
        require(failed, "the FP8 partial-forward memory fault did not fire");
        require(program.recover_after_failure(), "the device did not recover after a partial FP8 forward");
        require(complete(program, bind(program, plan), 8) == reference,
                "a partially written FP8 cache changed a fresh request after recovery");
    }
    return reference;
}
void fp8_kv_program(const char* artifact) {
    DeviceContext device(0);
    auto model = g4::load_model(artifact, {}, device);
    const g4::execution::Parameters parameters(*model, device.stream);
    g4::Frontend frontend(model->resources(), {.max_context = 2048});
    g4::ProgramOptions options{.max_context = 2048, .prefill_chunk = 128};
    options.qk_mode = ops::Gemma4QkMode::Fp8;
    require(refused(*model, options), "FP8 QK with BF16 KV was accepted");
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.qk_mode = ops::Gemma4QkMode::BFloat16;
    require(refused(*model, options), "INT8 KV was accepted by Gemma4");
    for (const auto qk : {ops::Gemma4QkMode::BFloat16, ops::Gemma4QkMode::Fp8}) {
        const auto eager = fp8_run(parameters, device, frontend, qk, false);
        require(fp8_run(parameters, device, frontend, qk, true) == eager,
                "CUDA graphs changed the FP8 KV continuation");
    }
    std::puts("PASS FP8 KV Program (BF16-QK and FP8-QK, graphs on/off, accounting, refusals, recovery)");
}
void public_fp8(const char* artifact) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 2048;
    options.prefill_chunk = 128;
    options.max_concurrency = 1;
    options.context_cache.enabled = false;
    options.kv_cache = KvCacheStorage::BFloat16;
    options.gemma4_qk = ops::Gemma4QkMode::Fp8;
    options.logits_dtype = LogitsDtype::BF16;
    options.speculative.backend_auto = false;
    options.speculative.backend = SpeculativeBackend::None;
    bool refused_qk = false;
    try { Engine engine(options); }
    catch (const std::invalid_argument&) { refused_qk = true; }
    require(refused_qk, "the public Engine accepted FP8 QK with BF16 KV");
    options.kv_cache = KvCacheStorage::Fp8E4M3Row256;
    Engine engine(options);
    require(!engine.generate(engine.prepare_tokens(tokens(257, 21)), request()).generated_token_ids.empty(),
            "the FP8 public Engine generated nothing");
    const auto memory = engine.memory_summary();
    require(memory.kv_cache == KvCacheStorage::Fp8E4M3Row256 && memory.gemma4_kv.has_value() &&
            memory.gemma4_kv->qk_mode == ops::Gemma4QkMode::Fp8, "the Engine memory summary lost the FP8 settings");
    require(engine.load_summary().prefill_signature.find("fp8-kv/qk=fp8") != std::string::npos,
            "the load summary does not identify the FP8 KV and QK route");
    std::puts("PASS public FP8 KV/FP8-QK Engine and QK refusal");
}
} // namespace
int main() {
    const char* artifact = std::getenv("INFERNIX_GEMMA4_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::puts("SKIP: INFERNIX_GEMMA4_ARTIFACT is not set");
        return 77;
    }
    try {
        native_transactions(artifact);
        public_recovery(artifact);
        fp8_kv_program(artifact);
        public_fp8(artifact);
        return 0;
    } catch (const std::exception& error) {
        testing::arm_memory_exhausted_here(0);
        runtime::testing::request_memory_fault(0);
        runtime::testing::set_nonfinite_fault(0);
        runtime::testing::hold_units(false);
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
