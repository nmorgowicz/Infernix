#include "runtime/engine/gemma4_instance.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/device.h"
#include "core/host_memory.h"
#include "core/startup.h"
#include "core/vram_budget.h"
#include "models/load_options.h"
#include "models/gemma4/program/kv_layout.h"
#include "runtime/engine/engine_plan.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/expert_offload/memory_plan.h"

#include <chrono>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::runtime {
namespace {
using Clock = std::chrono::steady_clock;
using models::gemma4::Program;
using models::gemma4::ProgramOptions;

ProgramOptions program_options(const EngineOptions& options) {
    ProgramOptions out;
    out.max_context = options.max_context;
    out.max_concurrency = options.max_concurrency;
    out.prefill_chunk = options.prefill_chunk;
    out.kv_capacity_tokens = options.kv_capacity.mode == KvCapacityMode::Automatic ? 0U :
        ((options.kv_capacity.explicit_tokens + 63U) / 64U) * 64U;
    out.kv_cache = options.kv_cache;
    out.qk_mode = options.gemma4_qk;
    out.causal_scoring = options.purpose == EnginePurpose::CausalScoring;
    out.prefix_cache = options.context_cache.enabled;
    out.reproducible = !options.fast_paths;
    out.use_cuda_graph = options.use_cuda_graph;
    out.diagnostics = options.diagnostic_observer;
    out.startup_observer = options.startup_observer;
    return out;
}

ModelMetadata metadata(const models::gemma4::Model& model) {
    ModelMetadata out;
    const auto& text = model.config().text;
    out.model_id = model.info().name;
    out.vocab_size = text.vocab_size;
    out.embedding_size = text.hidden_size;
    out.native_context = text.max_position_embeddings;
    std::set<const WeightParent*> counted;
    std::map<std::string, std::uint64_t> by_format;
    for (const auto& weight : model.weight_data()) {
        for (const auto& part : weight.view.parts) {
            if (!part.parent || !counted.insert(part.parent).second) { continue; }
            out.parameters += part.parent->geometry.elements;
            out.weight_bytes += part.parent->geometry.bytes;
            by_format[std::string(artifact::format_name(part.parent->geometry.format))] +=
                part.parent->geometry.elements;
        }
    }
    std::uint64_t dominant = 0;
    for (const auto& [name, count] : by_format) {
        if (count > dominant) { dominant = count; out.weights_id = name; }
    }
    return out;
}

} // namespace

void validate_gemma4_options(const EngineOptions& options) {
    if (!options.fast_paths || options.lossless || options.reference) {
        throw std::invalid_argument("Gemma4 does not support --reproducible, --lossless or --reference");
    }
    if (options.max_concurrency != 1) {
        throw std::invalid_argument("Gemma4 requires --max-concurrency 1");
    }
    if (options.context_cache.enabled) {
        throw std::invalid_argument("Gemma4 cold-cache execution requires --no-prefix-reuse; prefix reuse is not implemented yet");
    }
    (void)models::gemma4::gemma4_kv_format(options.kv_cache);
    validate_gemma4_qk(options.gemma4_qk, true, options.kv_cache);
    const auto& spec = options.speculative;
    if (spec.backend_auto || spec.backend != SpeculativeBackend::None || spec.ngram_draft_tokens != 0 ||
        spec.ngram_draft_tokens_auto || spec.ngram_archive_bytes != 0 || !spec.draft_model.empty()) {
        throw std::invalid_argument("Gemma4 cold-cache execution requires --spec off and no ngram drafting");
    }
    if (options.enable_vision || options.vision_offload == VisionOffload::On) {
        throw std::invalid_argument("Gemma4 vision and audio input are not supported by the text route");
    }
    if (options.rope_yarn_factor != 1.0F || options.embedding_host || options.a8_activations ||
        options.expert_a4_activations || options.expert_a16_activations || options.expert_csf ||
        !options.fast_path_candidates.empty() || !options.fast_path_exclusions.empty()) {
        throw std::invalid_argument("Gemma4 runs its artifact-declared projection policies; Qwen activation, fast-path, embedding-host and YaRN overrides are unsupported");
    }
    if (!options.thinking_budget_message.empty()) {
        throw std::invalid_argument("Gemma4 thinking budgets are accounting-only; a budget-control message is unsupported");
    }
    if (options.logits_dtype != LogitsDtype::BF16) {
        throw std::invalid_argument("Gemma4 preserves its native BF16 head; select --logits-dtype bf16 (unrounded FP32 logits are unsupported)");
    }
}

Gemma4Instance::Gemma4Instance(std::unique_ptr<models::gemma4::Model> source,
                               const EngineOptions& options, DeviceContext& device)
    : model(std::move(source)), parameters(*model, device.stream),
      frontend(model->resources(), {.max_context = options.max_context,
                                    .chat_template_path = options.chat_template_path,
                                    .grammar_cache_bytes = options.grammar_cache_bytes}),
      capacity(options.max_context) {}

Gemma4Instance::~Gemma4Instance() = default;

ConstructedGemma4 construct_gemma4(const EngineOptions& supplied, DeviceContext& device, EnginePlan* planned) {
    EngineOptions options = supplied;
    validate_gemma4_options(options);
    const auto start = Clock::now();
    const artifact::Reader reader(options.artifact_path);
    auto load_options = models::load_options(options, models::Architecture::Gemma4);
    const auto budget = open_vram_budget_source(device.device, options.vram_past_budget);
    const auto before = budget->query();
    const std::size_t headroom = static_cast<std::size_t>(
        expert_offload::display_headroom(before.display, false, options.vram_headroom_bytes));

    auto cold_plan = models::gemma4::plan_load(reader, load_options);
    const std::uint64_t weight_bytes = cold_plan.device_bytes();
    auto geometry = models::gemma4::plan_model(std::move(cold_plan));
    if (options.max_context > geometry->config().text.max_position_embeddings) {
        throw std::invalid_argument("Gemma4 --max-context exceeds the artifact's native position capacity");
    }
    const auto available = planned_free_after_weights(before.device_free, weight_bytes);
    if (options.prefill_chunk_auto) {
        try {
            options.prefill_chunk = models::gemma4::auto_prefill_chunk(
                options.max_context, available, headroom,
                [&](std::uint32_t chunk, std::uint32_t capacity) {
                    auto candidate = program_options(options);
                    candidate.prefill_chunk = chunk;
                    candidate.kv_capacity_tokens = capacity;
                    const auto curve = Program::plan_device(
                        *geometry, candidate, geometry->config().text.vocab_size).capacity_curve();
                    return curve.reservation_bytes(capacity / curve.main_page_tokens);
                });
        } catch (const std::runtime_error& error) {
            if (!planned) { throw; }
            planned->architecture = "Gemma4ForCausalLM";
            plan_readings(*planned, device.props.name, before, host_memory_snapshot(), options);
            planned->vram.weights_bytes = weight_bytes;
            plan_refuse(*planned, error.what());
            return {};
        }
    }
    const auto request_plan = Program::plan_device(*geometry, program_options(options), geometry->config().text.vocab_size);
    const auto curve = request_plan.capacity_curve();
    KvCapacityResolution resolution;
    if (planned) {
        planned->architecture = "Gemma4ForCausalLM";
        plan_readings(*planned, device.props.name, before, host_memory_snapshot(), options);
        planned->vram.weights_bytes = weight_bytes;
        planned->vram.workspace_bytes = request_plan.workspace;
        planned->vram.graph_allowance_bytes = request_plan.graph_allowance;
        const auto fitted = plan_capacity(*planned, options.kv_capacity, curve, available, headroom);
        if (!fitted) { return {}; }
        resolution = *fitted;
        if (resolution.resolved_tokens < options.max_context) {
            plan_refuse(*planned, "Gemma4 KV capacity cannot hold --max-context");
            return {};
        }
        auto resolved_options = program_options(options);
        resolved_options.kv_capacity_tokens = resolution.resolved_tokens;
        const auto final_plan = Program::plan_device(*geometry, resolved_options, geometry->config().text.vocab_size);
        planned->vram.kv_bytes = final_plan.kv;
        plan_ram_totals(*planned, geometry->storage_stats().retained_host_bytes);
        return {};
    }
    resolution = resolve_kv_capacity(options.kv_capacity, curve, available, headroom);
    if (resolution.resolved_tokens < options.max_context) {
        throw std::runtime_error("Gemma4 KV capacity cannot hold --max-context; reduce context/chunk or free device memory");
    }
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(resolution.resolved_tokens);
    geometry.reset();
    std::unique_ptr<models::gemma4::Model> model;
    {
        StartupPhaseScope upload(options.startup_observer, StartupPhase::WeightsMaterialize);
        SpillGuard guard(*budget);
        guard.begin();
        model = models::gemma4::materialize_model(models::gemma4::plan_load(reader, load_options), device,
                                                  &options.startup_observer);
        if (guard.end(model->storage_stats().device_capacity_bytes) != 0) {
            throw std::runtime_error("Gemma4 weights spilled into host memory: " +
                                     guard.readings(model->storage_stats().device_capacity_bytes));
        }
        upload.complete();
    }
    auto instance = std::make_unique<Gemma4Instance>(std::move(model), options, device);
    const auto resident_plan = Program::plan_device(*instance->model, program_options(options),
                                                    instance->model->config().text.vocab_size);
    // Parameters already owns these planned bytes. Add them back to the free reading
    // while resolving the curve that includes them, rather than charging them twice.
    const std::size_t runtime_room = budget->query().device_free + instance->parameters.derived_device_bytes();
    resolution = resolve_kv_capacity(options.kv_capacity, resident_plan.capacity_curve(), runtime_room, headroom);
    instance->kv_capacity_resolution = resolution;
    {
        StartupPhaseScope construct(options.startup_observer, StartupPhase::ProgramInitialize);
        instance->program = std::make_unique<Program>(instance->parameters, device, program_options(options));
        construct.complete();
    }
    device.synchronize();
    instance->kv_capacity_resolution.available_after_startup_bytes = budget->query().device_free;

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = "Gemma4ForCausalLM";
    summary.model_name = instance->model->info().name;
    summary.prefill_signature = std::string("gemma4/") +
        (options.kv_cache == KvCacheStorage::Fp8E4M3Row256 ? "fp8" : "bf16") + "-kv/qk=" +
        gemma4_qk_name(options.gemma4_qk) + "/cold/chunk=" + std::to_string(options.prefill_chunk);
    std::set<std::string> formats;
    for (const auto& weight : instance->model->weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    summary.prefill_chunk = options.prefill_chunk;
    summary.request_invariant = false;
    summary.load_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    summary.upload_seconds = stats.upload_seconds;
    summary.artifact_bytes_read = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes = stats.peak_staging_bytes;
    summary.device_object_count = stats.device_object_count;
    summary.host_object_count = stats.host_object_count;
    auto model_metadata = metadata(*instance->model);
    return {std::move(instance), std::move(summary), std::move(model_metadata), std::move(options)};
}

} // namespace infernix::runtime
