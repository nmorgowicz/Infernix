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
#include "runtime/engine/expert_memory.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/expert_offload/memory_plan.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infernix::runtime {
namespace {
using Clock = std::chrono::steady_clock;
using models::gemma4::Program;
using models::gemma4::ProgramOptions;

models::gemma4::KvStorageModel global_kv_storage(const EngineOptions& options) {
    return options.gemma4_key_only_kv ? models::gemma4::KvStorageModel::KeyOnly
                                     : models::gemma4::KvStorageModel::KeyEqualsValue;
}

ProgramOptions program_options(const EngineOptions& options, std::uint64_t host_bytes) {
    ProgramOptions out;
    out.max_context = options.max_context;
    out.max_concurrency = options.max_concurrency;
    out.prefill_chunk = options.prefill_chunk;
    out.kv_capacity_tokens = options.kv_capacity.mode == KvCapacityMode::Automatic ? 0U :
        ((options.kv_capacity.explicit_tokens + 63U) / 64U) * 64U;
    out.kv_cache = options.kv_cache;
    out.global_kv_storage = global_kv_storage(options);
    out.sliding_kv_cache = options.gemma4_sliding_kv;
    out.qk_mode = options.gemma4_qk;
    out.causal_scoring = options.purpose == EnginePurpose::CausalScoring;
    out.draft_tokens = options.speculative.backend == SpeculativeBackend::Mtp ? options.speculative.draft_tokens : 0U;
    // A Host tier is the prefix cache's only capacity on Gemma: 0 (--host-context-mib 0) is the
    // off switch and leaves a cold lane.
    out.prefix_cache = options.context_cache.enabled && host_bytes != 0;
    out.host_bytes = options.context_cache.enabled ? host_bytes : 0;
    out.reproducible = !options.fast_paths;
    // free-cache-resume is Gemma's one optional fast path: on unless --no-fast-path leaves it out.
    out.free_cache_resume = std::find(options.fast_path_exclusions.begin(), options.fast_path_exclusions.end(),
                                      FastPath::FreeCacheResume) == options.fast_path_exclusions.end();
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

// The opt-ins serve native Q5 trunk projections, not a Q5 head or assistant alone.
bool has_native_q5(const models::gemma4::Model& model) {
    for (const auto& weight : model.weight_data()) {
        const bool trunk = std::any_of(weight.uses.begin(), weight.uses.end(), [](const auto& use) {
            return use.input.starts_with("text/layers/");
        });
        if (!trunk) { continue; }
        for (const auto& part : weight.view.parts) {
            if (part.parent && part.parent->geometry.format == QType::Q5_G64_FP16) { return true; }
        }
    }
    return false;
}

} // namespace

std::uint32_t resolve_gemma4_auto_context(
    std::uint32_t upper, std::size_t available, std::size_t headroom,
    const std::function<std::size_t(std::uint32_t)>& reservation, bool allow_shared_memory) {
    constexpr std::uint32_t page = ops::kGemma4KvPageTokens;
    constexpr std::uint32_t minimum = 9216;
    upper = std::min(upper, 131072U) / page * page;
    if (upper >= minimum && available >= headroom) {
        // Search page indices relative to the usable floor, not tiny contexts whose ring curve
        // may not exist. Every candidate asks Program::plan_device for its own full reservation.
        const auto index = largest_fitting_context((upper - minimum) / page + 1U, 1U,
            [&](std::uint32_t candidate) {
                return reservation(minimum + (candidate - 1U) * page) <= available - headroom;
            });
        if (index) { return minimum + (*index - 1U) * page; }
    }
    if (allow_shared_memory && upper >= minimum) { return minimum; }
    std::string detail;
    if (upper >= minimum) {
        const auto mib = [](std::size_t bytes) { return std::to_string(bytes >> 20) + " MiB"; };
        const std::size_t need = reservation(minimum);
        const std::size_t room = available > headroom ? available - headroom : 0;
        detail = " [" + std::to_string(minimum) + " tokens need " + mib(need) + " beside the weights; " + mib(room) +
                 " remain after weights, headroom and startup margin: short by " +
                 mib(need > room ? need - room : 0) + "]";
    }
    throw std::runtime_error(
        "Gemma4 --max-context auto cannot fit the minimum usable 9216 tokens (1024 prompt + "
        "8192 output) within the artifact/131072 ceiling, resident VRAM and headroom; free GPU "
        "memory, lower --vram-headroom-mib or use a smaller artifact/chunk. Host-paged startup is not permitted; "
        "--allow-shared-memory starts anyway with the shortfall in shared GPU memory (much slower)" +
        detail);
}

std::size_t gemma4_shared_memory_room(const SequenceCapacityCurve& curve, std::size_t free,
                                      std::uint32_t context, std::size_t headroom) {
    const std::uint32_t wanted = (context + curve.main_page_tokens - 1U) / curve.main_page_tokens;
    const std::uint32_t groups = std::clamp(wanted, curve.minimum_main_page_groups, curve.maximum_main_page_groups);
    return std::max(free, curve.reservation_bytes(groups) + headroom);
}

void validate_gemma4_options(const EngineOptions& options) {
    switch (options.gemma4_q5_prefill) {
    case Gemma4Q5Prefill::A16:
    case Gemma4Q5Prefill::Fp8:
    case Gemma4Q5Prefill::Int8G32:
    case Gemma4Q5Prefill::Int8G64: break;
    default: throw std::invalid_argument("invalid Gemma4 Q5 prefill mode");
    }
    if (!options.lora_path.empty() || !options.ablate_sidecar_path.empty()) {
        throw std::invalid_argument("Gemma4 does not support Qwen runtime --lora or --ablate sidecars");
    }
    if (!options.fast_paths || options.lossless || options.reference) {
        throw std::invalid_argument("Gemma4 does not support --reproducible, --lossless or --reference");
    }
    if (options.max_concurrency != 1) {
        throw std::invalid_argument("Gemma4 requires --max-concurrency 1");
    }
    if (options.context_cache.enabled &&
        (options.context_cache.ssd_capacity_bytes != 0 || !options.context_cache.ssd_directory.empty())) {
        throw std::invalid_argument("Gemma4's prefix cache has no SSD tier; --ssd-context-gib and --ssd-context-dir are unsupported");
    }
    if (options.context_cache.enabled && !options.context_cache.hybrid.persistent_file.empty()) {
        throw std::invalid_argument("Gemma4 prefix cache persistence is not implemented; --prefix-cache-file is unsupported");
    }
    (void)models::gemma4::gemma4_kv_format(options.kv_cache);
    validate_gemma4_qk(options.gemma4_qk, true, options.kv_cache);
    if (options.gemma4_key_only_kv && options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument("--gemma4-key-only-kv requires --kv-dtype fp8");
    }
    if (options.gemma4_sliding_kv != KvCacheStorage::BFloat16 && options.gemma4_sliding_kv != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument("--gemma4-sliding-kv must be bf16 or fp8");
    }
    if (options.gemma4_sliding_kv == KvCacheStorage::Fp8E4M3Row256 && options.kv_cache != KvCacheStorage::Fp8E4M3Row256) {
        throw std::invalid_argument("--gemma4-sliding-kv fp8 requires --kv-dtype fp8");
    }
    if (options.max_context_auto && options.vram_past_budget) {
        throw std::invalid_argument("Gemma4 --max-context auto refuses --vram-past-budget; startup must remain resident");
    }
    if (options.max_context_auto && options.kv_capacity.mode != KvCapacityMode::Automatic) {
        throw std::invalid_argument(
            "Gemma4 --max-context auto requires --kv-capacity auto; an explicit KV capacity stays fixed");
    }
    const auto& spec = options.speculative;
    if (spec.backend_auto || (spec.backend != SpeculativeBackend::None && spec.backend != SpeculativeBackend::Mtp) ||
        spec.ngram_draft_tokens != 0 || spec.ngram_draft_tokens_auto || spec.ngram_archive_bytes != 0 ||
        !spec.draft_model.empty()) {
        throw std::invalid_argument("Gemma4 drafts only with its MTP assistant (--spec mtp or off); no ngram drafting");
    }
    if (spec.backend == SpeculativeBackend::Mtp && (spec.draft_tokens < 1 || spec.draft_tokens > 7)) {
        throw std::invalid_argument("Gemma4 --spec mtp drafts 1..7 tokens per round (--draft-tokens)");
    }
    if (spec.draft_min_p || spec.proposal_head != ProposalHead::Full) {
        throw std::invalid_argument("Gemma4 MTP has no draft-confidence cut (--spec-min-p) or proposal head");
    }
    if (spec.draft_tree_auto || std::any_of(spec.draft_tree_nodes.begin(), spec.draft_tree_nodes.end(),
                                          [](std::uint32_t n) { return n != 0; })) {
        throw std::invalid_argument("Gemma4 MTP supports chain verification only; draft trees are unsupported");
    }
    if (options.enable_vision || options.vision_offload == VisionOffload::On) {
        throw std::invalid_argument("Gemma4 vision and audio input are not supported by the text route");
    }
    if (options.rope_yarn_factor != 1.0F || options.embedding_host || options.a8_activations ||
        options.expert_a4_activations || options.expert_a16_activations || options.expert_csf ||
        !options.fast_path_candidates.empty() ||
        std::any_of(options.fast_path_exclusions.begin(), options.fast_path_exclusions.end(),
                    [](FastPath path) { return path != FastPath::FreeCacheResume; })) {
        throw std::invalid_argument("Gemma4 runs its artifact-declared projection policies; Qwen activation, fast-path "
                                    "(other than --no-fast-path free-cache-resume), embedding-host and YaRN overrides "
                                    "are unsupported");
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
                                    .grammar_cache_bytes = options.grammar_cache_bytes,
                                    .thinking_budget_message = options.thinking_budget_message}),
      capacity(options.max_context) {}

Gemma4Instance::~Gemma4Instance() = default;

ConstructedGemma4 construct_gemma4(const EngineOptions& supplied, DeviceContext& device, EnginePlan* planned) {
    EngineOptions options = supplied;
    validate_gemma4_options(options);
    const auto start = Clock::now();
    const artifact::Reader reader(options.artifact_path);
    auto load_options = models::load_options(options, models::Architecture::Gemma4);
    const auto budget = open_vram_budget_source(device.device, options.vram_past_budget, true);
    const auto before = budget->query();
    const std::size_t headroom = static_cast<std::size_t>(
        expert_offload::display_headroom(before.display, false, options.vram_headroom_bytes));
    const auto require_auto_readings = [&](const VramSnapshot& reading) {
        if (!options.max_context_auto) { return; }
        if (!reading.physical_free) {
            throw std::runtime_error(
                "Gemma4 --max-context auto needs an NVML physical free-VRAM reading; ensure nvml.dll/"
                "libnvidia-ml.so.1 is available and the NVIDIA driver can report memory");
        }
#ifdef _WIN32
        if (!reading.has_budget) {
            throw std::runtime_error(
                "Gemma4 --max-context auto needs the Windows per-process VRAM budget from DXGI; "
                "cannot guarantee a resident startup without it");
        }
#endif
    };

    // --allow-shared-memory: capacity checks count what the reservation needs when free VRAM falls short.
    const auto shared_room = [&](const SequenceCapacityCurve& curve, std::size_t free) {
        return options.allow_shared_memory ? gemma4_shared_memory_room(curve, free, options.max_context, headroom) : free;
    };

    auto cold_plan = models::gemma4::plan_load(reader, load_options);
    const std::uint64_t weight_bytes = cold_plan.device_bytes();
    auto geometry = models::gemma4::plan_model(std::move(cold_plan));
    if (options.gemma4_q5_prefill != Gemma4Q5Prefill::A16 && !has_native_q5(*geometry)) {
        throw std::invalid_argument(std::string("--gemma4-q5-prefill ") + gemma4_q5_prefill_name(options.gemma4_q5_prefill) +
                                    " needs an artifact with native Q5 (q5_g64_fp16) trunk projections");
    }
    if (options.max_context_auto) {
        options.max_context = std::min({options.max_context, 131072U, geometry->config().text.max_position_embeddings});
    } else if (options.max_context > geometry->config().text.max_position_embeddings) {
        throw std::invalid_argument("Gemma4 --max-context exceeds the artifact's native position capacity");
    }
    // The prefix cache's Host tier: explicit --host-context-mib, or auto like Qwen3.8-Flash-Next
    // (the RAM ledger's share beyond its reserve and plan, at least 1 GiB; expert_memory.h).
    // 0 (--host-context-mib 0) disables the cache: the lane stays cold.
    const bool cache_on = options.context_cache.enabled;
    const bool prefix_auto = cache_on && !options.context_cache.host_capacity_bytes;
    std::uint64_t prefix_bytes =
        cache_on && !prefix_auto ? *options.context_cache.host_capacity_bytes : 0;
    if (prefix_auto) {
        auto demand = expert_ram_demand(options, {}, 0U, budget->query());
        demand.pageable = frontend_cache_bytes(options);
        prefix_bytes = auto_host_context_bytes(expert_offload::plan_host_memory(host_memory_snapshot(), demand));
    }
    // HPC-min packing (D1): one image is the ring window alone; the Host tier must hold one live
    // snapshot, one staging image and a slab of headroom (image_slabs + 2 slabs). The requirement
    // is computed from the artifact's geometry, never assumed from a model size.
    if (cache_on && prefix_bytes != 0) {
        const auto& text = geometry->config().text;
        std::vector<models::gemma4::KvLayerFact> facts;
        facts.reserve(text.layer_types.size());
        for (const auto kind : text.layer_types) {
            facts.push_back({kind == models::gemma4::LayerKind::Global,
                             static_cast<int>(text.head_dim_of(kind)),
                             static_cast<int>(text.kv_heads_of(kind))});
        }
        const auto tier = models::gemma4::prefix_cache_geometry(
            facts, text.sliding_window,
            {options.kv_cache, global_kv_storage(options), options.gemma4_sliding_kv});
        if (prefix_bytes < tier.host_floor_bytes) {
            const std::uint64_t mib = (tier.host_floor_bytes + (1ULL << 20) - 1) >> 20;
            const std::string refusal =
                "--host-context-mib is too small for the Gemma4 prefix cache: one ring snapshot needs " +
                std::to_string(mib) + " MiB (" + std::to_string(tier.image_slabs) + "+2 slabs of " +
                std::to_string((tier.slab_bytes + 1023) >> 10) + " KiB from this artifact's geometry)";
            if (!planned) { throw std::invalid_argument(refusal); }
            planned->architecture = "Gemma4ForCausalLM";
            plan_readings(*planned, device.props.name, before, host_memory_snapshot(), options);
            planned->vram.weights_bytes = weight_bytes;
            plan_refuse(*planned, refusal);
            return {};
        }
    }
    const auto resolve_options = [&](const models::gemma4::Model& model, std::size_t available) {
        const auto reservation = [&](std::uint32_t context, std::uint32_t chunk) {
            auto candidate = program_options(options, prefix_bytes);
            candidate.max_context = context;
            candidate.prefill_chunk = chunk;
            candidate.kv_capacity_tokens = ((context + 63U) / 64U) * 64U;
            const auto curve = Program::plan_device(model, candidate, model.config().text.vocab_size).capacity_curve();
            return curve.reservation_bytes(candidate.kv_capacity_tokens / curve.main_page_tokens);
        };
        if (options.max_context_auto) {
            // Parameters, kernel modules loaded at first use and CUDA Graph state allocate after this reading
            // (103-146 MiB past the plan in IQ3 startups at chunk 128, the most with MTP), and the device
            // grows about 3 KiB per context token beyond the plan (ledger: 23.2 against 20.2 MiB per 1k tokens at
            // key-only, chunk 128); auto keeps both so the final headroom check holds after startup instead of failing it.
            constexpr std::size_t kAutoStartupMargin = 160ULL << 20;
            constexpr std::size_t kAutoUnplannedBytesPerToken = 4096;
            options.max_context = resolve_gemma4_auto_context(
                options.max_context, available, headroom + kAutoStartupMargin,
                [&](std::uint32_t context) {
                    // Maximize context first; then choose the widest chunk that fits it.
                    return reservation(context, options.prefill_chunk_auto ? 128U : options.prefill_chunk) +
                           static_cast<std::size_t>(context) * kAutoUnplannedBytesPerToken;
                }, options.allow_shared_memory);
        }
        if (options.prefill_chunk_auto) {
            options.prefill_chunk = models::gemma4::auto_prefill_chunk(
                options.max_context, available, headroom,
                [&](std::uint32_t chunk, std::uint32_t) {
                    return reservation(options.max_context, chunk);
                });
        }
        return Program::plan_device(model, program_options(options, prefix_bytes), model.config().text.vocab_size);
    };
    if (planned) {
        planned->architecture = "Gemma4ForCausalLM";
        plan_readings(*planned, device.props.name, before, host_memory_snapshot(), options);
        planned->vram.weights_bytes = weight_bytes;
        const auto available = planned_free_after_weights(before.device_free, weight_bytes);
        try {
            require_auto_readings(before);
            const auto request_plan = resolve_options(*geometry, available);
            planned->vram.workspace_bytes = request_plan.workspace;
            planned->vram.graph_allowance_bytes = request_plan.graph_allowance;
            const auto fitted = plan_capacity(*planned, options.kv_capacity, request_plan.capacity_curve(),
                                               shared_room(request_plan.capacity_curve(), available), headroom);
            if (!fitted) { return {}; }
            if (fitted->resolved_tokens < options.max_context) {
                plan_refuse(*planned, "Gemma4 KV capacity cannot hold --max-context");
                return {};
            }
            auto resolved_options = program_options(options, prefix_bytes);
            resolved_options.kv_capacity_tokens = fitted->resolved_tokens;
            planned->vram.kv_bytes = Program::plan_device(
                *geometry, resolved_options, geometry->config().text.vocab_size).kv;
            if (options.max_context_auto) { planned->max_context = options.max_context; }
            plan_ram_totals(*planned, geometry->storage_stats().retained_host_bytes);
        } catch (const std::exception& error) {
            plan_refuse(*planned, error.what());
        }
        return {};
    }
    require_auto_readings(before);
    // Cold feasibility refuses a startup that cannot keep even the weights/minimum context
    // resident, but never freezes auto's ceiling: resolve again after the actual upload.
    const auto ceiling = options.max_context;
    const auto requested_chunk = options.prefill_chunk;
    const auto available = planned_free_after_weights(before.device_free, weight_bytes);
    const auto cold_request = resolve_options(*geometry, available);
    const auto fitted = resolve_kv_capacity(options.kv_capacity, cold_request.capacity_curve(),
                                            shared_room(cold_request.capacity_curve(), available), headroom);
    if (fitted.resolved_tokens < options.max_context) {
        throw std::runtime_error("Gemma4 KV capacity cannot hold --max-context; reduce context/chunk or free device memory");
    }
    const auto cold_context = options.max_context;
    if (options.max_context_auto) { options.max_context = ceiling; }
    if (options.prefill_chunk_auto) { options.prefill_chunk = requested_chunk; }
    geometry.reset();
    std::unique_ptr<models::gemma4::Model> model;
    std::uint64_t shared_weights_bytes = 0;
    {
        StartupPhaseScope upload(options.startup_observer, StartupPhase::WeightsMaterialize);
        SpillGuard guard(*budget);
        // --allow-shared-memory with the runtime resident: hold the runtime state's bytes before the weights so that
        // the driver's overflow lands on the weights, then release them for the Program's own allocations.
        DeviceBuffer runtime_hold;
        if (options.allow_shared_memory && options.shared_memory_placement == SharedMemoryPlacement::RuntimeResident) {
            const auto curve = cold_request.capacity_curve();
            const auto need = gemma4_shared_memory_room(curve, 0, cold_context, 0);
            if (before.device_free < weight_bytes + need + headroom) { runtime_hold = DeviceBuffer(need); }
        }
        guard.begin();
        model = models::gemma4::materialize_model(models::gemma4::plan_load(reader, load_options), device,
                                                  &options.startup_observer);
        const auto weights_shortfall = options.allow_shared_memory
            ? guard.end(model->storage_stats().device_capacity_bytes)
            : guard.end_without_host_allocation(model->storage_stats().device_capacity_bytes);
        if (weights_shortfall != 0) {
            if (!options.allow_shared_memory) {
                throw std::runtime_error("Gemma4 weights spilled into host memory: " +
                                         guard.readings(model->storage_stats().device_capacity_bytes));
            }
            shared_weights_bytes = weights_shortfall;
        }
        runtime_hold = DeviceBuffer();
        upload.complete();
    }
    const auto actual = budget->query();
    require_auto_readings(actual);
    std::optional<models::gemma4::ProgramDevicePlan> resolved_plan;
    try {
        resolved_plan = resolve_options(*model, actual.device_free);
    } catch (const std::runtime_error& error) {
        const auto mib = [](std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; };
        throw std::runtime_error(std::string(error.what()) + " [resident free after weights " +
                                 mib(actual.device_free) + " (CUDA " + mib(actual.cuda_free.value_or(0)) +
                                 ", physical " + mib(actual.physical_free.value_or(0)) + ", budget " +
                                 mib(actual.local_budget) + " less usage " + mib(actual.local_usage) +
                                 "), planned " + mib(available) + ", headroom " + mib(headroom) + "]");
    }
    const auto& request_plan = *resolved_plan;
    const auto curve = request_plan.capacity_curve();
    auto resolution = resolve_kv_capacity(options.kv_capacity, curve,
                                          shared_room(curve, actual.device_free), headroom);
    if (resolution.resolved_tokens < options.max_context) {
        throw std::runtime_error("Gemma4 KV capacity cannot hold --max-context; reduce context/chunk or free device memory");
    }
    SpillGuard runtime_guard(*budget);
    runtime_guard.begin();
    auto instance = std::make_unique<Gemma4Instance>(std::move(model), options, device);
    // Parameters already owns these planned bytes. Add them back instead of charging twice.
    const auto after_parameters = budget->query();
    require_auto_readings(after_parameters);
    const std::size_t runtime_room = after_parameters.device_free + instance->parameters.derived_device_bytes();
    resolution = resolve_kv_capacity(options.kv_capacity, curve, shared_room(curve, runtime_room), headroom);
    if (resolution.resolved_tokens < options.max_context) {
        const auto mib = [](std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; };
        throw std::runtime_error(
            "Gemma4 resident VRAM changed during startup; cannot hold --max-context and headroom [free after weights " +
            mib(actual.device_free) + ", after parameters " + mib(after_parameters.device_free) + " (CUDA " +
            mib(after_parameters.cuda_free.value_or(0)) + ", physical " + mib(after_parameters.physical_free.value_or(0)) +
            "), derived " + mib(instance->parameters.derived_device_bytes()) + ", room " + mib(runtime_room) +
            ", headroom " + mib(headroom) + ", resolved " + std::to_string(resolution.resolved_tokens) + " of " +
            std::to_string(options.max_context) + " tokens]");
    }
    instance->kv_capacity_resolution = resolution;
    {
        StartupPhaseScope construct(options.startup_observer, StartupPhase::ProgramInitialize);
        instance->program = std::make_unique<Program>(instance->parameters, device, program_options(options, prefix_bytes));
        construct.complete();
    }
    device.synchronize();
    const auto memory = instance->program->memory_summary();
    const auto allocated = request_plan.fixed_bytes() - request_plan.graph_allowance + memory.cuda_graph_measured_bytes;
    std::uint64_t shared_runtime_bytes = 0;
    if (const auto runtime_shortfall = runtime_guard.end(allocated); runtime_shortfall != 0) {
        if (!options.allow_shared_memory) {
            throw std::runtime_error("Gemma4 runtime spilled into host memory: " + runtime_guard.readings(allocated));
        }
        shared_runtime_bytes = runtime_shortfall;
    }
    const auto after_startup = budget->query();
    require_auto_readings(after_startup);
    if (options.max_context_auto && !options.allow_shared_memory && after_startup.device_free < headroom) {
        const auto mib = [](std::uint64_t bytes) { return std::to_string(bytes >> 20) + " MiB"; };
        throw std::runtime_error(
            "Gemma4 resident VRAM changed during startup; required headroom is no longer free. "
            "Free GPU memory and restart; host-paged startup is not permitted [free " +
            mib(after_startup.device_free) + " (CUDA " + mib(after_startup.cuda_free.value_or(0)) + ", physical " +
            mib(after_startup.physical_free.value_or(0)) + ", budget " + mib(after_startup.local_budget) +
            " less usage " + mib(after_startup.local_usage) + "), headroom " + mib(headroom) + ", context " +
            std::to_string(options.max_context) + "]");
    }
    instance->kv_capacity_resolution.available_after_startup_bytes = after_startup.device_free;
    instance->kv_capacity_resolution.shared_memory_weights_bytes = shared_weights_bytes;
    instance->kv_capacity_resolution.shared_memory_runtime_bytes = shared_runtime_bytes;
    // The Engine and the request log report the tier actually resolved (--host-context-mib auto).
    if (prefix_auto) { options.context_cache.host_capacity_bytes = prefix_bytes; }

    const auto& stats = instance->model->storage_stats();
    LoadSummary summary;
    summary.architecture = "Gemma4ForCausalLM";
    summary.model_name = instance->model->info().name;
    summary.prefill_signature = std::string("gemma4/") +
        (options.kv_cache == KvCacheStorage::Fp8E4M3Row256 ? "fp8" : "bf16") + "-kv" +
        (global_kv_storage(options) == models::gemma4::KvStorageModel::KeyOnly ? "/k-only" : "") +
        (options.gemma4_sliding_kv == KvCacheStorage::Fp8E4M3Row256 ? "/s-fp8" : "") + "/qk=" +
        gemma4_qk_name(options.gemma4_qk) +
        (options.gemma4_q5_prefill != Gemma4Q5Prefill::A16
             ? std::string("/q5=") + gemma4_q5_prefill_name(options.gemma4_q5_prefill)
             : std::string()) +
        "/cold/chunk=" + std::to_string(options.prefill_chunk);
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
