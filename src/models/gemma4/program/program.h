#pragma once

// Native single-lane Gemma 4 execution. The prefix cache reserves its Host tier but binds every
// request at the root: no cached state is published, reused or persisted yet.
#include "models/gemma4/config.h"
#include "infernix/ops/gemma4_cache.h"
#include "models/gemma4/model.h"
#include "models/gemma4/frontend/frontend.h"
#include "models/gemma4/frontend/output_session.h"
#include "models/gemma4/frontend/prompt.h"
#include "runtime/contract/execution.h"
#include "runtime/contract/resources.h"
#include "runtime/prefix_cache/persist_result.h"
#include "runtime/prefix_cache/ssd_stats.h"

#include <array>
#include <chrono>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace infernix::models::gemma4 {
namespace execution { class Parameters; }
namespace detail { struct ProgramImpl; struct RequestPlan; struct ResumeData; }

struct PhysicalUsageSnapshot {
    runtime::ContextResourceUsage occupied;
    runtime::ContextResourceUsage capacity;
};
struct HybridPrefixCacheStats {
    std::uint32_t nodes = 0, snapshots = 0, device_resident_blocks = 0,
        device_evictable_blocks = 0, host_slabs = 0, host_free_slabs = 0;
    std::uint64_t host_slab_bytes = 0, snapshot_hits = 0, reused_tokens = 0,
        blocks_inserted = 0, blocks_reattached = 0, blocks_duplicate = 0,
        taps_created = 0, taps_skipped = 0, endpoints_created = 0,
        host_image_writes = 0, host_block_writes = 0, host_image_restores = 0,
        host_block_restores = 0, prefetched_blocks = 0, host_write_bytes = 0,
        host_restore_bytes = 0, evicted_blocks = 0, host_snapshot_evictions = 0,
        host_dead_reclaims = 0, unbacked_node_losses = 0;
    std::uint32_t held_snapshots = 0;
    std::uint64_t held_device_evictions = 0, held_snapshot_losses = 0,
        held_host_refusals = 0, tool_call_holds = 0;
    runtime::prefix_cache::SsdTierStats ssd;
};
class SequenceHandle {
public:
    SequenceHandle() noexcept = default;
    friend bool operator==(SequenceHandle, SequenceHandle) noexcept = default;
private:
    const void* owner_ = nullptr;
    std::uint32_t lane_ = 0;
    std::uint64_t epoch_ = 0;
    friend struct detail::ProgramImpl;
};
struct SourceCandidate {
    std::uint32_t reused_tokens = 0;
    PrefixReusePath reuse_path = PrefixReusePath::Root;
    std::uint32_t cached_block_tokens = 0;
};
struct BindingReservation {
    bool reserved = false;
    bool source_valid = true;
    bool capacity_possible = true;
    runtime::ContextResourceUsage shortage;
    std::exception_ptr refusal;
    explicit operator bool() const noexcept { return reserved; }
};
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept = default;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept = default;
    RequestBasePlan(const RequestBasePlan&) = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
private:
    explicit RequestBasePlan(std::shared_ptr<const detail::RequestPlan> plan) : impl_(std::move(plan)) {}
    std::shared_ptr<const detail::RequestPlan> impl_;
    friend class Program;
};
class ResumeState {
public:
    ResumeState(ResumeState&&) noexcept;
    ResumeState& operator=(ResumeState&&) noexcept;
    ~ResumeState();
    ResumeState(const ResumeState&) = delete;
    ResumeState& operator=(const ResumeState&) = delete;
    [[nodiscard]] std::uint32_t frontier() const noexcept;
private:
    explicit ResumeState(std::unique_ptr<detail::ResumeData>) noexcept;
    std::unique_ptr<detail::ResumeData> impl_;
    friend class Program;
};
enum class ExecutionUnitKind : std::uint8_t { Prefill, Replay, Decode, Control, Normalize };
struct ExecutionUnit {
    SequenceHandle sequence;
    ExecutionUnitKind kind = ExecutionUnitKind::Decode;
    std::uint32_t tokens = 0;
};
class PendingBatch {
public:
    PendingBatch() noexcept = default;
    PendingBatch(PendingBatch&& other) noexcept
        : owner_(std::exchange(other.owner_, nullptr)), transaction_(std::exchange(other.transaction_, 0)),
          sequence_(other.sequence_), tokens_(std::exchange(other.tokens_, {})), timing_(other.timing_),
          failure_(other.failure_) {}
    PendingBatch& operator=(PendingBatch&&) = delete;
    PendingBatch(const PendingBatch&) = delete;
    PendingBatch& operator=(const PendingBatch&) = delete;
    [[nodiscard]] std::size_t row_count() const noexcept { return owner_ ? 1 : 0; }
    [[nodiscard]] std::span<const TokenId> tokens() const noexcept { return tokens_; }
    [[nodiscard]] std::span<const std::int32_t> row_counts() const noexcept { return {}; }
    [[nodiscard]] std::uint32_t row_stride() const noexcept { return 1; }
    [[nodiscard]] runtime::ExecutionTiming execution_timing() const noexcept { return timing_; }
    [[nodiscard]] runtime::PendingRowFailure row_failure(std::size_t row) const {
        if (row >= row_count()) { throw std::out_of_range("Gemma4 pending row"); }
        return failure_;
    }
private:
    const void* owner_ = nullptr;
    std::uint64_t transaction_ = 0;
    SequenceHandle sequence_;
    std::span<const TokenId> tokens_;
    runtime::ExecutionTiming timing_;
    runtime::PendingRowFailure failure_ = runtime::PendingRowFailure::None;
    friend struct detail::ProgramImpl;
};
struct PrefillProgress {
    runtime::BeginSummary summary;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete = false;
    bool capacity_released = false;
    runtime::ExecutionTiming timing;
    std::optional<PendingBatch> pending;
    std::optional<PromptReadout> readout;
};
struct CommitRowResult {
    runtime::CommitDisposition disposition = runtime::CommitDisposition::Active;
    GenerationTimings timings;
    SpeculativeStats speculative;
    struct SpeculativeCounters {
        std::uint64_t rounds = 0, drafted_tokens = 0, accepted_tokens = 0, fallback_steps = 0;
    } speculative_counters;
};
struct CommitResult {
    std::array<CommitRowResult, kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
    runtime::ExecutionTiming timing;
};
struct DiscardResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    std::size_t row_count = 0;
};
struct FinishResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
    std::vector<ConstrainedDraw> constrained_draws;
    std::optional<ExpertCacheStats> expert_cache;
    std::optional<std::uint32_t> cache_created_prompt_tokens;
};
struct AbortResult {
    runtime::ConsumeStatus status = runtime::ConsumeStatus::InvariantMismatch;
    GenerationTimings timings;
    SpeculativeStats speculative;
};
struct ReplayProgress {
    std::uint32_t processed_tokens = 0;
    bool complete = false;
    runtime::ExecutionTiming timing;
};
enum class ContextOperationKind : std::uint8_t { Bind, Pause };
struct ContextProgress {
    ContextOperationKind kind = ContextOperationKind::Bind;
    bool advanced = false, complete = false, published = false;
    std::optional<SequenceHandle> sequence;
    std::optional<ResumeState> paused;
    bool replaying = false;
    runtime::ContextOperationCounts operations;
    std::optional<GenerationTimings> request_timings;
    SpeculativeStats request_speculative;
};
struct ProgramDevicePlan {
    std::uint32_t kv_pages = 0;
    // KV includes global pools (BF16, or FP8 codes plus FP16 scales) and BF16 sliding rings, plus both page tables.
    std::uint64_t kv = 0, kv_max = 0, workspace = 0, state = 0, io = 0;
    std::uint64_t prepared_row_scales = 0, graph_allowance = 0;
    std::uint32_t graph_bound = 0;
    runtime::SequenceCapacityCurve curve;
    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept { return curve; }
    [[nodiscard]] std::uint64_t fixed_bytes() const noexcept {
        return kv + workspace + state + io + prepared_row_scales + graph_allowance;
    }
};
struct ProgramOptions {
    std::uint32_t max_context = 0;
    std::uint32_t max_concurrency = 1;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t kv_capacity_tokens = 0;
    KvCacheStorage kv_cache = KvCacheStorage::BFloat16;
    ops::Gemma4QkMode qk_mode = ops::Gemma4QkMode::BFloat16;
    bool causal_scoring = false;
    bool prefix_cache = false;
    // The prefix cache's pinned Host tier in bytes; 0 disables its construction (a cold lane).
    std::uint64_t host_bytes = 0;
    bool reproducible = false;
    bool use_cuda_graph = true;
    DiagnosticObserver diagnostics;
    StartupObserver startup_observer;
};
class Program {
public:
    Program(const execution::Parameters&, DeviceContext&, ProgramOptions);
    ~Program() noexcept;
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    [[nodiscard]] static ProgramDevicePlan plan_device(const Model&, const ProgramOptions&, std::uint32_t public_tokens);
    [[nodiscard]] const ProgramDevicePlan& device_plan() const noexcept;
    // The prefix cache's pinned Host tier the Program was constructed with (0: none constructed).
    [[nodiscard]] std::uint64_t host_bytes() const noexcept;
    [[nodiscard]] RequestBasePlan plan_request(PreparedPrompt&&, const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] std::vector<SourceCandidate> hybrid_sources(const RequestBasePlan&, std::uint32_t);
    [[nodiscard]] BindingReservation start_binding(const RequestBasePlan&, runtime::LaneId,
        const SourceCandidate&, ResumeState* = nullptr, ExecutionUnitKind = ExecutionUnitKind::Decode, std::uint32_t = 1);
    [[nodiscard]] ContextProgress poll_context(runtime::CancellationFlagView);
    [[nodiscard]] bool has_context_transaction() const noexcept;
    [[nodiscard]] bool context_blocks(SequenceHandle) const noexcept;
    [[nodiscard]] runtime::ResourceReservation reserve_units(std::span<const ExecutionUnit>);
    void release_units(std::span<const SequenceHandle>) noexcept {}
    [[nodiscard]] bool recovery_pending(SequenceHandle) const noexcept { return false; }
    [[nodiscard]] bool start_pause(SequenceHandle, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] ReplayProgress advance_replay(SequenceHandle, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] PrefillProgress advance_prefill(SequenceHandle, runtime::ExecutionTiming* = nullptr,
        runtime::TokenMaskProvider* = nullptr);
    [[nodiscard]] PendingBatch decode(std::span<const SequenceHandle>, std::span<const runtime::RoundBudget>,
        runtime::ExecutionTiming* = nullptr, runtime::TokenMaskProvider* = nullptr);
    [[nodiscard]] runtime::ExecutionTiming append_forced_tokens(std::span<const SequenceHandle>,
        std::span<const TokenId>, std::uint32_t, std::span<const std::optional<std::uint32_t>>,
        runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] CommitResult commit(PendingBatch&&, std::span<const runtime::CommitDecision>,
        runtime::CommitObservation = runtime::CommitObservation::AllRows, runtime::ExecutionTiming* = nullptr);
    [[nodiscard]] DiscardResult abort_pending(PendingBatch&&) noexcept;
    [[nodiscard]] FinishResult finish(SequenceHandle) noexcept;
    [[nodiscard]] AbortResult abort(SequenceHandle) noexcept;
    void fail_all_cleanup() noexcept;
    [[nodiscard]] bool recover_after_failure() noexcept;
    void shutdown_cleanup() noexcept;
    // This dense cold-cache Program has no elastic Host prefix/expert tier, SSD transfers or
    // maintenance producer. Its fixed KV, graph operands and pinned staging remain reserved for
    // execution, so idle and between-round maintenance have no resize or deferred work to apply.
    void maintain() noexcept {}
    void maintain_busy() noexcept {}
    void set_maintenance_waker(std::function<void()>) noexcept {}
    [[nodiscard]] std::uint64_t engine_thread_cpus() const noexcept { return 0; }
    [[nodiscard]] std::optional<ExpertCacheTotals> expert_cache_totals() const noexcept { return std::nullopt; }
    [[nodiscard]] bool hybrid_reclaim(runtime::ContextResourceUsage) noexcept { return false; }
    [[nodiscard]] std::optional<std::uint32_t> hybrid_prefetch(const RequestBasePlan&) const noexcept { return 0; }
    [[nodiscard]] std::uint32_t hybrid_prefetch_room() const noexcept { return 0; }
    void hybrid_hold_queue(std::span<const RequestBasePlan* const>) noexcept {}
    void hybrid_hold_continuation(std::chrono::steady_clock::time_point);
    [[nodiscard]] bool hybrid_holds_stale() const noexcept { return false; }
    [[nodiscard]] std::uint64_t hybrid_cache_epoch() const noexcept { return 0; }
    [[nodiscard]] bool hybrid_poll_storage() noexcept { return false; }
    [[nodiscard]] bool hybrid_source_pending() const noexcept { return false; }
    [[nodiscard]] bool hybrid_storage_due() const noexcept { return false; }
    [[nodiscard]] std::uint32_t hybrid_resume_grid() const noexcept { return 1; }
    [[nodiscard]] HybridPrefixCacheStats hybrid_stats() const noexcept { return {}; }
    [[nodiscard]] runtime::prefix_cache::PersistResult attach_prefix_cache_file(
        const std::filesystem::path&, std::string, const StartupObserver&);
    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> prefix_shutdown_save() const { return std::nullopt; }
    [[nodiscard]] std::optional<runtime::prefix_cache::PersistResult> save_prefix_cache_now(const CancellationView&) { return std::nullopt; }
    [[nodiscard]] ScoreResult causal_score(PreparedPrompt&&, std::uint32_t, const ScoreOptions&);
    [[nodiscard]] PhysicalUsageSnapshot physical_usage() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    // What the device and host hold now, by use (RuntimeStats::memory; the prefix cache's Host and SSD tiers and the
    // headroom are the Engine's to fill). Read from the plan and arena sizes already known; engine thread, O(1).
    [[nodiscard]] MemoryUsage memory_usage() const noexcept;
    void reset_memory_peaks() noexcept;
private:
    std::unique_ptr<detail::ProgramImpl> impl_;
};
struct RuntimeTypes {
    using Frontend = gemma4::Frontend;
    using PreparedPrompt = gemma4::PreparedPrompt;
    using NgramArchive = gemma4::NgramArchive;
    using OutputSession = gemma4::OutputSession;
    using PublishedOutput = gemma4::PublishedOutput;
    using RequestBasePlan = gemma4::RequestBasePlan;
    using SequenceHandle = gemma4::SequenceHandle;
    using SourceCandidate = gemma4::SourceCandidate;
    using ResumeState = gemma4::ResumeState;
    using ExecutionUnit = gemma4::ExecutionUnit;
    using ExecutionUnitKind = gemma4::ExecutionUnitKind;
    using ContextProgress = gemma4::ContextProgress;
    using PendingBatch = gemma4::PendingBatch;
    using PrefillProgress = gemma4::PrefillProgress;
    using ReplayProgress = gemma4::ReplayProgress;
    using CommitResult = gemma4::CommitResult;
    using DiscardResult = gemma4::DiscardResult;
    using FinishResult = gemma4::FinishResult;
    using AbortResult = gemma4::AbortResult;
    using BindingReservation = gemma4::BindingReservation;
    using Program = gemma4::Program;
};
} // namespace infernix::models::gemma4
