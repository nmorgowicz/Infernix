#pragma once

// Private workspace layout shared by token sampling and stochastic speculative acceptance.
// The same builder supplies the public sizing query and launcher-side binding.
//
// Two candidate capacities exist. The fast capacity (20) is the original pipeline: 512-item
// group tiles hold 25 partials. The wide capacity (64) serves top_k 21..64: its group stage
// sorts 2048-item tiles (256 threads x 8 items) that hold 32 partials, so one final merge
// still covers vocabularies up to 32 groups x 32 partials x 512 = 524288 tokens. A launch picks
// one capacity; rows whose top_k exceeds it are clamped to it. Speculative acceptance runs the
// fast capacity only.

#include "core/layout.h"
#include "infernix/ops/sampling.h"
#include "ops/common/math.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops {

inline constexpr int kSamplerBlock                 = 256;
inline constexpr int kSamplerTileItems             = 256;
inline constexpr int kSamplerItemsPerThread        = 2;
inline constexpr int kSamplerPartialTileItems      = kSamplerBlock * kSamplerItemsPerThread;
inline constexpr int kSamplerGroupBlock            = 256;
inline constexpr int kSamplerFastCandidates        = kSamplingFastTopK;
inline constexpr int kSamplerWideCandidates        = kSamplingMaxTopK;
inline constexpr int kSamplerMaxColumns            = 16;
inline constexpr int kSpeculativeSamplerMaxColumns = 64;

// Column limit of the multi-block route of speculative acceptance: the 20-wide route serves up to
// 64 verification columns, the wide route the sampler's 16 (wider rounds take the single-block
// fallback).
__host__ __device__ constexpr int sampler_speculative_max_columns(int cap) {
    return cap == kSamplerFastCandidates ? kSpeculativeSamplerMaxColumns : kSamplerMaxColumns;
}

static_assert(kSamplerFastCandidates == 20 && kSamplerWideCandidates == 64);

__host__ __device__ constexpr bool sampler_capacity_valid(int cap) {
    return cap == kSamplerFastCandidates || cap == kSamplerWideCandidates;
}

// Items each group-stage thread sorts. A group tile must hold partials_per_group * cap keys.
__host__ __device__ constexpr int sampler_group_items_per_thread(int cap) {
    return cap == kSamplerFastCandidates ? 2 : 8;
}

__host__ __device__ constexpr int sampler_group_tile_items(int cap) {
    return kSamplerGroupBlock * sampler_group_items_per_thread(cap);
}

__host__ __device__ constexpr int sampler_partials_per_group(int cap) {
    return sampler_group_tile_items(cap) / cap;
}

// The 20-wide constants keep their historical values; the speculative round uses them directly.
inline constexpr int kSamplerGroupItemsPerThread = sampler_group_items_per_thread(kSamplerFastCandidates);
inline constexpr int kSamplerGroupTileItems      = sampler_group_tile_items(kSamplerFastCandidates);
inline constexpr int kSamplerPartialsPerGroup    = sampler_partials_per_group(kSamplerFastCandidates);
inline constexpr int kSamplerCandidateCap        = kSamplerFastCandidates;

static_assert(kSamplerGroupTileItems == 512);
static_assert(kSamplerPartialsPerGroup == 25);
static_assert(kSamplerPartialsPerGroup * kSamplerFastCandidates <= kSamplerGroupTileItems,
              "group merge tile must hold one group's candidates");
static_assert(sampler_group_tile_items(kSamplerWideCandidates) == 2048);
static_assert(sampler_partials_per_group(kSamplerWideCandidates) == 32);
static_assert(sampler_partials_per_group(kSamplerWideCandidates) * kSamplerWideCandidates <=
                  sampler_group_tile_items(kSamplerWideCandidates),
              "group merge tile must hold one group's candidates");
// Every group-stage rank below cap must land in a thread's items.
static_assert(sampler_group_items_per_thread(kSamplerWideCandidates) * kSamplerGroupBlock >=
              kSamplerWideCandidates);
static_assert(kSamplerWideCandidates <= kSamplerPartialTileItems);

__host__ __device__ inline int sampler_group_count(int partial_blocks,
                                                   int cap = kSamplerFastCandidates) {
    return div_up(partial_blocks, sampler_partials_per_group(cap));
}

// The multi-block route is deliberately finite. A single final merge tile must
// hold every group candidate. Ordinary sampling retains its 16-column route;
// speculative verification explicitly opts into up to 64 columns.
__host__ __device__ inline bool sampler_multiblock_ok(int vocab, int cols, int partial_blocks,
                                                      int group_count,
                                                      int max_columns = kSamplerMaxColumns,
                                                      int cap         = kSamplerFastCandidates) {
    return vocab > kSamplerTileItems && cols > 0 && cols <= max_columns && partial_blocks > 0 &&
           group_count > 0 && (group_count * cap) <= sampler_group_tile_items(cap);
}

struct SamplingWorkspace {
    unsigned long long* partial_keys         = nullptr;
    std::int32_t* dist_idx                   = nullptr;
    float* dist_prob                         = nullptr;
    std::int32_t* dist_support               = nullptr;
    std::int32_t* group_done                 = nullptr;
    std::int32_t* speculative_finalize_count = nullptr;
    std::int32_t partial_stride              = 0;
};

struct SamplingWorkspaceLayout {
    TensorRegion partial_keys;
    TensorRegion dist_idx;
    TensorRegion dist_prob;
    TensorRegion dist_support;
    TensorRegion group_done;
    TensorRegion speculative_finalize_count;
    std::size_t bytes           = 0;
    std::int32_t partial_stride = 0;
    bool multiblock             = false;

    [[nodiscard]] SamplingWorkspace bind(DeviceSpan backing) const {
        if (!multiblock) { return {}; }
        return {
            static_cast<unsigned long long*>(partial_keys.bind(backing).data),
            static_cast<std::int32_t*>(dist_idx.bind(backing).data),
            static_cast<float*>(dist_prob.bind(backing).data),
            static_cast<std::int32_t*>(dist_support.bind(backing).data),
            static_cast<std::int32_t*>(group_done.bind(backing).data),
            static_cast<std::int32_t*>(speculative_finalize_count.bind(backing).data),
            partial_stride,
        };
    }
};

inline SamplingWorkspaceLayout
make_sampling_workspace_layout(std::int32_t token_domain, std::int32_t columns,
                               std::int32_t max_columns = kSamplerMaxColumns,
                               std::int32_t cap         = kSamplerFastCandidates) {
    SamplingWorkspaceLayout out;
    if (token_domain <= 0 || columns <= 0 || !sampler_capacity_valid(cap)) { return out; }

    const std::int32_t partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const std::int32_t groups         = sampler_group_count(partial_blocks, cap);
    if (!sampler_multiblock_ok(token_domain, columns, partial_blocks, groups, max_columns, cap)) {
        return out;
    }

    out.multiblock     = true;
    out.partial_stride = partial_blocks + groups;

    LayoutBuilder layout;
    out.partial_keys = layout.add_tensor(DType::I64, {cap, out.partial_stride, columns}, 256,
                                         "sampling partial keys");
    out.dist_idx     = layout.add_tensor(DType::I32, {cap, columns}, 256,
                                         "sampling speculative distribution indices");
    out.dist_prob    = layout.add_tensor(DType::FP32, {cap, columns}, 256,
                                         "sampling speculative distribution probabilities");
    out.dist_support =
        layout.add_tensor(DType::I32, {columns}, 256, "sampling speculative support sizes");
    out.group_done = layout.add_tensor(DType::I32, {columns}, 256, "sampling group counters");
    out.speculative_finalize_count =
        layout.add_tensor(DType::I32, {1}, 256, "sampling speculative finalize counter");
    // Speculative acceptance replicates this complete layout once per batch row. Preserve the
    // strongest member alignment in the row stride so every replicated partial-key array remains
    // correctly aligned.
    out.bytes = layout.finish(256, "sampling workspace");
    return out;
}

} // namespace infernix::ops
