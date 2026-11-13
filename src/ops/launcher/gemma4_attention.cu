#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/gemma4_fp8_kv.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kWarp       = 32;
constexpr int kMaxThreads = 1024;
// Enough resident warps to cover the device several times over before splitting keys.
constexpr int kTargetWarps = 16384;
// Grids of (tokens, kv heads) below half this many blocks split their keys into chunks.
constexpr int kTargetBlocks = 768;
constexpr int kMaxChunks    = 128;
// A split warp should see at least this many keys of its chunk.
constexpr int kMinKeysPerWarp = 4;
constexpr int kCombineThreads = 128;
// From one full 64-row query tile on, prefill goes through the tensor-core kernel.
constexpr int kMmaMinTokens = kGemma4AttentionMmaMinTokens;

template <int E>
__device__ __forceinline__ void load_row(const __nv_bfloat16* row, float (&out)[E]) {
    static_assert(E % 8 == 0);
#pragma unroll
    for (int chunk = 0; chunk < E / 8; ++chunk) {
        const uint4 packed = reinterpret_cast<const uint4*>(row)[chunk];
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const float2 value            = __bfloat1622float2(pairs[pair]);
            out[chunk * 8 + pair * 2]     = value.x;
            out[chunk * 8 + pair * 2 + 1] = value.y;
        }
    }
}

template <int E, Gemma4KvFormat Fmt>
__device__ __forceinline__ void load_cache(const void* pages, const __half* scales,
                                          std::int64_t base, float (&out)[E]) {
    if constexpr (Fmt == Gemma4KvFormat::Fp8E4m3Row256) {
        const auto* codes = static_cast<const std::uint8_t*>(pages) + base;
        const __half scale = scales[(base / 512) * 2 + (base % 512) / kGemma4KvFp8Group];
#pragma unroll
        for (int chunk = 0; chunk < E / 8; ++chunk) {
            const uint2 packed = reinterpret_cast<const uint2*>(codes)[chunk];
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&packed);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out[chunk * 8 + i] = __bfloat162float(gemma4_fp8_expand(bytes[i], scale));
            }
        }
    } else {
        load_row<E>(static_cast<const __nv_bfloat16*>(pages) + base, out);
    }
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarp / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

// grid (tokens, kv_heads, chunks); block = group * splits warps. Chunk c of a query covers the
// c-th of `chunks` equal parts of its visible keys. Warp (h, s) handles query head
// kvh * group + h over the chunk's keys begin + s, begin + s + splits, ... with an online
// softmax, holding D / 32 contiguous dimensions per lane. The splits of one head are merged in
// shared memory. With one chunk the block writes the BF16 output; otherwise it writes its
// normalized FP32 result and (max, sum) for gemma4_attention_combine_kernel.
template <int D, Gemma4KvFormat Fmt>
__global__ void gemma4_attention_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pages,
    const void* __restrict__ v_pages, const __half* __restrict__ k_scales,
    const __half* __restrict__ v_scales, const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ out,
    float* __restrict__ partial, float* __restrict__ partial_stats, std::int32_t q_heads,
    std::int32_t kv_heads, int group, int splits, int chunks, int window, float scale,
    std::int32_t logical_pages, std::int32_t physical_pages) {
    constexpr int E = D / kWarp;
    extern __shared__ float shared[];
    float* merged = shared;             // [group][D]
    float* stats  = shared + group * D; // [group][splits][2]

    const int token         = blockIdx.x;
    const int kvh           = blockIdx.y;
    const int chunk         = blockIdx.z;
    const int warp          = threadIdx.x / kWarp;
    const int lane          = threadIdx.x % kWarp;
    const int head_in_group = warp / splits;
    const int split         = warp - head_in_group * splits;
    const int qh            = kvh * group + head_in_group;
    const int qpos          = positions[token];
    const int first         = window == 0 ? 0 : max(0, qpos + 1 - window);
    const int span          = (qpos + 1 - first + chunks - 1) / chunks;
    const int begin         = first + chunk * span;
    const int end           = min(qpos + 1, begin + span);

    for (int index = threadIdx.x; index < group * D; index += blockDim.x) { merged[index] = 0.0f; }

    float qv[E];
    load_row<E>(q + (static_cast<std::int64_t>(token) * q_heads + qh) * D + lane * E, qv);
#pragma unroll
    for (int dim = 0; dim < E; ++dim) { qv[dim] *= scale; }

    float acc[E];
#pragma unroll
    for (int dim = 0; dim < E; ++dim) { acc[dim] = 0.0f; }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    for (int key = begin + split; key < end; key += splits) {
        const int page = key >> kPagedKVPageShift;
        if (page >= logical_pages) { continue; }
        const int physical = block_table[page];
        if (physical < 0 || physical >= physical_pages) { continue; }
        const std::int64_t base =
            static_cast<std::int64_t>(D) * kPagedKVPageSize *
                (static_cast<std::int64_t>(kvh) + static_cast<std::int64_t>(kv_heads) * physical) +
            static_cast<std::int64_t>(D) * (key & kPagedKVPageMask) + lane * E;
        float kv[E];
        load_cache<E, Fmt>(k_pages, k_scales, base, kv);
        float dot = 0.0f;
#pragma unroll
        for (int dim = 0; dim < E; ++dim) { dot = fmaf(qv[dim], kv[dim], dot); }
        const float score    = warp_sum(dot);
        const float next_max = fmaxf(running_max, score);
        const float rescale  = expf(running_max - next_max);
        const float weight   = expf(score - next_max);
        running_sum          = running_sum * rescale + weight;
        running_max          = next_max;
        load_cache<E, Fmt>(v_pages, v_scales, base, kv);
#pragma unroll
        for (int dim = 0; dim < E; ++dim) { acc[dim] = fmaf(acc[dim], rescale, weight * kv[dim]); }
    }

    if (lane == 0) {
        stats[(head_in_group * splits + split) * 2]     = running_max;
        stats[(head_in_group * splits + split) * 2 + 1] = running_sum;
    }
    __syncthreads();
    float head_max = -INFINITY;
    for (int other = 0; other < splits; ++other) {
        head_max = fmaxf(head_max, stats[(head_in_group * splits + other) * 2]);
    }
    float head_sum = 0.0f;
    for (int other = 0; other < splits; ++other) {
        const float other_max = stats[(head_in_group * splits + other) * 2];
        if (other_max != -INFINITY) {
            head_sum +=
                stats[(head_in_group * splits + other) * 2 + 1] * expf(other_max - head_max);
        }
    }
    const bool contributes = running_max != -INFINITY && head_sum > 0.0f;
    const float factor     = contributes ? expf(running_max - head_max) / head_sum : 0.0f;
    float* row             = merged + head_in_group * D + lane * E;
    // Splits add in a fixed order so the result does not depend on scheduling.
    for (int turn = 0; turn < splits; ++turn) {
        if (turn == split && contributes) {
#pragma unroll
            for (int dim = 0; dim < E; ++dim) { row[dim] += acc[dim] * factor; }
        }
        __syncthreads();
    }
    const std::int64_t first_head =
        static_cast<std::int64_t>(token) * q_heads + static_cast<std::int64_t>(kvh) * group;
    if (chunks == 1) {
        __nv_bfloat16* destination = out + first_head * D;
        for (int index = threadIdx.x; index < group * D; index += blockDim.x) {
            destination[index] = __float2bfloat16_rn(merged[index]);
        }
        return;
    }
    if (split == 0 && lane == 0) {
        float* head_stats = partial_stats + ((first_head + head_in_group) * chunks + chunk) * 2;
        head_stats[0]     = head_max;
        head_stats[1]     = head_sum;
    }
    for (int index = threadIdx.x; index < group * D; index += blockDim.x) {
        const int head                                                         = index / D;
        partial[((first_head + head) * chunks + chunk) * D + index - head * D] = merged[index];
    }
}

// grid (T * Hq); merges the chunks of one query head in chunk order.
template <int D>
__global__ void gemma4_attention_combine_kernel(const float* __restrict__ partial,
                                                const float* __restrict__ partial_stats,
                                                __nv_bfloat16* __restrict__ out, int chunks) {
    constexpr int E         = D / kCombineThreads;
    const std::int64_t head = blockIdx.x;
    const float* head_stats = partial_stats + head * chunks * 2;
    float head_max          = -INFINITY;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        head_max = fmaxf(head_max, head_stats[chunk * 2]);
    }
    float total = 0.0f;
    float acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const float chunk_max = head_stats[chunk * 2];
        if (chunk_max == -INFINITY) { continue; }
        const float weight = head_stats[chunk * 2 + 1] * expf(chunk_max - head_max);
        total += weight;
        const float* row = partial + (head * chunks + chunk) * D + threadIdx.x;
#pragma unroll
        for (int e = 0; e < E; ++e) { acc[e] = fmaf(weight, row[e * kCombineThreads], acc[e]); }
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        out[head * D + threadIdx.x + e * kCombineThreads] = __float2bfloat16_rn(acc[e] * inverse);
    }
}

template <int D, Gemma4KvFormat Fmt>
void launch(const Tensor& q, const Gemma4CacheView& cache,
            const Tensor& positions, int window, float scale, Tensor& out,
            float* partial, float* partial_stats, cudaStream_t stream) {
    const Tensor& k_pages = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    const int q_heads        = q.ne[1];
    const int tokens         = q.ne[2];
    const int kv_heads       = k_pages.ne[2];
    const int group          = q_heads / kv_heads;
    const int chunks         = gemma4_attention_chunks(tokens, kv_heads);
    const int keys           = window > 0 ? window : block_table.ne[0] * kPagedKVPageSize;
    const int keys_per_chunk = (keys + chunks - 1) / chunks;
    const int max_splits     = std::max(1, kMaxThreads / (kWarp * group));
    int splits               = 1;
    while (splits * 2 <= max_splits && tokens * q_heads * chunks * splits < kTargetWarps &&
           splits * 2 * kMinKeysPerWarp <= keys_per_chunk) {
        splits *= 2;
    }
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(kv_heads),
                    static_cast<unsigned>(chunks));
    const int threads        = kWarp * group * splits;
    const std::size_t shared = sizeof(float) * (static_cast<std::size_t>(group) * D +
                                                static_cast<std::size_t>(group) * splits * 2);
    if (shared > 48U * 1024U) {
        throw std::invalid_argument("gemma4_attention: query group is too wide for shared memory");
    }
    if (chunks > 1 && (partial == nullptr || partial_stats == nullptr)) {
        throw std::invalid_argument("gemma4_attention: split keys need workspace");
    }
    constexpr bool Quantized = Fmt != Gemma4KvFormat::BFloat16;
    gemma4_attention_kernel<D, Fmt><<<grid, threads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), k_pages.data, cache.v_pages->data,
        Quantized ? static_cast<const __half*>(cache.k_scales->data) : nullptr,
        Quantized ? static_cast<const __half*>(cache.v_scales->data) : nullptr,
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        partial, partial_stats, q_heads, kv_heads, group, splits, chunks, window, scale,
        block_table.ne[0], k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
    if (chunks == 1) { return; }
    gemma4_attention_combine_kernel<D>
        <<<static_cast<unsigned>(tokens * q_heads), kCombineThreads, 0, stream>>>(
            partial, partial_stats, static_cast<__nv_bfloat16*>(out.data), chunks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

int gemma4_attention_chunks(std::int32_t tokens, std::int32_t kv_heads) {
    const std::int64_t blocks = static_cast<std::int64_t>(tokens) * kv_heads;
    if (blocks * 2 > kTargetBlocks) { return 1; }
    return static_cast<int>(std::min<std::int64_t>(kMaxChunks, kTargetBlocks / blocks));
}

int gemma4_attention_simt_chunks(std::int32_t tokens, std::int32_t q_heads,
                                 std::int32_t kv_heads, std::int32_t head_dim) {
    return gemma4_decode_d512_selected(head_dim, q_heads, kv_heads)
               ? gemma4_decode_d512_chunks(tokens, kv_heads)
               : gemma4_attention_chunks(tokens, kv_heads);
}

void gemma4_attention_launch(const Tensor& q, const Gemma4CacheView& cache,
                             const Tensor& positions, int window, float scale, Tensor& out,
                             float* partial, float* stats, cudaStream_t stream) {
    if (q.ne[1] / cache.k_pages->ne[2] * kWarp > kMaxThreads) {
        throw std::invalid_argument("gemma4_attention: at most 32 query heads per KV head");
    }
    if (q.ne[2] >= kMmaMinTokens) {
        gemma4_attention_mma_launch(q, cache, positions, window, scale, out, partial, stats,
                                    stream);
        return;
    }
    if (gemma4_decode_d512_selected(q.ne[0], q.ne[1], cache.k_pages->ne[2])) {
        gemma4_attention_decode_d512_launch(q, cache, positions, window, scale, out, partial,
                                            stats, stream);
        return;
    }
    if (q.ne[0] == 256) {
        launch<256, Gemma4KvFormat::BFloat16>(q, cache, positions, window, scale, out, partial,
                                              stats, stream);
    } else if (cache.format == Gemma4KvFormat::Fp8E4m3Row256) {
        launch<512, Gemma4KvFormat::Fp8E4m3Row256>(q, cache, positions, window, scale, out,
                                                   partial, stats, stream);
    } else {
        launch<512, Gemma4KvFormat::BFloat16>(q, cache, positions, window, scale, out, partial,
                                              stats, stream);
    }
}

} // namespace infernix::ops::detail
