#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kD       = 512;
constexpr int kWarp    = 32;
constexpr int kPerLane = kD / kWarp; // 16 contiguous dimensions per lane
// One tile is kTile keys. The block has one warp per key of a tile for staging and, in the
// compute phase, one warp per query head of the group (group <= kTile).
constexpr int kTile    = 8;
constexpr int kThreads = kTile * kWarp;
static_assert(kGemma4DecodeD512MaxGroup <= kTile);
// Resident CTAs per SM the launch bounds ask for; the chunk count fills whole waves of it.
constexpr int kBlocksPerSm = 2;
constexpr int kMaxChunks   = 128;
// A chunk covers at least this many keys (a multiple of kTile): short contexts leave the
// trailing chunks empty instead of splitting a few keys across many CTAs.
constexpr int kMinChunkKeys = 128;
static_assert(kMinChunkKeys % kTile == 0);
constexpr int kCombineThreads = 128;
constexpr int kCombineBatch   = 8;

__device__ __forceinline__ float bf16_low(std::uint32_t word) { return __uint_as_float(word << 16); }
__device__ __forceinline__ float bf16_high(std::uint32_t word) {
    return __uint_as_float(word & 0xffff0000U);
}

// Eight BF16 values (one uint4) times 8 floats, accumulated in dimension order.
__device__ __forceinline__ float dot8(const float* q, const uint4& packed, float dot) {
    dot = fmaf(q[0], bf16_low(packed.x), dot);
    dot = fmaf(q[1], bf16_high(packed.x), dot);
    dot = fmaf(q[2], bf16_low(packed.y), dot);
    dot = fmaf(q[3], bf16_high(packed.y), dot);
    dot = fmaf(q[4], bf16_low(packed.z), dot);
    dot = fmaf(q[5], bf16_high(packed.z), dot);
    dot = fmaf(q[6], bf16_low(packed.w), dot);
    dot = fmaf(q[7], bf16_high(packed.w), dot);
    return dot;
}

__device__ __forceinline__ void axpy8(float weight, const uint4& packed, float* acc) {
    acc[0] = fmaf(weight, bf16_low(packed.x), acc[0]);
    acc[1] = fmaf(weight, bf16_high(packed.x), acc[1]);
    acc[2] = fmaf(weight, bf16_low(packed.y), acc[2]);
    acc[3] = fmaf(weight, bf16_high(packed.y), acc[3]);
    acc[4] = fmaf(weight, bf16_low(packed.z), acc[4]);
    acc[5] = fmaf(weight, bf16_high(packed.z), acc[5]);
    acc[6] = fmaf(weight, bf16_low(packed.w), acc[6]);
    acc[7] = fmaf(weight, bf16_high(packed.w), acc[7]);
}

// A lane's raw slice of one K or V row: dimensions lane * 16 .. + 15.
template <Gemma4KvFormat Fmt>
struct RawRow {
    static constexpr int kVectors = Fmt == Gemma4KvFormat::BFloat16 ? 2 : 1;
    uint4 code[kVectors];
    float scale;
};

// One 16-byte load per lane for the quantized codes (two for BF16); the row is a warp-wide
// contiguous 512 B (1 KiB for BF16). The group scale is a single FP16 load.
template <Gemma4KvFormat Fmt>
__device__ __forceinline__ void load_raw(RawRow<Fmt>& raw, const void* pages, const __half* scales,
                                         std::int64_t row, int lane) {
    if constexpr (Fmt == Gemma4KvFormat::BFloat16) {
        const auto* source = reinterpret_cast<const uint4*>(
            static_cast<const __nv_bfloat16*>(pages) + row * kD + lane * kPerLane);
        raw.code[0] = __ldg(source);
        raw.code[1] = __ldg(source + 1);
        raw.scale   = 1.0f;
    } else {
        const auto* source = reinterpret_cast<const uint4*>(
            static_cast<const std::uint8_t*>(pages) + row * kD + lane * kPerLane);
        raw.code[0] = __ldg(source);

            // Lanes 0..15 hold dimensions 0..255, lanes 16..31 dimensions 256..511.
            raw.scale = __half2float(__ldg(scales + row * 2 + (lane >> 4)));

    }
}

// Two consecutive codes (low 16 bits) -> one packed BF16 pair. The represented value is the
// shared one: FP32(code) x FP32(scale) rounded to BF16 (gemma4_fp8_expand).
template <Gemma4KvFormat Fmt>
__device__ __forceinline__ std::uint32_t expand_pair(std::uint32_t codes, float scale) {
    float low;
    float high;

        // E4M3 -> FP16 is exact, FP16 -> FP32 is exact: the same value as static_cast<float>.
        const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(
            static_cast<__nv_fp8x2_storage_t>(codes & 0xffffU), __NV_E4M3);
        const float2 value = __half22float2(__half2(pair));
        low                = value.x;
        high               = value.y;

    const __nv_bfloat162 packed =
        __floats2bfloat162_rn(__fmul_rn(low, scale), __fmul_rn(high, scale));
    return *reinterpret_cast<const std::uint32_t*>(&packed);
}

template <Gemma4KvFormat Fmt>
__device__ __forceinline__ uint4 expand8(std::uint32_t first, std::uint32_t second, float scale) {
    uint4 out;
    out.x = expand_pair<Fmt>(first, scale);
    out.y = expand_pair<Fmt>(first >> 16, scale);
    out.z = expand_pair<Fmt>(second, scale);
    out.w = expand_pair<Fmt>(second >> 16, scale);
    return out;
}

// Writes the BF16 row into shared memory as 64 uint4: slot lane holds dimensions
// lane * 16 .. + 7, slot 32 + lane dimensions lane * 16 + 8 .. + 15, so both 128-bit reads of
// the compute phase are conflict-free.
template <Gemma4KvFormat Fmt>
__device__ __forceinline__ void store_row(const RawRow<Fmt>& raw, uint4* row, int lane) {
    if constexpr (Fmt == Gemma4KvFormat::BFloat16) {
        row[lane]      = raw.code[0];
        row[32 + lane] = raw.code[1];
    } else {
        row[lane]      = expand8<Fmt>(raw.code[0].x, raw.code[0].y, raw.scale);
        row[32 + lane] = expand8<Fmt>(raw.code[0].z, raw.code[0].w, raw.scale);
    }
}

// Physical row index of `key`, or false when the key is outside the chunk or its page is not
// mapped (such keys do not take part in the softmax, as in the original kernel).
__device__ __forceinline__ bool key_row(int key, int end, int kvh, int kv_heads,
                                        const std::int32_t* block_table,
                                        std::int32_t logical_pages, std::int32_t physical_pages,
                                        std::int64_t& row) {
    if (key >= end) { return false; }
    const int page = key >> kPagedKVPageShift;
    if (page >= logical_pages) { return false; }
    const int physical = block_table[page];
    if (physical < 0 || physical >= physical_pages) { return false; }
    row = (static_cast<std::int64_t>(kvh) + static_cast<std::int64_t>(kv_heads) * physical) *
              kPagedKVPageSize +
          (key & kPagedKVPageMask);
    return true;
}

// grid (tokens, kv_heads, chunks); block = 8 warps. Chunk c of a query covers the c-th of
// `chunks` equal parts of its visible keys, widened to at least kMinChunkKeys and rounded up
// to whole tiles of kTile keys; positions are read on the device, so the grid depends only on
// (tokens, kv_heads) and replays under a captured graph.
//
// Each tile of kTile keys is loaded once per KV head: warp w loads key w of the tile (16 B
// per lane), expands it to BF16 once and stores it in shared memory. All query heads of the
// group then read the tile from shared memory (warp h owns head kvh * group + h; its softmax
// state and a 16-dimension output slice per lane live in registers). The next tile's global
// loads are issued before the current tile is reduced, and tiles alternate between two shared
// buffers, so a tile costs one __syncthreads.
//
// Numerics: the represented K/V values are unchanged. Per query head the score sum runs in
// FP32 over dimension order per lane and then a different shuffle tree than the original
// kernel, and the softmax is updated once per tile of 8 keys instead of once per key, so
// results agree with the original kernel to FP32 rounding, not bitwise.
//
// With one chunk the block writes the BF16 output; otherwise it writes its normalized FP32
// result and (max, sum) for gemma4_decode_d512_combine_kernel, with the contract of
// gemma4_attention_combine_kernel. An empty chunk writes max = -inf, sum = 0 and no partial.
template <Gemma4KvFormat Fmt>
__global__ void __launch_bounds__(kThreads, kBlocksPerSm) gemma4_decode_d512_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pages,
    const void* __restrict__ v_pages, const __half* __restrict__ k_scales,
    const __half* __restrict__ v_scales, const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ out,
    float* __restrict__ partial, float* __restrict__ partial_stats, std::int32_t q_heads,
    std::int32_t kv_heads, int group, int chunks, int window, float scale,
    std::int32_t logical_pages, std::int32_t physical_pages) {
    __shared__ __align__(16) uint4 k_tile[2][kTile][kD / 8];
    __shared__ __align__(16) uint4 v_tile[2][kTile][kD / 8];
    __shared__ unsigned char key_valid[2][kTile];

    const int token = blockIdx.x;
    const int kvh   = blockIdx.y;
    const int chunk = blockIdx.z;
    const int warp  = threadIdx.x / kWarp;
    const int lane  = threadIdx.x % kWarp;
    const int qpos  = positions[token];
    const int first = window == 0 ? 0 : max(0, qpos + 1 - window);
    const int visible = max(qpos + 1 - first, 0);
    int span          = (visible + chunks - 1) / chunks;
    span              = (max(span, kMinChunkKeys) + kTile - 1) / kTile * kTile;
    const int begin   = first + chunk * span;
    const int end     = min(qpos + 1, begin + span);
    const int tiles   = end > begin ? (end - begin + kTile - 1) / kTile : 0;

    const bool active = warp < group;
    const int qh      = kvh * group + warp;
    float qv[kPerLane];
    float acc[kPerLane];
#pragma unroll
    for (int dim = 0; dim < kPerLane; ++dim) { acc[dim] = 0.0f; }
    if (active) {
        const auto* source = reinterpret_cast<const uint4*>(
            q + (static_cast<std::int64_t>(token) * q_heads + qh) * kD + lane * kPerLane);
        const uint4 low  = source[0];
        const uint4 high = source[1];
        const std::uint32_t words[8]{low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qv[2 * i]     = bf16_low(words[i]) * scale;
            qv[2 * i + 1] = bf16_high(words[i]) * scale;
        }
    } else {
#pragma unroll
        for (int dim = 0; dim < kPerLane; ++dim) { qv[dim] = 0.0f; }
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    RawRow<Fmt> k_raw;
    RawRow<Fmt> v_raw;
    // Issues the global loads of the key this warp stages for `tile`; returns whether it is
    // a live key. The result is uniform across the warp.
    const auto fetch = [&](int tile) {
        std::int64_t row = 0;
        const bool valid = key_row(begin + tile * kTile + warp, end, kvh, kv_heads, block_table,
                                   logical_pages, physical_pages, row);
        if (valid) {
            load_raw<Fmt>(k_raw, k_pages, k_scales, row, lane);
            load_raw<Fmt>(v_raw, v_pages, v_scales, row, lane);
        }
        return valid;
    };
    // Expands the fetched rows into shared buffer `buffer`; dead keys become zero rows.
    const auto commit = [&](int buffer, bool valid) {
        uint4* k_row = k_tile[buffer][warp];
        uint4* v_row = v_tile[buffer][warp];
        if (valid) {
            store_row<Fmt>(k_raw, k_row, lane);
            store_row<Fmt>(v_raw, v_row, lane);
        } else {
            const uint4 zero{0U, 0U, 0U, 0U};
            k_row[lane]      = zero;
            k_row[32 + lane] = zero;
            v_row[lane]      = zero;
            v_row[32 + lane] = zero;
        }
        if (lane == 0) { key_valid[buffer][warp] = valid ? 1 : 0; }
    };

    if (tiles > 0) { commit(0, fetch(0)); }
    __syncthreads();

    for (int tile = 0; tile < tiles; ++tile) {
        const int buffer = tile & 1;
        bool next_valid  = false;
        if (tile + 1 < tiles) { next_valid = fetch(tile + 1); }

        if (active) {
            // Scores of the tile's keys for this head: lane partials, then a transposed
            // butterfly (4 + 2 + 1 shuffles instead of 8 x 5) leaves key lane / 4 complete
            // in every lane of its 4-lane group.
            float s[kTile];
#pragma unroll
            for (int key = 0; key < kTile; ++key) {
                const uint4* row = k_tile[buffer][key];
                float dot        = dot8(qv, row[lane], 0.0f);
                s[key]           = dot8(qv + 8, row[32 + lane], dot);
            }
            const bool upper16 = (lane & 16) != 0;
            float t[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float send = upper16 ? s[i] : s[i + 4];
                const float keep = upper16 ? s[i + 4] : s[i];
                t[i]             = keep + __shfl_xor_sync(0xffffffffU, send, 16);
            }
            const bool upper8 = (lane & 8) != 0;
            float u[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float send = upper8 ? t[i] : t[i + 2];
                const float keep = upper8 ? t[i + 2] : t[i];
                u[i]             = keep + __shfl_xor_sync(0xffffffffU, send, 8);
            }
            const bool upper4 = (lane & 4) != 0;
            const float send  = upper4 ? u[0] : u[1];
            const float keep  = upper4 ? u[1] : u[0];
            float score       = keep + __shfl_xor_sync(0xffffffffU, send, 4);
            score += __shfl_xor_sync(0xffffffffU, score, 2);
            score += __shfl_xor_sync(0xffffffffU, score, 1);
            if (key_valid[buffer][lane >> 2] == 0) { score = -INFINITY; }

            float tile_max = score;
            tile_max       = fmaxf(tile_max, __shfl_xor_sync(0xffffffffU, tile_max, 16));
            tile_max       = fmaxf(tile_max, __shfl_xor_sync(0xffffffffU, tile_max, 8));
            tile_max       = fmaxf(tile_max, __shfl_xor_sync(0xffffffffU, tile_max, 4));
            // tile_max is warp-uniform; a tile without a live key leaves the state untouched.
            if (tile_max != -INFINITY) {
                const float next_max = fmaxf(running_max, tile_max);
                const float rescale  = expf(running_max - next_max);
                const float weight   = expf(score - next_max);
                float tile_sum       = weight;
                tile_sum += __shfl_xor_sync(0xffffffffU, tile_sum, 16);
                tile_sum += __shfl_xor_sync(0xffffffffU, tile_sum, 8);
                tile_sum += __shfl_xor_sync(0xffffffffU, tile_sum, 4);
                running_sum = running_sum * rescale + tile_sum;
                running_max = next_max;
#pragma unroll
                for (int dim = 0; dim < kPerLane; ++dim) { acc[dim] *= rescale; }
#pragma unroll
                for (int key = 0; key < kTile; ++key) {
                    const float p    = __shfl_sync(0xffffffffU, weight, key * 4);
                    const uint4* row = v_tile[buffer][key];
                    axpy8(p, row[lane], acc);
                    axpy8(p, row[32 + lane], acc + 8);
                }
            }
        }

        if (tile + 1 < tiles) { commit(buffer ^ 1, next_valid); }
        __syncthreads();
    }

    if (!active) { return; }
    const bool contributes = running_max != -INFINITY && running_sum > 0.0f;
    const float factor     = contributes ? 1.0f / running_sum : 0.0f;
    const std::int64_t head = static_cast<std::int64_t>(token) * q_heads + qh;
    if (chunks == 1) {
        uint4 packed[2];
        auto* words = reinterpret_cast<std::uint32_t*>(packed);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const __nv_bfloat162 pair =
                __floats2bfloat162_rn(acc[2 * i] * factor, acc[2 * i + 1] * factor);
            words[i] = *reinterpret_cast<const std::uint32_t*>(&pair);
        }
        auto* destination = reinterpret_cast<uint4*>(out + head * kD + lane * kPerLane);
        destination[0]    = packed[0];
        destination[1]    = packed[1];
        return;
    }
    if (lane == 0) {
        float* stats = partial_stats + (head * chunks + chunk) * 2;
        stats[0]     = contributes ? running_max : -INFINITY;
        stats[1]     = contributes ? running_sum : 0.0f;
    }
    if (contributes) {
        auto* destination =
            reinterpret_cast<float4*>(partial + (head * chunks + chunk) * kD + lane * kPerLane);
#pragma unroll
        for (int i = 0; i < kPerLane / 4; ++i) {
            destination[i] = float4{acc[4 * i] * factor, acc[4 * i + 1] * factor,
                                    acc[4 * i + 2] * factor, acc[4 * i + 3] * factor};
        }
    }
}

// grid (T * Hq); merges the chunks of one query head in chunk order, with the arithmetic of
// gemma4_attention_combine_kernel. The chunk weights are computed once in shared memory and
// the partial rows are read kCombineBatch chunks at a time to overlap the loads.
__global__ void __launch_bounds__(kCombineThreads) gemma4_decode_d512_combine_kernel(
    const float* __restrict__ partial, const float* __restrict__ partial_stats,
    __nv_bfloat16* __restrict__ out, int chunks) {
    constexpr int E = kD / kCombineThreads;
    __shared__ float chunk_max[kMaxChunks];
    __shared__ float chunk_weight[kMaxChunks];
    const std::int64_t head = blockIdx.x;
    const float* head_stats = partial_stats + head * chunks * 2;
    for (int chunk = threadIdx.x; chunk < chunks; chunk += kCombineThreads) {
        chunk_max[chunk]    = head_stats[chunk * 2];
        chunk_weight[chunk] = head_stats[chunk * 2 + 1];
    }
    __syncthreads();
    float head_max = -INFINITY;
    for (int chunk = 0; chunk < chunks; ++chunk) { head_max = fmaxf(head_max, chunk_max[chunk]); }
    // A negative weight marks a chunk without keys; real weights are never negative.
    for (int chunk = threadIdx.x; chunk < chunks; chunk += kCombineThreads) {
        const float max_value = chunk_max[chunk];
        chunk_weight[chunk] =
            max_value == -INFINITY ? -1.0f : chunk_weight[chunk] * expf(max_value - head_max);
    }
    __syncthreads();
    float total = 0.0f;
    float acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
    for (int base = 0; base < chunks; base += kCombineBatch) {
        float weight[kCombineBatch];
        float row[kCombineBatch][E];
#pragma unroll
        for (int u = 0; u < kCombineBatch; ++u) {
            const int chunk = base + u;
            weight[u]       = chunk < chunks ? chunk_weight[chunk] : -1.0f;
#pragma unroll
            for (int e = 0; e < E; ++e) { row[u][e] = 0.0f; }
            if (weight[u] >= 0.0f) {
                const float* source = partial + (head * chunks + chunk) * kD + threadIdx.x;
#pragma unroll
                for (int e = 0; e < E; ++e) { row[u][e] = source[e * kCombineThreads]; }
            }
        }
#pragma unroll
        for (int u = 0; u < kCombineBatch; ++u) {
            if (weight[u] >= 0.0f) {
                total += weight[u];
#pragma unroll
                for (int e = 0; e < E; ++e) { acc[e] = fmaf(weight[u], row[u][e], acc[e]); }
            }
        }
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        out[head * kD + threadIdx.x + e * kCombineThreads] = __float2bfloat16_rn(acc[e] * inverse);
    }
}

int device_sm_count() {
    static const int sm_count = [] {
        int device = 0;
        int count  = 1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) !=
                cudaSuccess) {
            return 1;
        }
        return std::max(count, 1);
    }();
    return sm_count;
}

template <Gemma4KvFormat Fmt>
void launch(const Tensor& q, const Gemma4CacheView& cache, const Tensor& positions, int window,
            float scale, Tensor& out, float* partial, float* partial_stats, cudaStream_t stream) {
    constexpr bool Quantized = Fmt != Gemma4KvFormat::BFloat16;
    const Tensor& k_pages     = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    const int q_heads         = q.ne[1];
    const int tokens          = q.ne[2];
    const int kv_heads        = k_pages.ne[2];
    const int group           = q_heads / kv_heads;
    const int chunks          = gemma4_decode_d512_chunks(tokens, kv_heads);
    if (chunks > 1 && (partial == nullptr || partial_stats == nullptr)) {
        throw std::invalid_argument("gemma4_attention: split keys need workspace");
    }
    const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(kv_heads),
                    static_cast<unsigned>(chunks));
    gemma4_decode_d512_kernel<Fmt><<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), k_pages.data, cache.v_pages->data,
        Quantized ? static_cast<const __half*>(cache.k_scales->data) : nullptr,
        Quantized ? static_cast<const __half*>(cache.v_scales->data) : nullptr,
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        partial, partial_stats, q_heads, kv_heads, group, chunks, window, scale, block_table.ne[0],
        k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
    if (chunks == 1) { return; }
    gemma4_decode_d512_combine_kernel<<<static_cast<unsigned>(tokens * q_heads),
                                        kCombineThreads, 0, stream>>>(
        partial, partial_stats, static_cast<__nv_bfloat16*>(out.data), chunks);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

bool gemma4_decode_d512_selected(std::int32_t head_dim, std::int32_t q_heads,
                                 std::int32_t kv_heads) {
    return head_dim == kD && kv_heads > 0 &&
           q_heads > 0 && q_heads % kv_heads == 0 &&
           q_heads / kv_heads <= kGemma4DecodeD512MaxGroup;
}

int gemma4_decode_d512_chunks(std::int32_t tokens, std::int32_t kv_heads) {
    const std::int64_t blocks = static_cast<std::int64_t>(tokens) * kv_heads;
    if (blocks <= 0) { return 1; }
    const std::int64_t wave = static_cast<std::int64_t>(device_sm_count()) * kBlocksPerSm;
    return static_cast<int>(std::clamp<std::int64_t>(wave / blocks, 1, kMaxChunks));
}

void gemma4_attention_decode_d512_launch(const Tensor& q, const Gemma4CacheView& cache,
                                         const Tensor& positions, int window, float scale,
                                         Tensor& out, float* partial, float* partial_stats,
                                         cudaStream_t stream) {
    if (q.ne[0] != kD || q.ne[1] / cache.k_pages->ne[2] > kGemma4DecodeD512MaxGroup) {
        throw std::invalid_argument("gemma4_attention: decode kernel needs D=512, group <= 8");
    }
    if (cache.format == Gemma4KvFormat::Fp8E4m3Row256) {
        launch<Gemma4KvFormat::Fp8E4m3Row256>(q, cache, positions, window, scale, out, partial,
                                              partial_stats, stream);
    } else {
        launch<Gemma4KvFormat::BFloat16>(q, cache, positions, window, scale, out, partial,
                                         partial_stats, stream);
    }
}

} // namespace infernix::ops::detail
