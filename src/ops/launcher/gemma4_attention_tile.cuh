#pragma once

#include "ops/common/memory.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"

namespace infernix::ops::detail {

// Gemma 4's online-softmax row, bit-identical to ninfer.spo b330aaf. The shared Bf16KvSoftmaxRow
// rescales with causal_rescale (two rounded products, candidates UV-01), which differs from the
// fork's alpha by an ulp wherever the maximum holds; the qualified Gemma logits use the fork's one
// FMA of the previous maximum against the new scaled maximum. Keep this copy Gemma-local.
struct Gemma4KvSoftmaxRow {
    float maximum;
    float sum;
    float scaled_maximum;

    __device__ Gemma4KvSoftmaxRow()
        : maximum(kBf16KvNegativeInfinity), sum(0.0f), scaled_maximum(0.0f) {}

    template <bool MayBeEmpty = true>
    __device__ __forceinline__ float update(float tile_maximum, float scale_log2) {
        const float previous = maximum;
        maximum              = fmaxf(maximum, tile_maximum);
        scaled_maximum       = maximum * scale_log2;
        if constexpr (MayBeEmpty)
            if (maximum == kBf16KvNegativeInfinity) scaled_maximum = 0.0f;
        const float alpha = exp2_approx(__fmaf_rn(previous, scale_log2, -scaled_maximum));
        return alpha;
    }

    __device__ __forceinline__ void accumulate(float alpha, float tile_sum) {
        sum = __fmaf_rn(sum, alpha, tile_sum);
    }

    __device__ __forceinline__ float probability(float score, float scale_log2) const {
        return exp2_approx(__fmaf_rn(score, scale_log2, -scaled_maximum));
    }

    __device__ __forceinline__ void finish() { sum = warp_sum<4>(sum, 0xffffffffu); }
};


// One [Bc, Width] tile of keys [k0, k0 + Bc) from columns [column, column + Width) of the
// [D,64,Hkv,pages] cache into swizzled shared memory. Keys past `last` are zeroed so the
// unwritten cache tail never reaches the tensor cores.
template <int D, int Width, int Bc, int Threads>
__device__ __forceinline__ void gemma4_stage_keys(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                                  int kv_heads, int kv_head, int physical, int k0,
                                                  int last, int column, int tid) {
    constexpr int VecPerRow = Width / 8;
    // Form addresses only for mapped rows, not negative physical page IDs.
#pragma unroll
    for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
        const int key    = chunk / VecPerRow;
        const int d      = (chunk % VecPerRow) * 8;
        __nv_bfloat16* p = &dst[key * Width + causal_swizzle(key, d)];
        if (physical >= 0 && k0 + key <= last) {
            const auto row = gemma4_kv_row(physical, kv_heads, kv_head,
                                           (k0 & kPagedKVPageMask) + key);
            cp_async<16, Cache::cg>(p, cache + row * D + column + d);
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}

// Raw quantized codes share the destination BF16 tile's storage. cp.async fills its
// compact first half while the other attention product computes; every thread then
// reads its codes into registers before a block barrier permits in-place expansion.
template <int Width, int Bc, int Threads>
struct Gemma4QuantStage {
    static constexpr int Chunks = Bc * (Width / 8) / Threads;
    static_assert(Chunks * Threads == Bc * (Width / 8), "tile chunks must divide over threads");
    static_assert(Chunks >= 1, "each thread must own at least one chunk");

    uint2 codes[Chunks];
    __half scale[Chunks];
    bool valid[Chunks];
};

// One compact [Bc, Width] byte tile. Width is the full D512 quantized row.
template <int Width, int Bc, int Threads, typename Code>
__device__ __forceinline__ void gemma4_stage_quant_codes(
    __nv_bfloat16* dst, const Code* pages, int kv_heads, int kv_head, int physical,
    int k0, int last, int tid) {
    static_assert(Width == 512);
    auto* raw = reinterpret_cast<std::uint8_t*>(dst);
#pragma unroll
    for (int chunk = tid; chunk < Bc * (Width / 16); chunk += Threads) {
        const int key = chunk / (Width / 16);
        const int d = (chunk % (Width / 16)) * 16;
        if (physical >= 0 && k0 + key <= last) {
            const auto row = gemma4_kv_row(physical, kv_heads, kv_head,
                                           (k0 & kPagedKVPageMask) + key);
            cp_async<16, Cache::cg>(raw + key * Width + d, pages + row * Width + d);
        } else {
            store_vec(raw + key * Width + d, make_int4(0, 0, 0, 0));
        }
    }
}

// Call after cp_wait and a block barrier; all threads must finish reading before
// any thread overwrites the compact codes with the expanded BF16 tile.
template <int Width, int Bc, int Threads, int GroupElems, int ScaleStride>
__device__ __forceinline__ void gemma4_load_quant_stage(
    Gemma4QuantStage<Width, Bc, Threads>& stage, const __nv_bfloat16* src,
    const __half* scales, int kv_heads, int kv_head, int physical, int k0, int last, int tid) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(src);
#pragma unroll
    for (int i = 0; i < Gemma4QuantStage<Width, Bc, Threads>::Chunks; ++i) {
        const int chunk = tid + i * Threads;
        const int key = chunk / (Width / 8);
        const int d = (chunk % (Width / 8)) * 8;
        stage.valid[i] = physical >= 0 && k0 + key <= last;
        if (stage.valid[i]) {
            const auto row = gemma4_kv_row(physical, kv_heads, kv_head,
                                           (k0 & kPagedKVPageMask) + key);
            stage.codes[i] = *reinterpret_cast<const uint2*>(raw + key * Width + d);
            stage.scale[i] = scales[row * ScaleStride + d / GroupElems];
        }
    }
}

// Expand a prefetched tile into shared memory. Callers keep their existing block barriers:
// the previous tile's reads of dst must have passed a barrier, and a barrier is needed after
// this before any cross-thread consumption of dst.
template <int Width, int Bc, int Threads, typename Code>
__device__ __forceinline__ void gemma4_expand_quant_stage(
    const Gemma4QuantStage<Width, Bc, Threads>& stage, __nv_bfloat16* dst, int column, int tid) {
#pragma unroll
    for (int i = 0; i < Gemma4QuantStage<Width, Bc, Threads>::Chunks; ++i) {
        const int chunk = tid + i * Threads;
        const int key   = chunk / (Width / 8);
        const int d     = column + (chunk % (Width / 8)) * 8;
        __nv_bfloat16* p = dst + key * Width + causal_swizzle(key, d - column);
        if (stage.valid[i]) {
            const auto* bytes = reinterpret_cast<const std::uint8_t*>(&stage.codes[i]);
            alignas(16) __nv_bfloat16 values[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {

                    values[j] = gemma4_fp8_expand(bytes[j], stage.scale[i]);

            }
            store_vec(p, *reinterpret_cast<const int4*>(values));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    }
}


} // namespace infernix::ops::detail
