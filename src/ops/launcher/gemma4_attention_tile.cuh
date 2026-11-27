#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/math.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"

namespace infernix::ops::detail {

// Gemma 4's online-softmax row, qualified bit for bit against the pinned Gemma logits. The shared Bf16KvSoftmaxRow
// rescales with causal_rescale (two rounded products, candidates UV-01), which differs from
// this row's alpha by an ulp wherever the maximum holds; the qualified Gemma logits use one
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

// One compact [Bc, Width] byte tile. Width is the full quantized row (512 global, 256 sliding).
template <int Width, int Bc, int Threads, typename Code>
__device__ __forceinline__ void gemma4_stage_quant_codes(
    __nv_bfloat16* dst, const Code* pages, int kv_heads, int kv_head, int physical,
    int k0, int last, int tid) {
    static_assert(Width == 512 || Width == 256);
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

// The Bc FP16 row scales of keys [k0, k0 + Bc) of a [1,64,Hkv,pages] scale cache, by cp.async into
// `dst`: the caller commits them with the codes gemma4_stage_quant_codes stages for the same tile.
// An unmapped page stages nothing. Scales of rows past the last key are copied like any other and
// never used (gemma4_expand_quant_tile zeroes those rows).
template <int Bc, int Threads>
__device__ __forceinline__ void gemma4_stage_quant_scales(__half* dst, const __half* scales, int kv_heads,
                                                          int kv_head, int physical, int k0, int tid) {
    constexpr int Vectors = Bc / 8;
    static_assert(Vectors <= Threads && kPagedKVPageSize % Bc == 0);
    if (physical >= 0 && tid < Vectors) {
        const auto row = gemma4_kv_row(physical, kv_heads, kv_head, k0 & kPagedKVPageMask);
        cp_async<16, Cache::cg>(dst + tid * 8, scales + row + tid * 8);
    }
}

// The two BF16 values of the E4M3 codes in the low 16 bits of `codes`: BF16(FP32(code) x scale), the
// value gemma4_fp8_expand returns for each, two to a register.
__device__ __forceinline__ std::uint32_t gemma4_expand_code_pair(std::uint32_t codes, float scale) {
    const __half2 halves = __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(codes), __NV_E4M3));
    const float2 values  = __half22float2(halves);
    return pack_bf16x2(__fmul_rn(values.x, scale), __fmul_rn(values.y, scale));
}

// Expands the compact [Bc, Width] code tile gemma4_stage_quant_codes left in `tile` into the BF16
// tile it occupies, one 8-code chunk per thread and pass, with the row scales gemma4_stage_quant_scales
// staged in `row_scales`. BF16 row j covers code rows 2j and 2j + 1: the BF16 rows of the last half of
// the keys lie past every code byte and expand straight away; the codes of the first half are read
// into registers, one block barrier separates every thread's reads from the first write, and the
// first half's rows follow. Rows past `last` or of an unmapped page expand to zeros unless
// `AllValid` (every row of the tile is mapped and at or before `last`). The caller has waited for
// the codes and scales and passed a block barrier; another barrier must follow before any
// cross-thread read of the tile.
template <int Width, int Bc, int Threads, bool AllValid>
__device__ __forceinline__ void gemma4_expand_quant_tile(__nv_bfloat16* tile, const __half* row_scales,
                                                         int physical, int k0, int last, int tid) {
    constexpr int ChunksPerRow = Width / 8;
    constexpr int RowsPerPass  = Threads / ChunksPerRow;
    constexpr int Passes       = Bc / RowsPerPass;
    static_assert(Threads % ChunksPerRow == 0 && Passes % 2 == 0);
    const int row_in_pass = tid / ChunksPerRow;
    const int column      = (tid % ChunksPerRow) * 8;
    const auto* codes     = reinterpret_cast<const std::uint8_t*>(tile);
    const bool mapped     = physical >= 0;
    const auto key_of     = [&](int pass) { return pass * RowsPerPass + row_in_pass; };
    const auto read       = [&](int pass) {
        return *reinterpret_cast<const uint2*>(codes + key_of(pass) * Width + column);
    };
    const auto write = [&](int pass, uint2 packed) {
        const int key    = key_of(pass);
        __nv_bfloat16* p = tile + key * Width + causal_swizzle(key, column);
        if (AllValid || (mapped && k0 + key <= last)) {
            const float scale = __half2float(row_scales[key]);
            store_vec(p, make_int4(static_cast<int>(gemma4_expand_code_pair(packed.x, scale)),
                                   static_cast<int>(gemma4_expand_code_pair(packed.x >> 16, scale)),
                                   static_cast<int>(gemma4_expand_code_pair(packed.y, scale)),
                                   static_cast<int>(gemma4_expand_code_pair(packed.y >> 16, scale))));
        } else {
            store_vec(p, make_int4(0, 0, 0, 0));
        }
    };
#pragma unroll
    for (int pass = Passes - 1; pass >= Passes / 2; --pass) { write(pass, read(pass)); }
    uint2 rest[Passes / 2];
#pragma unroll
    for (int pass = 0; pass < Passes / 2; ++pass) { rest[pass] = read(pass); }
    __syncthreads();
#pragma unroll
    for (int pass = Passes / 2 - 1; pass >= 0; --pass) { write(pass, rest[pass]); }
}

// Gemma4ValueSource::Key: expands a prefetched tile of K codes (keys k0 .. k0 + Bc - 1) as the V
// tile, with gemma4_expand_quant_stage's barriers and swizzled layout. Each value is FP32(code) x
// FP32(scale); dimensions [0, 128) pair d with d + 64 and are turned back by gemma4_unrope_pair at
// the key's position (`frequency` holds the 64 pair frequencies); every value is then multiplied by
// value_scale and rounded once to BF16. Chunk c holds dimensions 8 (c % 64) ..; its partner chunk
// (d +/- 64) is c ^ 8, i.e. lane ^ 8 of the same warp, so every lane joins the exchange.
template <int Bc, int Threads>
__device__ __forceinline__ void gemma4_expand_key_values(
    const Gemma4QuantStage<512, Bc, Threads>& stage, __nv_bfloat16* dst, int k0,
    const float* frequency, float value_scale, int tid) {
    static_assert(Threads % 64 == 0, "a row's chunks must map onto whole warps");
#pragma unroll
    for (int i = 0; i < Gemma4QuantStage<512, Bc, Threads>::Chunks; ++i) {
        const int chunk = tid + i * Threads;
        const int key   = chunk / 64;
        const int d     = (chunk % 64) * 8;
        const uint2 partner_codes{__shfl_xor_sync(0xffffffffU, stage.codes[i].x, 8),
                                  __shfl_xor_sync(0xffffffffU, stage.codes[i].y, 8)};
        const float row_scale = stage.valid[i] ? __half2float(stage.scale[i]) : 0.0f;
        const auto* own   = reinterpret_cast<const std::uint8_t*>(&stage.codes[i]);
        const auto* other = reinterpret_cast<const std::uint8_t*>(&partner_codes);
        float values[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            __nv_fp8_e4m3 code;
            code.__x = own[j];
            values[j] = __fmul_rn(static_cast<float>(code), row_scale);
        }
        if (d < kGemma4KeyValueRotaryDim) {
            const bool first = d < kGemma4KeyValueRotaryDim / 2;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                __nv_fp8_e4m3 code;
                code.__x = other[j];
                const float partner = __fmul_rn(static_cast<float>(code), row_scale);
                float a, b;
                gemma4_unrope_pair(first ? values[j] : partner, first ? partner : values[j], k0 + key,
                                   frequency[(d % (kGemma4KeyValueRotaryDim / 2)) + j], a, b);
                values[j] = first ? a : b;
            }
        }
        alignas(16) __nv_bfloat16 out[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) { out[j] = __float2bfloat16_rn(__fmul_rn(values[j], value_scale)); }
        __nv_bfloat16* p = dst + key * 512 + causal_swizzle(key, d);
        store_vec(p, stage.valid[i] ? *reinterpret_cast<const int4*>(out) : make_int4(0, 0, 0, 0));
    }
}


} // namespace infernix::ops::detail
