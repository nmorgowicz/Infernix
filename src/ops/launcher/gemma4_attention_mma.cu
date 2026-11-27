#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/math.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gemma4_attention_tile.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <climits>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kRows    = 64; // query rows per CTA, 16 per warp
constexpr int kThreads = 128;
constexpr int kSlice   = 256; // output dimensions per CTA

template <int D, int Bc>
constexpr std::size_t kSharedBytes =
    sizeof(__nv_bfloat16) *
    (static_cast<std::size_t>(kRows) * D + static_cast<std::size_t>(Bc) * D +
     static_cast<std::size_t>(Bc) * kSlice);

template <int D, int Width, int Bc>
__device__ __forceinline__ void stage_keys(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                           int kv_heads, int kv_head, int physical, int k0,
                                           int last, int column, int tid) {
    gemma4_stage_keys<D, Width, Bc, kThreads>(dst, cache, kv_heads, kv_head, physical, k0, last,
                                              column, tid);
}

// grid (query tiles, Hq, D / 256); CTA (tile, h, s) writes dimensions [256 s, 256 s + 256) of
// query rows [64 tile, 64 tile + 64) of head h. Online softmax over the union of the rows'
// visible keys in Bc-key tiles; each row masks with its own position and the window. Used for
// D256; D512 runs gemma4_attention_mma_d512.cu, one 8-warp CTA for both output halves.
// FP8 (the opt-in D256 ring): K and V codes arrive by cp.async in the compact first half of their
// BF16 tile, their 64 row scales in a small shared array, and expand in place
// (gemma4_expand_quant_tile) to BF16(FP32(code) x FP32(scale)), the values every Gemma FP8 reader
// consumes; QK and PV then run unchanged on the BF16 tiles.
template <int D, int Bc, Gemma4KvFormat Fmt>
__launch_bounds__(kThreads, 1) __global__
    void gemma4_attention_mma_kernel(const __nv_bfloat16* __restrict__ q,
                                     const void* __restrict__ k_pages,
                                     const void* __restrict__ v_pages,
                                     const __half* __restrict__ k_scales,
                                     const __half* __restrict__ v_scales,
                                     const std::int32_t* __restrict__ block_table,
                                     const std::int32_t* __restrict__ positions,
                                     __nv_bfloat16* __restrict__ out, int tokens, int q_heads,
                                     int kv_heads, int window, float scale,
                                     std::int32_t logical_pages, std::int32_t physical_pages) {
    constexpr bool kQuantKv     = Fmt != Gemma4KvFormat::BFloat16;
    static_assert(!kQuantKv || D == kSlice, "the FP8 ring is D256: one V slice is the whole row");
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = kSlice / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr unsigned FullMask = 0xffffffffu;
    static_assert(kSharedBytes<D, Bc> <= 99 * 1024);
    static_assert(Bc % 16 == 0 && kPagedKVPageSize % Bc == 0);

    extern __shared__ __align__(16) __nv_bfloat16 gemma4_smem[];
    __nv_bfloat16* q_s = gemma4_smem;     // [kRows, D] swizzled
    __nv_bfloat16* k_s = q_s + kRows * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D;    // [Bc, kSlice] swizzled
    __shared__ int range[3];              // first visible key, highest position, lowest position
    __shared__ __align__(16) __half k_scale_s[kQuantKv ? Bc : 1]; // FP8 row scales of the K / V tile
    __shared__ __align__(16) __half v_scale_s[kQuantKv ? Bc : 1];

    // Later tiles see more keys; start them first.
    const int q0     = (static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x)) * kRows;
    const int q_head = static_cast<int>(blockIdx.y);
    const int column = static_cast<int>(blockIdx.z) * kSlice;
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int group  = q_heads / kv_heads;
    const int kvh    = q_head / group;
    const int rows   = min(kRows, tokens - q0);

    if (tid == 0) {
        range[0] = INT_MAX;
        range[1] = -1;
        range[2] = INT_MAX;
    }
    __syncthreads();
    if (tid < rows) {
        const int position = positions[q0 + tid];
        atomicMin(&range[0], window == 0 ? 0 : max(0, position + 1 - window));
        atomicMax(&range[1], position);
        atomicMin(&range[2], position);
    }
    __syncthreads();
    const int first_key       = range[0];
    const int last_key        = range[1];
    const int lowest_position = range[2];

    const int gid      = lane >> 2;
    const int lid      = lane & 3;
    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int row_base = warp * 16;

    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>((row_base + a_rowoff) * (D * 2));
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(k_s) + static_cast<unsigned>(b_rin * (D * 2)) +
                                 (static_cast<unsigned>(lane >> 4) * (8 * D * 2));
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(v_s) +
                                 static_cast<unsigned>(((lane >> 3) & 1) * (8 * kSlice * 2)) +
                                 static_cast<unsigned>(b_rin * (kSlice * 2));
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    {
        constexpr int VecPerRow       = D / 8;
        const std::int64_t row_stride = static_cast<std::int64_t>(q_heads) * D;
        const __nv_bfloat16* q_block  = q + (static_cast<std::int64_t>(q0) * q_heads + q_head) * D;
#pragma unroll
        for (int chunk = tid; chunk < kRows * VecPerRow; chunk += kThreads) {
            const int row    = chunk / VecPerRow;
            const int d      = (chunk % VecPerRow) * 8;
            __nv_bfloat16* p = &q_s[row * D + causal_swizzle(row, d)];
            if (row < rows) {
                cp_async<16, Cache::cg>(p, &q_block[row * row_stride + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }

    const int row0      = row_base + gid;
    const int row1      = row0 + 8;
    const int position0 = row0 < rows ? positions[q0 + row0] : -1;
    const int position1 = row1 < rows ? positions[q0 + row1] : -1;
    const int low0      = window == 0 ? 0 : position0 + 1 - window;
    const int low1      = window == 0 ? 0 : position1 + 1 - window;

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    Gemma4KvSoftmaxRow state0, state1;

    const float scale_l2   = scale * kLog2E;
    const int block_begin  = first_key / Bc;
    const int block_end    = last_key < 0 ? block_begin : last_key / Bc + 1;
    const auto physical_of = [&](int block) {
        const int page = (block * Bc) >> kPagedKVPageShift;
        if (page >= logical_pages) { return -1; }
        const int physical = block_table[page];
        return physical < physical_pages ? physical : -1;
    };
    const bool all_rows = rows == kRows;

    // Keys and the V column slice of tile `k0`: BF16 rows, or the compact FP8 codes and row scales
    // gemma4_expand_quant_tile expands after they arrive.
    const auto stage_key = [&](int mapped, int k0) {
        if constexpr (kQuantKv) {
            gemma4_stage_quant_codes<D, Bc, kThreads>(k_s, static_cast<const std::uint8_t*>(k_pages), kv_heads,
                                                      kvh, mapped, k0, last_key, tid);
            gemma4_stage_quant_scales<Bc, kThreads>(k_scale_s, k_scales, kv_heads, kvh, mapped, k0, tid);
        } else {
            stage_keys<D, D, Bc>(k_s, static_cast<const __nv_bfloat16*>(k_pages), kv_heads, kvh, mapped, k0,
                                 last_key, 0, tid);
        }
    };
    const auto stage_value = [&](int mapped, int k0) {
        if constexpr (kQuantKv) {
            gemma4_stage_quant_codes<D, Bc, kThreads>(v_s, static_cast<const std::uint8_t*>(v_pages), kv_heads,
                                                      kvh, mapped, k0, last_key, tid);
            gemma4_stage_quant_scales<Bc, kThreads>(v_scale_s, v_scales, kv_heads, kvh, mapped, k0, tid);
        } else {
            stage_keys<D, kSlice, Bc>(v_s, static_cast<const __nv_bfloat16*>(v_pages), kv_heads, kvh, mapped,
                                      k0, last_key, column, tid);
        }
    };
    // FullTile: every row of the tile is mapped and at or before the last key, so no row is zeroed.
    const auto expand = [&]<bool FullTile>(__nv_bfloat16* tile, const __half* row_scales, int mapped, int k0) {
        if constexpr (kQuantKv) {
            gemma4_expand_quant_tile<D, Bc, kThreads, FullTile>(tile, row_scales, mapped, k0, last_key, tid);
            __syncthreads();
        }
    };

    int physical = block_begin < block_end ? physical_of(block_begin) : -1;
    cp_commit();
    if (block_begin < block_end) { stage_key(physical, block_begin * Bc); }
    cp_commit();

    const auto step = [&]<bool FullTile>(int block) {
        const int k0            = block * Bc;
        const int next_physical = block + 1 < block_end ? physical_of(block + 1) : physical;
        const bool mapped       = physical >= 0;

        cp_wait<0>();
        __syncthreads();
        expand.template operator()<FullTile>(k_s, k_scale_s, physical, k0);
        stage_value(physical, k0);
        cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                    causal_swizzle_address(q_lane_base, 0u, q_as, q_r));
#pragma unroll
        for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
            ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                        causal_swizzle_address(
                            k_lane_base + static_cast<unsigned>(nt2 * (8 * D * 2)), 0u, k_as, k_r));
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            causal_swizzle_address(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        causal_swizzle_address(
                            k_lane_base + static_cast<unsigned>(nt2 * (8 * D * 2)), ck, k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        cp_wait<0>();
        __syncthreads();
        expand.template operator()<FullTile>(v_s, v_scale_s, physical, k0);
        if (block + 1 < block_end) {
            physical = next_physical;
            stage_key(physical, (block + 1) * Bc);
            cp_commit();
        }

        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
        if constexpr (FullTile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = mapped ? k0 + nt * 8 + 2 * lid : INT_MAX - 1;
                const int key1 = key0 + 1;
                score[nt][0]   = (key0 <= position0 && key0 >= low0) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (key1 <= position0 && key1 >= low0) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (key0 <= position1 && key0 >= low1) ? score[nt][2] : -CUDART_INF_F;
                score[nt][3]   = (key1 <= position1 && key1 >= low1) ? score[nt][3] : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float alpha0 = state0.update<!FullTile>(bm0, scale_l2);
        const float alpha1 = state1.update<!FullTile>(bm1, scale_l2);
        const float m0     = state0.scaled_maximum;
        const float m1     = state1.scaled_maximum;

        // P goes to the tensor cores as hi + lo BF16 terms (about 16 mantissa bits): a single BF16
        // P moved Gemma's greedy tokens and lowered logit cosine below HF bf16's.
        float bl0 = 0.0f;
        float bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        unsigned p_low[PVKs][4];
        const auto split = [](float a, float b, unsigned& high, unsigned& low) {
            const __nv_bfloat162 rounded = __floats2bfloat162_rn(a, b);
            const float2 back            = __bfloat1622float2(rounded);
            high                         = *reinterpret_cast<const unsigned*>(&rounded);
            low                          = pack_bf16x2(a - back.x, b - back.y);
        };
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -m0));
            float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -m0));
            float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -m1));
            float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -m1));
            if constexpr (!FullTile) {
                p00 = score[nt][0] > -CUDART_INF_F ? p00 : 0.0f;
                p01 = score[nt][1] > -CUDART_INF_F ? p01 : 0.0f;
                p10 = score[nt][2] > -CUDART_INF_F ? p10 : 0.0f;
                p11 = score[nt][3] > -CUDART_INF_F ? p11 : 0.0f;
            }
            bl0 += p00 + p01;
            bl1 += p10 + p11;
            const int pk   = nt >> 1;
            const int slot = (nt & 1) * 2;
            split(p00, p01, p_frag[pk][slot], p_low[pk][slot]);
            split(p10, p11, p_frag[pk][slot + 1], p_low[pk][slot + 1]);
        }
        state0.accumulate(alpha0, bl0);
        state1.accumulate(alpha1, bl1);
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PVHalf  = PVNt / 2;
        constexpr int PVLoads = PVKs * PVHalf;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      causal_swizzle_address(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              causal_swizzle_address(
                                  v_lane_base + static_cast<unsigned>(k2 * (16 * kSlice * 2)), ckv,
                                  v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_low[k][0], p_low[k][1],
                     p_low[k][2], p_low[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_low[k][0],
                     p_low[k][1], p_low[k][2], p_low[k][3], vf[cur][2], vf[cur][3]);
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    };
    for (int block = block_begin; block < block_end; ++block) {
        const int k0     = block * Bc;
        const bool whole = all_rows && physical_of(block) >= 0 && k0 + Bc - 1 <= lowest_position &&
                           (window == 0 || k0 >= last_key + 1 - window);
        if (whole) {
            step.template operator()<true>(block);
        } else {
            step.template operator()<false>(block);
        }
    }
    cp_wait<0>();

    state0.finish();
    state1.finish();
    const float inv0 = state0.sum > 0.0f ? __frcp_rn(state0.sum) : 0.0f;
    const float inv1 = state1.sum > 0.0f ? __frcp_rn(state1.sum) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d = column + n * 8 + 2 * lid;
        if (row0 < rows) {
            *reinterpret_cast<unsigned*>(
                out + (static_cast<std::int64_t>(q0 + row0) * q_heads + q_head) * D + d) =
                pack_bf16x2(acc[n][0] * inv0, acc[n][1] * inv0);
        }
        if (row1 < rows) {
            *reinterpret_cast<unsigned*>(
                out + (static_cast<std::int64_t>(q0 + row1) * q_heads + q_head) * D + d) =
                pack_bf16x2(acc[n][2] * inv1, acc[n][3] * inv1);
        }
    }
}

template <int D, int Bc, Gemma4KvFormat Fmt>
void launch(const Tensor& q, const Gemma4CacheView& cache, const Tensor& positions, int window,
            float scale, Tensor& out, cudaStream_t stream) {
    constexpr bool Quantized     = Fmt != Gemma4KvFormat::BFloat16;
    constexpr std::size_t shared = kSharedBytes<D, Bc>;
    static bool configured       = false;
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma4_attention_mma_kernel<D, Bc, Fmt>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared)));
        configured = true;
    }
    const Tensor& k_pages     = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    // One CTA per query head. Packing a GQA group onto one CTA would reload Q on every key
    // tile (one Q tile is already 32 KB, and K plus V fill the rest of the 99 KB), which costs
    // as much traffic as the shared K/V it would save. The grid stays one head.
    const int tokens = q.ne[2];
    const dim3 grid(static_cast<unsigned>((tokens + kRows - 1) / kRows),
                    static_cast<unsigned>(q.ne[1]), static_cast<unsigned>(D / kSlice));
    gemma4_attention_mma_kernel<D, Bc, Fmt><<<grid, kThreads, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), k_pages.data, cache.v_pages->data,
        Quantized ? static_cast<const __half*>(cache.k_scales->data) : nullptr,
        Quantized ? static_cast<const __half*>(cache.v_scales->data) : nullptr,
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        tokens, q.ne[1], k_pages.ne[2], window, scale, block_table.ne[0], k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void gemma4_attention_mma_launch(const Tensor& q, const Gemma4CacheView& cache,
                                 const Tensor& positions, int window, float scale,
                                 Tensor& out, float* partial, float* stats, cudaStream_t stream) {
    if (q.ne[0] == 256) {
        if (cache.format == Gemma4KvFormat::Fp8E4m3Row256) {
            launch<256, 64, Gemma4KvFormat::Fp8E4m3Row256>(q, cache, positions, window, scale, out, stream);
        } else {
            launch<256, 64, Gemma4KvFormat::BFloat16>(q, cache, positions, window, scale, out, stream);
        }
    } else if (q.ne[0] == 512) {
        gemma4_attention_mma_d512_launch(q, cache, positions, window, scale, out, partial, stats,
                                         stream);
    } else {
        throw std::invalid_argument("gemma4_attention: the tensor-core path needs D=256 or 512");
    }
}

} // namespace infernix::ops::detail
