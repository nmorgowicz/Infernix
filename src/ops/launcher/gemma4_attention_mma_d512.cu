#include "ops/launcher/gemma4_attention.h"

#include "core/device.h"
#include "core/temporary_switches.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/launcher/gemma4_attention_fp8_qk.cuh"
#include "ops/launcher/gemma4_attention_tile.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"
#include "ops/softmax_attention/dense/causal_cache/bf16/softmax.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kD       = 512;
constexpr int kHalf    = 256; // output columns per warp
constexpr int kRows    = kGemma4AttentionD512TileRows;
constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kBc      = 16;
constexpr int kQkNt    = kBc / 8;
constexpr int kQkKs    = kD / 16;
constexpr int kPvNt    = kHalf / 8;
constexpr int kPvKs    = kBc / 16;

constexpr std::size_t kSharedBytes = sizeof(__nv_bfloat16) * (kRows + 2 * kBc) * kD;
static_assert(kSharedBytes <= 99 * 1024);
static_assert(kPagedKVPageSize % kBc == 0);

template <bool Fp8Qk>
constexpr std::size_t shared_bytes() {
    constexpr std::size_t qk = Fp8Qk ? (kRows + kBc) * kD :
                                                     sizeof(__nv_bfloat16) * (kRows + kBc) * kD;
    return qk + sizeof(__nv_bfloat16) * kBc * kD;
}

// grid (query tiles, Hq, segments); CTA (tile, h, seg) covers query rows [64 tile, 64 tile +
// 64) of head h over its segment of the visible keys. Warps w and w ^ 4 own the same 16 rows
// and each computes their full QK scores, so both hold the same softmax state; warp w writes
// output columns [256 (w / 4), 256 (w / 4) + 256). One staged K and V tile serves both
// halves. The arithmetic per row and column is the same as a 4-warp CTA per output half, so
// the result does not depend on the split. With segments > 1 each CTA writes FP32 partials
// and (max, sum) stats for gemma4_attention_d512_combine_kernel instead of the final rows.
template <Gemma4KvFormat Fmt, bool Fp8Qk = false>
__launch_bounds__(kThreads, 1) __global__ void gemma4_attention_d512_kernel(
    const __nv_bfloat16* __restrict__ q, const void* __restrict__ k_pages,
    const void* __restrict__ v_pages, const __half* __restrict__ k_scales,
    const __half* __restrict__ v_scales, const std::int32_t* __restrict__ block_table,
    const std::int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ out, int tokens,
    int q_heads, int kv_heads, int window, float scale, std::int32_t logical_pages,
    std::int32_t physical_pages, float* __restrict__ partial,
    float* __restrict__ partial_stats) {
    constexpr unsigned FullMask = 0xffffffffu;

    extern __shared__ __align__(1024) __nv_bfloat16 gemma4_d512_smem[]; // Preserve the qualified fork module's alignment.
    static_assert(!Fp8Qk || Fmt == Gemma4KvFormat::Fp8E4m3Row256);
    __nv_bfloat16* q_s = gemma4_d512_smem;
    auto* q_fp8 = reinterpret_cast<std::uint8_t*>(q_s);
    auto* k_fp8 = q_fp8 + kRows * kD;
    __nv_bfloat16* k_s = Fp8Qk ? reinterpret_cast<__nv_bfloat16*>(k_fp8) : q_s + kRows * kD;
    __nv_bfloat16* v_s = Fp8Qk ? reinterpret_cast<__nv_bfloat16*>(k_fp8 + kBc * kD) : k_s + kBc * kD;
    __shared__ __align__(4) __half q_scale_s[Fp8Qk ? kRows * 2 : 1];
    __shared__ __align__(4) __half k_scale_s[Fp8Qk ? kBc * 2 : 1];
    __shared__ int range[3]; // first visible key, highest position, lowest position

    // Later tiles see more keys; start them first.
    const int q0       = (static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x)) * kRows;
    const int q_head   = static_cast<int>(blockIdx.y);
    const int tid      = static_cast<int>(threadIdx.x);
    const int warp     = tid >> 5;
    const int lane     = tid & 31;
    const int half     = warp >> 2;
    const int row_base = (warp & 3) * 16;
    const int kvh      = q_head / (q_heads / kv_heads);
    const int rows     = min(kRows, tokens - q0);

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

    const unsigned q_lane_base =
        smem_addr(q_s) + static_cast<unsigned>((row_base + a_rowoff) * (kD * 2));
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    const unsigned k_lane_base = smem_addr(k_s) + static_cast<unsigned>(b_rin * (kD * 2)) +
                                 (static_cast<unsigned>(lane >> 4) * (8 * kD * 2));
    const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
    const unsigned v_lane_base = smem_addr(v_s) +
                                 static_cast<unsigned>(((lane >> 3) & 1) * (8 * kD * 2)) +
                                 static_cast<unsigned>(b_rin * (kD * 2));
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // De-duplicated QK: warp (r, h) contracts only the 256-dim half h of every key (one
    // FP8 scale group) and the warp pair (w, w ^ 4) exchanges its partial score tile
    // through this region, which aliases the K staging bytes: by exchange time every
    // warp finished its ldmatrix reads of the tile, and the next K stage starts only
    // after the CTA barrier below the exchange. Lane i of both paired warps holds the
    // identical (row, key) C-fragment coordinates, so the exchange is lane-symmetric.
    // The two contributions are added in opposite order by the two warps; FP addition is
    // commutative, so both hold bit-identical full scores and softmax state without the
    // duplicate QK MMAs of the old "each warp computes all 512 dims" layout.
    float* xchg = Fp8Qk ? reinterpret_cast<float*>(k_fp8) : reinterpret_cast<float*>(k_s);
    const int pair = 1 + (warp & 3); // named 64-thread barrier for warps r and r + 4

    if constexpr (Fp8Qk) {
        const auto* q_block = q + (static_cast<std::int64_t>(q0) * q_heads + q_head) * kD;
        gemma4_fp8_qk_queries<kRows, kWarps>(
            q_block, static_cast<std::int64_t>(q_heads) * kD, rows,
            q_fp8, q_scale_s, warp, lane);
    } else {
        constexpr int VecPerRow       = kD / 8;
        const std::int64_t row_stride = static_cast<std::int64_t>(q_heads) * kD;
        const __nv_bfloat16* q_block  = q + (static_cast<std::int64_t>(q0) * q_heads + q_head) * kD;
#pragma unroll
        for (int chunk = tid; chunk < kRows * VecPerRow; chunk += kThreads) {
            const int row    = chunk / VecPerRow;
            const int d      = (chunk % VecPerRow) * 8;
            __nv_bfloat16* p = &q_s[row * kD + causal_swizzle(row, d)];
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

    float acc[kPvNt][4];
#pragma unroll
    for (int n = 0; n < kPvNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    Bf16KvSoftmaxRow state0, state1;

    const float scale_l2   = scale * kLog2E;
    const int block_begin  = first_key / kBc;
    const int block_end   = last_key < 0 ? block_begin : last_key / kBc + 1;

    // grid.z splits the visible 16-key block range into equal segments (flash-decoding style
    // context split-K); segment 0 owns the lowest keys. With one segment the CTA writes the
    // final BF16 rows exactly like the single-kernel path.
    const int segments  = static_cast<int>(gridDim.z);
    const int span      = (max(0, block_end - block_begin) + segments - 1) / segments;
    const int seg       = static_cast<int>(blockIdx.z);
    const int seg_begin = block_begin + seg * span;
    const int seg_end   = min(block_end, seg_begin + span);
    const auto physical_of = [&](int block) {
        const int page = (block * kBc) >> kPagedKVPageShift;
        if (page >= logical_pages) { return -1; }
        const int physical = block_table[page];
        return physical < physical_pages ? physical : -1;
    };
    const bool all_rows = rows == kRows;

    // Quantized codes arrive asynchronously in the compact first half of their BF16
    // tile, then expand in place after every thread has read its register slice.
    constexpr bool kQuantKv = Fmt != Gemma4KvFormat::BFloat16;
    const auto stage = [&](auto* dst, const void* pages, const __half* scales, int mapped,
                           int k0) {
        if constexpr (kQuantKv) {
            gemma4_stage_quant_codes<kD, kBc, kThreads>(
                dst, static_cast<const std::uint8_t*>(pages), kv_heads, kvh, mapped,
                k0, last_key, tid);
        } else {
            gemma4_stage_keys<kD, kD, kBc, kThreads>(
                dst, static_cast<const __nv_bfloat16*>(pages), kv_heads, kvh, mapped, k0,
                last_key, 0, tid);
        }
    };
    const auto expand = [&](__nv_bfloat16* dst, const __half* scales, int mapped, int k0) {
        Gemma4QuantStage<kD, kBc, kThreads> staged;

            gemma4_load_quant_stage<kD, kBc, kThreads, kGemma4KvFp8Group, 2>(
                staged, dst, scales, kv_heads, kvh, mapped, k0, last_key, tid);

        __syncthreads();

            gemma4_expand_quant_stage<kD, kBc, kThreads, std::uint8_t>(staged, dst, 0, tid);

    };
    const auto stage_key = [&](int mapped, int k0) {
        if constexpr (Fp8Qk) {
            gemma4_fp8_qk_keys<kBc, kThreads>(
                k_fp8, k_scale_s, static_cast<const std::uint8_t*>(k_pages), k_scales,
                kv_heads, kvh, mapped, k0, last_key, tid);
        } else {
            stage(k_s, k_pages, k_scales, mapped, k0);
        }
    };

    int physical = seg_begin < seg_end ? physical_of(seg_begin) : -1;
    cp_commit();
    if (seg_begin < seg_end) { stage_key(physical, seg_begin * kBc); }
    cp_commit();

    const auto step = [&]<bool FullTile>(int block) {
        const int k0            = block * kBc;
        const int next_physical = block + 1 < seg_end ? physical_of(block + 1) : physical;
        const bool mapped       = physical >= 0;

        cp_wait<0>();
        __syncthreads();
        if constexpr (kQuantKv && !Fp8Qk) {
            expand(k_s, k_scales, physical, k0);
            __syncthreads();
        }
        stage(v_s, v_pages, v_scales, physical, k0);

        cp_commit();

        float score[kQkNt][4];
        if constexpr (Fp8Qk) {
            // Only this warp's 256-dim group; the pair exchange adds the other group.
            gemma4_fp8_qk_scores<kBc>(score, q_fp8, k_fp8, q_scale_s, k_scale_s, row_base,
                                      lane, half);
        } else {
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }
            unsigned af[2][4];
            unsigned bf[2][kQkNt][2];
            const auto load = [&](unsigned(&a)[4], unsigned(&b)[kQkNt][2], int k) {
                const unsigned ck = static_cast<unsigned>(k << 5);
                ldmatrix_x4(a[0], a[1], a[2], a[3], causal_swizzle_address(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < kQkNt; nt2 += 2) {
                    ldmatrix_x4(
                        b[nt2][0], b[nt2][1], b[nt2 + 1][0], b[nt2 + 1][1],
                        causal_swizzle_address(k_lane_base + static_cast<unsigned>(nt2 * (8 * kD * 2)),
                                               ck, k_as, k_r));
                }
            };
            // Half of the k-steps: dims [256 half, 256 (half + 1)).
            const int k_lo = half * (kQkKs / 2);
            load(af[0], bf[0], k_lo);
#pragma unroll
            for (int i = 0; i < kQkKs / 2; ++i) {
                const int cur = i & 1;
                if (i + 1 < kQkKs / 2) { load(af[cur ^ 1], bf[cur ^ 1], k_lo + i + 1); }
#pragma unroll
                for (int nt = 0; nt < kQkNt; ++nt) {
                    mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                             af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
                }
            }
        }

        cp_wait<0>();
        __syncthreads();
        // Exchange the half contractions: publish, meet the pair, add the partner's
        // partials. Every warp's K ldmatrix reads finished above, so the aliased K
        // staging bytes are dead; the trailing CTA barrier retires the reads before
        // the next K stage (or the V expand) reuses any of them.
        {
            float* slot = xchg + (warp * 32 + lane) * 8;
            *reinterpret_cast<float4*>(slot) = *reinterpret_cast<const float4*>(&score[0][0]);
            *reinterpret_cast<float4*>(slot + 4) = *reinterpret_cast<const float4*>(&score[1][0]);
            asm volatile("bar.sync %0, 64;" :: "r"(pair) : "memory");
            const float4* other = reinterpret_cast<const float4*>(xchg + ((warp ^ 4) * 32 + lane) * 8);
            const float4 lo = other[0];
            const float4 hi = other[1];
            score[0][0] += lo.x;
            score[0][1] += lo.y;
            score[0][2] += lo.z;
            score[0][3] += lo.w;
            score[1][0] += hi.x;
            score[1][1] += hi.y;
            score[1][2] += hi.z;
            score[1][3] += hi.w;
            __syncthreads();
        }
        if constexpr (kQuantKv) {
            expand(v_s, v_scales, physical, k0);
            __syncthreads();
        }
        if (block + 1 < seg_end) {
            physical = next_physical;
            stage_key(physical, (block + 1) * kBc);
            cp_commit();
        }

        float bm0 = -CUDART_INF_F;
        float bm1 = -CUDART_INF_F;
        if constexpr (FullTile) {
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < kQkNt; ++nt) {
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
        // or FP16 P moved Gemma's greedy tokens and lowered logit cosine below HF bf16's.
        float bl0 = 0.0f;
        float bl1 = 0.0f;
        unsigned p_frag[kPvKs][4];
        unsigned p_low[kPvKs][4];
        const auto split = [](float a, float b, unsigned& high, unsigned& low) {
            const __nv_bfloat162 rounded = __floats2bfloat162_rn(a, b);
            const float2 back            = __bfloat1622float2(rounded);
            high                         = *reinterpret_cast<const unsigned*>(&rounded);
            low                          = pack_bf16x2(a - back.x, b - back.y);
        };
#pragma unroll
        for (int nt = 0; nt < kQkNt; ++nt) {
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
        for (int n = 0; n < kPvNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PvHalf   = kPvNt / 2;
        constexpr int PvLoads  = kPvKs * PvHalf;

        const auto load_values = [&](unsigned(&dst)[4], int li) {
            const int k  = li / PvHalf;
            const int n2 = (li % PvHalf) * 2;

                ldmatrix_x4_t(dst[0], dst[1], dst[2], dst[3],
                              causal_swizzle_address(
                                  v_lane_base + static_cast<unsigned>(k * (16 * kD * 2)),
                                  static_cast<unsigned>((half * (kHalf / 8) + n2) << 4), v_as, v_r));

        };
        unsigned vf[2][4];
        load_values(vf[0], 0);
#pragma unroll
        for (int li = 0; li < PvLoads; ++li) {
            const int k   = li / PvHalf;
            const int n2  = (li % PvHalf) * 2;
            const int cur = li & 1;
            if (li + 1 < PvLoads) { load_values(vf[cur ^ 1], li + 1); }
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
    for (int block = seg_begin; block < seg_end; ++block) {
        const int k0     = block * kBc;
        const bool whole = all_rows && physical_of(block) >= 0 && k0 + kBc - 1 <= lowest_position &&
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
    const int column = half * kHalf;
    if (segments == 1) {
        const float inv0 = state0.sum > 0.0f ? __frcp_rn(state0.sum) : 0.0f;
        const float inv1 = state1.sum > 0.0f ? __frcp_rn(state1.sum) : 0.0f;
#pragma unroll
        for (int n = 0; n < kPvNt; ++n) {
            const int d = column + n * 8 + 2 * lid;
            if (row0 < rows) {
                *reinterpret_cast<unsigned*>(
                    out + (static_cast<std::int64_t>(q0 + row0) * q_heads + q_head) * kD + d) =
                    pack_bf16x2(acc[n][0] * inv0, acc[n][1] * inv0);
            }
            if (row1 < rows) {
                *reinterpret_cast<unsigned*>(
                    out + (static_cast<std::int64_t>(q0 + row1) * q_heads + q_head) * kD + d) =
                    pack_bf16x2(acc[n][2] * inv1, acc[n][3] * inv1);
            }
        }
        return;
    }

    // Split context: publish unnormalized FP32 partials plus (log2-unit max, sum) stats;
    // gemma4_attention_d512_combine_kernel merges the segments of each row. A segment that
    // saw no visible key keeps a -inf maximum so the combine skips it, which reproduces the
    // zero row of the single-kernel path.
    const float max0 = state0.sum > 0.0f ? state0.scaled_maximum : -CUDART_INF_F;
    const float max1 = state1.sum > 0.0f ? state1.scaled_maximum : -CUDART_INF_F;
    if (half == 0 && lid == 0) {
        if (row0 < rows) {
            float* row_stats =
                partial_stats + ((static_cast<std::int64_t>(q0 + row0) * q_heads + q_head) *
                                     segments + seg) * 2;
            row_stats[0] = max0;
            row_stats[1] = state0.sum;
        }
        if (row1 < rows) {
            float* row_stats =
                partial_stats + ((static_cast<std::int64_t>(q0 + row1) * q_heads + q_head) *
                                     segments + seg) * 2;
            row_stats[0] = max1;
            row_stats[1] = state1.sum;
        }
    }
#pragma unroll
    for (int n = 0; n < kPvNt; ++n) {
        const int d = column + n * 8 + 2 * lid;
        if (row0 < rows) {
            *reinterpret_cast<float2*>(
                partial + ((static_cast<std::int64_t>(q0 + row0) * q_heads + q_head) * segments +
                           seg) * kD + d) = make_float2(acc[n][0], acc[n][1]);
        }
        if (row1 < rows) {
            *reinterpret_cast<float2*>(
                partial + ((static_cast<std::int64_t>(q0 + row1) * q_heads + q_head) * segments +
                           seg) * kD + d) = make_float2(acc[n][2], acc[n][3]);
        }
    }
}

// grid (T * Hq); merges the context segments of one query head in segment order. Stats
// maxima are in log2 units, matching the mainloop's exp2 weights, so the merge uses exp2f.
constexpr int kD512CombineThreads = 128;

__global__ void gemma4_attention_d512_combine_kernel(
    const float* __restrict__ partial, const float* __restrict__ partial_stats,
    __nv_bfloat16* __restrict__ out, int segments) {
    constexpr int E         = kD / kD512CombineThreads;
    const std::int64_t head = blockIdx.x; // token * q_heads + q_head
    const float* head_stats = partial_stats + head * segments * 2;
    float head_max          = -CUDART_INF_F;
    for (int seg = 0; seg < segments; ++seg) {
        head_max = fmaxf(head_max, head_stats[seg * 2]);
    }
    float total = 0.0f;
    float acc[E];
#pragma unroll
    for (int e = 0; e < E; ++e) { acc[e] = 0.0f; }
    for (int seg = 0; seg < segments; ++seg) {
        const float seg_max = head_stats[seg * 2];
        if (seg_max == -CUDART_INF_F) { continue; }
        const float weight = exp2f(seg_max - head_max);
        total              = fmaf(weight, head_stats[seg * 2 + 1], total);
        const float* row   = partial + (head * segments + seg) * kD + threadIdx.x;
#pragma unroll
        for (int e = 0; e < E; ++e) { acc[e] = fmaf(weight, row[e * kD512CombineThreads], acc[e]); }
    }
    const float inverse = total > 0.0f ? 1.0f / total : 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        out[head * kD + threadIdx.x + e * kD512CombineThreads] =
            __float2bfloat16_rn(acc[e] * inverse);
    }
}

constexpr int kMinSegmentBlocks = 16; // shorter contexts keep the single-kernel path
constexpr int kMaxSegments      = 32;
constexpr int kWaveSegments     = 8; // beyond this, combine traffic outweighs the fill

// Context split-K heuristic (wave-aware, adapted from mxfp8_tiled_partition): at the
// kernel's 1-CTA/SM occupancy each wave holds sm_count CTAs, and a CTA's mainloop time
// scales with its share of the key range, so relative cost is waves(segments)/segments
// with waves = ceil(ctas * segments / SMs). Minimize that over segments, keeping the
// fewest segments on a tie. Unlike ceil(SMs / ctas) alone this reaches whole waves at
// every chunk width (chunk 128: 64 CTAs x 5 = 320 CTAs, 94% of one 170-SM wave pair).
// The segment choice depends only on (tokens, q_heads, SMs): the live key range comes
// from positions on device, so extra segments beyond the live range launch, see an
// empty block range, and publish -inf stats the combine skips. The grid therefore stays
// fixed per captured width and replay is capture-safe.
// pages <= 0 returns the context-unclamped bound, which sizes the preallocated workspace
// for every context length a bucket can later reach. INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K pins
// the count for sweeps; the pinned value skips the context clamp like pages <= 0.
int d512_segment_count(std::int32_t tokens, std::int32_t q_heads, std::int32_t pages,
                       std::int32_t window) {
    if (tokens < kRows || q_heads <= 0) { return 1; }
    static const int forced_segments = static_cast<int>(
        temporary_switches::integer("INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K", 0, 0, 13));
    if (forced_segments > 0) {
        return std::min(forced_segments, kMaxSegments);
    }
    static const int sm_count = [] {
        int device = 0;
        int count  = 1;
        if (cudaGetDevice(&device) != cudaSuccess ||
            cudaDeviceGetAttribute(&count, cudaDevAttrMultiProcessorCount, device) !=
                cudaSuccess) {
            return 1;
        }
        return count;
    }();
    const std::int64_t ctas =
        ((static_cast<std::int64_t>(tokens) + kRows - 1) / kRows) * q_heads;
    std::int64_t segments = 1;
    std::int64_t waves    = (ctas + sm_count - 1) / sm_count;
    for (int splits = 2; splits <= kWaveSegments; ++splits) {
        const std::int64_t next = (ctas * splits + sm_count - 1) / sm_count;
        if (next * segments < waves * splits) {
            segments = splits;
            waves    = next;
        }
    }
    segments = std::min(segments, static_cast<std::int64_t>(kMaxSegments));
    if (pages > 0) {
        std::int64_t keys = static_cast<std::int64_t>(pages) * kPagedKVPageSize;
        if (window > 0) { keys = std::min(keys, static_cast<std::int64_t>(window)); }
        const std::int64_t by_context = keys / (kBc * kMinSegmentBlocks);
        segments                      = std::min(segments, std::max<std::int64_t>(by_context, 1));
    }
    return static_cast<int>(std::max<std::int64_t>(segments, 1));
}

template <Gemma4KvFormat Fmt, bool Fp8Qk = false>
void launch_d512(const Tensor& q, const Gemma4CacheView& cache, const Tensor& positions,
                  int window, float scale, Tensor& out, float* partial, float* partial_stats,
                  cudaStream_t stream) {
    constexpr bool Quantized = Fmt != Gemma4KvFormat::BFloat16;
    const Tensor& k_pages = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    static bool configured = false;
    if (!configured) {
        CUDA_CHECK(cudaFuncSetAttribute(gemma4_attention_d512_kernel<Fmt, Fp8Qk>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        static_cast<int>(shared_bytes<Fp8Qk>())));
        configured = true;
    }
    const int tokens   = q.ne[2];
    const int segments = d512_segment_count(tokens, q.ne[1], block_table.ne[0], window);
    if (segments > 1 && (partial == nullptr || partial_stats == nullptr)) {
        throw std::invalid_argument("gemma4_attention: split context needs workspace");
    }
    const dim3 grid(static_cast<unsigned>((tokens + kRows - 1) / kRows),
                    static_cast<unsigned>(q.ne[1]), static_cast<unsigned>(segments));
    gemma4_attention_d512_kernel<Fmt, Fp8Qk>
        <<<grid, kThreads, shared_bytes<Fp8Qk>(), stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), k_pages.data, cache.v_pages->data,
        Quantized ? static_cast<const __half*>(cache.k_scales->data) : nullptr,
        Quantized ? static_cast<const __half*>(cache.v_scales->data) : nullptr,
        static_cast<const std::int32_t*>(block_table.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(out.data),
        tokens, q.ne[1], k_pages.ne[2], window, scale, block_table.ne[0], k_pages.ne[3], partial,
        partial_stats);
    if (segments > 1) {
        gemma4_attention_d512_combine_kernel
            <<<static_cast<unsigned>(tokens * q.ne[1]), kD512CombineThreads, 0, stream>>>(
                partial, partial_stats, static_cast<__nv_bfloat16*>(out.data), segments);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Keep diagnostics and real dispatch on the same specialization selector.
template <typename Visitor>
decltype(auto) visit_d512(const Gemma4CacheView& cache, Visitor&& visitor) {
    if (cache.format == Gemma4KvFormat::Fp8E4m3Row256) {

        if (cache.qk_mode == Gemma4QkMode::Fp8) {
            return visitor.template operator()<Gemma4KvFormat::Fp8E4m3Row256, true>();
        }
        return visitor.template operator()<Gemma4KvFormat::Fp8E4m3Row256, false>();
    }

    return visitor.template operator()<Gemma4KvFormat::BFloat16, false>();
}

} // namespace

int gemma4_attention_mma_d512_segments(std::int32_t tokens, std::int32_t q_heads,
                                       std::int32_t pages, std::int32_t window) {
    return d512_segment_count(tokens, q_heads, pages, window);
}

Gemma4AttentionD512Resources gemma4_attention_mma_d512_resources(const Gemma4CacheView& cache) {
    return visit_d512(cache, []<Gemma4KvFormat Fmt, bool Fp8Qk>() {
        Gemma4AttentionD512Resources result;
        result.dynamic_shared_bytes = shared_bytes<Fp8Qk>();
        result.threads = kThreads;
        CUDA_CHECK(cudaFuncSetAttribute(gemma4_attention_d512_kernel<Fmt, Fp8Qk>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(result.dynamic_shared_bytes)));
        CUDA_CHECK(cudaFuncGetAttributes(
            &result.attributes, gemma4_attention_d512_kernel<Fmt, Fp8Qk>));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &result.active_blocks_per_sm, gemma4_attention_d512_kernel<Fmt, Fp8Qk>,
            kThreads, result.dynamic_shared_bytes));
        return result;
    });
}

void gemma4_attention_mma_d512_launch(const Tensor& q, const Gemma4CacheView& cache,
                                      const Tensor& positions, int window, float scale,
                                      Tensor& out, float* partial, float* partial_stats,
                                      cudaStream_t stream) {
    visit_d512(cache, [&]<Gemma4KvFormat Fmt, bool Fp8Qk>() {
        launch_d512<Fmt, Fp8Qk>(
            q, cache, positions, window, scale, out, partial, partial_stats, stream);
    });
}

} // namespace infernix::ops::detail
