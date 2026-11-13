#pragma once

#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/softmax_attention/common/causal_tile_io.cuh"

namespace infernix::ops::detail {

// FP8 ldmatrix views the bytes as 16-bit cells. This is the reference FP8
// fragment swizzle, with a D512 row stride rather than the D256 codec stride.
__device__ __forceinline__ int gemma4_fp8_qk_offset(int row, int d) {
    return row * 512 + causal_swizzle(row, d >> 1) * 2 + (d & 1);
}

// Each row is quantized once, before the key loop. One warp owns each row;
// the two 256-element groups use exactly the persistent K codec's represented
// FP16 scale, bounded reciprocal, zero-row and whole-group nonfinite policy.
template <int Rows, int Warps>
__device__ __forceinline__ void gemma4_fp8_qk_queries(
    const __nv_bfloat16* q, std::int64_t stride, int valid_rows,
    std::uint8_t* codes, __half* scales, int warp, int lane) {
    for (int row = warp; row < Rows; row += Warps) {
#pragma unroll
        for (int group = 0; group < 2; ++group) {
            float2 values[4];
            float maximum = 0.0f;
            bool nonfinite = false;
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int d = group * 256 + 2 * (lane + 32 * s);
                values[s] = row < valid_rows ?
                    __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(q + row * stride + d)) :
                    make_float2(0.0f, 0.0f);
                nonfinite |= !isfinite(values[s].x) || !isfinite(values[s].y);
                maximum = fmaxf(maximum, fmaxf(fabsf(values[s].x), fabsf(values[s].y)));
            }
            const bool poison = __any_sync(0xffffffffU, nonfinite);
            maximum = warp_max(maximum, 0xffffffffU);
            const auto params = kv_cache_fp8_quant_params(maximum);
            if (lane == 0) { scales[row * 2 + group] = poison ? __float2half_rn(1.0f) : params.scale; }
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                const int d = group * 256 + 2 * (lane + 32 * s);
                const std::uint16_t packed = poison ? std::uint16_t{0x7f7f} :
                    kv_cache_fp8_quant_code2(values[s].x, values[s].y, params.inverse_scale);
                *reinterpret_cast<std::uint16_t*>(codes + gemma4_fp8_qk_offset(row, d)) = packed;
            }
        }
    }
}

// Stage the cached codes directly: there is no K dequantization or re-encoding.
// Invalid pages and past-last rows do not form source code OR scale addresses.
template <int Bc, int Threads>
__device__ __forceinline__ void gemma4_fp8_qk_keys(
    std::uint8_t* dst, __half* dst_scales, const std::uint8_t* codes,
    const __half* scales, int kv_heads, int kv_head, int physical,
    int k0, int last, int tid) {
    for (int row = tid; row < Bc; row += Threads) {
        if (physical >= 0 && k0 + row <= last) {
            const auto source = gemma4_kv_row(physical, kv_heads, kv_head, (k0 & 63) + row);
            *reinterpret_cast<__half2*>(dst_scales + row * 2) =
                *reinterpret_cast<const __half2*>(scales + source * 2);
        } else {
            dst_scales[row * 2] = dst_scales[row * 2 + 1] = __float2half_rn(0.0f);
        }
    }
#pragma unroll
    for (int chunk = tid; chunk < Bc * (512 / 16); chunk += Threads) {
        const int row = chunk / (512 / 16);
        const int d = (chunk % (512 / 16)) * 16;
        auto* destination = dst + gemma4_fp8_qk_offset(row, d);
        if (physical >= 0 && k0 + row <= last) {
            const auto source = gemma4_kv_row(physical, kv_heads, kv_head, (k0 & 63) + row);
            cp_async<16, Cache::cg>(destination, codes + source * 512 + d);
        } else {
            store_vec(destination, make_int4(0, 0, 0, 0));
        }
    }
}

// Warps paired for PV output halves each reconstruct the complete score from their own
// 256-dim group when only_group >= 0: only that group's mma runs and only its scaled
// partial is produced, for the caller's pair exchange. mma_fp8_e4m3 supplies unit
// UE8M0 block scales, not these row scales, so EACH partial is corrected before it is
// published.
template <int Bc>
__device__ __forceinline__ void gemma4_fp8_qk_scores(
    float (&score)[Bc / 8][4], const std::uint8_t* q, const std::uint8_t* k,
    const __half* q_scales, const __half* k_scales, int row_base, int lane,
    int only_group = -1) {
    const int gid = lane >> 2;
    const int lid = lane & 3;
    const int a_mat = lane >> 3;
    const int rin = lane & 7;
    const unsigned qa = smem_addr(q) + (row_base + rin + ((a_mat & 1) << 3)) * 512;
    const unsigned ka = smem_addr(k) + rin * 512 + (lane >> 4) * (8 * 512);
    const unsigned q_as = (a_mat >> 1) << 4;
    const unsigned k_as = ((lane >> 3) & 1) << 4;
    const unsigned r = rin << 4;
#pragma unroll
    for (int nt = 0; nt < Bc / 8; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { score[nt][i] = 0.0f; }
    }
#pragma unroll
    for (int group = 0; group < 2; ++group) {
        if (only_group >= 0 && group != only_group) { continue; }
        float partial[Bc / 8][4]{};
        unsigned af[2][4];
        unsigned bf[2][Bc / 8][2];
        const auto load = [&](int slot, int step) {
            const unsigned column = (group * 8 + step) << 5;
            ldmatrix_x4(af[slot][0], af[slot][1], af[slot][2], af[slot][3],
                        causal_swizzle_address(qa, column, q_as, r));
#pragma unroll
            for (int nt = 0; nt < Bc / 8; nt += 2) {
                ldmatrix_x4(bf[slot][nt][0], bf[slot][nt][1], bf[slot][nt + 1][0], bf[slot][nt + 1][1],
                            causal_swizzle_address(ka + nt * (8 * 512), column, k_as, r));
            }
        };
        load(0, 0);
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            const int current = step & 1;
            if (step + 1 < 8) { load(current ^ 1, step + 1); }
#pragma unroll
            for (int nt = 0; nt < Bc / 8; ++nt) {
                mma_fp8_e4m3(partial[nt][0], partial[nt][1], partial[nt][2], partial[nt][3],
                             af[current][0], af[current][1], af[current][2], af[current][3],
                             bf[current][nt][0], bf[current][nt][1]);
            }
        }
        const float qs0 = __half2float(q_scales[(row_base + gid) * 2 + group]);
        const float qs1 = __half2float(q_scales[(row_base + gid + 8) * 2 + group]);
#pragma unroll
        for (int nt = 0; nt < Bc / 8; ++nt) {
            const int key = nt * 8 + 2 * lid;
            const float ks0 = __half2float(k_scales[key * 2 + group]);
            const float ks1 = __half2float(k_scales[(key + 1) * 2 + group]);
            score[nt][0] += partial[nt][0] * (qs0 * ks0);
            score[nt][1] += partial[nt][1] * (qs0 * ks1);
            score[nt][2] += partial[nt][2] * (qs1 * ks0);
            score[nt][3] += partial[nt][3] * (qs1 * ks1);
        }
    }
}

} // namespace infernix::ops::detail
