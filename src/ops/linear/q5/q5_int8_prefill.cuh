#pragma once

// Exact-code Q5 x INT8-activation prefill GEMM (LinearRoutes::Q5Int8G32 / Q5Int8G64).
//
//   out[token, row] = sum_g  ws[row, g64] * as[token, g] * (sum_{k in g} q[row, k] * a[token, k])
//
// q: the stored signed five-bit codes, exact (no weight bake); ws: the stored FP16 group-64 scale; a: INT8
// activations quantized per token per K group (Group = 32 or 64 values) with scale `as` = max|x| / 127 (FP32,
// two low mantissa bits cleared), FP32 reciprocal multiply, round-half-even, clamp +-127.
//
// * The producers sign-extend the stored codes to INT8 shared tiles; the tensor core runs m16n8k32 s8 x s8 -> s32,
//   exact, so a group's integer dot is exact whatever the K order.
// * The accumulator starts at 0x4B400000: its s32 result is the bit pattern of the float 1.5 * 2^23 + dot
//   (|dot| < 2^22), so the int-to-float conversion is free. fma(F, as, nb) with nb = -(1.5 * 2^23) * as yields
//   dot * as correctly rounded (as keeps <= 22 mantissa bits, so 1.5 * 2^23 * as is exact), and one more FFMA applies
//   the weight scale into the FP32 running sum, ascending K groups.
// * K inside one 32-value chunk is permuted [even values | odd values] so the Q5 low and high nibble planes unpack
//   with mask and shift; the activation quantizer writes the same permutation. A K32 dot is order invariant.
//
// Warp-specialized: consumer warps run the MMAs, producer warps unpack the weights and copy the activation tile and
// its group scales through an mbarrier ring of StageK = 128 K values per stage (the row-split staging ring's design).

#include "ops/common/mbarrier.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace infernix::ops::detail {

// The activation scale of one (token, K group) and nb = -(1.5 * 2^23) * scale, exact.
struct Q5Int8Meta {
    float scale;
    float nb;
};
inline constexpr unsigned kQ5Int8Magic = 0x4B400000u;
inline constexpr float kQ5Int8MagicF   = 12582912.0f;
inline constexpr int kQ5Int8StageK     = 128;

template <int BM_, int BN_, int WM_, int WN_, int Stages_, int ProducerWarps_, int Group_>
struct Q5Int8Config {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_, Stages = Stages_, ProducerWarps = ProducerWarps_,
                         Group = Group_, StageK = kQ5Int8StageK;
    static constexpr int WarpsM = BM / WM, WarpsN = BN / WN, ConsumerWarps = WarpsM * WarpsN;
    static constexpr int MT = WM / 16, NT = WN / 8;
    static constexpr int kConsumerThreads = ConsumerWarps * 32, kProducerThreads = ProducerWarps * 32;
    static constexpr int kThreads         = kConsumerThreads + kProducerThreads;
    static constexpr int KC               = StageK / 32;         // K32 chunks per stage
    static constexpr int Units            = StageK / 16;         // 16-byte units per tile row
    static constexpr int WG               = StageK / 64;         // weight scale groups per stage
    static constexpr int kGroupsPerStage  = StageK / Group;      // activation scale groups a stage spans
    static constexpr int kWeightBytes = BM * StageK, kActBytes = BN * StageK;
    static constexpr int kWScaleBytes = BM * WG * 4;              // [BM][WG] float
    static constexpr int kMetaBytes   = kGroupsPerStage * BN * 8; // [groups][BN] Q5Int8Meta
    static constexpr int kStageBytesRaw = kWeightBytes + kActBytes + kWScaleBytes + kMetaBytes;
    static constexpr int kStageBytes    = (kStageBytesRaw + 1023) / 1024 * 1024;
    static constexpr int kBarrierOffset = Stages * kStageBytes;
    static constexpr int kSharedBytes   = kBarrierOffset + Stages * 2 * 8;
    static constexpr int kItems         = BM * KC / kProducerThreads; // weight (row, K32 chunk) items per producer thread
    static_assert(BM % WM == 0 && BN % WN == 0 && WM % 16 == 0 && WN % 16 == 0, "tile shape");
    static_assert(BM * KC % kProducerThreads == 0, "producer items");
    static_assert(Group == 32 || Group == 64);
    static_assert(kSharedBytes <= 99 * 1024, "one CTA per SM");
    static_assert(kThreads <= 1024);
    static_assert(NT % 2 == 0);
};

// 16-byte unit `unit` of tile row `row` (a row is Units = 8 units): XOR swizzle so ldmatrix's eight rows hit eight
// different bank groups.
__device__ __forceinline__ int q5int8_swizzle(int row, int unit) { return unit ^ (row & 7); }

__device__ __forceinline__ void q5int8_mma(int& d0, int& d1, int& d2, int& d3, unsigned a0, unsigned a1, unsigned a2,
                                           unsigned a3, unsigned b0, unsigned b1, int c0, int c1, int c2, int c3) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
}

// xq: INT8 activations [padded_tokens][padded_k]; meta: [padded_k / Group][padded_tokens]; rows a multiple of BM,
// padded_tokens a multiple of BN, k a multiple of StageK; the weight planes' rows are padded_k wide. Stores only
// tokens < `tokens`.
template <class Cfg>
__global__ __launch_bounds__(Cfg::kThreads, 1) void q5_int8_prefill_kernel(
    const std::int8_t* __restrict__ xq, const Q5Int8Meta* __restrict__ meta, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int rows, int tokens, int padded_tokens, int k, int padded_k) {
    constexpr int BM = Cfg::BM, BN = Cfg::BN, WM = Cfg::WM, WN = Cfg::WN, Stages = Cfg::Stages;
    constexpr int MT = Cfg::MT, NT = Cfg::NT, G = Cfg::kGroupsPerStage;
    constexpr int PT = Cfg::kProducerThreads, CT = Cfg::kConsumerThreads;
    constexpr int SK = Cfg::StageK, KC = Cfg::KC, WG = Cfg::WG;

    extern __shared__ __align__(1024) unsigned char q5int8_smem[];
    auto weight_tile = [&](int s) { return q5int8_smem + s * Cfg::kStageBytes; };
    auto act_tile    = [&](int s) { return q5int8_smem + s * Cfg::kStageBytes + Cfg::kWeightBytes; };
    auto wscale_tile = [&](int s) {
        return reinterpret_cast<float*>(q5int8_smem + s * Cfg::kStageBytes + Cfg::kWeightBytes + Cfg::kActBytes);
    };
    auto meta_tile = [&](int s) {
        return reinterpret_cast<Q5Int8Meta*>(q5int8_smem + s * Cfg::kStageBytes + Cfg::kWeightBytes + Cfg::kActBytes +
                                             Cfg::kWScaleBytes);
    };
    auto* full  = reinterpret_cast<std::uint64_t*>(q5int8_smem + Cfg::kBarrierOffset);
    auto* empty = full + Stages;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < Stages; ++s) {
            cta_mbarrier_init(full + s, PT);
            cta_mbarrier_init(empty + s, Cfg::ConsumerWarps);
        }
        cta_mbarrier_fence_init();
    }
    __syncthreads();

    const int k_tiles = k / SK;
    // Token tiles are the fast grid dimension: the CTAs of a wave share a few weight row tiles (one DRAM read each).
    const int row0 = static_cast<int>(blockIdx.y) * BM;
    const int col0 = static_cast<int>(blockIdx.x) * BN;

    if (threadIdx.x >= CT) {
        // ------------------------------ producers ------------------------------
        const int p = static_cast<int>(threadIdx.x) - CT;
        struct Raw {
            uint4 c;
            unsigned h, s;
        } raw[Cfg::kItems];
        const std::int64_t code_stride = padded_k / 2, high_stride = padded_k / 8, scale_stride = padded_k / 32;
        auto load = [&](int kt) {
#pragma unroll
            for (int u = 0; u < Cfg::kItems; ++u) {
                const int item = p + u * PT, row = item / KC, j = item % KC;
                const std::int64_t gr = row0 + row;
                raw[u].c = ld_nc_na<uint4>(codes + gr * code_stride + kt * (SK / 2) + j * 16);
                raw[u].h = __ldg(reinterpret_cast<const unsigned*>(high + gr * high_stride + kt * (SK / 8) + j * 4));
                raw[u].s = j == 0 ? __ldg(reinterpret_cast<const unsigned*>(scales + gr * scale_stride + kt * 4)) : 0u;
            }
        };
        load(0);
        for (int kt = 0; kt < k_tiles; ++kt) {
            const int stage = kt % Stages;
            cta_mbarrier_wait(empty + stage, 1U ^ ((kt / Stages) & 1U));
            unsigned char* At = act_tile(stage);
#pragma unroll 1
            for (int c = p; c < BN * Cfg::Units; c += PT) {
                const int tl = c / Cfg::Units, unit = c % Cfg::Units;
                cp_async<16, Cache::cg>(At + tl * SK + (q5int8_swizzle(tl, unit) << 4),
                                        xq + static_cast<std::int64_t>(col0 + tl) * k + kt * SK + unit * 16);
            }
            Q5Int8Meta* Mt = meta_tile(stage);
#pragma unroll 1
            for (int c = p; c < G * BN / 2; c += PT) {
                const int g = c / (BN / 2), r = c % (BN / 2);
                cp_async<16, Cache::cg>(Mt + g * BN + r * 2,
                                        meta + static_cast<std::int64_t>(kt * G + g) * padded_tokens + col0 + r * 2);
            }
            unsigned char* Wt = weight_tile(stage);
            float* Ws         = wscale_tile(stage);
#pragma unroll
            for (int u = 0; u < Cfg::kItems; ++u) {
                const int item = p + u * PT, row = item / KC, j = item % KC;
                // q = low nibble - 16 * fifth bit, per byte: byte = lo + 0xF0 * f (no carry between bytes).
                const unsigned w[4] = {raw[u].c.x, raw[u].c.y, raw[u].c.z, raw[u].c.w};
                unsigned a[4], b[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const unsigned hm = raw[u].h >> (8 * m);
                    const unsigned f0 = (hm & 0x55u) * 0x410410u;
                    const unsigned f1 = (hm & 0xAAu) * 0x208208u;
                    a[m] = (w[m] & 0x0F0F0F0Fu) + 15u * (f0 & 0x10101010u);
                    b[m] = ((w[m] >> 4) & 0x0F0F0F0Fu) + 15u * (f1 & 0x10101010u);
                }
                *reinterpret_cast<uint4*>(Wt + row * SK + (q5int8_swizzle(row, 2 * j) << 4))     = make_uint4(a[0], a[1], a[2], a[3]);
                *reinterpret_cast<uint4*>(Wt + row * SK + (q5int8_swizzle(row, 2 * j + 1) << 4)) = make_uint4(b[0], b[1], b[2], b[3]);
                if (j == 0) {
                    *reinterpret_cast<float2*>(Ws + row * 2) = __half22float2(*reinterpret_cast<const __half2*>(&raw[u].s));
                }
            }
            asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" : : "r"(smem_addr(full + stage)) : "memory");
            if (kt + 1 < k_tiles) load(kt + 1);
        }
        asm volatile("cp.async.wait_all;\n" ::: "memory");
        return;
    }

    // ------------------------------ consumers ------------------------------
    const int tid = static_cast<int>(threadIdx.x), warp = tid >> 5, lane = tid & 31;
    const int wm = warp / Cfg::WarpsN, wn = warp % Cfg::WarpsN;
    const int id = lane >> 3, r8 = lane & 7;
    const int a_row  = wm * WM + r8 + ((id & 1) << 3); // + mi * 16
    const int a_unit = id >> 1;
    const int b_tok  = wn * WN + ((id >> 1) << 3) + r8; // + np * 16
    const int b_unit = id & 1;
    const int magic  = static_cast<int>(kQ5Int8Magic);

    float run[MT][NT][4];
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e) { run[mi][ni][e] = 0.0f; }

    for (int kt = 0; kt < k_tiles; ++kt) {
        const int stage = kt % Stages;
        cta_mbarrier_wait(full + stage, (kt / Stages) & 1U);
        const unsigned wbase = smem_addr(weight_tile(stage));
        const unsigned abase = smem_addr(act_tile(stage));
        const float* Ws      = wscale_tile(stage);
        const Q5Int8Meta* Mt = meta_tile(stage);
        float wsv[MT][2][WG];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int r    = wm * WM + mi * 16 + (lane >> 2) + h * 8;
                const float2 v = *reinterpret_cast<const float2*>(Ws + r * 2);
                wsv[mi][h][0]  = v.x;
                wsv[mi][h][1]  = v.y;
            }
        constexpr int Steps   = Cfg::Group == 32 ? KC : KC / 2; // apply points per stage
        constexpr int PerStep = KC / Steps;                      // K32 chunks chained into one INT32 accumulator
        constexpr int kShift  = Cfg::Group == 32 ? 1 : 0;        // a step's weight 64-group is step >> kShift
#pragma unroll
        for (int step = 0; step < Steps; ++step) {
            // activation meta of this step's group: tokens (2t, 2t+1) of every token tile
            float asv[NT][2], nbv[NT][2];
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int tok  = wn * WN + ni * 8 + 2 * (lane & 3);
                const float4 m = *reinterpret_cast<const float4*>(Mt + step * BN + tok);
                asv[ni][0]     = m.x;
                nbv[ni][0]     = m.y;
                asv[ni][1]     = m.z;
                nbv[ni][1]     = m.w;
            }
            unsigned b[PerStep][NT][2];
#pragma unroll
            for (int c = 0; c < PerStep; ++c) {
                const int j = step * PerStep + c;
#pragma unroll
                for (int np = 0; np < NT / 2; ++np) {
                    const int trow = b_tok + np * 16;
                    ldmatrix_x4(b[c][2 * np][0], b[c][2 * np][1], b[c][2 * np + 1][0], b[c][2 * np + 1][1],
                                abase + trow * SK + (q5int8_swizzle(trow, 2 * j + b_unit) << 4));
                }
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                unsigned a[PerStep][4];
#pragma unroll
                for (int c = 0; c < PerStep; ++c) {
                    const int j    = step * PerStep + c;
                    const int arow = a_row + mi * 16;
                    ldmatrix_x4(a[c][0], a[c][1], a[c][2], a[c][3], wbase + arow * SK + (q5int8_swizzle(arow, 2 * j + a_unit) << 4));
                }
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    int d0, d1, d2, d3;
                    q5int8_mma(d0, d1, d2, d3, a[0][0], a[0][1], a[0][2], a[0][3], b[0][ni][0], b[0][ni][1], magic, magic, magic,
                               magic);
                    if constexpr (PerStep == 2) {
                        q5int8_mma(d0, d1, d2, d3, a[1][0], a[1][1], a[1][2], a[1][3], b[1][ni][0], b[1][ni][1], d0, d1, d2, d3);
                    }
                    const int wg = step >> kShift;
                    const float t0 = fmaf(__int_as_float(d0), asv[ni][0], nbv[ni][0]);
                    const float t1 = fmaf(__int_as_float(d1), asv[ni][1], nbv[ni][1]);
                    const float t2 = fmaf(__int_as_float(d2), asv[ni][0], nbv[ni][0]);
                    const float t3 = fmaf(__int_as_float(d3), asv[ni][1], nbv[ni][1]);
                    run[mi][ni][0] = fmaf(t0, wsv[mi][0][wg], run[mi][ni][0]);
                    run[mi][ni][1] = fmaf(t1, wsv[mi][0][wg], run[mi][ni][1]);
                    run[mi][ni][2] = fmaf(t2, wsv[mi][1][wg], run[mi][ni][2]);
                    run[mi][ni][3] = fmaf(t3, wsv[mi][1][wg], run[mi][ni][3]);
                }
            }
        }
        __syncwarp();
        if (lane == 0) cta_mbarrier_arrive(empty + stage);
    }

#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int ni = 0; ni < NT; ++ni) {
            const int row   = row0 + wm * WM + mi * 16 + (lane >> 2);
            const int token = col0 + wn * WN + ni * 8 + 2 * (lane & 3);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = token + (e & 1);
                if (t < tokens) {
                    out[static_cast<std::int64_t>(t) * rows + row + (e >> 1) * 8] = __float2bfloat16_rn(run[mi][ni][e]);
                }
            }
        }
}

// ------------------------------ activation quantizer ------------------------------
// One thread per eight consecutive K values of one token: the 16-byte BF16 load of a warp is contiguous, and a K
// group is four (Group 32) or eight (Group 64) neighbouring lanes, so its maximum is a shuffle reduction.
// xq: INT8 [padded_tokens][k], each 32-value chunk stored [even | odd] (the lane that holds values 8j..8j+7 of a
// chunk writes the four even codes at bytes 4j.. and the four odd codes at 16 + 4j..). meta: [k / Group][padded_tokens].
// Scale: max|x| / 127 (1 for an all-zero group), its two low mantissa bits cleared; NaN for a group with a nonfinite
// value, so the outputs it reaches are NaN rather than silently finite. Tokens >= `tokens` are zero codes, scale 1.
inline constexpr int kQ5Int8QuantizeThreads = 256;

template <int Group>
__global__ __launch_bounds__(kQ5Int8QuantizeThreads) void q5_int8_quantize_kernel(
    const __nv_bfloat16* __restrict__ x, std::int8_t* __restrict__ xq, Q5Int8Meta* __restrict__ meta, int tokens,
    int padded_tokens, int k) {
    constexpr int kLanesPerGroup = Group / 8;
    const int unit  = static_cast<int>(blockIdx.y) * kQ5Int8QuantizeThreads + static_cast<int>(threadIdx.x);
    const int token = static_cast<int>(blockIdx.x);
    if (unit >= k / 8) return; // k / 8 is a multiple of every group's lanes, so no group is split
    const unsigned active = __activemask(); // the lanes past k / 8 have left; a group's lanes are all still here
    const int j     = unit & 3;
    const int chunk = unit >> 2;
    std::int8_t* dst = xq + static_cast<std::int64_t>(token) * k + chunk * 32;
    Q5Int8Meta* mdst = meta + static_cast<std::int64_t>(unit / kLanesPerGroup) * padded_tokens + token;
    const bool leader = (unit % kLanesPerGroup) == 0;
    if (token >= tokens) {
        *reinterpret_cast<unsigned*>(dst + 4 * j)      = 0u;
        *reinterpret_cast<unsigned*>(dst + 16 + 4 * j) = 0u;
        if (leader) *mdst = Q5Int8Meta{1.0f, -kQ5Int8MagicF};
        return;
    }
    const uint4 raw = *reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * k + unit * 8);
    const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float v[8];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float2 f = __bfloat1622float2(pairs[h]);
        v[2 * h]       = f.x;
        v[2 * h + 1]   = f.y;
    }
    float ma = 0.0f;
    bool bad = false;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        ma  = fmaxf(ma, fabsf(v[i]));
        bad = bad || !(fabsf(v[i]) <= 3.402823466e38f);
    }
    unsigned any_bad = bad ? 1u : 0u;
#pragma unroll
    for (int offset = 1; offset < kLanesPerGroup; offset <<= 1) {
        ma      = fmaxf(ma, __shfl_xor_sync(active, ma, offset));
        any_bad |= __shfl_xor_sync(active, any_bad, offset);
    }
    float s = ma > 0.0f ? __fdiv_rn(ma, 127.0f) : 1.0f;
    s       = __uint_as_float(__float_as_uint(s) & 0xFFFFFFFCu);
    if (any_bad != 0u) s = __uint_as_float(0x7FC00000u);
    const float inv = __frcp_rn(s);
    unsigned lo = 0u, hi = 0u;
#pragma unroll
    for (int b = 0; b < 4; ++b) {
        const int e0 = static_cast<int>(fminf(fmaxf(rintf(v[2 * b] * inv), -127.0f), 127.0f));
        const int e1 = static_cast<int>(fminf(fmaxf(rintf(v[2 * b + 1] * inv), -127.0f), 127.0f));
        lo |= (static_cast<unsigned>(e0) & 0xFFu) << (8 * b);
        hi |= (static_cast<unsigned>(e1) & 0xFFu) << (8 * b);
    }
    *reinterpret_cast<unsigned*>(dst + 4 * j)      = lo;
    *reinterpret_cast<unsigned*>(dst + 16 + 4 * j) = hi;
    if (leader) *mdst = Q5Int8Meta{s, -kQ5Int8MagicF * s};
}

} // namespace infernix::ops::detail
