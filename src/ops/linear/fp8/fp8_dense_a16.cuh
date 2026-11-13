#pragma once
// Weight-only FP8 routes for a dense [N,K] projection: one token uses the GEMV, compact batches
// and prefill segments up to 32-48 tokens the K-split Tensor Core kernel, and longer prefill the
// A16 GEMM. The GEMM walks all of K in each of its N/64 x T/128 CTAs, so when that grid leaves
// most SMs idle a CTA cluster splits K and reduces in shared memory.
// Schedules were measured on 48 cold weight copies per Flash-Next shape (RTX PRO 6000); the
// medium-T routes on 10 cold samples per shape at T=16..1024. Long unsplit prefill moves to the
// pipelined GEMM, which keeps the A16 GEMM's per-output K order and so its bits.
#include "ops/linear/fp8/fp8_shapes.h"
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_ksplit_mma.cuh"
#include "ops/linear/fp8/fp8_a16_pipe_mma.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_template_launch.cuh"

namespace infernix::ops::detail::fp8_dense {

// The T=1 GEMV is infernix's FP8 A16 GEMV at the fork's schedule: the same FP32 FMA chains and the
// same single row-scale multiply, so a column has the fork's bits.
using Gemv        = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;
using Prefill     = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
using LongPrefill = Fp8A16PipeSchedule<128, 256, 64, 64, 2>;
// Measured at T=129..8192 (RTX PRO 6000, 300 W): the pipelined GEMM wins 1.1-1.4x from about
// 1.5k tokens once its 128x256 grid has 120 CTAs; smaller grids leave SMs walking K alone.
inline constexpr int kLongPrefillMinTokens = 1536;
inline constexpr int kLongPrefillMinCtas   = 120;
inline constexpr int kKSplitMaxTokens      = 16;
inline constexpr int kWideKSplitMaxTokens  = 48;
// Measured: a grid of more than 160 CTAs gains nothing from a split, and a split grid of up to 384
// CTAs (two per SM on 188 SMs plus a short tail) still beats the next smaller split.
inline constexpr int kSplitMaxBaseCtas = 160;
inline constexpr int kSplitTargetCtas  = 384;

// Eight K-split warps walk K in 512-wide groups; a K that is only a multiple of 256 takes four.
template <class Geometry>
inline constexpr int kKSplitWarps = (Geometry::kInputRows % 512) == 0 ? 8 : 4;

template <class Geometry>
void launch_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_gemv<Fp8ScheduleInstance<Gemv, Geometry::kInputRows>>(
        fp8_a16_operands(x, weight),
        LinearBf16Output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        LinearIdentityEpilogue{}, stream);
}

template <class Geometry, int TileTokens, int KWarps = kKSplitWarps<Geometry>,
          int MinBlocksPerSm = 1>
void launch_ksplit(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Fp8A16KSplitSchedule<KWarps, TileTokens, MinBlocksPerSm>;
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    fp8_a16_ksplit_mma_kernel<Geometry, TileTokens, Schedule, Fp8ContiguousOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, bool FullTokens>
void launch_prefill_slice(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Prefill::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Prefill::kBlockK) == 0);
    const dim3 grid(static_cast<unsigned>(Geometry::kOutputRows / Prefill::kBlockRows),
                    static_cast<unsigned>(div_up(x.ne[1], Prefill::kBlockTokens)), 1U);
    if constexpr (Prefill::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Prefill::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens>
        <<<grid, Prefill::kThreads, Prefill::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, bool FullTokens, int KSplit>
void launch_prefill_split(const Tensor& x, const Weight& weight, Tensor& out,
                          cudaStream_t stream) {
    static_assert(Prefill::kSharedBytes <= 48 * 1024);
    cudaLaunchAttribute cluster{};
    cluster.id               = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim.x = 1;
    cluster.val.clusterDim.y = 1;
    cluster.val.clusterDim.z = KSplit;
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(static_cast<unsigned>(Geometry::kOutputRows / Prefill::kBlockRows),
                           static_cast<unsigned>(div_up(x.ne[1], Prefill::kBlockTokens)),
                           static_cast<unsigned>(KSplit));
    config.blockDim         = dim3(Prefill::kThreads, 1U, 1U);
    config.dynamicSmemBytes = Prefill::kSharedBytes;
    config.stream           = stream;
    config.attrs            = &cluster;
    config.numAttrs         = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &config,
        fp8_a16_gemm_mma_kernel<Geometry, Prefill, FullTokens, Fp8ContiguousOutput, KSplit>,
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
        static_cast<std::int32_t>(x.ne[1])));
}

// A cluster split divides the K tiles evenly among its CTAs.
template <class Geometry, int Split>
inline constexpr bool kSplitFits = ((Geometry::kInputRows / Prefill::kBlockK) % Split) == 0;

// Largest fitting cluster split that keeps the launch within about two CTAs per SM.
template <class Geometry>
int prefill_k_split(int row_tiles, int token_tiles) {
    const int ctas = row_tiles * token_tiles;
    if (ctas > kSplitMaxBaseCtas) return 1;
    constexpr bool kFits[] = {kSplitFits<Geometry, 8>, kSplitFits<Geometry, 4>,
                              kSplitFits<Geometry, 2>};
    constexpr int kSplits[] = {8, 4, 2};
    for (int index = 0; index < 3; ++index) {
        if (kFits[index] && ctas * kSplits[index] <= kSplitTargetCtas) return kSplits[index];
    }
    return 1;
}

template <class Geometry, bool FullTokens>
void launch_long_prefill(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static const cudaError_t attribute = cudaFuncSetAttribute(
        fp8_a16_pipe_mma_kernel<Geometry, LongPrefill, FullTokens>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, LongPrefill::kSharedBytes);
    CUDA_CHECK(attribute);
    const dim3 grid(static_cast<unsigned>(Geometry::kOutputRows / LongPrefill::kBlockRows),
                    static_cast<unsigned>(div_up(x.ne[1], LongPrefill::kBlockTokens)), 1U);
    fp8_a16_pipe_mma_kernel<Geometry, LongPrefill, FullTokens>
        <<<grid, LongPrefill::kThreads, LongPrefill::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows},
            x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// Only where the A16 GEMM would run unsplit, so the route never changes an output bit.
template <class Geometry>
bool uses_long_prefill(int tokens, int split) {
    return split == 1 && tokens >= kLongPrefillMinTokens &&
           (Geometry::kOutputRows / LongPrefill::kBlockRows) *
                   div_up(tokens, LongPrefill::kBlockTokens) >=
               kLongPrefillMinCtas;
}

template <class Geometry, bool FullTokens>
void launch_prefill(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int split = prefill_k_split<Geometry>(Geometry::kOutputRows / Prefill::kBlockRows,
                                                div_up(x.ne[1], Prefill::kBlockTokens));
    if constexpr (kSplitFits<Geometry, 8>) {
        if (split == 8) return launch_prefill_split<Geometry, FullTokens, 8>(x, weight, out, stream);
    }
    if constexpr (kSplitFits<Geometry, 4>) {
        if (split == 4) return launch_prefill_split<Geometry, FullTokens, 4>(x, weight, out, stream);
    }
    if constexpr (kSplitFits<Geometry, 2>) {
        if (split == 2) return launch_prefill_split<Geometry, FullTokens, 2>(x, weight, out, stream);
    }
    launch_prefill_slice<Geometry, FullTokens>(x, weight, out, stream);
}

template <class Geometry>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int kWarps = kKSplitWarps<Geometry>;
    const int tokens     = x.ne[1];
    if (tokens == 1) return launch_gemv<Geometry>(x, weight, out, stream);
    if (tokens <= 8) return launch_ksplit<Geometry, 8>(x, weight, out, stream);
    if (tokens <= kKSplitMaxTokens) return launch_ksplit<Geometry, 16>(x, weight, out, stream);
    if (tokens <= 24) return launch_ksplit<Geometry, 24, kWarps, 1>(x, weight, out, stream);
    if (tokens <= 32) return launch_ksplit<Geometry, 32, kWarps, 1>(x, weight, out, stream);
    // Four-warp tiles walk K in 256-wide groups; on the 10240-wide input the split GEMM is faster.
    if constexpr (Geometry::kInputRows <= 6144) {
        if (tokens <= 40) return launch_ksplit<Geometry, 40, 4, 2>(x, weight, out, stream);
        if (tokens <= kWideKSplitMaxTokens) {
            return launch_ksplit<Geometry, 48, 4, 2>(x, weight, out, stream);
        }
    }
    if constexpr ((Geometry::kOutputRows % LongPrefill::kBlockRows) == 0) {
        const int split = prefill_k_split<Geometry>(Geometry::kOutputRows / Prefill::kBlockRows,
                                                    div_up(tokens, Prefill::kBlockTokens));
        if (uses_long_prefill<Geometry>(tokens, split)) {
            for_each_token_slice(
                tokens, LongPrefill::kBlockTokens, [&](std::int32_t offset, std::int32_t count) {
                    const Tensor input = x.slice(1, offset, count);
                    Tensor output      = out.slice(1, offset, count);
                    if ((count % LongPrefill::kBlockTokens) == 0) {
                        launch_long_prefill<Geometry, true>(input, weight, output, stream);
                    } else {
                        launch_long_prefill<Geometry, false>(input, weight, output, stream);
                    }
                });
            return;
        }
    }
    for_each_token_slice(tokens, Prefill::kBlockTokens, [&](std::int32_t offset, std::int32_t count) {
        const Tensor input = x.slice(1, offset, count);
        Tensor output      = out.slice(1, offset, count);
        if ((count % Prefill::kBlockTokens) == 0) {
            launch_prefill<Geometry, true>(input, weight, output, stream);
        } else {
            launch_prefill<Geometry, false>(input, weight, output, stream);
        }
    });
}

inline bool uses_a8(std::int32_t, std::int32_t) { return false; }

template <int N, int K>
constexpr Fp8LinearShape shape() {
    return {N, K, launch_a16<Fp8Geometry<N, K>>, nullptr, uses_a8};
}

} // namespace infernix::ops::detail::fp8_dense
