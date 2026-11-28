#pragma once

#include "core/pdl.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_schedule.cuh"
#include "ops/lora/lora_device.cuh"

namespace infernix::ops::detail {

// Direct global loads retain the split-2/split-4 SIMT dataflow. Each warp owns
// contiguous 256-value phases; the CTA reduces K before invoking the epilogue. A schedule with
// kRowsPerWarp > 1 shares one warp across that many output rows: the weight planes are per-row
// loads and decodes as before, while the x vector is row-independent and its load and BF16
// converts are amortized across the rows. Every row keeps its own accumulator, its per-phase FMA
// order, its warp reduction and its cross-warp sum, so outputs are bit-identical to the
// one-row-per-warp schedules at any kRowsPerWarp.
template <class Schedule, bool FullK, bool FullTokens, class Output, class Epilogue,
          bool TriggerPdl = false, bool JoinPdl = false>
__global__
__launch_bounds__(Schedule::kThreads,
                  lora_launch_min_blocks<Epilogue>(Schedule::kThreads, Schedule::kMinBlocksPerSm)) void
q5_a16_direct_simt_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ high, const std::uint8_t* __restrict__ scales, Output output,
    Epilogue epilogue, int rows, int k, int tokens, int padded_k, int token_begin) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int RP = Schedule::kRowsPerWarp;
    constexpr int W  = Schedule::kWarpsPerRow;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int P  = Schedule::kPhasesPerWarp;
    constexpr int BK = Schedule::kBlockK;
    static_assert(RP == 1 || !kLoraEpilogue<Epilogue>,
                  "LoRA terms keep the one-row-per-warp schedules");
    // A programmatic consumer waits before it reads x or writes (a no-op in a serialized launch); a fork's
    // joining sibling (JoinPdl) computes first and waits at its end (candidates §58).
    if constexpr (!JoinPdl) pdl::enter_streaming();
    if constexpr (TriggerPdl) {
        if (threadIdx.x == 0) pdl::trigger_dependents();
    }
    __shared__ float partial[R * RP][W][T];
    const int warp          = int(threadIdx.x) >> 5;
    const int lane          = int(threadIdx.x) & 31;
    const int local_row     = warp / W;
    const int part          = warp % W;
    const int row           = int(blockIdx.x) * (R * RP) + local_row * RP;
    const int token0        = Schedule::kExactTokens ? 0 : int(blockIdx.y) * T;
    const int global_token0 = Schedule::kExactTokens ? 0 : token_begin + token0;
    if constexpr (R == 1 && RP == 1) {
        if (row >= rows) return;
    }
    const int active_tokens = FullTokens ? T : min(T, tokens - token0);
    const int logical_k     = Schedule::kStaticK ? Schedule::kStaticK : k;
    const int iterations    = (logical_k + BK - 1) / BK;
    // Excess rows (rows not divisible by the block tile) clamp their weight-row index to 0 and
    // never store; their loads then read valid plane bytes of row 0.
    std::int64_t group_base[RP];
#pragma unroll
    for (int r = 0; r < RP; ++r) {
        const int row_r = row + r;
        group_base[r]   = std::int64_t(row_r < rows ? row_r : 0) * (padded_k / 64);
    }
    [[maybe_unused]] LoraWarpStage<1, T, kLoraStageRank<Epilogue>>* lora_stage = nullptr;
    if constexpr (kLoraEpilogue<Epilogue>) {
        // The row's term operands (B row, U columns) stage in shared memory beside the main loop.
        __shared__ LoraWarpStage<1, T, kLoraStageRank<Epilogue>> lora_stages[R];
        lora_stage = &lora_stages[local_row];
        if (part == 0) {
            const int term_rows[1] = {row < rows ? row : -1};
            lora_stage_warp_b(epilogue.lora, term_rows, *lora_stage);
            lora_stage_warp_u(epilogue.lora, term_rows, global_token0, global_token0 + active_tokens, *lora_stage);
        }
    }
    float acc[RP][T]        = {};
    constexpr int kUnroll   = Schedule::kStaticK ? (Schedule::kStaticK + BK - 1) / BK : 1;
#pragma unroll kUnroll
    for (int iteration = 0; iteration < iterations; ++iteration) {
#pragma unroll
        for (int phase = 0; phase < P; ++phase) {
            const int kk = iteration * BK + (part * P + phase) * 256 + lane * 8;
            const bool active = FullK || kk + 8 <= logical_k;
            std::uint32_t scale_bits[RP];
#pragma unroll
            for (int r = 0; r < RP; ++r) {
                scale_bits[r] = 0;
                if ((lane & 7) == 0 && active)
                    scale_bits[r] = load_vec<std::uint16_t>(scales + (group_base[r] + kk / 64) * 2);
            }
#pragma unroll
            for (int r = 0; r < RP; ++r) {
                scale_bits[r] = __shfl_sync(kFullWarpMask, scale_bits[r], lane & ~7);
            }
            if (active) {
                std::uint32_t word[RP];
                std::uint8_t high_byte[RP];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    const auto group_index = group_base[r] + kk / 64;
                    word[r]      = load_vec<std::uint32_t>(codes + group_index * 32 + (kk % 64) / 2);
                    high_byte[r] = high[group_index * 8 + (kk % 64) / 8];
                }
                float weights[RP][8];
#pragma unroll
                for (int r = 0; r < RP; ++r) {
                    Q5SimtDecodeAtom::decode_eight(word[r], high_byte[r], scale_bits[r], weights[r]);
                }
#pragma unroll
                for (int token = 0; token < T; ++token) {
                    if (FullTokens || token < active_tokens) {
                        const uint4 values =
                            load_vec<uint4>(x + std::int64_t(token0 + token) * logical_k + kk);
                        const float2 x0 = bf16x2_bits_to_float2(values.x);
                        const float2 x1 = bf16x2_bits_to_float2(values.y);
                        const float2 x2 = bf16x2_bits_to_float2(values.z);
                        const float2 x3 = bf16x2_bits_to_float2(values.w);
#pragma unroll
                        for (int r = 0; r < RP; ++r) {
                            acc[r][token] = fmaf(weights[r][0], x0.x, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][1], x0.y, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][2], x1.x, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][3], x1.y, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][4], x2.x, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][5], x2.y, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][6], x3.x, acc[r][token]);
                            acc[r][token] = fmaf(weights[r][7], x3.y, acc[r][token]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) {
#pragma unroll
        for (int token = 0; token < T; ++token) {
            const float sum = warp_reduce_sum(acc[r][token]);
            if (lane == 0) partial[local_row * RP + r][part][token] = sum;
        }
    }
    __syncthreads();
    constexpr bool kRowConsumer = requires {
        epilogue.apply_row(output, row, global_token0, partial[local_row * RP][0], active_tokens);
    };
    if constexpr (kLoraEpilogue<Epilogue>) {
        // --lora: the row's reduced values plus its term (the first warp's MMA chain, lane t column t), then the
        // epilogue as below.
        static_assert(!JoinPdl, "a LoRA term reads U, which a joining launch reads before its wait");
        static_assert(T <= 32);
        if (part == 0) {
            float sum = 0.0f;
            if (lane < active_tokens) {
                sum = partial[local_row][0][lane];
#pragma unroll
                for (int split = 1; split < W; ++split) sum += partial[local_row][split][lane];
            }
            bool has;
            const float term = lora_warp_row_tokens_staged<T>(epilogue.lora, row < rows ? row : -1, global_token0,
                                                              global_token0 + active_tokens, has, *lora_stage);
            sum              = lora_add(sum, term, has);
            if (lane < active_tokens) {
                if constexpr (kRowConsumer)
                    partial[local_row][0][lane] = sum;
                else if (row < rows)
                    output.store(row, global_token0 + lane, epilogue.apply(row, global_token0 + lane, sum));
            }
        }
        if constexpr (kRowConsumer) {
            __syncthreads();
            if (part == 0 && lane == 0 && row < rows)
                linear_finish_row(output, epilogue, row, global_token0, partial[local_row][0], active_tokens);
        }
        return;
    }
#pragma unroll
    for (int r = 0; r < RP; ++r) {
        const int row_r = row + r;
        if (part == 0 && lane < active_tokens) {
            float sum = partial[local_row * RP + r][0][lane];
#pragma unroll
            for (int split = 1; split < W; ++split) sum += partial[local_row * RP + r][split][lane];
            if constexpr (kRowConsumer)
                partial[local_row * RP + r][0][lane] = sum;
            else if (row_r < rows)
                output.store(row_r, global_token0 + lane, epilogue.apply(row_r, global_token0 + lane, sum));
        }
    }
    if constexpr (kRowConsumer) {
        __syncthreads();
#pragma unroll
        for (int r = 0; r < RP; ++r) {
            if (part == 0 && lane == 0 && row + r < rows)
                linear_finish_row(output, epilogue, row + r, global_token0, partial[local_row * RP + r][0],
                                  active_tokens);
        }
    }
    if constexpr (JoinPdl) pdl::wait_for_dependencies();
}

} // namespace infernix::ops::detail
