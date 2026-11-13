#pragma once

// Large-T row-scaled E4M3 weight x BF16 activation GEMM with a multi-stage cp.async pipeline.
//
//   out[M,T] = (scale[M] * e4m3_codes[M,K]) * x[K,T]
//
// E4M3 codes stay 8-bit in shared memory and are widened exactly into the BF16 MMA A fragments,
// so a K tile costs one barrier and no shared-memory widening pass. Each output accumulates its
// 64-wide K tiles and 16-wide MMA steps in the same order as the unsplit fp8_a16_gemm_mma_kernel,
// so both kernels produce the same bits.

#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace infernix::ops::detail {

template <int BlockRows, int BlockTokens, int WarpRows, int WarpTokens, int Stages>
struct Fp8A16PipeSchedule {
    static constexpr int kBlockRows   = BlockRows;
    static constexpr int kBlockTokens = BlockTokens;
    static constexpr int kBlockK      = 64;
    static constexpr int kWarpRows    = WarpRows;
    static constexpr int kWarpTokens  = WarpTokens;
    static constexpr int kStages      = Stages;
    static constexpr int kWarpsRows   = kBlockRows / kWarpRows;
    static constexpr int kWarpsTokens = kBlockTokens / kWarpTokens;
    static constexpr int kWarps       = kWarpsRows * kWarpsTokens;
    static constexpr int kThreads     = kWarps * 32;
    static constexpr int kMmaRows     = kWarpRows / 16;
    static constexpr int kMmaTokens   = kWarpTokens / 8;
    static constexpr int kCodeStageBytes       = kBlockRows * kBlockK;
    static constexpr int kActivationStageBytes = kBlockTokens * kBlockK * 2;
    static constexpr int kStageBytes           = kCodeStageBytes + kActivationStageBytes;
    static constexpr int kSharedBytes          = kStages * kStageBytes;

    static_assert((kBlockRows % kWarpRows) == 0 && (kBlockTokens % kWarpTokens) == 0);
    static_assert((kWarpRows % 16) == 0 && (kWarpTokens % 16) == 0);
    static_assert(kStages >= 2 && kStages <= 6);
    static_assert(kSharedBytes <= 99 * 1024);
};

// The 16-byte chunk of a 64-code row rotates with the row pair, so the eight rows of one A
// fragment load fall in distinct banks.
__device__ __forceinline__ int fp8_a16_pipe_code_offset(int row, int col) {
    return row * 64 + ((((col >> 4) ^ (row >> 1)) & 3) << 4) + (col & 15);
}

template <class Geometry, class Schedule, bool FullTokens, class Output = Fp8ContiguousOutput>
__global__ __launch_bounds__(Schedule::kThreads, 1) void fp8_a16_pipe_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ weight_codes,
    const __nv_bfloat16* __restrict__ row_scales, Output output, std::int32_t tokens) {
    constexpr int K       = Geometry::kInputRows;
    constexpr int BM      = Schedule::kBlockRows;
    constexpr int BN      = Schedule::kBlockTokens;
    constexpr int BK      = Schedule::kBlockK;
    constexpr int WM      = Schedule::kWarpRows;
    constexpr int WN      = Schedule::kWarpTokens;
    constexpr int MT      = Schedule::kMmaRows;
    constexpr int NT      = Schedule::kMmaTokens;
    constexpr int STAGES  = Schedule::kStages;
    constexpr int THREADS = Schedule::kThreads;
    constexpr int kTiles  = K / BK;
    static_assert((Geometry::kOutputRows % BM) == 0 && (K % BK) == 0);

    extern __shared__ __align__(16) unsigned char shared_raw[];
    const auto codes_at = [&](int stage) { return shared_raw + stage * Schedule::kStageBytes; };
    const auto activations_at = [&](int stage) {
        return reinterpret_cast<__nv_bfloat16*>(shared_raw + stage * Schedule::kStageBytes +
                                                Schedule::kCodeStageBytes);
    };

    const int tid  = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm   = warp / Schedule::kWarpsTokens;
    const int wn   = warp - wm * Schedule::kWarpsTokens;
    const int gid  = lane >> 2;
    const int lid  = lane & 3;

    const int row_begin   = static_cast<int>(blockIdx.x) * BM;
    const int token_begin = static_cast<int>(blockIdx.y) * BN;

    const auto load_stage = [&](int stage, int k_tile) {
        const int k_begin          = k_tile * BK;
        __nv_bfloat16* activations = activations_at(stage);
#pragma unroll
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int local_token = item / (BK / 8);
            const int kk          = (item - local_token * (BK / 8)) * 8;
            auto* destination =
                &activations[local_token * BK + fp8_a16_shared_col_64(local_token, kk)];
            const int token = token_begin + local_token;
            if constexpr (FullTokens) {
                cp_async<16, Cache::cg>(destination,
                                        x + static_cast<std::int64_t>(token) * K + k_begin + kk);
            } else {
                const bool valid = token < tokens;
                cp_async_zfill<16, Cache::cg>(
                    destination, x + static_cast<std::int64_t>(valid ? token : 0) * K + k_begin + kk,
                    valid ? 16 : 0);
            }
        }
        unsigned char* codes = codes_at(stage);
#pragma unroll
        for (int item = tid; item < BM * (BK / 16); item += THREADS) {
            const int local_row = item / (BK / 16);
            const int col       = (item - local_row * (BK / 16)) * 16;
            cp_async<16, Cache::cg>(
                &codes[fp8_a16_pipe_code_offset(local_row, col)],
                weight_codes + static_cast<std::int64_t>(row_begin + local_row) * K + k_begin + col);
        }
    };

    float accumulators[MT][NT][4] = {};

#pragma unroll
    for (int stage = 0; stage < STAGES - 1; ++stage) {
        if (stage < kTiles) { load_stage(stage, stage); }
        cp_commit();
    }

    const int b_row = (lane & 7) + ((lane >> 4) << 3);
    const int b_col = ((lane >> 3) & 1) << 3;

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = k_tile + STAGES - 1;
        if (next < kTiles) { load_stage(next % STAGES, next); }
        cp_commit();

        const int stage                  = k_tile % STAGES;
        const unsigned char* codes       = codes_at(stage);
        const __nv_bfloat16* activations = activations_at(stage);
        const auto widen                 = [&](int row, int col) {
            return fp8_e4m3x2_to_bf16x2_bits(static_cast<unsigned>(
                *reinterpret_cast<const std::uint16_t*>(&codes[fp8_a16_pipe_code_offset(row, col)])));
        };

        unsigned a_fragments[2][MT][4];
        unsigned b_fragments[2][NT][2];
        const auto load_fragments = [&](int slot, int k_step) {
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
                const int row0                = wm * WM + mma_row * 16 + gid;
                const int col0                = k_step * 16 + 2 * lid;
                a_fragments[slot][mma_row][0] = widen(row0, col0);
                a_fragments[slot][mma_row][1] = widen(row0 + 8, col0);
                a_fragments[slot][mma_row][2] = widen(row0, col0 + 8);
                a_fragments[slot][mma_row][3] = widen(row0 + 8, col0 + 8);
            }
#pragma unroll
            for (int pair = 0; pair < NT / 2; ++pair) {
                const int row = wn * WN + pair * 16 + b_row;
                const int col = k_step * 16 + b_col;
                ldmatrix_x4(b_fragments[slot][2 * pair][0], b_fragments[slot][2 * pair][1],
                            b_fragments[slot][2 * pair + 1][0], b_fragments[slot][2 * pair + 1][1],
                            smem_addr(&activations[row * BK + fp8_a16_shared_col_64(row, col)]));
            }
        };

        load_fragments(0, 0);
#pragma unroll
        for (int k_step = 0; k_step < BK / 16; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < BK / 16) { load_fragments(slot ^ 1, k_step + 1); }
#pragma unroll
            for (int mma_row = 0; mma_row < MT; ++mma_row) {
#pragma unroll
                for (int mma_token = 0; mma_token < NT; ++mma_token) {
                    mma_bf16(
                        accumulators[mma_row][mma_token][0], accumulators[mma_row][mma_token][1],
                        accumulators[mma_row][mma_token][2], accumulators[mma_row][mma_token][3],
                        a_fragments[slot][mma_row][0], a_fragments[slot][mma_row][1],
                        a_fragments[slot][mma_row][2], a_fragments[slot][mma_row][3],
                        b_fragments[slot][mma_token][0], b_fragments[slot][mma_token][1]);
                }
            }
        }
    }
    cp_wait<0>();

#pragma unroll
    for (int mma_row = 0; mma_row < MT; ++mma_row) {
        const int row0     = row_begin + wm * WM + mma_row * 16 + gid;
        const int row1     = row0 + 8;
        const float scale0 = __bfloat162float(row_scales[row0]);
        const float scale1 = __bfloat162float(row_scales[row1]);
#pragma unroll
        for (int mma_token = 0; mma_token < NT; ++mma_token) {
            const int token0    = token_begin + wn * WN + mma_token * 8 + 2 * lid;
            const int token1    = token0 + 1;
            const float* values = accumulators[mma_row][mma_token];
            if (FullTokens || token0 < tokens) {
                output.store(row0, token0, values[0] * scale0);
                output.store(row1, token0, values[2] * scale1);
            }
            if (FullTokens || token1 < tokens) {
                output.store(row0, token1, values[1] * scale0);
                output.store(row1, token1, values[3] * scale1);
            }
        }
    }
}

} // namespace infernix::ops::detail
