#include "ops/linear/q4/q4_g32_wide.cuh"

#include "ops/common/q4_scale_mode.h"
#include "ops/linear/q4/q4_rowsplit_storage.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kWideWarps = 8;
constexpr int kWideRowsPerWarp = 4;
constexpr int kWideRows = kWideWarps * kWideRowsPerWarp;
constexpr int kWideTokens = 8;
constexpr int kWideTileGroups = 32;

// kScaleMode (q4_scale_mode.h): the code plane is K/2 bytes per row for every format. Modes 0, 2, 3:
// one binary16 scale covers 32 << kScaleMode columns (0 is q4_g32).
template <int kScaleMode>
__global__ void q4g32_wide_kernel(const __nv_bfloat16* __restrict__ x,
                                  const std::uint8_t* __restrict__ codes,
                                  const std::uint8_t* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ out, int rows, int k, int tokens) {
    constexpr int kGroup = 32;
    const int groups = k / kGroup;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row_base =
        (static_cast<int>(blockIdx.x) * kWideRows) + warp * kWideRowsPerWarp;
    __shared__ __align__(16) __nv_bfloat16 tile[kWideTokens][kWideTileGroups * kGroup];
    float acc[kWideRowsPerWarp][kWideTokens];
#pragma unroll
    for (int row = 0; row < kWideRowsPerWarp; ++row) {
#pragma unroll
        for (int token = 0; token < kWideTokens; ++token) { acc[row][token] = 0.0f; }
    }

    for (int group0 = 0; group0 < groups; group0 += kWideTileGroups) {
        const int live = min(kWideTileGroups, groups - group0);
        const int tile_k = live * kGroup;
        for (int item = static_cast<int>(threadIdx.x); item < tokens * tile_k; item += blockDim.x) {
            const int token = item / tile_k;
            const int elem  = item - token * tile_k;
            const int source = token < tokens ? token : 0;
            tile[token][elem] =
                token < tokens ? x[static_cast<long long>(source) * k + group0 * kGroup + elem]
                               : __nv_bfloat16{};
        }
        __syncthreads();
        if (lane < live) {
#pragma unroll
            for (int local = 0; local < kWideRowsPerWarp; ++local) {
                const int row = row_base + local;
                if (row >= rows) { continue; }
                const int group = group0 + lane;
                const std::uint8_t* code =
                    codes + (static_cast<long long>(row) * groups + group) * 16;
                const float scale = __half2float(__ushort_as_half(
                    *reinterpret_cast<const std::uint16_t*>(
                        scales + (static_cast<long long>(row) * (groups >> kScaleMode) + (group >> kScaleMode)) * 2)));
                float weights[32];
#pragma unroll
                for (int word = 0; word < 4; ++word) {
                    float piece[8];
                    Q4SimtDecodeAtom::decode_eight(
                        *reinterpret_cast<const std::uint32_t*>(code + word * 4), 0x3c00u, piece);
#pragma unroll
                    for (int i = 0; i < 8; ++i) { weights[word * 8 + i] = piece[i] * scale; }
                }
                for (int token = 0; token < tokens; ++token) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        dot = fmaf(weights[i], __bfloat162float(tile[token][lane * kGroup + i]), dot);
                    }
                    acc[local][token] += dot;
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int local = 0; local < kWideRowsPerWarp; ++local) {
        const int row = row_base + local;
        for (int token = 0; token < tokens; ++token) {
            float sum = acc[local][token];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_xor_sync(0xffffffffu, sum, offset);
            }
            if (lane == 0 && row < rows) {
                out[row + static_cast<long long>(token) * rows] = __float2bfloat16(sum);
            }
        }
    }
}

} // namespace

void q4_g32_wide(const Q4LinearOperands& operands, __nv_bfloat16* out, cudaStream_t stream,
                 int scale_mode) {
    const dim3 block(kWideWarps * 32);
    const dim3 grid((operands.rows + kWideRows - 1) / kWideRows);
    const auto run = [&]<int kScaleMode>() {
        for (int done = 0; done < operands.tokens; done += kWideTokens) {
            const int tile = std::min(kWideTokens, operands.tokens - done);
            q4g32_wide_kernel<kScaleMode><<<grid, block, 0, stream>>>(
                operands.x + static_cast<long long>(done) * operands.k, operands.codes,
                operands.scales, out + static_cast<long long>(done) * operands.rows, operands.rows,
                operands.k, tile);
        }
    };
    switch (scale_mode) {
    case 0: run.template operator()<0>(); break;
    case 2: run.template operator()<2>(); break;
    case 3: run.template operator()<3>(); break;
    default: throw std::invalid_argument("q4 wide GEMV: unsupported scale mode");
    }
}

} // namespace infernix::ops::detail
