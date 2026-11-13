#include "ops/launcher/topk_columns.h"

#include "core/device.h"

#include <cuda_bf16.h>

#include <math_constants.h>

#include <climits>
#include <cstdint>

namespace infernix::ops::detail {
namespace {

constexpr int kTopkThreads = 1024;
constexpr int kTopkWarps   = kTopkThreads / 32;
constexpr std::int32_t kNone = INT32_MAX;

// The ops::argmax order as a total order: a NaN in row 0 beats everything (argmax starts from
// row 0 and nothing compares greater than NaN), a NaN in any other row loses to every number,
// a larger value wins, and an equal value keeps the lower index. kNone is an empty slot.
__device__ __forceinline__ int topk_class(float value, std::int32_t index) {
    if (index == kNone) { return 0; }
    if (isnan(value)) { return index == 0 ? 3 : 1; }
    return 2;
}

__device__ __forceinline__ bool topk_better(float value, std::int32_t index, float best_value,
                                            std::int32_t best_index) {
    const int a = topk_class(value, index);
    const int b = topk_class(best_value, best_index);
    if (a != b) { return a > b; }
    if (a != 2) { return index < best_index; }
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void topk_warp_best(float& value, std::int32_t& index) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other_value        = __shfl_xor_sync(0xffffffffU, value, offset);
        const std::int32_t other_index = __shfl_xor_sync(0xffffffffU, index, offset);
        if (topk_better(other_value, other_index, value, index)) {
            value = other_value;
            index = other_index;
        }
    }
}

// One CTA per column. Each thread keeps its best K rows, sorted. Rank r is then the best head
// over all threads; the thread that owned it moves to its next row.
template <int K>
__launch_bounds__(kTopkThreads) __global__
    void topk_columns_kernel(const __nv_bfloat16* logits, std::int32_t* ids,
                             std::int32_t valid_rows, std::int32_t physical_rows, bool vector) {
    const std::int32_t token = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(token) * physical_rows;
    const int tid            = static_cast<int>(threadIdx.x);
    const int lane           = tid & 31;
    const int warp           = tid >> 5;

    float local_value[K];
    std::int32_t local_index[K];
#pragma unroll
    for (int slot = 0; slot < K; ++slot) {
        local_value[slot] = -CUDART_INF_F;
        local_index[slot] = kNone;
    }
    const auto consider = [&](std::int32_t row, float value) {
        // A number below the last slot loses; NaN, an empty slot, and ties take the full order.
        if (value < local_value[K - 1]) { return; }
        if (!topk_better(value, row, local_value[K - 1], local_index[K - 1])) { return; }
        int place = K - 1;
#pragma unroll
        for (int slot = K - 1; slot > 0; --slot) {
            if (topk_better(value, row, local_value[slot - 1], local_index[slot - 1])) {
                local_value[slot] = local_value[slot - 1];
                local_index[slot] = local_index[slot - 1];
                place             = slot - 1;
            }
        }
        // The list is sorted, so once a slot beats the row every earlier slot does too.
#pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            if (slot == place) {
                local_value[slot] = value;
                local_index[slot] = row;
            }
        }
    };
    // The order is total, so the result does not depend on which thread sees which row.
    std::int32_t scalar_from = 0;
    if (vector) {
        const auto* chunks = reinterpret_cast<const int4*>(logits + base);
        const int full     = valid_rows / 8;
#pragma unroll 2
        for (int chunk = tid; chunk < full; chunk += kTopkThreads) {
            const int4 raw    = chunks[chunk];
            const auto* pairs = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
            for (int pair = 0; pair < 4; ++pair) {
                const float2 values = __bfloat1622float2(pairs[pair]);
                consider(chunk * 8 + 2 * pair, values.x);
                consider(chunk * 8 + 2 * pair + 1, values.y);
            }
        }
        scalar_from = full * 8;
    }
    for (std::int32_t row = scalar_from + tid; row < valid_rows; row += kTopkThreads) {
        consider(row, __bfloat162float(logits[base + row]));
    }

    __shared__ float warp_value[kTopkWarps];
    __shared__ std::int32_t warp_index[kTopkWarps];
    __shared__ std::int32_t winner;
    int head = 0;
    for (int rank = 0; rank < K; ++rank) {
        float value        = 0.0f;
        std::int32_t index = kNone;
#pragma unroll
        for (int slot = 0; slot < K; ++slot) {
            if (slot == head) {
                value = local_value[slot];
                index = local_index[slot];
            }
        }
        topk_warp_best(value, index);
        if (lane == 0) {
            warp_value[warp] = value;
            warp_index[warp] = index;
        }
        __syncthreads();
        if (warp == 0) {
            value = lane < kTopkWarps ? warp_value[lane] : 0.0f;
            index = lane < kTopkWarps ? warp_index[lane] : kNone;
            topk_warp_best(value, index);
            if (lane == 0) {
                winner = index;
                ids[static_cast<std::int64_t>(token) * K + rank] = index == kNone ? -1 : index;
            }
        }
        __syncthreads();
        if (winner != kNone && head < K) {
            std::int32_t mine = kNone;
#pragma unroll
            for (int slot = 0; slot < K; ++slot) {
                if (slot == head) { mine = local_index[slot]; }
            }
            if (mine == winner) { ++head; }
        }
    }
}

} // namespace

void topk_columns_launch(const Tensor& logits, std::int32_t valid_rows, std::int32_t k,
                         Tensor& ids, cudaStream_t stream) {
    const auto* values = static_cast<const __nv_bfloat16*>(logits.data);
    auto* out          = static_cast<std::int32_t*>(ids.data);
    const auto tokens  = static_cast<unsigned>(logits.ne[1]);
    const bool vector  = reinterpret_cast<std::uintptr_t>(values) % 16 == 0 && logits.ne[0] % 8 == 0;
    const auto launch  = [&](auto kernel) {
        kernel<<<tokens, kTopkThreads, 0, stream>>>(values, out, valid_rows, logits.ne[0], vector);
        CUDA_CHECK(cudaGetLastError());
    };
    switch (k) {
    case 1: launch(topk_columns_kernel<1>); break;
    case 2: launch(topk_columns_kernel<2>); break;
    case 3: launch(topk_columns_kernel<3>); break;
    case 4: launch(topk_columns_kernel<4>); break;
    case 5: launch(topk_columns_kernel<5>); break;
    case 6: launch(topk_columns_kernel<6>); break;
    case 7: launch(topk_columns_kernel<7>); break;
    case 8: launch(topk_columns_kernel<8>); break;
    default: break;
    }
}

} // namespace infernix::ops::detail
