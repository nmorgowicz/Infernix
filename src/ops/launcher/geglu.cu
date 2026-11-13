// infernix::ops — geglu launcher: grid/block/stream configuration + kernel launch.
// The only translation unit that includes this op's kernel header.
// See docs/op-development.md §2.
#include "ops/launcher/geglu.h"

#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/kernel/geglu.cuh"
#include "core/device.h" // CUDA_CHECK

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace infernix::ops::detail {
namespace {

bool can_use_dim0_split_fast_path(const Tensor& gate, const Tensor& up, const Tensor& out) {
    if (gate.ne[2] != 1 || gate.ne[3] != 1) { return false; }
    if ((gate.ne[0] & 1) != 0) { return false; }
    if (gate.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16)) ||
        up.nb[0] != static_cast<std::int64_t>(sizeof(__nv_bfloat16))) {
        return false;
    }
    if ((gate.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0 ||
        (up.nb[1] % static_cast<std::int64_t>(sizeof(__nv_bfloat162))) != 0) {
        return false;
    }

    const auto gate_addr = reinterpret_cast<std::uintptr_t>(gate.data);
    const auto up_addr   = reinterpret_cast<std::uintptr_t>(up.data);
    const auto out_addr  = reinterpret_cast<std::uintptr_t>(out.data);
    return ((gate_addr | up_addr | out_addr) & (alignof(__nv_bfloat162) - 1)) == 0;
}

// One column per CTA. GeGLU matches geglu_kernel element for element. The FP8 codes are the
// absmax/448 E4M3 quantize of those stored BF16 values, the same rule as fp8_a8_quantize_kernel.
// The maximum does not depend on reduction order, and each pair is converted on its own.
__global__ void geglu_column_fp8_kernel(const __nv_bfloat16* __restrict__ gate,
                                        const __nv_bfloat16* __restrict__ up,
                                        __nv_bfloat16* __restrict__ out,
                                        std::uint8_t* __restrict__ codes, float* __restrict__ scales,
                                        int rows, std::int64_t gate_stride, std::int64_t up_stride) {
    constexpr int kBlock = 256;
    const int col        = static_cast<int>(blockIdx.x);
    const int tid        = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* gate_col = gate + static_cast<std::int64_t>(col) * gate_stride;
    const __nv_bfloat16* up_col   = up + static_cast<std::int64_t>(col) * up_stride;
    __nv_bfloat16* out_col        = out + static_cast<std::int64_t>(col) * rows;

    float local = 0.0f;
    for (int row = tid; row < rows; row += kBlock) {
        const float activated = geglu_gelu_tanh(__bfloat162float(gate_col[row])) *
                                __bfloat162float(up_col[row]);
        const __nv_bfloat16 stored = __float2bfloat16_rn(activated);
        out_col[row]               = stored;
        local                      = fmaxf(local, fabsf(__bfloat162float(stored)));
    }
    __shared__ float scratch[kBlock];
    scratch[tid] = local;
    __syncthreads();
    for (int offset = kBlock / 2; offset > 0; offset >>= 1) {
        if (tid < offset) { scratch[tid] = fmaxf(scratch[tid], scratch[tid + offset]); }
        __syncthreads();
    }
    __shared__ float scale;
    if (tid == 0) { scale = scratch[0] > 0.0f ? scratch[0] / 448.0f : 0.0f; }
    __syncthreads();
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    auto* pairs         = reinterpret_cast<std::uint16_t*>(codes + static_cast<std::int64_t>(col) * rows);
    for (int pair = tid; pair < rows / 2; pair += kBlock) {
        const float2 scaled = make_float2(__bfloat162float(out_col[2 * pair]) * inverse,
                                          __bfloat162float(out_col[2 * pair + 1]) * inverse);
        pairs[pair]         = __nv_cvt_float2_to_fp8x2(scaled, __NV_SATFINITE, __NV_E4M3);
    }
    if (tid == 0) { scales[col] = scale; }
}

} // namespace

void geglu_launch(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream,
                  std::uint8_t* fp8_codes, float* fp8_scales) {
    if (fp8_codes != nullptr) {
        const int rows = gate.ne[0];
        if (rows <= 0 || (rows % 256) != 0 || gate.ne[1] <= 0) {
            throw std::invalid_argument("geglu fp8: columns must be a positive multiple of 256");
        }
        const auto gate_stride = gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
        const auto up_stride   = up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16));
        geglu_column_fp8_kernel<<<gate.ne[1], 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data),
            fp8_codes, fp8_scales, rows, gate_stride, up_stride);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const std::int64_t n = out.numel();
    constexpr int kBlock = 256;
    if (!gate.is_contiguous() || !up.is_contiguous()) {
        if (can_use_dim0_split_fast_path(gate, up, out)) {
            const std::int64_t row_pairs          = gate.ne[0] / 2;
            constexpr std::int64_t kPairsPerBlock = kBlock * kGegluPairsPerThread;
            const int grid_x =
                static_cast<int>(std::max<std::int64_t>(1, div_up(row_pairs, kPairsPerBlock)));
            for_each_token_slice(gate.ne[1], 1, [&](int token_offset, int token_count) {
                const Tensor gate_slice = gate.slice(1, token_offset, token_count);
                const Tensor up_slice   = up.slice(1, token_offset, token_count);
                Tensor out_slice        = out.slice(1, token_offset, token_count);
                const dim3 grid(static_cast<unsigned int>(grid_x),
                                static_cast<unsigned int>(token_count));
                geglu_dim0_split_kernel<<<grid, kBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(gate_slice.data),
                    static_cast<const __nv_bfloat16*>(up_slice.data),
                    static_cast<__nv_bfloat16*>(out_slice.data), gate.ne[0],
                    gate.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
                    up.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)));
                CUDA_CHECK(cudaGetLastError());
            });
            return;
        }
        const int scalar_grid = static_cast<int>(std::min<std::int64_t>(
            div_up(n, static_cast<std::int64_t>(kBlock)), std::numeric_limits<int>::max()));
        geglu_strided_input_kernel<<<scalar_grid, kBlock, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(gate.data),
            static_cast<const __nv_bfloat16*>(up.data), static_cast<__nv_bfloat16*>(out.data), n,
            gate.ne[0], gate.ne[1], gate.ne[2], gate.nb[0], gate.nb[1], gate.nb[2], gate.nb[3],
            up.nb[0], up.nb[1], up.nb[2], up.nb[3]);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int grid = static_cast<int>(std::min<std::int64_t>(
        div_up(n, static_cast<std::int64_t>(kBlock)), std::numeric_limits<int>::max()));
    geglu_kernel<<<grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(gate.data),
                                              static_cast<const __nv_bfloat16*>(up.data),
                                              static_cast<__nv_bfloat16*>(out.data), n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
