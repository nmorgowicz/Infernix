#include "ops/launcher/gemma4_qkv_prep.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/kernel/gemma4_fp8_kv.cuh"
#include "ops/kernel/rmsnorm.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace infernix::ops::detail {
namespace {

constexpr int kWarpsPerBlock = 4;

// Lane l holds BF16 pairs l + 32*s. The sum of squares is reduced in the same order as the
// unfused rmsnorm launch: one warp per row for D256, one 256-thread CTA per row for D512 (thread
// t holds pair t, so CTA warp w is slot w here). That keeps the result bit-identical.
template <int D>
__device__ __forceinline__ float head_inv(const float2 (&x)[D / 64], int d, float eps, int lane) {
    float total = 0.0f;
    if constexpr (D == 256) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < 4; ++s) { sum += x[s].x * x[s].x + x[s].y * x[s].y; }
        total = warp_reduce_sum(sum);
    } else {
        static_assert(D == 512);
        float part[8];
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            float sum = 0.0f;
            sum += x[s].x * x[s].x + x[s].y * x[s].y;
            part[s] = warp_reduce_sum(sum);
        }
        total = ((part[0] + part[4]) + (part[2] + part[6])) +
                ((part[1] + part[5]) + (part[3] + part[7]));
    }
    float inv = lane == 0 ? rsqrtf(total / static_cast<float>(d) + eps) : 0.0f;
    return __shfl_sync(kFullWarpMask, inv, 0);
}

template <int D>
__device__ __forceinline__ void load_head(const __nv_bfloat162* source, int lane,
                                          float2 (&x)[D / 64]) {
#pragma unroll
    for (int s = 0; s < D / 64; ++s) { x[s] = __bfloat1622float2(source[lane + 32 * s]); }
}

template <int D>
__device__ __forceinline__ void normalize(const float2 (&x)[D / 64], const __nv_bfloat162* weight,
                                          float inv, int lane, __nv_bfloat162 (&out)[D / 64]) {
#pragma unroll
    for (int s = 0; s < D / 64; ++s) {
        const float2 wf =
            weight != nullptr ? __bfloat1622float2(weight[lane + 32 * s]) : make_float2(1.f, 1.f);
        out[s] =
            __floats2bfloat162_rn(rmsnorm_epilogue<RmsEpilogue::Plain>(x[s].x, inv, wf.x, 0.0f),
                                  rmsnorm_epilogue<RmsEpilogue::Plain>(x[s].y, inv, wf.y, 0.0f));
    }
}

// Rotates slots [0, Rot) against slots [Rot, 2*Rot): element e pairs with e + rotary_dim/2.
template <int D, int Rot>
__device__ __forceinline__ void rotate(__nv_bfloat162 (&v)[D / 64], int position, int rotary_dim,
                                       float theta, int lane) {
#pragma unroll
    for (int s = 0; s < Rot; ++s) {
        const float2 first  = __bfloat1622float2(v[s]);
        const float2 second = __bfloat1622float2(v[s + Rot]);
        const int element   = 2 * (lane + 32 * s);
        float c[2];
        float sn[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float exponent =
                -2.0F * static_cast<float>(element + j) / static_cast<float>(rotary_dim);
            const float frequency = powf(theta, exponent);
            const float angle     = static_cast<float>(position) * frequency;
            sincosf(angle, &sn[j], &c[j]);
        }
        v[s]       = __floats2bfloat162_rn(first.x * c[0] - second.x * sn[0],
                                           first.y * c[1] - second.y * sn[1]);
        v[s + Rot] = __floats2bfloat162_rn(second.x * c[0] + first.x * sn[0],
                                           second.y * c[1] + first.y * sn[1]);
    }
}

template <int D, int Rot, Gemma4KvFormat Fmt>
__global__ __launch_bounds__(kWarpsPerBlock * 32) void gemma4_qkv_prep_kernel(
    const __nv_bfloat162* qkv, std::int64_t column_pairs, const __nv_bfloat162* q_norm,
    const __nv_bfloat162* k_norm, const std::int32_t* positions, __nv_bfloat162* q_out,
    void* k_pages, void* v_pages, __half* k_scales, __half* v_scales,
    const std::int32_t* block_table, int q_heads,
    int kv_heads, int rotary_dim, int d, float theta, float eps, bool v_from_k, int logical_pages,
    int physical_pages) {
    constexpr int kHeadPairs = D / 2;
    const int lane           = static_cast<int>(threadIdx.x) & 31;
    const int head =
        static_cast<int>(blockIdx.x) * kWarpsPerBlock + (static_cast<int>(threadIdx.x) >> 5);
    if (head >= q_heads + kv_heads) { return; }
    const int token              = static_cast<int>(blockIdx.y);
    const int position           = positions[token];
    const __nv_bfloat162* column = qkv + static_cast<std::int64_t>(token) * column_pairs;

    float2 x[D / 64];
    __nv_bfloat162 out[D / 64];
    if (head < q_heads) {
        load_head<D>(column + static_cast<std::int64_t>(head) * kHeadPairs, lane, x);
        normalize<D>(x, q_norm, head_inv<D>(x, d, eps, lane), lane, out);
        rotate<D, Rot>(out, position, rotary_dim, theta, lane);
        __nv_bfloat162* dest =
            q_out + (static_cast<std::int64_t>(token) * q_heads + head) * kHeadPairs;
#pragma unroll
        for (int s = 0; s < D / 64; ++s) { dest[lane + 32 * s] = out[s]; }
        return;
    }

    const int kv_head   = head - q_heads;
    const int page      = position >> kPagedKVPageShift;
    const bool writable = position >= 0 && page < logical_pages;
    const int physical  = writable ? block_table[page] : -1;
    if (physical < 0 || physical >= physical_pages) { return; }
    const std::int64_t slot =
        (static_cast<std::int64_t>(D) * kPagedKVPageSize *
             (static_cast<std::int64_t>(kv_head) + static_cast<std::int64_t>(kv_heads) * physical) +
         static_cast<std::int64_t>(D) * (position & kPagedKVPageMask)) /
        2;
    const std::int64_t k_rows = static_cast<std::int64_t>(q_heads) * kHeadPairs;
    load_head<D>(column + k_rows + static_cast<std::int64_t>(kv_head) * kHeadPairs, lane, x);
    const float k_inv = head_inv<D>(x, d, eps, lane);
    normalize<D>(x, k_norm, k_inv, lane, out);
    rotate<D, Rot>(out, position, rotary_dim, theta, lane);
    if constexpr (Fmt == Gemma4KvFormat::Fp8E4m3Row256) {
        gemma4_fp8_store(out, static_cast<std::uint8_t*>(k_pages), k_scales, slot * 2 / D, lane);
    } else {
#pragma unroll
        for (int s = 0; s < D / 64; ++s) {
            static_cast<__nv_bfloat162*>(k_pages)[slot + lane + 32 * s] = out[s];
        }
    }

    if (!v_from_k) {
        const std::int64_t v_rows = k_rows + static_cast<std::int64_t>(kv_heads) * kHeadPairs;
        load_head<D>(column + v_rows + static_cast<std::int64_t>(kv_head) * kHeadPairs, lane, x);
    }
    normalize<D>(x, nullptr, v_from_k ? k_inv : head_inv<D>(x, d, eps, lane), lane, out);
    if constexpr (Fmt == Gemma4KvFormat::Fp8E4m3Row256) {
        gemma4_fp8_store(out, static_cast<std::uint8_t*>(v_pages), v_scales, slot * 2 / D, lane);
    } else {
#pragma unroll
        for (int s = 0; s < D / 64; ++s) {
            static_cast<__nv_bfloat162*>(v_pages)[slot + lane + 32 * s] = out[s];
        }
    }
}

template <int D, int Rot, Gemma4KvFormat Fmt>
void launch(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm, const Tensor& positions,
            int rotary_dim, float theta, float eps, bool v_from_k, Tensor& q_out,
            const Gemma4CacheView& cache, cudaStream_t stream) {
    constexpr bool Quantized = Fmt != Gemma4KvFormat::BFloat16;
    const Tensor& k_pages = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    const int q_heads  = q_out.ne[1];
    const int kv_heads = k_pages.ne[2];
    const int tokens   = q_out.ne[2];
    const dim3 grid(
        static_cast<unsigned>((q_heads + kv_heads + kWarpsPerBlock - 1) / kWarpsPerBlock),
        static_cast<unsigned>(tokens));
    gemma4_qkv_prep_kernel<D, Rot, Fmt><<<grid, kWarpsPerBlock * 32, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(qkv.data), qkv.ne[0] / 2,
        static_cast<const __nv_bfloat162*>(q_norm.data),
        static_cast<const __nv_bfloat162*>(k_norm.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat162*>(q_out.data),
        k_pages.data, cache.v_pages->data,
        Quantized ? static_cast<__half*>(cache.k_scales->data) : nullptr,
        Quantized ? static_cast<__half*>(cache.v_scales->data) : nullptr,
        static_cast<const std::int32_t*>(block_table.data), q_heads, kv_heads, rotary_dim, D, theta,
        eps, v_from_k, block_table.ne[0], k_pages.ne[3]);
}

} // namespace

void gemma4_qkv_prep_launch(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                            const Tensor& positions, int rotary_dim, float theta, float eps,
                            bool v_from_k, Tensor& q_out, const Gemma4CacheView& cache,
                            cudaStream_t stream) {
    const int head_dim = q_out.ne[0];
    if (head_dim == 256 && rotary_dim == 256) {
        launch<256, 2, Gemma4KvFormat::BFloat16>(qkv, q_norm, k_norm, positions, rotary_dim, theta,
                                                 eps, v_from_k, q_out, cache, stream);
    } else if (head_dim == 512 && rotary_dim == 128) {
        if (cache.format == Gemma4KvFormat::Fp8E4m3Row256) {
            launch<512, 1, Gemma4KvFormat::Fp8E4m3Row256>(qkv, q_norm, k_norm, positions,
                                                          rotary_dim, theta, eps, v_from_k, q_out,
                                                          cache, stream);
        } else {
            launch<512, 1, Gemma4KvFormat::BFloat16>(qkv, q_norm, k_norm, positions, rotary_dim,
                                                     theta, eps, v_from_k, q_out, cache, stream);
        }
    } else {
        throw std::invalid_argument("gemma4_qkv_prep: D/rotary_dim must be 256/256 or 512/128");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
