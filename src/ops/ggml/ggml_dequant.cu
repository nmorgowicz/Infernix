// Dense decodes of a GGUF projection's blocks for tensor-core GEMMs (include/infernix/ops/ggml_linear.h):
// ggml_dequantize_bf16 (BF16 A16 route) and ggml_widen_e4m3 (E4M3 rows for the FP8 A8 route).

#include "infernix/ops/ggml_linear.h"

#include "core/ggml_format.h"
#include "ops/ggml/ggml_exact.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

constexpr int kThreads = 256;

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(std::string("ggml_dequantize_bf16: ") + message); }
}

// One thread per 8-element chunk: its exact value kappa * (d * sub * code - dmin * min_sub) rounded once to BF16,
// stored as 16 bytes at the chunk's (possibly permuted) column. IQ2_S, IQ3_XXS and IQ3_S (the bulk of an IQ3 GGUF:
// |sub * code| < 2^11 with an 11-bit binary16 mantissa) form d * sub * code in binary32; the other formats form it in
// binary64, which matched the oracle on random blocks where binary32 did not (Q6_K, IQ4_XS).
template <QType F>
__global__ void __launch_bounds__(kThreads)
    dequantize_kernel(const std::uint8_t* __restrict__ rows, std::int64_t chunks, int chunks_per_row, int row_bytes,
                      int k, std::uint16_t* __restrict__ out, const std::int32_t* __restrict__ perm, int perm_block) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    if (index >= chunks) { return; }
    const int row   = static_cast<int>(index / chunks_per_row);
    const int chunk = static_cast<int>(index - static_cast<std::int64_t>(row) * chunks_per_row);
    const ggml::Chunk c = ggml::row_chunk<F>(rows + static_cast<std::size_t>(row) * row_bytes, chunk);
    __align__(16) __nv_bfloat16 v[8];
    if constexpr (F != QType::GGML_IQ2_S && F != QType::GGML_IQ3_XXS && F != QType::GGML_IQ3_S) {
        const double d      = static_cast<double>(__half2float(__ushort_as_half(c.d))) * c.sub;
        const double offset = F == QType::GGML_Q4_K || F == QType::GGML_Q5_K
                                  ? static_cast<double>(__half2float(__ushort_as_half(c.dmin))) * c.min_sub
                                  : 0.0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = __double2bfloat16(ldexp(d * c.code[j] - offset, ggml::kappa_log2(F))); }
    } else {
        const float d = __half2float(__ushort_as_half(c.d));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = __float2bfloat16_rn(ggml::scale_pow2(__fmul_rn(d, static_cast<float>(c.sub * c.code[j])),
                                                        ggml::kappa_log2(F)));
        }
    }
    int column = chunk * 8;
    if (perm != nullptr) { column = perm[column / perm_block] * perm_block + column % perm_block; }
    *reinterpret_cast<uint4*>(out + static_cast<std::size_t>(row) * k + column) = *reinterpret_cast<const uint4*>(v);
}

template <QType F>
void launch(const GgmlSegment& s, int k, std::uint16_t* out, const std::int32_t* perm, int perm_block, cudaStream_t stream) {
    const int chunks_per_row = k / 8;
    const std::int64_t chunks = static_cast<std::int64_t>(s.rows) * chunks_per_row;
    const auto blocks = static_cast<unsigned>((chunks + kThreads - 1) / kThreads);
    dequantize_kernel<F><<<blocks, kThreads, 0, stream>>>(s.data, chunks, chunks_per_row,
                                                          static_cast<int>(ggml_row_bytes(F, k)), k,
                                                          out + static_cast<std::size_t>(s.out_offset) * k, perm,
                                                          perm_block);
}

// A chunk's eight values in binary32: exact for IQ2_S/IQ3_XXS/IQ3_S, the binary64 value rounded once otherwise.
template <QType F>
__device__ __forceinline__ void chunk_f32(const ggml::Chunk& c, float (&v)[8]) {
    if constexpr (F != QType::GGML_IQ2_S && F != QType::GGML_IQ3_XXS && F != QType::GGML_IQ3_S) {
        const double d      = static_cast<double>(__half2float(__ushort_as_half(c.d))) * c.sub;
        const double offset = F == QType::GGML_Q4_K || F == QType::GGML_Q5_K
                                  ? static_cast<double>(__half2float(__ushort_as_half(c.dmin))) * c.min_sub
                                  : 0.0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { v[j] = static_cast<float>(ldexp(d * c.code[j] - offset, ggml::kappa_log2(F))); }
    } else {
        const float d = __half2float(__ushort_as_half(c.d));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            v[j] = ggml::scale_pow2(__fmul_rn(d, static_cast<float>(c.sub * c.code[j])), ggml::kappa_log2(F));
        }
    }
}

// One CTA per row: the row's largest |value| m (binary32, of chunk_f32's values), the BF16 row multiplier
// s = bf16_ru(fl32(m / 448)) (1 for an all-zero row), then every value as e4m3_rn_satfinite(fl32(v * fl32(1 / s)))
// at its (possibly permuted) column: q4's bake contract (s * e4m3 code approximates v) for the FP8 A8 GEMM.
template <QType F>
__global__ void __launch_bounds__(kThreads)
    widen_kernel(const std::uint8_t* __restrict__ rows, int row_bytes, int k, std::uint8_t* __restrict__ codes,
                 __nv_bfloat16* __restrict__ row_scales, const std::int32_t* __restrict__ perm, int perm_block) {
    const int row = blockIdx.x;
    const std::uint8_t* src = rows + static_cast<std::size_t>(row) * row_bytes;
    const int chunks = k / 8;
    float largest = 0.0F;
    for (int i = threadIdx.x; i < chunks; i += kThreads) {
        float v[8];
        chunk_f32<F>(ggml::row_chunk<F>(src, i), v);
#pragma unroll
        for (int j = 0; j < 8; ++j) { largest = fmaxf(largest, fabsf(v[j])); }
    }
    __shared__ float warp_max[kThreads / 32];
    for (int offset = 16; offset > 0; offset >>= 1) { largest = fmaxf(largest, __shfl_xor_sync(0xffffffffU, largest, offset)); }
    if (threadIdx.x % 32 == 0) { warp_max[threadIdx.x / 32] = largest; }
    __syncthreads();
    largest = 0.0F;
#pragma unroll
    for (int w = 0; w < kThreads / 32; ++w) { largest = fmaxf(largest, warp_max[w]); }
    const __nv_bfloat16 scale = largest > 0.0F ? __float2bfloat16_ru(__fdiv_rn(largest, 448.0F)) : __float2bfloat16(1.0F);
    if (threadIdx.x == 0) { row_scales[row] = scale; }
    const float inverse = __frcp_rn(__bfloat162float(scale));
    for (int i = threadIdx.x; i < chunks; i += kThreads) {
        float v[8];
        chunk_f32<F>(ggml::row_chunk<F>(src, i), v);
        __align__(8) __nv_fp8x2_storage_t out[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[j] = __nv_cvt_float2_to_fp8x2(make_float2(__fmul_rn(v[2 * j], inverse), __fmul_rn(v[2 * j + 1], inverse)),
                                              __NV_SATFINITE, __NV_E4M3);
        }
        int column = i * 8;
        if (perm != nullptr) { column = perm[column / perm_block] * perm_block + column % perm_block; }
        *reinterpret_cast<uint2*>(codes + static_cast<std::size_t>(row) * k + column) = *reinterpret_cast<const uint2*>(out);
    }
}

template <QType F>
void launch_widen(const GgmlSegment& s, int k, std::uint8_t* codes, __nv_bfloat16* scales, const std::int32_t* perm,
                  int perm_block, cudaStream_t stream) {
    widen_kernel<F><<<s.rows, kThreads, 0, stream>>>(s.data, static_cast<int>(ggml_row_bytes(F, k)), k,
                                                     codes + static_cast<std::size_t>(s.out_offset) * k,
                                                     scales + s.out_offset, perm, perm_block);
}

// Calls f.template operator()<F>() for the segment's format.
template <class Function>
void with_format(QType q, Function&& f) {
    switch (q) {
    case QType::GGML_Q2_0: f.template operator()<QType::GGML_Q2_0>(); break;
    case QType::GGML_Q4_K: f.template operator()<QType::GGML_Q4_K>(); break;
    case QType::GGML_Q5_K: f.template operator()<QType::GGML_Q5_K>(); break;
    case QType::GGML_Q6_K: f.template operator()<QType::GGML_Q6_K>(); break;
    case QType::GGML_IQ2_S: f.template operator()<QType::GGML_IQ2_S>(); break;
    case QType::GGML_IQ3_XXS: f.template operator()<QType::GGML_IQ3_XXS>(); break;
    case QType::GGML_IQ3_S: f.template operator()<QType::GGML_IQ3_S>(); break;
    case QType::GGML_IQ4_XS: f.template operator()<QType::GGML_IQ4_XS>(); break;
    case QType::GGML_IQ4_NL: f.template operator()<QType::GGML_IQ4_NL>(); break;
    default: require(false, "unsupported ggml format");
    }
}

void check_segments(std::span<const GgmlSegment> segments, std::int32_t k, std::int32_t n, const std::int32_t* col_perm,
                    std::int32_t perm_block) {
    require(k > 0 && k % 8 == 0, "K must be a positive multiple of 8");
    require(col_perm == nullptr || (perm_block > 0 && perm_block % 8 == 0 && k % perm_block == 0),
            "a column permutation needs whole blocks of a multiple of 8 columns");
    for (const auto& s : segments) {
        require(s.layout == GgmlLayout::Rows && s.output == 0 && s.data != nullptr && s.rows > 0, "segments must be "
                "device-visible ggml_rows_v1 rows of one output");
        require(s.out_offset >= 0 && s.out_offset + s.rows <= n, "a segment's rows exceed out");
        require(k % static_cast<std::int32_t>(ggml_block_elems(s.format)) == 0, "K must be whole blocks");
    }
}

void check_launch() {
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml dense decode: ") + cudaGetErrorString(error)); }
}

} // namespace

void ggml_widen_e4m3(std::span<const GgmlSegment> segments, std::int32_t k, Tensor& codes, Tensor& row_scales,
                     const std::int32_t* col_perm, std::int32_t perm_block, cudaStream_t stream) {
    require(codes.data != nullptr && codes.dtype == DType::U8 && codes.is_contiguous() && codes.ne[0] == k,
            "codes must be contiguous U8 [K, N]");
    require(row_scales.data != nullptr && row_scales.dtype == DType::BF16 && row_scales.is_contiguous() &&
                row_scales.ne[0] == codes.ne[1],
            "row_scales must be contiguous BF16 [N]");
    require(reinterpret_cast<std::uintptr_t>(codes.data) % 8 == 0, "codes must be 8-byte aligned");
    check_segments(segments, k, codes.ne[1], col_perm, perm_block);
    auto* c = static_cast<std::uint8_t*>(codes.data);
    auto* s = static_cast<__nv_bfloat16*>(row_scales.data);
    for (const auto& segment : segments) {
        with_format(segment.format, [&]<QType F>() { launch_widen<F>(segment, k, c, s, col_perm, perm_block, stream); });
        check_launch();
    }
}

void ggml_dequantize_bf16(std::span<const GgmlSegment> segments, std::int32_t k, Tensor& out,
                          const std::int32_t* col_perm, std::int32_t perm_block, cudaStream_t stream) {
    require(out.data != nullptr && out.dtype == DType::BF16 && out.is_contiguous() && out.ne[0] == k,
            "out must be contiguous BF16 [K, N]");
    require(k > 0 && k % 8 == 0 && reinterpret_cast<std::uintptr_t>(out.data) % 16 == 0,
            "K must be a positive multiple of 8 and out 16-byte aligned");
    require(col_perm == nullptr || (perm_block > 0 && perm_block % 8 == 0 && k % perm_block == 0),
            "a column permutation needs whole blocks of a multiple of 8 columns");
    auto* o = static_cast<std::uint16_t*>(out.data);
    for (const auto& s : segments) {
        require(s.layout == GgmlLayout::Rows && s.output == 0 && s.data != nullptr && s.rows > 0, "segments must be "
                "device-visible ggml_rows_v1 rows of one output");
        require(s.out_offset >= 0 && s.out_offset + s.rows <= out.ne[1], "a segment's rows exceed out");
        require(k % static_cast<std::int32_t>(ggml_block_elems(s.format)) == 0, "K must be whole blocks");
        switch (s.format) {
        case QType::GGML_Q2_0: launch<QType::GGML_Q2_0>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_Q4_K: launch<QType::GGML_Q4_K>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_Q5_K: launch<QType::GGML_Q5_K>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_Q6_K: launch<QType::GGML_Q6_K>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_IQ2_S: launch<QType::GGML_IQ2_S>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_IQ3_XXS: launch<QType::GGML_IQ3_XXS>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_IQ3_S: launch<QType::GGML_IQ3_S>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_IQ4_XS: launch<QType::GGML_IQ4_XS>(s, k, o, col_perm, perm_block, stream); break;
        case QType::GGML_IQ4_NL: launch<QType::GGML_IQ4_NL>(s, k, o, col_perm, perm_block, stream); break;
        default: require(false, "unsupported ggml format");
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) { throw std::runtime_error(std::string("ggml_dequantize_bf16: ") + cudaGetErrorString(error)); }
    }
}

} // namespace infernix::ops
