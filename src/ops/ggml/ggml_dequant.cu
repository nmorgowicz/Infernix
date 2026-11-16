// Dense BF16 decode of a GGUF projection's blocks for a tensor-core GEMM (include/infernix/ops/ggml_linear.h,
// ggml_dequantize_bf16).

#include "infernix/ops/ggml_linear.h"

#include "core/ggml_format.h"
#include "ops/ggml/ggml_exact.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

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

} // namespace

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
