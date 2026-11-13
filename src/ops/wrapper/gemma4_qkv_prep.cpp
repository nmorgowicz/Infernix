#include "infernix/ops/gemma4_qkv_prep.h"

#include "ops/launcher/gemma4_qkv_prep.h"
#include "ops/wrapper/gemma4_cache_validation.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_qkv_prep: ") + name +
                                    " must be contiguous non-null BF16");
    }
}

void require_i32(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::I32 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_qkv_prep: ") + name +
                                    " must be contiguous non-null I32");
    }
}

} // namespace

void gemma4_qkv_prep(const Tensor& qkv, const Tensor& q_norm, const Tensor& k_norm,
                     const Tensor& positions, int rotary_dim, float theta, float eps, bool v_from_k,
                     Tensor& q_out, const Gemma4CacheView& cache, cudaStream_t stream) {
    require_bf16(qkv, "qkv");
    require_bf16(q_norm, "q_norm");
    require_bf16(k_norm, "k_norm");
    require_bf16(q_out, "q_out");
    detail::validate_gemma4_cache(cache, q_out.ne[0], v_from_k, "gemma4_qkv_prep");
    const Tensor& k_pages = *cache.k_pages;
    const Tensor& block_table = *cache.block_table;
    require_i32(positions, "positions");
    require_i32(block_table, "block_table");
    const int head_dim = q_out.ne[0];
    const int q_heads  = q_out.ne[1];
    const int tokens   = q_out.ne[2];
    const int kv_heads = k_pages.ne[2];
    if ((head_dim != 256 && head_dim != 512) || q_heads <= 0 || tokens <= 0 || q_out.ne[3] != 1) {
        throw std::invalid_argument("gemma4_qkv_prep: q_out must be [256|512,Hq,T]");
    }
    const int rows = (q_heads + (v_from_k ? 1 : 2) * kv_heads) * head_dim;
    if (qkv.ne[0] != rows || qkv.ne[1] != tokens || qkv.ne[2] != 1) {
        throw std::invalid_argument("gemma4_qkv_prep: qkv must be [" + std::to_string(rows) +
                                    ",T]");
    }
    if (q_norm.numel() != head_dim || k_norm.numel() != head_dim) {
        throw std::invalid_argument("gemma4_qkv_prep: q_norm and k_norm must be [D]");
    }
    if (positions.ne[0] != tokens || positions.numel() != tokens) {
        throw std::invalid_argument("gemma4_qkv_prep: positions must be [T]");
    }
    if (!(std::isfinite(theta) && theta > 0.0f) || !(std::isfinite(eps) && eps > 0.0f)) {
        throw std::invalid_argument("gemma4_qkv_prep: theta and eps must be positive and finite");
    }
    detail::gemma4_qkv_prep_launch(qkv, q_norm, k_norm, positions, rotary_dim, theta, eps, v_from_k,
                                   q_out, cache, stream);
}

} // namespace infernix::ops
