#include "infernix/ops/gemma4_kv_append.h"

#include "ops/launcher/gemma4_kv_append.h"

#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_kv_append: ") + name +
                                    " must be contiguous non-null BF16");
    }
}

} // namespace

void gemma4_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions, Tensor& k_pages,
                      Tensor& v_pages, const Tensor& block_table, cudaStream_t stream) {
    require_bf16(k, "k");
    require_bf16(v, "v");
    require_bf16(k_pages, "k_pages");
    require_bf16(v_pages, "v_pages");
    if (positions.dtype != DType::I32 || block_table.dtype != DType::I32 ||
        !positions.is_contiguous() || !block_table.is_contiguous() || positions.data == nullptr ||
        block_table.data == nullptr) {
        throw std::invalid_argument("gemma4_kv_append: positions and block_table must be I32");
    }
    const auto head_dim = k.ne[0];
    const auto kv_heads = k.ne[1];
    const auto tokens   = k.ne[2];
    if ((head_dim != 256 && head_dim != 512) || kv_heads <= 0 || tokens <= 0 || k.ne[3] != 1) {
        throw std::invalid_argument("gemma4_kv_append: k must be [256|512,Hkv,T]");
    }
    if (v.ne[0] != head_dim || v.ne[1] != kv_heads || v.ne[2] != tokens || v.ne[3] != 1) {
        throw std::invalid_argument("gemma4_kv_append: v shape must match k");
    }
    if (positions.ne[0] != tokens) {
        throw std::invalid_argument("gemma4_kv_append: positions must be [T]");
    }
    if (k_pages.ne[0] != head_dim || k_pages.ne[1] != 64 || k_pages.ne[2] != kv_heads ||
        v_pages.ne[0] != k_pages.ne[0] || v_pages.ne[1] != k_pages.ne[1] ||
        v_pages.ne[2] != k_pages.ne[2] || v_pages.ne[3] != k_pages.ne[3]) {
        throw std::invalid_argument("gemma4_kv_append: pages must be [D,64,Hkv,pages]");
    }
    detail::gemma4_kv_append_launch(k, v, positions, k_pages, v_pages, block_table, stream);
}

} // namespace infernix::ops
