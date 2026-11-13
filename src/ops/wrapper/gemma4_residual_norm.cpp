#include "infernix/ops/gemma4_residual_norm.h"

#include "ops/launcher/gemma4_residual_norm.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

void require_bf16(const Tensor& tensor, const char* name) {
    if (tensor.dtype != DType::BF16 || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("gemma4_residual_norm: ") + name +
                                    " must be contiguous non-null BF16");
    }
}

bool same_shape(const Tensor& a, const Tensor& b) {
    return a.ne[0] == b.ne[0] && a.ne[1] == b.ne[1] && a.ne[2] == b.ne[2] && a.ne[3] == b.ne[3];
}

} // namespace

void gemma4_residual_norm(const Tensor& projected, const Tensor& post_norm, float scale,
                          const Tensor* next_norm, float eps, Tensor& x, Tensor* next_out,
                          cudaStream_t stream) {
    require_bf16(projected, "projected");
    require_bf16(post_norm, "post_norm");
    require_bf16(x, "x");
    const int hidden    = projected.ne[0];
    constexpr int kSpan = detail::kGemma4ResidualNormBlock;
    if (hidden <= 0 || hidden % kSpan != 0 ||
        hidden > kSpan * detail::kGemma4ResidualNormMaxPerThread || projected.ne[1] <= 0 ||
        projected.ne[2] != 1 || !same_shape(projected, x)) {
        throw std::invalid_argument("gemma4_residual_norm: projected and x must be [H,T] with H a "
                                    "multiple of 256 up to 6144");
    }
    if (projected.data == x.data) {
        throw std::invalid_argument("gemma4_residual_norm: projected must not alias x");
    }
    if (post_norm.numel() != hidden) {
        throw std::invalid_argument("gemma4_residual_norm: post_norm must be [H]");
    }
    if ((next_norm == nullptr) != (next_out == nullptr)) {
        throw std::invalid_argument("gemma4_residual_norm: next_norm and next_out go together");
    }
    if (next_norm != nullptr) {
        require_bf16(*next_norm, "next_norm");
        require_bf16(*next_out, "next_out");
        if (next_norm->numel() != hidden || !same_shape(*next_out, x) || next_out->data == x.data ||
            next_out->data == projected.data) {
            throw std::invalid_argument(
                "gemma4_residual_norm: next_norm must be [H] and next_out a separate [H,T]");
        }
    }
    if (!std::isfinite(scale) || !(std::isfinite(eps) && eps > 0.0f)) {
        throw std::invalid_argument("gemma4_residual_norm: scale and eps must be finite");
    }
    detail::gemma4_residual_norm_launch(projected, post_norm, scale, next_norm, eps, x, next_out,
                                        stream);
}

} // namespace infernix::ops
