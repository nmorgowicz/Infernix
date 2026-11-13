#include "infernix/ops/residual_scale.h"

#include "ops/launcher/residual_scale.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops {

void residual_scale(Tensor& x, float scale, cudaStream_t stream) {
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("residual_scale: scale must be finite");
    }
    if (x.dtype != DType::BF16) { throw std::invalid_argument("residual_scale: x must be BF16"); }
    if (!x.is_contiguous()) { throw std::invalid_argument("residual_scale: x must be contiguous"); }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("residual_scale: negative dimension"); }
        if (x.ne[i] == 0) { return; }
        if (n > std::numeric_limits<std::int64_t>::max() / x.ne[i]) {
            throw std::overflow_error("residual_scale: tensor size overflows int64");
        }
        n *= x.ne[i];
    }
    if (x.data == nullptr) { throw std::invalid_argument("residual_scale: x data must be non-null"); }
    detail::residual_scale_launch(x, scale, stream);
}

} // namespace infernix::ops
