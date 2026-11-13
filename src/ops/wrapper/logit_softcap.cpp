#include "infernix/ops/logit_softcap.h"

#include "ops/launcher/logit_softcap.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infernix::ops {

void logit_softcap(Tensor& x, float cap, cudaStream_t stream) {
    if (!(cap > 0.0f) || !std::isfinite(cap)) {
        throw std::invalid_argument("logit_softcap: cap must be positive and finite");
    }
    if (x.dtype != DType::BF16) { throw std::invalid_argument("logit_softcap: x must be BF16"); }
    if (!x.is_contiguous()) { throw std::invalid_argument("logit_softcap: x must be contiguous"); }
    std::int64_t n = 1;
    for (int i = 0; i < 4; ++i) {
        if (x.ne[i] < 0) { throw std::invalid_argument("logit_softcap: negative dimension"); }
        if (x.ne[i] == 0) { return; }
        if (n > std::numeric_limits<std::int64_t>::max() / x.ne[i]) {
            throw std::overflow_error("logit_softcap: tensor size overflows int64");
        }
        n *= x.ne[i];
    }
    if (x.data == nullptr) { throw std::invalid_argument("logit_softcap: x data must be non-null"); }
    detail::logit_softcap_launch(x, cap, stream);
}

} // namespace infernix::ops
