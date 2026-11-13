// infernix::ops — geglu wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "infernix/ops/geglu.h"

#include "ops/launcher/geglu.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace infernix::ops {
namespace {

bool same_shape(const Tensor& a, const Tensor& b) {
    for (int i = 0; i < 4; ++i) {
        if (a.ne[i] != b.ne[i]) { return false; }
    }
    return true;
}

} // namespace

void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream,
           std::uint8_t* fp8_codes, float* fp8_scales) {
    if (gate.dtype != DType::BF16 || up.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("geglu: gate/up/out must be BF16");
    }
    if (!same_shape(gate, up) || !same_shape(gate, out)) {
        throw std::invalid_argument("geglu: gate, up, and out must share a shape");
    }
    if (!out.is_contiguous()) { throw std::invalid_argument("geglu: out must be contiguous"); }
    const std::int64_t n = out.numel();
    if (n == 0) { return; }
    if (gate.data == nullptr || up.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("geglu: gate/up/out data must be non-null");
    }
    if ((fp8_codes == nullptr) != (fp8_scales == nullptr)) {
        throw std::invalid_argument("geglu: fp8 codes and scales are set together");
    }
    if (gate.is_contiguous() && up.is_contiguous()) {
        const auto* gate_bytes = static_cast<const char*>(gate.data);
        const auto* up_bytes   = static_cast<const char*>(up.data);
        const auto* out_bytes  = static_cast<const char*>(out.data);
        const auto bytes       = n * static_cast<std::int64_t>(sizeof(std::uint16_t));
        if ((out_bytes >= gate_bytes && out_bytes < gate_bytes + bytes) ||
            (gate_bytes >= out_bytes && gate_bytes < out_bytes + bytes) ||
            (out_bytes >= up_bytes && out_bytes < up_bytes + bytes) ||
            (up_bytes >= out_bytes && up_bytes < out_bytes + bytes)) {
            throw std::invalid_argument("geglu: out must not overlap gate or up");
        }
    }
    detail::geglu_launch(gate, up, out, stream, fp8_codes, fp8_scales);
}

} // namespace infernix::ops
