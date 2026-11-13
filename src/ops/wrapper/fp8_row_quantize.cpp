#include "infernix/ops/fp8_row_quantize.h"

#include "core/device.h"
#include "ops/launcher/fp8_row_quantize.h"
#include "ops/linear/fp8/fp8_format.h"

#include <cuda_bf16.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace infernix::ops {
namespace {

void bind_fp8_row_weight(Weight& weight, void* payload, std::uint64_t bytes, std::int32_t n,
                         std::int32_t k) {
    const auto geometry = detail::fp8_row_geometry(n, k, "fp8_row_quantize");
    auto* bytes_ptr     = static_cast<std::byte*>(payload);
    weight              = Weight{};
    weight.qtype        = QType::FP8_E4M3FN_ROW_BF16;
    weight.layout       = QuantLayout::RowScale;
    weight.scale_dtype  = DType::BF16;
    weight.payload      = payload;
    weight.payload_bytes = bytes;
    weight.qdata        = payload;
    weight.scales       = bytes_ptr + geometry.scale_plane_offset;
    weight.group_size   = static_cast<std::uint32_t>(k);
    weight.group        = k;
    weight.ndim         = 2;
    weight.n = weight.shape[0] = weight.padded_shape[0] = n;
    weight.k = weight.shape[1] = weight.padded_shape[1] = k;
    weight.shape[2] = weight.shape[3] = weight.padded_shape[2] = weight.padded_shape[3] = 1;
    weight.scale_ne[0] = n;
    weight.scale_ne[1] = weight.scale_ne[2] = weight.scale_ne[3] = 1;
    weight.scale_nb[0] = 2;
    weight.scale_nb[1] = weight.scale_nb[2] = weight.scale_nb[3] = static_cast<std::int64_t>(n) * 2;
    (void)detail::validate_fp8_weight(weight, "fp8_row_quantize");
}

} // namespace

void quantize_fp8_row_bf16(Weight& weight, DeviceBuffer& storage, cudaStream_t stream) {
    if (weight.qtype == QType::FP8_E4M3FN_ROW_BF16) { return; }
    if (weight.qtype != QType::BF16) {
        throw std::invalid_argument("fp8_row_quantize: weight must be BF16 or FP8_E4M3FN_ROW_BF16");
    }
    if (weight.n <= 0 || weight.k <= 0 || weight.qdata == nullptr) {
        throw std::invalid_argument("fp8_row_quantize: BF16 matrix must be resident and positive");
    }

    const auto geometry = detail::fp8_row_geometry(weight.n, weight.k, "fp8_row_quantize");
    DeviceBuffer payload(static_cast<std::size_t>(geometry.required_payload_bytes));
    auto* codes  = static_cast<std::uint8_t*>(payload.p);
    auto* scales = reinterpret_cast<__nv_bfloat16*>(
        static_cast<std::byte*>(payload.p) + geometry.scale_plane_offset);
    detail::fp8_row_quantize_launch(static_cast<const __nv_bfloat16*>(weight.qdata), codes, scales,
                                    weight.n, weight.k, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    storage = std::move(payload);
    bind_fp8_row_weight(weight, storage.p, storage.bytes, weight.n, weight.k);
}

} // namespace infernix::ops
