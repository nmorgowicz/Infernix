#pragma once

#include "core/dtype.h"
#include "infernix/kv_format.h"
#include "infernix/ops/gemma4_cache.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace infernix::models::gemma4 {

// Gemma stores global-layer K/V as BF16 or FP8 E4M3 with one FP16 scale per 256 columns. Every
// other KvCacheStorage is refused here, not mapped.
[[nodiscard]] inline ops::Gemma4KvFormat gemma4_kv_format(KvCacheStorage requested) {
    switch (requested) {
    case KvCacheStorage::BFloat16: return ops::Gemma4KvFormat::BFloat16;
    case KvCacheStorage::Fp8E4M3Row256: return ops::Gemma4KvFormat::Fp8E4m3Row256;
    default: throw std::invalid_argument("Gemma4 KV cache supports only --kv-dtype bf16 and fp8");
    }
}

struct KvLayerLayout {
    ops::Gemma4KvFormat format = ops::Gemma4KvFormat::BFloat16;
    DType code_dtype = DType::BF16;
    int scale_groups = 0;
    std::size_t code_page_bytes = 0;  // One K or V plane.
    std::size_t scale_page_bytes = 0; // One K or V plane.

    [[nodiscard]] std::size_t page_bytes() const noexcept {
        return 2U * (code_page_bytes + scale_page_bytes);
    }
};

// Sliding layers always retain BF16, independently of the requested global format.
[[nodiscard]] inline KvLayerLayout kv_layer_layout(KvCacheStorage requested, bool global,
                                                    int head_dim, int kv_heads) {
    const auto requested_format = gemma4_kv_format(requested);
    if (head_dim <= 0 || kv_heads <= 0) {
        throw std::invalid_argument("Gemma4 KV cache geometry is invalid");
    }
    KvLayerLayout layout;
    layout.format = global ? requested_format : ops::Gemma4KvFormat::BFloat16;
    const auto rows = static_cast<std::size_t>(ops::kGemma4KvPageTokens) * kv_heads;
    if (layout.format == ops::Gemma4KvFormat::Fp8E4m3Row256) {
        if (head_dim != 512) {
            throw std::invalid_argument("Gemma4 fp8 global KV requires head dimension 512");
        }
        layout.code_dtype = DType::U8;
        layout.scale_groups = head_dim / ops::kGemma4KvFp8Group;
        layout.code_page_bytes = rows * head_dim;
        layout.scale_page_bytes = rows * layout.scale_groups * sizeof(std::uint16_t);
    } else {
        layout.code_page_bytes = rows * head_dim * sizeof(std::uint16_t);
    }
    return layout;
}

// The callback plans each chunk independently, including its sliding rings, workspace,
// I/O and graph allowance, and returns the reservation for the full page-rounded context.
template <class Reservation>
[[nodiscard]] std::uint32_t auto_prefill_chunk(std::uint32_t max_context,
                                               std::size_t available_bytes,
                                               std::size_t headroom_bytes,
                                               Reservation&& reservation) {
    constexpr std::uint32_t page = ops::kGemma4KvPageTokens;
    if (max_context == 0 || max_context > std::numeric_limits<std::uint32_t>::max() - (page - 1)) {
        throw std::invalid_argument("Gemma4 auto prefill requires a valid max_context");
    }
    const auto capacity = ((max_context + page - 1) / page) * page;
    if (headroom_bytes <= available_bytes) {
        const auto budget = available_bytes - headroom_bytes;
        for (const std::uint32_t chunk : {2048U, 1024U, 512U, 256U, 128U}) {
            if (reservation(chunk, capacity) <= budget) { return chunk; }
        }
    }
    throw std::runtime_error(
        "Gemma4 --prefill-chunk auto: no candidate fits the complete page-rounded "
        "--max-context and VRAM headroom; reduce context or free device memory");
}

} // namespace infernix::models::gemma4
