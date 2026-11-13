#pragma once

#include "core/dtype.h"
#include "infernix/kv_format.h"
#include "infernix/ops/gemma4_cache.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <span>
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

// How one layer's page bundle stores K relative to V. P7 implements KeyEqualsValue only;
// ValueOnly (addendum E3's derived-key layout) is carried in the descriptor and the identity so a
// future layout is a different format, never a smaller alias of a K==V cache.
enum class KvStorageModel : std::uint8_t { KeyEqualsValue, ValueOnly };

// One plane of a page bundle: its element type and per-page byte size.
struct KvPlane {
    DType dtype = DType::BF16;
    std::size_t page_bytes = 0;
};

struct KvLayerLayout {
    ops::Gemma4KvFormat format = ops::Gemma4KvFormat::BFloat16;
    DType code_dtype = DType::BF16;
    int scale_groups = 0;
    std::size_t code_page_bytes = 0;  // One K or V plane.
    std::size_t scale_page_bytes = 0; // One K or V plane.
    KvStorageModel storage = KvStorageModel::KeyEqualsValue;

    // The page bundle as a plane-descriptor list: K's planes, then V's (codes, then their scales).
    // FP8-row256: K codes U8, K scales FP16, V codes U8, V scales FP16 (four planes). BF16: two
    // BF16 planes. A value-only bundle keeps only the V planes (documented; refused at runtime).
    [[nodiscard]] std::size_t plane_count() const noexcept {
        const auto sides = storage == KvStorageModel::KeyEqualsValue ? 2U : 1U;
        return sides * (scale_page_bytes != 0 ? 2U : 1U);
    }
    [[nodiscard]] std::array<KvPlane, 4> planes() const noexcept {
        std::array<KvPlane, 4> out{};
        const auto scaled = scale_page_bytes != 0;
        const auto sides = storage == KvStorageModel::KeyEqualsValue ? 2U : 1U;
        for (std::size_t side = 0; side < sides; ++side) {
            out[side * (scaled ? 2U : 1U)] = {code_dtype, code_page_bytes};
            if (scaled) { out[side * 2U + 1U] = {DType::FP16, scale_page_bytes}; }
        }
        return out;
    }

    [[nodiscard]] std::size_t page_bytes() const noexcept {
        // KeyEqualsValue stores two code planes (+ two scale planes when scaled). ValueOnly keeps
        // only the V side's planes — half these bytes — and is its own identity.
        const auto sides = storage == KvStorageModel::KeyEqualsValue ? 2U : 1U;
        return sides * (code_page_bytes + scale_page_bytes);
    }
};

// Refuses a layout P7 does not implement, with the message a value-only descriptor earns.
[[nodiscard]] inline const KvLayerLayout& require_implemented_storage(const KvLayerLayout& layout) {
    if (layout.storage != KvStorageModel::KeyEqualsValue) {
        throw std::invalid_argument(
            "Gemma4 value-only global KV storage is not implemented; its layout identity is a "
            "different format, never a K==V alias");
    }
    return layout;
}

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

// Per-layer KV facts a geometry summary needs: the caller derives them from config
// (layer_types, per-kind head_dim and KV heads); no model type leaks into this header.
struct KvLayerFact {
    bool global = false;
    int head_dim = 0;
    int kv_heads = 0;
};

// The prefix cache's byte geometry, computed from config + the per-layer layouts (plan §10.3 P7:
// nothing is a model-size literal). HPC-min packing: the image is exactly the ring window's
// (window-1) tokens, one K and one V part per sliding layer, each 256-aligned behind a 256-byte
// header. The Host tier's floor is image_slabs + 2 slabs (one live snapshot, one staging, one
// slab of headroom); slabs are 4 KiB aligned, one block per slab.
struct PrefixCacheGeometry {
    std::size_t global_bytes_per_token = 0;
    std::size_t sliding_bytes_per_token = 0;
    std::size_t block_bytes = 0;      // One global block: page tokens × global bytes/token, 4 KiB aligned.
    std::size_t slab_bytes = 0;       // The Host tier's slab: block_bytes.
    std::size_t image_bytes = 0;      // One state snapshot: header + per-layer K/V window parts.
    std::uint32_t image_slabs = 0;
    std::size_t host_floor_bytes = 0; // (image_slabs + 2) × slab_bytes; smaller Host budgets refuse.
};

[[nodiscard]] inline std::size_t align_bytes(std::size_t value, std::size_t boundary) noexcept {
    return (value + boundary - 1U) / boundary * boundary;
}

[[nodiscard]] inline PrefixCacheGeometry prefix_cache_geometry(std::span<const KvLayerFact> layers,
                                                               std::uint32_t window,
                                                               KvCacheStorage global_format) {
    constexpr std::size_t kHeaderBytes = 256;
    constexpr std::size_t kPartAlignment = 256;
    constexpr std::size_t kSlabAlignment = 4096;
    const auto page_tokens = static_cast<std::size_t>(ops::kGemma4KvPageTokens);
    PrefixCacheGeometry out;
    const std::size_t window_tokens = window > 0 ? static_cast<std::size_t>(window - 1U) : 0U;
    out.image_bytes = kHeaderBytes;
    for (const auto& layer : layers) {
        const auto layout = require_implemented_storage(kv_layer_layout(
            layer.global ? global_format : KvCacheStorage::BFloat16, layer.global,
            layer.head_dim, layer.kv_heads));
        const auto per_token = layout.page_bytes() / page_tokens;
        if (layer.global) {
            out.global_bytes_per_token += per_token;
            continue;
        }
        out.sliding_bytes_per_token += per_token;
        // The image carries each sliding layer's window as its own K part then V part.
        const auto code_share  = layout.code_page_bytes / page_tokens;
        const auto scale_share = layout.scale_page_bytes / page_tokens;
        // K==V storage keeps both sides (value-only is refused above), so every sliding layer
        // contributes a K part then a V part, each its own 256-aligned codes (+ scales) run.
        for (int side = 0; side < 2; ++side) {
            out.image_bytes += align_bytes(window_tokens * code_share, kPartAlignment);
            if (layout.scale_page_bytes != 0) {
                out.image_bytes += align_bytes(window_tokens * scale_share, kPartAlignment);
            }
        }
    }
    out.block_bytes = align_bytes(out.global_bytes_per_token * page_tokens, kSlabAlignment);
    out.slab_bytes = out.block_bytes;
    out.image_slabs = static_cast<std::uint32_t>(
        (out.image_bytes + out.slab_bytes - 1U) / out.slab_bytes);
    out.host_floor_bytes = (static_cast<std::size_t>(out.image_slabs) + 2U) * out.slab_bytes;
    return out;
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
