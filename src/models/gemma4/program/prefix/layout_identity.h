#pragma once

// The prefix cache's cache/restore identity (P7 scope contract 2): everything that makes a
// cached state alias-free. It is computed from config + kv_layout.h only, lives in the state
// image's header, and is validated before any restore — a mismatch refuses instead of aliasing,
// so a different geometry (26B-A4B), KV dtype, storage model or QK mode can never reuse another
// layout's blocks or snapshots. Pure header, CPU-only; no device or model types.

#include "core/dtype.h"
#include "infernix/kv_format.h"
#include "infernix/ops/gemma4_cache.h"
#include "models/gemma4/program/kv_layout.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace infernix::models::gemma4::prefix {

// Bumped whenever the on-device/host packing of blocks or images changes shape or order; the
// identity refuses across versions.
inline constexpr std::uint32_t kLayoutPackingVersion = 1;

struct LayoutIdentity {
    // KV dtype/format and the scale-group shape (a row-256 FP8 group, one FP16 scale each; zero
    // groups for BF16).
    ops::Gemma4KvFormat kv_format = ops::Gemma4KvFormat::BFloat16;
    DType code_dtype = DType::BF16;
    std::uint32_t scale_group_columns = 0;                        // grp (kGemma4KvFp8Group when scaled)
    std::uint32_t scale_element_bytes = 0;                        // scale width (2 for FP16)
    // Storage model (K==V today; the value-only flag makes a future layout its own identity).
    KvStorageModel storage = KvStorageModel::KeyEqualsValue;
    // QK mode of the global layers (sliding layers always retain BF16 QK).
    ops::Gemma4QkMode qk_mode = ops::Gemma4QkMode::BFloat16;
    // Global layers: count, KV heads, head dimension. Sliding layers: count, KV heads, head dim.
    std::uint32_t global_layers = 0;                              // G
    std::uint32_t global_kv_heads = 0;                            // Hg
    std::uint32_t global_head_dim = 0;                            // dg
    std::uint32_t sliding_layers = 0;                             // S (one restore group per layer)
    std::uint32_t sliding_kv_heads = 0;                           // Hs
    std::uint32_t sliding_head_dim = 0;                           // ds
    // Sliding window in tokens, and the prefill chunk the state was produced with.
    std::uint32_t window_tokens = 0;                              // W
    std::uint32_t prefill_chunk = 0;                              // C
    // Packing version and the page the blocks are cut into.
    std::uint32_t packing_version = kLayoutPackingVersion;
    std::uint32_t page_tokens = 0;                                // T (kGemma4KvPageTokens)
    // The geometry's byte totals (kv_layout.h's prefix_cache_geometry).
    std::uint64_t block_bytes = 0;
    std::uint64_t image_bytes = 0;

    friend bool operator==(const LayoutIdentity&, const LayoutIdentity&) noexcept = default;

    // A stable 64-bit fingerprint for indexing and file headers: field order is fixed, every
    // field is a mixed integer, and the mix (splitmix64 finalizer) is byte-stable across
    // processes and endianness-neutral on little-endian hosts. Never persisted raw without the
    // full struct beside it.
    [[nodiscard]] std::uint64_t fingerprint() const noexcept {
        auto mix = [](std::uint64_t x) noexcept {
            x += 0x9E3779B97F4A7C15ULL;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
            return x ^ (x >> 31);
        };
        std::uint64_t h = 0x243F6A8885A308D3ULL;
        const auto absorb = [&](std::uint64_t value) noexcept {
            h = mix(h ^ mix(value));
        };
        absorb(static_cast<std::uint64_t>(kv_format));
        absorb(static_cast<std::uint64_t>(code_dtype));
        absorb(scale_group_columns);
        absorb(scale_element_bytes);
        absorb(static_cast<std::uint64_t>(storage));
        absorb(static_cast<std::uint64_t>(qk_mode));
        absorb(static_cast<std::uint64_t>(global_layers) |
               (static_cast<std::uint64_t>(global_kv_heads) << 32));
        absorb(static_cast<std::uint64_t>(global_head_dim) |
               (static_cast<std::uint64_t>(sliding_layers) << 32));
        absorb(static_cast<std::uint64_t>(sliding_kv_heads) |
               (static_cast<std::uint64_t>(sliding_head_dim) << 32));
        absorb(static_cast<std::uint64_t>(window_tokens) |
               (static_cast<std::uint64_t>(prefill_chunk) << 32));
        absorb(static_cast<std::uint64_t>(packing_version) |
               (static_cast<std::uint64_t>(page_tokens) << 32));
        absorb(block_bytes);
        absorb(image_bytes);
        return h;
    }

    // A short human-readable tag for refusal messages and logs.
    [[nodiscard]] std::string describe() const {
        return "kv=" + std::to_string(static_cast<int>(kv_format)) +
               (scale_group_columns != 0 ? "/row" + std::to_string(scale_group_columns) : "") +
               (storage == KvStorageModel::KeyEqualsValue ? "" : "/value-only") +
               "/qk=" + std::to_string(static_cast<int>(qk_mode)) +
               "/global=" + std::to_string(global_layers) + "x" + std::to_string(global_kv_heads) +
               "x" + std::to_string(global_head_dim) +
               "/sliding=" + std::to_string(sliding_layers) + "x" + std::to_string(sliding_kv_heads) +
               "x" + std::to_string(sliding_head_dim) +
               "/w=" + std::to_string(window_tokens) +
               "/chunk=" + std::to_string(prefill_chunk) +
               "/pack=" + std::to_string(packing_version);
    }
};

} // namespace infernix::models::gemma4::prefix
