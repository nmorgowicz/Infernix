#pragma once

// Shared by the CPU-only Gemma4 prefix-cache tests (test_prefix_geometry.cpp, test_prefix_store.cpp):
// the failure counter, the synthetic descriptors' per-layer facts and the contract-1 byte formulas
// derived from them independently of prefix_cache_geometry (FP8 global pages carry one FP16 scale per
// 256 columns, K==V doubles the planes, sliding windows stay BF16 with a K part then a V part, the
// header and every part are 256-aligned, blocks and slabs 4096-aligned, the floor is image slabs
// plus two). No device or model types.

#include "models/gemma4/program/kv_layout.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace prefix_test {

inline int failures = 0;

inline void check(bool ok, const std::string& label, const std::string& detail = {}) {
    if (ok) { return; }
    std::cerr << "FAIL " << label << ": " << detail << '\n';
    ++failures;
}

// One descriptor's layer facts: the global layers first, then the sliding ones.
inline std::vector<infernix::models::gemma4::KvLayerFact>
layer_facts(std::size_t global_count, std::size_t global_heads, std::size_t sliding_count,
            std::size_t sliding_heads) {
    std::vector<infernix::models::gemma4::KvLayerFact> out;
    for (std::size_t i = 0; i < global_count; ++i) {
        out.push_back({true, 512, static_cast<int>(global_heads)});
    }
    for (std::size_t i = 0; i < sliding_count; ++i) {
        out.push_back({false, 256, static_cast<int>(sliding_heads)});
    }
    return out;
}

struct GeometryExpectation {
    std::size_t global_bytes_per_token = 0;
    std::size_t sliding_bytes_per_token = 0;
    std::size_t block_bytes = 0;
    std::size_t image_bytes = 0;
    std::uint32_t image_slabs = 0;
    std::size_t host_floor_bytes = 0;
};

inline GeometryExpectation expected_geometry(std::size_t global_layers, std::size_t global_heads,
                                             std::size_t sliding_layers, std::size_t sliding_heads,
                                             std::size_t window) {
    constexpr std::size_t page  = 64;   // ops::kGemma4KvPageTokens
    constexpr std::size_t group = 256;  // ops::kGemma4KvFp8Group
    const std::size_t window_tokens = window - 1;
    const std::size_t global_per_layer =
        2U * (page * global_heads * 512U + page * global_heads * (512U / group) * 2U);
    const std::size_t sliding_per_layer = 2U * 2U * page * sliding_heads * 256U;
    GeometryExpectation out;
    out.global_bytes_per_token  = global_layers * global_per_layer / page;
    out.sliding_bytes_per_token = sliding_layers * sliding_per_layer / page;
    out.block_bytes = infernix::models::gemma4::align_bytes(out.global_bytes_per_token * page, 4096);
    const std::size_t part = infernix::models::gemma4::align_bytes(
        window_tokens * sliding_per_layer / (2U * page), 256);
    out.image_bytes  = 256U + sliding_layers * 2U * part;
    out.image_slabs = static_cast<std::uint32_t>(
        (out.image_bytes + out.block_bytes - 1U) / out.block_bytes);
    out.host_floor_bytes = (static_cast<std::size_t>(out.image_slabs) + 2U) * out.block_bytes;
    return out;
}

} // namespace prefix_test
