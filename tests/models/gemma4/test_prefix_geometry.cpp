// P7 C.0: prefix-cache geometry and cache/restore identity from pure headers. Two in-process
// descriptors — 31B-like (10 global x 4 KV heads x D512 FP8; 50 sliding x 16 x D256; W=1024) and
// 26B-A4B-like (5 x 2 x D512 FP8; 50 x 8 x D256; W=1024, C=512) — assert the contract-1 byte
// formulas (block, image, slab count, host floor) against independent arithmetic, the identity
// fingerprint's stability and per-field sensitivity, and the value-only storage flag's carry and
// refusal. CPU only: no GPU, no artifact, no model types.

#include "models/gemma4/program/kv_layout.h"
#include "models/gemma4/program/prefix/layout_identity.h"
#include "prefix_geometry_support.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {


namespace g4  = infernix::models::gemma4;
namespace g4p = infernix::models::gemma4::prefix;
namespace ops = infernix::ops;

using namespace prefix_test;

void assert_geometry(const char* label, std::size_t global_layers, std::size_t global_heads,
                     std::size_t sliding_layers, std::size_t sliding_heads, std::size_t window) {
    const auto facts = layer_facts(global_layers, global_heads, sliding_layers, sliding_heads);
    const g4::PrefixCacheGeometry g =
        g4::prefix_cache_geometry(facts, static_cast<std::uint32_t>(window),
                                  infernix::KvCacheStorage::Fp8E4M3Row256);
    const GeometryExpectation want =
        expected_geometry(global_layers, global_heads, sliding_layers, sliding_heads, window);
    const auto sized = [](std::size_t value) { return std::to_string(value); };
    check(g.global_bytes_per_token == want.global_bytes_per_token,
          std::string(label) + " global bytes/token", sized(g.global_bytes_per_token));
    check(g.sliding_bytes_per_token == want.sliding_bytes_per_token,
          std::string(label) + " sliding bytes/token", sized(g.sliding_bytes_per_token));
    check(g.block_bytes == want.block_bytes, std::string(label) + " block bytes",
          sized(g.block_bytes));
    check(g.slab_bytes == g.block_bytes, std::string(label) + " slab equals block",
          sized(g.slab_bytes));
    check(g.image_bytes == want.image_bytes, std::string(label) + " image bytes",
          sized(g.image_bytes));
    check(g.image_slabs == want.image_slabs, std::string(label) + " image slabs",
          sized(g.image_slabs));
    check(g.host_floor_bytes == want.host_floor_bytes, std::string(label) + " host floor bytes",
          sized(g.host_floor_bytes));
    check(g.block_bytes % 4096 == 0, std::string(label) + " block is 4096-aligned");
    check(g.image_bytes >= 256U, std::string(label) + " image carries its header");
    check(g.host_floor_bytes == (static_cast<std::size_t>(g.image_slabs) + 2U) * g.slab_bytes,
          std::string(label) + " floor is image slabs plus two");
}

g4p::LayoutIdentity identity(std::uint32_t global_layers, std::uint32_t global_heads,
                             std::uint32_t sliding_heads, std::uint32_t prefill_chunk,
                             std::uint64_t block_bytes, std::uint64_t image_bytes) {
    g4p::LayoutIdentity out;
    out.kv_format          = ops::Gemma4KvFormat::Fp8E4m3Row256;
    out.code_dtype         = infernix::DType::U8;
    out.scale_group_columns = 256;  // grp: one FP16 scale per 256 columns.
    out.scale_element_bytes = 2;
    out.storage            = g4::KvStorageModel::KeyEqualsValue;
    out.qk_mode            = ops::Gemma4QkMode::BFloat16;
    out.global_layers      = global_layers;
    out.global_kv_heads    = global_heads;
    out.global_head_dim    = 512;
    out.sliding_layers     = 50;
    out.sliding_kv_heads   = sliding_heads;
    out.sliding_head_dim   = 256;
    out.window_tokens      = 1024;
    out.prefill_chunk      = prefill_chunk;
    out.packing_version    = g4p::kLayoutPackingVersion;
    out.page_tokens        = 64;
    out.block_bytes        = block_bytes;
    out.image_bytes        = image_bytes;
    return out;
}

} // namespace

int main() {
    // Byte formulas hold for both geometry descriptors.
    assert_geometry("31B-like", 10, 4, 50, 16, 1024);
    assert_geometry("26B-A4B-like", 5, 2, 50, 8, 1024);

    // The scope's worked examples pin the exact totals, so a formula drift cannot cancel between
    // the derivation above and the implementation.
    {
        const auto g31 = g4::prefix_cache_geometry(layer_facts(10, 4, 50, 16), 1024,
                                                   infernix::KvCacheStorage::Fp8E4M3Row256);
        check(g31.global_bytes_per_token == 41280, "31B-like global 41,280 B/token",
              std::to_string(g31.global_bytes_per_token));
        check(g31.sliding_bytes_per_token == 819200, "31B-like sliding 819,200 B/token",
              std::to_string(g31.sliding_bytes_per_token));
        check(g31.block_bytes == 2641920, "31B-like block 2,641,920 B", std::to_string(g31.block_bytes));
        check(g31.image_bytes == 838041856, "31B-like image 838,041,856 B",
              std::to_string(g31.image_bytes));
        check(g31.image_slabs == 318, "31B-like image needs 318 slabs", std::to_string(g31.image_slabs));
        check(g31.host_floor_bytes == 845414400, "31B-like floor 320 slabs, 845,414,400 B",
              std::to_string(g31.host_floor_bytes));
    }
    {
        const auto g26 = g4::prefix_cache_geometry(layer_facts(5, 2, 50, 8), 1024,
                                                   infernix::KvCacheStorage::Fp8E4M3Row256);
        check(g26.global_bytes_per_token == 10320, "26B-like global 10,320 B/token",
              std::to_string(g26.global_bytes_per_token));
        check(g26.sliding_bytes_per_token == 409600, "26B-like sliding 409,600 B/token",
              std::to_string(g26.sliding_bytes_per_token));
        check(g26.block_bytes == 663552, "26B-like block 663,552 B", std::to_string(g26.block_bytes));
        check(g26.image_bytes == 419021056, "26B-like image 419,021,056 B",
              std::to_string(g26.image_bytes));
        check(g26.image_slabs == 632, "26B-like image needs 632 slabs", std::to_string(g26.image_slabs));
        check(g26.host_floor_bytes == 420691968, "26B-like floor 634 slabs, 420,691,968 B",
              std::to_string(g26.host_floor_bytes));
    }

    // The identity's fingerprint is stable for a given struct, field-sensitive to every single
    // change, and distinguishes the two geometries from each other.
    const g4p::LayoutIdentity gemma31 = identity(10, 4, 16, 128, 2641920, 838041856);
    const g4p::LayoutIdentity gemma26 = identity(5, 2, 8, 512, 663552, 419021056);
    const g4p::LayoutIdentity copy31  = gemma31;
    check(copy31 == gemma31 && copy31.fingerprint() == gemma31.fingerprint(),
          "the identity round-trips bit-identically");
    check(gemma31.fingerprint() != gemma26.fingerprint(),
          "the two geometries have different fingerprints");
    const auto differs = [&gemma31](auto mutate, const char* field) {
        g4p::LayoutIdentity changed = gemma31;
        mutate(changed);
        check(changed.fingerprint() != gemma31.fingerprint(),
              std::string("fingerprint rejects a ") + field + " change");
    };
    differs([](g4p::LayoutIdentity& i) { i.kv_format = ops::Gemma4KvFormat::BFloat16; }, "kv_format");
    differs([](g4p::LayoutIdentity& i) { i.code_dtype = infernix::DType::BF16; }, "code_dtype");
    differs([](g4p::LayoutIdentity& i) { i.scale_group_columns = 128; }, "scale_group_columns");
    differs([](g4p::LayoutIdentity& i) { i.scale_element_bytes = 1; }, "scale_element_bytes");
    differs([](g4p::LayoutIdentity& i) { i.storage = g4::KvStorageModel::ValueOnly; }, "storage");
    differs([](g4p::LayoutIdentity& i) { i.qk_mode = ops::Gemma4QkMode::Fp8; }, "qk_mode");
    differs([](g4p::LayoutIdentity& i) { ++i.global_layers; }, "global_layers");
    differs([](g4p::LayoutIdentity& i) { ++i.global_kv_heads; }, "global_kv_heads");
    differs([](g4p::LayoutIdentity& i) { ++i.global_head_dim; }, "global_head_dim");
    differs([](g4p::LayoutIdentity& i) { ++i.sliding_layers; }, "sliding_layers");
    differs([](g4p::LayoutIdentity& i) { ++i.sliding_kv_heads; }, "sliding_kv_heads");
    differs([](g4p::LayoutIdentity& i) { ++i.sliding_head_dim; }, "sliding_head_dim");
    differs([](g4p::LayoutIdentity& i) { ++i.window_tokens; }, "window_tokens");
    differs([](g4p::LayoutIdentity& i) { ++i.prefill_chunk; }, "prefill_chunk");
    differs([](g4p::LayoutIdentity& i) { ++i.packing_version; }, "packing_version");
    differs([](g4p::LayoutIdentity& i) { ++i.page_tokens; }, "page_tokens");
    differs([](g4p::LayoutIdentity& i) { ++i.block_bytes; }, "block_bytes");
    differs([](g4p::LayoutIdentity& i) { ++i.image_bytes; }, "image_bytes");

    // The value-only flag rides the layout and the identity (a different format, never a smaller
    // alias) and the forward path refuses it while P7 implements K==V only.
    {
        const g4::KvLayerLayout layout =
            g4::kv_layer_layout(infernix::KvCacheStorage::Fp8E4M3Row256, true, 512, 2);
        check(layout.storage == g4::KvStorageModel::KeyEqualsValue && layout.plane_count() == 4 &&
                  layout.page_bytes() == 132096,
              "the K==V FP8 bundle carries four planes", std::to_string(layout.page_bytes()));
        const g4p::LayoutIdentity value_only = [&gemma26] {
            g4p::LayoutIdentity out = gemma26;
            out.storage = g4::KvStorageModel::ValueOnly;
            return out;
        }();
        check(value_only.fingerprint() != gemma26.fingerprint() &&
                  value_only.describe().find("/value-only") != std::string::npos,
              "the value-only flag is its own identity");
        bool refused   = false;
        std::string refusal;
        try {
            g4::KvLayerLayout flagged = layout;
            flagged.storage           = g4::KvStorageModel::ValueOnly;
            check(flagged.plane_count() == 2 && flagged.page_bytes() == 66048,
                  "the value-only bundle halves the planes", std::to_string(flagged.page_bytes()));
            (void)g4::require_implemented_storage(flagged);
        } catch (const std::invalid_argument& error) {
            refused  = true;
            refusal = error.what();
        }
        check(refused, "value-only storage refuses via require_implemented_storage", refusal);
        bool accepted = true;
        try { (void)g4::require_implemented_storage(layout); }
        catch (const std::invalid_argument&) { accepted = false; }
        check(accepted, "K==V storage passes require_implemented_storage");
    }

    // describe() carries the fields a refusal message needs to tell the layouts apart.
    const std::string d31 = gemma31.describe();
    const std::string d26 = gemma26.describe();
    check(d31 != d26, "describe() distinguishes the geometries");
    check(d31.find("row256") != std::string::npos &&
              d31.find("global=10x4x512") != std::string::npos &&
              d31.find("sliding=50x16x256") != std::string::npos,
          "31B-like describe() names its fields", d31);
    check(d26.find("global=5x2x512") != std::string::npos &&
              d26.find("sliding=50x8x256") != std::string::npos,
          "26B-like describe() names its fields", d26);

    std::cout << (failures ? "FAIL" : "OK") << " gemma4 prefix geometry\n";
    return failures ? 1 : 0;
}
