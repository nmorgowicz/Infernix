#include "core/device.h"
#include "infernix/ops/gemma4_cache.h"
#include "infernix/ops/gemma4_ring_copy.h"
#include "ops/op_family.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <deque>
#include <iostream>
#include <random>
#include <vector>

using namespace infernix;
using namespace infernix::test;

namespace {

constexpr int kPage = 64;

// One sliding layer's K/V planes over `pages` physical pages, filled with random bytes, with the block
// table logical page -> page % pages (the engine's ring mapping).
struct Ring {
    int pages;
    int heads;
    int head_dim;
    bool fp8;
    std::vector<std::vector<std::uint8_t>> host; // K, V, K scales, V scales
    std::deque<GuardedDeviceBuffer> device;
    std::vector<std::int32_t> table;
    GuardedDeviceBuffer table_device;
    Tensor table_tensor;
    Tensor k, v, ks, vs;

    Ring(int pages_, int heads_, int head_dim_, bool fp8_, int logical_pages, std::uint32_t seed)
        : pages(pages_), heads(heads_), head_dim(head_dim_), fp8(fp8_), table(static_cast<std::size_t>(logical_pages)),
          table_device(static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t)) {
        const std::size_t rows = static_cast<std::size_t>(pages) * heads * kPage;
        const std::size_t code_bytes = rows * head_dim * (fp8 ? 1 : 2);
        const std::size_t scale_bytes = rows * 2;
        std::mt19937 rng(seed);
        host.resize(fp8 ? 4 : 2);
        for (std::size_t i = 0; i < host.size(); ++i) {
            host[i].resize(i < 2 ? code_bytes : scale_bytes);
            for (auto& b : host[i]) { b = static_cast<std::uint8_t>(rng()); }
            device.emplace_back(host[i].size());
            device.back().copy_from_host(host[i].data(), host[i].size());
        }
        for (int p = 0; p < logical_pages; ++p) { table[static_cast<std::size_t>(p)] = p % pages; }
        table_device.copy_from_host(table.data(), table.size() * sizeof(std::int32_t));
        table_tensor = Tensor(table_device.data(), DType::I32, {logical_pages});
        const DType code = fp8 ? DType::U8 : DType::BF16;
        k = Tensor(device[0].data(), code, {head_dim, kPage, heads, pages});
        v = Tensor(device[1].data(), code, {head_dim, kPage, heads, pages});
        if (fp8) {
            ks = Tensor(device[2].data(), DType::FP16, {1, kPage, heads, pages});
            vs = Tensor(device[3].data(), DType::FP16, {1, kPage, heads, pages});
        }
    }

    ops::Gemma4CacheView view() {
        ops::Gemma4CacheView cache;
        cache.format = fp8 ? ops::Gemma4KvFormat::Fp8E4m3Row256 : ops::Gemma4KvFormat::BFloat16;
        cache.k_pages = &k;
        cache.v_pages = &v;
        cache.k_scales = fp8 ? &ks : nullptr;
        cache.v_scales = fp8 ? &vs : nullptr;
        cache.block_table = &table_tensor;
        return cache;
    }

    // Byte offset of row `slot` of head `h` on physical page `page` in plane `i`.
    std::size_t offset(std::size_t i, int page, int h, int slot) const {
        const std::size_t row_bytes = i < 2 ? static_cast<std::size_t>(head_dim) * (fp8 ? 1 : 2) : 2;
        return row_bytes * (static_cast<std::size_t>(slot) + kPage * (static_cast<std::size_t>(h) + static_cast<std::size_t>(heads) * page));
    }
    std::size_t row_bytes(std::size_t i) const { return i < 2 ? static_cast<std::size_t>(head_dim) * (fp8 ? 1 : 2) : 2; }
};

struct Case {
    const char* label;
    bool fp8;
    bool window;
    int first;
    int tokens;
    int window_rows;
    int src_pages;
    int dst_pages;
    int heads;
};

int run_case(const Case& c, std::uint32_t seed) {
    constexpr int kLogical = 400;
    Ring src(c.src_pages, c.heads, 256, c.fp8, kLogical, seed);
    Ring dst(c.dst_pages, c.heads, 256, c.fp8, kLogical, seed + 1);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(c.tokens));
    for (int i = 0; i < c.tokens; ++i) { positions[static_cast<std::size_t>(i)] = c.first + i; }
    GuardedDeviceBuffer positions_device(positions.size() * sizeof(std::int32_t));
    positions_device.copy_from_host(positions.data(), positions_device.bytes());
    Tensor positions_tensor(positions_device.data(), DType::I32, {c.tokens});
    const int capacity_rows = c.dst_pages * kPage;

    ops::gemma4_ring_copy(src.view(), dst.view(), positions_tensor,
                          c.window ? ops::Gemma4RingCopyRows::Window : ops::Gemma4RingCopyRows::Tail, c.tokens,
                          c.window_rows, capacity_rows, nullptr);
    cuda_synchronize();

    // Expected: dst bytes with the selected rows replaced by the source rows.
    int lo = 0;
    int hi = 0;
    if (c.window) {
        hi = c.first;
        lo = std::max(0, c.first - c.window_rows);
    } else {
        hi = c.first + c.tokens;
        lo = std::max(c.first, hi - capacity_rows);
    }
    auto expected = dst.host;
    for (int p = lo; p < hi; ++p) {
        const int sp = src.table[static_cast<std::size_t>(p / kPage)];
        const int dp = dst.table[static_cast<std::size_t>(p / kPage)];
        for (std::size_t plane = 0; plane < expected.size(); ++plane) {
            for (int h = 0; h < c.heads; ++h) {
                const std::size_t n = src.row_bytes(plane);
                std::copy_n(src.host[plane].begin() + static_cast<std::ptrdiff_t>(src.offset(plane, sp, h, p % kPage)), n,
                            expected[plane].begin() + static_cast<std::ptrdiff_t>(dst.offset(plane, dp, h, p % kPage)));
            }
        }
    }
    int failures = 0;
    for (std::size_t plane = 0; plane < expected.size(); ++plane) {
        std::vector<std::uint8_t> got(expected[plane].size());
        dst.device[plane].copy_to_host(got.data(), got.size());
        if (got != expected[plane]) {
            std::cout << "FAIL " << c.label << ": plane " << plane << " differs from the row-exact expectation\n";
            ++failures;
        }
        failures += dst.device[plane].verify_guards(c.label);
        std::vector<std::uint8_t> source(src.host[plane].size());
        src.device[plane].copy_to_host(source.data(), source.size());
        if (source != src.host[plane]) {
            std::cout << "FAIL " << c.label << ": source plane " << plane << " modified\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace

INFERNIX_OP_TEST(gemma4_ring_copy) {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    // Persistent 24 pages, staging 48 (chunk 2,048: pages_for(2,048 + 1,023)), window 1,023 rows.
    const std::vector<Case> cases = {
        {"window bf16 first=0 (empty)", false, true, 0, 2048, 1023, 24, 48, 4},
        {"window bf16 first=100", false, true, 100, 2048, 1023, 24, 48, 4},
        {"window bf16 first=1023", false, true, 1023, 2048, 1023, 24, 48, 4},
        {"window bf16 first=2048 (aligned)", false, true, 2048, 2048, 1023, 24, 48, 4},
        {"window bf16 first=5000 (unaligned, wrapped)", false, true, 5000, 2048, 1023, 24, 48, 4},
        {"window fp8 first=5000", true, true, 5000, 2048, 1023, 24, 48, 4},
        {"window fp8 first=1040 16 heads", true, true, 1040, 1024, 1023, 24, 32, 16},
        {"tail bf16 first=0 w=2048 (wider than the ring)", false, false, 0, 2048, 1023, 48, 24, 4},
        {"tail bf16 first=2048 w=2048", false, false, 2048, 2048, 1023, 48, 24, 4},
        {"tail bf16 first=5000 w=2048 (unaligned)", false, false, 5000, 2048, 1023, 48, 24, 4},
        {"tail bf16 first=5001 w=700 (narrower than the ring)", false, false, 5001, 700, 1023, 48, 24, 4},
        {"tail fp8 first=7777 w=2048", true, false, 7777, 2048, 1023, 48, 24, 4},
        {"tail fp8 first=100 w=1024 16 heads", true, false, 100, 1024, 1023, 32, 24, 16},
    };
    int failures = 0;
    std::uint32_t seed = 100;
    for (const auto& c : cases) { failures += run_case(c, seed += 7); }
    std::cout << (failures ? "FAIL" : "OK") << " gemma4_ring_copy\n";
    return failures ? 1 : 0;
}
