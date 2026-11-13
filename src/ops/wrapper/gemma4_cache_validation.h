#pragma once

#include "core/tensor.h"
#include "infernix/ops/gemma4_cache.h"

#include <stdexcept>
#include <string>

namespace infernix::ops::detail {

inline void validate_gemma4_cache(const Gemma4CacheView& cache, int head_dim, bool global,
                                  const char* op) {
    const auto fail = [&](const char* reason) {
        throw std::invalid_argument(std::string(op) + ": " + reason);
    };
    if (!cache.k_pages || !cache.v_pages || !cache.block_table) {
        fail("cache requires k/v pages and block_table");
    }
    const bool fp8 = cache.format == Gemma4KvFormat::Fp8E4m3Row256;
    if (!fp8 && cache.format != Gemma4KvFormat::BFloat16) { fail("unknown cache format"); }
    if (fp8 && (head_dim != 512 || !global)) {
        fail("FP8 requires D512 global attention");
    }
    if (cache.qk_mode != Gemma4QkMode::BFloat16 && cache.qk_mode != Gemma4QkMode::Fp8) {
        fail("unknown QK mode");
    }
    if (cache.qk_mode == Gemma4QkMode::Fp8 && (!fp8 || head_dim != 512 || !global)) {
        fail("FP8 QK requires D512 global FP8 cache");
    }
    const Tensor& k = *cache.k_pages;
    const Tensor& v = *cache.v_pages;
    const DType dtype = fp8 ? DType::U8 : DType::BF16;
    if (k.dtype != dtype || v.dtype != dtype || !k.is_contiguous() || !v.is_contiguous() ||
        !k.data || !v.data) { fail("cache pages must have matching contiguous non-null storage"); }
    if (k.ne[0] != head_dim || k.ne[1] != kGemma4KvPageTokens || k.ne[2] <= 0 ||
        k.ne[3] <= 0 || v.ne[0] != k.ne[0] || v.ne[1] != k.ne[1] ||
        v.ne[2] != k.ne[2] || v.ne[3] != k.ne[3]) {
        fail("k/v pages must be matching [D,64,Hkv,pages]");
    }
    const Tensor& table = *cache.block_table;
    if (table.dtype != DType::I32 || !table.is_contiguous() || !table.data || table.ne[0] <= 0 ||
        table.ne[1] != 1 || table.ne[2] != 1 || table.ne[3] != 1) {
        fail("block_table must be contiguous non-null nonempty I32 [logical_pages]");
    }
    if (!fp8) {
        if (cache.k_scales || cache.v_scales) { fail("BF16 cache must not have scale planes"); }
        return;
    }
    if (!cache.k_scales || !cache.v_scales) { fail("FP8 cache requires both scale planes"); }
    for (const Tensor* scales : {cache.k_scales, cache.v_scales}) {
        if (scales->dtype != DType::FP16 || !scales->is_contiguous() || !scales->data ||
            scales->ne[0] != head_dim / kGemma4KvFp8Group ||
            scales->ne[1] != kGemma4KvPageTokens || scales->ne[2] != k.ne[2] ||
            scales->ne[3] != k.ne[3]) {
            fail("FP8 scales must be contiguous non-null FP16 [2,64,Hkv,pages]");
        }
    }
}

} // namespace infernix::ops::detail
