#pragma once

namespace infernix { struct Tensor; }

namespace infernix::ops {

inline constexpr int kGemma4KvPageTokens = 64;
inline constexpr int kGemma4KvFp8Group = 256;

enum class Gemma4KvFormat { BFloat16, Fp8E4m3Row256 };
enum class Gemma4QkMode { BFloat16, Fp8 };

// Non-owning view; finalize tensor owners before taking their addresses. BF16 pages are
// [D,64,Hkv,pages]. Global-only FP8 pages are U8 [512,64,Hkv,pages], with separate
// FP16 scales [2,64,Hkv,pages]. All planes use the same physical page IDs.
struct Gemma4CacheView {
    Gemma4KvFormat format = Gemma4KvFormat::BFloat16;
    Tensor* k_pages = nullptr;
    Tensor* v_pages = nullptr;
    Tensor* k_scales = nullptr;
    Tensor* v_scales = nullptr;
    const Tensor* block_table = nullptr;
    Gemma4QkMode qk_mode = Gemma4QkMode::BFloat16;
};

} // namespace infernix::ops
