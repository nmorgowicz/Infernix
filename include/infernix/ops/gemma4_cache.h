#pragma once

namespace infernix { struct Tensor; }

namespace infernix::ops {

inline constexpr int kGemma4KvPageTokens = 64;
inline constexpr int kGemma4KvFp8Group = 256;

enum class Gemma4KvFormat { BFloat16, Fp8E4m3Row256 };
enum class Gemma4QkMode { BFloat16, Fp8 };

// Where attention reads V. Stored: the v_pages (and v_scales) planes. Key: there are no V planes;
// a global D512 FP8 layer whose V is its normed raw key without RoPE (Gemma 4's K == V layers) and
// whose key-norm weight is one constant c has V = unrope(K) / c, so readers rebuild V from the K
// codes and scales: dimensions [0, rotary_dim) are pairs (i, i + rotary_dim / 2) turned back by the
// FP32 angle gemma4_qkv_prep applied (position x theta^(-2i/rotary_dim)), every dimension is then
// multiplied by value_scale = 1 / c and rounded once to BF16.
inline constexpr int kGemma4KeyValueRotaryDim = 128;
enum class Gemma4ValueSource { Stored, Key };

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
    // Gemma4ValueSource::Key only (zero otherwise): 1 / c, the RoPE base and kGemma4KeyValueRotaryDim.
    Gemma4ValueSource value_source = Gemma4ValueSource::Stored;
    float value_scale = 0.0f;
    float rope_theta = 0.0f;
    int rotary_dim = 0;
};

} // namespace infernix::ops
