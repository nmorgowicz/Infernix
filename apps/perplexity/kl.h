// Ported from ninfer.spo full-vocabulary KL (NKLREF01 format and statistics preserved).
#pragma once

// Exact paired full-vocabulary KL for infernix-perplexity.
//
// A reference run writes the F32 log-probability distribution of every scored position to a
// cache; a candidate run streams that cache while scoring and accumulates D_KL(ref || cand) per
// position in F64. Nothing here touches the GPU.

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <fstream>
#include <span>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace infernix::perplexity::kl {

// F32 natural-log softmax of one BF16 logit column (host bits). The normalizer is accumulated in
// F64 so that the rounding of the stored F32 values, not of the sum, bounds the error.
void log_softmax_bf16(const std::uint16_t* logits, std::int32_t vocab, float* out);

// The same operation for BF16 logits widened exactly to F32 by Engine::score_tokens.
void log_softmax_f32(const float* logits, std::int32_t vocab, float* out);

// D_KL(p_ref || p_cand) = sum_v exp(r_v) (r_v - c_v), accumulated in F64. Not clamped: identical
// inputs give exactly zero and numerical noise stays visible as small signed values.
[[nodiscard]] double kl_divergence(const float* ref, const float* cand, std::int32_t vocab);

// Index of the largest value; the first on ties.
[[nodiscard]] std::int32_t argmax(const float* values, std::int32_t vocab);

// Runs fn(begin, end) over [0, count) split across up to `workers` threads; blocks until done.
void parallel_columns(std::int32_t count, const std::function<void(std::int32_t, std::int32_t)>& fn);

// Linear-interpolated percentile (q in [0,1]) of a copy of `values`; NaN for an empty input.
[[nodiscard]] double percentile(std::span<const float> values, double q);

struct Interval {
    double mean = 0.0;
    double low  = 0.0;
    double high = 0.0;
};

// Ratio-of-sums mean (sum of window sums / sum of window counts) with a 95% percentile interval
// from a bootstrap that resamples whole windows. Deterministic for a fixed seed.
[[nodiscard]] Interval window_bootstrap(std::span<const double> window_sums,
                                        std::span<const std::uint64_t> window_counts,
                                        std::uint32_t resamples, std::uint64_t seed);

// Same statistic for the paired difference of two arms scored on the same windows.
[[nodiscard]] Interval paired_window_bootstrap(std::span<const double> sums_a,
                                               std::span<const double> sums_b,
                                               std::span<const std::uint64_t> window_counts,
                                               std::uint32_t resamples, std::uint64_t seed);

// 64-bit FNV-1a, used for token-stream/window identity (not a security hash).
class Fnv64 {
public:
    void add(const void* data, std::size_t bytes) noexcept;
    template <class T>
    void add_value(const T& value) noexcept {
        add(&value, sizeof(T));
    }
    [[nodiscard]] std::uint64_t value() const noexcept { return state_; }

private:
    std::uint64_t state_ = 14695981039346656037ULL;
};

// Hash raw BF16 bits from the public API's exactly widened F32 logits, in position/vocab order.
// Rejects unrounded logits instead of silently truncating them; matches the evidence adapter.
void hash_bf16_logits(Fnv64& hash, std::span<const float> logits);

// File layout: "NKLREF01", u32 version, u32 vocab, u64 positions (0 until finished), u64 header
// json bytes, u64 data offset (4096-aligned), then the json, then [positions, vocab] F32 rows.
// The json holds {"protocol": {...}, "reference": {...}}. Candidate mode requires `protocol` to
// equal its own protocol exactly; `reference` is informational provenance of the writer.
constexpr std::uint32_t kCacheVersion = 1;
constexpr std::uint64_t kCacheAlign   = 4096;

class CacheWriter {
public:
    CacheWriter(const std::filesystem::path& path, std::int32_t vocab,
                const nlohmann::json& protocol, const nlohmann::json& reference);
    void append(std::span<const float> rows, std::int32_t columns);
    // Rewrites the header with the final position count. A cache that was never finished has
    // positions == 0 and is refused by the reader.
    void finish();
    [[nodiscard]] std::uint64_t positions() const noexcept { return positions_; }

private:
    std::filesystem::path path_;
    std::ofstream file_;
    std::int32_t vocab_;
    std::uint64_t json_bytes_ = 0;
    std::uint64_t data_offset_ = 0;
    std::uint64_t positions_  = 0;
    bool finished_            = false;
};

class CacheReader {
public:
    // Throws std::runtime_error naming the cause on a missing, truncated, unfinished or
    // protocol-mismatched cache.
    CacheReader(const std::filesystem::path& path, std::int32_t vocab,
                const nlohmann::json& expected_protocol);
    void read(std::int32_t columns, std::span<float> rows);
    [[nodiscard]] std::uint64_t positions() const noexcept { return positions_; }
    [[nodiscard]] std::uint64_t consumed() const noexcept { return consumed_; }
    [[nodiscard]] const nlohmann::json& reference() const noexcept { return reference_; }

private:
    std::ifstream file_;
    std::int32_t vocab_;
    std::uint64_t positions_ = 0;
    std::uint64_t consumed_  = 0;
    nlohmann::json reference_;
};

// Per-position results of a candidate run, in scoring order. All vectors have one entry per
// scored position. `ref_lp`/`cand_lp` are the target-token log-probabilities under each arm.
struct PositionStats {
    std::vector<float> kl;
    std::vector<float> ref_lp;
    std::vector<float> cand_lp;
    std::vector<std::uint8_t> top1_agree;
};

struct WindowRecord {
    std::string stream;
    std::string domain;
    std::uint64_t begin = 0; // first position index in PositionStats
    std::uint64_t count = 0;
};

// JSON report: overall and per-domain mean KL with window-bootstrap interval, median/p95/p99/p99.9
// KL, mean change in target NLL (candidate minus reference) with interval, top-1 disagreement,
// and the per-window sums needed to pair two arms later.
[[nodiscard]] nlohmann::json summarize(const PositionStats& stats,
                                       std::span<const WindowRecord> windows,
                                       std::uint32_t resamples, std::uint64_t seed);

} // namespace infernix::perplexity::kl
