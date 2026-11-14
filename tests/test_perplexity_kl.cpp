// Ported from ninfer.spo full-vocabulary KL behavioral tests.
#include "kl.h"
#include "evaluation.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace kl = infernix::perplexity::kl;

namespace {

int require(bool condition, const char* label) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << label << '\n';
    return 1;
}

std::uint16_t bf16_bits(float value) {
    return static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
}

template <class Fn>
bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

std::filesystem::path scratch(const char* name) {
    auto path = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove(path);
    return path;
}

} // namespace

int main() {
    int failures = 0;

    // Log-softmax of exactly representable logits {0, ln2, ln2, ...} has a known normalizer.
    {
        const float ln2 = 0.69140625F; // exact in BF16 (close to ln 2; used only as a value)
        const std::vector<std::uint16_t> logits{bf16_bits(0.0F), bf16_bits(ln2), bf16_bits(ln2),
                                                bf16_bits(-1.0F)};
        std::vector<float> out(4);
        kl::log_softmax_bf16(logits.data(), 4, out.data());
        const double z = 1.0 + 2.0 * std::exp(static_cast<double>(ln2)) + std::exp(-1.0);
        double total   = 0.0;
        for (int i = 0; i < 4; ++i) { total += std::exp(static_cast<double>(out[i])); }
        failures += require(std::abs(total - 1.0) < 1e-6, "log-softmax normalizes to one");
        failures += require(std::abs(static_cast<double>(out[0]) + std::log(z)) < 1e-6,
                            "log-softmax value matches the analytic normalizer");
        std::vector<float> widened;
        for (const auto bits : logits) {
            widened.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16));
        }
        std::vector<float> native(4);
        kl::log_softmax_f32(widened.data(), 4, native.data());
        failures += require(native == out, "native widened logits preserve fork log-probability bits");
        kl::Fnv64 adapter_hash;
        adapter_hash.add(logits.data(), logits.size() * sizeof(std::uint16_t));
        kl::Fnv64 native_hash;
        kl::hash_bf16_logits(native_hash, std::span<const float>(widened).first(2));
        kl::hash_bf16_logits(native_hash, std::span<const float>(widened).subspan(2));
        failures += require(native_hash.value() == adapter_hash.value(),
                            "window hash matches raw BF16 adapter bytes across tile boundaries");
        failures += require(throws([&] {
            const std::vector<float> unrounded{0.1F};
            kl::hash_bf16_logits(native_hash, unrounded);
        }), "window hash refuses logits that were not rounded to BF16");
    }

    // Analytic KL between two 3-point distributions; the reverse direction differs.
    {
        const std::vector<double> p{0.5, 0.3, 0.2};
        const std::vector<double> q{0.25, 0.25, 0.5};
        std::vector<float> lp(3);
        std::vector<float> lq(3);
        double forward = 0.0;
        double reverse = 0.0;
        for (int i = 0; i < 3; ++i) {
            lp[i] = static_cast<float>(std::log(p[i]));
            lq[i] = static_cast<float>(std::log(q[i]));
            forward += p[i] * std::log(p[i] / q[i]);
            reverse += q[i] * std::log(q[i] / p[i]);
        }
        failures += require(std::abs(kl::kl_divergence(lp.data(), lq.data(), 3) - forward) < 1e-6,
                            "KL(p||q) matches the analytic value");
        failures += require(std::abs(kl::kl_divergence(lq.data(), lp.data(), 3) - reverse) < 1e-6,
                            "KL(q||p) matches the analytic value");
        failures += require(std::abs(forward - reverse) > 1e-3, "KL is asymmetric");
        failures += require(kl::kl_divergence(lp.data(), lp.data(), 3) == 0.0,
                            "identical distributions give exactly zero");
        failures += require(kl::argmax(lq.data(), 3) == 2, "argmax picks the largest log-prob");
    }

    // Percentiles interpolate linearly and ignore input order.
    {
        const std::vector<float> values{5.0F, 1.0F, 3.0F, 2.0F, 4.0F};
        failures += require(kl::percentile(values, 0.0) == 1.0, "percentile minimum");
        failures += require(kl::percentile(values, 1.0) == 5.0, "percentile maximum");
        failures += require(kl::percentile(values, 0.5) == 3.0, "percentile median");
        failures += require(std::abs(kl::percentile(values, 0.875) - 4.5) < 1e-12,
                            "percentile interpolates");
        failures += require(std::isnan(kl::percentile({}, 0.5)), "empty percentile is NaN");
    }

    // Window bootstrap: deterministic per seed, interval brackets the ratio-of-sums mean, and a
    // constant-per-position statistic has a degenerate interval.
    {
        const std::vector<double> sums{1.0, 2.0, 3.0, 10.0};
        const std::vector<std::uint64_t> counts{1, 1, 1, 1};
        const auto a = kl::window_bootstrap(sums, counts, 1000, 7);
        const auto b = kl::window_bootstrap(sums, counts, 1000, 7);
        const auto c = kl::window_bootstrap(sums, counts, 1000, 8);
        failures += require(std::abs(a.mean - 4.0) < 1e-12, "bootstrap mean is the ratio of sums");
        failures += require(a.low == b.low && a.high == b.high, "bootstrap is deterministic per seed");
        failures += require(a.low != c.low || a.high != c.high, "bootstrap depends on the seed");
        failures += require(a.low <= a.mean && a.mean <= a.high, "interval brackets the mean");
        const std::vector<double> flat{2.0, 4.0, 6.0};
        const std::vector<std::uint64_t> flat_counts{1, 2, 3};
        const auto d = kl::window_bootstrap(flat, flat_counts, 500, 1);
        failures += require(std::abs(d.low - 2.0) < 1e-12 && std::abs(d.high - 2.0) < 1e-12,
                            "constant per-position value gives a degenerate interval");
        const auto paired = kl::paired_window_bootstrap(sums, sums, counts, 200, 3);
        failures += require(paired.mean == 0.0 && paired.low == 0.0 && paired.high == 0.0,
                            "identical arms have a zero paired difference");
        failures += require(throws([&] {
            (void)kl::window_bootstrap(std::vector<double>{1.0}, std::vector<std::uint64_t>{}, 10, 1);
        }), "bootstrap rejects mismatched windows");
    }

    // Cache round trip, tile-boundary equivalence, and refusals.
    {
        const std::int32_t vocab = 5;
        const nlohmann::json protocol{{"vocab", vocab}, {"context", 64}, {"token_hash", "abc"}};
        const nlohmann::json reference{{"artifact", "ref"}};
        std::vector<float> rows(static_cast<std::size_t>(7) * vocab);
        std::iota(rows.begin(), rows.end(), 0.5F);

        const auto path = scratch("infernix_kl_cache_roundtrip.nklref");
        {
            kl::CacheWriter writer(path, vocab, protocol, reference);
            writer.append({rows.data(), static_cast<std::size_t>(3) * vocab}, 3);
            writer.append({rows.data() + static_cast<std::size_t>(3) * vocab,
                           static_cast<std::size_t>(4) * vocab},
                          4);
            writer.finish();
        }
        failures += require(std::filesystem::file_size(path) % 4 == 0, "cache is float-sized");
        {
            kl::CacheReader reader(path, vocab, protocol);
            failures += require(reader.positions() == 7, "reader reports the position count");
            failures += require(reader.reference() == reference, "reader exposes writer provenance");
            std::vector<float> back(rows.size());
            reader.read(2, {back.data(), static_cast<std::size_t>(2) * vocab});
            reader.read(5, {back.data() + static_cast<std::size_t>(2) * vocab,
                            static_cast<std::size_t>(5) * vocab});
            failures += require(back == rows, "rows round-trip across different tile boundaries");
            std::vector<float> one(vocab);
            failures += require(throws([&] { reader.read(1, one); }),
                                "reading past the end is refused");
        }
        failures += require(throws([&] {
            kl::CacheReader reader(path, vocab, nlohmann::json{{"vocab", vocab}, {"context", 65},
                                                                {"token_hash", "abc"}});
        }), "protocol mismatch is refused");
        failures += require(throws([&] { kl::CacheReader reader(path, vocab + 1, protocol); }),
                            "vocab mismatch is refused");
        failures += require(throws([&] { kl::CacheWriter again(path, vocab, protocol, reference); }),
                            "an existing cache is never overwritten");
        for (const char* field : {"token_hash", "context", "kv_dtype", "gemma4_qk",
                                   "prefill_chunk_tokens", "prefix_ids", "positions"}) {
            auto different = protocol;
            different[field] = "changed";
            failures += require(throws([&] { kl::CacheReader reader(path, vocab, different); }),
                                "every protocol difference is refused");
        }

        // Independent NKLREF01 fixture, not produced by CacheWriter: checks fork wire compatibility.
        const auto legacy = scratch("infernix_kl_cache_legacy.nklref");
        {
            const auto header = nlohmann::json{{"protocol", protocol}, {"reference", reference}}.dump();
            std::ofstream file(legacy, std::ios::binary);
            const std::uint32_t version = 1;
            const std::uint32_t vocabulary = vocab;
            const std::uint64_t positions = 7;
            const std::uint64_t json_bytes = header.size();
            const std::uint64_t data_offset = 4096;
            file.write("NKLREF01", 8);
            file.write(reinterpret_cast<const char*>(&version), sizeof(version));
            file.write(reinterpret_cast<const char*>(&vocabulary), sizeof(vocabulary));
            file.write(reinterpret_cast<const char*>(&positions), sizeof(positions));
            file.write(reinterpret_cast<const char*>(&json_bytes), sizeof(json_bytes));
            file.write(reinterpret_cast<const char*>(&data_offset), sizeof(data_offset));
            file.write(header.data(), static_cast<std::streamsize>(header.size()));
            const std::vector<char> padding(data_offset - 40 - header.size(), 0);
            file.write(padding.data(), static_cast<std::streamsize>(padding.size()));
            file.write(reinterpret_cast<const char*>(rows.data()),
                       static_cast<std::streamsize>(rows.size() * sizeof(float)));
        }
        {
            kl::CacheReader reader(legacy, vocab, protocol);
            std::vector<float> back(rows.size());
            reader.read(7, back);
            failures += require(back == rows, "independent fork-format fixture reads exactly");
        }
        std::filesystem::remove(legacy);

        const auto corrupt = scratch("infernix_kl_cache_header.nklref");
        std::filesystem::copy_file(path, corrupt);
        {
            std::fstream file(corrupt, std::ios::binary | std::ios::in | std::ios::out);
            const auto huge = std::numeric_limits<std::uint64_t>::max();
            file.seekp(24); // json_bytes must fit before the aligned data offset.
            file.write(reinterpret_cast<const char*>(&huge), sizeof(huge));
        }
        failures += require(throws([&] { kl::CacheReader reader(corrupt, vocab, protocol); }),
                            "corrupt header bounds are refused before allocation");
        std::filesystem::remove(corrupt);

        // Truncation.
        const auto truncated = scratch("infernix_kl_cache_truncated.nklref");
        std::filesystem::copy_file(path, truncated);
        std::filesystem::resize_file(truncated, std::filesystem::file_size(truncated) - 4);
        failures += require(throws([&] { kl::CacheReader reader(truncated, vocab, protocol); }),
                            "a truncated cache is refused");

        // An unfinished cache (crashed reference run) has zero positions and is refused.
        const auto unfinished = scratch("infernix_kl_cache_unfinished.nklref");
        {
            kl::CacheWriter writer(unfinished, vocab, protocol, reference);
            writer.append({rows.data(), static_cast<std::size_t>(vocab)}, 1);
        }
        failures += require(throws([&] { kl::CacheReader reader(unfinished, vocab, protocol); }),
                            "an unfinished cache is refused");

        // Not a cache at all.
        const auto garbage = scratch("infernix_kl_cache_garbage.nklref");
        { std::ofstream(garbage, std::ios::binary) << "not a cache"; }
        failures += require(throws([&] { kl::CacheReader reader(garbage, vocab, protocol); }),
                            "a foreign file is refused");

        for (const auto& file : {path, truncated, unfinished, garbage}) {
            std::filesystem::remove(file);
        }
    }

    // Pinned ref512-en metadata: prefix tokens are conditioning only, never scored targets.
    {
        const std::vector<std::size_t> stream_tokens{
            1075, 1148, 1077, 981, 1190, 1165, 1211, 1097, 1426, 1209, 941, 1220};
        const std::vector<std::size_t> tails{51, 124, 53, 213, 166, 141, 187, 73, 146, 185, 173, 196};
        std::size_t total_positions = 0;
        std::size_t total_windows = 0;
        for (std::size_t i = 0; i < stream_tokens.size(); ++i) {
            const auto windows = infernix::perplexity::plan_windows(stream_tokens[i], 512, 256);
            failures += require(windows.front().target_end - windows.front().target_begin == 511,
                                "pinned initial window scores 511 targets before prefix conditioning");
            failures += require(windows.back().target_end - windows.back().target_begin == tails[i],
                                "pinned final window preserves its short target suffix");
            for (const auto& window : windows) {
                total_positions += window.target_end - window.target_begin;
                const auto prefixed_first_target = window.first_target + 17;
                failures += require(window.input_end - window.input_begin + 17 - prefixed_first_target ==
                                        window.target_end - window.target_begin,
                                    "17-token pinned prefix does not change the number of targets");
            }
            total_windows += windows.size();
        }
        failures += require(total_positions == 13728 && total_windows == 47,
                            "pinned ref512-en plans 13728 positions over 47 windows");
    }

    // Identity hash is order-sensitive.
    {
        kl::Fnv64 a;
        kl::Fnv64 b;
        const std::int32_t x = 1;
        const std::int32_t y = 2;
        a.add_value(x);
        a.add_value(y);
        b.add_value(y);
        b.add_value(x);
        failures += require(a.value() != b.value(), "token-stream hash is order sensitive");
    }

    // Summary: per-domain statistics, window tiling validation and ΔNLL sign.
    {
        kl::PositionStats stats;
        stats.kl         = {0.1F, 0.1F, 0.3F, 0.3F, 0.3F};
        stats.ref_lp     = {-1.0F, -1.0F, -2.0F, -2.0F, -2.0F};
        stats.cand_lp    = {-1.5F, -1.5F, -2.0F, -2.0F, -2.0F};
        stats.top1_agree = {1, 0, 1, 1, 1};
        const std::vector<kl::WindowRecord> windows{{"a", "alpha", 0, 2}, {"b", "beta", 2, 3}};
        const auto report = kl::summarize(stats, windows, 100, 5);
        const auto& overall = report.at("overall");
        failures += require(overall.at("positions") == 5 && overall.at("windows") == 2,
                            "summary counts positions and windows");
        failures += require(std::abs(overall.at("mean_kl").get<double>() - 0.22) < 1e-7,
                            "overall mean KL is the ratio of sums");
        failures += require(std::abs(overall.at("top1_disagreement").get<double>() - 0.2) < 1e-12,
                            "top-1 disagreement counts positions");
        failures += require(std::abs(overall.at("mean_delta_nll").get<double>() - 0.2) < 1e-6,
                            "worse candidate has positive change in NLL");
        failures += require(report.at("domains").size() == 2 &&
                                report.at("domains")[0].at("domain") == "alpha" &&
                                std::abs(report.at("domains")[1].at("mean_kl").get<double>() - 0.3) < 1e-7,
                            "per-domain statistics are separated");
        failures += require(report.at("windows")[0].at("kl_sum").get<double>() > 0.19 &&
                                report.at("windows").size() == 2,
                            "per-window sums are retained for pairing");
        const std::vector<kl::WindowRecord> gap{{"a", "alpha", 0, 2}, {"b", "beta", 3, 2}};
        failures += require(throws([&] { (void)kl::summarize(stats, gap, 10, 1); }),
                            "non-contiguous windows are refused");
        const std::vector<kl::WindowRecord> short_cover{{"a", "alpha", 0, 2}};
        failures += require(throws([&] { (void)kl::summarize(stats, short_cover, 10, 1); }),
                            "windows that miss positions are refused");
    }

    std::cout << (failures == 0 ? "OK" : "FAIL") << " perplexity_kl\n";
    return failures == 0 ? 0 : 1;
}
