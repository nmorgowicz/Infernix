// Ported from ninfer.spo full-vocabulary KL (NKLREF01 format and statistics preserved).
#include "kl.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace infernix::perplexity::kl {
namespace {

float bf16_to_float(std::uint16_t bits) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16);
}

constexpr char kMagic[8] = {'N', 'K', 'L', 'R', 'E', 'F', '0', '1'};
constexpr std::size_t kFixedHeader = 8 + 4 + 4 + 8 + 8 + 8;

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

void write_header(std::ofstream& file, std::uint32_t vocab, std::uint64_t positions,
                  std::uint64_t json_bytes, std::uint64_t data_offset) {
    const std::uint32_t version = kCacheVersion;
    file.write(kMagic, sizeof(kMagic));
    file.write(reinterpret_cast<const char*>(&version), sizeof(version));
    file.write(reinterpret_cast<const char*>(&vocab), sizeof(vocab));
    file.write(reinterpret_cast<const char*>(&positions), sizeof(positions));
    file.write(reinterpret_cast<const char*>(&json_bytes), sizeof(json_bytes));
    file.write(reinterpret_cast<const char*>(&data_offset), sizeof(data_offset));
}

std::string describe_protocol_difference(const nlohmann::json& expected,
                                         const nlohmann::json& actual) {
    std::string out;
    for (auto it = expected.begin(); it != expected.end(); ++it) {
        if (!actual.contains(it.key()) || actual.at(it.key()) != it.value()) {
            out += (out.empty() ? "" : ", ") + it.key() + " (this run " + it.value().dump() +
                   " vs cache " +
                   (actual.contains(it.key()) ? actual.at(it.key()).dump() : "missing") + ")";
        }
    }
    for (auto it = actual.begin(); it != actual.end(); ++it) {
        if (!expected.contains(it.key())) { out += (out.empty() ? "" : ", ") + it.key() + " (extra in cache)"; }
    }
    return out;
}

Interval interval_from(std::vector<double>& resampled, double mean) {
    std::sort(resampled.begin(), resampled.end());
    const auto at = [&](double q) {
        const double position = q * static_cast<double>(resampled.size() - 1);
        const auto lower      = static_cast<std::size_t>(std::floor(position));
        const auto upper      = static_cast<std::size_t>(std::ceil(position));
        const double fraction = position - static_cast<double>(lower);
        return resampled[lower] * (1.0 - fraction) + resampled[upper] * fraction;
    };
    return Interval{.mean = mean, .low = at(0.025), .high = at(0.975)};
}

} // namespace

void log_softmax_bf16(const std::uint16_t* logits, std::int32_t vocab, float* out) {
    float peak = -std::numeric_limits<float>::infinity();
    for (std::int32_t i = 0; i < vocab; ++i) { peak = std::max(peak, bf16_to_float(logits[i])); }
    double sum = 0.0;
    for (std::int32_t i = 0; i < vocab; ++i) {
        sum += std::exp(static_cast<double>(bf16_to_float(logits[i])) - static_cast<double>(peak));
    }
    const double log_total = static_cast<double>(peak) + std::log(sum);
    for (std::int32_t i = 0; i < vocab; ++i) {
        out[i] = static_cast<float>(static_cast<double>(bf16_to_float(logits[i])) - log_total);
    }
}

void log_softmax_f32(const float* logits, std::int32_t vocab, float* out) {
    float peak = -std::numeric_limits<float>::infinity();
    for (std::int32_t i = 0; i < vocab; ++i) { peak = std::max(peak, logits[i]); }
    double sum = 0.0;
    for (std::int32_t i = 0; i < vocab; ++i) {
        sum += std::exp(static_cast<double>(logits[i]) - static_cast<double>(peak));
    }
    const double log_total = static_cast<double>(peak) + std::log(sum);
    for (std::int32_t i = 0; i < vocab; ++i) {
        out[i] = static_cast<float>(static_cast<double>(logits[i]) - log_total);
    }
}

double kl_divergence(const float* ref, const float* cand, std::int32_t vocab) {
    double total = 0.0;
    for (std::int32_t i = 0; i < vocab; ++i) {
        const double r = ref[i];
        total += std::exp(r) * (r - static_cast<double>(cand[i]));
    }
    return total;
}

std::int32_t argmax(const float* values, std::int32_t vocab) {
    std::int32_t best = 0;
    for (std::int32_t i = 1; i < vocab; ++i) {
        if (values[i] > values[best]) { best = i; }
    }
    return best;
}

void parallel_columns(std::int32_t count,
                      const std::function<void(std::int32_t, std::int32_t)>& fn) {
    if (count <= 0) { return; }
    const auto workers = static_cast<std::int32_t>(
        std::clamp<unsigned>(std::thread::hardware_concurrency(), 1U, 32U));
    const std::int32_t per = (count + workers - 1) / workers;
    std::vector<std::jthread> threads;
    for (std::int32_t begin = per; begin < count; begin += per) {
        threads.emplace_back(fn, begin, std::min(count, begin + per));
    }
    fn(0, std::min(count, per));
}

double percentile(std::span<const float> values, double q) {
    if (values.empty()) { return std::numeric_limits<double>::quiet_NaN(); }
    std::vector<float> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    const double position = std::clamp(q, 0.0, 1.0) * static_cast<double>(sorted.size() - 1);
    const auto lower      = static_cast<std::size_t>(std::floor(position));
    const auto upper      = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return static_cast<double>(sorted[lower]) * (1.0 - fraction) +
           static_cast<double>(sorted[upper]) * fraction;
}

Interval window_bootstrap(std::span<const double> window_sums,
                          std::span<const std::uint64_t> window_counts, std::uint32_t resamples,
                          std::uint64_t seed) {
    if (window_sums.size() != window_counts.size() || window_sums.empty() || resamples == 0) {
        throw std::invalid_argument("bootstrap needs matching nonempty windows and resamples>0");
    }
    double sum = 0.0;
    double n   = 0.0;
    for (std::size_t i = 0; i < window_sums.size(); ++i) {
        sum += window_sums[i];
        n += static_cast<double>(window_counts[i]);
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::size_t> pick(0, window_sums.size() - 1);
    std::vector<double> resampled(resamples);
    for (auto& value : resampled) {
        double s = 0.0;
        double c = 0.0;
        for (std::size_t i = 0; i < window_sums.size(); ++i) {
            const std::size_t j = pick(rng);
            s += window_sums[j];
            c += static_cast<double>(window_counts[j]);
        }
        value = s / c;
    }
    return interval_from(resampled, sum / n);
}

Interval paired_window_bootstrap(std::span<const double> sums_a, std::span<const double> sums_b,
                                 std::span<const std::uint64_t> window_counts,
                                 std::uint32_t resamples, std::uint64_t seed) {
    if (sums_a.size() != sums_b.size()) {
        throw std::invalid_argument("paired bootstrap needs the same windows in both arms");
    }
    std::vector<double> difference(sums_a.size());
    for (std::size_t i = 0; i < difference.size(); ++i) { difference[i] = sums_a[i] - sums_b[i]; }
    return window_bootstrap(difference, window_counts, resamples, seed);
}

void Fnv64::add(const void* data, std::size_t bytes) noexcept {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < bytes; ++i) {
        state_ ^= p[i];
        state_ *= 1099511628211ULL;
    }
}

void hash_bf16_logits(Fnv64& hash, std::span<const float> logits) {
    for (const float value : logits) {
        const auto raw = std::bit_cast<std::uint32_t>(value);
        if ((raw & 0xffffU) != 0) {
            throw std::runtime_error("public BF16 head returned an unrounded logit");
        }
        const auto bits = static_cast<std::uint16_t>(raw >> 16);
        hash.add_value(bits);
    }
}

CacheWriter::CacheWriter(const std::filesystem::path& path, std::int32_t vocab,
                         const nlohmann::json& protocol, const nlohmann::json& reference)
    : path_(path), vocab_(vocab) {
    if (vocab <= 0) { throw std::invalid_argument("KL cache vocab must be positive"); }
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("KL reference cache already exists: " + path.string());
    }
    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_) { throw std::runtime_error("cannot create KL reference cache: " + path.string()); }
    const std::string json =
        nlohmann::json{{"protocol", protocol}, {"reference", reference}}.dump();
    json_bytes_  = json.size();
    data_offset_ = align_up(kFixedHeader + json_bytes_, kCacheAlign);
    write_header(file_, static_cast<std::uint32_t>(vocab), 0, json_bytes_, data_offset_);
    file_.write(json.data(), static_cast<std::streamsize>(json.size()));
    const std::vector<char> pad(data_offset_ - kFixedHeader - json_bytes_, 0);
    file_.write(pad.data(), static_cast<std::streamsize>(pad.size()));
    if (!file_) { throw std::runtime_error("cannot write KL cache header: " + path.string()); }
}

void CacheWriter::append(std::span<const float> rows, std::int32_t columns) {
    if (finished_ || columns < 0 ||
        rows.size() != static_cast<std::size_t>(columns) * static_cast<std::size_t>(vocab_)) {
        throw std::invalid_argument("KL cache append has the wrong shape");
    }
    file_.write(reinterpret_cast<const char*>(rows.data()),
                static_cast<std::streamsize>(rows.size() * sizeof(float)));
    if (!file_) { throw std::runtime_error("cannot write KL cache rows: " + path_.string()); }
    positions_ += static_cast<std::uint64_t>(columns);
}

void CacheWriter::finish() {
    if (finished_) { return; }
    if (positions_ == 0) { throw std::runtime_error("KL cache holds no positions"); }
    file_.flush();
    file_.seekp(0);
    write_header(file_, static_cast<std::uint32_t>(vocab_), positions_, json_bytes_, data_offset_);
    file_.flush();
    if (!file_) { throw std::runtime_error("cannot finalize KL cache: " + path_.string()); }
    file_.close();
    finished_ = true;
}

CacheReader::CacheReader(const std::filesystem::path& path, std::int32_t vocab,
                         const nlohmann::json& expected_protocol)
    : file_(path, std::ios::binary), vocab_(vocab) {
    if (vocab <= 0) { throw std::invalid_argument("KL cache vocab must be positive"); }
    if (!file_) { throw std::runtime_error("cannot open KL reference cache: " + path.string()); }
    char magic[8]{};
    std::uint32_t version   = 0;
    std::uint32_t cache_vocab = 0;
    std::uint64_t json_bytes  = 0;
    std::uint64_t data_offset = 0;
    file_.read(magic, sizeof(magic));
    file_.read(reinterpret_cast<char*>(&version), sizeof(version));
    file_.read(reinterpret_cast<char*>(&cache_vocab), sizeof(cache_vocab));
    file_.read(reinterpret_cast<char*>(&positions_), sizeof(positions_));
    file_.read(reinterpret_cast<char*>(&json_bytes), sizeof(json_bytes));
    file_.read(reinterpret_cast<char*>(&data_offset), sizeof(data_offset));
    if (!file_ || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("not a KL reference cache: " + path.string());
    }
    if (version != kCacheVersion) {
        throw std::runtime_error("unsupported KL cache version " + std::to_string(version));
    }
    if (positions_ == 0) {
        throw std::runtime_error("KL reference cache was never finished (positions=0): " +
                                 path.string());
    }
    if (static_cast<std::int32_t>(cache_vocab) != vocab) {
        throw std::runtime_error("KL cache vocab " + std::to_string(cache_vocab) +
                                 " does not match the engine vocab " + std::to_string(vocab));
    }
    const std::uint64_t file_bytes = std::filesystem::file_size(path);
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(vocab) * sizeof(float);
    if (data_offset < kFixedHeader || data_offset % kCacheAlign != 0 ||
        data_offset > file_bytes || json_bytes > data_offset - kFixedHeader ||
        json_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::streamsize>::max()) ||
        data_offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        positions_ > (std::numeric_limits<std::uint64_t>::max() - data_offset) / row_bytes ||
        file_bytes - data_offset != positions_ * row_bytes) {
        throw std::runtime_error("KL reference cache is truncated or corrupt: " + path.string());
    }
    std::string json(json_bytes, '\0');
    file_.read(json.data(), static_cast<std::streamsize>(json_bytes));
    if (!file_) { throw std::runtime_error("cannot read KL cache header json"); }
    const nlohmann::json header = nlohmann::json::parse(json);
    const nlohmann::json& protocol = header.at("protocol");
    if (!protocol.is_object() || !expected_protocol.is_object()) {
        throw std::runtime_error("KL reference cache protocol must be an object");
    }
    if (protocol != expected_protocol) {
        throw std::runtime_error("KL reference cache protocol mismatch: " +
                                 describe_protocol_difference(expected_protocol, protocol));
    }
    reference_ = header.at("reference");
    file_.seekg(static_cast<std::streamoff>(data_offset));
    if (!file_) { throw std::runtime_error("cannot seek to KL reference cache rows"); }
}

void CacheReader::read(std::int32_t columns, std::span<float> rows) {
    if (columns < 0 ||
        rows.size() != static_cast<std::size_t>(columns) * static_cast<std::size_t>(vocab_)) {
        throw std::invalid_argument("KL cache read has the wrong shape");
    }
    if (consumed_ + static_cast<std::uint64_t>(columns) > positions_) {
        throw std::runtime_error("this run scores more positions than the KL reference cache holds");
    }
    file_.read(reinterpret_cast<char*>(rows.data()),
               static_cast<std::streamsize>(rows.size() * sizeof(float)));
    if (!file_) { throw std::runtime_error("cannot read KL reference cache rows"); }
    consumed_ += static_cast<std::uint64_t>(columns);
}

namespace {

nlohmann::json group_stats(const PositionStats& stats, std::span<const WindowRecord> windows,
                           std::span<const std::size_t> members, std::uint32_t resamples,
                           std::uint64_t seed) {
    std::vector<double> kl_sums;
    std::vector<double> dnll_sums;
    std::vector<std::uint64_t> counts;
    std::vector<float> kl_values;
    std::uint64_t positions = 0;
    std::uint64_t disagree  = 0;
    for (const std::size_t index : members) {
        const WindowRecord& window = windows[index];
        double kl_sum              = 0.0;
        double dnll_sum            = 0.0;
        for (std::uint64_t i = window.begin; i < window.begin + window.count; ++i) {
            kl_sum += static_cast<double>(stats.kl[i]);
            // Change in target NLL: (-cand_lp) - (-ref_lp).
            dnll_sum += static_cast<double>(stats.ref_lp[i]) - static_cast<double>(stats.cand_lp[i]);
            kl_values.push_back(stats.kl[i]);
            disagree += stats.top1_agree[i] == 0 ? 1U : 0U;
        }
        kl_sums.push_back(kl_sum);
        dnll_sums.push_back(dnll_sum);
        counts.push_back(window.count);
        positions += window.count;
    }
    const Interval kl_interval   = window_bootstrap(kl_sums, counts, resamples, seed);
    const Interval dnll_interval = window_bootstrap(dnll_sums, counts, resamples, seed + 1);
    return nlohmann::json{
        {"positions", positions},
        {"windows", members.size()},
        {"mean_kl", kl_interval.mean},
        {"mean_kl_ci95", {kl_interval.low, kl_interval.high}},
        {"median_kl", percentile(kl_values, 0.5)},
        {"p95_kl", percentile(kl_values, 0.95)},
        {"p99_kl", percentile(kl_values, 0.99)},
        {"p999_kl", percentile(kl_values, 0.999)},
        {"mean_delta_nll", dnll_interval.mean},
        {"mean_delta_nll_ci95", {dnll_interval.low, dnll_interval.high}},
        {"top1_disagreement", static_cast<double>(disagree) / static_cast<double>(positions)},
    };
}

} // namespace

nlohmann::json summarize(const PositionStats& stats, std::span<const WindowRecord> windows,
                         std::uint32_t resamples, std::uint64_t seed) {
    const std::size_t positions = stats.kl.size();
    if (stats.ref_lp.size() != positions || stats.cand_lp.size() != positions ||
        stats.top1_agree.size() != positions || windows.empty()) {
        throw std::invalid_argument("KL summary needs equal per-position vectors and windows");
    }
    std::uint64_t covered = 0;
    for (const WindowRecord& window : windows) {
        if (window.begin != covered || window.count == 0 || window.begin + window.count > positions) {
            throw std::invalid_argument("KL windows must tile the positions contiguously");
        }
        covered += window.count;
    }
    if (covered != positions) {
        throw std::invalid_argument("KL windows do not cover every position");
    }
    std::vector<std::size_t> all(windows.size());
    std::iota(all.begin(), all.end(), std::size_t{0});
    std::vector<std::string> domain_names;
    for (const WindowRecord& window : windows) {
        if (std::find(domain_names.begin(), domain_names.end(), window.domain) == domain_names.end()) {
            domain_names.push_back(window.domain);
        }
    }
    std::sort(domain_names.begin(), domain_names.end());
    nlohmann::json report;
    report["bootstrap"] = {{"resamples", resamples}, {"seed", seed}, {"unit", "window"}};
    report["overall"]   = group_stats(stats, windows, all, resamples, seed);
    report["domains"]   = nlohmann::json::array();
    for (const std::string& name : domain_names) {
        std::vector<std::size_t> members;
        for (std::size_t i = 0; i < windows.size(); ++i) {
            if (windows[i].domain == name) { members.push_back(i); }
        }
        nlohmann::json item = group_stats(stats, windows, members, resamples, seed);
        item["domain"]      = name;
        report["domains"].push_back(std::move(item));
    }
    report["windows"] = nlohmann::json::array();
    for (const WindowRecord& window : windows) {
        double kl_sum   = 0.0;
        double dnll_sum = 0.0;
        for (std::uint64_t i = window.begin; i < window.begin + window.count; ++i) {
            kl_sum += static_cast<double>(stats.kl[i]);
            dnll_sum += static_cast<double>(stats.ref_lp[i]) - static_cast<double>(stats.cand_lp[i]);
        }
        report["windows"].push_back({{"stream", window.stream},
                                     {"domain", window.domain},
                                     {"begin", window.begin},
                                     {"count", window.count},
                                     {"kl_sum", kl_sum},
                                     {"delta_nll_sum", dnll_sum}});
    }
    return report;
}

} // namespace infernix::perplexity::kl
