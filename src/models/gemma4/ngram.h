#pragma once

#include "infernix/types.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

namespace infernix::models::gemma4 {

// P4 has no speculative archive. Engine bookkeeping may query this truthful absence adapter;
// any attempt to publish or bind an archive request is rejected, never reported as successful.
class NgramArchive {
public:
    struct Limits {
        std::size_t session_bytes = 0;
        std::size_t total_bytes = 0;
    };
    class Request {};
    explicit NgramArchive(Limits) noexcept {}
    bool publish(std::unique_ptr<Request>, std::span<const TokenId>, std::uint32_t) {
        throw std::invalid_argument("gemma4 P4 does not support ngram archives");
    }
    [[nodiscard]] NgramArchiveStats stats(std::string_view) const noexcept { return {}; }
};

} // namespace infernix::models::gemma4
