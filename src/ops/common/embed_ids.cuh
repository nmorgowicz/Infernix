#pragma once

#include "core/device.h" // DeviceFault

#include <cstdint>

namespace infernix::ops {

// A gather's token ids and the table's row count: an id outside [0, vocab) writes zeros for its column
// and stores `fault.value` into `fault.word` when set (core DeviceFault), instead of reading outside the
// table.
struct EmbedIds {
    const std::int32_t* ids;
    std::int32_t vocab;
    DeviceFault fault;

    // ids[t], or -1 (reported) when it lies outside the table.
    __device__ __forceinline__ std::int32_t row(std::int32_t t) const {
        const std::int32_t id = ids[t];
        if (id >= 0 && id < vocab) { return id; }
        if (fault.word != nullptr) { *reinterpret_cast<volatile std::uint32_t*>(fault.word) = fault.value; }
        return -1;
    }
};

} // namespace infernix::ops
