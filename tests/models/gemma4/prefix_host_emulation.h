#pragma once

// Host-memory emulation of the Gemma4 prefix cache's Device side, for the CPU-only store test: the
// sliding rings as the kernels address them ([physical page][head][64 tokens][row]), a Host image over
// separate slab allocations, RingImage::copy_group's piece construction (the pure seam of its copies,
// whose cudaMemcpyAsync cannot run without the driver), a replay of planned CopyOps with memcpy, and
// an oracle image built from the documented layout (state_image.h) without using the planner.

#include "models/gemma4/program/prefix/state_image.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace prefix_test {

namespace sp = infernix::models::gemma4::prefix;

using Bytes = std::vector<std::byte>;
inline constexpr std::uint32_t kPage = 64;

inline std::uint64_t mix(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline void fill_row(std::byte* row, std::size_t bytes, std::uint64_t seed) {
    for (std::size_t i = 0; i + 8 <= bytes; i += 8) {
        const std::uint64_t v = mix(seed + i);
        std::memcpy(row + i, &v, 8);
    }
}

// One token row of one head of one ring plane at one logical position.
inline std::uint64_t row_seed(std::uint32_t group, std::uint32_t plane, std::uint32_t head,
                              std::uint32_t position, std::uint32_t salt) {
    return mix((static_cast<std::uint64_t>(group) << 56) ^ (static_cast<std::uint64_t>(plane) << 52) ^
               (static_cast<std::uint64_t>(head) << 40) ^ (static_cast<std::uint64_t>(salt) << 32) ^ position);
}

// Ring planes as RingImage addresses them.
struct Rings {
    std::vector<std::vector<Bytes>> storage; // [group][plane]
    std::vector<std::vector<sp::DevicePlane>> planes;
    std::uint32_t ring_pages = 0;

    Rings(const sp::StateImageLayout& layout, std::uint32_t pages) : ring_pages(pages) {
        storage.resize(layout.groups());
        planes.resize(layout.groups());
        for (std::uint32_t g = 0; g < layout.groups(); ++g) {
            for (std::uint32_t i = layout.group_begin[g]; i < layout.group_begin[g + 1]; ++i) {
                storage[g].emplace_back(static_cast<std::size_t>(pages) * layout.parts[i].page_bytes);
                planes[g].push_back({storage[g].back().data(), layout.parts[i].page_bytes});
            }
        }
        reset();
    }
    Rings(const Rings&)            = delete;
    Rings& operator=(const Rings&) = delete;

    void reset() {
        for (auto& group : storage) {
            for (auto& plane : group) { std::fill(plane.begin(), plane.end(), std::byte{0xA5}); }
        }
    }
    // Writes logical `position` of every ring (a later position overwrites its slot, as the model does).
    void write(const sp::StateImageLayout& layout, std::uint32_t position, std::uint32_t salt) {
        for (std::uint32_t g = 0; g < layout.groups(); ++g) {
            for (std::uint32_t i = layout.group_begin[g]; i < layout.group_begin[g + 1]; ++i) {
                const sp::ImagePart& part = layout.parts[i];
                const std::size_t page = (position / kPage) % ring_pages;
                for (std::uint32_t head = 0; head < part.kv_heads; ++head) {
                    std::byte* row = planes[g][part.plane].base + page * part.page_bytes +
                                     (static_cast<std::size_t>(head) * kPage + position % kPage) * part.row_bytes;
                    fill_row(row, part.row_bytes, row_seed(g, part.plane, head, position, salt));
                }
            }
        }
    }
    bool operator==(const Rings& other) const { return storage == other.storage; }
};

// A Host image over separate slab allocations: no two segments are adjacent in memory.
struct Slabs {
    std::vector<Bytes> slabs;
    std::vector<std::byte*> pointers;
    std::size_t slab_bytes;
    Slabs(std::size_t bytes, std::uint32_t count) : slabs(count), slab_bytes(bytes) {
        for (auto& slab : slabs) {
            slab.assign(bytes, std::byte{0});
            pointers.push_back(slab.data());
        }
    }
    void zero() { for (auto& slab : slabs) { std::fill(slab.begin(), slab.end(), std::byte{0}); } }
    [[nodiscard]] sp::HostImage image() const { return {pointers, slab_bytes}; }
    [[nodiscard]] Bytes flatten(std::size_t bytes) const {
        Bytes out;
        for (const auto& slab : slabs) { out.insert(out.end(), slab.begin(), slab.end()); }
        out.resize(bytes);
        return out;
    }
};

// RingImage::copy_group's piece construction for one part.
inline std::vector<sp::CopyPiece> part_pieces(const sp::ImagePart& part, const sp::DevicePlane& plane,
                                              std::span<const sp::WindowRun> runs, const sp::HostImage& image) {
    std::vector<sp::CopyPiece> pieces;
    const std::size_t head_stride = static_cast<std::size_t>(kPage) * part.row_bytes;
    for (const sp::WindowRun& run : runs) {
        const std::size_t run_bytes = static_cast<std::size_t>(run.tokens) * part.row_bytes;
        for (std::uint32_t head = 0; head < part.kv_heads; ++head) {
            std::byte* device = plane.base + static_cast<std::size_t>(run.physical_page) * plane.page_bytes +
                                head * head_stride + static_cast<std::size_t>(run.first_row) * part.row_bytes;
            const std::size_t offset = part.offset +
                                       static_cast<std::size_t>(part.kv_heads) * run.first * part.row_bytes +
                                       head * run_bytes;
            sp::append_image_pieces(image, offset, run_bytes, device, pieces);
        }
    }
    return pieces;
}

// Replays planned copies on host memory: save = Device -> Host.
inline void replay(std::span<const sp::CopyOp> ops, bool save) {
    for (const sp::CopyOp& op : ops) {
        for (std::size_t i = 0; i < op.height; ++i) {
            std::byte* device = op.device + i * op.device_pitch;
            std::byte* host   = op.host + i * op.host_pitch;
            if (save) { std::memcpy(host, device, op.width); } else { std::memcpy(device, host, op.width); }
        }
    }
}

// Gathers (save) or scatters (restore) every part of the image for the window [start, frontier).
inline void transfer(const sp::StateImageLayout& layout, Rings& rings, const sp::HostImage& image,
                     std::uint32_t start, std::uint32_t frontier, bool save) {
    const auto runs = sp::window_runs(start, frontier, rings.ring_pages);
    for (const sp::ImagePart& part : layout.parts) {
        const auto ops = sp::plan_copy_ops(part_pieces(part, rings.planes[part.group][part.plane], runs, image));
        replay(ops, save);
    }
}

// The image the documented layout prescribes: header, then per part and per run [head][tokens][row];
// every byte outside the runs (alignment, unused tail) zero. Runs are cut here independently.
inline Bytes expected_image(const sp::StateImageLayout& layout, const sp::LayoutIdentity& id,
                            const sp::StateImageHeader& header, std::uint32_t salt) {
    Bytes flat(layout.image_bytes);
    sp::write_state_image_header(id, header, flat.data());
    for (const sp::ImagePart& part : layout.parts) {
        for (std::uint32_t p = header.image_start; p < header.frontier;) {
            const std::uint32_t end = std::min((p / kPage + 1) * kPage, header.frontier);
            const std::uint32_t n = end - p, i0 = p - header.image_start;
            for (std::uint32_t head = 0; head < part.kv_heads; ++head) {
                for (std::uint32_t t = 0; t < n; ++t) {
                    const std::size_t at = part.offset +
                                           static_cast<std::size_t>(part.kv_heads) * i0 * part.row_bytes +
                                           (static_cast<std::size_t>(head) * n + t) * part.row_bytes;
                    fill_row(flat.data() + at, part.row_bytes, row_seed(part.group, part.plane, head, p + t, salt));
                }
            }
            p = end;
        }
    }
    return flat;
}

} // namespace prefix_test
