#include "ops/linear/fp8/shapes/gemma4_a8.h"

#include "ops/linear/fp8/fp8_launch.cuh"

namespace infernix::ops::detail::gemma4_fp8 {
namespace {
// Tile choice is per shape, from the cold bench. The 256-row bulk is the default.
using Narrow = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Mid    = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 128, 128, 2, 4, 2, 1>, 170, 4, 8>;
using Bulk   = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<128, 256, 128, 2, 4, 2, 1>, 170, 4, 8>;
using NarrowSplit = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>, 188, 8, 8>;
using Thin   = Fp8A8TmaMmaSchedule<32, 64, 128, 1, 2, 3, 2>;
// Up to 32 tokens the A16 kernels read the weights at full bandwidth; from 33 they slow down
// and A8 is at least as fast on every trunk shape.
constexpr std::int32_t kA8MinTokens = 33;
} // namespace

template <int N, int K>
void launch_a8_impl(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
                    cudaStream_t stream, bool quantize) {
    using Geometry = Fp8Geometry<N, K>;
    const int tokens = x.ne[1];
    // Below 128 tokens (tmp/infernix/gemma4/fp8short/sweep.csv): the 64-token tile on the wide
    // shapes; on the 5376-row shapes too few CTAs fill the GPU, so the 32x64 tile up to 64
    // tokens and the 64-token tile split over K above that.
    if (tokens < 128) {
        if constexpr (N == 5376) {
            if (tokens <= 64) {
                return launch_fp8_a8_tma<Geometry, Thin>(x, weight, out, scratch, stream, quantize);
            }
            return launch_fp8_a8_tma<Geometry, NarrowSplit>(x, weight, out, scratch, stream,
                                                            quantize);
        } else {
            return launch_fp8_a8_tma<Geometry, Narrow>(x, weight, out, scratch, stream, quantize);
        }
    }
    // Cold-bench on this GPU (tmp/infernix/gemma4/nsys/speed/linear_a8_split.txt): the 64-token
    // tile wins at 128 tokens except on 43008x5376 and 5376x21504, and the 128-row tile wins
    // at 512 tokens when N is below 18000. The 256-row bulk stays fastest everywhere else,
    // including every shape at 4096.
    const bool narrow_wins = N != 43008 && !(N == 5376 && (K == 16384 || K == 21504));
    if (tokens <= 128 && narrow_wins) {
        return launch_fp8_a8_tma<Geometry, Narrow>(x, weight, out, scratch, stream, quantize);
    }
    if (tokens > 128 && tokens <= 512 && N < 18000) {
        return launch_fp8_a8_tma<Geometry, Mid>(x, weight, out, scratch, stream, quantize);
    }
    launch_fp8_a8_tma<Geometry, Bulk>(x, weight, out, scratch, stream, quantize);
}

template <int N, int K>
void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    launch_a8_impl<N, K>(x, weight, out, scratch, stream, true);
}

void launch_down_ready(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
                       cudaStream_t stream) {
    launch_a8_impl<5376, 21504>(x, weight, out, scratch, stream, false);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= kA8MinTokens; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    // Sized for the bulk tile, which is the largest partial any of these schedules writes.
    static_assert(Bulk::kPartialBytes >= NarrowSplit::kPartialBytes);
    return max_tokens >= kA8MinTokens ? Bulk::kPartialBytes : 0;
}

#define INFERNIX_GEMMA4_FP8_A8(N, K)                                                                 \
    template void launch_a8<N, K>(const Tensor&, const Weight&, Tensor&, Fp8A8Workspace,           \
                                  cudaStream_t);
INFERNIX_GEMMA4_FP8_A8(16384, 5376)
INFERNIX_GEMMA4_FP8_A8(18432, 5376)
INFERNIX_GEMMA4_FP8_A8(43008, 5376)
INFERNIX_GEMMA4_FP8_A8(5376, 8192)
INFERNIX_GEMMA4_FP8_A8(5376, 16384)
INFERNIX_GEMMA4_FP8_A8(5376, 21504)
#undef INFERNIX_GEMMA4_FP8_A8

} // namespace infernix::ops::detail::gemma4_fp8
