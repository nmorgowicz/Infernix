#include "ops/linear/fp8/shapes/gemma4_a16.h"

#include "ops/linear/fp8/fp8_a16_wide_gemv.cuh"
#include "ops/linear/fp8/fp8_dense_a16.cuh"

namespace infernix::ops::detail::gemma4_fp8 {
namespace {
template <class G, int Tile, int RowBlocks, int Warps>
void wide(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_fp8_a16_wide_gemv<G, Fp8A16WideSchedule<Tile, RowBlocks, Warps>>(x, weight, out,
                                                                            stream);
}
} // namespace

// Cold-bench choices per shape and token band (17..127; decode and MTP verify at <= 16 tokens
// keep the generic routes and their bits). A route is kept where it is at least 10% faster than
// the generic one or reaches 85% of read bandwidth. Above 64 tokens the wide GEMV's larger tiles
// spill, so the generic routes stay, except the 4-way K-split GEMM on 16384x5376 up to 96.
template <int N, int K>
void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using G     = Fp8Geometry<N, K>;
    const int t = x.ne[1];
    if constexpr (K == 5376) {
        if (t >= 17 && t <= 24 && N != 16384) { return wide<G, 32, 2, 4>(x, weight, out, stream); }
        if (t >= 25 && t <= 32) { return wide<G, 32, 4, 4>(x, weight, out, stream); }
        if (t >= 33 && t <= 48 && N == 16384) { return wide<G, 64, 2, 4>(x, weight, out, stream); }
        if (t >= 33 && t <= 64) { return wide<G, 64, 4, 4>(x, weight, out, stream); }
        if (t >= 65 && t <= 96 && N == 16384) {
            return fp8_dense::launch_prefill_split<G, false, 4>(x, weight, out, stream);
        }
    } else if constexpr (K == 8192) {
        if (t >= 25 && t <= 32) { return wide<G, 32, 4, 4>(x, weight, out, stream); }
        if (t >= 33 && t <= 64) { return wide<G, 64, 2, 4>(x, weight, out, stream); }
    } else if constexpr (K == 16384) {
        if (t >= 17 && t <= 32) { return wide<G, 32, 4, 4>(x, weight, out, stream); }
        if (t >= 33 && t <= 56) { return wide<G, 64, 2, 4>(x, weight, out, stream); }
    } else {
        if (t >= 17 && t <= 32) { return wide<G, 32, 2, 8>(x, weight, out, stream); }
        if (t >= 33 && t <= 56) { return wide<G, 64, 2, 4>(x, weight, out, stream); }
    }
    fp8_dense::launch_a16<G>(x, weight, out, stream);
}

#define INFERNIX_GEMMA4_FP8_A16(N, K)                                                                \
    template void launch_a16<N, K>(const Tensor&, const Weight&, Tensor&, cudaStream_t);
INFERNIX_GEMMA4_FP8_A16(16384, 5376)
INFERNIX_GEMMA4_FP8_A16(18432, 5376)
INFERNIX_GEMMA4_FP8_A16(43008, 5376)
INFERNIX_GEMMA4_FP8_A16(5376, 8192)
INFERNIX_GEMMA4_FP8_A16(5376, 16384)
INFERNIX_GEMMA4_FP8_A16(5376, 21504)
#undef INFERNIX_GEMMA4_FP8_A16

} // namespace infernix::ops::detail::gemma4_fp8
