#include "ops/linear/q5/q5_int8_prefill.h"

#include "core/device.h"
#include "ops/linear/q5/q5_int8_prefill.cuh"

#include <array>
#include <stdexcept>
#include <string>

namespace infernix::ops::detail {
namespace {

constexpr int kTokenTile = 128;
constexpr std::size_t kArenaAlignment = 256;

// One tile shape per width class, the same for both group sizes: 128 x 128 for the wide ring widths, 64 x 128 with a
// third stage for the 64..128 token tail. The phase-1 microbenchmark's sweep found the best single shape of a class
// within 1 % (T = 2048) and 3 % (T = 128) of the per-shape best; no further tile tuning is done here.
template <int Group>
using WideConfig = Q5Int8Config<128, kTokenTile, 64, 32, 2, 4, Group>;
template <int Group>
using NarrowConfig = Q5Int8Config<64, kTokenTile, 32, 32, 3, 4, Group>;

struct Shape {
    std::int32_t n, k;
};
// Parameters::geometry's six matrices: sliding Q/K/V, global Q/K (V is K), gate/up, and the three outputs.
constexpr std::array kShapes{Shape{16384, 5376}, Shape{18432, 5376}, Shape{43008, 5376},
                             Shape{5376, 8192},  Shape{5376, 16384}, Shape{5376, 21504}};

constexpr std::int64_t round_up(std::int64_t value, std::int64_t unit) { return (value + unit - 1) / unit * unit; }

template <class Cfg>
void launch_gemm(const std::int8_t* xq, const Q5Int8Meta* meta, const Weight& w, __nv_bfloat16* out, int tokens,
                 int padded_tokens, cudaStream_t stream) {
    static const cudaError_t reserved = cudaFuncSetAttribute(q5_int8_prefill_kernel<Cfg>,
                                                            cudaFuncAttributeMaxDynamicSharedMemorySize, Cfg::kSharedBytes);
    CUDA_CHECK(reserved);
    const dim3 grid(static_cast<unsigned>(padded_tokens / Cfg::BN), static_cast<unsigned>(w.n / Cfg::BM));
    q5_int8_prefill_kernel<Cfg><<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(
        xq, meta, static_cast<const std::uint8_t*>(w.qdata), static_cast<const std::uint8_t*>(w.qhigh),
        static_cast<const std::uint8_t*>(w.scales), out, w.n, tokens, padded_tokens, w.k, w.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <int Group>
void run(const Weight& w, Tensor& out, const std::int8_t* xq, const Q5Int8Meta* meta, int tokens, int padded_tokens,
         cudaStream_t stream) {
    if (tokens <= 128) {
        launch_gemm<NarrowConfig<Group>>(xq, meta, w, static_cast<__nv_bfloat16*>(out.data), tokens, padded_tokens, stream);
    } else {
        launch_gemm<WideConfig<Group>>(xq, meta, w, static_cast<__nv_bfloat16*>(out.data), tokens, padded_tokens, stream);
    }
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

} // namespace

bool q5_int8_prefill_shape(std::int32_t n, std::int32_t k) {
    for (const Shape& shape : kShapes) {
        if (shape.n == n && shape.k == k) { return true; }
    }
    return false;
}

bool q5_int8_prefill_supported(const Weight& w, std::int32_t tokens) {
    return tokens >= kQ5Int8MinTokens && w.qtype == QType::Q5_G64_FP16 && w.layout == QuantLayout::RowSplit &&
           q5_int8_prefill_shape(w.n, w.k);
}

std::size_t q5_int8_prefill_workspace_bytes(std::int32_t k, std::int32_t tokens, int group) {
    if (k <= 0 || tokens <= 0 || (group != 32 && group != 64)) {
        throw std::invalid_argument("q5 int8 prefill: invalid workspace extents");
    }
    const auto padded = static_cast<std::size_t>(round_up(tokens, kTokenTile));
    const auto codes  = padded * static_cast<std::size_t>(k);
    const auto groups = padded * static_cast<std::size_t>(k / group) * sizeof(Q5Int8Meta);
    return codes + groups + 2 * kArenaAlignment;
}

void q5_int8_prefill(const Tensor& x, const Weight& w, Tensor& out, int group, WorkspaceArena& workspace,
                     cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (!q5_int8_prefill_supported(w, tokens) || (group != 32 && group != 64) || w.group != 64 || w.group_size != 64 ||
        w.scale_dtype != DType::FP16 || w.padded_shape[1] < w.k || w.k % kQ5Int8StageK != 0 || w.n % 128 != 0 ||
        !aligned(w.qdata, 16) || !aligned(w.qhigh, 16) || !aligned(w.scales, 4) || x.dtype != DType::BF16 ||
        out.dtype != DType::BF16 || x.ne[0] != w.k || out.ne[0] != w.n || out.ne[1] != tokens) {
        throw std::invalid_argument("q5 int8 prefill: expected a registered native Q5 trunk matrix and at least " +
                                    std::to_string(kQ5Int8MinTokens) + " BF16 columns");
    }
    const int padded_tokens = static_cast<int>(round_up(tokens, kTokenTile));
    auto scope              = workspace.scope();
    const auto codes        = workspace.alloc_bytes(static_cast<std::size_t>(padded_tokens) * w.k, kArenaAlignment);
    const auto groups       = workspace.alloc_bytes(static_cast<std::size_t>(padded_tokens) * (w.k / group) * sizeof(Q5Int8Meta),
                                                    kArenaAlignment);
    auto* xq   = static_cast<std::int8_t*>(codes.data);
    auto* meta = static_cast<Q5Int8Meta*>(groups.data);
    const dim3 quantize_grid(static_cast<unsigned>(padded_tokens),
                             static_cast<unsigned>((w.k / 8 + kQ5Int8QuantizeThreads - 1) / kQ5Int8QuantizeThreads));
    if (group == 32) {
        q5_int8_quantize_kernel<32><<<quantize_grid, kQ5Int8QuantizeThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), xq, meta, tokens, padded_tokens, w.k);
        CUDA_CHECK(cudaGetLastError());
        run<32>(w, out, xq, meta, tokens, padded_tokens, stream);
    } else {
        q5_int8_quantize_kernel<64><<<quantize_grid, kQ5Int8QuantizeThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), xq, meta, tokens, padded_tokens, w.k);
        CUDA_CHECK(cudaGetLastError());
        run<64>(w, out, xq, meta, tokens, padded_tokens, stream);
    }
}

} // namespace infernix::ops::detail
