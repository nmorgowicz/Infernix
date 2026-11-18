#pragma once

#include "core/arena.h"
#include "infernix/ops/linear.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infernix::ops::detail {

// The opt-in INT8-activation route of the six Gemma 4 31B trunk Q5 projections (LinearRoutes::Q5Int8G32 /
// Q5Int8G64): from this width on a call runs on exact stored codes and per-group INT8 activations; narrower calls
// (decode, MTP verification) keep the native A16 routes.
inline constexpr std::int32_t kQ5Int8MinTokens = 64;

// 32 or 64 for LinearRoutes::Q5Int8G32 / Q5Int8G64, else 0.
[[nodiscard]] constexpr int q5_int8_route_group(LinearRoutes routes) noexcept {
    return has_route(routes, LinearRoutes::Q5Int8G32) ? 32 : has_route(routes, LinearRoutes::Q5Int8G64) ? 64 : 0;
}

// One of the six registered trunk shapes (Parameters::geometry's matrices).
[[nodiscard]] bool q5_int8_prefill_shape(std::int32_t n, std::int32_t k);

// True for a native Q5_G64_FP16 RowSplit matrix of a registered trunk shape called with at least kQ5Int8MinTokens
// columns.
[[nodiscard]] bool q5_int8_prefill_supported(const Weight& w, std::int32_t tokens);

// Transient bytes of a call of up to `tokens` columns of K = `k`: the INT8 activations and their group scales,
// padded to whole 128-token tiles, plus the arena's two alignments. `group` is 32 or 64.
[[nodiscard]] std::size_t q5_int8_prefill_workspace_bytes(std::int32_t k, std::int32_t tokens, int group);

// out[N,T] = W[N,K] x[K,T]. `workspace` supplies the transient storage inside its own scope.
void q5_int8_prefill(const Tensor& x, const Weight& w, Tensor& out, int group, WorkspaceArena& workspace,
                     cudaStream_t stream);

} // namespace infernix::ops::detail
