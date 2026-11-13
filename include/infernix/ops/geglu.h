#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace infernix::ops {

/**
 * GeGLU with the tanh GELU approximation:
 *
 *   ideal[i] = gelu_tanh(gate[i]) * up[i]
 *   gelu_tanh(z) = 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3))).
 *
 * `gate`, `up`, and `out` are same-shaped BF16 tensors. out is contiguous; gate and up may use
 * arbitrary valid Tensor strides. out must not overlap either input (the two read-only inputs may
 * overlap one another). The oracle evaluates `ideal` in FP64 from the represented inputs. The BF16
 * output is promoted and compared directly with that result; output storage rounding belongs to
 * the Op's numerical criterion, not the oracle. The Op writes all of out and uses no workspace.
 */
// fp8_codes and fp8_scales, when both set, receive the per-token E4M3 row quantize of `out`
// (scale = absmax/448) from the same BF16 values this call stores. Both are null, or both are
// set to one column of `out.ne[0]` codes and one float scale per column.
void geglu(const Tensor& gate, const Tensor& up, Tensor& out, cudaStream_t stream,
           std::uint8_t* fp8_codes = nullptr, float* fp8_scales = nullptr);

} // namespace infernix::ops
