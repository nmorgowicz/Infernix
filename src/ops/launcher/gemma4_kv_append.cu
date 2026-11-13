#include "ops/launcher/gemma4_kv_append.h"

#include "core/device.h"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>

namespace infernix::ops::detail {
namespace {

__global__ void gemma4_kv_append_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                        const std::int32_t* positions, __nv_bfloat16* k_pages,
                                        __nv_bfloat16* v_pages, const std::int32_t* block_table,
                                        std::int32_t head_dim, std::int32_t kv_heads,
                                        std::int32_t tokens, std::int32_t logical_pages,
                                        std::int32_t physical_pages) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int count = tokens * kv_heads * head_dim;
    if (index >= count) { return; }
    const int dim   = index % head_dim;
    const int head  = (index / head_dim) % kv_heads;
    const int token = index / (head_dim * kv_heads);
    const int pos   = positions[token];
    if (pos < 0) { return; }
    const int page = pos >> kPagedKVPageShift;
    if (page >= logical_pages) { return; }
    const int physical = block_table[page];
    if (physical < 0 || physical >= physical_pages) { return; }
    const std::int64_t dest =
        static_cast<std::int64_t>(head_dim) * kPagedKVPageSize *
            (static_cast<std::int64_t>(head) + static_cast<std::int64_t>(kv_heads) * physical) +
        static_cast<std::int64_t>(head_dim) * (pos & kPagedKVPageMask) + dim;
    k_pages[dest] = k[index];
    v_pages[dest] = v[index];
}

} // namespace

void gemma4_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                             Tensor& k_pages, Tensor& v_pages, const Tensor& block_table,
                             cudaStream_t stream) {
    const int count = k.ne[0] * k.ne[1] * k.ne[2];
    constexpr int block = 256;
    const int grid = (count + block - 1) / block;
    gemma4_kv_append_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data), static_cast<__nv_bfloat16*>(k_pages.data),
        static_cast<__nv_bfloat16*>(v_pages.data), static_cast<const std::int32_t*>(block_table.data),
        k.ne[0], k.ne[1], k.ne[2], block_table.ne[0], k_pages.ne[3]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace infernix::ops::detail
