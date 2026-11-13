#include "infernix/ops/topk_columns.h"

#include "ops/launcher/topk_columns.h"

#include <stdexcept>

namespace infernix::ops {

void topk_columns(const Tensor& logits, std::int32_t valid_rows, std::int32_t k, Tensor& ids,
                  cudaStream_t stream) {
    if (logits.dtype != DType::BF16 || ids.dtype != DType::I32) {
        throw std::invalid_argument("topk_columns: logits must be BF16 and ids I32");
    }
    if (logits.ne[2] != 1 || logits.ne[3] != 1 || ids.ne[2] != 1 || ids.ne[3] != 1) {
        throw std::invalid_argument("topk_columns: logits is [rows,T] and ids is [k,T]");
    }
    if (k < 1 || k > 8) { throw std::invalid_argument("topk_columns: k must be in [1,8]"); }
    if (valid_rows <= 0 || valid_rows > logits.ne[0]) {
        throw std::invalid_argument("topk_columns: valid_rows must be in [1, rows]");
    }
    if (ids.ne[0] != k || ids.ne[1] != logits.ne[1]) {
        throw std::invalid_argument("topk_columns: ids shape must be [k,T]");
    }
    if (logits.ne[1] == 0) { return; }
    if (!logits.is_contiguous() || !ids.is_contiguous() || logits.data == nullptr ||
        ids.data == nullptr) {
        throw std::invalid_argument("topk_columns: tensors must be contiguous and non-null");
    }
    detail::topk_columns_launch(logits, valid_rows, k, ids, stream);
}

} // namespace infernix::ops
