// Gemma 4 31B projections and output heads in the row-scaled FP8 profile. The trunk projections
// take their own A16 routes for 17..32 tokens (gemma4_a16.h) and admit A8 from 33 tokens
// (gemma4_a8.h). Both vocabulary heads and the drafter run the dense A16 routes.
#include "ops/linear/fp8/fp8_dense_a16.cuh"
#include "ops/linear/fp8/shapes/gemma4_a16.h"
#include "ops/linear/fp8/shapes/gemma4_a8.h"

namespace infernix::ops::detail {
namespace {
template <int N, int K>
constexpr Fp8LinearShape trunk() {
    Fp8LinearShape shape         = fp8_dense::shape<N, K>();
    shape.a16                    = gemma4_fp8::launch_a16<N, K>;
    shape.a8                     = gemma4_fp8::launch_a8<N, K>;
    shape.uses_a8                = gemma4_fp8::uses_a8;
    shape.partial_capacity_bytes = gemma4_fp8::partial_capacity_bytes;
    return shape;
}
} // namespace

const Fp8LinearShape kFp8N16384K5376  = trunk<16384, 5376>();
const Fp8LinearShape kFp8N18432K5376  = trunk<18432, 5376>();
const Fp8LinearShape kFp8N43008K5376  = trunk<43008, 5376>();
const Fp8LinearShape kFp8N5376K8192   = trunk<5376, 8192>();
const Fp8LinearShape kFp8N5376K16384  = trunk<5376, 16384>();
const Fp8LinearShape kFp8N5376K21504  = trunk<5376, 21504>();
const Fp8LinearShape kFp8N262144K5376 = fp8_dense::shape<262144, 5376>();
const Fp8LinearShape kFp8N262144K1024 = fp8_dense::shape<262144, 1024>();
const Fp8LinearShape kFp8N1024K10752  = fp8_dense::shape<1024, 10752>();
const Fp8LinearShape kFp8N5376K1024   = fp8_dense::shape<5376, 1024>();
const Fp8LinearShape kFp8N8192K1024   = fp8_dense::shape<8192, 1024>();
const Fp8LinearShape kFp8N16384K1024  = fp8_dense::shape<16384, 1024>();
const Fp8LinearShape kFp8N1024K8192   = fp8_dense::shape<1024, 8192>();
const Fp8LinearShape kFp8N1024K16384  = fp8_dense::shape<1024, 16384>();
} // namespace infernix::ops::detail
