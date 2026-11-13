# Gemma 4 31B FP8 row quantize, cuBLAS FP8 row, and the ready-codes A8 down projection. Included after
# tests/ops/linear/tests.cmake, which defines infernix_linear_test_support. The Gemma FP8 trunk, head and
# draft shapes under ops::linear are covered by tests/ops/linear/test_gemma4_shapes.cpp (gemma4_q4_tests.cmake).
set(GEMMA4_FP8_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}/linear")

# BF16 rows to row-scaled E4M3: zero row, scale rule and reconstruction bound; an FP8 Weight is left alone.
infernix_add_family_op_test(infernix_linear_fp8_row_quantize_test FAMILY linear
  SOURCES "${GEMMA4_FP8_TEST_DIR}/test_fp8_row_quantize.cpp"
  LIBRARIES infernix_linear_test_support)

# fp8_row_cublas (E4M3 widened to BF16, cuBLAS FP32 accumulation, row scale applied once) against the FP64
# oracle at the A16 criterion.
infernix_add_family_op_test(infernix_linear_fp8_row_cublas_test FAMILY linear
  SOURCES "${GEMMA4_FP8_TEST_DIR}/test_fp8_row_cublas.cpp"
  LIBRARIES infernix_linear_test_support)

# The A8 route of the 5376x21504 down projection with the activation codes and scales written by the
# producer (launch_fp8_a8_tma quantize=false), at the four tile bands, against the FP64 oracle.
infernix_add_family_op_test(infernix_linear_gemma4_fp8_ready_test FAMILY linear
  SOURCES "${GEMMA4_FP8_TEST_DIR}/test_gemma4_fp8_ready.cpp"
  LIBRARIES infernix_linear_test_support)
