# Gemma 4 31B q4_g32 / q4_g128 / q4_g256 (FP16 scales) linear tests. Included after tests/ops/linear/tests.cmake,
# which defines infernix_linear_test_support.
set(GEMMA4_Q4_TEST_DIR "${CMAKE_CURRENT_LIST_DIR}/linear")

# g32 A16 GEMV / tensor-core / wide kernels against the FP64 oracle, the fused/baked W4A8 prefill against
# the host bake oracle, and the ready-codes call.
infernix_add_family_op_test(infernix_linear_q4_g32_a16_test FAMILY linear
  SOURCES "${GEMMA4_Q4_TEST_DIR}/test_q4_g32_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_q4_coarse_a16_test FAMILY linear
  SOURCES "${GEMMA4_Q4_TEST_DIR}/test_q4_coarse_a16.cpp"
  LIBRARIES infernix_linear_test_support)

infernix_add_family_op_test(infernix_linear_q4_coarse_a8_test FAMILY linear
  SOURCES "${GEMMA4_Q4_TEST_DIR}/test_q4_coarse_a8.cpp"
  LIBRARIES infernix_linear_test_support)

# The same test with INFERNIX_TMP_Q4_EXACT: the exact route on the trunk shapes, deterministic.
add_test(NAME infernix_linear_q4_coarse_a8_exact_test COMMAND ${infernix_linear_q4_coarse_a8_test_COMMAND})
set_tests_properties(infernix_linear_q4_coarse_a8_exact_test PROPERTIES
  ENVIRONMENT "INFERNIX_TMP_Q4_EXACT=1" SKIP_RETURN_CODE 77)

# Gemma 31B trunk shapes: NVFP4, FP8 and Q4 g32/g128/g256 A16 and A8 against the FP64 oracle.
infernix_add_family_op_test(infernix_linear_gemma4_shapes_test FAMILY linear
  SOURCES "${GEMMA4_Q4_TEST_DIR}/test_gemma4_shapes.cpp"
  LIBRARIES infernix_linear_test_support)
