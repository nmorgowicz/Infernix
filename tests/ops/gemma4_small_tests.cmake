# Gemma 4 31B small-op tests (new files only; the Gemma cases of test_embedding, test_rmsnorm and
# test_rope live in those tests, registered by tests/ops/tests.cmake). Included after tests/ops/tests.cmake.
set(infernix_gemma4_small_op_tests
  geglu
  gemma4_qkv_prep
  gemma4_residual_norm
  logit_softcap
  residual_scale)
foreach(op IN LISTS infernix_gemma4_small_op_tests)
  infernix_add_family_op_test(infernix_${op}_test FAMILY elementwise
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.cpp"
    LIBRARIES infernix_ops)
endforeach()

infernix_add_family_op_test(infernix_topk_columns_test FAMILY sampling
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_topk_columns.cpp"
  LIBRARIES infernix_ops)
