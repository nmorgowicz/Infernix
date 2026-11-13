# Gemma 4 attention oracles, adapted from ninfer.spo b330aaf. Include from tests/ops/tests.cmake
# before infernix_finish_op_families(). Each split arm is a fresh process: temporary switches
# are immutable after their first snapshot. --split 0 selects one segment; --split 1 selects
# the automatic rule. --segments 1..13 forces that many segments through the same switch.
foreach(op IN ITEMS gemma4_attention gemma4_fp8_kv)
  if(op STREQUAL "gemma4_attention")
    set(extension cpp)
  else()
    set(extension cu)
  endif()
  infernix_add_family_op_test(infernix_${op}_test FAMILY gemma4_attention
    SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_${op}.${extension}"
    LIBRARIES infernix_ops)
  list(APPEND infernix_gemma4_attention_oracle_tests infernix_${op}_test)
  foreach(split IN ITEMS 0 1)
    add_test(NAME infernix_${op}_split${split}_test
      COMMAND ${infernix_${op}_test_COMMAND} --split ${split})
    list(APPEND infernix_gemma4_attention_oracle_tests infernix_${op}_split${split}_test)
  endforeach()
endforeach()
set_tests_properties(${infernix_gemma4_attention_oracle_tests} PROPERTIES
  SKIP_RETURN_CODE 77 TIMEOUT 900 RESOURCE_LOCK infernix_attention_oracle
  ENVIRONMENT "INFERNIX_TMP_GEMMA4_ATTN_SPLIT_K=0")

# Manual event benchmark (not a CTest): infernix_ops_gemma4_attention_test gemma4_fp8_kv
# --benchmark --chunks 128,512,640,1024 --histories 8192,32768,131072
# This is 36 cells: four chunks x three histories x bf16/fp8_phase1/fp8_phase2.
# Resource usage and rotating-order samples/medians retain the fork's record names;
# the 128-wide cells also retain its independent sampled long-context FP64 oracle.
# A segment sweep runs each family member in a fresh process with --segments N, N=1..13.
