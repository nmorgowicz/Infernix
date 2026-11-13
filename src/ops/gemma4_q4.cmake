# Gemma-4 Q4 g32/g128/g256 FP16-scale linear kernels (non-RDC archive infernix_gemma4_ops).
target_sources(infernix_gemma4_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/linear/q4/q4_g32_gemv.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear/q4/q4_g32_tc.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear/q4/q4_g32_w4a8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear/q4/q4_g32_wide.cu"
)
