target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q5_dispatch.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1024_k5120.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n6144_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n7168_k5120.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k6144.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n5120_k17408.cu"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k1152.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/n1152_k4304.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/shapes/gemma4.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_sliced_k_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/q5_a16_simt.cu"
)

# Per-call Q5 -> row-FP8 widening is self-contained; reuse the existing non-RDC Gemma archive.
target_sources(infernix_gemma4_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/q5_widen_e4m3.cu")
