# Gemma-4 FP8 row quantize, cuBLAS FP8 row, and Gemma-shape FP8 GEMM (non-RDC archive infernix_gemma4_ops).
# The shape instances (fp8_shapes.h's Gemma Fp8LinearShape objects, which fp8_dispatch.cpp in infernix_ops
# reads) are plain host data and host launchers: every kernel they launch is a template instantiated in
# the same translation unit. They call back into infernix_ops for launch_fp8_a8_quantize (fp8_a8.cu).
target_sources(infernix_gemma4_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/launcher/fp8_row_quantize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/fp8_row_quantize.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/fp8_row_cublas.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/fp8_row_cublas.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/linear/fp8/shapes/gemma4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear/fp8/shapes/gemma4_a16.cu"
  "${CMAKE_CURRENT_LIST_DIR}/linear/fp8/shapes/gemma4_a8.cu"
)
# fp8_row_cublas drives its GEMM through cuBLAS (Fp8RowCublasContext, wrapper/fp8_row_cublas.cpp).
target_link_libraries(infernix_gemma4_ops PRIVATE CUDA::cublas)
