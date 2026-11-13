# Gemma-4 attention and paged append, ported from ninfer.spo b330aaf.
# Keep the fork's RDC code generation without adding these kernels to infernix_ops'
# large device link. The non-RDC D512 FP8 specializations spill on CUDA 13.5.
add_library(infernix_gemma4_attention_ops STATIC)
infernix_internal_includes(infernix_gemma4_attention_ops)
infernix_cuda_archive(infernix_gemma4_attention_ops)
target_link_libraries(infernix_gemma4_attention_ops PRIVATE
  infernix_core infernix_cuda_runtime CUDA::cuda_driver)
target_link_libraries(infernix_ops PRIVATE infernix_gemma4_attention_ops)
target_sources(infernix_gemma4_attention_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_attention.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_attention_mma.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_attention_mma_d512.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_attention_decode_d512.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_kv_append.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gemma4_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gemma4_kv_append.cpp"
)
