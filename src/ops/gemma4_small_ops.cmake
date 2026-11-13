# Gemma-4 small ops (non-RDC archive infernix_gemma4_ops; see CMakeLists.txt).
# gemma4_rmsnorm.cu and gemma4_embed_gather.cu hold the Gemma additions to the shared rmsnorm and
# embedding ops: rmsnorm_launch / ops::embedding in infernix_ops call their host launchers.
target_sources(infernix_gemma4_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/launcher/geglu.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_embed_gather.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_qkv_prep.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_residual_norm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/gemma4_rmsnorm.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/logit_softcap.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/residual_scale.cu"
  "${CMAKE_CURRENT_LIST_DIR}/launcher/topk_columns.cu"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/geglu.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gemma4_qkv_prep.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/gemma4_residual_norm.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/logit_softcap.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/residual_scale.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/wrapper/topk_columns.cpp"
)
