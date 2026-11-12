# Gemma 4 (text component and the MTP assistant): its own model architecture. Config, binding and
# loading join infernix_model_loading.
target_sources(infernix_model_loading PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/config.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/load.cpp"
)
