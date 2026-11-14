# Gemma 4 native single-lane Program with its prefix cache, and causal scoring.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/prefix_cache.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/program/prefix/state_image.cpp"
)
