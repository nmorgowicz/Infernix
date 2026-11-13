# Gemma 4 native single-lane cold-cache Program and causal scoring.
target_sources(infernix_model_runtime PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/program/program.cpp"
)
