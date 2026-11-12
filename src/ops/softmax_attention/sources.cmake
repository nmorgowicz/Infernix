target_sources(infernix_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/column_invariant.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/bf16/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/tiled_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/fp8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/int8/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/grouped_vq2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/grouped_q4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/prompt_vq2.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/vq2/prompt_q4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# Experiment (Gemma port P0): the nvfp4 and k8v4 KV-format launch objects (about 30 MB each) compiled non-RDC so they stay out
# of the infernix_ops device link, whose nvlink memory explodes on CUDA 13.5. The vq2 launches stay RDC: they read the
# __device__ codebook defined in kv_cache/vq2_codebook.cu.
target_sources(infernix_nvfp4_non_rdc PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/fast_prompt_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/nvfp4/tiled_launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/k8v4/tiled_launch.cu"
)
