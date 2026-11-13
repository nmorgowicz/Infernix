# Compile policy is owned here; each semantic family explicitly contributes sources.
add_library(infernix_ops STATIC)
infernix_internal_includes(infernix_ops)
infernix_cuda_archive(infernix_ops)
target_link_libraries(infernix_ops PUBLIC infernix_core PRIVATE infernix_nvfp4_non_rdc infernix_gemma4_ops)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from
# producer to consumer warpgroups. RDC makes ptxas discard that contract. These
# self-contained kernels therefore use a non-RDC archive and host launchers (and the
# f16-accumulate split-register launches).
add_library(infernix_nvfp4_non_rdc STATIC)
infernix_internal_includes(infernix_nvfp4_non_rdc)
infernix_cuda_non_rdc_archive(infernix_nvfp4_non_rdc)
target_link_libraries(infernix_nvfp4_non_rdc PRIVATE infernix_core infernix_cuda_runtime CUDA::cuda_driver)

# Gemma-4 ops. A separate non-RDC archive: the CUDA 13.5 device link of infernix_ops needs over 46 GB, and
# every kernel added there raises it (porting plan §8.5). Every translation unit here is self-contained: its
# launcher is a host function, and it calls no device code defined in another translation unit.
add_library(infernix_gemma4_ops STATIC)
infernix_internal_includes(infernix_gemma4_ops)
infernix_cuda_non_rdc_archive(infernix_gemma4_ops)
target_link_libraries(infernix_gemma4_ops PRIVATE infernix_core infernix_cuda_runtime CUDA::cuda_driver)
include("${CMAKE_CURRENT_LIST_DIR}/gemma4_small_ops.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gemma4_q4.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gemma4_fp8.cmake")

include(basic_sources.cmake)
include("${CMAKE_CURRENT_LIST_DIR}/attn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/candidate_selector/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/dynamic_grouped_conv/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_gating_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/gdn_input_proj/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ggml/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/hyper_connection/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/kv_cache/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_add/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_pair/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_swiglu/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/linear_topk/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/offloaded_sparse_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ple/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/lossless/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/projection_fp32/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/qsa/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/resident_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rows/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/rmsnorm_rope/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/screened_head/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/softmax_attention/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/sparse_moe/sources.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/speculative_tree/sources.cmake")
