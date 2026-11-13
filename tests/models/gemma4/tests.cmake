infernix_add_test(infernix_gemma4_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES infernix_model_loading)

# A real Gemma 4 text(+mtp) artifact by explicit path (--artifact); skips (77) without one. Host only.
infernix_add_test(infernix_gemma4_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES infernix_model_loading)

# Explicit real artifact, no downloads. Exercises real partial-ring writes and public queued recovery.
infernix_add_test(infernix_gemma4_program_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_program_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 900)
