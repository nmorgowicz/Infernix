infernix_add_test(infernix_gemma4_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES infernix_model_loading)

# A real Gemma 4 text(+mtp) artifact by explicit path (--artifact); skips (77) without one. Host only.
infernix_add_test(infernix_gemma4_loading_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading_real.cpp"
  LIBRARIES infernix_model_loading)

# Explicit real artifact, no downloads. Exercises real partial-ring writes, public queued recovery and
# the FP8 KV / FP8-QK Program (graphs on/off, accounting, refusals).
infernix_add_test(infernix_gemma4_program_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_program_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 900)

# Tool-call body grammar (`call:NAME{k:v}`, `<|"|>` strings); CPU only, no model.
infernix_add_test(infernix_gemma4_tool_parser_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_tool_parser.cpp"
          "${PROJECT_SOURCE_DIR}/src/models/gemma4/frontend/tool_parser.cpp"
  LIBRARIES infernix::json)

# P7 C.0: prefix-cache byte geometry (31B-like and 26B-A4B-like descriptors), the cache/restore
# identity's fingerprint stability and field sensitivity, and the value-only storage flag's carry
# and refusal. Pure headers, CPU only: no GPU, no artifact, no model.
infernix_add_test(infernix_gemma4_prefix_geometry_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_geometry.cpp")

# P7 store mechanics on the pure helpers, over the 31B-like and 26B-A4B-like descriptors: image and
# block layouts equal the geometry, header round trip and refusal of foreign/value-only identities,
# window runs across the ring wrap, gather/scatter round trips on a host ring emulation, copy
# planning, the Host floor refusal text, the tap ladder and call planning. CPU only: no GPU, no
# artifact (PrefixCache itself pins Host memory with the driver, so its scenarios are the real tests').
infernix_add_test(infernix_gemma4_prefix_store_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_store.cpp"
  LIBRARIES infernix_model_runtime)

# P7 scenarios on a real artifact (INFERNIX_GEMMA4_ARTIFACT; INFERNIX_GEMMA4_HOST_MIB sizes the Host tier):
# frontiers around pages and the ring wrap, displaced/branch/return conversations, three layout identities,
# chat turns through the Engine and a cache-off Engine. Every resume equals a cache-off lane's output.
infernix_add_test(infernix_gemma4_prefix_scenarios_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_scenarios_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_prefix_scenarios_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# P7 recovery on a real artifact: cancel mid-prefill/decode, non-finite withdrawal, abort during restore,
# capture then cancel, a memory fault after a restore, and the Engine's cancel/heal paths; each leaves the
# lane and cache at baseline and the next request equal to a cache-off lane's.
infernix_add_test(infernix_gemma4_prefix_recovery_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefix_recovery_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_prefix_recovery_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# Original HF-template fixture byte identity, plus fixed-template override behavior, tool calls
# through the output session, required tool choice, EOS arrays and thinking precedence. CPU only. Skips (77)
# without INFERNIX_GEMMA4_HF (a Hugging Face Gemma 4 directory with tokenizer.json and chat_template.jinja);
# INFERNIX_GEMMA4_FIXTURES optionally replaces the fixture directory.
infernix_add_test(infernix_gemma4_frontend_fixtures_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_frontend_fixtures.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine infernix_core infernix::json)
target_compile_definitions(infernix_gemma4_frontend_fixtures_test PRIVATE
  INFERNIX_GEMMA4_FIXTURE_DIR="${CMAKE_CURRENT_LIST_DIR}/fixtures")
set_tests_properties(infernix_gemma4_frontend_fixtures_test PROPERTIES SKIP_RETURN_CODE 77)

# Isolated behavioral constraints against the real 262144-token vocabulary: JSON-schema, GBNF,
# regex, choice, Gemma-style tool arguments, transactional masks, split streams and multi-EOS.
infernix_add_test(infernix_gemma4_grammar_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_grammar.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine infernix_core infernix::json)
set_tests_properties(infernix_gemma4_grammar_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
