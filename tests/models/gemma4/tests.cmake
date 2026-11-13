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
