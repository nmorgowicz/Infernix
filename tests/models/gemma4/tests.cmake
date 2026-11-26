infernix_add_test(infernix_gemma4_loading_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_loading.cpp"
  LIBRARIES infernix_model_loading)

# The device reservation Program::plan_device plans per route on a geometry-only model with the real FP8 head
# dimensions: the FP8 ring's first-launch allowance with and without CUDA graphs, none on a BF16 ring. CPU only.
infernix_add_test(infernix_gemma4_program_plan_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_program_plan.cpp"
  LIBRARIES infernix_model_loading infernix_model_runtime)

# Projections stored as GGUF blocks (a llama.cpp GGUF import) through the execution layer's project(): fused
# segments of mixed ggml formats and the global output projection's input permutation, against a binary64
# oracle over independently decoded blocks in source order; the permutation table on the real 31B geometry.
infernix_add_test(infernix_gemma4_gguf_projection_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_gguf_projection.cpp"
  LIBRARIES infernix_model_runtime)
set_tests_properties(infernix_gemma4_gguf_projection_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

# Native Q5 through model project(): explicit A8 prefill vs A16Only/narrow, independently
# encoded operands and original packed-weight FP64 oracles, eager and changed-input graphs.
infernix_add_test(infernix_gemma4_q5_projection_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_q5_projection.cpp"
  LIBRARIES infernix_model_runtime infernix_linear_test_support)
infernix_op_oracle_options(infernix_gemma4_q5_projection_test)
set_tests_properties(infernix_gemma4_q5_projection_test PROPERTIES SKIP_RETURN_CODE 77 LABELS "gpu")

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

# Teacher-forced one-token scoring matches real generation and preserves shared-prefix A16 arithmetic.
infernix_add_test(infernix_gemma4_incremental_score_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_incremental_score_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_incremental_score_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 300)

# Final-layer early exit on an explicit MTP artifact. Direct native full/KV-only calls compare
# every layer's cache bytes; EVIDENCE fingerprints can be compared with retained baseline runs.
infernix_add_test(infernix_gemma4_prefill_earlyexit_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_prefill_earlyexit_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_prefill_earlyexit_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The experimental key-only global KV cache on an explicit MTP artifact: refusals, writer bytes equal
# to the K==V FP8 cache through the first global layer, K-planes-only accounting, deterministic
# generation, root replay and Host snapshot restore equal to the cold key-only run with MTP off/on.
# METRIC lines report V, logits, token and causal-score differences against K==V FP8 and BF16 K/V.
infernix_add_test(infernix_gemma4_key_only_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_key_only_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_key_only_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

# The opt-in FP8 sliding ring on an explicit MTP artifact: refusal with BF16 globals, the first sliding
# layer's FP8 rows equal to the row codec of the BF16 ring's, FP8-ring accounting, deterministic
# generation, root replay and Host snapshot restore equal to the cold FP8-ring run (stored and key-only
# globals, MTP off/on). METRIC lines report scale-granularity error, logits, token and causal-score
# differences against the BF16 ring.
infernix_add_test(infernix_gemma4_fp8_ring_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_fp8_ring_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_fp8_ring_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

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

# P8 MTP on a real artifact with its assistant (INFERNIX_GEMMA4_ARTIFACT; INFERNIX_GEMMA4_KV/QK=fp8 select the
# FP8 KV and QK routes): greedy MTP at 1 and 5 drafts equals plain decode up to a scored near tie, sampled MTP
# keeps plain decode's opening distribution and seed determinism, and a restored turn equals a cache-off turn.
infernix_add_test(infernix_gemma4_mtp_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_mtp_real_test
  PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE LABELS "gpu;real" TIMEOUT 3600)

infernix_add_test(infernix_gemma4_mtp_options_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_options.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)

infernix_add_test(infernix_gemma4_mtp_transactions_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_mtp_transactions_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
set_tests_properties(infernix_gemma4_mtp_transactions_real_test
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

# Thinking control against the real vocabulary: the budget's round license, the control span committed at the cap and
# the answer after it, the budget message, the capacity check, the reasoning-loop guard, grammar-constrained replies and
# that an unreached budget changes nothing. CPU only. Skips (77) without INFERNIX_GEMMA4_HF.
infernix_add_test(infernix_gemma4_thinking_budget_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_thinking_budget.cpp"
  NEEDS_SOURCE_DIR
  LIBRARIES infernix_engine infernix_core infernix::json)
set_tests_properties(infernix_gemma4_thinking_budget_test PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

# The thinking budget through the public Engine on a real artifact (INFERNIX_GEMMA4_ARTIFACT): the control span is
# committed to the resident sequence at the cap with plain and MTP decoding (no round past the cap, the speculative
# counters consistent around it), a reply whose budget is not reached equals one with none, and a follow-up turn that
# resumes from the capped reply's cached state equals a cache-off turn.
infernix_add_test(infernix_gemma4_thinking_budget_real_test
  SOURCES "${CMAKE_CURRENT_LIST_DIR}/test_thinking_budget_real.cpp"
  LIBRARIES infernix_engine infernix_model_runtime)
