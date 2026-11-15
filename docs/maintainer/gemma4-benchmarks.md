# Gemma4 benchmark tools

The workloads in `tools/bench/kv_eval/`, `rp_ladder.py`, `bench_tail.py`,
`fixed_tail.py`, `decode_abab.py`, and `retrieval-smoke.py`
run from this checkout,
use local artifacts and corpus files, and do not import an operator/tooling checkout.
They do not download models or corpus data. Run commands from the repository root
with Python 3.11 and the dependencies in `tools/bench/kv_eval/requirements.txt`.
`derive_prefix_ids` additionally uses Transformers, with local files only.

All executable, artifact, corpus, reference and output paths below are placeholders
chosen explicitly by the operator. Use a fresh output directory for each campaign.
Do not benchmark alongside another GPU workload. The process-owning runners share
an exclusive lock in the OS temporary directory (`infernix-bench-gpu.lock`); `--lock`
can select another common path. A stale lock must be inspected before removal.
The lock coordinates these runners, not arbitrary external GPU processes.
Servers bind loopback, have a health/readiness timeout, capture logs, and are stopped
on failure as well as success. No firewall or machine-specific paths are assumed.

### MTP selecting-logit evidence

The family-private `gemma4::ProgramOptions::logits_observer` observes the BF16
logits already used by ordinary prefill, plain decode or MTP verification. It runs
after sampling/acceptance and outside CUDA-graph capture; normal execution does
no observer staging, copy or synchronization when the callback is absent.
Columns are vocabulary-contiguous; column zero predicts the output at
`first_predictor + 1`. Only the pending row's licensed columns are output
evidence; columns after a rejected draft must not be compared as generated text.

A greedy near-tie check records both original selecting distributions at the
first token divergence, while their generated contexts still match. A fresh
causal-scoring replay is not a substitute: its prefill/verification widths may
change the arithmetic. The real MTP test requires each chosen token to be its
own column's argmax and the candidate gap under the original plain reference
to be at most 0.125. The candidate distribution's gap is retained as diagnostic
evidence, not used as a second reference. This check does not replace the
separate paired KL/NLL or sampled-distribution gates.

The real MTP test also prints, per prompt, the logit drift between the MTP and plain paths over the
positions before their first token difference. Measured 2026-11-14 (FP8 KV and QK, 128 outputs, ten
prompt/draft cells each): the plain winner's logit moves by a mean of 0.03-0.04 (BF16 logits step 0.125 at
16-32) in both q4_g256 and q4_g32, and by up to 1.0 at worst; any-token maxima reach 1.4-1.9 on g256 and up
to 6.2 on g32 (low-probability tokens). A greedy divergence therefore needs a near tie, and g256 simply
met none on those prompts while g32 met a few. It is rounding-scale and is not an arithmetic defect of one
quantization. Forcing the q4 GEMV for T=2..8 did not remove the g32 divergences; BF16 KV/QK did not either.

### Decode speed, best-medium (q4_g32 GPTQ, MTP d1)

Port only, FP8 KV and FP8 QK, 3 timed reps per length after a discarded warmup, one server per run (median
decode tokens/s; draft acceptance in brackets). `INFERNIX_TMP_Q4_EXACT=1` changes only prefill passes of 128
tokens and up, not decode.

| Context | Default route | `INFERNIX_TMP_Q4_EXACT=1` |
|---:|---:|---:|
| 1k | 113.78 (0.82) | 116.12 (0.83) |
| 8k | 110.71 (0.80) | 111.35 (0.80) |
| 32k | 101.34 (0.85) | 101.16 (0.83) |
| 100k | 83.01 (0.85) | 84.04 (0.85) |

Cold prefill at 32k is 4,178-4,303 tok/s on the default route against 3,450-3,455 with the exact flag
(about -18%), and at 100k 2,393-2,436 against 2,123-2,162 (about -11%). For reference, best-small (g256)
decodes 121.35 / 120.76 / 108.75 / 86.75 tok/s at the same four lengths.

## Native full-vocabulary KL

`infernix-perplexity --full-vocab-reference-write CACHE` creates the
`.nklref` full-vocabulary cache. `--full-vocab-reference CACHE` scores against it,
requiring the same pinned protocol and BF16-rounded logits (`--logits-dtype bf16`).
These flags are distinct from the existing `--kl-reference` top-token lower-bound
instrument. Do not substitute one for the other.

The native comparison writes the `report.json` `kl` structure and
`kl_positions.f32` (four F32 values per scored position). Reference metadata and
protocol checks remain fail-closed; neither corpus nor tokenizer nor window settings
may be silently changed. Native prefix IDs are explicit `--full-vocab-prefix-ids`
and must exactly match the cache. `derive_prefix_ids` renders the local artifact's
chat preamble; the runner accepts its JSON via `--prefix-ids-file` (or a direct
`--prefix-ids` CSV) and maps it to the native flag. Non-native scoring has no prefix.

The single-arm runner supports both native modes and keeps argv, stdout/stderr,
executable/artifact/corpus hashes, GPU snapshots and a receipt:

```text
py -3.11 -m tools.bench.kv_eval.run_ppl --exe <infernix-perplexity> --artifact <reference.infernix> --corpus <manifest.json> --arms-json <arms.json> --arm bf16 --out-dir <reference-run> --context 512 --stride 256 --full-vocab-reference-write <new-reference.nklref>
py -3.11 -m tools.bench.kv_eval.run_ppl --exe <infernix-perplexity> --artifact <candidate.infernix> --corpus <pinned-manifest.json> --arms-json <arms.json> --arm fp8 --out-dir <native-run> --context 512 --stride 256 --full-vocab-reference <ref512-en.nklref> --full-vocab-reference-kv-dtype fp8 --full-vocab-reference-gemma4-qk bf16 --prefix-ids-file <pinned-prefix.json> --save-tokens <native-windows.json>
```

The expected-reference configuration options describe the cache, not the candidate:
`--full-vocab-reference-kv-dtype` and `--full-vocab-reference-gemma4-qk` default to
candidate settings. An FP8-QK candidate using the pinned BF16-QK cache must explicitly
pass `--full-vocab-reference-gemma4-qk bf16`. Put candidate execution settings such as
`--prefill-chunk 128`, `--kv-dtype fp8`, `--gemma4-qk fp8` in the selected arm's
`flags`. The runner provides `--full-vocab-seed` (20261105) and
`--full-vocab-resamples` (1000), matching native report defaults.

`kl_corpus` makes deterministic UTF-8 head-truncated corpus subsets and records their
hashes. Do not recreate or substitute a subset when measuring against a pinned cache.
Full-vocabulary cache storage is about one MiB per position at vocabulary 262144.

```text
py -3.11 -m tools.bench.kv_eval.kl_corpus --source <manifest.json> --out <new-subset>
py -3.11 -m tools.bench.kv_eval.kl_compare --baseline <baseline/report.json> --candidate <candidate/report.json> --out <paired.json>
```

`kl_compare` runs a paired whole-window bootstrap: 2000 resamples by
default, seed 20261105, overall and per domain. `--delta` has no default: only the
operator chooses an acceptance factor. When supplied, acceptance still requires the
upper 95% mean-KL ratio bound to be at most `1 + delta` overall and in every domain,
and the point-estimate p999-KL ratio to be within that same factor. Protocol, cache,
reference artifact and scored window/domain identities must match.

### T1 parity evidence

The required pinned rescore is best-small FP8 KV against `ref512-en.nklref`, at
context 512 / stride 256 / chunk 128: 47 windows and 13,728 positions. The existing
adapter evidence gives mean KL 0.12347 for BF16-QK and 0.12384 for FP8-QK (rounded
figures, not tolerances). Native results must reproduce the underlying adapter
results exactly, including window hashes. Compare the full `kl.statistics` values
and per-position sidecars, not merely those rounded means. Keep reference cache,
corpus, tokenizer, window order, logits dtype, seed and resample count identical.
The pinned cache's prefix IDs are
`2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101`;
its token-window hash is `e07b21a0629a9f4e`. Supply these exact prefix IDs rather
than deriving a possibly different template preamble for this pinned rescore.
`--save-tokens` provides native window-token evidence for this comparison. Do not
retire the external scoring adapter until native parity is actually established.
This document records the required evidence, not a claim that it has been run.

## Top-token diagnostic and retrieval kit

An arms file is a JSON list or `{ "arms": [...] }`; each entry contains `name`,
`flags` (one argv token per element), and role `reference`, `noise`, or `quant`.
Exactly one arm is the reference. `tools/bench/kv_eval/arms.example.json` supplies
BF16, repeat, FP8, FP8-QK and INT8 examples; select only formats the artifact supports.

Without a full-vocabulary mode, `run_ppl` requests the target's `--save-top-tokens`
format (`NINFTOPK`, version 1). Its
reader checks the native header, complete end marker, position count and window
layout against the report. `analyze` computes the instrument definitions:
PPL/dPPL, top-1 agreement, renormalized top-64 truncated KL, both-in-top-64 target
log-probability differences, paired-window statistics, domains and position bins.
The truncated diagnostic is **not exact full-vocabulary KL** and is also distinct
from the native executable's top-token-plus-remainder lower-bound KL.

```text
py -3.11 -m tools.bench.kv_eval.run_ppl --exe <infernix-perplexity> --artifact <artifact.infernix> --corpus <manifest.json> --arms-json <arms.json> --arm bf16 --out-dir <ppl-run> --profile short --dry-run
py -3.11 -m tools.bench.kv_eval.analyze --dir <ppl-run> --arms-json <arms.json> --artifact <artifact.infernix> --corpus-manifest <manifest.json>
```

Run each arm sequentially before analysis. Output layout is `<arm>/report.json`,
`dumps/<arm>.topk`, `logs/`, followed by `analysis.json` and `tables.md`.
`common` handles embedded tokenizer extraction and corpus provenance; `corpus_pool`
provides token-aligned natural text; `metrics` defines the diagnostic statistics;
`gpu_state` records available GPU/process information without requiring nvidia-smi.

For an already-running server with `--model-id gemma-bench`:

```text
py -3.11 -m tools.bench.kv_eval.needle run --label fp8 --out <needle/fp8.jsonl> --manifest <manifest.json> --artifact <artifact.infernix> --model gemma-bench --url http://127.0.0.1:8041/v1/chat/completions
py -3.11 -m tools.bench.kv_eval.needle summarize --dir <needle> --arms bf16,fp8
py -3.11 -m tools.bench.kv_eval.speed_client --label fp8-mtp --out <speed/fp8-mtp.cells.jsonl> --manifest <manifest.json> --artifact <artifact.infernix> --model gemma-bench --url http://127.0.0.1:8041/v1/chat/completions
py -3.11 -m tools.bench.kv_eval.speed_summary --dir <speed> --arms fp8 --modes mtp
```

Needle retrieval uses deterministic corpus prompts and exact six-digit scoring;
summaries reject differing prompt hashes and report Wilson intervals and paired
exact McNemar tests. The speed client records one warmup plus timed unique prompts
per length; summaries exclude warmup, retain failures and flag unexpected output
lengths/cache reuse. Optional `<label>.serve.jsonl` server-start records add memory
and capacity information. `--dry-run` on these clients builds prompts without HTTP.

## HTTP tail and MTP campaigns

All five campaign scripts accept explicit local `--artifact`, `--manifest`, optional
`--tokenizer`, and `--pool-domains`. Corpus counts are nominal: use server-reported
prompt token counts for actual lengths.

```text
py -3.11 tools/bench/rp_ladder.py --exe <infernix-serve> --artifact <artifact.infernix> --manifest <manifest.json> --out-dir <ladder> --contexts 1000,8000,32000,100000 --arms off,d1,d2,d4
py -3.11 tools/bench/bench_tail.py --artifact <artifact.infernix> --manifest <manifest.json> --label fp8-mtp --out <tail.jsonl> --url http://127.0.0.1:8041/v1/chat/completions --model gemma-bench --history 100000 --tail 1200
py -3.11 tools/bench/fixed_tail.py --artifact <artifact.infernix> --manifest <manifest.json> --label fp8-mtp --out <fixed.jsonl> --url http://127.0.0.1:8041/v1/chat/completions --model gemma-bench
py -3.11 tools/bench/retrieval-smoke.py --artifact <artifact.infernix> --manifest <manifest.json> --label fp8 --out <retrieval.jsonl> --url http://127.0.0.1:8041/v1/chat/completions --model gemma-bench
py -3.11 tools/bench/decode_abab.py --exe <infernix-serve> --artifact <artifact.infernix> --manifest <manifest.json> --arms-json <decode-arms.json> --out-dir <decode-abab> --mtp-draft 1
```

- `rp_ladder` owns one server per arm, with FP8 KV/FP8-QK, chunk `auto`, concurrency
  one and an explicit MTP-off or draft count 1/2/4. Each fresh context gets sampled
  ingest, three sampled tail turns (temperature 1, top-p .95, top-k 64) and a greedy
  probe. Replies differ across arms; compare median timings, not output equality.
  `prompt_ms` is a server-side prefill/TTFT proxy, not wire first-token latency.
- `bench_tail` appends the real generated reply before sending the next tail.
- `fixed_tail` replaces the reply with fixed corpus text. It records both message
  hashes and measures scripted divergence, not reusable generated continuations.
- `retrieval-smoke` runs six cases: 32k/100k at depths .05/.50/.95, one needle per
  cell. It exits nonzero for any failed answer or request. This is only a smoke.
- `decode_abab` starts fresh servers in A/B/A/B order and uses the in-tree speed
  client. No decode-attention selector variable is forwarded. Instead, explicitly specify executable/argument/environment differences:

  ```json
  {"a": {"exe": "<baseline-serve>", "args": [], "env": {}},
   "b": {"exe": "<candidate-serve>", "args": [], "env": {}}}
  ```

  Arm executable paths resolve relative to the JSON file. `--exe` supplies the
  common default. Environment values are string maps; argument values are argv
  lists. Both runs use the same corpus prompts and sampling settings.

The process-owning ladder/ABAB tools use the serve flags `--vram-headroom-mib` and
`--host-context-mib`; they pass no `--model-match`, `--kv-headroom-mib` or `--device-state-slots`
options. Additional
settings are explicit `--serve-arg=--flag` / `--serve-arg=value` tokens. To disable
cache reuse, pass `--no-prefix-reuse` (then no Host-cache option is emitted).

Every chat-issuing tool (`rp_ladder`, `bench_tail`, `fixed_tail`, `decode_abab`, `soak`,
`retrieval-smoke`, `kv_eval/needle.py`, `kv_eval/speed_client.py`) takes
`--enable-thinking {true,false,default}`. `true`/`false` send `enable_thinking` in every
request body; `default` omits the field so the server decides, and infernix resolves an
absent value to thinking ON on Gemma 4. The chosen value is recorded in each tool's receipt or cells. Campaigns
that compare engines, or that measure roleplay behaviour, MUST pass the same explicit
value on every arm; otherwise the arms may measure different generation lengths and
prompts. `retrieval-smoke` and `needle` default to `false`: their 32-token answer budget
cannot hold a reasoning block, so with thinking left on the reply comes back empty and
every case scores as failed. All other tools default to `default` (field omitted).

## Soak and cross-engine tools

- `python -m tools.bench.soak` starts one production-profile server and runs the cached-tail probe at near-capacity history sizes
  (a stress pass: GPU memory before and after, liveness, log error lines, a sampled final request).
- `python -m tools.bench.roleplay_soak` is a client-only sustained session against any running server (start the real launcher
  yourself, so the exact production command is what is tested): growing conversations to `--context-limit`, sampled turns with
  thinking as stated, a streamed request dropped every `--abort-every` turns, GPU memory sampled in the background. It reports
  failed requests, liveness, VRAM growth (second quarter against last quarter) and the fraction of turns that restored most of
  their prompt from the cache; exit 0 only if all hold. It measures stability, not output quality.
- `rp_ladder.py --arm-argv-json FILE` replaces the built server command per arm with a complete argv, so another engine can be the
  control while the client, requests and corpus stay identical. Cross-engine pairing and gates live in the tooling repo.

## Verification handoff

These commands are for the integration owner; porting the scripts alone does not
establish native numerical parity, server behavior or performance:

```text
py -3.11 -m compileall -q tools/bench/kv_eval tools/bench/gemma4_common.py tools/bench/rp_ladder.py tools/bench/bench_tail.py tools/bench/fixed_tail.py tools/bench/decode_abab.py tools/bench/retrieval-smoke.py
py -3.11 -m tools.bench.kv_eval.run_ppl --help
py -3.11 tools/bench/rp_ladder.py --help
py -3.11 tools/bench/decode_abab.py --help
```

Then exercise a synthetic native `NINFTOPK` file/report through `metrics.open_dump`
and paired synthetic reports through `kl_compare`, including mismatch/truncation
refusals. Perform the pinned native rescoring and adapter comparisons above, retain
Qwen perplexity regression evidence, and run the ladder and HTTP smoke on the
approved serialized GPU session. No thresholds may be relaxed to obtain parity.
