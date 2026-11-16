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

## Four-tier comparison

This is the canonical measured comparison of `best-medium`, `best-small`, `iq3-xs` and `iq3-xxs`.
All four are **the same Wicked-Stuff-31B-Final diode-s15-t005 model**, text-only **including the FP8 MTP
assistant**, without vision. Their HF text-weight/config/index identities match and the bound assistant
parameter payloads are byte-identical; the encoded trunk weights intentionally differ. File bytes are the
actual measured artifacts, not estimates or a different build's receipt. Artifact SHA-256 prefixes, in table
order: `11347ddc8860`, `fc32d729d5ba`, `4404999e58ae`, `d0fdeef561a5`.

**Quality instrument:** each artifact against the **same diode-own Q8_0 full-vocabulary reference**,
native `infernix-perplexity`, 47 windows / 13,728 positions, context 512, stride 256, prefill chunk 128,
FP8 KV, BF16 QK and BF16-rounded logits, identical prefix and token-window bounds. Reference window hash
`e07b21a0629a9f4e`; the corpus includes English prose and code. The two q4 files were newly scored;
the matching IQ3 reports were reused. These are teacher-forced causal KL results, not generated-output/MTP
quality. Older preset hints such as 0.0845 / 0.1213 used an FP8-row reference and different artifact builds;
they are **not** the matched-reference quality column below. Plain-model IQ3 results remain separate below.

**Serving instrument:** RTX 5090, CUDA 13.5 development server, FP8 KV/QK, BF16 logits, MTP depth 1,
maximum context 32768, **auto prefill actually selected 2048**, prefix reuse and host cache disabled.
At each of ~1k and ~16k: one excluded warmup and three timed, cold, natural English corpus prompts,
byte-identical cross-arm and Gemma-tokenized. Actual timed prompt counts are 992–993 and 15,989–15,993.
Greedy temperature 0, top-k **20**, seed 1234, top-p omitted, cap **256**, explicit `enable_thinking=true`:
the established cold four-bit HTTP comparison settings, **not** the roleplay launcher's sampling defaults.
All arms explicitly use the same `tools/chat_templates/gemma4_fixed.jinja` because their embedded template
bytes differ. Serving binary SHA-256 prefix: `f0073e790fb9`. Every timing run used the 8 GiB free-RAM floor.

**Thinking is proven, but this is capped thought generation:** all **24 timed responses** published nonempty
`reasoning_content` SSE deltas and reported **256 reasoning / 256 total completion tokens**, `finish_reason=length`,
with **zero final-answer tokens**. Rendered thinking controls use the system `<|think|>` opener and the
generation suffix `<|turn>model\n<|channel>thought\n`; request logs also resolve thinking true.
Decode rates and MTP counts include those reasoning tokens. This is neither a completed-answer comparison,
a long-thinking soak nor a measure of reasoning quality.

All paired serving columns below are **~1k / ~16k**; rates and latency are medians of three timed reps.
Accepted/drafted counts sum those three reps. Wire TTFT ends at the first nonempty reasoning/content delta;
the prompt-time proxy excludes template/tokenization/transport and is shown separately.

| Tier | Actual file bytes, text + MTP | Own-Q8 KL [95% CI] | Prefill tok/s | Decode + MTP tok/s | MTP accepted/drafted | Wire TTFT ms | Prompt proxy ms | GPU used MiB |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| `best-medium` | 18,405,559,556 | 0.0941 [0.0803, 0.1099] | 6552.9 / 5545.4 | 118.38 / 110.04 | 344/420 / 340/425 | 182 / 2934 | 151 / 2884 | 25834 / 25839 |
| `best-small` | 16,852,045,060 | 0.1291 [0.1099, 0.1503] | 6615.8 / 5531.8 | 127.09 / 118.37 | 348/415 / 336/429 | 173 / 2950 | 150 / 2891 | 24366 / 24346 |
| `iq3-xs` | 13,902,732,292 | 0.2256 [0.1971, 0.2547] | 6519.0 / 5412.8 | 128.59 / 118.03 | 347/417 / 341/422 | 159 / 3003 | 152 / 2955 | 21534 / 21542 |
| `iq3-xxs` | 13,019,175,940 | 0.3154 [0.2711, 0.3701] | 6511.6 / 5399.0 | 126.92 / 120.14 | 335/427 / 332/430 | 175 / 3035 | 152 / 2962 | 20695 / 20694 |

Decode repetition ranges, ~1k / ~16k (tok/s; **not confidence intervals**): best-medium
116.10–122.92 / 109.80–115.51; best-small 126.31–133.95 / 117.20–119.20; IQ3_XS
127.27–129.60 / 117.35–120.59; IQ3_XXS 124.59–134.71 / 118.44–120.43.
IQ3_XS reaches 101.2% / **99.7%** of best-small's decode medians; IQ3_XXS **99.9%** / 101.5%.
Thus neither IQ3 tier meets strict median decode parity at **both** lengths; overlapping repetition ranges
are not proof of equality. Best-medium is 93.1% / 93.0% of best-small here, in exchange for lower causal KL.
IQ3 prefill is about 97.6–98.5% of best-small. Smaller files trade quality for storage, not a blanket speed win.

Memory is whole-device `nvidia-smi` used/total snapshots after requests (total 32607 MiB), **not model-only
weights, peak allocation or an allocator high-water mark**. No smaller-card fit, 131k capacity or longer-prompt
throughput claim follows from this 32k-reservation campaign. Results are unavailable for completed final
answers under these capped requests, long-thought stability and generated-output KL; unavailable is not zero.

Optional per-token streaming timing telemetry initially returned `invalid OpenAI Chat live-timing state`
before generation on every arm. It was omitted for the successful campaign; terminal timing/usage chunks,
raw SSE and independently timestamped wire TTFT were retained. An initial incompatible redundant cache flag
was also rejected before loading. Those failed attempts are retained, not counted as timed repetitions.
No server telemetry fix or per-token timing qualification is claimed.

For creating either small file from HF BF16, see
[IQ3 creation](../weight-conversion.md#create-the-smaller-iq3-files-from-hf-bf16). Use this table rather than
combining the earlier unmatched/route-specific measurements below into a new ranking.

## Long-context baseline (~65k and ~122k prompts in a 131072 context)

Same four artifacts, serving binary (SHA-256 prefix `f0073e790fb9`), fixed `gemma4_fixed.jinja` template and RTX 5090 as the
[four-tier comparison](#four-tier-comparison), but with the roleplay launcher profile instead of its 32768 / no-cache
configuration: `--max-context 131072 --kv-capacity auto --vram-headroom-mib 1024 --kv-dtype fp8 --gemma4-qk fp8
--prefill-chunk auto --logits-dtype bf16 --spec mtp --draft-tokens 1 --host-context-mib 8192`, prefix reuse on, loopback
host and a dedicated port. Thinking is **off** (`enable_thinking=false`) except the explicitly labelled capped thinking
rows; greedy temperature 0, top-k 20, seed 1234, streaming with usage. Every timing job ran under the 8 GiB free-RAM
floor (minimum free RAM observed 27.2 GiB). Rates are server `timings`; wire TTFT is the first nonempty content
delta seen by the client; whole-device MiB is the maximum of `nvidia-smi` `memory.used` sampled about every 0.4 s
during each request, **including the 2.1-2.7 GiB the idle desktop already used**, so it is not model-only memory and a
sampler can miss shorter spikes. This is one 5090 under its software power cap (server throughput lines report
`limited sw-power-cap`); it says nothing about other cards.

**KV capacity is verified, not read from metadata.** All four starts log
`capacity | 131,072 tokens of context across all requests (fp8 KV)` with `pages 2,048/2,048`, global FP8 KV 5.04 GiB
plus BF16 sliding rings 2.34 GiB, and `prefix cache | on | 8.00 GiB in RAM`; `--plan` reports `kv_tokens` 131072 and
`fits` for each. `--kv-capacity auto` is therefore capped by `--max-context` here (the log labels it `explicit`) and
the remaining device memory after startup is 2.68 (best-medium), 4.09 (best-small), 6.51 (IQ3_XS) and 9.35 GiB (IQ3_XXS).
`/v1/models` reports `n_ctx = max_model_len = 131072`, equal to the verified pool in this configuration, and the
~122k-prompt requests below completed up to 8,192 generated tokens inside it. Nothing above 131072 was attempted:
that needs a separate start with a larger `--max-context`, which was not run, so no extension claim is made.

**Prompts.** Natural English corpus windows from the pinned pool, byte-identical across tiers: actual prompt counts
**64,986-64,987** and **121,996-121,998** tokens (not nominal). Each point is one excluded warmup then three timed cold
requests, each a distinct document (distinct leading reference id, so no prefix reuse between them); the ~122k timed
requests ask for up to 8,192 tokens, the ~65k ones for 256. Servers log an actual automatic chunk of **2048**
(`signature gemma4/fp8-kv/qk=fp8/cold/chunk=2048`) with 33 prefill units at ~65k and 61 at ~122k.

**Three runs, because the primary run had a transient.** The primary campaign ran the tiers in the order IQ3_XXS, IQ3_XS,
best-small, best-medium. Its IQ3_XXS (and, less, IQ3_XS) cold ~65k and ~122k results were 9-21% slower than its best-small
and best-medium results, while the same tiers at ~16k had matched. A reverse-order control (best-medium first, IQ3_XXS
last; ~65k and one ~122k request set) and a third run in the primary's order (cold ~65k, continuations of conversations
that were no longer the latest, and ~122k cold/continuation pairs) did **not** reproduce it: all four tiers cold-prefill ~65k
at ~3.1k tok/s there. The primary XXS/XS depression is therefore neither a tier property nor a fixed-order effect; it is a
transient during those primary runs whose cause was not identified [INFERENCE: GPU power/thermal state or other host
activity; unmeasured]. The primary rows are published as measured, next to the other runs, not replaced. Decode rows
and MTP counts are deterministic under greedy sampling (identical accepted/drafted counts in every run).

Cold ~65k prefill, thinking off, 256 generated tokens (medians of three timed requests; primary / reverse-order control /
third run):

| Tier | Prefill tok/s | Decode tok/s | Wire TTFT s | MTP accepted/drafted (3 reps) | Run-peak device MiB (primary / control / third) |
|---|---|---|---|---|---|
| `best-medium` | 3098 / 3161 / 3144 | 88.1 / 91 / 89 | 21.16 / 20.76 / 20.83 | 332/432 | 29164 / 29318 / 29300 |
| `best-small` | 3197 / 3105 / 3186 | 94.7 / 93 / 95 | 20.53 / 21.09 / 20.55 | 323/439 | 27683 / 27945 / 27828 |
| `iq3-xs` | 2922 / 3101 / 3112 | 82.6 / 88 / 90 | 22.42 / 21.10 / 21.08 | 313/451 | 25483 / 25139 / 25061 |
| `iq3-xxs` | **2529** / 3153 / 3186 | **72.6** / 93 / 95 | **25.93** / 20.77 / 20.58 | 317/447 | 24608 / 24297 / 24184 |

Primary-run ranges across the three timed requests: prefill 3083-3109, 3187-3219, 2915-2927 and 2480-2554 tok/s;
decode 85.7-92.1, 92.9-96.3, 80.6-86.0 and 71.6-73.9 tok/s. The prefill drop from ~5.4-5.5k tok/s at ~16k to ~3.1k at
~65k and ~2.1-2.17k at ~122k is attention cost growing with context. The best-medium / best-small / IQ3 decode spread at
~65k in the control is within ordinary run-to-run variation (~5%).

Cold ~122k (prompt 121,996-121,998), thinking off. The primary rows ask for 8,192 tokens and stop on their own at
different lengths, so their decode is an average over different generation lengths; the clean decode-at-122k number is the
256-token cached continuation in a later table.

| Tier | Prefill tok/s (primary median [range]; control; third-run x122) | Wire TTFT s (primary) | Generated tokens (3 timed requests, finish) | MTP accepted/drafted | Peak device MiB |
|---|---|---:|---|---|---:|
| `best-medium` | 2139 [2138-2140]; 2053; 2160 | 57.41 | 4986, 2121, 2502 (stop) | 4175/5432 (76.9%) | 29155 |
| `best-small` | 2166 [2161-2173]; 2141; 2156 | 56.57 | 5678, 3783, 3295 (stop) | 5457/7297 (74.8%) | 27672 |
| `iq3-xs` | 1970 [1794-1996]; 2124; 2163 | 62.21 | 3767 (stop), 8192, 8192 (length) | 9652/10495 (92.0%) | 25483 |
| `iq3-xxs` | 1868 [1826-2023]; 2164; 2167 | 65.73 | 8192, 8192, 8192 (length) | 11946/12624 (94.6%) | 24585 |

(The third run's ~122k column is its 256-token `x122` request; control and third-run values are single measurements.)
Cold ~122k prefill is ~2.1-2.17k tok/s on all tiers in the runs without the transient, ~57 s to first token.

**Greedy long generations degenerate on the IQ3 tiers at ~122k.** Five of the six IQ3 ~122k timed generations (iq3-xs
2 of 3, iq3-xxs 3 of 3) ended in a degenerate repetition loop of a few tokens (`l' l' l' ...`, ` la la la ...`) until the
8,192 cap (tail compressibility 0.003-0.005, against 0.42-0.48 for all six best-medium/best-small replies, which each
ended naturally with `stop`). The control run's single ~122k request per tier reproduced the primary run's length exactly
(best-medium 4,986, best-small 5,678, iq3-xs 3,767 `stop`; iq3-xxs 8,192 loop with an identical tail). The 92-95% MTP
acceptance of the looping runs is the loop being trivially predictable, not healthy speculation. This is greedy sampling
on one prompt set, not the launcher's sampler and not a quality metric; it is nevertheless a measured difference between
tiers at this length.

Cached continuation of the **most recent** conversation (the previous user turn, a fixed short assistant message, a new
user turn; 256 generated tokens, thinking off; primary / control TTFT):

| Tier | ~65k: cached / computed tokens, TTFT s | ~65k decode tok/s | ~122k: cached / computed tokens, TTFT s | ~122k decode tok/s |
|---|---|---|---|---|
| `best-medium` | 64,982-64,983 / 47, 0.34 / 0.38 | 87 | 121,989-121,991 / 50, 0.74 / 0.66 | 72 |
| `best-small` | 64,982-64,983 / 47, 0.35 / 0.42 | 90 | 121,989-121,991 / 50, 0.68 / 0.72 | 77 |
| `iq3-xs` | 64,982-64,983 / 47, **0.88 / 0.88** | 82 | 121,989-121,991 / 50, **1.25 / 1.26** | 69 |
| `iq3-xxs` | 64,982-64,983 / 47, **1.05 / 0.84** | 72 | 121,989-121,991 / 50, **1.19 / 1.26** | 68 |

The ~65k resume is `lane_resident` (the Gemma lane continued its own state, 99.9% of the prompt cached) and the ~122k one
`hybrid_snapshot`. The IQ3 tiers take 1.8-3x the q4 tiers' time on the ~47-50 token tail (reproduced in the reverse-order
control); both are under 1.3 s against 20-57 s cold. The cause was not isolated. Server throughput lines during
prefill also show the IQ3 tiers' exposed host-side time at ~14% against ~1.3% on the q4 tiers.

**Resuming a conversation that is not the latest was a full miss on every tier.** In 32 attempts (16 in the primary
campaign: three ~65k continuations after ~122k traffic and one ~122k per tier; 16 in the third run: three ~65k
continuations after two other ~65k conversations and one ~122k after one other ~122k conversation, per tier) the
continuation reused **0** tokens and re-prefilled at roughly the cold rate (~3.1k tok/s at ~65k, ~2.1k tok/s at ~122k,
i.e. ~21 s and ~57 s to first token); the iq3-xxs third-run request log records `prefix_miss` `kind: diverged`,
`lost_reuse_tokens: 64,982` and no snapshot or Host restores. That run's server counters show the 8,192 MiB Host tier at
8,588,881,920 of 8,588,881,920 bytes, 41 snapshot images written, 37 Host snapshot evictions, 10,956 Device blocks
evicted, 10,885 Host slabs reclaimed as unreachable and 0 Host block or image restores (best-small: 39 / 37 / 10,956 /
10,229 / 0). No tier kept a resumable ~65k or ~122k history across even one intervening conversation with this Host
budget. [INFERENCE] A larger `--host-context-mib` (or the SSD tier) might change this; it was not varied, so no claim is made.

Capped thinking at ~65k (`enable_thinking=true`, cap 256, three timed requests, prompts 64,991-64,992): every one of the 12
responses streamed nonempty reasoning and reported 256 reasoning / 256 total tokens with `finish_reason=length`
and no final answer, so decode includes reasoning only. Decode median [range] tok/s: best-medium 93.0 [90.5-95.0],
best-small 99.8 [97.5-100.1], iq3-xs 93.0 [91.5-93.0], iq3-xxs 100.8 [75.3-102.0] (the wide XXS range is the primary-run
transient); MTP accepted/drafted 352/410, 344/420, 354/409 and 342/420. This does not measure a completed thought, answer
quality, or the 8,192-token total budget of long-thinking roleplay.

Not measured: sampling other than greedy, any prompt beyond 131072, a larger Host tier, concurrency above one, cold
prefill with a different chunk, and quality at these lengths (no KL or retrieval score exists for them).

## Long-thinking qualification (8192 total tokens including reasoning)

Stability and throughput only; **no reasoning or answer quality is claimed**. Same four artifacts, binary (`f0073e790fb9`),
fixed template and RTX 5090 as above, with the roleplay launcher profile (131072 context, FP8 KV/QK, auto chunk 2048, MTP d1,
8192 MiB host tier unless labelled), one fresh server per tier, 8 GiB free-RAM floor (minimum free RAM observed 15.7 GiB with the
20480 MiB host tier). Every request sets `enable_thinking=true`, `max_tokens=8192` (reasoning and answer share it). Two
sampling arms: **matched greedy** (temperature 0, top-k 20, seed 1234) and the **roleplay sampler**: temperature 1.0,
top_p 0.95, top_k 64, min_p 0.03, a different seed per request. The launcher passes no sampling flags, so the sampler is
written explicitly in each request (Gemma's recommended 1.0 / 0.95 / 64 plus the SillyTavern preset's min_p 0.03). The server
default answer room was kept: the log shows `thinking N/6,144` for a request with an 8,192 limit.

**Per response** the harness records reasoning, answer and total tokens (from usage), `finish_reason`, empty final, wire TTFT,
MTP accepted/drafted, decode tok/s and errors, and stores the full text. A response is **complete** only if it ends with
`finish_reason=stop`, has a non-empty final answer and is not flagged as a loop; throughput is never a gate, because a loop
inflates both MTP acceptance and decode rate (some looping responses below report 99% acceptance, one garbage-token loop 33%). Loop flag, computed on
reasoning plus answer text: the last 400 words are periodic (best fraction of words equal to the word *p* positions earlier,
p = 1-64, at least 0.95), or the last 4000 characters compress to under 0.10 under zlib, or the last 500 words contain under
10% distinct words. Healthy replies compress to 0.33-0.55 and had at least 0.41 distinct words; none of the 425
`stop` responses is flagged and none of the 18 `length` responses with a 8,192-token cap is unflagged. A `repetitive` soft flag
(distinct word 8-grams below 0.5) is recorded for human reading. This is a stuck-output detector, not a quality judgement.

**Part A, cold prompts (~1k, ~16k, ~65k, ~122k; actual 988-992, 15,990-15,993, 64,990-64,992, 121,992-121,993 tokens),
completed / attempted and decode tok/s median [range] over completed and looping runs:** greedy reps 5 / 5 / 2 / 2 (the ~1k and
~16k points are the primary run plus the reversed-tier repeat, below) and sampler reps 5 / 5 / 2 / 1; each timed request is a
distinct document and one excluded 256-token warmup precedes them.

| Prompt | Tier | Greedy complete | Greedy decode tok/s | Sampler complete | Sampler decode tok/s | Wire TTFT s (greedy) |
|---|---|---|---|---|---|---:|
| ~1k | `best-medium` | 5/5 | 119.6 [115.4-121.0] | 4/5 | 109.8 [105.1-125.8] | 0.25 |
| ~1k | `best-small` | 5/5 | 125.6 [114.9-127.0] | 5/5 | 116.9 [115.4-121.9] | 0.25 |
| ~1k | `iq3-xs` | 5/5 | 123.7 [119.4-127.6] | 4/5 | 109.6 [106.0-133.8] | 0.25 |
| ~1k | `iq3-xxs` | 5/5 | 126.1 [122.8-128.9] | 4/5 | 117.1 [114.3-135.9] | 0.27 |
| ~16k | `best-medium` | 5/5 | 108.2 [105.4-110.4] | 5/5 | 106.2 [105.7-108.9] | 3.08 |
| ~16k | `best-small` | 4/5 | 115.0 [111.2-129.2] | 5/5 | 112.6 [107.6-114.5] | 3.06 |
| ~16k | `iq3-xs` | 4/5 | 110.6 [106.7-124.2] | 5/5 | 107.8 [103.0-111.3] | 3.26 |
| ~16k | `iq3-xxs` | 4/5 | 113.3 [110.2-128.2] | 5/5 | 110.8 [109.2-114.6] | 3.18 |
| ~65k | `best-medium` | 2/2 | 92.5 [92.4-92.7] | 2/2 | 87.7 [86.5-88.8] | 20.70 |
| ~65k | `best-small` | 2/2 | 97.1 [95.5-98.6] | 2/2 | 91.1 [89.9-92.2] | 20.64 |
| ~65k | `iq3-xs` | 2/2 | 92.2 [91.3-93.1] | 2/2 | 88.2 [86.7-89.8] | 21.34 |
| ~65k | `iq3-xxs` | 2/2 | 94.3 [94.1-94.5] | 1/2 | 79.3 [70.4-88.3] | 21.21 |
| ~122k | `best-medium` | 2/2 | 76.1 [75.9-76.3] | 1/1 | 74.0 | 56.86 |
| ~122k | `best-small` | 2/2 | 79.1 [77.6-80.6] | 1/1 | 80.1 | 56.62 |
| ~122k | `iq3-xs` | 2/2 | 76.8 [76.6-77.0] | 1/1 | 73.6 | 58.32 |
| ~122k | `iq3-xxs` | 2/2 | 77.8 [76.9-78.7] | 1/1 | 77.0 | 58.51 |

Rates of the 122k points are single or paired measurements. The item-1 transient did **not** recur: the reversed-tier repeat of the ~1k
and ~16k points (iq3-xxs first, best-medium last) gave the same decode rates as the primary order within run-to-run variation.
Every *completed* response was short: median reasoning 726-1,129 and answer 563-684 tokens (maximum over all 425 completed
thinking responses, soak and roleplay turns included: 3,777 reasoning, 4,351 total), so the 8,192 cap bound only looping responses.

**Thinking loops occur on every tier, with and without the sampler, and are more frequent on `iq3-xxs`.** All 18 thinking-on
responses that hit 8,192 tokens were degenerate: 16 never left the reasoning channel (reasoning 8,192, empty final answer,
`thinking 8,192/6,144` in the server line, i.e. the server's own 6,144-token thinking budget did not close the loop) and 2
looped inside the final answer. Overall completion over all 443 thinking-on responses
(Part A + repeat, Part B turns, Part C soak; labelled reruns included): best-medium 42/43, best-small 188/191, iq3-xs 41/43,
iq3-xxs 154/166. Excluding the roleplay turns (all 16-turn conversations completed): Part A + repeat + soak loop counts are
1/27, 3/143, 2/27 and 12/134. The same ~16k document looped under greedy on `best-small`, `iq3-xs` and `iq3-xxs` but not
`best-medium`, so loops are partly prompt-specific; the sampler loops were at other prompts. These counts are small and no
significance test was run.

Cause of the `thinking 8,192/6,144` rows: in the measured binary Gemma's thinking budget was accounting only, so the budget the
server computed was logged and never enforced. Later builds enforce it (the control span, the budget message and
`--reasoning-loop` as on Qwen, see [serving](../serving.md#openai-chat-completions)); the rows above are the unenforced
behaviour and were not re-measured.

**Part D, item-1's degeneracy condition under the roleplay sampler** (thinking off, three ~122k prompts from the long-context
baseline, cap 8192, seeds 5001-5003): `best-small` 1/3, `iq3-xs` 2/3 and `iq3-xxs` 2/3 generated until the 8,192 cap in a
degenerate low-vocabulary token loop (the same prompt on all three tiers); the other runs stopped on their own at
1,653-3,887 tokens. The greedy baseline had 0/6 on q4 and 5/6 on IQ3, so the sampler reduces the IQ3 rate in this small test
and shows that the q4 tier is not immune; it does not remove the IQ3 gap. No engine change was attempted.

**Part B, multi-turn roleplay (16 turns per tier, thinking on, sampler, cap 8192):** a synthetic fantasy road-journey
conversation with a ~50k-token compendium inside the system prompt (corpus text), scripted benign user turns, and, like a chat
frontend, only the `content` of each reply sent back as history. Prompt length grows from 50,147 to 55,036-55,494 tokens.
**All 16 turns completed on every tier** (finish `stop`, no loop, no empty final) at both host-tier sizes; every turn generated 451-2,183
tokens. Turn 1 (cold) TTFT was 14.3 (`best-medium`), 14.9, 14.6 and 15.5 s (`iq3-xxs`); later turns resumed 99.0-99.9% of the prompt
(`lane_resident`, plus one `hybrid_snapshot` on `iq3-xxs`) at median TTFT 0.46, 0.44, 0.85 and 0.78 s (maximum 0.52, 0.51, 0.95, 0.93).
With the 8192 MiB Host tier each conversation wrote 1,020-1,060 blocks and 19-30 snapshot images and left the tier at 8.23-8.34 GB of
8.59 GB with 12-23 snapshot evictions. A 20480 MiB tier (`best-small` and `iq3-xxs`, labelled, same 16 turns) also filled
(21.17-21.18 of 21.47 GB, 45 images written) and changed nothing visible: the same classes and TTFTs. Host restores were 0 (`best-small`) and 1 (`iq3-xxs`);
because turns are served from the lane, this does not exercise the Host tier. [INFERENCE] The sliding-window image dominates the Host bytes.

**Part C, soak** (45 minutes, one server each, mixed thinking on and off, greedy and sampler, ~1k to ~65k prompts, caps 1,024-8,192,
about 13% cached continuations, fixed RNG so both tiers received the same request sequence until continuations diverged):

| Tier | Requests | Finish stop / length | HTTP or transport errors | Loops (thinking-on) | Whole-device MiB: ready / first-10-min median / last-10-min median / max | Linear slope |
|---|---|---|---|---|---|---|
| `best-small` | 158 | 156 / 2 | 0 | 2 | 27510 / 27605 / 27780 / 28074 | +393 MiB/h |
| `iq3-xxs` | 143 | 134 / 9 | 0 | 9 | 23860 / 23879 / 23871 / 23969 | -20 MiB/h |

`best-small` stepped up by about 175 MiB between minutes 25 and 30 and was flat afterwards; `iq3-xxs` did not grow. Samples are two seconds
apart, include the idle desktop and can miss shorter spikes. Neither server died or returned an error; the 2 and 9 `length` finishes are the loops above (in the `iq3-xxs`
soak 7 stayed in reasoning and 2 looped in the answer). Slow cached turns were rare (one 3.3-3.4 s continuation on each tier whose prompt needed 16.7k tokens recomputed).

**Needs human reading:** every response's reasoning and answer are in the retained evidence; no quality was judged, the loop
detector's flags were not audited by a person, and the `repetitive` soft flag was not reviewed. Not measured: other samplers or
seeds, prompts above 122k with thinking, host tiers other than 8192 and 20480 MiB, or any comparison of reasoning quality between tiers.

## Route-specific measurements

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

FP8 KV and FP8 QK, 3 timed reps per length after a discarded warmup, one server per run (median
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

This earlier route comparison is not the matched four-tier study: its request thinking field was omitted.
Use [Four-tier comparison](#four-tier-comparison) for current, explicitly thinking-enabled tier comparisons.

### GGUF-block artifacts (IQ3_XS import): prefill and decode

`infernix_bench`, FP8 KV, BF16 logits, 3 reps after a warmup, against an absmax `q4_g256_fp16` build of the same model
(W4A8 fused/baked routes). Prefill, `-p 8192`, no MTP, tok/s by `--prefill-chunk`:

| Route | 128 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|
| q4_g256 (W4A8) | 3,655 | 4,681 | 5,535 | 6,049 |
| GGUF, AllowA8 (default `--weights q4_g32`): E4M3 rows + FP8 A8 GEMM | 2,146 | 4,498 | 5,300 | 5,922 |
| GGUF, A16Only (`--weights q4_g32_a16`): dense BF16 + cuBLAS | 1,198 | 2,182 | 2,436 | 2,529 |

Decode, chunk 2048, tok/s (tg128 / pp4096 + tg128): q4_g256 80.2 / 73.7; GGUF AllowA8 78.5 / 73.5; GGUF A16Only
62.6 / 59.9. With MTP (one draft, pp4096 + tg128, 183-186 of ~200 drafts accepted): q4_g256 124.2, GGUF AllowA8
117.8, GGUF A16Only 82.2. The bench's prompts are Qwen-tokenized ids, so MTP acceptance there is content-dependent.

Native KL (plain model, Q8_0 reference, chunk 128): A16Only IQ3_XS 0.163, IQ3_XXS 0.218 (equal to the exact GGUF
route, paired 0.000 [−0.001, +0.001]); AllowA8 prefill +0.013 [+0.008, +0.018] (IQ3_XS 0.176) and +0.010 [+0.005,
0.015] (IQ3_XXS 0.228). The 8-bit decode route against the exact one, scored at prefill chunk 16 (every projection
on its narrow route) against the exact run's own cache: KL 0.0035 [0.0027, 0.0043], dNLL −0.001 [−0.003, +0.001].

#### Combined-default IQ3 imports, real Gemma serving (2026-11-16)

Fresh artifacts store `AllowA8` on the GGUF projection Uses and the same FP8 MTP assistant. No temporary
activation environment switches are set. Native full-vocabulary KL uses the model variant's own Q8_0
reference: 47 English windows / 13,728 positions, context 512, stride 256, prefill chunk 128, FP8 KV,
BF16 QK/logits. Paired differences reuse the corresponding exact-A16 reports:

| Model / tier | Mean KL [95% CI] | ΔNLL | Top-1 disagreement | Paired default − A16 KL [95% CI] |
|---|---:|---:|---:|---:|
| plain IQ3_XS | 0.1762 [0.1534, 0.1996] | +0.0662 | 12.63% | +0.0129 [+0.0084, +0.0175] |
| plain IQ3_XXS | 0.2273 [0.1984, 0.2589] | +0.0826 | 14.40% | +0.0095 [+0.0050, +0.0146] |

The cost is measurable: all paired KL intervals exclude zero despite overlapping marginal intervals.
The diode IQ3 results are kept with the matched diode q4 results in the canonical four-tier comparison;
their paired default-minus-A16 KL costs are +0.0159 (XS) and +0.0120 (XXS), also excluding zero.
Causal scoring at chunk 128 predominantly exercises wide projections; it is not generated-output KL.

Real HTTP MTP: RTX 5090, FP8 KV/QK, BF16 logits, same assistant, draft depth 1, automatic prefill,
32,768 maximum context, no prefix reuse, greedy top-k 20 / seed 1234, **thinking explicitly disabled**.
Each length has an excluded warmup and three timed repetitions with identical cross-arm natural English
prompts tokenized by Gemma. Generation is capped at 256 tokens; actual prompt counts are 986–987 and
15,983–15,987. Rates are server timings, not wire TTFT; brackets are repetition ranges, not confidence intervals.
The q4 baseline is absmax q4_g256, a performance/format baseline, not quality-matched GPTQ.

| Arm | Prompt | Auto chunk | Prefill tok/s | Decode median [range] tok/s | Accepted / drafted |
|---|---:|---:|---:|---:|---:|
| q4 | ~1k | 2048 | 6,583 | 121.2 [117.7, 124.6] | 320 / 444 |
| IQ3_XS | ~1k | 2048 | 6,407 | 120.7 [119.2, 122.7] | 322 / 441 |
| IQ3_XXS | ~1k | 2048 | 6,331 | 125.5 [116.2, 127.2] | 315 / 448 |
| q4 | ~16k | 2048 | 5,502 | 115.9 [115.9, 116.4] | 327 / 436 |
| IQ3_XS | ~16k | 2048 | 5,390 | 112.4 [112.0, 114.7] | 321 / 441 |
| IQ3_XXS | ~16k | 2048 | 5,380 | 117.2 [116.6, 119.8] | 326 / 436 |

Observed prefill reaches 96.2–98.0% of q4. Strict median MTP decode parity is **not met by IQ3_XS**:
99.5% / 97.0% of q4 at ~1k / ~16k. IQ3_XXS reaches 103.5% / 101.1% on these prompts, with overlapping
short-run ranges. These measurements do not qualify every prompt length, thinking-enabled throughput or
smaller-card capacity. All four plain/diode default imports also passed loopback model-list and chat probes.
The canonical generated-launcher `serve-check` also passed all six checks on each of the four imports (24/24),
using the default 131072-context / FP8 KV/QK / auto-chunk / MTP d1 / 8192 MiB host-cache profile with only
host/port changed to development loopback. Those checks did not override the template's default thinking setting.


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
