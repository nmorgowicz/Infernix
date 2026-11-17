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
server computed was logged and never enforced. Current builds enforce the thought cap with a native `<channel|>` close,
a configurable budget message and the reasoning-loop guard; see [serving](../serving.md#openai-chat-completions).
The historical tables above remain unchanged; the focused enforcement checks below are separate, not a new stability campaign.

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

### Thinking-budget enforcement checks (2026-11-16)

Focused public-Engine GPU checks on the same diode `iq3-xxs` text+MTP artifact, RTX 5090 and CUDA 13.5 development runtime
passed. No production binary was replaced. The default message plus the native `"\n<channel|>"` close totals 24 tokens;
the control counts against the total output limit and is published in reasoning, not answer content.

With MTP at 3 and 7 drafts, each capped reply contained exactly **24 model thinking + 24 control + 40 answer tokens**,
`finish_reason=length`, with nonempty content. Every token was accounted for by first-token, draft-round, plain-step and
control counters. Plain decoding with the custom message `Time to act now:` used **24 + 7 + 40** tokens instead.
Insufficient post-budget capacity was refused; the smallest sufficient limit admitted the control and one answer token.
The final native test passed **88 assertions**, including changed-control prefix invalidation, live continuation and
displaced Host-snapshot restore. Free-cache-resume restored 1,113 tokens through the control ending at 1,018, but is not
a cold-prefill exact-output contract: its first difference was at zero-based output token 3. Reproducible chunk-grid
Host restore reused 1,024 tokens and generated the same 24 output tokens as a cache-off request.

The archived greedy `g1k-r0` request (992 prompt tokens), run with no budget and with budget 6,144 on the same native
executable and with prefix reuse disabled, was token-identical: **979 thinking + 631 answer = 1,610**, natural `stop`,
no injection. The archived `g16k-r2` request (15,990 prompt tokens), with the usual MTP d1 / FP8 KV/QK / BF16 logits /
auto-chunk profile, closed at **6,144 model thinking + 24 control** and emitted **2,024 answer tokens**, reaching
8,192 total with `length`. Its 8,432-byte final answer **still loops** (periodic-tail score 1.0, zlib ratio 0.0275,
distinct-word fraction 0.034). Closing the thought is enforced; a good or non-looping final answer is not guaranteed.

Real loopback HTTP also passed after the operator approved the development executable in the app firewall.
With `--default-thinking-budget 24 --thinking-budget-message "Time to act now:"`, a 128-token request with no budget
overrides produced **24 model thinking + 7 control + 51 post-close = 82 tokens**, natural `stop`. Sending
`"thinking_budget_tokens": 8` produced **8 + 7 + 113 = 128**, `length`. Both reasoning fields ended with the custom
message and both final answers were nonempty. A 31-token limit with the default cap returned HTTP 400
`thinking_budget_capacity_insufficient`; `/health` returned 200. This HTTP smoke used plain decoding, FP8 KV/QK,
BF16 logits, chunk 128 and context 4,096; MTP is covered by the native checks above.

Evidence: `<camp>/convert/iq3xs-20261115/gemma-budget-20261116/` (`real-host-oracle.log`, archived request outputs,
`archived-native/analysis.json`, `http-open.{results.json,requests.jsonl,server.log}`); GPU lease receipts
`gemma-budget-host-oracle`, `gemma-budget-recovery-final` and `gemma-budget-http-approved`. These are behavior checks,
not throughput, quality, soak or cross-build identity measurements. Earlier connection-timeout failures and test-oracle
failures are retained: the app firewall caused the former; the latter incorrectly assumed free-resume/cold token
identity and fixed-length raw-history output with model turn stops enabled. No engine failure was hidden by either fix.

## 16 GB-class capacity envelope (simulated available memory on an RTX 5090)

Everything in this section is **simulated available memory on an RTX 5090**, not a physical 16 GB card: no claim is made about
a real card's fit, driver behaviour, display load or speed. Prefill and decode rates are 5090 relative indicators only.
Artifacts, binary (`f0073e790fb9`), template and request shape are the same as above; `iq3-xxs` and `iq3-xs` text + FP8 MTP files,
FP8 KV and FP8 QK, `--max-concurrency 1`, `--vram-headroom-mib 1024` (not binding here), 4096 MiB host cache (host RAM, not VRAM),
one server per cell, thinking off, greedy. Every cell loads the server, then sends a near-capacity prompt (`min(context, KV capacity)`
minus 450 tokens, 256 generated) and two ~1k prompts, so a configuration is demonstrated by running, not by loading.

**Ballast.** A torch process holds VRAM until `nvidia-smi` reports the intended free memory, verified before every load
(`summary.json` in each variant directory). Simulated card total 16303 MiB (RTX 5080-class `memory.total`). The server-usable
memory is the card total minus a resident reserve: **nominal** 14,767 MiB (1.5 GiB for display, driver and other CUDA contexts),
**conservative** 13,743 MiB (2.5 GiB, which is what this machine's desktop uses at idle), **headless** 15,791 MiB (0.5 GiB, no display).
These are assumptions, not measurements of any 16 GB card. This 5090 reports 32,607 MiB but its Windows budget is 31,419 MiB, so
the ballast is set to usable + 1,161 MiB; ballasting only by `nvidia-smi` free let the server run past the budget and page
(a first probe at 4k, default chunk: prefill 273 tok/s and decode 21 tok/s). A cell is `fit` only when the whole-device peak used
(`nvidia-smi`, 0.4 s samples, including the desktop and the ballast) stays within 31,419 MiB; `marginal` is over that line with decode
within 70% (and prefill within 60%) of the no-ballast reference; `degraded` is over and slower; `FAIL` is a refusal. A sampler can miss
short spikes, and over-the-line `marginal` cells run at full speed here, which means the ballast is partly evictable: only `fit` is a demonstrated limit.

**What sets the footprint.** Server-added peak MiB at 4,096 context, no ballast: `iq3-xxs` with MTP **14,172** (prefill chunk 128),
14,191 (256), ~14,470 (512) and **16,833 at the default `auto` chunk (2048)**; without MTP 13,695 (128) and 16,389 (`auto`).
The plan and logs give the parts: weights 12,385 MiB with MTP and 11,921 MiB with `--spec off` (the plan and the `GPU weights` log line drop the
464 MiB assistant when speculation is off); the sliding-window ring **900 MiB at chunk 128 against 2.34 GiB at `auto`**; prefill workspace 274, 306, 370, 765 and
1,530 MiB at chunks 128, 256, 512, 1024 and `auto`; global FP8 KV 41,280 B per token (161 MiB per 4k, 5.04 GiB at 131,072);
graph allowance 64 MiB; and about 290-390 MiB that no plan line names (CUDA context and kernels). `iq3-xs` adds about 840 MiB over `iq3-xxs`
(its weights). So at a 16 GB-class card the **default chunk cannot fit** (16.4-16.8 GiB needed at only 4k context); chunk 128-512 is what
makes `iq3-xxs` fit, and the usable memory left over buys context at 0.04 MiB per token. `--prefill-split-workspace-mib 64/128` and
`--no-cuda-graph` did not change the plan's workspace; `--no-cuda-graph` saved 16 MiB (14,155 to 14,139 at 4k, chunk 128) and cost
about 15-20% decode (127 to 106 tok/s with MTP, 84 to 65 without), so it is not a useful fit lever.

Largest demonstrated context by configuration, **`fit` / including `marginal`** (ladder 4k, 8k, 12k, 16k, 20k, 24k, 32k, 40k, 49k, 65k; each
ladder stops after two consecutive non-`fit` rows; chunk 256 was tested only at 4k and 16k):

| Tier, MTP | Chunk | Headless 15,791 MiB | Nominal 14,767 MiB | Conservative 13,743 MiB |
|---|---|---|---|---|
| `iq3-xxs`, MTP d1 | 128 | **40,960** / 49,152 | **16,384** / 24,576 | none (4,096 marginal, +472 MiB over) |
| `iq3-xxs`, MTP d1 | 512 | 32,768 / 49,152 | 8,192 / 16,384 | none (4,096 marginal) |
| `iq3-xxs`, no MTP | 128 | **49,152** / 65,536 | **24,576** / 40,960 | **4,096** / 6,144 |
| `iq3-xxs`, no MTP | 512 | 40,960 / 40,960 | 20,480 / 32,768 | none (4,096 marginal) |
| `iq3-xs`, MTP d1 | 128 | 20,480 / 32,768 | none (8,192 marginal) | not run |
| `iq3-xs`, MTP d1 | 512 | 12,288 / 20,480 | none (8,192 marginal) | not run |
| `iq3-xs`, no MTP | 128 | 32,768 / 40,960 | 8,192 / 16,384 | not run |
| `iq3-xs`, no MTP | 512 | 24,576 / 40,960 | none (8,192 marginal) | not run |

Default `auto` chunk at these sizes never fit: every ballasted `auto` cell (4k context, `iq3-xxs`, headless and nominal and two scan points) was
`degraded`, with near-capacity prefill 515-2,311 tok/s and decode 26-54 tok/s against 5,547 and 125 unconstrained. The 65,536-context
`iq3-xxs` MTP chunk-128 cell (headless) is `degraded` (prefill 343, decode 19-47); the no-MTP chunk-128 cell at 65,536 is `marginal`
(1,571 / 71-83). Nothing above 65,536 was run on a ballasted card.

Relative decode and prefill, `iq3-xxs` on the 5090 (near-capacity prompt, then ~1k prompts): MTP d1 decodes at 125-135 tok/s
(120-124 at 16k, 100-111 at 32k-65k) against 82-85 (80-84, 71-78) without it, +45-50% at the same context; `iq3-xs` 113-127 against 80-81.
MTP costs 464 MiB of weights plus about 13 MiB of draft state (14,172 against 13,695 at 4k), which is 8,192 tokens of context in both
the headless and nominal chunk-128 ladders (49,152 to 40,960 and 24,576 to 16,384). At this card class that is a trade of
roughly a third of the context for about +50% decode; it is worth it when 16k or more is enough [INFERENCE on workload; the rates are 5090 ones].
Chunk costs prefill: near-capacity prefill of `iq3-xxs` MTP is 1,715 / 1,972 tok/s at 4k / 16k with chunk 128, 2,410 / 2,984 at 256,
2,775 / 3,619 at 512 and 5,547 / 5,256 at the default `auto` (unconstrained), i.e. chunk 128 costs about 2.7-3.2x of the default and chunk 512 about 1.5-2.0x.

**Refusals.** 17 of 161 ballasted cell attempts (and 0 of 10 unconstrained) were refused at startup, all with
`FATAL server failed during startup | Gemma4 CUDA graph preparation exceeded its planned device allowance; reduce prefill_chunk or disable CUDA graphs`,
including cells with more than 1 GiB spare (for example headless `iq3-xxs` no-MTP chunk 512 at 4k), and not monotonic with context. A retry
of the same configuration succeeded for every refused ladder cell (rerun in a later lease, or by the harness's own two retries added after the first
ladders); all refused attempts are retained. The two refusals that remain (scan cells, `iq3-xxs` MTP chunk 512 at 4k, 15,500 and 15,200 MiB usable) were not retried. The 14
`--no-cuda-graph` cells were never refused. [INFERENCE] The allowance check reads free memory around graph capture and is perturbed by other
processes' memory changes in this simulation; it was not isolated.

**Demonstrated launcher flags for a 16 GB-class card** (add to the usual profile; every value below was run with this serve binary):

- No display (headless), `iq3-xxs`, MTP: `--spec mtp --draft-tokens 1 --prefill-chunk 128 --max-context 40960 --kv-dtype fp8 --gemma4-qk fp8
  --host-context-mib 4096`; no MTP: `--spec off` with `--max-context 49152`. `--prefill-chunk 512` with `--max-context 32768` trades context for 1.5x faster prefill.
- Display-sharing card (1.5 GiB reserved), `iq3-xxs`: MTP with `--prefill-chunk 128 --max-context 16384`; without MTP `--max-context 24576`.
  `iq3-xs` fits only without MTP at `--max-context 8192`.
- A card with 2.5 GiB reserved: only `iq3-xxs` without MTP at `--max-context 4096` demonstrated; no context worth serving.

The default launcher profile (131072 context, `--prefill-chunk auto`) does **not** fit any of these simulated cards. Not measured: a physical 16 GB
card, Linux or a headless driver stack, any reserve other than the three above, soak, thinking-on generation, contexts above 65,536 under ballast,
`iq3-xs` at the conservative reserve, and quality at the reduced chunk (the chunk changes scheduling, not the stored weights;
no KL was scored for it).

## Final-layer early exit in intermediate prefill chunks (2026-11-16)

The default [mechanism](engine-architecture.md) skips the last layer's attention, output projection and FFN
in prefill forwards whose final hidden state nothing reads, after that layer's K/V is written.

**Historical equivalence (switch unset against `=1`, separate processes, same build).** All of these matched exactly:
- Native direct calls: every BF16 and global FP8 code/scale plane after full versus KV-only forwards, at intermediate and final boundaries,
  and final full-vocabulary logits. 32 cells (four diode artifacts x chunk 128/2048 x CUDA graphs off/on x MTP off/on), 160 fingerprints
  (selecting-distribution, readout, token, MTP counters), Host snapshot restore and two-intermediate-chunk root replay equal to cold generation.
- HTTP: 112 requests (four artifacts, chunk 128 and `auto`, MTP on/off), responses, usage and speculative counters identical.
- Native full-vocabulary KL at chunk 128 (pinned Q8 reference, BF16 QK, FP8 KV): reports and position sidecars identical
  (mean KL 0.3154030109 in both arms). Chunk 512 cannot use that reference (the pinned reference guard refuses a chunk other than 128, unchanged), so
  chunk 512 compares causal NLL and the saved top-256 sidecar of the same input (`diode-IQ3_XXS`): report fields other than timings and
  the sidecar path, and the 28,171,234-byte sidecar, are identical. No Q8 KL was scored at chunk 512.

**Cold-prefill timing.** RTX 5090, one build (`build-infernix-prefill-earlyexit-cuda13.5`, `infernix-serve.exe`), serialized, default 8 GiB RAM
floor, one fresh server per arm, FP8 KV and FP8 QK, BF16 logits, MTP d1, thinking off, greedy, 8 output tokens, max context 131072, 4096 MiB
Host tier, prompts byte-identical across arms. `--no-fast-path free-cache-resume` is set in both arms (otherwise six common opener tokens
are reused from the live lane even for distinct prompts) and every timed request reported `cached_tokens` 0 with equal prompt and computed counts.
**This is a changed timing profile**: the earlier mixed-cache instrument is not comparable to these rates. Each cell has one excluded warmup, then
3 (1k) or 2 (65k, 122k) timed requests per arm; AB and BA orders (BA also reverses tier order) are reported separately. Ranges are repetitions,
not confidence intervals. Rate = server-reported computed prompt tokens per second.

| Chunk | Prompt (nominal) | Rate change, all four tiers, AB and BA medians | Example (best-medium, A to B tok/s; wire TTFT) |
|---|---|---|---|
| 128 | 1k | +0.69% to +2.63% | 3,668 to 3,731 |
| 128 | 65k | +3.19% to +5.41% | 2,292 to 2,414; 28.5 to 27.1 s |
| 128 | 122k | +5.19% to +6.88% | 1,673 to 1,787; 73.2 to 68.5 s |
| 2048 (`auto`) | 1k | -0.56% to +0.71% (no-op; single-chunk prompt) | 6,496 to 6,460 |
| 2048 (`auto`) | 65k | +6.13% to +6.43% | 3,215 to 3,415; 20.3 to 19.2 s |
| 2048 (`auto`) | 122k | +7.26% to +8.01% | 2,182 to 2,348; 56.2 to 52.3 s |

All 48 cells (4 tiers x 2 chunks x 3 prompts x 2 orders; 32 server runs) completed with no failure, no cached prefix and identical output
text, token counts and draft counters between arms. AB and BA gains agree within about one point (largest gap: IQ3_XS chunk 128 at 65k, +3.19%
against +4.08%). The 1k prompt at chunk 2048 is one chunk whose last call is complete, so no skipping applies and its spread is noise.
[INFERENCE] The gain growing with prompt length points at the last layer's long-context attention being a larger share of each intermediate chunk than its
weight-GEMM cost; this was not profiled per layer. That campaign did not measure decode or MTP-round speed, physical 16 GB cards, free-cache-resume equality
(that route's partition arithmetic differs from cold by contract), and any tier or chunk other than those above.

Full tables, receipts and drivers: the camp's `convert/iq3xs-20261115/prefill-earlyexit-20261116/` (`results.md`, `results.json`, `receipts/`).

**Default cutover and decode (2026-11-17).** The switch is removed. The rebuilt development binary's real-artifact
native test passed all eight diode IQ3_XXS cells (chunk 128/2048, graphs and MTP on/off); 36 unique fingerprints
matched the retained switch-off baseline across builds. Full versus KV-only calls compared BF16 and FP8 KV planes
and full-vocabulary logits; generation, readout, scoring, root replay and Host snapshot restore passed. This is not a
same-build on/off comparison.

Real HTTP on the rebuilt binary (`a81341d60b1d`), RTX 5090, diode IQ3_XXS text+MTP artifact, FP8 KV/QK,
BF16 logits, auto chunk 2048, max context 131072, 4096 MiB Host, free-cache-resume disabled, thinking off,
greedy top-k 20, seed 1234: one excluded warmup and three timed distinct prompts per length, 256 output tokens
each, all `length`, with zero cached tokens. Fresh server per mode, plain first then MTP d1; no reversed-order control.
Rates are server decode timings; ranges are repetitions, not confidence intervals.

| Prompt tokens (timed) | Plain tok/s median [range] | MTP d1 tok/s median [range] | Accepted / drafted (3 reps) | MTP ms / round proxy median [range] |
|---|---|---|---|---|
| 984–986 | 83.85 [83.32–84.18] | 122.27 [121.48–125.67] | 312 / 451 | 13.80 [13.72–13.81] |
| 64,983–64,985 | 71.37 [70.65–71.59] | 94.74 [94.39–102.26] | 331 / 431 | 18.13 [18.07–18.69] |
| 121,983–121,985 | 62.18 [60.79–62.53] | 76.16 [73.94–77.63] | 306 / 457 | 22.05 [21.88–22.25] |

The round figure is **not a direct device-round timer**: aggregate server `predicted_ms` divided by drafted tokens
(one draft per speculative round), including any fallback and other decode overhead. It is a duration proxy, not
an isolated MTP kernel latency. Plain versus MTP differs in backend and acceptance; these measurements do not isolate
the early exit's effect on decode. Historical prefill A/B used eight-token outputs, so it is not a matched decode baseline.
The 24 HTTP requests succeeded under the guarded GPU lease (minimum free RAM 29.33 GiB); raw responses, logs,
driver and summaries are in `prefill-earlyexit-20261116/default-20261117/`.

## Key-only global KV cache (experimental, off by default)

`--gemma4-key-only-kv` (explicit opt-in; omission keeps the default K==V FP8 layout) stores only the
FP8 K codes and scales of the global (D512) layers and rebuilds V in attention. It requires `--kv-dtype fp8`; other
formats are refused. Sliding layers keep BF16 K and V.

**Why V can be rebuilt.** Gemma 4's global layers have no V projection: `gemma4_qkv_prep` forms V as the raw K row after
the weightless norm, and K as the same row times the key-norm weight, then RoPE on the 128-wide rotated prefix (pairs
`(i, i + 64)` after the converter's `prope_perm`). When the key-norm weight is one constant `c`, `V = unrope(K) / c`. The
converter records such constants as `global_k_norm_scalar`; the Program refuses the layout unless every global layer has
one and the loaded BF16 weights equal it (all ten global layers of the 31B artifacts do, c = 0.0600-0.0654).

**What runs.** `KvStorageModel::KeyOnly` is its own layout identity (`/key-only` in the prefix-cache identity, different
block and image fingerprints), so its Host blocks and snapshots never restore into a K==V cache or the reverse; block
records hold K codes and scales only. Cache views carry `Gemma4ValueSource::Key` with `1 / c`, the RoPE base and
the rotary width. The writer stores K only. The D512 decode kernel turns each staged K row into its V row (no V row is
read); the tensor-core kernel stages the V tile from the K pages and expands it the same way. Both form the writer's FP32
angle, reduce it to [-pi, pi] and rotate back, multiply by `1 / c` and round each V element once to BF16. The generic
SIMT path (query groups above 8) refuses the layout. The MTP assistant reads the same views. The memory summary, the
startup `Gemma KV` log line (`fp8 key-only`) and the request log's `memory.gemma4_kv.global_key_only` report the layout;
the load summary's prefill signature carries `/k-only`.

**Bytes.** Global KV per token 41,280 -> 20,640 B on the 31B geometry (10 global layers x 4 KV heads); at 131,072
tokens 5.04 -> 2.52 GiB. Sliding rings, workspace and prefix images are unchanged.

**Tests.** `infernix_ops_gemma4_attention_test gemma4_key_only_kv` (cache validation; the writer's K planes byte-equal
to the K==V writer's; `unrope(K_bf16) / c` against the stored BF16 V; decode and tensor-core attention, BF16 and FP8 QK,
split context and graph replay against an FP64 oracle up to position 262,143; attention error of key-only and stored FP8
V against BF16 K/V from the same writer input). `infernix_gemma4_key_only_real_test` (`INFERNIX_GEMMA4_ARTIFACT`):
refusals, K bytes equal to the K==V cache through the first global layer, K-planes-only accounting, deterministic
generation, root replay and Host snapshot restore equal to the cold key-only run with MTP off and on.

**Historical measurement setup (2026-11-17, RTX 5090, diode IQ3_XXS with its MTP assistant, one build):**
the retained campaign compared `INFERNIX_TMP_GEMMA4_KEY_ONLY_KV` unset against `=1`. That temporary
environment interface is removed; current commands use `--gemma4-key-only-kv`. The measured results
below retain their original build/setup identity, rather than being relabelled as new-CLI qualification.
Camp evidence: `iq3xs-20261115/konly-20261117/` (drivers, raw requests/responses, server logs, `results.json`).

- Op level, same writer input, attention error against BF16 K/V (stored FP8 / key-only): decode 0.0379 / 0.0379,
  prefill 0.0428 / 0.0433, decode at position ~262k 0.0553 / 0.0561, FP8 QK prefill 0.0551 / 0.0554. First global
  layer of the real model, V error against BF16 V: stored FP8 0.0260, rebuilt 0.0261.
- Native full-vocabulary KL against the diode Q8 reference (context 512, stride 256, chunk 128, BF16 QK): K==V FP8
  0.3154030109 (unchanged), key-only 0.3168307746; top-1 disagreement 0.1684 / 0.1658; mean delta NLL 0.1585 / 0.1564.
- Long context, top-256 KL of key-only against K==V FP8 (context 32768, stride 16384, 12 English/code streams, 809,397
  positions, chunk 2048, no chat prefix): mean 0.379, top-1 agreement 0.780, NLL 7.872 -> 7.881. By preceding context:
  0.361 (<1k), 0.356 (1-2k), 0.445 (2-4k), 0.469 (4-8k), 0.473 (8-16k), 0.355 (16-32k): no growth with position. This
  instrument has no chat prefix and its baseline NLL is near-degenerate (7.9 against 2.6 with the prefix), so these KLs
  are between two high-entropy distributions; they are not a quality figure on the pinned scale.
- Needle retrieval (three 7-digit codes at 10/50/90 % depth, greedy, MTP off, ~16k/65k/120k prompts, four each): 36/36
  for both arms.
- Capacity (`--plan`, MTP d1): global KV at 131,072 tokens 5.04 -> 2.52 GiB; planned KV 7,560 -> 4,980 MiB (chunk
  `auto`) and 6,060 -> 3,480 MiB (chunk 128); at 262,144 tokens 12,720 -> 7,560 MiB (`auto`). Serving at 131,072 tokens,
  whole-device peak 23,243-23,311 -> 20,665-20,693 MiB plain and 23,701-23,745 -> 21,126-21,143 MiB with MTP.
- Simulated 16 GB-class card (the ballast method of the 16 GB envelope section, headless variant, 15,791 MiB usable;
  MTP d1, chunk 128, near-capacity prompt plus 256 decode): key-only 40,960 fits (14,820 MiB added), 65,536 fits
  (15,305), 81,920 tight (15,602). The K==V FP8 ladder there: 40,960 fit, 49,152 marginal.
- Speed, matched instrument of the early-exit section (131,072 context, FP8 KV/QK, auto chunk, 256 greedy tokens,
  free-cache-resume off, AB and BA orders; medians of 3 timed requests, K==V FP8 -> key-only):

| Prompt | Prefill tok/s | Plain decode tok/s | MTP d1 decode tok/s (accepted / drafted) |
|---:|---|---|---|
| ~1k | 6,294-6,351 -> 6,358-6,463 | 82.6-82.7 -> 83.7-84.0 | 119.8-122.1 -> 123.6-123.8 (312/451 -> 312/452) |
| ~65k | 3,351-3,386 -> 2,951-2,972 (-12 %) | 71.0-71.2 -> 68.8-69.2 (-3 %) | 96.5-97.7 -> 87.9-88.5 (331/431 -> 306/456) |
| ~122k | 2,320-2,329 -> 1,979-1,983 (-15 %) | 61.6-62.5 -> 59.7-60.5 (-3 %) | 75.2-75.4 -> 71.6-72.7 (306/457 -> 313/450) |

The rebuild (a sine and cosine per rotated pair and key) costs more than the halved V read saves at these lengths;
outputs differ from K==V FP8 (a different lossy V) but are identical across AB and BA runs of the same arm.

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
