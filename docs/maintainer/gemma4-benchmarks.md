# Gemma4 benchmark tools

The workloads in `tools/bench/kv_eval/`, `rp_ladder.py`, `bench_tail.py`,
`fixed_tail.py`, `decode_abab.py`, and `retrieval-smoke.py`
run from this checkout,
use local artifacts and corpus files, and do not import any checkout outside this repository.
They do not download models or corpus data. Run commands from the repository root
with Python 3.11 and the dependencies in `tools/bench/kv_eval/requirements.txt`.
`derive_prefix_ids` additionally uses Transformers, with local files only.

All executable, artifact, corpus, reference and output paths below are placeholders
chosen explicitly by the user. Use a fresh output directory for each campaign.
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

Real loopback HTTP also passed once the development executable was allowed in the app firewall.
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

## 16 GB profile qualification

**Setup (2026-11-17, RTX 5090, simulated capacity, not a physical 16 GB card).** The diode IQ3_XXS text+MTP artifact, key-only
FP8 KV, chunk 128, `--kv-dtype fp8 --gemma4-qk fp8`, under the ballast method of the envelope section (ballast set to usable + 1,161 MiB;
the Windows budget line is 31,419 MiB; a desktop idles at 1.9-2.7 GiB on this machine, so the envelopes are headless 15,791,
nominal 14,767, ~1.9 GiB reserve 14,357 and conservative 13,743 MiB usable). Camp evidence: `iq3xs-20261115/preset16-20261117/`
(`ladder.py`, `appopen.py`, `qualify.py`, per-cell logs/SSE/`results.json`, plan breakdown `plan-chunks.json`).

**Footprint (added MiB, headless MTP, 9,216 tokens).** Weights 12,385 (11,921 without MTP); sliding ring plus key-only KV 1,081 (ring 900
at chunk 128, KV 0.0197 MiB per token); prefill workspace 274; graph allowance 64; state 3; planned 13,808 (13,343 without MTP). Measured
server-added 14,176. The engine's ledger shows 103-146 MiB past the plan at startup (modules loaded at first use, graph state)
plus about 3 KiB per context token (23.2 against 20.2 MiB per 1k tokens). Scaling with chunk (plan, 9,216 tokens): ring+KV 1,081 / 1,181 /
1,381 / 1,781 MiB and workspace 274 / 306 / 370 / 765 MiB at chunk 128 / 256 / 512 / 1024, so minimum reservation 1,264 / 1,400 / 1,672 / 2,482 MiB.
Chunks are multiples of 128: 64 is not supported. The measurements of this section use the BF16 sliding ring (`memory-16gb` now defaults to the FP8 ring: see [Adoption decision](#adoption-decision-and-coverage-limits)); `--gemma4-sliding-kv fp8` halves it ([FP8 sliding ring](#fp8-sliding-ring-experimental-off-by-default));
`--no-cuda-graph` saves the 24 MiB graph state at about -15-20 % decode.

**Context ladder** (explicit context and KV capacity, headroom 0, 48-token request at near-capacity prompt; `fit` = starts, answers, no spill;
cell value = MiB below the 31,419 MiB budget line at peak, so including a ~385 MiB gap between that line and the engine's own resident reading):

| Envelope | MTP | fitting contexts (spare MiB) | refused |
|---|---|---|---|
| nominal 14,767 | d1 | 4,096 (653), 8,192 (587), 9,216 (594), 12,288 (535), 16,384 (423), 20,480 (347) | not run above |
| nominal 14,767 | off | 9,216 (1,021), 16,384 (885), 24,576 (717), 40,960 (396) | 32,768 (spill, free left 72 MiB), 49,152 (short 22 MiB) |
| ~1.9 GiB 14,357 | d1 | none | 4,096: short 8 MiB |
| ~1.9 GiB 14,357 | off | 4,096 (620), 8,192 (534), 9,216 (515), 12,288 (467), 16,384 (366), 20,480 (272) | 24,576 (short 64 MiB) |
| conservative 13,743 | d1 / off | none | 2,048: short 569 / 124 MiB |

**Safe-fit criterion.** The earlier `fit` line (whole-device peak within 31,419 MiB) is a budget line, not a safe rule. Startup spills
(`Gemma4 runtime spilled into host memory`) occurred in explicit starts that left 72-287 MiB free, and a graph-preparation check measured
327 MiB where another program had allocated meanwhile; both are free-memory drift of other programs, which the engine now separates
from this process's own usage (per-process DXGI usage for graphs; a shortfall with this process's shared-memory use unchanged is not a
weights spill). The rule the profile uses is in engine terms: after startup the resident free memory (NVML free less the budget gap latched at the
first reading) must stay at least `--vram-headroom-mib` (profile default 64), and auto context additionally keeps 160 MiB plus 4 KiB per token
for what the plan does not name. Refusals report the shortfall in MiB.

**A program opening after launch** (nominal, no MTP, 16,384 context, headroom 0; a second process allocates 256-2,300 MiB more, up to the
physical card, while the server serves cached 8k-token requests): decode stayed 80-83 tok/s against 82 tok/s before, no errors, no device
memory recoveries, whole-device used reached 32,093 of 32,607 MiB; after the program exits the server's whole-device use had dropped by
~1.3 GiB (WDDM demoted idle allocations) and decode stayed 78-82 tok/s. So an app opening later did not page the weights or slow decode in
this test; this was a cached-prefix workload on the cells above and no new prefill of a long prompt, and one run per pressure level.
Headroom therefore protects the next startup and spill-guard noise, not a promised later spike.

**Profile qualification** (generated launcher CMD, default MTP-first with fallback, thinking on, output cap 8192, sampler 1.0/0.95/64/0.03,
seeds 4100/4101/4200; two ~1k prompts exercise prefix reuse, a third fills the resolved context less 8,320 tokens; over-context request must
return HTTP 400 and an explicit `--max-context 131072` start must be refused; all cells below did):

| Envelope | launcher result | resolved context | decode tok/s | requests |
|---|---|---:|---|---|
| headless 15,791 | MTP | 58,752 | 119-121 short, 92 at a 50,422-token prompt | 3/3 `stop`, TTFT 29.7 s at 50k |
| nominal 14,767 | MTP | 11,392 / 13,248 (two runs) | 115-121 | 3/3 `stop` each |
| nominal 14,767 | third run | refused (short 24 MiB, another program held ~450 MiB more) | - | clean refusal, nothing paged |
| ~1.9 GiB 14,357 | no MTP (fallback) | 13,184 | 82-83 | 3/3 `stop` |
| ~1.9 GiB 14,357 | `--vram-headroom-mib 0` | 16,192 | 81-83 | 3/3 `stop` |
| conservative 13,743 | refused | - | - | 9,216 tokens short by 512 MiB at the default |

Outputs were 1,310-1,482 tokens (697-864 reasoning) and ended with `stop`; no request reached the 8192-token cap, loop flags were false and
no quality rating follows. Rates belong to the RTX 5090.

**What does not fit, and why.** Conservative (13,743 MiB): without MTP 9,216 tokens need 1,422 MiB beside the 11,921 MiB of weights
(the engine plan with ring, workspace and graphs), 1,169 MiB are available after weights: **short by about 253 MiB** with no headroom and no
margin, 512 MiB at the profile default; MTP adds 464 MiB of weights (short about 717 MiB). This historical result
used BF16 rings. The later [FP8-ring ladder](#fp8-ring-desktop-ladders-2026-11-18) demonstrates conservative
9,216/12,288-token starts without MTP; `memory-16gb` now defaults to the FP8 ring on the strength of that ladder, accepting the measured fidelity and prefill costs.
Smaller weight tiers and unsupported chunk 64 were not implemented by that study; `--no-cuda-graph` saved 24 MiB at
about -15-20 % decode, and moving the display to an iGPU was not tested.
The vocabulary/output head (FP8, ~1.3 GiB) cannot leave the device without paging.

Retained process finding: the Windows OS budget shrinks to usage plus free memory as device memory fills (31,419 MiB at open, 17,154 MiB after the
weights under the ballast), so resident sizing latches the budget gap at the first reading.

## Shared GPU memory (`--allow-shared-memory`)

**What it is.** Default behavior is unchanged: a Gemma 4 start that cannot keep weights and runtime resident is refused. With
`--allow-shared-memory` (opt-in, Gemma only) startup continues: `--max-context auto` takes the 9,216-token minimum when nothing fits,
explicit contexts are never reduced, capacity checks count the reservation of the requested context, the weights and runtime spill
guards record instead of throwing, and a WARN line states "SHARED GPU MEMORY ALLOWED", the placement and the estimated bytes in shared
memory (allocated bytes missing from the drop in free VRAM, weights and KV/ring/workspace separately). The `memory-16gb` launcher generated
with `--allow-shared-memory` first tries resident starts (plan with MTP, plan without) and only then starts without MTP with the flag. Refusal
messages name the option.

**Measurement method and its limit (2026-11-17, RTX 5090, no MTP, key-only, chunk 128, 9,216 tokens, 8,501-token prompt then 128 tokens).**
An idle torch ballast does not create real overcommit: Windows demoted the idle ballast first, so even a 4 GiB estimated shortfall ran at
77-80 tok/s with whole-device use staying near 32 GB. The cells below therefore use an **active ballast** (every held chunk touched each
10 ms; `vram_ballast_active.py`), whose baseline is lower because of GPU contention: 72 tok/s decode and 1.74k tok/s prefill with no
spill. One run per cell, the GPU lease held and no build or quantizer overlapping; a CPU-side conversion of another job may have run
alongside, so quote the numbers as estimates.

| Usable VRAM | Estimated shortfall (weights / runtime) | Placement | Decode tok/s | Prefill tok/s |
|---|---|---|---:|---:|
| 16,000 (no spill) | 0 / 0 | - | 72.2 | 1.74k |
| 13,743 (2.5 GiB reserve) | 0 / 224 MiB | runtime | 72.2 | 1.73k |
| 13,743 | 0 / 231 MiB | blind | 72.0 | 1.73k |
| 13,743, context 16,384 | 0 / 367 MiB | runtime | 73.8 | 1.82k |
| 13,743, context 24,576 | 0 / 532 MiB | runtime | 72.0 | 1.77k |
| 12,543 | 108 MiB / 670 MiB | runtime | 23.0 | 677 |
| 12,543 | 183 MiB / 1.35 GiB | blind | 72.5 | 1.75k |
| 11,543 | 975 MiB / 703 MiB | runtime | 30.3 | 549 |
| 11,543 | 1.16 GiB / 1.35 GiB | blind | 28.6 | 744 |
| 10,543 | 1.91 GiB / 727 MiB | runtime | 22.9 (18.5 repeat) | 576 |
| 10,543 | 1.91 GiB / 1.35 GiB | blind | 24.9 (17.1 repeat) | 672 |

**Reading.** (1) At the conservative 13,743 MiB envelope the whole shortfall is 0.2-0.5 GiB and the cost is not measurable: 9,216, 16,384
and 24,576 tokens all run at 72 tok/s, so the 2.5 GiB-reserve desktop that the default refuses is usable with the option (minimum measured
context 2,048 with 0 B estimated shortfall; the idle-ballast run of the same envelope decoded 84 tok/s). Auto takes only the 9,216-token
minimum there; pass `--max-context` and `--kv-capacity` explicitly for more. (2) Beyond roughly 0.8 GiB short the driver places hot
allocations in shared memory and decode falls to 18-30 tok/s (-58 to -75 %) and prefill to 550-750 tok/s (-57 to -68 %); one 12,543 cell
with a larger estimate was unaffected, so the estimate does not predict the cost and the driver, not the engine, decides which
allocations spill. (3) Which allocations spill cannot be read per allocation; the estimates split into weights and runtime, and both
grow with the shortfall. (4) Keeping the runtime resident and spilling weights (`--shared-memory-placement runtime`, the default) was not
consistently better than `blind` (23 against 72 tok/s at 12,543 and 30 against 29 at 11,543): there is no demonstrated placement win, and
a placement that spills only cold weights does not exist because every layer's weights are read for every token.
Launcher end to end (`conservative`, staged launcher, idle ballast): planning with and without MTP failed, the `WARNING` line was printed,
the server started with the warning, 9,216 tokens and an estimated 237 MiB in shared memory.

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
formats are refused. The sliding ring follows `--gemma4-sliding-kv` (BF16 by default).

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

## FP8 sliding ring (experimental, off by default)

`--gemma4-sliding-kv fp8` (`bf16` keeps the BF16 ring) stores the sliding (D256) layers'
K and V ring as FP8 E4M3 codes with one FP16 scale per 256-value row, the codec of the global FP8 path (amax / 448, FP16
rounded and bounded scale, round-to-nearest-even saturating codes; a row with any nonfinite value stores NaN codes and a
unit scale). It requires `--kv-dtype fp8` and combines with `--gemma4-key-only-kv` and `--gemma4-qk fp8`.

Current default: BF16 for the general profile, `q5-quality` and native commands; FP8 for `memory-16gb` and `q5-long-context` (see [Adoption decision](#adoption-decision-and-coverage-limits)). The measurements below were made while the FP8 ring was opt-in only.

**What runs.** `gemma4_qkv_prep` quantizes each sliding K and V row once as it writes its ring slot. The decode SIMT
kernel (T < 64: decode, MTP verify and the drafter) expands codes on load; the D256 tensor-core prefill kernel stages
the codes by `cp.async` into the compact half of each BF16 tile and expands them in place, half a tile at a time.
Every reader consumes BF16(FP32 code x FP32 scale), so sliding QK and PV keep their BF16
math; FP8 QK stays a global-layer option. `KvCacheLayout` carries the sliding format as its own axis; the prefix-cache
identity carries it (`/sliding-fp8`, layout packing version 2), so blocks and snapshots never restore across ring
formats. Host snapshot images hold the FP8 codes and row scales. The memory summary, the `Gemma KV` startup line
(`sliding fp8`), the request log's `memory.gemma4_kv.sliding`, the perplexity report's `execution.gemma4_sliding_kv`
and the prefill signature (`/s-fp8`) report the layout; `--plan`, the capacity curve and `auto` sizing count its bytes.

**Bytes (layout, confirmed by the real test's accounting).** A ring row is 258 of BF16's 512 bytes (50.39 %). 31B ring
(50 layers x 16 KV heads): 900 -> 454 MiB at chunk 128 (18 pages), 2,400 -> 1,209 MiB at chunk 2048 (48 pages). One
Host snapshot image 838,041,856 -> 422,297,856 B (Host floor 320 -> 162 slabs of 2,641,920 B).

**Scale granularity.** One scale per row was kept: on every sliding layer's real K/V rows (diode IQ3_XXS, positions
0-128 of a 129-token prompt) the FP8 representation error is 0.0215 relative RMS with 256 columns per scale, 0.0210
with 128 and 0.0203 with 64 (worst layer 0.0255 / 0.0251 / 0.0244; largest row peak/RMS 15.9): smaller groups would buy
about 2-6 % of the error for 1-6 more scale bytes per row.

**Tests.** `infernix_ops_gemma4_attention_test gemma4_fp8_ring` (validation; the writer's FP8 rows equal the row codec
of the BF16 ring's rows, nonfinite rows included; decode and tensor-core attention over a wrapped ring with unmapped and
out-of-range pages, T = 1-2048, equal bit for bit to BF16 attention over the expanded ring and within the BF16 attention
criterion of an FP64 oracle; never-written rows not read; graph replay). Its writer-input comparison against the BF16
ring reports 0.057-0.068 relative RMS attention error with moderate scores and 0.10-0.14 with key-norm weights
0.4-1.7 (a near-argmax softmax). The original experimental 0.10 criterion **failed**, as detailed below.
`infernix_gemma4_fp8_ring_real_test`
(`INFERNIX_GEMMA4_ARTIFACT`): refusal with BF16 globals; the first sliding layer's FP8 rows equal the row codec of the
BF16 ring's; sliding payload and plan equal to the FP8 layout; deterministic generation, root replay and Host snapshot
restore equal to the cold FP8-ring run, chunk 128 and 2048, CUDA graphs off and on, stored globals, MTP
off and on; key-only globals are additionally checked with graphs on (2026-11-18, diode IQ3_XXS: pass).

**Measurement setup (2026-11-18, RTX 5090, preserved development binary SHA-256 prefix `96a403667812`, implementation committed as `e03a5e26`).** One binary serves every arm; its embedded build label predates that commit (`gcbc85732-dirty`). Arms differ
only by `--gemma4-sliding-kv`. Camp evidence: `iq3xs-20261115/fp8ring-20261118/` (drivers, receipts, reports, raw
requests/responses, server logs, `results.json`).

- **Default unchanged.** With the BF16 ring the new binary reproduces the pre-change binary's native KL runs bit for bit
  (per-position sidecars and report statistics equal): diode IQ3_XXS short (a baseline run before the change), plain
  GPTQ5 A8 short stored and key-only, and the GPTQ5 A8 bounded long protocol stored and key-only (the Q5 study's runs).
- **Native full-vocabulary KL** (pinned protocol: context 512, stride 256, chunk 128, chat prefix, FP8 KV, BF16 logits,
  47 windows, 13,728 positions, each artifact against its own source's Q8 reference; paired window bootstrap, 4000
  resamples, FP8 ring minus BF16 ring):

| Artifact, arm | BF16 ring KL | FP8 ring KL | Paired delta KL [95% CI] | Relative | Delta dNLL [95% CI] |
|---|---:|---:|---|---:|---|
| diode IQ3_XXS, BF16 QK | 0.31540 | 0.32219 | +0.0068 [+0.0001, +0.0130] | +2.2 % | -0.0046 [-0.0127, +0.0036] |
| diode IQ3_XXS, FP8 QK | 0.31222 | 0.32092 | +0.0087 [+0.0020, +0.0154] | +2.8 % | +0.0005 [-0.0085, +0.0092] |
| diode IQ3_XXS, key-only, BF16 QK | 0.31683 | 0.32163 | +0.0048 [-0.0033, +0.0110] | +1.5 % | +0.0039 [-0.0056, +0.0120] |
| plain GPTQ5 A8, BF16 QK | 0.02965 | 0.03242 | +0.0028 [+0.0004, +0.0055] | +9.3 % | +0.0053 [-0.0005, +0.0111] |
| plain GPTQ5 A8, key-only, BF16 QK | 0.02985 | 0.03319 | +0.0034 [+0.0009, +0.0059] | +11.2 % | +0.0046 [-0.0011, +0.0104] |

- **Bounded long context** (the Q5 study's chat-prefixed protocol: context 32,768, stride 16,384, chunk 2048, last 64
  targets per window, 3 streams, 704 positions, plain Q8 reference; plain GPTQ5 A8): stored 0.03463 -> 0.03305, paired
  -0.0016 [-0.0138, +0.0076]; key-only 0.03838 -> 0.03877, +0.0004 [-0.0039, +0.0041]. No change is resolvable at this
  sample size (CIs about +/-0.01).
- **Retrieval** (diode IQ3_XXS, 131,072 context, auto chunk, MTP off, greedy; four 7-digit codes per prompt at 10/50/90 %
  depth and one inside the last ~2,600 characters before the question; four prompts each at ~16k/65k/120k tokens): 48/48
  for both rings, including all 12 tail codes.
- **Plans** (`--plan`, MTP d1, headroom 1024 MiB, KV includes rings and tables; BF16 -> FP8 ring): diode IQ3_XXS stored at
  131,072 tokens 7,560 -> 6,369 MiB (`auto` = chunk 2048) and 6,060 -> 5,614 MiB (chunk 128); key-only 4,980 -> 3,789 and
  3,480 -> 3,034 MiB. Plain GPTQ5 A8 stored at 131,072 tokens and chunk 2048: BF16 ring refused (101,888 tokens fit),
  FP8 ring's **estimated plan** fits with 43 MiB left; `auto` keeps chunk 2048 with the FP8 ring. This is not a
  successful 131k resident start: the actual MTP start below resolves only 114,368. Planning before weights load
  does not include every first-use allocation or the live auto-context safety margin; available desktop memory
  also changes between starts. Use the startup capacity, not this tight static plan, for a serving claim.
- **Speed** (diode IQ3_XXS, the matched instrument of the key-only section: 131,072 context, FP8 KV/QK, auto chunk 2048,
  256 greedy tokens, free-cache-resume off, rep 0 excluded, medians of 3, AB and BA orders, one fresh server per cell,
  no build or quantizer running; BF16 -> FP8 ring):

| Prompt | Plain prefill tok/s | Plain decode tok/s | MTP d1 decode tok/s (accepted / drafted) | Whole-device peak MiB, plain |
|---:|---|---|---|---|
| ~1k | 6,393-6,583 -> 5,746-5,761 (-11 %) | 83.1-84.7 -> 83.6-84.1 | 122.8-124.9 -> 121.5-122.1 (314/448 -> 325/439) | 22,971-23,067 -> 21,908-21,972 |
| ~65k | 3,337-3,399 -> 3,102-3,113 (-8 %) | 70.5-71.2 -> 71.0-71.1 | 95.5-95.7 -> 91.5-92.0 (-4 %; 320/444 -> 312/452) | 22,971-23,074 -> 21,996-22,001 |
| ~122k | 2,310-2,332 -> 2,212-2,213 (-4 %) | 61.3-62.8 -> 62.3-63.1 | 79.6-80.1 -> 77.3 (-3 %; 323/441 -> 317/447) | 23,040-23,078 -> 22,014-22,022 |

The in-place expansion of each staged tile (a register round trip and two barriers per K and V tile) costs prefill more
than the halved ring read saves; plain decode is unchanged within the AB/BA spread; MTP rounds lose 2-4 %, partly through
different acceptance (a different lossy ring). Outputs differ from the BF16 ring but are identical across AB and BA runs
of the same arm. Paging counters (Pages/sec, which include file-backed reads) spiked in single rows of both arms with
no visible rate change in those rows.

MTP prefill medians (BF16 -> FP8 ring, AB/BA ranges) are 6,405-6,521 -> 5,789-5,802 at ~1k,
3,349-3,384 -> 3,107-3,108 at ~65k and 2,328-2,350 -> 2,184-2,228 at ~122k. In matched orders the
plain prefill loss is 10.1-12.5 % / 7.1-8.4 % / 4.2-5.1 %; MTP prefill loss is 9.6-11.0 % /
7.2-8.2 % / 5.2-6.2 %. Thus the planned -5 % prefill gate is **not met**. MTP decode loses 3.0-3.5 %
at ~122k and 3.6-4.4 % at ~65k, also failing the planned -3 % gate. Neither is dismissed as noise.

- **32 GB frontier, plain GPTQ5 A8** (`--max-context auto` up to 131,072, chunk 2048, headroom 1024 MiB, the same
  instrument plus one near-capacity request; capacity is the startup's resolved context and moves with the desktop's
  free VRAM by several thousand tokens between starts):

| Globals, ring | Mode | Capacity tokens | ~1k prefill / decode | ~65k | ~122k |
|---|---|---:|---|---|---|
| stored, BF16 (this campaign) | MTP | 86,976 | 6,278 / 84.6 (322/442) | 3,365 / 67.8 | not run (outside capacity) |
| stored, BF16 (Q5 study, pre-change binary) | plain / MTP | 108,096 / 97,344 | 6,303 / 53.7; 6,286 / 82.4 | 3,332 / 48.0; 3,334 / 65.9 | not run |
| stored, FP8 | plain | 125,376 | 5,754 / 54.8 | 3,130 / 49.0 | 2,217 / 44.6 (frontier 124,608: 2,190 / 44.5) |
| stored, FP8 | MTP | 114,368 | 5,750 / 84.9 (329/434) | 3,131 / 66.9 | not run (frontier 113,600: 2,318 / 62.6) |
| key-only, FP8 | plain | 131,072 | 5,751 / 54.7 | 2,783 / 48.3 | 1,902 / 43.6 (130,304: 1,821 / 43.1) |
| key-only, FP8 | MTP | 131,072 | 5,732 / 84.2 (329/434) | 2,789 / 66.2 | 1,903 / 55.7 |

  With normal FP8 global KV the FP8 ring served a 122k prompt without MTP on this 32 GB 5090,
  not with MTP; key-only plus the FP8 ring reaches the 131,072 ceiling in both modes. The +17,280
  tokens against the earlier BF16-ring plain run is a cross-start observation, not an isolated ring-only
  capacity gain: free VRAM and startup margin differ. The new binary's BF16-ring MTP cell reproduced the
  Q5 study's responses, usage and draft counts for all six shared requests.

### FP8-ring desktop ladders (2026-11-18)

The inherited batch completed without interruption (receipt `fp8ring-ladder2`, exit 0, 915 s, minimum
free host RAM 30.03 GiB). Six ladders, **idle** torch ballast on the RTX 5090, not active ballast:
the driver's original comment was incorrect (`fit16g.start_ballast` runs `vram_ballast.py`, not
`vram_ballast_active.py`). Same simulated 16 GB usable envelopes and 31,419 MiB budget line as above;
key-only global FP8 KV, FP8 QK, chunk 128, explicit context and KV capacity, headroom **0**,
8,192 MiB Host tier, one near-capacity 48-token thinking-off greedy request per successful start.
The BF16-ring comparison is the retained `preset16-20261117/ladder2-*` campaign, not rerun.

| Usable MiB | Mode | Largest demonstrated FP8-ring context (peak spare MiB) | Next attempted refusals | Retained BF16-ring largest demonstrated |
|---:|---|---:|---|---:|
| 14,767 nominal | plain | 65,536 (182) | not tested above | 40,960 |
| 14,767 nominal | MTP d1 | 32,768 (240) | 40,960; 49,152 | 20,480 (not tested above) |
| 14,357 (~1.9 GiB reserve) | plain | 40,960 (153) | 49,152; 57,344 | 20,480 |
| 14,357 | MTP d1 | 20,480 (198) | 24,576 (spill guard); 32,768 | none (4,096 refused) |
| 13,743 conservative | plain | 12,288 (169); 9,216 also fits (232) | 16,384; 20,480 | none (2,048 refused) |
| 13,743 conservative | MTP d1 | none | 2,048; 4,096 | none |

All 35 admitted cells completed their one request within the sampled budget; the other 10 starts were
refused (nine reservation shortfalls, one spill guard). The 14,357 MiB/MTP/24,576 start reports shared-system
memory rising from 106 to 8,299 MiB while allocating runtime; it is **not a fit** despite a sampled peak
112 MiB under the budget. Conservative/plain/16,384 missed its reservation by only 306,176 bytes;
this remains a refusal, not a rounded success. Ladders stop after two consecutive failures.

These are sparse, drifting desktop envelopes, not guaranteed hardware limits, a physical 16 GB-card test,
active-overcommit proof or qualification of the default 64 MiB-headroom automatic profile. Idle ballast can
be evicted; sampling can miss spikes. The conservative plain result establishes the intended memory
benefit at zero headroom; it does not promise that the generated memory-16gb profile will fit every such desktop.

### Generated-launcher sampled thinking smoke (2026-11-18)

The missing end-to-end surface check ran the **actual generated CMD**, eight fresh servers:
BF16/FP8 ring x MTP off/d1 x chunk 128/2048, normal stored global FP8 KV, FP8 QK, BF16 logits,
32,768 context, 4,096 MiB Host tier, loopback, fixed Gemma template. Each receives the same two
natural-text prompts (990 / 15,993 actual tokens), thinking explicitly on, cap **8192 total tokens**,
sampler temperature 1.0 / top_p 0.95 / top_k 64 / min_p 0.03, seeds 4100 / 4200.
`fp8ring-finish-qualification` completed (exit 0, 536 s, minimum free RAM **34.72 GiB**).

All **16 requests** returned `stop`, nonempty reasoning and final answers, with no HTTP/transport
errors (1,224-6,672 completion tokens). That is transport/stability evidence, **not 16 healthy thoughts**:
three ~1k requests visibly degenerated in reasoning before the enforced thinking budget closed them:
FP8/chunk128/plain and BF16/chunk2048/plain/MTP. Each reports **6,168 reasoning tokens**
(6,144 model thinking + 24 control); they then produced 476 / 504 / 444 final-answer tokens and stopped.
The retained thought tails show repeated self-correction, repeated phrases or low-vocabulary garbage.
The existing whole-response `loop` flag is false for all 16 (the final answer masks the bad reasoning);
the softer `repetitive` flag catches only two of these three. No detector criterion was changed.
This is an observed failure on **both** rings, not evidence of sampled-quality parity, a loop-free option
or a statistically resolved difference. No final answer was rated for correctness or roleplay quality.
Other requests used 630-1,195 reasoning tokens. No request exhausted the total 8192 cap.

The new generator selection passed **15 focused CPU fixtures** (135 deselected) in the existing
tooling Python 3.13 environment; Python 3.11 syntax compilation passed. Neither installed Python 3.11
nor infernix's Python 3.13 venv had pytest, so no passing 3.11 pytest execution is claimed.
The test-only removal of the unjustified synthetic cutoff compiled in the approved development tree
(`infernix_ops_gemma4_attention_test`, guarded receipt `fp8ring-finish-test-build`: exit 0, 17 s,
minimum free RAM 43.18 GiB). The already-passing codec/FP64 and real-artifact GPU proofs were reused,
not rerun; the serving/perplexity binaries used for the measurements were not rebuilt.


### Adoption decision and coverage limits

The initial 2026-11-18 recommendation was BF16 for every default. The measured memory/fidelity/speed trade was then
accepted for **`memory-16gb` IQ3_XXS**, which now defaults to the FP8 ring, and for the explicit
`q5-long-context` profile ([Q5 adoption](gemma4-q5-prefill.md#q5-recommended-profile-adoption-2026-11-18)).
The general profile, `q5-quality` and native rings remain BF16. Explicit
`--gemma4-sliding-kv bf16` overrides either FP8-ring profile. Planning and fallback stages
preserve the selection ([usage](../../tools/README.md#gemma-sliding-ring-memory-opt-in)). No quality gate changes.
The short plain-GPTQ5 KL increase of 9.3-11.2 % exceeds the plan's order-of-magnitude warning relative
to key-only's historical +0.45 %, and the speed gates above fail.

**Retained failed synthetic criterion, not a pass.** The precommit `fp8ring-ops-first` experiment required
BF16-versus-FP8 ring relative RMS below **0.10**; five of six high-key-norm cases failed (maximum **0.139054**).
This was a newly invented experiment cutoff, not a preexisting repository attention gate: the entire test file
first appears in `e03a5e26`. That commit raised the cutoff to **0.25**, after observing the failures. Passing
that relaxed gross-error tripwire does **not** pass the original criterion or qualify quality. The completion
change removes the unjustified bound altogether and keeps these comparisons as measurements. The original
0.10 failure remains the result; exact codec checks and the unchanged FP64 represented-value attention
criterion establish kernel correctness only; the independent KL/speed findings did not support making the FP8 ring a general default.
The earlier successful Op receipts ran the 0.25 revision; they are not relabelled as an original-criterion pass.
Retained random-token diagnostics report final-position KL 1.15148 (chunk 128) / 0.489734 (2048) between
the rings, changed argmaxes and causal-score differences; their NLL is ~19.8, so they are adversarial
state/correctness probes, not the natural-text Q8 quality instrument.

The initial `fp8ring-quality` receipt exited 1 when an FP8-QK candidate omitted the explicit BF16-QK
reference expectation; the native protocol guard refused it. Corrected commands declare that reference,
and `fp8ring-quality2` passed without changing corpus, reference, numerical thresholds or scored windows.
An earlier ladder attempt used the wrong client port and timed out; its two cells are retained in
`receipts/failed/ladder-nominal-off-ringfp8-port-mismatch`, not counted as fit or as server failures.
The tooling repo's `verify_run_comparability.py` cannot parse these native `report.json` runs (it
supports Sixcat/gate receipts); paired analysis instead checks exact native protocol, window counts
and reference target log-probabilities before computing deltas.
Full-vocabulary KL at chunk 2048/context 512 is refused by the pinned chunk-128 reference guard; the
32k/chunk-2048 evidence instead uses the existing bounded Q5 reference (11 windows / 704 targets, three
streams), **not** the original plan's full 12-stream study. There is no diode-own long-context KL,
65k/122k KL, natural-text generated-distribution KL, all-factor key-only + FP8-QK short KL,
physical 16 GB qualification or human-rated reasoning/roleplay-quality result. Retrieval (including tail
codes) uses both global and sliding layers; a tail success cannot be attributed exclusively to the ring.

## Five-bit text projections: quality, speed and 32 GB capacity (2026-11-17)

Explicit alternatives to best-medium when measured (2026-11-17), not defaults at that time. Current default: the `q5-g64` preset builds one A16Only artifact and the Q5 serving profiles select `int8-g64` prefill ([Q5 adoption](gemma4-q5-prefill.md#q5-recommended-profile-adoption-2026-11-18)); the rows below were measured with the then-explicit `a16`/A8 routes. Each arm is scored on the pinned full-vocabulary protocol
(context 512, stride 256, chunk 128, chat prefix, BF16 QK, FP8 KV, BF16 logits; 47 windows, 13,728 positions)
against the Q8_0 reference of its **own** model; plain-HF rows and additive-diode rows are never compared with each
other. Artifacts include the FP8 vocabulary and FP8 MTP assistant. Routes: own-imatrix `llama-quantize` Q5_K_S/Q5_K_M
imported verbatim (A8 default), and native GPTQ group-64 INT5 (`--weights q5_g64_a16` stays A16; `--weights q5_g64`
permits the wide-prefill (T >= 64) transient row-E4M3 bake into the FP8 A8 GEMM, decode/MTP stay A16).

| Arm | Artifact | Bytes | Mean KL [95% CI] | dNLL | Top-1 disagr. |
|---|---|---:|---|---:|---:|
| diode | actual best-medium (Q4 g32, A8) | 18,405,559,556 | 0.0941 [0.0803, 0.1099] | +0.0316 | 0.0905 |
| diode | Q5_K_S import | 22,066,330,628 | 0.0403 [0.0348, 0.0468] | +0.0180 | 0.0587 |
| diode | Q5_K_M import | 22,600,064,004 | 0.0370 [0.0323, 0.0424] | +0.0091 | 0.0575 |
| plain | GPTQ Q4 g32, A8 (matched best-medium analogue) | 18,405,489,668 | 0.0687 [0.0587, 0.0803] | +0.0167 | 0.0798 |
| plain | GPTQ Q4 g32, A16 | 18,405,489,668 | 0.0591 [0.0499, 0.0702] | +0.0076 | 0.0718 |
| plain | Q5_K_S import | 22,066,330,628 | 0.0244 [0.0211, 0.0277] | +0.0058 | 0.0458 |
| plain | Q5_K_M import | 22,600,064,004 | 0.0236 [0.0204, 0.0269] | +0.0073 | 0.0474 |
| plain | GPTQ5 g64, A16 | 21,151,120,388 | 0.0199 [0.0173, 0.0225] | -0.0008 | 0.0442 |
| plain | GPTQ5 g64, A8 | 21,151,120,388 | 0.0296 [0.0260, 0.0332] | +0.0027 | 0.0543 |

Same weights, A8 minus A16 (paired): GPTQ5 +0.0097 [+0.0080, +0.0115] KL, GPTQ4 +0.0096 [+0.0072, +0.0121]. Key-only
KV changes short-context KL by no measurable amount on the same artifact (GPTQ4 A8 paired -0.0014 [-0.0035, +0.0007];
GPTQ5 A8 +0.0002 [-0.0014, +0.0019]); that says nothing about long context. The same GGUF blocks scored through
libllama give about half the native-import KL (plain Q5_K_S 0.0117), so roughly 0.012 KL comes from the native
serving path rather than the K-quant weights. The native projection tests are `infernix_linear_gemma4_q5_a16_test`,
`infernix_linear_q5_widen_e4m3_test` and `infernix_gemma4_q5_projection_test`.

**Serving instrument.** RTX 5090, development server (build `windows-2026.11.07-187-g9dc29360-dirty`), FP8 KV, FP8 QK, BF16 logits,
`--max-context auto --kv-capacity auto`, `--vram-headroom-mib 1024`, prefix reuse off, thinking-off cold natural-text prompts,
greedy, 256 output tokens; per cell one excluded warmup and two timed requests per length (Q5_K_S import: one), plus one request of
(capacity - 768) tokens. MTP d1 acceptance is accepted/drafted summed over the timed requests. RAM floor 8 GiB in every cell
(never relaxed); each row stores process-list overlap (no build, quantizer or other heavy process was seen in any row) and available RAM / Pages/sec samples
(available RAM stayed >= 33 GiB; Pages/sec spikes were file reads, not memory pressure). The best-medium artifact is the actual
G: file (diode model); the Q5 artifacts are the plain model, so speed, not KL, is comparable across the rows. Cells are prefill tok/s | decode tok/s
(MTP: accepted/drafted); "stored" is normal FP8 K==V, "key-only" is `--gemma4-key-only-kv`.

| Artifact, chunk, mode | ~1k | ~65k | ~122k | Capacity tokens (headroom 1024 MiB) |
|---|---|---|---|---:|
| best-medium c2048 stored, plain | 6302 / 71.0 | 3341 / 61.8 | 2305 / 55.3 | >= 131,072 |
| best-medium c2048 stored, MTP | 6324 / 114.6 (222/287) | 3342 / 88.9 | 2305 / 75.7 | >= 131,072 |
| best-medium c2048 key-only, plain / MTP | 6273 / 70.8; 6441 / 113.7 | 2948 / 60.2; 2956 / 87.0 | 1963 / 53.7; 1963 / 69.8 | >= 131,072 |
| best-medium c128 stored, plain / MTP | 3710 / 71.0; 3696 / 114.0 | 2360 / 61.9; 2358 / 87.4 | 1750 / 55.2; 1749 / 76.0 | >= 131,072 |
| GPTQ5 W5A8 c2048 stored, plain | 6303 / 53.7 | 3332 / 48.0 | not run (outside capacity) | 108,096 |
| GPTQ5 W5A8 c2048 stored, MTP | 6286 / 82.4 (213/297) | 3334 / 65.9 | not run | 97,344 |
| GPTQ5 W5A8 c2048 key-only, plain / MTP | 6205 / 53.6; 6329 / 83.7 | 2941 / 47.6; 2963 / 67.5 | 1985 / 43.1; 1987 / 58.8 | >= 131,072 |
| GPTQ5 W5A8 c128 stored, plain / MTP | 2059 / 53.6; 2125 / 83.1 | 1616 / 46.7; 1536 / 60.1 | 1311 / 43.7; 1110 / 52.0 | >= 131,072 |
| GPTQ5 W5A8 c128 key-only, plain / MTP | 1959 / 49.9; 2121 / 85.6 | 1545 / 47.2; 1547 / 65.3 | 1179 / 40.8; 1195 / 54.8 | >= 131,072 |
| GPTQ5 A16 c2048 stored, plain / MTP | 2719 / 53.3; 2683 / 83.1 | 1944 / 47.6; 1939 / 65.4 | not run | 95,744 / 84,608 |
| Q5_K_S import c2048 stored, plain / MTP | 1421 / 57.2; 1076 / 65.8 | 1803 / 51.3; 1749 / 73.2 | not run | 83,328 / 72,320 |
| Q5_K_S import c128 stored, plain | 212 / 58.1 | 209 / 50.7 | 195 / 42.4 | >= 131,072 |

">= 131,072" means the automatic context reached its 131,072 ceiling at this headroom and a 130,304-token request completed in every such cell; the true frontier of
those cells is not located. The measured frontiers are those below 131,072: normal FP8 KV at chunk 2048 does **not** hold a 122k prompt with five-bit weights
(GPTQ5 W5A8 97k-108k, GPTQ5 A16 85k-96k, Q5_K_S import 72k-83k tokens), whereas key-only KV reaches the ceiling with every artifact. Key-only costs
prefill (65k: 3332 -> 2941, 122k: ~2300 -> ~1990 tok/s for best-medium/GPTQ5) and 1-8 % decode, as in the key-only section above; it is an arm, not a free offset.
Five-bit weights cost 24 % of plain decode against best-medium (71.0 -> 53.7 tok/s at 1k) and 28 % of MTP decode (114.6 -> 82.4); prefill at chunk 2048 matches best-medium for the W5A8 bake (6303 vs 6302 at 1k, 3332 vs 3341 at 65k) but is 43 % (1k) and 58 % (65k) of it for GPTQ5 A16 and 54 % at 65k for the Q5_K_S import. The Q5_K_S GGUF import has no fast prefill route
at these shapes (about 200 tok/s at chunk 128), which makes it impractical for long prompts independent of its good KL.

**Long-context quality (bounded, chat-prefixed; plain model only).** Reference: the plain Q8_0 GGUF through libllama
(`token_logprobs.py reference`, no mmap, 40 GPU layers; a quality run at a 4 GiB RAM floor, minimum free 7.5 GiB) on the windows
of a native control run with the same 17-token chat prefix: 3 streams (wikitext-00, pg19-00, ninfer-00 of the 12-stream long corpus), context 32,768, stride
16,384, 11 windows, last 64 targets of each window (`--score-window-tail 64`) = 704 positions, chunk 2048, FP8 KV, BF16 QK,
BF16 logits. Reference sha256 `fc66c718...d7a5`, Q8_0 sha256 `02776a5f...c77`. Windows are unchanged independent 32k contexts, so these are late-context
tokens at 32.7k depth; the sample is small (CIs wide, the ninfer-00 domain has 4 windows) and says nothing about 65k/122k or a full corpus.

| Artifact | KV | KL [95% CI] | dNLL [95% CI] | Top-1 disagr. |
|---|---|---|---|---:|
| GPTQ5 g64 A16 | normal FP8 | 0.0317 [0.0145, 0.0579] | -0.0037 [-0.0347, +0.0268] | 0.0526 |
| GPTQ5 g64 A16 | key-only | 0.0319 [0.0148, 0.0582] | -0.0088 [-0.0437, +0.0245] | 0.0540 |
| GPTQ5 g64 A8 | normal FP8 | 0.0346 [0.0162, 0.0652] | +0.0115 [-0.0220, +0.0435] | 0.0526 |
| GPTQ5 g64 A8 | key-only | 0.0384 [0.0183, 0.0717] | +0.0048 [-0.0307, +0.0399] | 0.0526 |

Paired (same windows, 4000 window resamples): key-only minus normal dKL +0.0002 [-0.0010, +0.0015] for A16 and +0.0038 [+0.0004, +0.0077] for A8;
A8 minus A16 (normal KV) +0.0030 [-0.0059, +0.0107]. At this bound key-only is indistinguishable from normal KV with A16 weights and
shows a small but interval-excluding-zero KL increase with A8 weights; weight precision and KV precision are separate factors and neither offsets the other here.


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

`kl_compare` runs a paired bootstrap: 2000 resamples by default, seed 20261105,
overall and per domain. The resampling unit is a whole window by default; `--cluster stream`
resamples whole corpus streams instead (all windows of a stream move together), which is
the honest unit when each stream is one independent document, because the overlapping
windows of one stream are not independent (the roleplay corpus uses one forum thread per
stream). The output records the unit (`cluster`) and the number of resampled units
(`clusters`). `--delta` has no default: only the
user chooses an acceptance factor. When supplied, acceptance still requires the
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
