# Gemma 4 GPTQ5 prefill attribution (2026-11-18)

**Current state (read first).** The sections below are dated measurements; statements about defaults,
opt-in status or what remains to be done describe the state when that measurement was made, and instrument
and receipt identifiers (`item2a-http-v1/clean`, `item2d-*.receipt.json`, ...) keep their original campaign labels.

- Builder: `--preset q5-g64` builds one A16Only artifact (native GPTQ5 group 64, FP8 vocabulary, BF16-source
  MTP assistant). Compute is chosen at launch on that same artifact, never by rewriting weights.
- Builder: `--preset q4q5-mixed-a` (experimental, plain-model measurements only) builds best-medium with Q5 `o` and
  `down` in all layers from two GPTQ passes on the same tree and calibration; it keeps the `default` serve profile and
  takes `--gemma4-q5-prefill` for its Q5 cells only.
- Launcher profiles: omitted `--profile` selects `q5-quality` for a `q5-g64` build or a native-Q5 artifact.
  `q5-quality` and `q5-long-context` both select `--gemma4-q5-prefill int8-g64`, auto context, chunk 2048,
  1024 MiB headroom, top-k 64, MTP d1 and a host-only prefix cache; `q5-quality` keeps the BF16 sliding ring,
  `q5-long-context` uses the FP8 ring. Neither enables key-only KV. Explicit flags win; explicit
  `--profile default` keeps the general profile.
- Engine, CLI and perplexity: `--gemma4-q5-prefill` accepts `a16|fp8|int8-g32|int8-g64`; the default `a16`
  preserves the artifact's stored A16Only/AllowA8 policy. The general profile, Q4 and IQ3 tiers pass no flag
  (`a16`). `memory-16gb` (IQ3_XXS only) defaults to the FP8 ring; other rings default to BF16.
- Chunk 2048 is retained; chunk 1536 saves a measured 884 MiB of reservation but is not faster.
- The plain text60 tree and the Q5 A8-tagged artifact used in some measurements below were deleted later;
  they are reproducible (`tools/gptq/make_text_tree.py`; the A8 behavior as the one A16Only artifact with
  `--gemma4-q5-prefill fp8`, bit-identical to it on the bounded incremental protocol below).
- Open caveats that remain: INT8 long-context quality and the diode artifact's incremental quality are
  unmeasured; g32 shows a resolved incremental NLL increase; no blanket numerical-equivalence claim is made.

## Scope and implementation diagnosis

This attribution is measurement and diagnosis only: the existing plain GPTQ5 group-64 A16 and W5A8
artifacts, identical weights and other precision settings, natural ~1k/~65k prompts, chunk
2048 with chunk 128 as a sensitivity arm. No kernel implementation or default change belongs
to this item. The BF16 sliding ring, normal stored global FP8 K/V, FP8 global QK and BF16 logits
isolate the projection routes; no key-only or FP8-ring arm is mixed into this comparison.

**Native GPTQ5 A16 already fuses dequantization into GEMM.** It has no separate global-memory
BF16 weight materialization pass. Confusing it with the GGUF A16 route gives the wrong diagnosis:

- `src/models/gemma4/execution/projection.cpp`: native Q5 A16 calls `ops::linear`; native
  AllowA8 with T >= 64 first calls `q5_widen_e4m3`, then the row-FP8 A8 linear. The actual
  GGUF A16 branch separately calls `ggml_dequantize_bf16`, cuBLAS BF16-to-FP32 GEMM and an output cast.
- `src/ops/linear/q5/shapes/gemma4.cpp`: all six Gemma trunk geometries use the same native
  selection. T <= 64 selects sliced/direct routes, T = 65..128 the 32-row/128-token MMA,
  and T > 128 the wide staging ring in `q5_a16_mma.cu`.
- The wide ring stages **128 weight rows x 128 tokens x 64 K**, three stages, sixteen MMA
  consumer warps and four producer warps: **640 threads and 98,352 B dynamic shared memory**.
  The producer reads packed Q5 codes/high bits/FP16 scales, scales in FP32, rounds to BF16,
  and writes only the shared tile. Consumers accumulate ascending-K BF16 MMAs in FP32.
  The native Q5 linear workspace query is zero: there is no N x K BF16 global scratch to remove.
- W5A8's bake computes each row's actual maximum, a BF16 upward-rounded multiplier, and
  row-E4M3 codes. It decodes the stored weights twice inside that one kernel (maximum pass,
  then encode pass). That is a real standalone pass, but it belongs to the **faster A8 arm**.
  Every projection/chunk rebakes; it is not a persistent weight cache. Activations are then
  quantized per token, followed by the existing FP8 GEMM and any split-K reduction.
- Wide W5A8 uses the existing TMA bulk schedule: **128 tokens x 256 rows x 128 K**, two stages,
  eight consumer warps plus one producer (**288 threads**, **98,336 B shared memory**).
  It splits only an underfilled final wave and reserves a **22,282,240 B** FP32 partial plane.
  Smaller widths/shapes select the existing narrow/mid schedules. The shared-memory ceiling
  is almost identical to the native wide ring; a blanket claim of higher occupancy is not established.

The arithmetic and work differ beyond a removable pass: native Q5 performs packed decode and
FP32 scale/BF16 rounding repeatedly per token tile; W5A8 pays an additional lossy row bake but
uses 8-bit tensor-core operands, wider output-row reuse, larger K steps, TMA and a different
final-wave schedule. A source-level explanation is not a measurement of stall cycles, achieved
occupancy, DRAM throughput or tensor-core utilization.

## Alternative shared-tile design

A reference generic tiled kernel (a design with a non-Volta path that decodes packed weights differently)
is not evidence that shared-tile fusion is missing here.
Both implementations already decode packed weights into a shared BF16 tile. Its non-Volta
path instead stores exact integer codes in that tile and scales an FP32 partial at each group
boundary, with a conventional CTA-synchronized pipeline and activation-as-MMA-A orientation.
The current native wide ring scales/rounds weights before the MMA, overlaps producer/consumer
warps, and keeps one complete-K FP32 chain. The external grouped accumulation therefore changes
rounding/reduction semantics as well as scheduling; it is not a drop-in removal of a redundant
BF16 global pass. Relevant mechanisms must be compared against these actual routes and Gemma
shapes, not the name “fused dequant.” No such kernel is implemented by this attribution.

## Interpretation rule

A separately measured dequant time is a bound only for deleting that exact pass while holding
the downstream GEMM and all other work fixed. Shared-tile fusion can also alter rereads, operand
reuse, registers, shared memory, occupancy, tile shapes and the GEMM itself; its full performance
is not bounded by the removed pass alone. Conversely, such effects are not free or a speed promise.
For native GPTQ5 A16 the removable global-BF16-pass component is exactly absent. Optimizing its
existing fused GEMM requires evidence about that GEMM, not subtracting GGUF decode timings.

## Matched clean HTTP measurement

Hardware: RTX 5090, driver 621.01, CUDA 13.5 development build, 32,607 MiB reported device
memory, configured power limit 575 W. Source baseline `fca9cea236dec3c32ee9b6a1dc15e7248df25b72`.
Instrument `item2a-http-v1/clean`, server SHA-256
`96a40366781237f948be8109422dcd82c5231f2a101a938e6d6ff11b14b4d544`.
The sole executable/build path is `build-infernix-prefill-earlyexit-cuda13.5`.

Artifacts are `plain-native-gptq5-g64-a16.ninfer` / `-a8.ninfer`, both 21,151,120,388 B;
SHA-256 `757652af20b91cf6b61c6cc61e50c58750f008a8343f9c8a107fba50edb09c72` /
`ba17145cf7e6ee51a02e8bafdb1f9f571d7af2d466d2abf71078e229ed2efd1b`.
The bounded CPU comparison independently hashes **all 706 tensor objects**: every descriptor
and payload matches, as do all resources, model components and bindings. The 410 changed
Uses change A16Only to AllowA8, not weights. Both arms have the same FP8 vocabulary/assistant;
the assistant is not loaded for these MTP-off measurements.

Profile: context/KV 68,096, headroom 1,024 MiB, FP8 normal global K/V and QK, explicit BF16
sliding ring, BF16 logits, one lane, prefix reuse/Host cache off. Fixed Gemma template SHA-256
`dc99c0bae1b6113b7756e74fe119ad5933e8a181f99f152976cc650ff9104232`.
The retained English corpus requests are byte-identical across arms/modes: **986/987** short
tokens and **64,985** long tokens. Greedy, top-k 20, seed 1234, thinking off, **one output token**:
this isolates prefill, not decode or answer quality. The excluded short warmup has 982 tokens.
Default graphs remain enabled. Chunk 2048 has A16/A8/A8/A16 starts, four measured requests per
length/arm; chunk 128 has one A16/A8 pair, two per length/arm. Ranges are repetitions, not CIs.

| Chunk | Prompt | A16 tok/s median [range] | W5A8 tok/s median [range] | A16 / A8 prompt ms | A16 / A8 wire TTFT ms |
|---:|---|---:|---:|---:|---:|
| 2048 | ~1k | 2800 [2753, 2816] | 6431 [6327, 6443] | 352 / 153 | 382 / 183 |
| 2048 | ~65k | 2014 [1993, 2040] | 3425 [3417, 3435] | 32273 / 18975 | 32403 / 19104 |
| 128 | ~1k | 1734 [1733, 1735] | 2286 [2275, 2297] | 569 / 432 | 596 / 460 |
| 128 | ~65k | 1343 [1343, 1343] | 1650 [1637, 1663] | 48398 / 39392 | 48525 / 39520 |

The primary A8/A16 throughput ratios are **2.30x short, 1.70x long**; at chunk 128 only
**1.32x / 1.23x**. The larger chunk improves both arms; chunk 128 is not a faster alternative.
Historical 256-token serving measurements and these one-token measurements are separate
instruments, not pooled repetitions.

### Clean Host/GPU boundary accounting

Existing first-output request telemetry, medians in milliseconds (Host submit overlaps GPU
execution; **never add submit to GPU interval**):

| Chunk/prompt | A16 submit / wait / GPU interval | A8 submit / wait / GPU interval |
|---|---:|---:|
| 2048 / ~1k | 3.87 / 348.17 / 350.96 | 12.63 / 140.28 / 152.09 |
| 2048 / ~65k | 89.44 / 32182.46 / 32269.50 | 675.14 / 18288.36 / 18971.64 |
| 128 / ~1k | 4.82 / 563.40 / 566.81 | 16.28 / 414.82 / 429.87 |
| 128 / ~65k | 50.62 / 48336.84 / 48363.92 | 65.15 / 39316.27 / 39356.03 |

Post-processing is at most 3.53 ms in these medians. GPU events are whole-stream intervals,
including possible submission gaps, not active-kernel sums. Stream synchronization dominates
the CPU wall phases because it waits for the GPU; it is **not an independently removable
32-second Host/sync bottleneck**. The primary ~65k GPU-interval difference is about 13.30 s,
while A8 actually has more Host submission time. Wire overhead is about 30 ms short / 130 ms long.

`item2a-clean-prefill` receipt: exit 0, 476 s, minimum free physical RAM **38.47 GiB**;
the 8 GiB floor was not lowered. No build/quantizer/other named heavy-process overlap was
observed by the inherited polling sampler. Memory `Pages/sec` is not uniformly zero:
the maximum sampled rate is **6839.28**, on A16/2048's second long request, with >38.5 GiB
available. Short requests can finish between samples (`min_available_gib=null`); the lease
still samples its RAM floor. These counters do not establish zero paging or identify which
process caused it, and no paging-free claim is made.

Whole-program workspace reservations/high-water at startup: both arms at chunk 2048
**1,604,329,984 / 1,331,691,520 B**; at chunk 128 A16 **75,014,656 / 57,966,592 B**,
A8 **287,176,192 / 281,137,664 B**. These are shared whole-program arena figures, not
an N x K BF16 native-Q5 dequant matrix. Device weight allocations match at **20,631,888,384 B**.

Evidence lives in the model camp `convert/iq3xs-20261115/q5-20261117/prefill-attribution-20261118/`:
`measure.py`, `identity.py`, `artifact-comparison.json`, `summarize.py`, per-cell results,
request/SSE/metrics/logs, and the lease receipts `infernix-port-p3-20261112/item2a-*.receipt.json`.
The generic tooling comparability reader refuses these native-performance directories (exit 2:
it accepts Sixcat/gate artifacts only); the campaign analyzer instead asserts exact request
hash/count identity, no prefix reuse, completed rows and the named instrument before aggregation.
Existing [five-bit quality evidence](gemma4-benchmarks.md#five-bit-text-projections-quality-speed-and-32-gb-capacity-2026-11-17)
remains unchanged; no quality gate is rerun or relaxed by this diagnosis.

## Per-operation CUDA-event attribution

Instruments `item2a-http-v1/eager` (graphs off, event collection off) and
`item2a-http-v1/events` (graphs off, collection on) share executable SHA-256
`e3e0fd903cda1ab80f9a04847677b4cfba88172b03d07374cb87153aed4cfbb3`.
The temporary patch is retained as `event-instrument.patch`, SHA-256
`d1eeda82d188f922d8a291dfef9faddac73f1428239996464b3b36c309547988`.
It reuses the earlier IQ3 projection benchmark's CUDA-event method, **not its random-IQ3
GGUF decode/cuBLAS numbers**, and times actual model projection/attention calls over the
same HTTP prompts. Same-stream start/stop events surround each operation; collection
synchronizes once per chunk and prints shape/count/time sums. These intervals can include
launch gaps and are not hardware active-cycle counters.

Exact event categories map to: `q5_fused_bf16_gemm` = native Q5 `ops::linear`;
`q5_fp8_bake` = `ops::q5_widen_e4m3`; `activation_quant` =
`launch_fp8_a8_quantize`; `fp8_gemm` = selected `launch_a8_impl(..., false)`, including
split-K reduction; `global_attention` / `sliding_attention` = the corresponding
`ops::gemma4_attention` call. There is deliberately no native-Q5 global-BF16-dequant category.
The ~65k primary request has 32 chunks; chunk 128 has 508. Existing last-layer tail skipping
is retained, so a nonfinal chunk has 59 output/gate/down calls and 60 QKV calls.

The first analyzer attempt included the server's two 20-token startup forwards and failed
its token-count assertion. Parsing was corrected to the post-listener HTTP region; every
request's chunk sum now equals its actual prompt count, with no unused event rows.
No GPU cell was rerun for that parser correction. All **70 HTTP requests** completed
(56 timed, 14 excluded short warmups); generated first-token IDs match exactly across
clean/eager/events for each artifact, chunk and prompt. This is a bounded instrumentation
sanity check, not full-logit, KL or generated-answer qualification.

Means of the two timed request interval sums, **milliseconds**:

| Chunk/prompt | Arm | Fused Q5 BF16 GEMM | Q5→FP8 bake | Activation quant | FP8 GEMM + reduction | Global attention | Sliding attention | Other/launch gaps | Total layer interval |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048/~1k | A16 | 325.62 | absent | absent | — | 9.11 | 11.50 | 8.65 | 354.89 |
| 2048/~1k | A8 | — | 31.21 | 4.99 | 88.91 | 8.73 | 11.00 | 9.37 | 154.20 |
| 2048/~65k | A16 | 20622.18 | absent | absent | — | 10116.66 | 1145.46 | 606.05 | 32490.35 |
| 2048/~65k | A8 | — | 1022.41 | 304.71 | 5848.01 | 10089.88 | 1140.18 | 656.22 | 19061.41 |
| 128/~1k | A16 | 527.78 | absent | absent | — | 5.38 | 22.73 | 18.58 | 574.46 |
| 128/~1k | A8 | — | 235.95 | 11.66 | 153.32 | 5.93 | 23.64 | 24.54 | 455.04 |
| 128/~65k | A16 | 33867.13 | absent | absent | — | 10952.83 | 2602.98 | 1199.59 | 48622.52 |
| 128/~65k | A8 | — | 14968.22 | 743.37 | 9637.14 | 10951.47 | 2650.19 | 1558.24 | 40508.63 |

**Primary diagnosis:** projection time accounts for **200.52 ms of a 200.69 ms short
layer-interval gap**, and **13.447 s of a 13.429 s long gap**. Attention/other differences
slightly offset the long projection gap. The native fused GEMM, not a nonexistent external
BF16-dequant pass, is the measured source of the A16 slowdown. It is 91.8% of short / 63.5%
of long native layer time. In A8 at ~65k, attention is already 58.9% of layer time.

The actual standalone A8 bake costs **20.2% short / 5.36% long** at chunk 2048 and
**51.9% / 37.0%** at chunk 128. Removing only that exact bake with every other operation
fixed could remove those intervals, not the whole native-A16 gap. Fusing Q5 into the current
FP8 GEMM also changes representation/scale handling and may change GEMM throughput; the
subtraction is neither an implementation nor a promised speedup.

### Shape efficiency and compute limitations

~65k, chunk 2048, mean cumulative GEMM milliseconds and effective `2*N*K*T / interval`
TFLOP/s. These rates include native decode/staging or FP8 reduction respectively; they
are **not measured tensor-core utilization**:

| Projection N x K | Native A16 ms / TFLOP/s | FP8 GEMM ms / TFLOP/s |
|---|---:|---:|
| sliding QKV 16384 x 5376 | 3167 / 181 | 915 / 625 |
| global QK 18432 x 5376 | 689 / 187 | 202 / 639 |
| sliding output 5376 x 8192 | 1510 / 189 | 439 / 652 |
| global output 5376 x 16384 | 536 / 193 | 158 / 655 |
| gate/up 43008 x 5376 | 10011 / 177 | 2761 / 642 |
| down 5376 x 21504 | 4709 / 188 | 1373 / 646 |

Gate/up plus down consume **71.4%** of native projection time. All shapes show a GEMM
efficiency gap, not one missed shape dispatch. Chunk-128 native effective rates fall to
88-134 TFLOP/s; FP8 to 316-445 TFLOP/s. That is a different native 32-row MMA schedule and
much more frequent A8 rebaking, not just an attention-context effect.

[INFERENCE] BF16's smaller MMA K contraction (16 versus FP8's 32), repeated packed decode,
FP32 scale/BF16 conversion, native ring's four producer warps, smaller row reuse and
different raster/final-wave scheduling are concrete candidate costs. Source establishes
that the native wide ring is limited to one ~96 KiB shared-memory CTA per SM, but so is
the wide FP8 bulk shared-memory footprint: **higher achieved occupancy is not measured**.
The current evidence does not separate arithmetic throughput, producer starvation, register
pressure, shared-memory stalls or DRAM/L2 misses inside the fused kernel. Throughput logs
also record `limited sw-power-cap`; no uncapped or clock-locked comparison is claimed.

### Instrumentation effects, resources and limits

HTTP prompt-time medians, clean / eager / events, milliseconds:

| Chunk/prompt | A16 | A8 |
|---|---:|---:|
| 2048/~1k | 352.38 / 361.04 / 356.39 | 153.39 / 154.99 / 156.07 |
| 2048/~65k | 32272.93 / 32625.37 / 32502.07 | 18974.82 / 19090.79 / 19080.34 |
| 128/~1k | 568.84 / 590.47 / 578.41 | 431.56 / 452.53 / 460.78 |
| 128/~65k | 48398.02 / 48649.49 / 48785.42 | 39391.94 / 40034.34 / 40785.05 |

Events versus eager ranges from -2.04% to +1.88% in these small, separately started cells;
negative differences do not mean events accelerate kernels. Graph effects, clock/thermal
variation and event overhead are not perfectly separated by this limited repeat count.
At chunk 128, event attribution's 8.11 s long gap is smaller than clean HTTP's 9.01 s;
use clean HTTP for end-to-end performance, not the instrumented aggregate. The event collector's
internal synchronization is charged to Host submission by the existing coarse recorder;
its Host/wait percentages are therefore **not production Host attribution**.

Event high-water workspace matches the clean startup arena peaks: 1,331,691,520 B for
either arm at chunk 2048; 57,966,592 B A16 / 281,137,664 B A8 at 128. These are lifetime
arena high-water marks, not individual allocation peaks. Sampled whole-device peaks were
28,424 MiB primary / 25,525 MiB A16-128 / 25,721 MiB A8-128, including desktop usage.
No hardware read-byte counters were available in this instrument. Source arithmetic only:
native Q5 uses 42 encoded bytes/64 weights and writes **zero global BF16 weight-matrix
bytes**; W5A8 reads Q5 in two bake passes and writes N*K FP8 codes plus 2*N scale bytes
per projection/chunk. Cache/transaction traffic is not measured by those logical byte counts.

Eager/event leases exit 0, 346 / 344 s, minimum free RAM 38.73 / 38.75 GiB; no named
heavy overlap, unchanged 8 GiB floor. Nonzero paging samples remain in the receipts/results.
No new nsys attempt was made. Temporary code is removed after measurement; only the patch
and camp analysis drivers remain as reproducibility evidence. No profiler API, new kernel,
precision/default change, source-wiring test or relaxed numerical threshold is retained.

## Ranked next experiments (proposals, not implementations)

1. **Optimize the existing native A16 GEMM, gate/up then down.** First microbenchmark the
   six existing launch shapes at widths 128/~1k/2048 with cold-L2 and matched graph controls;
   compare the available native schedules and measure registers/occupancy/producer stalls
   if a supported instrument is available. Preserve the current represented-input FP64
   criteria and actual model shapes, then require matched clean HTTP and the pinned native
   KL gates for any changed arithmetic. This targets the measured 20.62 s long projection
   component; it does not predict how much is recoverable.
2. **Separate scale/decode work from tile scheduling before considering the external tile
   concept.** Compare the existing shared-scaled-BF16 ring with exact-integer shared codes
   plus per-group FP32 accumulation only as an independently qualified numerical experiment.
   The source already records an earlier 27B group-partial alternative being slower
   (0.985x pp4096); that history is not a Gemma result, but it rejects “fusion must win.”
3. **For small chunks, target repeated W5A8 bake rather than activation quant first.**
   Its 14.97 s long cost at 128 is large; at 2048 it is only 1.02 s, versus 0.30 s activation
   quant. Evaluate immutable per-row metadata/preparation versus fused/groupwise alternatives
   under explicit memory and represented-value contracts; a full persistent FP8 trunk would
   cost roughly a whole additional weight representation and is not assumed to fit. The
   separate groupwise-A8 question is answered for quality by the next section (INT8 per-32 recovers A16 KL); its kernel speed is unmeasured.
4. **Keep long-context attention separate.** Its ~11.23 s A8 primary cost limits the
   end-to-end return from projection work. Do not mix FP8 rings/key-only KV into a kernel
   attribution baseline or silently trade quality to report a projection improvement.

Unresolved: achieved hardware occupancy/read bandwidth/stall attribution, all-MTP behavior,
other GPUs/power limits, longer prompts and quality of any proposed alternative. Existing
quality gates and artifact/reference identities remain binding. Proposals 2 and 3 were pursued in the
compute-precision and INT8 kernel sections below.

## Compute-precision emulation (2026-11-18)

Quality-only, offline study of which compute precision explains the plain GPTQ5 W5A8 gap to A16.
No optimized kernel, default or artifact changed. A temporary emulation route (an explicit
`INFERNIX_TMP_PRECISION` switch in `project()` that dequantized codes and ran grouped BF16 tensor-core
GEMMs with an FP32 groupwise scale chain) was built into the approved development build for the
runs and then removed; it survives only as the camp patch `emulator.patch`
(SHA-256 `aafbc94c5f625c40f174c250443d7e9399f21059ac7ba10779ceabdcadac95d9`, on `8b9f67ac`).
Executable SHA-256 `17e6fa923d69c0832394749c9669cff3b8e49292d313ac05fcf216fbbb30f8fb`; clean rebuild
`d7a3e43fc66a48af4a8175776f6fd1eb3df33f5c0292972885d19aecc62693d2` reproduces the stored A16 positions bit for bit.

Protocol: plain `plain-native-gptq5-g64-a16.ninfer` against `ref512-en-q8-plain.nklref`, context 512,
stride 256, chunk 128, chat prefix, FP8 global KV, BF16 sliding ring, BF16 QK and logits;
47 windows / 13,728 positions; 4000-resample paired window bootstrap. Mean KL [95% CI]:

| Arm | KL | dKL vs exact A16 |
|---|---|---|
| native A16 / exact Q5 emulation | 0.0199 [0.0173, 0.0225] / 0.0199 | +0.00002 / 0 |
| exact Q5 codes, INT8 activations per 32 | 0.0200 [0.0174, 0.0226] | +0.00011 [-0.00044, +0.00057] |
| exact Q5 codes, INT8 activations per 64 | 0.0204 [0.0176, 0.0230] | +0.00050 [-0.00013, +0.00107] |
| row-E4M3 weight bake only | 0.0244 [0.0213, 0.0275] | +0.00452 [+0.00324, +0.00582] |
| row-E4M3 activations only (exact weights) | 0.0246 [0.0214, 0.0277] | +0.00470 [+0.00378, +0.00564] |
| MXFP8 block-32 activations only (exact weights) | 0.0255 [0.0222, 0.0288] | +0.00565 [+0.00430, +0.00705] |
| native W5A8 / emulated W5A8 | 0.0297 [0.0260, 0.0332] / 0.0295 | +0.00976 / +0.00965 |
| MXFP8 activations on row-E4M3 weights | 0.0315 [0.0276, 0.0353] | +0.01159 [+0.00946, +0.01367] |
| MXFP8 weights and activations | 0.0320 [0.0282, 0.0358] | +0.01211 [+0.01002, +0.01421] |

Exact Q5 codes are signed two's-complement five-bit integers, so they are exact INT8 operands; the
FP16 group-64 weight scale is applied after integer accumulation, per K group, never once per full-K dot.
A group dot of |code| <= 16 and |activation| <= 127 over 64 values is below 2^24, so FP32 accumulation of
exactly representable BF16 integers equals INT32 accumulation; the proposed INT8 path would differ only
in INT32-to-FP32 conversion (exact), scale-product association, K-split order and epilogue fusion
(FP32-rounding scale, below the BF16 output). Activation INT8: per-token per-group scale max/127, FP32
reciprocal multiply, round-half-even, clamp to +-127. MXFP8: E4M3 nearest-even saturating with the smallest
power-of-two block scale whose range covers the block maximum (flooring variants untested).

Independent NumPy/FP64 oracle (no CUDA/codec reuse) on the six real shapes (K = 5376, 8192, 16384, 21504):
all ten modes' captured weights, scales and quantized activations match with zero mismatching elements
and the BF16 outputs match its per-group FP32 chain with zero disagreements. Local projection error
ordering (INT8 per-32 about 4x below row FP8; MXFP8 no better than row scaling) predicted the model
ordering but used 8 tokens x 32 rows per shape, so it only filtered. Emulated W5A8 matches production
W5A8 in aggregate (paired -0.00010 [-0.00136, +0.00105]) but not per position (production FP8 GEMM order).

Reading: the W5A8 gap is two additive halves (weight bake +0.0045, activation rounding +0.0047); INT8
per-32 activations on exact codes remove both and reach the A16 ceiling; E4M3 block scaling does not.
Speed was measured separately in the next section (INT8 kernel microbenchmark).
Limits: one plain model, short pinned protocol, intervals about +-0.0025 KL, MTP off, no long-context
re-measurement; KL numbers measured on other models are not evidence here. Plan and full record:
tooling `docs/plans/20261118-gemma4-q5-compute-precision.md`; evidence in the camp's `q5-precision-20261118/`.

## INT8 per-32 kernel microbenchmark (phase 1, 2026-11-18)

Standalone CUDA microbenchmark, **no engine change during phase 1**. The per-32 soft gate was not met;
both group sizes were then implemented in phase 2 below. Sources live in the camp (`q5-int8-20261118/`: `q5_i8.cuh` kernel, `bench.cu`, `probe.cu`,
`oracle_check.py`, `summarize.py`, runner logs). The baseline is the engine's own native A16 code, instantiated from
the same headers: `launch_rowsplit_ring_linear<Q5RingFormat, RowSplitRingTile128, 3, 4>` for T = 2048 and
`q5_a16_mma_kernel<MmaR32T128>` for T = 128. RTX 5090 (170 SMs), CUDA 13.5 `sm_120a`, plain GPTQ5 g64 layout on
the six real shapes with random weights/activations; L2 flushed before each of 15 events-timed repetitions, median of
three serial lease cells (`item2c-final-r1..3`, exit 0, 9.1 s each, minimum free RAM 42.2 GiB, floor 8 GiB, no
build or other heavy job overlapped; Pages/sec samples 15.9 / 16.0 / 1,601.9 with >= 43.2 GiB available, so no
paging-free claim). Native A16 reproduces the in-situ rates of the attribution above (191-208 TFLOP/s versus 177-193).

**Tensor-core peaks** (register-only probe, 170 SMs): INT8 `mma.m16n8k32` s32 **1,002 TOPS**; MX block-scaled E4M3
(`kind::mxf8f6f4`, the production W5A8 GEMM's instruction) **998 TFLOP/s**; plain E4M3 `mma.sync` FP32-accumulate
504; BF16 FP32-accumulate 251. INT8 and block-scaled FP8 peaks are equal, so INT8 has no raw-rate advantage over
the production FP8 GEMM (which reaches 625-655 TFLOP/s in situ); both are 4x the BF16 path.

**Kernel measured.** Exact signed Q5 codes, no bake. Producer warps unpack the stored planes to sign-extended INT8
shared tiles (a 32-value chunk is stored [even | odd] values and the activation quantizer writes the same
permutation, so low/high nibble planes unpack with mask/shift); consumer warps run `mma.m16n8k32` s8 with the
accumulator initialised to `0x4B400000`, so the INT32 result is the bit pattern of `1.5*2^23 + dot` and costs no
conversion; per activation group two FFMA (`fma(F, as, -1.5*2^23*as)` is `dot*as` correctly rounded because the
activation scale keeps <= 22 mantissa bits, then `fma(., ws, run)` with the FP16 group-64 scale). The per-32
scale apply is therefore 8 FFMA per MMA; per-64 chains two MMAs per apply (4 per MMA). Activations: per token per
K group, scale `max|x|/127` (FP32, low two mantissa bits cleared), FP32 reciprocal multiply, round-half-even,
clamp +-127. Pipeline: mbarrier stages of K = 128 (64 also tried), 8 consumer warps (64 x 32 warp tiles of a
128 x 128 CTA tile, one CTA per SM) plus 4 producer warps; optional TMA copy of the activation tile and scales.

**Verification before timing.** Independent NumPy/FP64 oracle (`oracle_check.py`; real captured emulation-study operands tiled to
the full N and T with per-row scale and per-token magnitude variation; 46 configuration/T combinations over both
group sizes, fused / TMA / pre-baked / K64 variants): the activation quantizer matches the independent NumPy rule
with **0** mismatching codes or scales; 979,695 sampled BF16 outputs match the FP32 group-chain emulation
(`t = fl32(dot*as)`, `run = fl32(t*ws + run)`, ascending groups) with **0** disagreements; worst relative L2 to the
FP64 ideal 0.00199 (the BF16 output rounding; native A16 versus exact Q5 x BF16 is 0.0016-0.0021). Against the study's
emulator captures (real rows/tokens, 256 elements per shape) 4-19 elements per shape differ by one BF16 ulp
(relative L2 4e-5 to 4.3e-4): the 22-bit activation-scale mantissa and the association of the FP32 scale products,
as the emulation study predicted; 4-48 of 43k-172k int8 codes per shape differ from its quantizer for the truncated scale.

**Per-shape time at T = 2048** (median of three cells; effective TOP/s = `2*N*K*T / time`; best per shape):

| Shape N x K | native A16 ms (TOP/s) | INT8 per-32 ms (TOP/s) | INT8 per-64 ms (TOP/s) | quant per-32 ms | bake ms |
|---|---:|---:|---:|---:|---:|
| sliding QKV 16384 x 5376 | 1.845 (196) | 1.002 (360) | 0.834 (432) | 0.039 | 0.109 |
| global QK 18432 x 5376 | 2.017 (201) | 1.127 (360) | 0.951 (427) | 0.040 | 0.123 |
| sliding output 5376 x 8192 | 0.879 (205) | 0.496 (364) | 0.414 (435) | 0.058 | 0.054 |
| global output 5376 x 16384 | 1.738 (208) | 0.969 (372) | 0.807 (447) | 0.113 | 0.110 |
| gate/up 43008 x 5376 | 4.957 (191) | 2.600 (364) | 2.193 (432) | 0.040 | 0.286 |
| down 5376 x 21504 | 2.340 (202) | 1.297 (365) | 1.090 (434) | 0.148 | 0.144 |

At T = 128 the native A16 route runs 92-138 TOP/s and the best INT8 per-32 kernel 176-297 TOP/s (0.101 / 0.101 /
0.064 / 0.111 / 0.199 / 0.156 ms in the order above; native 0.205 / 0.209 / 0.123 / 0.234 / 0.429 / 0.312 ms).

**Gate projection at T = 2048.** Each candidate's per-shape time relative to native A16 in the same session
scales the attribution's in-situ cumulative A16 projection time of that shape (20.62 s); quantizer and bake calls are counted
per layer and chunk (50 sliding + 10 global layers, 32 chunks). The standalone quantizer is unfused and about 2x
its memory floor [INFERENCE: roughly 0.3 s fused, not measured]. Reference: W5A8 GEMM 5.85 s + bake 1.02 s + quant
0.31 s = 7.18 s.

| Candidate (exact codes) | GEMM s | quant s | bake s | total s | vs native A16 (20.62 s) | vs W5A8 stack (7.18 s) |
|---|---:|---:|---:|---:|---:|---:|
| INT8 per-32, fused unpack | 11.12 | 0.56 | - | **11.68** | **1.77x faster** | **1.63x** |
| INT8 per-32, pre-baked INT8 scratch | 10.04 | 0.56 | 1.16 | 11.77 | 1.75x | 1.64x |
| INT8 per-64, fused unpack | 9.34 | 0.55 | - | 9.89 | 2.09x | 1.38x |
| INT8 per-64, pre-baked INT8 scratch | 8.87 | 0.55 | 1.16 | 10.57 | 1.95x | 1.47x |

At chunk 128 (A16 33.87 s; W5A8 stack 9.64 + 14.97 + 0.74 = 25.35 s) the fused per-32 kernel projects to 16.43 s
GEMM + 0.95 s quantizer = 17.38 s (1.95x faster than native, **0.69x** of the W5A8 stack because W5A8 rebakes every
chunk), per-64 16.12 s. These chunk-128 projections split the attribution's A16 total by call-weighted measured per-call time
[INFERENCE for the split]; the 65k layer-interval projection (not measured) is 23.55 s for per-32 and 21.76 s for
per-64, against 32.49 s native A16 and 19.06 s W5A8.

**Verdict: the per-32 gate is not met.** It projects 1.77x faster than native A16 at T = 2048 (the 1.3x criterion
holds) but 1.63x the W5A8 stack (the ~1.5x criterion fails by 8 %; a fused quantizer at the memory floor would give
about 1.59x [INFERENCE]). Per-64 (KL 0.0204 in the emulation study against 0.0200, not separable at that sample size) projects 1.38x and passes.
Per-32 narrowly missed the ~1.5x soft criterion; both group sizes were then implemented behind an explicit opt-in load-time compute mode
(A16 was the default at that time; the current Q5 profile default is stated at the top); the engine kernel, the mode and their model-level checks are the next section.

**Where the per-32 rate goes** (gate/up shape, T = 2048, one 128 x 128 tile configuration, effective TOP/s):
consumers alone (staging replaced by barrier pacing) 548; plus real MMA/staging without scale apply 536;
consumers plus activation staging 485; plus weight staging without activations 406; everything 364; producers alone
894. Nsight Compute (profiler SM clock 2.30 GHz; ratios only): tensor pipe 46.5 % active on the full
kernel (56 % consumer-only, where the FMA pipes are about 32 % busy), issue slots 50 % busy, L2 throughput 67 % of
peak, consumers wait on the full barrier for 14 % of their cycles. Neither tensor, FMA, issue nor L2 is saturated
[INFERENCE: the 8 FFMA + 1 IMMA per 4,096 MACs and two consumer warps per scheduler leave the kernel latency-bound,
and the producer and consumer work largely add rather than overlap]. Variants that did not help: K64 stages
(313 versus 353 TOP/s), three-deep pipelines with 12 consumer
warps (334-344), 64 x 256 tiles (334-353 per-32; the best per-64), two or eight producer warps, TMA activation
copies (+2-3 %, kept as the best per-32 for four shapes), mbarrier wait policies (no change), L1-cached side planes
(+1 %), a pre-baked exact INT8 scratch (+11 % GEMM, but 1.16 s of bake and no net gain), two CTAs per SM (180).

Phase-1 limits: synthetic weights and activations ([INFERENCE]: timing does not depend on data values except through power), one GPU and power
limit; engine partial tiles, bounds, graphs, arena fit, quality and HTTP were not yet measured at that stage.
The FP8 reference is the attribution's in-situ number rather than the same harness. Phase 2 supplies the engine checks below.
Tooling plan: `docs/plans/20261118-gemma4-q5-int8-per32-microbench.md`.

## INT8 prefill engine kernel (phase 2, 2026-11-18)

Implemented in `74eff759`: `--gemma4-q5-prefill a16|int8-g32|int8-g64`, default `a16`, across Engine,
CLI, serving, native perplexity and the builder-generated launcher/serve-check profile. At that commit no profile enabled INT8
implicitly (the Q5 profiles now do; see the last section). **The default preserves the artifact's existing A16Only/AllowA8 policy**: an A16 artifact stays A16;
an AllowA8 artifact still uses the existing W5A8 bake. The INT8 choices override the six native Q5 trunk
shapes at **T >= 64** only, using the existing Q5 group-64 artifact without rewriting or duplicating weights.
They are refused with lossless execution, other model families or an artifact without native Q5 trunk weights.
Mode is included in the prefix-cache signature and request/execution records.

`q5_int8_prefill.cuh/.cu` retains the phase-1 exact-code IMMA/groupwise FP32 scale chain, with a bounded
activation quantizer and partial-token handling. The same two configurations serve both group sizes:
128 x 128 wide tiles, 64 x 128 tiles for T = 64..128. The quantizer writes a 128-token-padded INT8
activation buffer and two FP32 metadata values per group in the existing scoped workspace. There is no
persistent weight bake, new device allocation or environment-controlled inference route.

**Actual decode precision:** calls below 64 columns retain native Q5 A16, including single-token decode
and short MTP verification; the assistant/vocabulary keep their existing representations. Wide verification
can select INT8. This is not an “all-token INT8” implementation, and changed prefill KV can change later
argmaxes even when the decode kernel itself is unchanged.

### Numerical and integration evidence

RTX 5090, CUDA 13.5 `sm_120a`, approved `build-infernix-prefill-earlyexit-cuda13.5` only.
The integrated Linear test passes on all six shapes at T = 64/65/128/129/193/256, with T = 2048 on
gate/up and down, guard/poison checks, changed-input graph replay and below-threshold/default
route comparisons. Its independent host decoder uses integer dots and `std::fmaf`; the separately decoded
NumPy/FP64 capture oracle covers **92 cases / 110,600 elements, zero BF16 disagreements** against the
documented FP32 group chain, worst relative L2 **0.001961** against the FP64 represented-operand ideal.
`infernix_gemma4_q5_projection_test` also passes through model projection policy/workspace/graphs.
Receipts: `int8-phase2-optest-2` / `int8-phase2-q5proj-1`, exit 0, minimum free RAM 37.48 / 37.59 GiB.
The saved implementation session additionally ran the native CLI/serve parsers and affected Python builder
tests; those checks were not repeated during recovery.

Pinned native full-vocabulary KL: the same plain A16 artifact and Q8 reference as the emulation study, context 512,
stride 256, chunk 128, chat prefix, FP8 global KV, BF16 ring/QK/logits; **47 windows / 13,728 positions**,
4000-resample paired window bootstrap:

| Route | Mean KL [95% CI] | dKL versus default A16 [95% CI] | dNLL versus A16 | Top-1 disagreement |
|---|---|---|---:|---:|
| default A16 | 0.019914 [0.017267, 0.022466] | 0 | 0 | 0.04422 |
| INT8 g32 | 0.020425 [0.017716, 0.022952] | +0.000511 [-0.000095, +0.001127] | +0.002021 | 0.04538 |
| INT8 g64 | 0.020305 [0.017541, 0.022987] | +0.000391 [-0.000416, +0.001178] | +0.001506 | 0.04356 |
| W5A8 (retained matched emulation-study result) | 0.029649 [0.026021, 0.033243] | +0.009735 | +0.003517 | 0.05427 |

Default A16 reproduces every retained emulation-study position bit for bit. Native g32 versus its emulator is
+0.000422 [-0.000149, +0.000997] KL; g64 -0.000081 [-0.000647, +0.000426]. Neither is bit-identical to
the emulator (truncated scale/FP32 association), but this sample does not resolve an aggregate difference.
Nor does it establish exact equivalence to A16 or a quality ranking between g32/g64 (paired g64 minus g32
-0.000120 [-0.000728, +0.000470]). Both recover most of W5A8's quality gap: dKL -0.009224 / -0.009344.
Worst individual window increases versus A16 are +0.00636 / +0.00746; do not replace that distribution with
the mean. Long-context quality is unmeasured. The KL lease exits 0 in 166 s with >=36.62 GiB free RAM.

### Clean HTTP prefill

All arms use the same executable SHA-256 `ec6905ca4cc799623862814439316d56083d0bf68f1a2967e71b37bdeb83db53`,
with no event-instrumentation switch. Same hardware/power, artifacts and requests as the attribution; context/KV 68,096,
headroom 1024 MiB, normal global FP8 K/V and FP8 QK, **BF16 sliding ring**, BF16 logits, one lane,
graphs on, prefix reuse/Host cache off, `free-cache-resume` off, greedy/seed 1234/thinking off.
Four measured requests per arm/length at chunk 2048 (forward/reverse starts), two at chunk 128; one excluded
982-token warmup per start. Actual timed lengths are 986/987 and 64,985 tokens, one output token.
Request hashes/counts, no prefix reuse, completed requests, binary identity and no named heavy overlap
are checked before aggregation. Ranges below are repetitions, not confidence intervals.

| Chunk / prompt | A16 tok/s [range] | W5A8 tok/s [range] | INT8 g32 tok/s [range] | INT8 g64 tok/s [range] |
|---|---:|---:|---:|---:|
| 2048 / ~1k | 2797 [2748, 2834] | 6329 [6302, 6365] | 4660 [4569, 4709] | 5390 [5286, 5486] |
| 2048 / ~65k | 1968 [1866, 1993] | 3411 [3384, 3434] | 2734 [2723, 2757] | 2982 [2968, 3003] |
| 128 / ~1k | 1736 [1731, 1742] | 2273 [2261, 2286] | 3308 [3302, 3314] | 3716 [3660, 3772] |
| 128 / ~65k | 1326 [1303, 1348] | 1658 [1658, 1659] | 2071 [2070, 2072] | 2148 [2144, 2153] |

At chunk 2048 g32/g64 are **1.67x/1.93x** A16 throughput short and **1.39x/1.51x** long, but
**26.4%/14.8% slower** than W5A8 short and **19.9%/12.6% slower** long. At chunk 128 both beat W5A8,
but both remain slower than their own chunk-2048 route: smaller chunks are not recommended for speed.
Median prompt times in arm order A16/W5A8/g32/g64: 353/156/212/183 ms short and
33.014/19.050/23.769/21.793 s long at 2048; 568/434/298/266 ms and 49.041/39.187/31.379/30.251 s at 128.
Wire TTFT: 378/183/275/211 ms and 33.146/19.192/23.932/21.923 s at 2048;
597/463/315/291 ms and 49.169/39.307/31.505/30.387 s at 128.
All **60** prefill requests return the same first token (818), also matching the retained A16/W5A8 attribution
requests. That is a smoke observation, not proof of full-output/logit equivalence.

The three saved cold-L2 microbench repetitions of the integrated configurations (15 CUDA-event samples each)
give T=2048 GEMM medians in shape order sliding QKV/global QK/sliding out/global out/gate-up/down:
g32 **1.036/1.148/0.506/0.981/2.879/1.301 ms**, g64 **0.873/0.963/0.418/0.809/2.458/1.080 ms**.
G32 quantization costs 0.029/0.029/0.046/0.083/0.030/0.108 ms; these are kernel numbers, not HTTP totals.
The harness's `projection_65k_ms` rows are ratio-based projections using the attribution's chunk-2048 weights, not
measured engine intervals; its T=128 projection uses that same weighting and is not a valid chunk-128
end-to-end estimate. The clean HTTP table, not those projected rows, is the performance result.

### Plain decode, MTP and resources

Short smoke only, chunk 2048, two different ~1k requests per cell, **128 output tokens each**, all complete:

| Route | Plain decode tok/s [range] | MTP d1 decode tok/s [range] | Accepted / drafted, two requests |
|---|---:|---:|---|
| A16 | 50.57 [50.55, 50.58] | 83.49 [81.25, 85.74] | 56/71, 53/74 |
| W5A8 | 50.56 [50.53, 50.59] | 79.10 [75.99, 82.20] | 57/69, 51/76 |
| INT8 g32 | 50.83 [50.82, 50.85] | 81.65 [79.21, 84.09] | 56/70, 52/75 |
| INT8 g64 | 50.94 [49.99, 51.89] | 78.88 [73.93, 83.83] | 60/67, 51/76 |

Plain decode uses the same A16 kernels, with no measured slowdown in these medians; MTP g32/g64 medians are
2.2%/5.5% below A16. Different generated continuations/acceptance and two samples do not isolate a kernel
regression or qualify general MTP throughput. Every route/spec matches the first eight tokens for each
request; whole 128-token sequences can diverge (first observed difference from A16/plain at output 11).
Two different requests are not repeatability trials. No blanket token-identity claim is made.

All arms' weight allocations remain **20,631,888,384 B**. Whole-program workspace capacity/high-water
at chunk 2048 is **1,604,329,984 / 1,331,691,520 B**, unchanged across modes and these MTP starts;
at 128, A16/g32/g64 **75,014,656 / 57,966,592 B**, W5A8 **287,176,192 / 281,137,664 B**.
Transient INT8 codes/metadata fit inside that reused envelope: at K=21504, T=2048, g32/g64 need
52.5/47.25 MiB plus alignment, not another weight matrix. No VRAM-capacity-frontier claim is inferred.
Observed whole-device peaks (including desktop) are 28,421 MiB prefill and 29,256 MiB short decode/MTP.

All GPU leases retain the **8 GiB** physical-RAM floor and are serialized with builds. HTTP prefill/decode
leases exit 0 in 881/163 s, minimum free RAM 36.85/36.67 GiB; no named heavy-process overlap.
Maximum sampled Pages/sec is **3445.76** prefill / **16.85** decode, so no paging-free claim.
The original HTTP attempt failed before server startup because its system Python lacked NumPy;
the existing clean2 retry used the tooling venv and succeeded. Recovery preserved all its rows, reran
neither KL nor completed numerical/timing tests, and added only the missing short decode/MTP cells.
The default-mode startup description was then corrected to say artifact routing, not falsely label W5A8
as BF16; that logging-only change was rebuilt separately and is not the measured executable above.

**Disposition:** the opt-in routes provide near-A16 short-protocol KL at materially faster prefill, not
W5A8 parity at chunk 2048. G64 is faster with no resolved quality loss versus g32 on this sample; neither
was made a default by this work (the later Q5 profile adoption selects g64 for the Q5 profiles only). No tuning campaign, model promotion, artifact rewrite or production build change.
Evidence: `q5-int8-20261118/phase2/{oracle-engine,kl-summary,http-summary}.json`, test dumps, benchmark
sources, request/SSE/metrics/logs, and lease receipts `int8-phase2-*.receipt.json`. Plan:
tooling `docs/plans/20261118-gemma4-q5-int8-per32-microbench.md`.

### Final rebuild and path incident

The logging-only rebuild was first invoked with a basename rather than the required absolute `-BuildDir`.
The wrapper resolved it under the checkout, rebuilding the **pre-existing**
`D:/AI/loaders/infernix/build-infernix-prefill-earlyexit-cuda13.5` tree instead of the approved sibling.
That process tree was stopped. The initial claim that it had created a new directory was corrected after
checking its creation time (09:15, before recovery); the pre-existing tree and outputs were preserved,
not deleted. No production or approved-build output was changed by that mistaken invocation.

The corrected invocation used **`-BuildDir D:/AI/loaders/build-infernix-prefill-earlyexit-cuda13.5`**,
target `infernix-serve`: exit 0, minimum free RAM 39.51 GiB. Only the startup string changed in source
(`5d0c6343`); numerical kernels/settings are unchanged from the measured implementation.
Rebuilt executable SHA-256 **`4732960a69f40a4a32d56281c3a4af3c87045429bf7f384d941b601e4b7c643a`**.
A narrow, serialized startup/log smoke then passed on default W5A8 and opt-in g32: expected mode/scope in
human and JSON startup records, one-token requests both returned 818. Lease exit 0, 33 s, >=35.84 GiB RAM,
unchanged 8 GiB floor, no named heavy overlap. No quality/timing campaign was repeated.
`phase2/startup-scope-summary.json` connects the measured and rebuilt hashes to the implementation/logging
commits; `int8-phase2-log-scope-approved.*` and `int8-phase2-startup-scope.receipt.json` retain build/smoke evidence.

## Single-artifact compute selection and incremental scoring (2026-11-18)

The INT8 phase-2 work already implemented launch-time selection, scoped workspace, graph routing, cache signatures,
CLI/serve/perplexity/builder propagation and the narrow A16 split. This study reuses those mechanisms.
The same `--gemma4-q5-prefill` now accepts `fp8`, enabling the existing W5A8 bake for all four prepared
native Q5 trunk projections (QKV/QK, output, gate/up, down), including trunk verification. No second flag,
artifact rewrite, copied artifact or persistent extra weight representation is needed.

The Engine/CLI `a16` default remains **artifact policy** (Q5 launcher profiles pass `int8-g64` explicitly), not a forced override: an A16Only artifact runs
A16; an AllowA8 artifact keeps W5A8. For an actual launch-time A16/FP8/INT8 choice, use the **one A16Only
Q5 artifact** and select `a16|fp8|int8-g32|int8-g64`. All lossy routes apply at 64+ columns only; single-token
decode and narrow MTP verification remain native A16. Assistant/head representations stay unchanged.
The prepared policy feeds both planning and execution, and existing mode-specific signatures isolate caches.

`ScoreOptions::incremental` and `infernix-perplexity --incremental` add a Gemma4-only teacher-forced
schedule to the existing scoring/full-vocabulary KL API: independently reset each window, prefill exactly
`tokens[0:first_target]` in the configured chunks, then feed each preceding ground-truth suffix token in
one-column forwards with that window's retained KV. **Scored position zero is the final prefill state**;
only positions 1+ measure incremental arithmetic. This is not repeated prefix rescoring, sampled generation,
or a block-prefill quality result. `scoring_schedule=prefill-then-one-token/1` binds reference caches to the
new schedule; old block-prefill references cannot silently qualify it. The companion explicit-token Q8
reference writer inherits the same prefix/chunk/one-token schedule from that protocol.

Compare identical token IDs, window boundaries and target positions against the plain model's own Q8
reference. Cross-mode incremental differences include changed prefill KV; an identical narrow A16 prefix
control separately isolates whether the decode arithmetic itself changed. Short bounded evidence and
verification are reported below; no default, promotion, long-context qualification or chunk sweep follows
from this instrumentation.

### Behavioral verification

Implementation `e80af5ef` + protocol/API validation `132c9dcf`; the test-capacity correction is `3c8f0c23`.
The approved absolute development tree builds all affected executables/tests successfully; final measured
perplexity executable SHA-256 `c10f25f193c8faf6775c65efb6bd29334da9cb7e8fea827519794a7aeec39426`.
No production build or artifact is rewritten.

- The existing Q5 projection test now also exercises launch-time `fp8` on A16Only and AllowA8 policies
  over all six shapes and its existing T=1/2/8/63/64/65/127/128/129/2048 widths, independent represented
  and stored-operand FP64 gates, changed-input graph replay and workspace guards. It passes unchanged gates.
- `infernix_gemma4_incremental_score_real_test` passes all four compute modes on the one A16Only artifact:
  eight teacher-forced top-1 outputs equal real public greedy generation, repeated windows reset bit-exactly,
  and the first selected full-vocabulary row equals the corresponding ordinary final-prefill row.
  With an identical 31-token A16 prefix, selected full-vocabulary rows at scored positions 0/1/7 are
  bit-identical across all modes. This separately establishes the unchanged narrow arithmetic for the
  measured control, not equivalence of KV written by different wide-prefill modes.
- Native CLI/serve parser tests pass; 11 targeted builder launcher tests and 15 reference-window/schedule/
  byte-exact CRLF Python cases pass. The builder emits nondefault precision on every generated serve/planning command.

The first test lease passed projection but the Engine test failed before `main` (`0xC0000135`): targeted
builds had not staged its DLL closure. The retry uses the approved apps runtime directory on PATH (no copied
DLLs). That exposed a test setup error: max context 512 with the generation API's default KV capacity 2048.
The test now explicitly requests capacity 512; no engine gate or numerical threshold changed. Completed
projection work was not rerun. Successful corrected lease: exit 0, 71 s, minimum free RAM **35.05 GiB**;
the first projection lease retained >=34.84 GiB. Every GPU lease retains the 8 GiB floor and is serialized
with builds. Build minima: initial 20.77 GiB, final validation rebuild 37.56 GiB, test-only rebuild 38.81 GiB.
The first native Python test attempt lacked pytest; the tooling environment completed the same targeted cases.

### Bounded incremental protocol

Plain model only, same **A16Only Q5 artifact** for A16/FP8/INT8 g32/INT8 g64, with the legacy AllowA8
artifact used solely as a bit-equivalence control. Corpus: the first two streams each of English reference,
long-form prose and code. Selection used the first 768 HF tokens of each LF-normalized source prefix;
Windows serialization preserved CRLF in the resulting files. The authoritative native input counts are
**785/773/918/847/827/833** (4,983 total), not the original 768-token selection estimate.
Context 512, stride 256, 17 explicit chat-prefix IDs, chunk 128, last 17 targets per independent window;
ordinary stored global FP8 K/V, BF16 sliding ring, BF16 QK/logits, MTP off, default graphs.
Actual plan: **18 windows, 294 targets: 18 final-prefill distributions and 276 true incremental positions**.
Seventeen windows prefill 512 tokens; the short terminal window prefills 524 and scores five targets.
Each subsequent target is scored after one new ground-truth input, not a re-prefilled prefix.

The first Q8 reference attempt correctly refused before model loading: Python's universal-newline read
changed CRLF to LF, giving 768 tokens where native binary input recorded 785. The reference reader now
decodes UTF-8 bytes without newline translation, with explicit CRLF regressions for both window builders.
The completed native protocol is retained, not rerun; no target/window gate is relaxed. The Q8 reference
uses the plain model's own Q8_0 GGUF, exact native input IDs/predictor frontiers and the inherited chunk/
one-token schedule, with its existing default KV representation and FP32 logits. Candidates round logits
to BF16 as in the earlier native quality protocol. Counts and reference identity must match before analysis.

Confidence intervals use 4,000 **paired window** bootstrap resamples, seed 20261118, token weighting
within each slice. The windows overlap and come from only six short source prefixes: these are conditional
sample intervals, not population guarantees or long-context qualification. Final-prefill and incremental
slices are never pooled to make a decode claim; the shared-prefix A16 control above isolates unchanged
decode arithmetic separately from the inherited prefill-state error.

### Incremental quality results

Full-vocabulary `KL(Q8 || candidate)` and NLL differences, natural logs, paired-window 95% intervals.
Reference cache SHA-256 `42c8c8f47aa6893b79efb748183ccff7c41101c325ccb1a1de5f8bb02ff7ce00`,
own plain Q8 GGUF SHA-256 `02776a5fd33cb05a4cfee40d4bb4e88afef5d145f3f70ccede6bbccc4a49fc77`.
The preflight independently proves **every native/Q8 input token ID and first-target frontier identical**.

**True incremental positions only (276):**

| Prefill / decode | Mean KL [95% CI] | ΔNLL vs Q8 [95% CI] | Paired ΔKL vs A16 [95% CI] | Paired ΔNLL vs A16 [95% CI] |
|---|---|---|---|---|
| A16 / A16 | 0.009948 [0.006199, 0.014050] | -0.002820 [-0.025807, +0.019563] | 0 | 0 |
| FP8 / A16 | 0.011924 [0.008183, 0.015671] | +0.014275 [-0.007691, +0.036316] | +0.001976 [+0.000355, +0.004001] | +0.017095 [-0.004153, +0.038221] |
| INT8 g32 / A16 | 0.011127 [0.006619, 0.016588] | +0.010323 [-0.010305, +0.030062] | +0.001179 [-0.000047, +0.003028] | +0.013143 [+0.004969, +0.022636] |
| INT8 g64 / A16 | 0.010398 [0.006561, 0.014765] | +0.000793 [-0.020607, +0.020409] | +0.000449 [-0.000341, +0.001318] | +0.003613 [-0.009612, +0.017883] |

**Final-prefill distributions only (18), not decode:**

| Prefill | Mean KL [95% CI] | Paired ΔKL vs A16 [95% CI] | Paired ΔNLL vs A16 [95% CI] |
|---|---|---|---|
| A16 | 0.006828 [0.002273, 0.013841] | 0 | 0 |
| FP8 | 0.016623 [0.003429, 0.039364] | +0.009795 [+0.000785, +0.025828] | -0.113922 [-0.211589, -0.034989] |
| INT8 g32 | 0.006688 [0.002170, 0.013318] | -0.000141 [-0.000932, +0.000842] | +0.047943 [-0.006762, +0.127210] |
| INT8 g64 | 0.007830 [0.002216, 0.016752] | +0.001002 [-0.000395, +0.003062] | +0.027199 [-0.017296, +0.097572] |

The incremental FP8 KL penalty is resolved on this bounded sample despite unchanged A16 decode arithmetic.
G32/g64 paired KL intervals cross zero; that is **not equivalence**. G32's paired incremental NLL increase
is resolved in this conditional window bootstrap; do not carry the earlier block-prefill “near-A16” result
over as a blanket decode-quality claim. G64's NLL interval crosses zero. No general ranking or default follows.
Incremental top-1 disagreements with Q8: A16/FP8/g32/g64 **2.90%/4.35%/2.17%/2.90%**.
Worst window-mean incremental ΔKL versus A16: FP8 +0.015007, g32 +0.013925, g64 +0.004331.
The negative FP8 NLL change on the tiny final-prefill slice alongside worse KL also illustrates why NLL
alone is not a distribution-quality criterion. These are different target positions and a different
schedule from the INT8 phase-2 13,728-position block-prefill protocol; their absolute KL means are not comparable.

The new single-artifact `fp8` route and legacy A8-tagged artifact produce **all 18 full-logit window hashes
and all 294 per-position records bit-identically**. Default A16's protocol-write and comparison logit
hashes also match exactly. This is a bounded numerical equivalence proof for the new selection path,
not a reason to repeat the completed INT8 phase-2 ~65k timing campaign.

The byte-exact quality lease exits 0 in 224 s, minimum free RAM **32.27 GiB**, unchanged executable hash,
8 GiB floor. The earlier refused reference lease retained >=36.09 GiB and proved both original artifact
hashes unchanged. No model deletion, promotion, production build, long-context campaign or chunk sweep.
Explicit refusal checks pass for a real Q4 artifact, non-Gemma/unknown-artifact configuration, and incremental
top-token references whose format cannot bind the schedule (exit 1 with the intended errors; the lease
exit 0, >=37.67 GiB). The first Q4 refusal harness hit the unrelated FP32-logits guard; its corrected BF16
configuration reaches the intended native-Q5 artifact guard.

Evidence: camp `q5-20261117/item2d-20261118/` (`run.py`, `quality.py`, `summarize.py`, `quality-summary.json`,
`token-window-equivalence.json`, `reference.identity.json`, `refusals.json`, independent protocol/reference
caches, all window/position records and command/log files); lease receipts `item2d-*.receipt.json` and build logs.

### New-path short HTTP/MTP smoke and resource equivalence

Only the genuinely new `A16Only artifact + fp8` path gets a short serving comparison against the legacy
AllowA8 artifact; the prior INT8 phase-2 long and INT8 timings above are retained, not rerun. Same clean development
server SHA-256 `63af2c0d78a6788ee5b97ba16d8206a1d503e4d78c9abe106fa527925c010c8c`, default graphs,
context/KV 68,096, chunk 2048, FP8 global KV/QK, BF16 ring/logits, headroom 1,024 MiB, cache/reuse off,
MTP d1. Each start has one excluded 982-token/one-output warmup and the same 987-token/32-output request.
All tokens match across starts, including all 32 MTP outputs; both accept **14/16** drafts.

Single-artifact / legacy observations: prompt **163.84 / 175.32 ms** (6024 / 5630 tok/s), MTP decode
**85.59 / 83.96 tok/s**, wire TTFT **191.59 / 191.52 ms**. One pair is a smoke, not a performance
confidence interval, speedup claim or replacement for the phase-2 measurements. Startup/request records correctly expose
`fp8` versus historical `a16` artifact routing, with distinct `q5=fp8` execution signatures.

With the assistant loaded, **both** starts allocate weights **21,118,694,400 B**, workspace capacity/peak
**1,604,329,984 / 1,331,691,520 B**, and measured graphs **31,457,280 B** within the 64 MiB allowance.
This is the observed unchanged allocation envelope, not a VRAM-capacity-frontier test. The only new
selection changes the prepared trunk policy; no persistent widened trunk appears.
Lease exit 0, 33 s, minimum free physical RAM **36.12 GiB**, unchanged server hash, 8 GiB floor.
No named heavy-process overlap was sampled; Pages/sec reaches **4930.25**, so no paging-free claim.
Evidence: `http-summary.json`, both starts' request/SSE/metrics/logs and `item2d-fp8-http-smoke.receipt.json`.

**Disposition:** one A16Only Q5 artifact supports the explicit A16/FP8/INT8 launch choices;
default artifact behavior, production artifacts/builds and staging models remain untouched. The new path
is numerically equivalent to the prior A8-tagged artifact on the bounded protocol and MTP smoke.
True incremental quality is now measured separately from final-prefill state; narrow A16 arithmetic is
unchanged on the shared-prefix control, but prefill history still changes decode distributions and g32 shows
a measured NLL increase. No new default, general quality equivalence or long-context claim follows from this study.

## Matched Q5 chunk sweep (2026-11-18)

**Recommendation: retain chunk 2048.** On this RTX 5090 (170 SMs), FP8's highest measured
~65k median is at 2048. INT8 g64's 1536 median is only **0.64%** higher than 2048, with
overlapping repetition ranges; g32's 1536 median is 2.08% higher but also within its much
wider range. Neither is a clear speed winner. Chunk **1536 is an explicit memory tradeoff**,
not a new default or a demonstrated speed improvement. Compute modes remain independent
choices: these timings do not qualify quality or erase the significant g32 incremental
NLL increase. No source, kernel, artifact, preset, automatic chunk or production-build change; the Q5 profiles keep chunk 2048.

### Instrument and fixed serving profile

`item2e-cold-http/v1`, source `f9a882adc021fc9ce2cf3dae144a67154513ceb9`, clean tracked source
at instrument capture; the existing approved absolute executable is
`D:/AI/loaders/build-infernix-prefill-earlyexit-cuda13.5/apps/Release/infernix-serve.exe`,
SHA-256 **`63af2c0d78a6788ee5b97ba16d8206a1d503e4d78c9abe106fa527925c010c8c`**.
No build was needed. Every mode uses the **one** A16Only artifact
`plain-native-gptq5-g64-a16.ninfer`, 21,151,120,388 B, SHA-256
`757652af20b91cf6b61c6cc61e50c58750f008a8343f9c8a107fba50edb09c72`.
The lease verifies both hashes unchanged afterward. Driver 621.01, CUDA 13.5, 575 W limit;
no clock lock or hardware-profiler claim. Campaign driver SHA-256
`a502fb2b508da6f561fd822c055e41e720d6d636ce5b80af6ce3025077879ef0`;
`identity.json` also binds the reused HTTP helper and fixed template
(`dc99c0bae1b6113b7756e74fe119ad5933e8a181f99f152976cc650ff9104232`).

Same profile as the INT8 phase-2 measurements: context/KV **68,096**, 1024 MiB headroom, normal global FP8 K/V,
FP8 QK, **BF16 sliding ring**, BF16 logits, one lane, graphs on, MTP off, prefix reuse
and Host cache off, `free-cache-resume` off. “Cold” means **zero retained-prefix reuse**,
not cold disk or cold L2: startup warmup and one excluded 982-token HTTP warmup remain.
No temporary inference environment switches were set.

The retained attribution/phase-2 English request bodies, greedy/top-k 20/seed 1234/thinking off,
are identical across cells: short **986/987** tokens, long **64,985**, one output token.
Each route/chunk has **three independent starts**, two measured shorts and one identical
long per start: **six short / three long observations**. Chunk orders are forward,
reverse and interleaved; mode orders are rotated/reversed. Two minimal A16/2048 starts
bracket the sweep. All **38 starts / 152 HTTP requests** complete, including 38 excluded
warmups. The analyzer verifies actual mode/chunk/KV/cache/graph settings, request hashes,
native token counts, full computed-prefill counts and completed outputs. Actual long
prefill units for chunks 1024/1280/1536/2048 are **64/51/43/32**. Every one-token response
is token 818; this is only a smoke observation, not quality or full-output equivalence.

### Measured HTTP throughput

Tok/s **median [minimum, maximum]**; ranges are observed repetitions, not confidence intervals:

| Chunk | Prompt | FP8 | INT8 g64 | INT8 g32 |
|---:|---|---:|---:|---:|
| 1024 | ~1k | 6412 [5926, 6513] | 5357 [4846, 5534] | 4641 [4354, 4744] |
| 1280 | ~1k | 6322 [5734, 6391] | 5505 [5354, 5529] | 4629 [4428, 4721] |
| 1536 | ~1k | 6375 [5673, 6439] | 5463 [4958, 5510] | 4659 [3854, 4738] |
| 2048 | ~1k | 6378 [6072, 6420] | 5428 [5269, 5520] | 4356 [4182, 4731] |
| 1024 | ~65k | 3193 [3019, 3246] | 2904 [2844, 2967] | 2665 [2655, 2719] |
| 1280 | ~65k | 3314 [3309, 3345] | 2951 [2951, 2988] | 2704 [2702, 2733] |
| 1536 | ~65k | 3351 [3175, 3361] | 3010 [2974, 3014] | 2749 [2435, 2760] |
| 2048 | ~65k | 3421 [3187, 3436] | 2991 [2982, 3001] | 2694 [2674, 2705] |

Median **prompt / wire-TTFT milliseconds**, mode order FP8 / g64 / g32:

| Chunk | ~1k prompt / wire TTFT | ~65k prompt / wire TTFT |
|---:|---|---|
| 1024 | 154/176 ; 184/207 ; 213/235 | 20355/20489 ; 22379/22527 ; 24383/24496 |
| 1280 | 156/183 ; 179/209 ; 213/238 | 19611/19751 ; 22018/22145 ; 24032/24184 |
| 1536 | 155/182 ; 181/208 ; 212/273 | 19390/19526 ; 21591/21729 ; 23635/23826 |
| 2048 | 155/182 ; 182/199 ; 226/260 | 18996/19131 ; 21730/21865 ; 24126/24306 |

All short prompts fit in one chunk at every setting: their scatter is not evidence that
smaller long-prefill chunks accelerate this workload. Long FP8 medians at 1024/1280/1536
are **6.68%/3.14%/2.03% below 2048**. G64 at 1024/1280 is 2.90%/1.31% below 2048;
1536 saves only 139 ms median prompt time. A16 controls themselves span **1734–1997 tok/s**
long and **2318–2787** short (before/after long 1997/1734), demonstrating uncontrolled
variation. The historical phase-2 A16/2048 measurement remains the reliable larger baseline; these two
controls are not pooled into it or used to claim a fresh cross-precision speedup.
No timing outlier is silently discarded or causally attributed to paging.

### Measured memory, capacity and contamination

Startup reservations/high-water are identical across FP8/g64/g32 at each chunk. All weight
allocations stay **20,631,888,384 B**; global FP8 KV stays **2,811,002,880 B**. Every start
actually resolves **68,096 context/KV tokens** (1064 page groups). This fixed-capacity
experiment does **not** measure the auto-capacity frontier or a longer-context speed.

| Chunk | Workspace capacity / high-water MiB | BF16 ring MiB | Total runtime reservation MiB | Startup free VRAM GiB, range |
|---:|---:|---:|---:|---:|
| 1024 | 765.008 / 635.000 | 1600 | 5121.839 | 3.707–3.780 |
| 1280 | 827.758 / 635.000 | 1800 | 5387.216 | 3.438–3.531 |
| 1536 | 1051.133 / 856.125 | 2000 | 5813.218 | 2.972–3.105 |
| 2048 | 1530.008 / 1270.000 | 2400 | 6697.347 | 2.127–2.234 |

Relative to 2048, 1536 reduces the measured **runtime reservation by 927,076,352 B =
884.129 MiB**: **478.875 MiB workspace capacity + 400 MiB ring payload + 5.254 MiB
other sequence reservation** (the non-payload residual is not separately attributed).
That is not 884 MiB of KV payload or measured additional token capacity. Workspace
high-water alone falls **413.875 MiB**. Graph measured allocation remains 14 MiB inside
the unchanged 64 MiB allowance. Whole-device sampled peak is **29,004 MiB**, including
desktop; free-VRAM differences also include desktop variation and are not a capacity proof.

Lease `item2e-chunksweep`: exit 0, **1470 s**, minimum free physical RAM **35.257 GiB**;
unchanged 8 GiB floor. All starts succeed. No named build/quantizer/heavy overlap or
second Infernix server was sampled. **All cells nevertheless have nonzero system fault
activity**: 660 raw per-cell samples record available RAM, `Memory\Pages/sec`,
`Memory\Page Reads/sec` and `Memory\Pages Input/sec`; no counter-read failures.
The latter two are disk-backed-fault proxies, not per-process hard-fault counts.
Across the three starts per row, minimum available GiB and maximum sampled rates:

| Mode / chunk | Min RAM GiB | Max Pages/sec / Page Reads/sec / Pages Input/sec |
|---|---:|---:|
| FP8 / 1024 | 36.09 | 749 / 148 / 749 |
| FP8 / 1280 | 36.11 | 27212 / 10369 / 27212 |
| FP8 / 1536 | 35.27 | 1487 / 420 / 1487 |
| FP8 / 2048 | 36.09 | 6470 / 1229 / 6470 |
| g64 / 1024 | 36.07 | 293 / 117 / 293 |
| g64 / 1280 | 36.12 | 46033 / 516 / 46033 |
| g64 / 1536 | 36.09 | 1762 / 1582 / 1762 |
| g64 / 2048 | 36.20 | 108 / 47 / 108 |
| g32 / 1024 | 36.03 | 1140 / 243 / 1140 |
| g32 / 1280 | 36.17 | 1319 / 278 / 1319 |
| g32 / 1536 | 35.36 | 128 / 23 / 128 |
| g32 / 2048 | 35.98 | 697 / 384 / 697 |

These maxima include startup as well as requests; raw timestamps and request-overlap
summaries preserve that distinction. The 46,033 Pages/sec maximum is in g64/1280's first
start (its long request reaches only 27.88); FP8/1280's second long request itself reaches
27,212. A16 controls reach 737/612 Pages/sec. Classification: **system fault activity
observed, process attribution unknown**, not paging-free and not proof the model paged.
Three starts, variable desktop/clock/system conditions and polling limits make small
speed differences inconclusive. The conservative unchanged-2048 decision does not require
pretending the measurements are pollution-free.

Evidence: camp `chunksweep-20261118/{measure.py,summarize.py,identity.json,summary.json}`,
38 per-start directories with request/SSE/response/metrics/startup logs and
`resource-samples.jsonl`; lease receipt `infernix-port-p3-20261112/item2e-chunksweep.{log,receipt.json}`.
The first analyzer attempt expected backend spelling `off`; actual startup telemetry
correctly says `none`. Only that parser expectation was corrected; no GPU run was repeated.
No ~122k arm was needed (it exceeds this fixed 68,096 profile); no ring/global-KV change,
quality campaign or rerun of the completed phase-2/selection tests. Since the chosen chunk remains 2048,
the existing short MTP evidence is retained and no new MTP run is needed.
Tooling protocol/results: `docs/plans/20261118-gemma4-q5-chunk-sweep.md`.

## Calibrated builder on the diode model (2026-11-18)

The canonical `q5-g64` preset was run for real on the immutable BF16 variant `diode-s15-t005`,
with Hessians collected on **that model** (the plain model's calibration is never reused), the
BF16-source MTP assistant and `--gemma4-q5-prefill int8-g64` as the generated launcher's explicit
compute opt-in. All builder steps completed (calibration 26 s, text tree 92 s, GPTQ 1093 s,
checkpoint check 189 s, convert 561 s). Result: **one** A16Only artifact,
`diode-q5-g64.ninfer`, **21,151,124,484 B**, SHA-256
`55d5e8cf1fe33a96e09d0adf4e7330ad3906f55b58ab49d1f90bf155eeed07c7`, plus a build receipt
and a generated `serve-diode-q5-g64.cmd`. The campaign stays in the camp; nothing was promoted.

### Artifact identity

Reader-verified: 706 tensor objects, 240 `q5_g64_fp16` objects, 410 target text-projection
Uses all `A16Only`, 60 text layers, `mtp` present, no vision. Every `text/output_head` and `mtp/*`
binding (including range-split parts) is byte-identical to the same-source G32 best-medium
artifact: the FP8 row vocabulary (`fp8_e4m3fn_row_bf16`) and the BF16-source MTP assistant
are exactly the established encodings. Each object digest is recorded in `artifact-identity.json`.

### Own-Q8 KL against the matched best-medium diode

Instrument `item3-diode-own-q8-block-prefill-v1`: `refs/ref512-en-q8-diode.nklref`
(SHA-256 `695f365e…b49782`) own-Q8 diode reference, context 512, stride 256, chunk 128, chat
prefix, FP8 global KV, BF16 sliding ring, BF16 QK and logits, 47 windows / 13,728 positions,
4000-resample paired window bootstrap, executable
`infernix-perplexity.exe` SHA-256 `c10f25f1…426`. The analyzer asserts identical protocol,
window counts, reference log-probabilities and execution fields across all three rows. The
best-medium row **reproduces the matched 0.0941**, so the instrument is comparable.

| Row | Mean KL [95% CI] | Mean ΔNLL [95% CI] | Top-1 disagreement |
|---|---:|---:|---:|
| best-medium G32 (A16) | 0.0941 [0.0804, 0.1099] | 0.0316 [0.0155, 0.0484] | 9.05% |
| diode Q5 g64, `a16` | **0.0331** [0.0280, 0.0389] | 0.0134 [0.0056, 0.0222] | 5.41% |
| diode Q5 g64, `int8-g64` | **0.0336** [0.0288, 0.0396] | 0.0127 [0.0040, 0.0220] | 5.55% |

Paired against best-medium: A16 KL **−0.0610 [−0.0729, −0.0501]**, ΔNLL −0.0182
[−0.0332, −0.0033]; int8-g64 KL **−0.0605 [−0.0724, −0.0493]**, ΔNLL −0.0188
[−0.0334, −0.0043]. The prior Q5_K_S diode value (0.0403) came from a different encoder and
is a historical context number, not a paired comparison. The two compute modes' KL
intervals overlap almost entirely; this short block-prefill instrument does not separate them.
**Not measured:** long-context quality, incremental-decode quality or INT8 long-context
quality of this diode artifact [INFERENCE: the plain-model incremental figures above are
expected to transfer but were not re-measured here]. No promotion.

### Generated-launcher smoke

The generated launcher (FP8 KV, FP8 QK, BF16 sliding ring, chunk 2048, MTP d1, int8-g64) was
parsed and started unmodified under the GPU lease: 80,960 tokens of context, two chat
requests with default thinking (296 and 271 reasoning tokens, finish `stop`, `top_k` 64,
seed 1234) returned reasoning content and a two-sentence answer; metrics show MTP rounds
(354 verification rounds). This is a short functional smoke, not a quality or speed
measurement.

### Staging

The builder-owned `work/text60` tree (61,427,043,204 B, 20 files) was deleted after
recording per-file SHA-256 in `owned-stage-sha256.json`; it is reconstructible from the
immutable BF16 variant. The GPTQ checkpoint, calibration, build receipts, BF16 sources, plain
text60 and all earlier artifacts were left in place at that time. The plain text60 tree and the Q5 A8-tagged artifact were deleted later; see the current-state note at the top for how to reproduce them.

Evidence: camp `q5-20261117/builder-20261118/{qualify.py,variant/,quality-*,quality-summary.json,
artifact-identity.json,launcher-smoke.json,owned-stage-sha256.json,work/logs}`; lease receipts
`infernix-port-p3-20261112/item3-*.receipt.json`.

## Q5 recommended-profile adoption (2026-11-18)

The measured recommendation is a **Q5-specific launcher default**, not a new
native Engine/CLI/perplexity precision default (those keep `a16`, artifact policy). Historical opt-in/default statements above describe
their respective experiments. The general profile and the Q4/IQ3 tiers remain unchanged (`memory-16gb` is separate: IQ3_XXS only, FP8 ring by default).

`q5-quality` now defaults to INT8 g64 prefill, BF16 rings, FP8 global KV/QK, auto context, chunk 2048,
1024 MiB headroom, top-k 64, thinking enabled, MTP d1 and host-only prefix caching.
`q5-long-context` changes only the default ring to FP8. Explicit compute/ring/QK/sampling/context
choices win; no Q5 profile implicitly enables key-only KV. Omitted `--profile` on a Q5 preset build
or native-Q5 artifact launcher selects `q5-quality`; explicit `--profile default` retains the old
general roleplay profile. Python `ServeSettings`/`ServeProfile` constructors share the same named
profile resolution, rather than relying on argparse to fill them in.

This is a measured tradeoff, **not numerical equivalence or a maximum-fidelity mode**. The INT8 phase-2
g64 chunk-2048 prefill is 1.93x A16 short / 1.51x long, but 14.8% / 12.6% slower than FP8.
Plain short KL is 0.0203 g64 / 0.0199 A16 / 0.0296 FP8. The single-artifact study's bounded incremental g64
uncertainty crosses zero but does not establish equivalence; g32 has a resolved incremental NLL
increase. The real diode builder's own-Q8 short KL is 0.0336 g64 / 0.0331 A16 versus 0.0941
best-medium. Neither model has qualified long-context INT8 quality; diode incremental quality
remains unmeasured. MTP speed variation and all original measurement caveats remain above.

“Quality” identifies Q5 plus BF16 rings. The existing FP8 QK setting is retained **for roleplay/prose**;
it has a measurable fidelity cost. Coding/tool and fidelity-first launchers can explicitly choose
`--gemma4-qk bf16 --gemma4-q5-prefill a16`; no such existing launcher is rewritten.
Chunk 1536, FP8 rings, no MTP, shorter context, explicit experimental key-only KV and explicit
last-resort shared memory form the user option ladder, not a dynamic new profile framework.
Commands and exact memory/performance costs live in the
[conversion guide](../weight-conversion.md#lower-vram-q5-ladder-same-artifact).

### Profile acceptance evidence

Implementation `0dca0b55`; Python behavior suite **202 passed in 16.62 s** with the tooling
Python 3.13.10 environment, plus Python 3.11.9 `py_compile` of both changed modules. The initial
Python 3.11 pytest invocation could not run because that interpreter has no pytest; no dependency
was installed. Tests cover omitted versus explicit profile precedence, all compute overrides,
BF16 ring/QK overrides, direct Python construction, generated commands, native-Q5 trunk-directory
detection (including split bindings, excluding assistant/head-only Q5) and unchanged other tiers.

The actual generated CMDs were run on the verified diode artifact above, with the unchanged
approved absolute server (SHA-256 `63af2c0d…10c8c`), RTX 5090 / CUDA 13.5:
`q5-quality` resolved **81,792** context, INT8 g64 / BF16 ring; `q5-long-context` resolved
**109,376**, INT8 g64 / FP8 ring. Each completed one **174-token prompt** (wide prefill exercised)
with default thinking and MTP d1: 341 / 269 reasoning tokens, 178/205 / 139/168 accepted/drafted,
both finish `stop`. BF16 ring/QK plus A16 compute overrides and explicit general `default` were
proved through parser/direct-API/generated-config agreement, not additional GPU runs.
These are startup/short-function checks, not long-context quality or new performance measurements.

The first smoke lease failed its second start because its harness stopped only `cmd.exe`, leaving
its own first server resident. That server was identified and terminated; cleanup now stops the
owned process tree. Completed recommended-profile evidence was retained; only the refused
long-context start was retried successfully. No server remained afterward. Lease durations
65.81 / 44.37 s, minimum physical RAM 25.58 / 29.03 GiB, unchanged 8 GiB floor; artifact and
server SHA-256 values match before/after both leases. No build or quality campaign was repeated.

Evidence: camp `q5-20261117/profiles-20261118/{configuration-proof.json,acceptance.json,
verify_profiles.py,*.cmd,*response.json,*metrics.txt,*.log}`; lease receipts
`infernix-port-p3-20261112/q5-profiles-20261118*.receipt.json`.
