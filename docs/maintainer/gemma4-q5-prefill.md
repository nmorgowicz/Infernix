# Gemma 4 GPTQ5 prefill attribution (2026-11-18)

## Scope and implementation diagnosis

Item 2a is measurement and diagnosis only: the existing plain GPTQ5 group-64 A16 and W5A8
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

## Shared-tile external concept

The local NInferno `generic_tiled.cuh` at `e99ce56a3918d6f86427cc8cecc3b2b0597a021c`
is not evidence that shared-tile fusion is missing here.
Both implementations already decode packed weights into a shared BF16 tile. Its non-Volta
path instead stores exact integer codes in that tile and scales an FP32 partial at each group
boundary, with a conventional CTA-synchronized pipeline and activation-as-MMA-A orientation.
The current native wide ring scales/rounds weights before the MMA, overlaps producer/consumer
warps, and keeps one complete-K FP32 chain. The external grouped accumulation therefore changes
rounding/reduction semantics as well as scheduling; it is not a drop-in removal of a redundant
BF16 global pass. Relevant mechanisms must be compared against these actual routes and Gemma
shapes, not the name “fused dequant.” No port is implemented or authorized by this attribution.

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
request/SSE/metrics/logs, and the parent `infernix-port-p3-20261112/item2a-*.receipt.json`.
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
quality gates and artifact/reference identities remain binding. Parent review precedes 2c or
the separately approved serving-profile changes.

## Compute-precision emulation (item 2b, 2026-11-18)

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
Speed was measured separately in the next section (item 2c phase 1).
Limits: one plain model, short pinned protocol, intervals about +-0.0025 KL, MTP off, no long-context
re-measurement; Brontir's cross-model KL numbers are not evidence here. Plan and full record:
tooling `docs/plans/20261118-gemma4-q5-compute-precision.md`; evidence in the camp's `q5-precision-20261118/`.

## INT8 per-32 kernel microbenchmark (item 2c phase 1, 2026-11-18)

Standalone CUDA microbenchmark, **no engine change during phase 1**. The per-32 soft gate was not met;
the parent subsequently chose to proceed with both group sizes in phase 2 below. Sources live in the camp (`q5-int8-20261118/`: `q5_i8.cuh` kernel, `bench.cu`, `probe.cu`,
`oracle_check.py`, `summarize.py`, runner logs). The baseline is the engine's own native A16 code, instantiated from
the same headers: `launch_rowsplit_ring_linear<Q5RingFormat, RowSplitRingTile128, 3, 4>` for T = 2048 and
`q5_a16_mma_kernel<MmaR32T128>` for T = 128. RTX 5090 (170 SMs), CUDA 13.5 `sm_120a`, plain GPTQ5 g64 layout on
the six real shapes with random weights/activations; L2 flushed before each of 15 events-timed repetitions, median of
three serial lease cells (`item2c-final-r1..3`, exit 0, 9.1 s each, minimum free RAM 42.2 GiB, floor 8 GiB, no
build or other heavy job overlapped; Pages/sec samples 15.9 / 16.0 / 1,601.9 with >= 43.2 GiB available, so no
paging-free claim). Native A16 reproduces the in-situ rates of item 2a (191-208 TFLOP/s versus 177-193).

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

**Verification before timing.** Independent NumPy/FP64 oracle (`oracle_check.py`; real captured 2b operands tiled to
the full N and T with per-row scale and per-token magnitude variation; 46 configuration/T combinations over both
group sizes, fused / TMA / pre-baked / K64 variants): the activation quantizer matches the independent NumPy rule
with **0** mismatching codes or scales; 979,695 sampled BF16 outputs match the FP32 group-chain emulation
(`t = fl32(dot*as)`, `run = fl32(t*ws + run)`, ascending groups) with **0** disagreements; worst relative L2 to the
FP64 ideal 0.00199 (the BF16 output rounding; native A16 versus exact Q5 x BF16 is 0.0016-0.0021). Against the 2b
emulator captures (real rows/tokens, 256 elements per shape) 4-19 elements per shape differ by one BF16 ulp
(relative L2 4e-5 to 4.3e-4): the 22-bit activation-scale mantissa and the association of the FP32 scale products,
as 2b predicted; 4-48 of 43k-172k int8 codes per shape differ from the 2b quantizer for the truncated scale.

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
scales item 2a's in-situ cumulative A16 projection time of that shape (20.62 s); quantizer and bake calls are counted
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
chunk), per-64 16.12 s. These chunk-128 projections split item 2a's A16 total by call-weighted measured per-call time
[INFERENCE for the split]; the 65k layer-interval projection (not measured) is 23.55 s for per-32 and 21.76 s for
per-64, against 32.49 s native A16 and 19.06 s W5A8.

**Verdict: the per-32 gate is not met.** It projects 1.77x faster than native A16 at T = 2048 (the 1.3x criterion
holds) but 1.63x the W5A8 stack (the ~1.5x criterion fails by 8 %; a fused quantizer at the memory floor would give
about 1.59x [INFERENCE]). Per-64 (KL 0.0204 in 2b against 0.0200, not separable at that sample size) projects 1.38x and passes.
The parent assistant (not the operator) then decided to proceed with both group sizes behind an explicit opt-in load-time compute mode
(A16 stays the default); the engine kernel, the mode and their model-level checks are the next section.

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
The FP8 reference is item 2a's in-situ number rather than the same harness. Phase 2 supplies the engine checks below.
Tooling plan: `docs/plans/20261118-gemma4-q5-int8-per32-microbench.md`.

## INT8 prefill engine kernel (item 2c phase 2, 2026-11-18)

Implemented in `74eff759`: `--gemma4-q5-prefill a16|int8-g32|int8-g64`, default `a16`, across Engine,
CLI, serving, native perplexity and the builder-generated launcher/serve-check profile. No profile enables INT8
implicitly. **The default preserves the artifact's existing A16Only/AllowA8 policy**: an A16 artifact stays A16;
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

Pinned native full-vocabulary KL: the same plain A16 artifact and Q8 reference as 2b, context 512,
stride 256, chunk 128, chat prefix, FP8 global KV, BF16 ring/QK/logits; **47 windows / 13,728 positions**,
4000-resample paired window bootstrap:

| Route | Mean KL [95% CI] | dKL versus default A16 [95% CI] | dNLL versus A16 | Top-1 disagreement |
|---|---|---|---:|---:|
| default A16 | 0.019914 [0.017267, 0.022466] | 0 | 0 | 0.04422 |
| INT8 g32 | 0.020425 [0.017716, 0.022952] | +0.000511 [-0.000095, +0.001127] | +0.002021 | 0.04538 |
| INT8 g64 | 0.020305 [0.017541, 0.022987] | +0.000391 [-0.000416, +0.001178] | +0.001506 | 0.04356 |
| W5A8 (retained matched 2b result) | 0.029649 [0.026021, 0.033243] | +0.009735 | +0.003517 | 0.05427 |

Default A16 reproduces every retained 2b position bit for bit. Native g32 versus its 2b emulator is
+0.000422 [-0.000149, +0.000997] KL; g64 -0.000081 [-0.000647, +0.000426]. Neither is bit-identical to
the emulator (truncated scale/FP32 association), but this sample does not resolve an aggregate difference.
Nor does it establish exact equivalence to A16 or a quality ranking between g32/g64 (paired g64 minus g32
-0.000120 [-0.000728, +0.000470]). Both recover most of W5A8's quality gap: dKL -0.009224 / -0.009344.
Worst individual window increases versus A16 are +0.00636 / +0.00746; do not replace that distribution with
the mean. Long-context quality is unmeasured. The KL lease exits 0 in 166 s with >=36.62 GiB free RAM.

### Clean HTTP prefill

All arms use the same executable SHA-256 `ec6905ca4cc799623862814439316d56083d0bf68f1a2967e71b37bdeb83db53`,
with no event-instrumentation switch. Same hardware/power, artifacts and requests as item 2a; context/KV 68,096,
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
All **60** prefill requests return the same first token (818), also matching the retained A16/W5A8 item-2a
requests. That is a smoke observation, not proof of full-output/logit equivalence.

The three saved cold-L2 microbench repetitions of the integrated configurations (15 CUDA-event samples each)
give T=2048 GEMM medians in shape order sliding QKV/global QK/sliding out/global out/gate-up/down:
g32 **1.036/1.148/0.506/0.981/2.879/1.301 ms**, g64 **0.873/0.963/0.418/0.809/2.458/1.080 ms**.
G32 quantization costs 0.029/0.029/0.046/0.083/0.030/0.108 ms; these are kernel numbers, not HTTP totals.
The harness's `projection_65k_ms` rows are ratio-based projections using item-2a chunk-2048 weights, not
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
becomes a default. No tuning campaign, item 2d, model promotion, artifact rewrite or production build change.
Evidence: `q5-int8-20261118/phase2/{oracle-engine,kl-summary,http-summary}.json`, test dumps, benchmark
sources, request/SSE/metrics/logs, and parent `int8-phase2-*.receipt.json`. Plan:
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

## Single-artifact compute selection and incremental scoring (item 2d, 2026-11-18)

Item 2c already implemented INT8 launch-time selection, scoped workspace, graph routing, cache signatures,
CLI/serve/perplexity/builder propagation and the narrow A16 split. Item 2d reuses those mechanisms.
The same `--gemma4-q5-prefill` now accepts `fp8`, enabling the existing W5A8 bake for all four prepared
native Q5 trunk projections (QKV/QK, output, gate/up, down), including trunk verification. No second flag,
artifact rewrite, copied artifact or persistent extra weight representation is needed.

The historical `a16` default remains **artifact policy**, not a forced override: an A16Only artifact runs
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
verification are reported below; no default, promotion, long-context qualification or item-2e sweep follows
from this instrumentation.
