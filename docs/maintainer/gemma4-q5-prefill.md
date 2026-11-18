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
Speed is not inferred: the emulator is slow by construction, and INT8 versus FP8 tensor-core throughput,
per-K32 scale-apply cost and activation quantization cost on these shapes remain to be microbenchmarked.
Limits: one plain model, short pinned protocol, intervals about +-0.0025 KL, MTP off, no long-context
re-measurement; Brontir's cross-model KL numbers are not evidence here. Plan and full record:
tooling `docs/plans/20261118-gemma4-q5-compute-precision.md`; evidence in the camp's `q5-precision-20261118/`.
