# Gemma 4 31B: choose a card, build and serving policy

Status: 2026-11-28. This guide covers **31B dense BF16 fine-tunes**, text only, not 26B-A4B. A weight recipe, prefill compute route, KV/ring policy and MTP setting are separate decisions. `best-medium` remains the general medium default; `best-small` remains the recommended smaller Q4. Compatible global-FP8 roleplay builder/profile/generated-launcher routes default to FP8 sliding storage; explicit BF16 storage wins. Bundled quality/balanced/context **recommendations are labels** resolved to preset + profile + flag combinations (`--list-presets`); they are not new CLI profiles.

## Decide by usable memory, not the box label

Plan over **1.5–2.1 GiB desktop overhead**, usually **1.5–1.7 GiB**, and accept the actual free budget rather than imposing a fixed deduction. Minimum desired resident contexts are 32,768 / 65,536 / 131,072 tokens for 16 / 24 / 32 GB classes. These are requirements, not promises. Read startup and `/v1/models`; account for output tokens too.

| Card class | Practical starting point | Qualification and tradeoff |
|---|---|---|
| 16 GB, e.g. RTX 5070 Ti / 5080 | Bundled: balanced = `forum-iq3-16gb --assistant-encoding q4-g32` + `memory-16gb`; context = the same preset with `q4-g256`; quality = `forum-iq3-16gb-quality` (MTP off by default). Research-only labels, measured on the diode finetune. Public fallback: `iq3-xxs` plus `memory-16gb`; inspect resolved context, optionally disable MTP | F2 bundles passed explicit 32k+MTP under active ballast (dedicated peaks 13,815.95 / 13,791.95 MiB, g32 / g256); fresh-superiority unresolved, historical <0.85 gates still failed. The public XXS recipe **did not meet 32k with the registered 64 MiB margin**. IQ3_XS and Q4/Q5 do not solve this envelope. Standard RTX 5070 is not a 16 GB example. |
| 24 GB, including laptop class | `best-small` context-first or `best-medium` balanced, ordinary FP8 global KV/FP8 ring, chunk512 | Current medium plans span 94,400→80,256 MTP tokens across 1.5→2.1 GiB overhead. Mixed-A ordinary FP8-ring plans reach 64k only at 1.5/1.7 GiB (73,728/68,992; below 64k at 1.9/2.1). Q5 does not reach 64k. Plans are not real starts or laptop qualification; key-only remains a separate explicit trade. |
| 32 GB desktop | `best-medium`, ordinary FP8 global KV/FP8 ring, chunk2048, MTP | Current study starts resolved 131,072 for Q4/IQ3/mixed-A with either ring. Q5 ordinary-FP8-ring 131k at overhead through 1.9 GiB is derived sizing; a lower-free-budget real start resolved 112,768 (BF16 ring 99,200). Inspect actual startup instead of promising 131k or adding key-only implicitly. |

Only the **AM4/DDR4/PCIe4 x16 RTX 5090 Windows desktop** was measured. Historical smaller-card starts used active ballast; current card-range rows are plans/derived sizing, not physical smaller-card fits or speed. Neither laptop speed nor Linux is measured. The old fixed usable budgets 22,509 /14,357 /29,473 MiB for 24 /16 /32 GB correspond to 1.9 GiB overhead and are historical anchors, not required current budgets. Actual WDDM availability and other applications govern.

**Research presets, now reproducible from any BF16 fine-tune** (still research-only labels, measured on the diode
finetune, never defaults): `forum-iq3-16gb` is the measured F2 recipe — the balanced32k bundle uses
`--assistant-encoding q4-g32`, the smallest/context bundle `q4-g256`; both passed explicit 32k+MTP under ballast.
`forum-iq3-16gb-quality` is F1 (launcher without MTP, explicit `--spec mtp --draft-tokens 1 --allow-shared-memory`
override launcher beside it). `q4q5-mixed-b` is the measured higher-fidelity mixed candidate (fresh KL 0.62x
best-medium). All three require your **own** calibration: the forum presets build their imatrix from the pinned
forum roleplay corpus of your own model (`--rp-raw` from `roleplay_heldout fetch` plus `--rp-against` evaluation
texts; `--calibration-text` is refused), and mixed-B runs two GPTQ passes on your own Hessians. Supplying forum
calibration to plain `iq3-xxs` does **not** reproduce F2: the allocation and the tied `q4_g32` vocabulary are part
of the preset.

## What the supported recipes buy

| Builder preset | Measured diode native bytes including assistant | Use |
|---|---:|---|
| `iq3-xxs` | 13,019,175,940 | Smallest public recipe, highest fidelity cost here |
| `iq3-xxs-feedback` (research-only, measured on the diode finetune at XXS bytes) | 13,019,175,940 planned (same bytes as `iq3-xxs`) | The registered X5 Hessian-feedback recipe at XXS bytes: measured KL ratios 0.44-0.53 of deployed XXS, confirmed at the historical strength on an untouched confirmation draw; needs the research tooling checkout and a GPU lease; imatrix defaults to the campaign-measured generic templated corpus, the roleplay option is an unmeasured stack |
| `iq3-xs` | 13,902,732,292 | Intermediate size; not a resident32k solution for the 16 GB envelope |
| `best-small` | 16,852,045,060 | Context/size-first Q4 |
| `best-medium` | 18,405,559,556 | General medium baseline |
| `q4q5-mixed-a` | 19,344,788,484 | Q5 attention output/down, Q4 elsewhere; better measured roleplay KL at extra bytes/decode cost |
| `q5-g64` | 21,151,124,484 | Higher-fidelity weight tier; INT8-g64 prefill is the Q5 profile balance |
| `q4q5-mixed-b` (research-only) | 20,139,576,324 | Q5 `gate_up` all layers + `down` 20-59; measured fresh roleplay KL 0.6196 [0.6061, 0.6324] of best-medium; two GPTQ passes |
| `forum-iq3-16gb` (research-only) | 12,630,891,524 (FP8 assistant); 12,431,956,996 (q4-g32); 12,407,184,388 (q4-g256) | The measured F2 16 GB recipe: forum roleplay imatrix of your own model, pinned allocation, tied q4_g32 vocabulary |
| `forum-iq3-16gb-quality` (research-only) | 13,173,652,484 planned (measured artifact +4,096 B) | The measured F1 16 GB recipe; launcher MTP-off, explicit override launcher beside it |

Sizes belong to one fine-tune, not guaranteed output sizes. KL against the model's **own Q8** measures numerical fidelity, not human roleplay quality. Do not average short and long instruments or infer PPL from KL. Sparse/missing PPL is unmeasured.

**Hessian-feedback IQ recipe (research tool; preset `iq3-xxs-feedback`).** The research tool (`llama-local-tooling` `converter/gguf/iq_hessian_feedback.py`) applies iterative Hessian error feedback while solving existing IQ blocks; on the diode finetune it produced an IQ3_XXS-tier candidate at **exactly the same encoded bytes** as the deployed XXS recipe (container byte-identical except projection payload bytes; no format, size or inventory drift). Measured against the deployed XXS (paired thread-cluster 95%, 4,000 resamples): RP-dev ratio 0.4388 [0.4248, 0.4551], fresh 0.4463 [0.4225, 0.4714], 65k-16 0.5263 [0.5024, 0.5502], and an untouched new short confirmation 0.4455 [0.4239, 0.4720] — confirmed at the historical <0.85 strength. The builder now exposes it as the research-only preset `iq3-xxs-feedback` (see [weight-conversion.md](weight-conversion.md): the recipe is reproducible from any BF16 fine-tune; the campaign measurement was on the diode finetune at XXS bytes with the generic templated imatrix, so the preset's roleplay-imatrix option is an unmeasured stack, and no end-to-end user-finetune qualification has run yet). Not a default; capacity anchors quoted for 16 GB are measured on the F2 allocation, not this container.

Historical clean warmed65k MTP-on medians on that 5090: best-medium **3380 PP /95.0 TG**, mixed-A **3234/83.5**, mixed-B **3110/76.1** tok/s. Three cold starts per arm, Latin-square order, allocation warmups excluded,256-token caps, cache fully disabled, FP8 global KV/QK and BF16 ring, chunk2048, pre-staging frozen engine. These are throughput probes, not current-layout speed or completed-answer quality. Ranges are start-level ranges, **not confidence intervals**. Full observations, exact hashes, counters and quality intervals are in the [receipt matrix](https://github.com/nmorgowicz-org/llama-local-tooling/blob/research/tooling-reorganization/docs/GEMMA4-DECISION-MATRIX.md).

## BF16 fine-tune to supported artifact

Run from a clean infernix checkout. Preserve the full BF16 HF source/config/tokenizer read-only and use separate fresh output/work directories. Supply the **separate BF16 `google/gemma-4-31B-it-assistant` source** for MTP; a trunk GGUF cannot reconstruct it. Use a supported Blackwell server build; see [project build instructions](../README.md). The converter uses Python3.11 with torch/numpy/safetensors. GPTQ uses its separate pinned Python3.13 environment (`tools/gptq/requirements.txt`, compressed-tensors0.18.0 / LLM Compressor0.13.0).

These commands describe future conversion; they are **not CPU-only operations** and were not executed while writing this guide.

```text
python -m tools.convert.gemma4_build setup-gptq --venv .venv-gptq
python -m tools.convert.gemma4_build --preset best-medium --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/best-medium" --work-dir "<scratch>/best-medium" --serve "<serve>" --check
python -m tools.convert.gemma4_build --preset best-medium --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/best-medium" --work-dir "<scratch>/best-medium" --serve "<serve>" --yes
```

For `best-small` change the preset and both directories. For `q5-g64` likewise; its launcher selects `q5-quality` (INT8-g64/FP8 ring with global FP8/auto context). For mixed-A use the [documented complete command](weight-conversion.md#builder-preset-q4q5-mixed-a-supported-option-not-the-default), explicitly selecting `--gemma4-q5-prefill int8-g64`. Use explicit `--context auto` when building to avoid relying on version-specific implicit context defaults. Legacy builder-written default-profile launchers request131072 and can refuse with MTP; an explicitly requested131072 can still refuse. For an existing artifact, generate an auto-context launcher:

```text
python -m tools.convert.gemma4_build launcher --artifact "<mixed-A.ninfer>" --serve "<serve>" --out "<camp>/serve-mixed-auto.cmd" --context auto
```

Native Q5 detection selects `q5-quality` for this launcher, not a new weight conversion. Q4 cells retain stored A8; Q5 narrow decode stays A16. Explicit `--context` wins; explicit `--profile default` retains its historical fixed131072 unless context is overridden. Since `13356265`, mixed-A and mixed-B builds use auto context when **both profile and context are omitted**; other profiles and explicit settings retain their contracts. The implementation's CPU builder regression suite passed227/227; actual GPU `serve-check` remains pending. CPU launcher coverage is not runtime or capacity qualification.

IQ3 also needs GPU-enabled `llama-imatrix`, `llama-quantize`, their binary directory and llama.cpp source containing `convert_hf_to_gguf.py`; converter Python additionally needs gguf/transformers/sentencepiece/tokenizers, and pyarrow for default corpus export. It does **not** use GPTQ:

```text
python -m tools.convert.gemma4_build --preset iq3-xxs --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xxs" --work-dir "<scratch>/iq3-xxs" --yes
python -m tools.convert.gemma4_build launcher --profile memory-16gb --artifact "<IQ3_XXS.ninfer>" --serve "<serve>" --out "<camp>/serve-16gb.cmd"
```

For XS change preset/directories to `iq3-xs`; `memory-16gb` is specifically XXS. It plans MTP first and can fall back to off while keeping the assistant embedded. This is not a 32k guarantee.

Use `--dry-run` instead of `--yes` to inspect exact stages. Q4 requires approximately75GB minimum/103GB comfortable scratch; Q5 85/115GB; mixed-A102/132GB; mixed-B110/140GB; IQ3 at least100/120GB, excluding source and allowing separate final output space. These are planning estimates; preflight governs. IQ3 exports BF16, makes own-model Q8 for imatrix collection, quantizes pinned tensor allocations and imports blocks verbatim. GPTQ collects own-model Hessians; never borrow another fine-tune's matrix/checkpoint. Same-argument reruns resume; changed input/preset needs fresh work. Receipts, native verification, `BUILD-SUMMARY.md` and launcher are automated. `--keep-work` retains useful research intermediates but IQ3 still prunes BF16/Q8 after last use.

```text
python -m tools.convert.gemma4_q4 verify --artifact "<artifact.ninfer>"
python -m tools.convert.gemma4_build serve-check --launcher "<serve.cmd>"
```

Storage verification is not runtime qualification; serve-check is an HTTP smoke, not capacity or quality certification. See [complete conversion reference](weight-conversion.md#gemma-4-31b-builder-presets-and-the-self-contained-workflow).

## Choose runtime independently

Generate launchers with `launcher --artifact ... --serve ... --out ...`, adding documented options such as `--context auto`, `--prefill-chunk 512` or `--no-mtp` deliberately. Compatible global-FP8 roleplay builder/profile routes select FP8 ring storage, including general, `q5-quality` and the retained `q5-long-context`; neither Q5 profile adds key-only KV. Explicit `--gemma4-sliding-kv bf16` wins; omitted ring with global `--kv-dtype bf16` resolves BF16. Native Engine/CLI ring default stays BF16. `--gemma4-qk bf16` changes compute only, not storage; add the explicit BF16-ring flag when BF16 storage is wanted, and Q5 `--gemma4-q5-prefill a16` for A16 projections. Preserve existing explicit fidelity configurations. Key-only globals remain FP8-only and experimental; shared memory is an explicit last resort, not resident fit.

Current staged-ring BF16→FP8 savings are about **587 MiB reservation /620 MiB process at chunk2048**,
and **563 /596 MiB at chunk512**, at matched fixed capacity. Current paired timing is near parity;
old FP8-kernel gains and historical up-to-12.5% penalties are not current BF16/FP8 effects.
Short, 32k and shipped-Q5 roleplay fidelity costs and resolved mixed-A/Q5 65k costs remain; Q5 sampled MTP decode
was about 1.3% lower, while pooled acceptance intervals include zero. No equivalence or human-quality
claim follows. [Final paired evidence](https://github.com/nmorgowicz-org/llama-local-tooling/blob/research/tooling-reorganization/docs/plans/20261127-gemma4-sliding-kv-paired-study.md);
[adoption measurements, 32k/short/MTP-on gaps and real-start gates](https://github.com/nmorgowicz-org/llama-local-tooling/blob/research/tooling-reorganization/docs/plans/20261128-gemma4-fp8-ring-adoption.md).

## Calibration is not evaluation

The builder's default pinned English calibration is `eaddario/imatrix-calibration` revision `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4`. `--calibration-text FILE` supplies disjoint training/calibration input, not an evaluation corpus.

[Reproducible roleplay tools and exact commands](roleplay-heldout.md) separately implement `tools.eval.roleplay_calibration` and `tools.eval.roleplay_heldout`, using dataset `lemonilia/roleplaying-forums-raw` revision **`6ba79e71d8266428eee590f634bf77aedead39b9`**. Fetch/build/check, exclusions, shingle checks, hashes, environment versions and paired KL commands are documented there. The dataset card reports CC-BY-4.0; no broader licensing conclusion is made here. Raw forum text stays outside the repository. If reused evaluation windows influence selection they become development evidence, as for the combined F2 study. Builder `--score` uses its own FP8-row reference and is not automatically the own-Q8 instrument used above.
