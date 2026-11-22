# Gemma 4 31B: choose a card, build and serving policy

Status: 2026-11-22. This guide covers **31B dense BF16 fine-tunes**, text only, not 26B-A4B. A weight recipe, prefill compute route, KV/ring policy and MTP setting are separate decisions. `best-medium` remains the general medium default; `best-small` remains the recommended smaller Q4. No new profiles are implemented by this guide.

## Decide by usable memory, not the box label

The planning envelope reserves approximately **1.9 GiB** for the desktop. Minimum desired resident contexts are 32,768 / 65,536 / 131,072 tokens for 16 / 24 / 32 GB classes. These are requirements, not promises. Read startup and `/v1/models`; account for output tokens too.

| Card class | Practical starting point | Qualification and tradeoff |
|---|---|---|
| 16 GB, e.g. RTX 5070 Ti / 5080 | Supported `iq3-xxs` plus `memory-16gb`; inspect resolved context, optionally disable MTP | Existing public recipe **did not meet 32k with the registered 64 MiB margin** in active-ballast tests. IQ3_XS and Q4/Q5 do not solve this envelope. Standard RTX 5070 is not a 16 GB example. |
| 24 GB, including laptop class | `best-small`, FP8 global KV/BF16 ring, chunk512; or `best-medium` with FP8 ring for 64k+MTP | Ballast study: best-small 91.4k and medium with FP8 ring 71.2k auto MTP capacity. Mixed-A needs key-only KV to reach 64k with MTP (67.6k BF16 ring). Q5 did not meet 64k even with key-only/FP8 ring. Laptop WDDM gap and overhead are assumptions, not measured. |
| 32 GB desktop | `best-medium`, FP8 global KV/BF16 ring, chunk2048, MTP | Physically observed 131,072 startup capacity in this environment. Mixed-A reaches 131k with FP8 ring or key-only; ordinary BF16-ring auto can resolve below it. Q5 needs key-only for the registered 131k requirement. Fidelity costs must be chosen explicitly. |

Only the **AM4/DDR4/PCIe4 x16 RTX 5090 Windows desktop** was measured. Smaller cards were emulated by active ballast on that GPU; neither their speed nor a laptop's speed is measured. Linux is unmeasured. The 24 GB envelope assumes 22,509 MiB usable; 16 GB assumes 14,357 MiB; 32 GB study uses 29,473 MiB. Real WDDM budgets and other applications can lower these.

**Research, not public presets:** forum-calibrated F2 with Q4g32 assistant is the proposed balanced32k option; Q4g256 is the smallest option. Both passed explicit32k+MTP in ballast. Mixed-B is a measured higher-fidelity mixed candidate. None is a one-call builder preset. Do not invent preset names or assume that `--calibration-text` alone reproduces the F2 tensor/vocabulary/assistant allocation.

## What the supported recipes buy

| Builder preset | Measured diode native bytes including assistant | Use |
|---|---:|---|
| `iq3-xxs` | 13,019,175,940 | Smallest public recipe, highest fidelity cost here |
| `iq3-xs` | 13,902,732,292 | Intermediate size; not a resident32k solution for the 16 GB envelope |
| `best-small` | 16,852,045,060 | Context/size-first Q4 |
| `best-medium` | 18,405,559,556 | General medium baseline |
| `q4q5-mixed-a` | 19,344,788,484 | Q5 attention output/down, Q4 elsewhere; better measured roleplay KL at extra bytes/decode cost |
| `q5-g64` | 21,151,124,484 | Higher-fidelity weight tier; INT8-g64 prefill is the Q5 profile balance |

Sizes belong to one fine-tune, not guaranteed output sizes. KL against the model's **own Q8** measures numerical fidelity, not human roleplay quality. Do not average short and long instruments or infer PPL from KL. Sparse/missing PPL is unmeasured.

Latest clean warmed65k MTP-on medians on that 5090: best-medium **3380 PP /95.0 TG**, mixed-A **3234/83.5**, mixed-B **3110/76.1** tok/s. Three cold starts per arm, Latin-square order, allocation warmups excluded,256-token caps, cache fully disabled, FP8 global KV/QK and BF16 ring, chunk2048. These are throughput probes, not completed-answer quality. Ranges are start-level ranges, **not confidence intervals**. Full observations, exact hashes, counters and quality intervals are in the [receipt matrix](https://github.com/nmorgowicz-org/llama-local-tooling/blob/research/tooling-reorganization/docs/GEMMA4-DECISION-MATRIX.md).

## BF16 fine-tune to supported artifact

Run from a clean infernix checkout. Preserve the full BF16 HF source/config/tokenizer read-only and use separate fresh output/work directories. Supply the **separate BF16 `google/gemma-4-31B-it-assistant` source** for MTP; a trunk GGUF cannot reconstruct it. Use a supported Blackwell server build; see [project build instructions](../README.md). The converter uses Python3.11 with torch/numpy/safetensors. GPTQ uses its separate pinned Python3.13 environment (`tools/gptq/requirements.txt`, compressed-tensors0.18.0 / LLM Compressor0.13.0).

These commands describe future conversion; they are **not CPU-only operations** and were not executed while writing this guide.

```text
python -m tools.convert.gemma4_build setup-gptq --venv .venv-gptq
python -m tools.convert.gemma4_build --preset best-medium --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/best-medium" --work-dir "<scratch>/best-medium" --serve "<serve>" --check
python -m tools.convert.gemma4_build --preset best-medium --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/best-medium" --work-dir "<scratch>/best-medium" --serve "<serve>" --yes
```

For `best-small` change the preset and both directories. For `q5-g64` likewise; its launcher selects `q5-quality` (INT8-g64/BF16 ring/auto context). For mixed-A use the [documented complete command](weight-conversion.md#builder-preset-q4q5-mixed-a-supported-option-not-the-default), explicitly selecting `--gemma4-q5-prefill int8-g64`. Use explicit `--context auto` when building to avoid relying on version-specific implicit context defaults. Legacy builder-written default-profile launchers request131072 and can refuse with MTP; an explicitly requested131072 can still refuse. For an existing artifact, generate an auto-context launcher:

```text
python -m tools.convert.gemma4_build launcher --artifact "<mixed-A.ninfer>" --serve "<serve>" --out "<camp>/serve-mixed-auto.cmd" --context auto
```

Native Q5 detection selects `q5-quality` for this launcher, not a new weight conversion. Q4 cells retain stored A8; Q5 narrow decode stays A16. Explicit `--context` wins; explicit `--profile default` retains its historical fixed131072 unless context is overridden. An implicit mixed-A builder-auto correction is tracked separately from actual GPU `serve-check`: CPU launcher regression coverage alone is not runtime qualification, and no new GPU serve-check was performed for this guide.

IQ3 also needs GPU-enabled `llama-imatrix`, `llama-quantize`, their binary directory and llama.cpp source containing `convert_hf_to_gguf.py`; converter Python additionally needs gguf/transformers/sentencepiece/tokenizers, and pyarrow for default corpus export. It does **not** use GPTQ:

```text
python -m tools.convert.gemma4_build --preset iq3-xxs --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xxs" --work-dir "<scratch>/iq3-xxs" --yes
python -m tools.convert.gemma4_build launcher --profile memory-16gb --artifact "<IQ3_XXS.ninfer>" --serve "<serve>" --out "<camp>/serve-16gb.cmd"
```

For XS change preset/directories to `iq3-xs`; `memory-16gb` is specifically XXS. It plans MTP first and can fall back to off while keeping the assistant embedded. This is not a 32k guarantee.

Use `--dry-run` instead of `--yes` to inspect exact stages. Q4 requires approximately75GB minimum/103GB comfortable scratch; Q5 85/115GB; mixed-A102/132GB; IQ3 at least100/120GB, excluding source and allowing separate final output space. These are planning estimates; preflight governs. IQ3 exports BF16, makes own-model Q8 for imatrix collection, quantizes pinned tensor allocations and imports blocks verbatim. GPTQ collects own-model Hessians; never borrow another fine-tune's matrix/checkpoint. Same-argument reruns resume; changed input/preset needs fresh work. Receipts, native verification, `BUILD-SUMMARY.md` and launcher are automated. `--keep-work` retains useful research intermediates but IQ3 still prunes BF16/Q8 after last use.

```text
python -m tools.convert.gemma4_q4 verify --artifact "<artifact.ninfer>"
python -m tools.convert.gemma4_build serve-check --launcher "<serve.cmd>"
```

Storage verification is not runtime qualification; serve-check is an HTTP smoke, not capacity or quality certification. See [complete conversion reference](weight-conversion.md#gemma-4-31b-builder-presets-and-the-self-contained-workflow).

## Choose runtime independently

Generate launchers with `launcher --artifact ... --serve ... --out ...`, adding documented options: `--context auto`, `--prefill-chunk 512`, `--no-mtp`, `--gemma4-sliding-kv fp8`, or `--gemma4-key-only-kv` as deliberately selected. General rings are BF16; FP8 ring saves memory but changes numerics. Key-only is FP8-only and experimental. `--profile q5-long-context` changes ring, not weights. `--gemma4-qk bf16` and Q5 `--gemma4-q5-prefill a16` are fidelity-first arithmetic overrides, not a measured human-quality guarantee. `--allow-shared-memory` is an explicit last resort, not resident operation or free capacity. Never silently trade MTP, context or arithmetic for fit.

## Calibration is not evaluation

The builder's default pinned English calibration is `eaddario/imatrix-calibration` revision `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4`. `--calibration-text FILE` supplies disjoint training/calibration input, not an evaluation corpus.

[Reproducible roleplay tools and exact commands](roleplay-heldout.md) separately implement `tools.eval.roleplay_calibration` and `tools.eval.roleplay_heldout`, using dataset `lemonilia/roleplaying-forums-raw` revision **`6ba79e71d8266428eee590f634bf77aedead39b9`**. Fetch/build/check, exclusions, shingle checks, hashes, environment versions and paired KL commands are documented there. The dataset card reports CC-BY-4.0; no broader licensing conclusion is made here. Raw forum text stays outside the repository. If reused evaluation windows influence selection they become development evidence, as for the combined F2 study. Builder `--score` uses its own FP8-row reference and is not automatically the own-Q8 instrument used above.
