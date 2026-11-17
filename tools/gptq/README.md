# GPTQ recipe runner (`--gptq-source`)

These scripts produce the checkpoint that `python -m tools.convert.gemma4_q4 convert --gptq-source DIR` imports:
LLM Compressor GPTQ, symmetric INT4 or explicit group64 INT5, static contiguous groups, no activation order.
The converter copies the codes and the (binary16-exact) scales word for word; nothing is re-rounded.
Measured on the author's calibrated INT4 31B this route halves the KL of calibrated RTN at group 32
(see `docs/maintainer/gemma4-benchmarks.md`),
so use the recommended `imatrix_mse` observer: the default `memoryless_minmax` observer is worse than calibrated RTN.
(The numbers below were measured for Gemma 4 31B, the one family with a recipe so far.)

| Script | What it does |
|---|---|
| `make_calib.py` | Chat-prefixed calibration text (from `gemma4_q4 calibration-text`) -> `[chunks, 512]` token tensor |
| `make_text_tree.py` | BF16 multimodal checkpoint -> text-only `Gemma4ForCausalLM` tree that transformers 5.14 can load |
| `run_gptq.py` | GPTQ with LLM Compressor, layer by layer, writes the packed projections as it goes; model-specific choices come from `--recipe` |
| `recipes/` | One module per model family: sequential targets, ignore patterns, Hessian leader map, calibration defaults, load check |
| `fast_solve.py`, `fused_gptq_kernel.py` | The fused block solve `run_gptq.py` installs by default (Triton kernel vendored from LLM Compressor 0.14.0), with its start-up self-test |
| `check_checkpoint.py` | Verifies exact INT4 or explicit `--bits 5` native INT5 import (including packing/shape checks and binary16-exact scales) before you convert |

This is a separate environment from the converter's. The converter itself needs only what
[weight-conversion.md](../../docs/weight-conversion.md) lists; only these scripts need `requirements.txt`.

## 0. Recipes

A recipe (`recipes/<family>.py`, registered in `recipes/__init__.py`) holds everything about a family that the runner
does not hard-code:

| Field | Meaning |
|---|---|
| `sequential_targets` | module class names LLM Compressor treats as one sequential layer |
| `ignore` / `saved_ignore` | modules left unquantized; the `ignore` list written to the checkpoint's `quantization_config` |
| `hessian_leaders` | follower -> leader projection names that read one input tensor and so share one Hessian |
| `group_size`, `observer`, `dampening`, `block_size` | defaults, each overridable on the command line |
| `load_check` | three tree keys and the loaded parameters to compare, which catches a tree with wrong key names |

`--recipe gemma4_31b` is the recipe of the shipped Gemma 4 31B checkpoints: `Gemma4TextDecoderLayer`, ignore
`lm_head`, `re:.*embed.*`, `re:.*vision.*`, leaders `k_proj`/`v_proj` -> `q_proj` and `up_proj` -> `gate_proj`, group size
32, `imatrix_mse`, dampening 0.01, block size 128, no activation order. To add a family (26B-A4B, Qwen) write a new
recipe module; the runner needs no change unless the family needs a different module layout for the load check.

## 1. Environment

Needs an NVIDIA GPU with about 12 GiB free (peak torch memory was 10.3 GiB on an RTX 5090), 64 GiB RAM (see the
CPU cap below), and about 58 GiB of disk for the text-only tree plus 16 GiB per checkpoint.

```bash
uv venv --python 3.13 .venv-gptq
uv pip install --python /absolute/path/to/.venv-gptq/bin/python \
  --index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match -r tools/gptq/requirements.txt
```

(On Windows the interpreter is `.venv-gptq\Scripts\python.exe`.) Every version is pinned in `requirements.txt`; the
route was measured with llmcompressor 0.13.0, compressed-tensors 0.18.0, transformers 5.14.1, torch 2.13.0+cu130.
**Always call the environment's interpreter by absolute path.** A relative `.venv/Scripts/python.exe` resolved to a
different environment's interpreter in the shell the route was developed in (the symptom was transformers 5.15.1 with
its `per_layer_config` error instead of the pinned 5.14.1). Check with
`<python> -c "import transformers; print(transformers.__version__, transformers.__file__)"`.

## 2. Calibration tokens

Calibration text must be public text disjoint from whatever you evaluate on, rendered as chat-prefixed chunks exactly
as for the imatrix route (`gemma4_q4 calibration-text`, `docs/weight-conversion.md`); the same 400 x 512 file
serves both. A chat fine-tune sees the chat prefix at run time, so bare text gives worse statistics.

```bash
python tools/gptq/make_calib.py --text calibration-templated.txt \
  --tokenizer /path/to/gemma-4-31B-bf16/tokenizer.json --out work/calib.pt
```

Every chunk must encode to exactly 512 tokens and start with the 17 prefix tokens; anything else is an error.

## 3. Text-only tree

```bash
python tools/gptq/make_text_tree.py --model /path/to/gemma-4-31B-bf16 --out work/text60
```

About 90 seconds and 58 GiB. The checkpoint's `per_layer_config` (written by transformers 5.15) cannot be read by
transformers 5.14.1, so the tree's config carries `global_head_dim` and `num_global_key_value_heads` instead, exactly
what the converter's `normalize_config` produces. Weights are copied bit for bit with `model.language_model.*`
renamed to `model.*`. `--layers N` makes a truncated tree for smoke tests; N must end on a global layer (6, 12, ...),
because transformers rewrites a truncated model's last layer to full attention.

## 4. GPTQ

The default remains **`--bits 4`**, with the recipe's group size. Native INT5 is explicit:
`--bits 5 --group-size 64`; other five-bit groups are refused before model loading. Use the same
text tree and identity-checked calibration tensor for matched four/five-bit comparisons, with
separate fresh checkpoint and offload directories. The fused solver qualifies the requested bit
width at startup; the native compressor's dense cross-word five-bit packing is independently
checked first, and its actual version/layout are recorded in the checkpoint config.

See the [native GPTQ5 conversion instructions](../../docs/weight-conversion.md#native-gptq5-group64-explicit-a16a8-permissions)
for complete commands. `--weights q5_g64_a16` retains native A16 throughout; explicit
`--weights q5_g64` stores the same native codes/scales and permits a separate wide-prefill FP8
route (64 columns and above). Neither enables narrow/decode A8 or changes vocabulary/MTP policy.
The historical timings and quality results below concern INT4, not a measured GPTQ5 result.

```bash
python tools/gptq/run_gptq.py --recipe gemma4_31b --tree work/text60 --calib work/calib.pt --out work/gptq-g32 \
  --group-size 32 --max-cpu-gb 32 --offload-dir work/offload --peak-log work/gptq-g32.peak.jsonl
```

`--observer` defaults to `imatrix_mse` (importance-weighted scale search from the activations seen at each
layer). Expect about 17 minutes on an RTX 5090 for 400 chunks (measured: 1,042 s wall for group 32, 1,012 s for group
256, with the default fused solve below). The same run with the original solver takes 54 to 55 minutes: 2,400 s of
GPTQ solves (the 21504-wide down projection 15 s per layer, the others 4 to 7 s), the rest forward passes. Peak
torch memory 10.3 GiB; host RSS peaked at 39 GiB with the 32 GiB CPU cap. Group sizes 32, 128 and 256 take the same
time and memory (`--group-size`, defaults unchanged) and needed no observer or dampening change: on the author's
model, with `--native-groups` at conversion, group 256 scores KL 0.1192 (A16) / 0.1213 (W4A8) in a 16.37 GB file and
group 128 scores 0.1033 / 0.1081 in 16.59 GB, against 0.1504 / 0.1569 for calibrated RTN at group 32 (17.92 GB);
group 32 GPTQ scores 0.0805 / 0.0845.

### 4.1 Where the time goes, and what makes it fast

LLM Compressor 0.13.0 solves each weight matrix with a Python loop of one step per input column (about 30 tiny CUDA
launches and 2 to 3 host/device syncs each, 0.70 ms per column whatever the matrix shape; Gemma 4 31B has 3.42 million
columns): 2,381 of the 3,316 s of the original run, with the GPU 15% busy. The default now replaces that loop by one
Triton launch per 128-column block (`fast_solve.py`, with the kernel in `fused_gptq_kernel.py`, vendored from
LLM Compressor 0.14.0, PR #3128, Apache-2.0; the installed 0.13.0 package is not touched). Everything else of the solve
(damping, Cholesky chain, the error-feedback GEMM between blocks) is unchanged and the output is **bit-identical** to
the original solver:

| Check (RTX 5090, 31B text tree) | Result |
|---|---|
| Original solver, 6 layers, run twice | bit-deterministic: 0 of 123 tensors differ between runs, and 0 differ from layers 0-5 of the earlier full run |
| Fused solve vs original, 6 layers, group 32 and group 256 | 0 of 123 tensors differ, both groups |
| Full 60 layers, fused solve + shared Hessians, group 32 | 0 of 1,230 tensors differ from the original full run (`gptq-g32-imatrix-mse`) |
| Full 60 layers, group 256 | 0 of 1,230 tensors differ from the original full group-256 run |

Full group-32 run (1,042 s wall, 1,024 s inside LLM Compressor), measured split: calibration forward passes with
Hessian accumulation 473 s, the propagation passes 121 s, solving 117 s (of which the kernel and Cholesky are 69 s;
it was 2,381 s), packing and writing the checkpoint 127 s, observer search, offload write-back and bookkeeping the
remaining 186 s (by subtraction, not measured separately). Forward passes and Hessians are now the largest part (58%).

- `--fused-solve {auto,on,off}` (default `auto`). `auto` runs a self-test at start-up (a small random layer, two
  blocks, solved with the kernel and with the original per-column loop; a bit difference or a Triton compile failure
  disables the fast path), prints one line saying which solver runs, and otherwise keeps the original solver with a
  warning; the run is then about 3 times slower but correct. `on` refuses to start without the kernel, `off` is the
  original solver. Modules the kernel does not cover (anything but symmetric INT4 groups or
  group64 INT5, bf16 scales and a power-of-two block width dividing the columns) fall back per module.
- Needs Triton: `triton-windows` is pinned in `requirements.txt` (on Linux torch brings Triton). The kernel was
  compiled for `sm_120a` by triton-windows 3.8.0.post28 on an RTX 5090. It needs ptxas, which that package bundles.
- `--share-hessians {on,off}` (default `on`). q/k/v read one tensor, and so do gate/up, but LLM Compressor builds a
  Hessian per projection. With sharing, k, v and up take over the Hessian that q and gate just accumulated from the
  very same tensor (a follower that ever sees a different tensor accumulates its own). Measured bit-identical: the q/k/v
  and gate/up Hessians were equal (`torch.equal`) in all 17 follower projections of the 6-layer run before the solve,
  and the output is bit-identical; it saves about 0.8 s of 7.6 s per layer.

### 4.2 Faster variants that were measured and not adopted

These change the order of floating-point work, so they are not bit-identical; GPTQ amplifies such differences
(about 10% of the 4-bit codes and a few thousandths of the scales change for any of them), so each one needs the
quality gate, a full-model paired KL (47 windows, W4A8) against the group-32 baseline of 0.0845. None passed, so none
is in `run_gptq.py`; the code and logs of the experiments are not part of this repository.

| Variant (group 32, 60 layers unless noted) | Time | Codes differing from the default run | Paired KL against the baseline 0.0845 |
|---|---|---|---|
| Default (fused solve + shared Hessians) | 1,042 s wall | 0 (identical to the original solver) | the baseline |
| `batch_size` 8 | 835 s wall | 10.4% (6 layers) | 0.0906, +0.0060 (95% CI -0.0011..+0.0132); lower in 18 of 47 windows |
| `batch_size` 8 and bf16 tensor-core Hessian (`addmm` bf16 to float32) | 669 s wall | 9.5% | 0.0919, +0.0074 (95% CI -0.0018..+0.0165); lower in 16 of 47 windows |
| LLM Compressor `sequential_prefetch=True` (6 layers) | 94 s against 99 s | 14.9% (3.4% of the scales) | not scored: it should only overlap transfers, yet the output changed more than for any other variant, so it was dropped without investigating |

Both scored variants are 7 to 9% worse in the mean with a confidence interval that contains 0 but leans positive;
that is not the "comfortably contains 0" the route needs, so they are off. The bf16 Hessian alone is 3.4 times faster
than the float32 one on a 21504-wide Hessian (3.1 against 10.8 ms per call, microbenchmark), with a relative
difference of 1e-6. If you want the 3 to 6 minutes anyway, test it with a paired KL first.

The remaining time is calibration forward passes with Hessian accumulation (46%), packing and writing (12%) and the
propagation passes (12%): see the split in section 4.1.

Workarounds the script applies, each learned the hard way:

- **`processor=<tokenizer>`**: `oneshot` wants a processor when given a dataset, and the Gemma4 processor imports
  torchvision, which is not installed.
- **Per-layer dump instead of `save_pretrained`**: the model does not fit in 64 GiB of RAM, so weights beyond
  `--max-cpu-gb` are disk-offloaded, and transformers' `save_pretrained` then either dies silently while assembling a
  20 GB shard or raises "could not revert some weight conversions". Either loses the run. `run_gptq.py` instead calls
  compressed-tensors' own `PackedQuantizationCompressor.compress` on each layer right after its solve and writes
  `part-NNNN.safetensors`; on a 6-layer model that is byte-identical to the stock save (123 of 123 tensors).
- **`--max-cpu-gb 32`** keeps 32 GiB of weights in RAM and offloads the rest. `auto_offload` with all available RAM
  left too little for the save step; an extreme cap (4 GiB on a 6-layer model) failed at the last layer with
  "Cannot copy out of meta tensor", so do not cap far below the model size.
- **Load check**: a tree with wrong key names loads "successfully" with random weights. The script compares three
  loaded tensors with the files and refuses to start otherwise.
- **No activation order**: `actorder` is set to `None` explicitly (LLM Compressor's default is `static`); the importer
  refuses `weight_g_idx`.

Without `--max-cpu-gb` all weights stay on the CPU (fine for truncated trees; the full model needs about 60 GiB).

## 5. Check, convert, verify, score

```bash
python tools/gptq/check_checkpoint.py --gptq work/gptq-g32 --tree work/text60

python -m tools.convert.gemma4_q4 convert --model /path/to/gemma-4-31B-bf16 --text-only --no-mtp \
  --weights q4_g32_a16 --gptq-source work/gptq-g32 --out work/gptq-g32-a16.ninfer
python -m tools.convert.gemma4_q4 verify --artifact work/gptq-g32-a16.ninfer
```

Use `--weights q4_g32` for the W4A8 artifact and `--mtp-assistant DIR` instead of `--no-mtp` for MTP, as in
`docs/weight-conversion.md`. For a group-128 or group-256 checkpoint add `--native-groups` to store the real
`q4_g128_fp16`/`q4_g256_fp16` formats (without it the scales are repeated into g32 words: same values, no size win).
`--gptq-source` is mutually exclusive with `--scale-search`, `--imatrix`, `--group-size`, `--region-group` and
`--scale-bits`. The converter refuses a checkpoint it cannot store exactly: zero points, `weight_g_idx`, scales that
are not exactly representable in binary16, asymmetric or wrong-bit schemes. The commands above
are INT4; INT5 requires `check_checkpoint.py --bits 5` and a `q5_g64`/`q5_g64_a16` import recipe.
Score each activation policy separately with the quality tooling (`docs/quality.md`).

Global-layer permutations: the converter permutes global q/k rows and o columns (`prope_perm`). Row permutations never
touch groups. The o permutation maps every 32-group of the Hugging Face order onto a whole 32-group (the rotated blocks
`[0,64)` and `[256,320)` of a 512-wide head are 32-aligned), so group 32 imports every tensor exactly. For group 128
and 256 the permutation interleaves 64-wide runs of two source groups into one converter group, so the ten global-layer
o projections cannot be stored natively; with `--native-groups` they are stored as `q4_g32_fp16` with the repeated
source scales (still exact), and the receipt lists them under `kept_g32_storage`.

For INT5 group64, each global-o permutation run is exactly one 64-wide group: every text
projection stays native `q5_g64_fp16` with its original scale words; there is no g32 fallback.

## 6. Upgrading the GPTQ stack

The converter does not constrain this environment (it needs only `numpy`/`torch`), and nothing in `tools/convert`
imports these scripts. To move off the shipped pin:

- (a) First reproduce the shipped checkpoints on the **old** pin; that is the byte-identity gate.
- (b) Then rerun best-small GPTQ on the new set as a separate qualification: record the fraction of differing codes and
  the paired KL against the old build, and adopt only if KL is not worse within the confidence interval.
- (c) Check whether the new LLM Compressor ships the fused block solver natively; if so, retire `fused_gptq_kernel.py`
  and `fast_solve.py`.
- (d) Check whether the new transformers reads `per_layer_config` natively; if so, drop the config rewrite in
  `make_text_tree.py`.
- (e) Re-verify the `gemma4` MoE linearization of the new LLM Compressor before adding a 26B-A4B recipe.

Receipts record the full version freeze either way. The newest mutually compatible set known at the time of writing
is llmcompressor 0.15.0, transformers 5.21.0, torch 2.14.1, compressed-tensors 0.20.1.
