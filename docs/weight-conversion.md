# Weight conversion

Infernix's converter creates `.infernix` artifacts from local weights and a Python recipe; the
container is the one NInfer named `.ninfer`, and Infernix loads either extension, so the examples
below that write `.ninfer` work unchanged with `--out <name>.infernix`. A recipe
can reuse an official conversion, change selected layers or projections, combine sources, or call
your own conversion method. The artifact contains the resulting configuration, encoded weights,
logical bindings and frontend resources.

Run the commands below from the repository root.

## Upgrade an existing v2 artifact

The offline upgrade tool supports the official Qwen3.6/3.8-27B groupwise-int and NVFP4 artifacts,
and Qwen3.6-35B-A3B groupwise-int. Update your checkout to the current `master` and
[rebuild Infernix](../README.md#quick-start-windows), then run with Python 3.11:

```bash
python3 tools/upgrade_ninfer_v2_to_v3.py \
  models/qwen3_8_27b_nvfp4.ninfer \
  models/qwen3_8_27b_nvfp4.v3.ninfer
```

The output must use a new path. After upgrading, use it directly or rename it to replace the
original file. Stored weight values and formats are preserved. The upgrade also installs the
matching template from `tools/chat_templates/`. Published SHA-256 checksums apply only to
downloaded files.

## Start with an official recipe

Source-weight conversion requires a Python 3.11 environment with PyTorch and NumPy. It uses CUDA
by default; `--device cpu` selects CPU conversion, and `--workers N` converts up to N weight objects
at once on the CPU (written in order; each in-flight object, at most 2N, is held whole in RAM, and
`--max-buffered-bytes`, default 8 GiB, bounds their total; a larger object converts alone). The
input paths below are placeholders for your local checkpoint directories. A Safetensors checkpoint
is read through its `model.safetensors.index.json`; without one, `model.safetensors`, or else every
`*.safetensors` file in the directory (a tensor stored in two of them is rejected).

For Qwen3.6-27B floating-point source weights:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --name qwen3.6-27b \
  --out models/qwen3_6_27b.ninfer
```

`--components` defaults to `text`. Include only the optional components you want to distribute.
`--proposal` adds the indexed proposal head used by speculative decoding. Its rows are the most
frequent tokens of a token ranking, the repository's by default
(`--ranking tools/freq_corpus/fixtures/ranking/ranking.train.counts.i64`), and it keeps 131,072 of
them by default (`--proposal-rows`). The ordinary full-vocabulary output head is retained.

The built-in recipes are ordinary Python functions in
[`official_recipes.py`](../tools/convert/official_recipes.py):

| Recipe | Main representation choices | Additional source |
|---|---|---|
| `qwen3_6_27b` | Q4/Q5 projections, Q6 vocabulary weights | None |
| `qwen3_8_27b` | Q4/Q5 projections, Q8 vocabulary weights | None |
| `qwen3_8_27b_q6` | Q4/Q5 projections, Q6 MLP gate/up, Q8 vocabulary weights | None |
| `qwen3_6_35b_a3b` | Q4 experts, Q5/Q6 expert down, Q8 shared/projection weights | None |
| `qwen3_6_27b_nvfp4` | Imported NVFP4, selected BF16 projections, Q8 vocabulary weights | `quantized` |
| `qwen3_8_27b_nvfp4` | Imported NVFP4/FP8, FP8 embedding generated from BF16 | `quantized` |
| `qwen3_8_27b_nvfp4_nvidia` | NVIDIA ModelOpt layout: imported NVFP4 MLP and FP8 attention/GDN projections, FP8 embedding from BF16, FP8 vocabulary weights re-quantized from the NVFP4 head; a selected DFlash2 drafter stores its MLP gate/up as NVFP4 (`nvfp4_mse`, A16) and the rest as Q8 | `quantized` |
| `qwen3_8_27b_nvfp4_orcarouter` | orcarouter GPTQ layout: imported NVFP4 MLP for layers 0–55, imported FP8 for the last eight layers' MLP and every attention/GDN projection, FP8 embedding and vocabulary weights from BF16 | `quantized` |
| `qwen3_8_flash_next_nvfp4` | Qwen3.8-Flash-Next recipe A: every tensor as NVIDIA stores it (routed experts as exact `nvfp4_mul` banks, the FP8 n-gram table in its own volume); only the MTP drafter re-quantized (Q8 projections, Q4 experts) ([guide](qwen3_8-flash-next.md)) | None |
| `qwen3_8_flash_next_nvfp4_dense8` | Recipe B (Dense8): recipe A plus Q8 for the dense projection classes that dominate per-token weight reads (GDN, QSA output, shared experts, hyper-connection mixers, PLE projections, `lm_head`) | None |
| `qwen3_8_flash_next_fp4_local` | NVIDIA's QAD Qwen3.8-Flash-Next-FP4-Local, bit-exact: `nvfp4_mul` expert banks (NVFP4-CSF scales restored) with their calibrated activation scales, run W4A16 by default and W4A4 with `--expert-a4-activations`, MXFP8 GDN/QSA/shared-expert projections as `fp8_e4m3fn_g32_e8m0`, the NVFP4 vision MLPs and merger as dense `nvfp4_mul` in `nvfp4_rg16_v1`, FP32 norms kept, the NVFP4 n-gram table in its own volume; only the drafter re-quantized ([guide](qwen3_8-flash-next.md#nvidia-fp4-local-checkpoint)) | None |

These names select conversion choices. Runtime execution is selected from the architecture,
configuration and actual bindings stored in the artifact. `--name` sets the public model name;
it does not select kernels.

`qwen3_8_27b_q6` differs from `qwen3_8_27b` in one place: the MLP gate and up projections carry Q6
instead of Q4. Those two parameters are 43% of the text weights, and at 4.25 bits per weight Q4
leaves precision unused there (11.5% relative weight error against 2.6% for Q6 at 6.25 bits per
weight). The vocabulary endpoints deliberately stay Q8, which at 8.5 bits per weight already
outranks Q6.

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_q6 \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b-q6 \
  --out models/qwen3_8_27b_q6.ninfer
```

Measured with `infernix-perplexity` over a 47,917-token corpus at `--context 4096 --stride 2048`
with `--kv-dtype int8`, this artifact scores 1.520765 against 1.524538 for `qwen3_8_27b` and
1.530857 for `qwen3_8_27b_nvfp4`.

The FFN evaluates a Q6 gate/up pair through its existing materialized decomposition (the generic
`linear` op followed by `silu_mul`) rather than the fused `linear_swiglu` op, which has no Q6
variant. That decomposition costs nothing measurable: forcing the fused Q4 route through the same
decomposition moves the corpus perplexity from 1.524538 to 1.524618.

For a Qwen3.8-27B NVFP4/FP8 artifact with DFlash2:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-27B \
  --recipe qwen3_8_27b_nvfp4 \
  --source quantized=/path/to/Qwen3.8-27B-NVFP4 \
  --source dflash2=/path/to/Qwen3.8-27B-DFlash2 \
  --components text,vision,mtp,dflash2 \
  --resource chat_template.jinja=tools/chat_templates/qwen3_8.jinja \
  --proposal \
  --name qwen3.8-27b \
  --out models/qwen3_8_27b_nvfp4.ninfer
```

MTP and Vision use the main source. DFlash and DFlash2 use the corresponding named source, supplied
as `--source dflash=PATH` or `--source dflash2=PATH`. An artifact may contain several optional
components; the Engine loads only the ones selected at startup, including at most one speculative
backend. Component availability and startup selection are independent.

A Qwen3.8-Flash-Next DFlash2 drafter is converted on its own, into a companion artifact that runs
beside any Flash-Next artifact (`--spec dflash2 --draft-model FILE`): `--components dflash2` alone,
with `--model` naming any Flash-Next checkpoint for its config and tokenizer (no weights are read
from it):

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.8-Flash-Next-FP8 \
  --source dflash2=/path/to/Qwen3.8-Flash-Next-DFlash2 \
  --components dflash2 \
  --recipe qwen3_8_flash_next_dflash2 \
  --name Qwen3.8-Flash-Next-DFlash2-Infernix \
  --out models/Qwen3.8-Flash-Next-DFlash2-Infernix.infernix
```

The recipe stores the drafter as NInfer's official Qwen3.8-27B artifact stores its DFlash2:
`q8_g32_fp16` projections (absmax scales; the context K/V projections share the block's K/V words),
BF16 dynamic-convolution kernel projections, selector projection and codebooks, norms and base
kernels. The drafter reuses the main artifact's token embedding and its text or proposal head, so
the companion holds none of them (776 MB). See [container §4.3](maintainer/artifact-container.md#43-companion-artifacts).

## Change part of a recipe

Save the following as `my_recipe.py`:

```python
from tools.convert.official_recipes import qwen3_6_27b


def configure(model, recipe, sources):
    qwen3_6_27b(model, recipe, sources)
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method="grouped_absmax",
    )
```

Run it with the same source and component selection:

```bash
python3 -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe my_recipe.py \
  --components text,vision,mtp \
  --proposal \
  --out models/my_qwen.ninfer
```

The default entry function is `configure`; `--recipe my_recipe.py:customize` selects another
function. Alternatively, use an official `--recipe` and put only the changes in an `--override`
file. Overrides run after the base recipe and optional proposal-head setup.

`model.parameters` maps logical names to their shape, source and mathematical inputs. To see the
available names for your selected components, a recipe can print them:

```python
for name, parameter in model.parameters.items():
    print(name, parameter.shape, parameter.inputs)
```

`recipe.assign` accepts one name, a list of names, or shell-style patterns such as
`text/layers/*/mlp/down`. A selector that matches nothing fails. Assignments run in Python order;
later assignments replace only the explicitly supplied choices. Changing `format` does not
automatically change `method`. Use `layout="auto"` to select the registered layout for the format
when overriding an earlier explicit layout choice.

The principal choices are:

| Argument | Meaning |
|---|---|
| `format` | Persistent numeric format |
| `layout` | Physical encoding; inferred from format unless explicitly set |
| `method` | Built-in method name or Python callable |
| `source` | Logical values or encoded rows from the selected source |
| `parameters` | JSON-serializable numerical parameters passed to the method |
| `rows=(begin, end)` | Override complete leading-axis rows in a half-open range |
| `activation_policy` | Permission for activation precision at the parameter's mathematical inputs |

For example, `rows=(0, 128)` can give the first 128 rows a different format. This creates multiple
physical parts when necessary. The container can represent that result; the intended Op must also
support consuming those parts. Current native projections generally require a contiguous parent
region, so arbitrary splits of one projection are not automatically executable.

## Formats, methods and activation precision

The converter currently writes these formats:

| Format | Built-in method for floating-point input | Import of already encoded input |
|---|---|---|
| `bf16`, `fp32`, `int32` | `cast_direct` | Direct words through the source reader |
| `q4_g64_fp16`, `q5_g64_fp16`, `q6_g64_fp16`, `q8_g32_fp16` | `grouped_absmax` | Supply a custom method/source if needed |
| `fp8_e4m3fn_row_bf16` | `fp8_row_maxabs` | `import_encoded` |
| `fp8_e4m3fn_g32_e8m0` (MXFP8) | None | `import_encoded` (ModelOpt MXFP8 sources) |
| `nvfp4` | `nvfp4_absmax`, `nvfp4_mse` | `import_encoded` |

`grouped_absmax` stores one FP16 scale per group and signed integer codes. `fp8_row_maxabs` first
rounds input values to BF16, then produces E4M3FN codes and one BF16 multiplier per row.
`nvfp4_absmax` maps each 16-value group's largest magnitude to the largest E2M1 code (6) through
an E4M3FN group scale, with one FP32 divisor for the whole parent; `nvfp4_mse` evaluates every
positive finite E4M3FN group scale and keeps the one with the least squared reconstruction error
(ModelOpt's NVFP4 weight-MSE scale sweep), so it is never worse, group by group, than any fixed
max-magnitude target at the same divisor. Both round codes to nearest with ties to even. They quantize weights only: an `AllowA4` use still needs its activation
divisor supplied through `recipe.use`, so for an uncalibrated parent use `A16Only`.
`import_encoded` preserves compatible code and scale words, including NVFP4's matrix weight divisor.
It does not dequantize and requantize them. `cast_direct` rounds exact binary64 source values (an
MXFP8 or weight-only NVFP4 matrix's) to BF16 once, to nearest even.

The exact numeric and packing rules are in [numeric formats](maintainer/tensor-formats.md) and
[storage layouts](maintainer/storage-layouts.md). Source format names alone do not establish
compatibility: scale direction, granularity, code meaning and axis order must also match.

Activation permissions are independent of the stored weight format:

| Policy | Permitted activation paths |
|---|---|
| `A16Only` | A16 |
| `AllowA8` | A16, A8 |
| `AllowA4` | A16, A8, A4 |

They permit choices; they do not force a kernel to use the lowest precision. A fused operation that
shares one activation across several projections must respect the intersection of their
permissions. `recipe.use(parameter, input_name, ...)` can set one mathematical input independently;
the names are available in `parameter.inputs`.

An NVFP4 A4 input requires a positive finite activation divisor. `import_encoded` obtains it from
the selected source, or a recipe supplies it through
`recipe.use(..., auxiliaries={"activation_input_divisor": value})`. Shared weights retain separate
Use records; sharing weights does not share calibration implicitly.

## Fused parents and logical projections

The Qwen adapter exposes Q, K, gate and V separately, even when they came from fused source tensors.
It also supplies finite packing groups for attention, GDN, MLP and MoE. Compatible selections are
packed into a shared parent automatically by the built-in methods.

For the Dense groupwise recipe, attention Q/K form one Q4 parent and gate/V form one Q5 parent.
The native fused Op receives two weights. To use the supported single-parent FP8 form, an override
can assign all four projections together:

```python
def configure(model, recipe, sources):
    for layer, kind in enumerate(model.config["layer_types"]):
        if kind != "full_attention":
            continue
        prefix = f"text/layers/{layer}/attention/"
        recipe.assign(
            [prefix + role for role in ("query", "key", "gate", "value")],
            format="fp8_e4m3fn_row_bf16",
            layout="auto",
            method="fp8_row_maxabs",
            activation_policy="AllowA8",
        )
```

Use this file as `--override` after `qwen3_6_27b` or `qwen3_8_27b`. It leaves the other parameters
under that recipe. The adapter handles source Q/gate row order; the override works with logical
projections. Changing their representation can change numerical results and the physical kernels.

For explicit organization, `recipe.group([names...])` concatenates compatible selections in the
given order, `recipe.separate(names)` disables automatic grouping for those parameters, and
`recipe.share(parameter, target)` binds equal-shaped parameters to the same physical data.
Explicit groups must be disjoint and use unsplit selections with matching format, layout, method
and method parameters. NVFP4 parents also require a common weight divisor. Automatic grouping is
limited to built-in methods; custom methods can request explicit groups.

Grouping chooses storage. Model execution code chooses the supported fused implementation. The
loader uploads the stored representation, without repacking an inconvenient arrangement.

## Read another source

`--model` supplies the main config, default resources and the source named `base`. Add other
Safetensors sources with repeated `--source NAME=PATH`; they are opened when used. Single-file
Safetensors and indexed shards are supported. Additional tensor-only sources can omit model config;
sources carrying config are checked against the relevant model geometry.

To replace a logical parameter from another compatible checkpoint in a recipe:

```python
name = "text/layers/0/mlp/down"
recipe.assign(
    name,
    source=model.source(name, sources["alternate"]),
    format="q6_g64_fp16",
    method="grouped_absmax",
)
```

Supply `--source alternate=/path/to/alternate-checkpoint`. `model.source` applies the architecture's
source-name and axis mapping, including Q/gate extraction. The built-in compressed-tensors reader
understands the implemented per-row FP8 and NVFP4 code/scale conventions. It can expose decoded
values for another quantizer or encoded rows for exact import.

For another file format or quantization convention, provide a `LogicalSource`. Its value reader
accepts flat C-order element bounds and returns exactly that range. For example, a recipe can read
a logical matrix from a NumPy file stored beside the recipe:

```python
from pathlib import Path
import numpy as np
import torch
from tools.convert.sources.logical import LogicalSource


def configure(model, recipe, sources):
    name = "text/layers/0/mlp/down"
    path = Path(__file__).with_name("mlp-down.npy")
    data = np.load(path, mmap_mode="r")
    if tuple(data.shape) != model.parameters[name].shape or not data.flags.c_contiguous:
        raise ValueError("mlp-down.npy must have the logical shape and C-order storage")

    def read_values(begin, end):
        values = data.reshape(-1)[begin:end].astype(np.float32, copy=True)
        return torch.from_numpy(values)

    source = LogicalSource(tuple(data.shape), str(path), read_values)
    recipe.assign(name, source=source, format="q6_g64_fp16", method="grouped_absmax")
```

Use this as an override. The NumPy file must already follow the logical row/column order. For an
unfamiliar quantized source, its reader performs the corresponding decoding before returning
values. To preserve existing compatible encoded words, also provide `read_encoded` returning
`EncodedRows`, and the format's required divisor accessors. Their definitions are in
[`sources/logical.py`](../tools/convert/sources/logical.py).

## Import a GGUF bit-exactly

`--gguf` imports a llama.cpp GGUF's stored blocks as they are, with no dequantization and
re-quantization, so the artifact runs exactly the weights llama.cpp runs from that file. Two
architectures are supported: Qwen3.8-Flash-Next (see
[its guide](qwen3_8-flash-next.md#gguf-import-ista-daslab-gsq-rco-iq3_s)), and Qwen3.5 dense
(`qwen35`) GGUFs in `Q8_0`, text only. For the second, `--model` needs only the checkpoint's
config and tokenizer files:

```bash
python -m tools.convert --model /path/to/Qwen3.8-27B-config-only \
  --gguf /path/to/Qwen3.8-27B-Q8_0.gguf --recipe qwen3_5_gguf_q8_0 \
  --device cpu --out out/qwen3_8_27b_gguf_q8_0.infernix
```

Each Q8_0 block becomes the identical `q8_g32_fp16` word, including the token embedding and
the head. The F32 gating projections and every vector keep their exact values.
[`tools/convert/qwen3_5_gguf.py`](../tools/convert/qwen3_5_gguf.py) undoes llama.cpp's export
transforms and checks each one: the `+1` on the norms, the gated delta-net's tiled value-head
order (rows and whole output-projection column blocks), `-exp(A_log)`, and the squeezed
convolution. The GGUF's metadata must agree with the config. The MTP block and any mmproj are
not imported, and the recipe refuses other ggml formats.

This was checked on bartowski's Qwen3.8-27B `Q8_0` against Qwen's BF16 checkpoint:

- Sampled rows of every matrix role equal llama.cpp's Q8_0 quantization of the checkpoint's
  rows in Hugging Face order.
- `A_log` and `dt_bias` are exact on all 48 delta-net layers.
- The norms, convolutions and gating projections of seven layers are exact, except two gamma
  elements below 2^-16 that llama.cpp's binary32 `1 + g` rounded. The import keeps the GGUF's
  values there.

## Convert from a GGUF quant instead of the full checkpoint

The base `--model` checkpoint only needs to supply `config.json` and the frontend
resources (tokenizer, chat template, generation config); it does not need every
weight tensor present. If you already have a GGUF export of the model -- F16, or a
ggml quant level such as `Q8_0` or `Q4_K_M` -- an override recipe can read the actual
tensor values from that file instead of downloading the multi-gigabyte Safetensors
checkpoint again. Keep only the checkpoint's small config/tokenizer files locally, and
pass the GGUF file as a named `--source`:

```bash
python3 -m tools.convert \
  --model /path/to/checkpoint-config-only \
  --recipe qwen3_6_27b \
  --source quantized=/path/to/model-Q4_K_M.gguf \
  --override use_gguf_source.py \
  --out models/my_qwen.ninfer
```

`tools/convert/sources/gguf.py` reads the GGUF file (via the `gguf` package --
`pip install gguf`; it performs the actual block dequantization, since GGUF's quant
byte layouts are intricate enough that reusing the maintained ggml-org reader beats
reimplementing them) and exposes its tensors as FP32 values. Because GGUF tensor
names never match the HF Safetensors names a recipe's `source_name`s are written
against (`model.layers.0.self_attn.q_proj.weight` vs. `blk.0.attn_q.weight`), and
some architectures reorder tensors on export, `HFAliasSource` wraps a `GGUFSource`
with an explicit `{hf_name: gguf_name}` map (plus optional per-tensor transforms)
so it satisfies the same interface a recipe already expects from `sources["..."]`.

`use_gguf_source.py`:

```python
from tools.convert.sources.gguf import HFAliasSource, standard_dense_name_map


def configure(model, recipe, sources):
    gguf_source = sources["quantized"]  # opened as a GGUFSource because the path ends in .gguf
    name_map, transforms, shapes = standard_dense_name_map(
        model.config["num_hidden_layers"],
        tie_word_embeddings=model.config["tie_word_embeddings"],
        text_prefix="model.language_model.",  # "model." for a text-only (non-vision) checkpoint
    )
    quantized = HFAliasSource(gguf_source, name_map, transforms=transforms, shapes=shapes)
    for name, parameter in model.parameters.items():
        if parameter.source_factory is None:
            continue
        recipe.assign(name, source=model.source(name, quantized))
```

Parameters whose HF source name isn't in `name_map` (mixture-of-experts routing,
MTP, vision, ...) fail with a clear "no GGUF tensor mapped for ..." error rather
than silently reading the wrong bytes; extend `name_map`/`transforms` for those
before assigning them too, or leave them assigned to the base recipe's original
source by only looping over the names you've actually mapped.

### MTP, vision, and DFlash2 from GGUF

`--components text,vision,mtp,dflash2` works with GGUF sources too, matching a
normal multi-component conversion, though each component needs its own wiring:

- **MTP** is usually exported as a *separate* GGUF file (llama.cpp names these
  `mtp-*.gguf`), not fused into the main model's file. It appears as one more
  transformer layer appended after the base model's real layers (layer 64 for a
  64-layer model), reusing the same attention/MLP/norm tensor roles
  `standard_dense_name_map` already covers, plus four MTP-specific tensors.
  `qwen35_mtp_name_map(gguf_layer_index, hf_prefix=...)` builds that map --
  `hf_prefix="mtp."` when the MTP tensors are fused into the main checkpoint's
  state dict, or `hf_prefix=""` for a standalone MTP-only checkpoint whose
  tensors have no prefix. Pass it its own `--source mtp_quantized=mtp-*.gguf`
  and wrap that in its own `HFAliasSource`.

- **Vision** is exported as a separate `mmproj-*.gguf` file (GGUF architecture
  `clip`, not the text model's `qwen35`/`qwen35moe`) covering the ViT tower and
  the projector MLP. `qwen35_vision_name_map(depth, hidden_size=..., patch_size=...,
  temporal_patch_size=..., ...)` builds its name map from the checkpoint's own
  `vision_config`. It also needs a `--source vision_quantized=mmproj-*.gguf`.
  One tensor needs more than a name/value transform: the patch-embed Conv3d
  kernel's temporal dimension is split across two separate GGUF tensors
  (`v.patch_embd.weight` / `.weight.1`), which `HFAliasSource`'s `composites`
  argument reassembles by reading both from the underlying `GGUFSource` and
  stacking them -- see its docstring if you need the same pattern elsewhere.

- **DFlash2** needs no GGUF-specific code at all in the cases seen so far: it's
  a small enough draft head that a plain Safetensors checkpoint for it is the
  normal way to get one, and `tools.convert` already reads that natively via
  `--source dflash2=PATH` plus `--components ...,dflash2` -- the same mechanism
  `qwen3_8_27b_nvfp4`'s example in this doc already uses. An override recipe
  only needs to touch `dflash2/*` parameters if you actually have a GGUF export
  of the draft head to source them from instead.

All three name maps above were checked against a real checkpoint/GGUF pair on
this machine the same way `qwen35_linear_attention_name_map` was (see its
docstring): every tensor's dequantized GGUF value was compared back to the
original Safetensors value. MTP's own norm tensors turned out to need **no**
+1 shift (unlike the base model's layers -- verified, not assumed), vision
needs no shift at all (it's LayerNorm, not RMSNorm), and the patch-embed
temporal split reassembles to the exact original kernel. None of the three
cover mixture-of-experts tensors, and the vision map doesn't cover deepstack
layers (no checkpoint with those to verify against was available).

This keeps every format/method choice the base recipe already made (`qwen3_6_27b`
here); only the byte source for each parameter's values changes. Formats needing
higher source fidelity than the GGUF file provides will simply re-quantize its
already-lossy values -- e.g. re-deriving `q4_g64_fp16` from a `Q4_K_M` source
compounds two rounds of quantization error. Prefer the highest-fidelity GGUF you
have (F16 or `Q8_0`/`Q6_K`) as the source when the target format is more precise
than the GGUF's own quant level.

`standard_dense_name_map` only covers ordinary attention/MLP/norm tensors. Run
`python3 -m tools.convert.sources.gguf /path/to/model.gguf` to list every tensor
name, shape and ggml type in your file, and extend the map for anything it doesn't
cover -- mixture-of-experts routing/shared-expert tensors, MTP/nextn tensors, or
(for Qwen3.5's hybrid linear-attention layers) the GDN tensors handled by
`qwen35_linear_attention_name_map` and `reorder_linear_attention_v_heads` in the
same module. Verify a full conversion's outputs (`tools.artifact.inspect`, then the
normal CLI/serving smoke test) before relying on it -- name-mapping mistakes for a
component this map doesn't cover fail as a missing-tensor error, but a wrong
element ordering for a component it does claim to cover would not.

## Gemma 4 31B: builder presets and the self-contained workflow

`python -m tools.convert.gemma4_build` turns a BF16 Gemma 4 31B dense checkpoint (or a fine-tune of it) into a
Q4, Q5 or IQ3 `.ninfer` with the MTP assistant, then writes the receipts, a Windows `infernix-serve` launcher and
`BUILD-SUMMARY.md`. It builds command lines and records which step finished; conversion is owned by
`python -m tools.convert.gemma4_q4 {convert,verify,calibration-text}`, GPTQ by `tools/gptq/`, and the IQ3
collector/quantizer by an existing llama.cpp build supplied explicitly below. No other orchestration repository
is needed. Source weights are read-only, outputs are never overwritten, and a failed build resumes with the same command.

| preset | quantization method | stored text projections |
|---|---|---|
| `best-small` (recommended q4) | calibrated GPTQ, group 256 | `q4_g256_fp16` plus permutation-safe g32 global output projections |
| `best-medium` | calibrated GPTQ, group 32 | `q4_g32_fp16` |
| `q5-g64` (higher-quality Q5) | own-model calibrated GPTQ, 5 bits, group 64 | `q5_g64_fp16`, A16Only |
| `q4q5-mixed-a` (experimental) | own-model calibrated GPTQ g32 plus GPTQ 5-bit g64 for the Q5 groups | `q4_g32_fp16` with `q5_g64_fp16` (A16Only) attention output and `down` in all 60 layers |
| `iq3-xs` | own-model importance matrix, IQ3_XS allocation | GGUF IQ3_S / IQ3_XXS / Q6_K blocks |
| `iq3-xxs` | own-model importance matrix, IQ3_XXS allocation | GGUF IQ3_XXS / IQ2_S / IQ3_S blocks |
| `draft` | uncalibrated absmax, group 32 | `q4_g32_fp16`; a quick conversion smoke, not a calibrated tier |

See the [canonical four-tier comparison](maintainer/gemma4-benchmarks.md#four-tier-comparison) for measured
text-plus-MTP bytes, model/reference-specific KL, matched thinking-enabled serving and limitations. In particular,
do not compare the plain-model IQ3 KL numbers to a diode-model q4 number or mix FP8-row and Q8_0 references.
The two-level 6-bit `q4_g32_s6_fp16` and calibrated round-to-nearest `lite-*` presets are not built here.

GPTQ runs in its own interpreter, pinned by `tools/gptq/requirements.txt`, and never in the converter's: create
it once, then point the builder at it with `--gptq-python` or `INFERNIX_GPTQ_PYTHON` (default
`.venv-gptq` in the checkout).

```text
python -m tools.convert.gemma4_build setup-gptq --venv .venv-gptq
python -m tools.convert.gemma4_build --preset best-small --model <BF16 dir> --assistant <assistant dir> --out-dir <dir> --check
python -m tools.convert.gemma4_build --preset best-small --model <BF16 dir> --assistant <assistant dir> --out-dir <dir> --dry-run
python -m tools.convert.gemma4_build --preset best-small --model <BF16 dir> --assistant <assistant dir> --out-dir <dir> --yes
```

### Native GPTQ5 group64, explicit A16/A8 permissions

The canonical one-call preset is `q5-g64`: the same calibration/render/tokenize/text-tree/GPTQ/check/import
workflow as best-medium, with explicit five-bit/group64 quantization and `q5_g64_a16` import. It collects
Hessians on **your actual model**; plain-model Hessians must never be reused for a diode variant.
It writes one A16Only artifact with FP8 vocabulary and the BF16-source MTP assistant (assistant layer
projections encoded FP8, outer projections BF16). It does not replace or rename best-medium.

```text
python -m tools.convert.gemma4_build --preset q5-g64 --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/q5-g64" --work-dir "<camp>/convert/q5-g64" --serve "<absolute-infernix-serve-path>" --yes
```

Use `--dry-run` for exact commands and input/RAM/disk checks. Budget roughly 60 GB for the owned BF16
text stage plus the checkpoint/artifact (85 GB minimum scratch, 115 GB comfortable); these estimates
exclude the preserved original source. `--keep-work` retains logs/checkpoint/stage for qualification.

Artifact selection and compute selection are independent.
**Q5 profiles default to `int8-g64`**, the measured speed/fidelity balance; explicit
`--gemma4-q5-prefill a16|fp8|int8-g32|int8-g64` always wins. `fp8` is faster but less faithful;
`a16` preserves artifact policy (A16 on the builder's A16Only artifact). Narrow trunk decode/MTP
verification stays A16. Do not rebuild or duplicate weights to change compute.

Omitting `--profile` selects `q5-quality` for `--preset q5-g64` and for `launcher` when the artifact
directory contains native Q5 weights; the filename/HF model name is not used. Explicit `--profile default`
retains the historical general settings, including `a16`, 131072 context and automatic chunk.
Q4/IQ3 and `memory-16gb` defaults do not change. Direct Python callers use
`ServeSettings(profile="q5-quality")` or `ServeProfile(profile="q5-long-context")`; unspecified fields
resolve in the same place as the CLI. `write_launcher(..., profile=None)` detects the artifact.

| Serving profile | Prefill compute | Sliding ring | Context / chunk | Global KV |
|---|---|---|---|---|
| `q5-quality` (Q5 default) | INT8 g64 | BF16 | auto / 2048 | normal stored FP8 K/V |
| `q5-long-context` | INT8 g64 | FP8 | auto / 2048 | normal stored FP8 K/V |

Both retain 1024 MiB headroom, thinking on, top-k 64, MTP d1, BF16 logits and an 8192 MiB host-only
prefix cache. **“Quality” means the higher-quality Q5 tier with BF16 rings, not maximum-fidelity
arithmetic:** FP8 global QK remains the existing roleplay/prose choice, faster on long prompts but
measurably less faithful. For coding/tools or fidelity-first use, explicitly add `--gemma4-qk bf16`;
also add `--gemma4-q5-prefill a16` to avoid the lossy projection choice. No coding/tool configuration
is silently migrated. Explicit BF16 ring, QK, compute, sampling and capacity overrides win.
Neither Q5 profile implicitly enables key-only KV. Auto context fits actual available resident
VRAM at startup (minimum 9216), not a promise of 131k; read startup and `/v1/models`.

#### Lower-VRAM Q5 ladder (same artifact)

Generate a **new** CMD for the choice you need, then run that CMD. These are alternatives, not
automatically enabled fallbacks; combine options only deliberately. `<Q5.ninfer>` is the one
verified A16Only file and `<serve>` is the absolute `infernix-serve` executable path.

| Choice | Cost / limit |
|---|---|
| `q5-quality` | Recommended balance; BF16 ring, chunk 2048. |
| `q5-long-context` | FP8 ring saves about 1191 MiB at chunk 2048; measured short-protocol KL increases and prefill slows up to 12.5% in the ring study. Not long-context quality qualification. |
| Chunk 1536 | Measured BF16-ring reservation saving **884.129 MiB = 478.875 workspace + 400 ring + 5.254 other sequence reservation** versus 2048. Not all KV and not measured extra token capacity; g64's +0.64% long-prompt median is inconclusive, not a speed gain. FP8-ring combinations do not inherit that exact saving. |
| `--no-mtp` | Leaves the embedded assistant unloaded; forfeits speculative decode speed for more room. |
| Shorter `--context 16384` | Limits conversation capacity; explicit context must fit unchanged. Auto already chooses what fits. |
| Explicit `--gemma4-key-only-kv` | Experimental extra global capacity; changes attention arithmetic and has measured fidelity/long-context speed costs. Never implicit. |
| `--allow-shared-memory` | Explicit **last resort**, system RAM over PCIe, potentially much slower. Q5 does not automatically try no-MTP first: choose it explicitly. Not resident operation. |

```text
python -m tools.convert.gemma4_build launcher --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5.cmd"
python -m tools.convert.gemma4_build launcher --profile q5-long-context --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-fp8-ring.cmd"
python -m tools.convert.gemma4_build launcher --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-1536.cmd" --prefill-chunk 1536
python -m tools.convert.gemma4_build launcher --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-no-mtp.cmd" --no-mtp
python -m tools.convert.gemma4_build launcher --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-16k.cmd" --context 16384
python -m tools.convert.gemma4_build launcher --profile q5-long-context --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-key-only.cmd" --gemma4-key-only-kv
python -m tools.convert.gemma4_build launcher --profile q5-long-context --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-shared.cmd" --prefill-chunk 1536 --no-mtp --context 16384 --allow-shared-memory
python -m tools.convert.gemma4_build launcher --profile q5-long-context --artifact "<Q5.ninfer>" --serve "<serve>" --out "<camp>/serve-q5-fidelity.cmd" --gemma4-sliding-kv bf16 --gemma4-qk bf16 --gemma4-q5-prefill a16
```

Q5's roughly **21.15 GB native file** (about 20.63 GB device weights without MTP in the measured plain
artifact) cannot fit resident on a **16 GB card regardless of KV changes**. For resident 16 GB serving
use the smaller IQ3_XXS `memory-16gb` tier below, accepting its quality tradeoff. The ladder is useful
when Q5 weights fit but context/workspace do not; it is not a 16 GB Q5 fit claim.

The historical plain-model short block-prefill KL is A16 0.0199 / INT8-g64 0.0203 / FP8 0.0296.
The separate 276-position incremental sample is g64 0.0104 / A16 0.00995; overlapping-window
uncertainty is not equivalence. G32 has a significant measured incremental delta-NLL increase.
Long-context INT8 quality is unmeasured. Matched diode own-Q8 short block-prefill KL is 0.0331
A16 / 0.0336 g64 versus 0.0941 best-medium; that does not qualify diode incremental or long-context
quality. See [the measured protocols and limitations](maintainer/gemma4-q5-prefill.md).

The lower-level explicit conversion alternatives remain available: `q5_g64_a16` keeps A16Only Uses,
while `q5_g64` permits the historical wide A8 policy. Every target text projection uses
`q5_g64_fp16`. These are storage permissions, not a second serving-compute selector.

Start with a complete unquantized Gemma 4 **31B dense** BF16 HF checkpoint, its tokenizer and the
converter environment (Python 3.11 with torch/numpy/safetensors). GPTQ uses the separate pinned
environment in `tools/gptq/requirements.txt` (Python 3.13; compressed-tensors **0.18.0**, LLM
Compressor 0.13.0), not the converter interpreter. Supply absolute interpreter paths below.
Use fresh output/scratch directories outside the read-only HF source, with room for the text-only
BF16 tree, disk-offloaded weights, packed checkpoint and final artifact.

Render disjoint public calibration text with the matching tokenizer/template using
`gemma4_q4 calibration-text` first, or reuse an existing token tensor only after establishing its
source/tokenizer/chat-prefix identity. Matching `400 x 512` dimensions alone do not establish this.
Create a full text-only tree (omitting `--layers` keeps all 60 layers); the runner does not consume
the multimodal tree directly:

```text
"<converter-python-3.11>" tools/gptq/make_text_tree.py --model "<BF16-dir>" --out "<work>/text60"
"<gptq-python>" tools/gptq/make_calib.py --text "<matching-chat-prefixed-calibration.txt>" --tokenizer "<BF16-dir>/tokenizer.json" --out "<work>/calib.pt"
"<gptq-python>" tools/gptq/run_gptq.py --recipe gemma4_31b --tree "<work>/text60" --calib "<work>/calib.pt" --bits 5 --group-size 64 --max-cpu-gb 32 --offload-dir "<work>/offload5" --out "<work>/gptq5-g64"
"<converter-python-3.11>" tools/gptq/check_checkpoint.py --bits 5 --gptq "<work>/gptq5-g64" --tree "<work>/text60"
"<converter-python-3.11>" -m tools.convert.gemma4_q4 convert --model "<BF16-dir>" --text-only --no-mtp --weights q5_g64_a16 --gptq-source "<work>/gptq5-g64" --out "<out>/gemma4-gptq5-g64-a16.ninfer"
"<converter-python-3.11>" -m tools.convert.gemma4_q4 verify --artifact "<out>/gemma4-gptq5-g64-a16.ninfer"
```

To select the wide-prefill FP8 route, keep that **same A16Only artifact** and generate a launcher:

```text
"<converter-python-3.11>" -m tools.convert.gemma4_build launcher --artifact "<out>/gemma4-gptq5-g64-a16.ninfer" --serve "<absolute-infernix-serve-path>" --gemma4-q5-prefill fp8 --out "<out>/serve-gemma4-gptq5-fp8.cmd"
```

The lower-level `q5_g64` import remains a storage-permission alternative, not a required duplicate:

| Explicit weights recipe | Target projection Uses | Execution permission |
|---|---|---|
| `q5_g64_a16` | `A16Only` | Native Q5 with 16-bit activations at every width |
| `q5_g64` | `AllowA8` | Wide prefill (64 columns and above) may bake Q5 to FP8-row for FP8 GEMM; narrower calls stay native Q5 A16 |

Vocabulary, assistant and other non-target-projection Uses do not acquire `AllowA8`. Native server
defaults preserve stored policy; the Q5 launcher profile explicitly selects INT8 g64 unless overridden.
There is no Q5 narrow/decode A8 route.
The wide A8 composition belongs to Gemma's projection dispatcher (`project()`), not a new
generic Linear Q5-A8 kernel: generic `ops::linear` on Q5 remains A16 even with `AllowA8`.

The Q5 wide-prefill bake uses **actual decoded values**, not the Q4 scale-order bound.
For each stored signed five-bit code `q` and binary16 scale `d`, decode
`v = FP32_RN(q * d)` first (this finite product is exact), with no intermediate BF16
weight cast. Set `m = max(abs(v))` over the logical row, then the row multiplier
`s = BF16_round_toward_positive_infinity(FP32_RN(m / 448))`; a zero row uses `s = 1`.
Compute `r = FP32_RN(1 / FP32(s))` and encode
`E4M3FN_RN_ties_even_satfinite(FP32_RN(v * r))`. The represented baked weight is that
E4M3FN value times `FP32(s)`. In particular, this is **not** Q4's
`q * FP32_RN(d * r)` ordering, nor a BF16-rounded weight maximum. The FP8 activation
encoder/GEMM is then reused for allowed wide calls. This is a distinct approximate
execution path, not a hidden precision mode or a guarantee of native-A16 numerical identity.
Its quality/performance require separate measurements; the A16 artifact retains its own path.

The converter requires a clean tracked checkout for its revision receipt. Replace `--no-mtp` with
`--mtp-assistant "<assistant-BF16-dir>"` to include the separate assistant: its layer projections
default to FP8, its outer projections stay BF16, and its head stays FP8 as before. GPTQ supplies
**only target text projections**; the tied vocabulary is still FP8-row from `--model`, and norms,
frontend resources and optional components retain their existing explicit HF/BF16-source policy.

The [pinned compressor](https://github.com/vllm-project/compressed-tensors/blob/0.18.0/src/compressed_tensors/compressors/pack_quantized/helpers.py)
supports native five-bit compression with **dense cross-word offset-binary packing**: code `j`
starts at bit `5*j`, words have shape `[N, ceil(K*5/32)]`, and codes may straddle int32 words.
This is **not** six codes per word (`floor(32/5)`). The runner proves its installed compressor's
layout against an independent fixed bitstream before loading the model and refuses unsupported
compression instead of inventing a repacker. Its config records the actual compressor version,
bit width and packing; conversion receipts retain those facts. Other packing shapes, zero points,
activation-ordered groups, non-group64 INT5 and mismatched bit widths are refused.

Import copies every signed code and binary16-exact scale without dequantization/requantization
or scale rerounding. Global q/k reorder complete rows. Global o's rotary permutation moves
complete **64-wide** groups, so it too stays native `q5_g64_fp16`; a split group is an error,
not a fallback format. Scales not exactly representable in binary16 are refused.

Both `q5_g64` and `q5_g64_a16` fix native group64: do not add Q4 `--group-size`,
`--region-group`, `--native-groups` or scale-search flags. Without `--gptq-source` either
explicit recipe uses uncalibrated group64 RTN from HF BF16; that is a separate quality arm, not GPTQ.
Existing Q4 conversion and `run_gptq.py` defaults remain four-bit/group32. For a matched GPTQ4
baseline rerun with `--bits 4 --group-size 32`, the **same** text tree/calibration tensor and
separate fresh output/offload directories; import with `--weights q4_g32_a16` for A16 or the
unchanged `--weights q4_g32` for its separate W4A8 performance arm.

#### Mixed Q4 / Q5 artifacts (per fused projection group and layer)

One artifact may hold native `q4_g32_fp16` and `q5_g64_fp16` projections side by side. The unit is a fused
projection group of one layer, because q/k(/v) and gate/up are stored as one matrix and so share a format:
`qkv` (sliding q,k,v; global q,k), `o`, `gate_up` and `down`, across all 60 layers. Start from an INT4 g32 import and
name the groups that come from a matching INT5 group64 checkpoint of the same model (same text tree and calibration):

```text
"<converter-python-3.11>" -m tools.convert.gemma4_q4 convert --model "<BF16-dir>" --text-only --mtp-assistant "<assistant-BF16-dir>" --weights q4_g32_a16 --gptq-source "<work>/gptq4-g32" --q5-gptq-source "<work>/gptq5-g64" --q5-unit down --q5-unit qkv@0-19 --out "<out>/gemma4-mixed.ninfer"
```

`--q5-unit ROLE[@LAYERS]` repeats; `LAYERS` is a comma list of layer numbers or inclusive ranges (default all
layers) and overlapping selectors are a union. Every named group is stored `q5_g64_fp16` with an `A16Only` Use; every
other projection follows the INT4 import unchanged, and `--weights q4_g32` still gives those Q4 projections
`AllowA8` (the Q5 groups stay `A16Only`; `--gemma4-q5-prefill` selects their prefill compute at launch, and applies
to the Q5 groups only). Each group costs `0.75` bit per weight over Q4 (4.5 vs 5.25 bit including scales), so
all four roles over all layers reproduce the full Q5 artifact's size. The flags need `--weights q4_g32|q4_g32_a16`
with `--gptq-source`, refuse `--native-groups`, unknown roles, malformed or out-of-range layers and a source that
is not INT5 g64, and the build receipt lists the cells (`scale_search.q5_source`) and the per-format tensor counts.
The loader, the per-layer workspace arena and the route selection work per prepared projection, so no engine
option is involved.

##### Builder preset `q4q5-mixed-a` (experimental)

`--preset q4q5-mixed-a` is best-medium with Q5 `o` and `down` in every layer, built in one call: the shared
calibration/render/tokenize/text-tree steps, a group-32 GPTQ pass (`gptq`, `gptq-check`), a 5-bit group-64 pass
(`gptq-q5`, `gptq-q5-check`, checkpoint `gptq-q5`) from the same text tree and calibration tensor, then
`gemma4_q4 convert --weights q4_g32 --gptq-source <gptq> --q5-gptq-source <gptq-q5> --q5-unit o --q5-unit down`. Both
checkpoints therefore use Hessians of **your actual model**. The artifact is the best-medium file plus about 0.94 GB
(0.9393 GB measured on the plain model; 19.34 GB with MTP): Q4 projections keep `AllowA8`, the Q5 projections are `A16Only`.

**Experimental; measured on a diode variant.** The allocation was selected on the plain model (its own Q8_0
reference, plain-model Hessians; prose KL against best-medium on that model fell from 0.0684 to 0.0485 on the default
route, FULL KL 0.0458 against 0.0687 for best-medium and 0.0203 for full Q5). It was then built from the deployed
diode's own Hessians (19,344,788,484 B, byte-identical size) and scored against the diode's own Q8_0 reference on
development windows (block-prefill KL at context 512, one GPTQ pass per width). Roleplay-forum text (42 threads, 84
windows): KL 0.0340 against 0.0446 for best-medium on A16 compute, ratio 0.762 (95 % cluster interval 0.744-0.781; plain
model 0.724), and 0.0383 against 0.0513 on the default route (ratio 0.747, interval 0.723-0.770); PG-19 prose ratio
0.742 on A16 compute, 0.749 on the default route. Speed against the diode best-medium (default route, HTTP, chunk 2048,
FP8 KV/QK): decode 64.5 against 72.6 tok/s at 1k, prefill 3,263 against 3,424 tok/s at 65k, capacity 131,072 tokens with
MTP off and 122,432 with MTP d1. Nothing is promoted and the hint column is still the plain-model figure: the build
summary, `--list-presets` and the dry run keep the experimental mark. Measure your own build before relying on it.
Evidence: llama-local-tooling `docs/plans/20261123-gemma4-diode-mixed-a.md`.

**Fresh held-out roleplay text.** On the reproducible [held-out roleplay corpus](roleplay-heldout.md) (36 forum threads, 72
windows, disjoint from the development windows and every calibration text) the same diode artifact on the default route
scored KL 0.0405 against 0.0547 for the promoted best-medium: ratio 0.740 (95 % thread-cluster interval 0.722-0.759), below
best-medium in all 36 threads; the A16-compute plain-model rebuild scored 0.0272 against 0.0381 for plain Q4, ratio 0.713
(0.697-0.727). Same limits as above: block-prefill KL at context 512 against each model's own Q8_0, and the finetune may
have seen these forums. Evidence: llama-local-tooling `docs/plans/20261125-gemma4-fresh-heldout.md`.

The serve settings are the unchanged `default` profile (`--profile` omitted), and no Q5 profile is selected by this
preset. The Q5 cells' prefill compute is a launch choice on the same artifact; the measured route lets the Q4 cells
keep their A8 paths and runs Q5 prefill as INT8 g64:

```text
python -m tools.convert.gemma4_build --preset q4q5-mixed-a --model "<BF16-dir>" --assistant "<assistant-dir>" --gptq-python "<gptq-python>" --out-dir "<camp>/q4q5-mixed-a" --gemma4-q5-prefill int8-g64 --serve "<absolute-infernix-serve-path>" --yes
```

`--gemma4-q5-prefill` applies to the Q5 groups only (narrow decode stays A16 there). `launcher --artifact` without
`--profile` selects `q5-quality` for any artifact that holds native Q5 weights, this one included; pass
`--profile default` to keep the general settings. The builder needs a server that knows both `q4_g32_fp16` and
`q5_g64_fp16`, a second GPTQ pass (about 25 more minutes) and about 102 GB minimum scratch (132 GB comfortable) on the work
drive. `--score` measures the Q4 cells on their exact W4A8 route and the Q5 cells on A16, so it is not the default-route
figure above.

### Create the smaller IQ3 files from HF BF16

Start with the complete local BF16 Gemma 4 **31B dense** HF tree (including its tokenizer/config files), a clean
infernix Git checkout, and a Python environment with the converter dependencies. The 26B-A4B architecture is
not supported by this builder. Source files stay read-only; use output and scratch directories outside that tree.

IQ3 additionally needs an existing llama.cpp build with GPU-enabled `llama-imatrix` and `llama-quantize`, and
its source checkout containing `convert_hf_to_gguf.py`. Pass the **binary directory** as `--llama-cpp` and
the **source directory** as `--llama-cpp-source`, or set `INFERNIX_LLAMA_CPP` and `INFERNIX_LLAMA_CPP_SOURCE`.
The builder's Python needs `gguf`, `transformers`, `sentencepiece`, `tokenizers` and the ordinary
`torch`/`numpy`/`safetensors` converter dependencies; default corpus export also needs `pyarrow`.
The IQ3 route does not use or install the separate GPTQ environment. Conversion/collection needs CUDA;
serving targets the supported Blackwell GPU build. Use `--check` to see the actual input/tool/GPU/RAM/disk checks.

MTP is a **separate source**, not reconstructed from a trunk GGUF: supply the local
`google/gemma-4-31B-it-assistant` BF16 checkpoint with `--assistant`. Its projections are encoded FP8 by default.
To deliberately omit MTP, replace the whole `--assistant "<assistant-dir>"` pair with `--no-mtp`.
Both IQ3 presets are text-only by default; `--with-vision` is explicit and is outside the measurements below.

Run from this checkout; the two builds use different output/work directories:

```text
python -m tools.convert.gemma4_build --preset iq3-xs --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xs" --check
python -m tools.convert.gemma4_build --preset iq3-xs --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xs" --yes
python -m tools.convert.gemma4_build --preset iq3-xxs --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xxs" --yes
```

Here `<name>` defaults to the sanitized `<HF-directory-name>-<preset>`; `--name NAME` sets it explicitly.
The artifact is `<out-dir>/<name>.ninfer`, and the build prints its actual name and launcher path.

Replace `--yes` with `--dry-run` to preview exact commands without writing. `--work-dir "<scratch-dir>"`
selects another scratch drive. Each preset renders **400 × 512-token** chat-prefixed calibration chunks,
exports BF16 GGUF, quantizes a Q8_0 collection copy, runs `llama-imatrix` on **that model**, and quantizes
with the measured Gemma allocation pins. No publisher/base-model importance matrix is substituted.
The projections are then imported verbatim, while vocabulary/norms/scalars come from HF BF16 and MTP from
the explicit assistant. The build verifies storage and writes the artifact, two conversion receipts,
`BUILD-SUMMARY.md`, the builder receipt and `serve-<name>.cmd`.

**Corpus and reuse.** Unless `--calibration-text FILE` supplies disjoint UTF-8 training/calibration text,
the builder downloads the pinned English `eaddario/imatrix-calibration` `combined_en_large.parquet`
revision `e87ed55dcba9d9c3a3e41539f3e728e981b1daa4`, verifies its hash, exports the pinned text bytes, then
renders the model's chat prefix. Do not calibrate on your evaluation corpus or use another model's imatrix.
The generated `imatrix.gguf`, rendered corpus and step markers are local to that build's work directory.
Rerunning the **same arguments** resumes completed stages, including collection; it is not a shared
cross-preset imatrix cache. A different preset/source/corpus cannot borrow that directory's done markers.
`--keep-work` retains the matrix, corpus, IQ3 GGUF and logs after success, but does not disable normal
BF16/Q8 scratch pruning. Without it, the builder removes its work directory after a successful build.
There is no builder `--imatrix`/`--gguf-source` override; an existing own-model GGUF can instead use the
direct import workflow below without repeating BF16 export, collection or quantization.

**Time and storage.** The printed IQ3 plan estimates roughly 100 minutes, advertised as about two hours
on RTX 5090 + NVMe: about 15 min BF16 export, 8 min Q8 copy, 25 min collection, 45 min CPU quantization and
6 min import, plus small rendering/receipt steps. These are estimates, not guarantees or timings of every
stage on every model. Small output does not mean cheap construction: a 31B BF16 intermediate is about
62 GB and Q8 about 33 GB, versus final files around 13–14 GB including MTP. Preflight requires at least
100 GB free scratch (120 GB comfortable), in addition to the original HF source; a separate output drive
needs the estimated artifact size plus 2 GB. A shared output/work drive uses the peak-stage calculation,
not the sum of all sequential intermediates. BF16 and Q8 copies are deleted after their last use.
`--score` adds about 62 GB scratch and a long FP8-row reference build; its reference is not automatically
the Q8_0 instrument used in the canonical comparison.

After building, verify the file and the generated launcher separately:

```text
python -m tools.convert.gemma4_q4 verify --artifact "<camp>/iq3-xs/<name>.ninfer"
python -m tools.convert.gemma4_build serve-check --launcher "<camp>/iq3-xs/serve-<name>.cmd"
```

Repeat for IQ3_XXS. A storage receipt does not establish runtime behavior or model quality; `serve-check`
is an HTTP smoke, not a long-context/VRAM/quality certificate. Check the real startup KV capacity and chunk,
especially on smaller cards. Measured file sizes and matched throughput are in the canonical comparison.


The launcher runs `infernix-serve` with the operator roleplay profile: `--max-context 131072` (alias `131k`),
`--kv-dtype fp8`, `--gemma4-qk fp8`, `--prefill-chunk auto`, one lane, `--spec mtp --draft-tokens 1` when the
file carries the assistant, `--logits-dtype bf16`, a host-only prefix cache (`--device-snapshot-slots 0
--host-context-mib 8192`), `--any-model-name`, and `0.0.0.0:8001`. The server binary is
`apps/Release/infernix-serve.exe` (or a `build*/apps/Release` of the checkout) unless `--serve` names another.
Every setting has a flag; `--gemma4-qk bf16` omits the QK flag.
Sliding rings are BF16 for the general profile, `q5-quality` and native commands; `memory-16gb` and `q5-long-context`
default to the FP8 ring, and an explicit `--gemma4-sliding-kv bf16` overrides it. FP8 requires `--kv-dtype fp8`.
The FP8 ring saves 446 MiB at chunk 128 or 1,191 MiB at chunk 2048, but short-context KL increases and prefill slows:
see the [FP8-ring measurements and limits](maintainer/gemma4-benchmarks.md#fp8-sliding-ring-experimental-off-by-default).
Planning/fallback commands preserve the selected ring, and `serve-check` records the effective selection.
`--gemma4-q5-prefill a16|fp8|int8-g32|int8-g64` defaults to `int8-g64` for Q5 profiles and `a16`
for others. Non-`a16` selections are forwarded to every serve command; `serve-check` records them.
`fp8` enables the existing wide Q5 row-E4M3 bake on the same A16Only artifact; narrow decode/MTP remains A16.

```text
python -m tools.convert.gemma4_build launcher --artifact <file.ninfer> --out <serve.cmd>
python -m tools.convert.gemma4_build serve-check --launcher <serve.cmd>
```

`serve-check` starts the server the launcher runs, checks `/v1/models` and chat requests with the launcher's
model id, another model name (200 with `--any-model-name`, 404 without), no model field (400) and an arbitrary
bearer key, stops the server and writes a JSON receipt with the launcher's effective profile.

### Opt-in 16 GB memory profile

`--profile memory-16gb` extends the same builder and launcher machinery; it is not a new converter.
For a build it selects `iq3-xxs` when `--preset` is omitted and refuses any other recipe or vision.
The normal profile and all four artifact tiers are unchanged.

```text
python -m tools.convert.gemma4_build --profile memory-16gb --model "<BF16-dir>" --assistant "<assistant-dir>" --llama-cpp "<llama-bin-dir>" --llama-cpp-source "<llama-source-dir>" --out-dir "<camp>/iq3-xxs" --yes
python -m tools.convert.gemma4_build launcher --profile memory-16gb --artifact "<IQ3_XXS.ninfer>" --serve "<absolute-infernix-serve-path>" --out "<camp>/serve-16gb.cmd"
```

Run the resulting `serve-16gb.cmd`. It plans with the MTP assistant first (`--plan`, no GPU work) and starts without it
when only that fits (`echo` notice on stderr; the artifact keeps the assistant). `launcher ... --no-mtp` always omits it, which
buys context: the explicit ladder fits 24,576 tokens without it against 20,480 (the highest tried) with it at 14,767 MiB usable and zero headroom. Thinking stays enabled.
The profile emits `--max-context auto --kv-capacity auto --gemma4-key-only-kv --kv-dtype fp8 --gemma4-qk fp8 --gemma4-sliding-kv fp8 --prefill-chunk 128`,
BF16 logits, one lane, a host-only 8192 MiB prefix cache, 64 MiB VRAM headroom (`--vram-headroom-mib`, any value from 0) and one
MTP draft when enabled. The default alias is `gemma4-16gb`; host/port retain `0.0.0.0:8001` (use `--host 127.0.0.1` for local-only).
No environment variable enables key-only KV.

The Engine, not the launcher generator, fits context at each launch from the actual free resident VRAM (NVML free less the WDDM
budget gap latched at the first reading) and its complete weights/KV/workspace/graphs plan, checks again after upload and after
startup, and reports the resolved limit in `/v1/models` and startup logs. It never assumes that every 16 GB desktop can hold 65k, and it adapts
to whatever other programs hold at launch (a desktop idling at 1.9-2.7 GiB, a video app already open). Rule: automatic context is page-aligned,
capped at 131072/native positions, the largest context whose reservation, plus 4 KiB per token and 160 MiB for what the plan does
not name, fits beside `--vram-headroom-mib` free; below **9216 tokens** (about 1k of prompt plus an 8192-token output cap) it refuses and
states the shortfall in MiB, never paging, shrinking an explicit `--context` or starting with less than the headroom after startup.
Auto requires `--kv-capacity auto` and a physical-free reading; Gemma refuses `--vram-past-budget`. When a refusal is short by a few
tens of MiB lower `--vram-headroom-mib` (0 gives the 1.9 GiB desktop 16,192 tokens instead of a refusal), close programs holding VRAM,
or use `--no-mtp`.

Historical **BF16-ring** measurements on an RTX 5090 with the ballast method of the [16 GB envelopes](maintainer/gemma4-benchmarks.md#16-gb-profile-qualification)
(not a physical 16 GB card; rates are the 5090's): headless 15,791 MiB usable resolves **58,752 tokens with MTP**; a 14,767 MiB desktop (1.5 GiB
reserved) **11,392-13,248 tokens with MTP** (115-121 tok/s) or, [INFERENCE] from the explicit ladder, about 20k without; a ~1.9 GiB-idle desktop (14,357 MiB) **13,184 tokens without
MTP**, 16,192 with `--vram-headroom-mib 0`; a 2.5 GiB-reserve desktop (13,743 MiB) is **refused**, 9,216 tokens being short by about 253 MiB with
no headroom (512 MiB at the default). Context is bytes-limited, so the same card reads differently with a video app open: read `/v1/models` after startup.
Other programs opening after launch did not slow or page decode in the one test (80-83 tok/s with up to 2.3 GiB more taken), because
Windows demotes idle allocations first; the headroom setting protects the next startup, not a later spike, and no profile guarantees
complete answers or the absence of loops.

**Opt-in: `--allow-shared-memory`.** Users who must run below the refusal line can generate the launcher with
`launcher --profile memory-16gb --allow-shared-memory` (or add the flag to `infernix-serve`): the CMD still tries resident starts first
(plan with MTP, plan without it) and only when neither fits starts without MTP with the flag, printing a `WARNING` and a server `SHARED GPU
MEMORY ALLOWED` line with the estimated bytes in shared memory. Every refusal of the profile names the option. Measured cost on the 2.5 GiB-reserve
desktop (13,743 MiB usable): none up to 24,576 tokens (72 tok/s), because the shortfall is 0.2-0.5 GiB; beyond about 0.8 GiB short decode
falls 58-75 % and prefill 57-68 % ([measurements](maintainer/gemma4-benchmarks.md#shared-gpu-memory---allow-shared-memory)). Default behavior
without the flag is unchanged: refusal, nothing paged.

Profile overrides permit chunks `128`, `256`, `512` (larger chunks trade context for prompt speed: 1,081 / 1,381 MiB of ring plus KV at
chunk 128 / 512), any non-negative headroom, and one MTP draft only. FP8 KV and BF16 logits are required. Key-only KV is
experimental and changes attention arithmetic; the FP8 sliding ring is this profile's default, with an explicit BF16 override.
Key-only's historical numerical and long-context speed costs remain in the [key-only measurements](maintainer/gemma4-benchmarks.md#key-only-global-kv-cache-experimental-off-by-default).

### Gemma 4 31B from a llama.cpp GGUF quant (IQ3_XS, IQ3_XXS, ...)

If a same-model IQ3_XS/IQ3_XXS GGUF already exists, skip collection/quantization and import it directly.
The original BF16 tree is still required; this is not a GGUF-only conversion:

```text
python -m tools.convert.gemma4_q4 convert --model "<BF16-dir>" --gguf-source "<same-model-IQ3.gguf>" --mtp-assistant "<assistant-dir>" --text-only --out "<camp>/model-iq3-mtp.ninfer"
python -m tools.convert.gemma4_q4 verify --artifact "<camp>/model-iq3-mtp.ninfer"
python -m tools.convert.gemma4_build launcher --artifact "<camp>/model-iq3-mtp.ninfer" --serve "<absolute-infernix-serve-path>" --out "<camp>/serve-model-iq3.cmd"
python -m tools.convert.gemma4_build serve-check --launcher "<camp>/serve-model-iq3.cmd"
```

For no MTP replace `--mtp-assistant "<assistant-dir>"` with `--no-mtp`. Every text-layer projection is
copied **as GGUF blocks, not dequantized and requantized**. The GGUF vocabulary is **not** imported:
the tied embedding/output vocabulary is encoded from HF BF16 into FP8-row storage. Norms, scalars,
tokenizer/template and optional components likewise use their explicit HF sources. A published GGUF of a
different base or a pre-edit variant is not an equivalent source. Projection blocks are stored in
`ggml_rows_v1` in the GGUF's own format (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0, Q4_K, Q5_K or Q6_K) and run
by the GGUF block kernels. With the default `--weights q4_g32` the projections allow 8-bit activations (AllowA8), as
q4 trunks do: decode and MTP verification (up to 63 columns) take `ops::ggml_linear`'s 8-bit activation route, and
prefill (64 columns and more) widens the blocks once per call to E4M3 rows with a BF16 row multiplier
(`ops::ggml_widen_e4m3`) and runs the FP8 A8 GEMM. `--weights q4_g32_a16` keeps them on 16-bit activations
(A16Only): the exact A16 route at decode widths, and from 64 columns the blocks decoded into a dense BF16 matrix and
run as one cuBLAS GEMM. Existing artifacts keep their stored policies: loading an A16Only artifact does not
silently opt it into A8. There are no temporary GGUF activation environment switches.

For an explicit exact-A16 activation artifact, add `--weights q4_g32_a16` to the direct import command and
choose a fresh output path. The `q4_g32` part of this option name does **not** requantize imported IQ3
weights: `--gguf-source` decides their blocks; the option sets the activation permission. The builder presets
always choose AllowA8 and have no `--weights` flag. To use their retained GGUF on A16, build with
`--keep-work`, then import `work-<name>/model-IQ3_XS.gguf` or `model-IQ3_XXS.gguf` with that option.
This does not overwrite or silently change an existing artifact.

See the [canonical comparison](maintainer/gemma4-benchmarks.md#four-tier-comparison) for model-specific
quality costs and thinking-enabled rates. Smaller storage does not imply equal quality.
Global-layer query and key rows are reordered as whole rows; the global output projection keeps the
GGUF's column order and the text config records `global_output_order: "source"`, so the runtime reads the
attention output through a permutation of 64-element blocks. A 256-element block cannot be split, so this is the
only exact way to keep those tensors. `--gguf-source` excludes `--gptq-source`, scale searches, imatrix,
group sizes and `--native-groups`: the GGUF decides every projection's stored blocks. Code:
`tools/convert/gemma4_gguf.py`; native test `infernix_gemma4_gguf_projection_test`.

## Write a conversion method

A method receives a `PrepareRequest` and returns `request.job(produce=...)`. Preparation validates
the target and determines auxiliary values. The `produce` function reads bounded source regions
and writes values or codes/scales through `TensorOutput`; the writer owns placement and file I/O.

This example adds explicit clipping before the existing grouped quantizer. It demonstrates the
method interface; the clipping threshold is a numerical choice made by the recipe author.

```python
import math
import torch
from tools.artifact.formats import QuantFormat, get_format
from tools.convert.quantization.groupwise import quantize_matrix


def clipped_grouped(request):
    if len(request.target.shape) != 2 or not isinstance(
        get_format(request.target.format), QuantFormat
    ):
        raise ValueError("clipped_grouped requires a grouped-integer matrix")
    limit = float(request.parameters["clip"])
    if not math.isfinite(limit) or limit <= 0 or request.rows_per_chunk <= 0:
        raise ValueError("clip and rows_per_chunk must be positive")
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("source contains non-finite values")
            encoded = quantize_matrix(
                values.clamp(-limit, limit),
                request.target.format,
                device=request.device,
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def configure(model, recipe, sources):
    recipe.assign(
        "text/layers/0/mlp/down",
        format="q6_g64_fp16",
        method=clipped_grouped,
        parameters={"clip": 1.0},
    )
```

Use this as an override. `request.values` traverses the prepared logical inputs in parent order,
including explicit groups. `output.write_codes` performs the registered packing and validates
codes/scales; the method should not duplicate that byte-layout logic. Direct output uses
`output.write_values`. Keep source blocks and temporary device tensors bounded to the method's
working set. `--rows-per-chunk` defaults to 512; custom methods own how they use it.

## Resources, files and inspection

Text includes `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` and
`generation_config.json`. Vision adds its image and video processor configs. Resources come from
`--model`; `--resource ROLE=PATH` replaces a selected resource:

```text
--resource chat_template.jinja=/path/to/chat_template.jinja
```

`--resource sampling.json=PATH` adds the optional text resource `sampling.json`: a checkpoint's
recommended sampling presets, which the runtime uses as its defaults in place of the
architecture's (process flags and request fields still override them). It holds a `thinking` and a
`non_thinking` preset, each with `temperature` [0,2], `top_k` [1,20], `top_p` [0,1], `min_p` [0,1],
`presence_penalty` and `frequency_penalty` [-2,2], and optionally `repetition_penalty` (only 1: the
sampler has none), plus an optional `source` string. The converter and the runtime refuse a missing,
unknown or out-of-range value. [ornith-1.5-122b.json](../tools/convert/sampling/ornith-1.5-122b.json)
is Ornith-1.5-122B's model-card set and
[ornith-1.5-122b-uncensored.json](../tools/convert/sampling/ornith-1.5-122b-uncensored.json)
dealignai's Ornith-1.5-122B-UNCENSORED-NVFP4's.

The official conversion examples select these maintained templates:

| Model | Template | Defaults |
|---|---|---|
| Qwen3.6 Dense/MoE | [qwen3_6.jinja](../tools/chat_templates/qwen3_6.jinja) | thinking on; closed-turn reasoning omitted |
| Qwen3.8 27B and Flash-Next | [qwen3_8.jinja](../tools/chat_templates/qwen3_8.jinja) | thinking on; effort `xhigh`; closed-turn reasoning retained |

Use your own Jinja file to change the artifact's default template. A startup
[`--chat-template FILE`](cli.md#text-input) overrides the stored template.
`generation_config.json` is preserved; sampling presets remain determined by the architecture
and explicit application/request settings.

The default maximum file size is 32,000,000,000 bytes, including framing. Smaller artifacts remain
one file. Larger artifacts use an entry such as `models/my_qwen.ninfer` plus
`my_qwen.ninfer.part-0001`, `my_qwen.ninfer.part-0002`, and so on in the same directory. Pass only the
entry path to Infernix and keep all its recorded parts together. `--max-file-bytes` changes the limit.

Conversion writes `models/my_qwen.ninfer.conversion.json` alongside the artifact, recording sources,
methods, formats, component configs, files and timing. Existing output files are not overwritten.
The report is useful for reproducing a recipe; the Engine reads the artifact itself.

```bash
python3 -m tools.artifact.inspect models/my_qwen.ninfer --objects --bindings
python3 -m tools.artifact.inspect models/my_qwen.ninfer --json
```

Inspection reads directory facts without running inference.

```bash
python3 -m tools.artifact.rename models/my_qwen.ninfer my_qwen.infernix
```

Renaming moves a whole file set in place, for example a `.ninfer` artifact to `.infernix`: the parts
take the new entry's names and the entry's file table is rewritten inside its existing directory
space, so no payload moves. `--numbered` names the files as numbered shards instead,
`my_qwen-00001-of-00003.infernix` (the entry, the file to open) to `my_qwen-00003-of-00003.infernix`.
The conversion report follows; other sidecars (an n-gram volume, a saved expert state) are listed
and keep their names.

Conversion rejects missing logical coverage, invalid source geometry, unsupported encodings and
invalid method output. Actual Op
support is checked by consumers during preparation, resource queries, warmup or execution. A
valid file may need additional Op support before its chosen combination can run. Exercise the
phases and optional components you intend to use through the normal [CLI](cli.md) or
[serving](serving.md) route.
