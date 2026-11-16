"""Encoded sources for the Gemma 4 text projections of a llama.cpp GGUF, imported bit-exactly.

The GGUF (``convert_hf_to_gguf.py`` ``gemma4``, quantized by ``llama-quantize``) stores every text-layer projection in
Hugging Face order, row for row and column for column. Each projection keeps its stored GGUF blocks verbatim
(``ggml_rows_v1``; any ggml block format ``ops::ggml_linear`` runs, F32/F16/BF16 as exact values):

* global-layer query and key rows are permuted by the converter's ``prope_perm`` as whole rows, so the stored rows are
  the GGUF's rows in the converted order;
* global-layer output columns cannot be permuted (a 256-element block spans 64-wide runs of several permuted
  positions), so they stay in Hugging Face order and the text config records ``global_output_order: source``; the
  runtime then reads the attention output through the inverse permutation in 64-element blocks.

Every other parameter (vocabulary, norms, layer scalars, MTP, vision) comes from the BF16 checkpoint as before.
"""

from __future__ import annotations

from pathlib import Path

import torch

from tools.artifact.codecs.ggml import BY_GGML_ID

from .gemma4 import prope_perm
from .sources.gguf_logical import ggml_or_plain
from .sources.gguf_raw import GgufStore
from .sources.logical import LogicalSource, select_rows

PERM_BLOCK = 64
_GGUF_ROLES = {
    "attention/query": "attn_q",
    "attention/key": "attn_k",
    "attention/value": "attn_v",
    "attention/output": "attn_output",
    "mlp/gate": "ffn_gate",
    "mlp/up": "ffn_up",
    "mlp/down": "ffn_down",
}


def _metadata_checks(store: GgufStore, config: dict) -> None:
    m = store.metadata
    if m.get("general.architecture") != "gemma4":
        raise ValueError(f"{store.path}: GGUF architecture {m.get('general.architecture')!r} is not gemma4")
    expected = {
        "gemma4.block_count": config["num_hidden_layers"],
        "gemma4.embedding_length": config["hidden_size"],
        "gemma4.feed_forward_length": config["intermediate_size"],
        "gemma4.attention.head_count": config["num_attention_heads"],
    }
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError(f"{store.path}: {key} = {m.get(key)!r}, the checkpoint config says {value!r}")


def head_row_ranges(perm: torch.Tensor, heads: int) -> tuple[tuple[int, int], ...]:
    """Source row ranges, in converted order, of `heads` heads whose rows are permuted by `perm`; consecutive source
    rows are merged into one range."""

    dim = int(perm.numel())
    ranges: list[list[int]] = []
    for head in range(heads):
        for j in perm.tolist():
            row = head * dim + j
            if ranges and ranges[-1][1] == row:
                ranges[-1][1] += 1
            else:
                ranges.append([row, row + 1])
    return tuple((a, b) for a, b in ranges)


def output_input_perm(perm: torch.Tensor, heads: int, block: int = PERM_BLOCK) -> list[int]:
    """The global output projection's input permutation in `block`-element blocks: source-order block j (Hugging Face
    column block of the stored matrix) reads block result[j] of the attention output, which is in converted order
    (converted position p holds source dimension perm[p] of its head)."""

    dim = int(perm.numel())
    if dim % block:
        raise ValueError(f"head dim {dim} is not whole {block}-element blocks")
    per_head = dim // block
    position = torch.empty_like(perm)
    position[perm] = torch.arange(dim, dtype=perm.dtype)
    blocks = []
    for b in range(per_head):
        first = position[b * block : (b + 1) * block]
        start = int(first[0])
        if start % block or not torch.equal(first, torch.arange(start, start + block, dtype=perm.dtype)):
            raise ValueError(f"prope_perm does not move whole {block}-element blocks")
        blocks.append(start // block)
    return [head * per_head + b for head in range(heads) for b in blocks]


def gguf_projection_sources(model, path: Path) -> tuple[dict[str, LogicalSource], dict[str, str]]:
    """Sources (by logical parameter name) and stored formats of every text-layer projection of `model`, read from
    the GGUF at `path` (one file or the first shard of a split model)."""

    paths = [Path(path)]
    if "-00001-of-" in paths[0].name:
        stem, _, rest = paths[0].name.partition("-00001-of-")
        count = int(rest.split(".")[0])
        paths = [paths[0].with_name(f"{stem}-{i:05d}-of-{rest}") for i in range(1, count + 1)]
    store = GgufStore(paths)
    config = model.config
    _metadata_checks(store, config)
    perm = prope_perm(config["global_head_dim"], config["partial_rotary_factor"])
    heads, kv = config["num_attention_heads"], config["num_global_key_value_heads"]
    sources, formats = {}, {}
    for name, parameter in model.parameters.items():
        if not (name.startswith("text/layers/") and parameter.projection):
            continue
        _, _, layer, role = name.split("/", 3)
        index = int(layer)
        tensor = f"blk.{index}.{_GGUF_ROLES[role]}.weight"
        stored = store.tensor(tensor).ggml_type
        if stored not in BY_GGML_ID:
            raise ValueError(f"{tensor}: GGUF type {stored} is not a ggml block format Gemma 4 binds")
        source = ggml_or_plain(store, tensor, tuple(parameter.shape))
        if config["layer_types"][index] == "full_attention" and role in ("attention/query", "attention/key"):
            source = select_rows(source, head_row_ranges(perm, heads if role == "attention/query" else kv))
        formats[name] = BY_GGML_ID[stored].name
        sources[name] = source
    if not sources:
        raise ValueError("the model has no text-layer projections")
    return sources, formats
