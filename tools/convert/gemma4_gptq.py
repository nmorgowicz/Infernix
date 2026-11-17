"""Encoded sources for Gemma 4 text projections from compressed-tensors INT4/INT5 checkpoints.

The checkpoint is a text-only (`model.layers.N...`) or full (`model.language_model.layers.N...`) Hugging Face tree in
the original head order. Global-layer q/k rows and o columns are permuted by the converter (`prope_perm`); the same
permutation is applied here to the encoded words, so the stored codes and scales are the ones the external tool chose.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .gemma4 import prope_perm
from .sources.logical import LogicalSource
from .sources.packed_int import GROUP_SIZES, column_group_index, packed_matrix_source
from .sources.safetensors import SafetensorsSource

_HF_ROLES = {
    "attention/query": "self_attn.q_proj",
    "attention/key": "self_attn.k_proj",
    "attention/value": "self_attn.v_proj",
    "attention/output": "self_attn.o_proj",
    "mlp/gate": "mlp.gate_proj",
    "mlp/up": "mlp.up_proj",
    "mlp/down": "mlp.down_proj",
}


def checkpoint_group_size(config: dict, *, bits: int = 4) -> int:
    """Validate the source layout and requested bit width, returning the source group size."""
    if bits not in (4, 5):
        raise ValueError(f"GPTQ import supports 4 or 5 bits, not {bits}")
    quant = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
    if not quant or quant.get("quant_method") != "compressed-tensors" or quant.get("format") != "pack-quantized":
        raise ValueError("the GPTQ source must be a compressed-tensors pack-quantized checkpoint")
    groups = quant.get("config_groups") or {}
    if len(groups) != 1:
        raise ValueError(f"expected exactly one quantization config group, got {sorted(groups)}")
    weights = next(iter(groups.values())).get("weights") or {}
    group_size = weights.get("group_size")
    allowed_groups = (64,) if bits == 5 else GROUP_SIZES
    if (
        weights.get("num_bits") != bits
        or weights.get("type") != "int"
        or weights.get("symmetric") is not True
        or weights.get("strategy") != "group"
        or weights.get("actorder") not in (None, "null")
        or group_size not in allowed_groups
    ):
        raise ValueError(
            f"unsupported weight scheme {json.dumps(weights, sort_keys=True)}; "
            f"need {bits}-bit symmetric int, strategy group, group_size in {allowed_groups}, no activation order"
        )
    return group_size


def _head_index(perm: torch.Tensor, heads: int) -> torch.Tensor:
    """Converter index -> source index when every head of `perm.numel()` entries is permuted by `perm`."""
    dim = perm.numel()
    return (torch.arange(heads, dtype=torch.int64)[:, None] * dim + perm[None, :]).reshape(-1)


def gptq_projection_sources(
    model, root: Path, *, native: bool = False, bits: int = 4
) -> tuple[dict[str, LogicalSource], int, dict[str, str]]:
    """Exact text projections, source group size and target formats.

    INT4 defaults to g32 with repeated scales; native g128/g256 falls back to exact
    g32 for global o, whose permutation splits those groups. INT5 always keeps g64:
    the real Gemma global-o permutation moves complete 64-wide runs. Splits are refused.
    """
    store = SafetensorsSource(root)
    try:
        group_size = checkpoint_group_size(store.config, bits=bits)
        probe = "layers.0.mlp.down_proj.weight_packed"
        stem = next(
            (p for p in ("model.layers.", "model.language_model.layers.") if store.has(p[: -len("layers.")] + probe)),
            None,
        )
        if stem is None:
            raise ValueError(f"{root}: no packed Gemma 4 text projections (looked for model[.language_model].{probe})")
        config = model.config
        perm = prope_perm(config["global_head_dim"], config["partial_rotary_factor"])
        heads, kv = config["num_attention_heads"], config["num_global_key_value_heads"]
        sources, formats = {}, {}
        for name, parameter in model.parameters.items():
            if not (name.startswith("text/layers/") and parameter.projection):
                continue
            _, _, layer, role = name.split("/", 3)
            index = int(layer)
            global_layer = config["layer_types"][index] == "full_attention"
            row_index = col_index = None
            if global_layer and role in ("attention/query", "attention/key"):
                row_index = _head_index(perm, heads if role == "attention/query" else kv)
            elif global_layer and role == "attention/output":
                col_index = _head_index(perm, heads)
            keep_groups = bits == 5 or (native and group_size > 32)
            if bits == 4 and keep_groups and col_index is not None:
                try:
                    column_group_index(col_index, group_size)
                except ValueError:
                    keep_groups = False
            sources[name] = packed_matrix_source(
                store, f"{stem}{index}.{_HF_ROLES[role]}", parameter.shape, group_size,
                bits=bits, row_index=row_index, col_index=col_index, native=keep_groups,
            )
            formats[name] = f"q{bits}_g{group_size}_fp16" if keep_groups else "q4_g32_fp16"
        if not sources:
            raise ValueError("the model has no text-layer projections")
        return sources, group_size, formats
    finally:
        store.close()
