"""Gemma 4 31B text tree: the recipe of the shipped checkpoints (bit-for-bit)."""

from __future__ import annotations

from pathlib import Path

from . import Recipe


def load_check(model, tree: Path) -> list[tuple[str, object]]:
    layers = model.model.layers
    return [
        ("model.layers.0.mlp.down_proj.weight", layers[0].mlp.down_proj.weight),
        (f"model.layers.{len(layers) - 1}.self_attn.q_proj.weight", layers[-1].self_attn.q_proj.weight),
        ("model.embed_tokens.weight", model.model.embed_tokens.weight),
    ]


RECIPE = Recipe(
    name="gemma4_31b",
    sequential_targets=("Gemma4TextDecoderLayer",),
    ignore=("lm_head", "re:.*embed.*", "re:.*vision.*"),
    saved_ignore=("lm_head",),
    # q/k/v project the same hidden state and gate/up the same pre-MLP state: one Hessian serves all of them.
    hessian_leaders={"k_proj": "q_proj", "v_proj": "q_proj", "up_proj": "gate_proj"},
    group_size=32,
    observer="imatrix_mse",
    dampening=0.01,
    block_size=128,
    load_check=load_check,
)
