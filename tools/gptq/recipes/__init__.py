"""Per-family GPTQ recipes for `run_gptq.py`.

A recipe holds everything about a model family that the runner must not hard-code: which module class is one
sequential unit, which modules stay unquantized, which projections share a Hessian, and the calibration defaults.
Add a family by writing a module here that builds a `Recipe` and registering it below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Recipe:
    name: str
    sequential_targets: tuple[str, ...]  # class names LLM Compressor treats as one sequential layer
    ignore: tuple[str, ...]  # module patterns excluded from quantization (LLM Compressor `ignore`)
    saved_ignore: tuple[str, ...]  # `ignore` written into the checkpoint's quantization_config
    hessian_leaders: dict[str, str]  # follower leaf name -> leader leaf name reading the identical input tensor
    group_size: int
    observer: str
    dampening: float
    block_size: int
    # (checkpoint key, parameter) pairs that detect a tree loaded with wrong key names (random weights).
    load_check: Callable[[object, Path], list[tuple[str, object]]]


def _registry() -> dict[str, Recipe]:
    from .gemma4_31b import RECIPE as gemma4_31b

    return {recipe.name: recipe for recipe in (gemma4_31b,)}


def names() -> tuple[str, ...]:
    return tuple(_registry())


def get(name: str) -> Recipe:
    recipes = _registry()
    if name not in recipes:
        raise KeyError(f"unknown recipe {name!r}; available: {', '.join(recipes)}")
    return recipes[name]
