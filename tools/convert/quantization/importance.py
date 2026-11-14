"""Per-input-column importance vectors for diagonal-weighted scale search.

The registry is process-global and explicit: a trusted caller sets it before ``Recipe.prepare``
and ``grouped_mse_search`` reads it by logical parameter id. Vectors are
``sum(in_sum2) / sum(counts)`` of a llama.cpp imatrix, already in the converter's logical column order
(the caller applies any column permutation). An id with no vector is refused, never replaced by ones.
"""

from __future__ import annotations

import torch

_IMPORTANCE: dict[str, torch.Tensor] = {}


def set_importance(vectors: dict[str, torch.Tensor]) -> None:
    checked = {}
    for name, vector in vectors.items():
        vector = torch.as_tensor(vector, dtype=torch.float32).detach().cpu().contiguous()
        if vector.dim() != 1 or not bool(torch.isfinite(vector).all()) or bool((vector < 0).any()):
            raise ValueError(f"{name}: importance must be a finite non-negative vector")
        checked[name] = vector
    _IMPORTANCE.clear()
    _IMPORTANCE.update(checked)


def clear_importance() -> None:
    _IMPORTANCE.clear()


def importance_for(names: tuple[str, ...], width: int) -> torch.Tensor:
    """Return the one vector shared by every logical input of a packed parent."""
    vectors = []
    for name in names:
        if name not in _IMPORTANCE:
            raise ValueError(f"{name}: no importance vector registered")
        vectors.append(_IMPORTANCE[name])
    first = vectors[0]
    if first.numel() != width:
        raise ValueError(f"{names[0]}: importance width {first.numel()} != {width}")
    for name, vector in zip(names[1:], vectors[1:]):
        if not torch.equal(vector, first):
            raise ValueError(
                f"{name}: importance differs from {names[0]} inside one packed parent"
            )
    return first
