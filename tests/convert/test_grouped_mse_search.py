"""Scale search is never worse than absmax, keeps codes nearest for the stored scale, and ties to absmax."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import dequantize_row_split, encode_row_split
from tools.convert.methods import grouped_mse_search
from tools.convert.model import Model, Parameter
from tools.convert.quantization.groupwise import (
    SEARCH_DIVISORS,
    quantize_q4_scale_search,
    quantize_q4_virtual_groups,
)
from tools.convert.quantization.importance import clear_importance, set_importance
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def _decode(encoded, shape):
    payload = encode_row_split(encoded.codes, encoded.scales, "q4_g32_fp16", shape)
    return dequantize_row_split(payload, "q4_g32_fp16", shape, dtype=torch.float32).numpy()


def _group_errors(values, decoded, weights, group_size):
    n, k = values.shape
    out = np.zeros((n, -(-k // group_size)))
    for row in range(n):
        for index, begin in enumerate(range(0, k, group_size)):
            x = values[row, begin : begin + group_size].astype(np.float64)
            y = decoded[row, begin : begin + group_size].astype(np.float64)
            out[row, index] = float((weights[begin : begin + group_size] * (y - x) ** 2).sum())
    return out


def _absmax_candidate_error(values, group_size, weights):
    return _group_errors(
        values,
        _decode(quantize_q4_virtual_groups(torch.from_numpy(values), group_size, device="cpu"), values.shape),
        weights,
        group_size,
    )


def _random(k, seed=3):
    generator = np.random.default_rng(seed)
    values = (generator.standard_normal((5, k)) * 0.02).astype(np.float32)
    values[0] = 0.0
    values[1, :40] = 0.0
    values[2, 7] = 0.5  # outlier that clipping should often prefer to ignore
    return values


@pytest.mark.parametrize("group_size", [32, 128, 256])
@pytest.mark.parametrize("k", [128, 300])
@pytest.mark.parametrize("weighted", [False, True])
def test_search_error_never_exceeds_absmax_and_stays_nearest(group_size, k, weighted):
    values = _random(k)
    weights = (
        np.random.default_rng(11).random(k).astype(np.float32) ** 3 if weighted else np.ones(k, np.float32)
    )
    importance = torch.from_numpy(weights) if weighted else None
    encoded = quantize_q4_scale_search(torch.from_numpy(values), group_size, importance, device="cpu")
    decoded = _decode(encoded, values.shape)
    searched = _group_errors(values, decoded, weights, group_size)
    assert bool((searched <= _absmax_candidate_error(values, group_size, weights) * (1 + 1e-6) + 1e-30).all())
    # Codes are the nearest in [-8, 7] for the stored scale: no weight can change that.
    scale = np.repeat(encoded.scales.float().numpy(), 32, axis=1)[:, :k]
    with np.errstate(divide="ignore", invalid="ignore"):
        nearest = np.clip(np.rint(np.where(scale > 0, values / scale, 0.0)), -8, 7) * scale
    assert np.allclose(decoded, nearest, atol=1e-6, rtol=1e-3)
    # One scale word per logical group.
    words = encoded.scales.view(torch.int16).numpy()
    ratio = group_size // 32
    for first in range(0, words.shape[1], ratio):
        assert (words[:, first : first + ratio] == words[:, first : first + 1]).all()


def test_search_actually_improves_on_an_outlier_group():
    values = _random(128)
    ones = np.ones(128, np.float32)
    assert float(
        _group_errors(
            values,
            _decode(quantize_q4_scale_search(torch.from_numpy(values), 32, None, device="cpu"), values.shape),
            ones,
            32,
        )[2].sum()
    ) < float(_absmax_candidate_error(values, 32, ones)[2].sum())


def test_unit_weights_equal_the_unweighted_arm_and_zero_weights_fall_back_to_absmax():
    values = _random(300)
    plain = quantize_q4_scale_search(torch.from_numpy(values), 128, None, device="cpu")
    unit = quantize_q4_scale_search(torch.from_numpy(values), 128, torch.ones(300), device="cpu")
    assert torch.equal(plain.codes, unit.codes)
    assert torch.equal(plain.scales.view(torch.int16), unit.scales.view(torch.int16))
    zero = quantize_q4_scale_search(torch.from_numpy(values), 128, torch.zeros(300), device="cpu")
    absmax = quantize_q4_virtual_groups(torch.from_numpy(values), 128, device="cpu")
    assert torch.equal(zero.codes, absmax.codes)
    assert torch.equal(zero.scales.view(torch.int16), absmax.scales.view(torch.int16))
    assert SEARCH_DIVISORS[0] == 7.0 and len(set(SEARCH_DIVISORS)) == len(SEARCH_DIVISORS)


def test_search_is_deterministic():
    values = torch.from_numpy(_random(300))
    importance = torch.rand(300)
    first = quantize_q4_scale_search(values, 64, importance, device="cpu")
    again = quantize_q4_scale_search(values, 64, importance, device="cpu")
    assert torch.equal(first.codes, again.codes) and torch.equal(first.scales, again.scales)


@pytest.mark.parametrize("bad", [torch.tensor([-1.0] + [1.0] * 127), torch.tensor([float("nan")] + [1.0] * 127), torch.ones(127)])
def test_invalid_importance_is_refused(bad):
    with pytest.raises(ValueError, match="importance"):
        quantize_q4_scale_search(torch.zeros(2, 128), 32, bad, device="cpu")


def _model(names):
    model = Model({"text": {"config": {}}})
    for name in names:
        values = torch.linspace(-1, 1, 4 * 128).reshape(4, 128).to(torch.bfloat16)
        model.add(Parameter(name, (4, 128), array_source(values, name), inputs=("input",)))
    model.packing_groups = [tuple(names)]
    return model


def _prepare(names, parameters):
    recipe = Recipe(_model(names))
    recipe.assign("*", format="q4_g32_fp16", method=grouped_mse_search, parameters=parameters)
    return recipe.prepare(device="cpu")


def test_weighted_method_needs_a_matching_vector_for_every_packed_input():
    clear_importance()
    with pytest.raises(ValueError, match="no importance vector"):
        _prepare(("gate", "up"), {"group_size": 32, "weighted": True})
    set_importance({"gate": torch.ones(128), "up": torch.ones(128) * 2})
    with pytest.raises(ValueError, match="differs"):
        _prepare(("gate", "up"), {"group_size": 32, "weighted": True})
    set_importance({"gate": torch.ones(128), "up": torch.ones(128)})
    assert len(_prepare(("gate", "up"), {"group_size": 32, "weighted": True}).weights) == 1
    set_importance({"gate": torch.ones(64), "up": torch.ones(64)})
    with pytest.raises(ValueError, match="width"):
        _prepare(("gate", "up"), {"group_size": 32, "weighted": True})
    clear_importance()


@pytest.mark.parametrize("parameters", [{}, {"group_size": 48}, {"group_size": 32, "weighted": "yes"}, {"group_size": 32, "extra": 1}, {"group_size": 32, "scale_bits": 6}])
def test_invalid_parameters_are_refused(parameters):
    with pytest.raises(ValueError):
        _prepare(("gate",), parameters)
