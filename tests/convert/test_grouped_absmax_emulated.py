"""Virtual coarse groups stored as q4_g32 decode exactly to coarse-group absmax values."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tools.artifact.codecs.row_split import dequantize_row_split, encode_row_split
from tools.convert.methods import grouped_absmax_emulated
from tools.convert.model import Model, Parameter
from tools.convert.quantization.groupwise import (
    quantize_matrix,
    quantize_q4_virtual_groups,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

FP16_MIN_SUBNORMAL = 2.0**-24


def _oracle(values: np.ndarray, group_size: int) -> np.ndarray:
    """Independent float64 coarse-group absmax round trip (logical columns only)."""
    n, k = values.shape
    out = np.zeros((n, k), dtype=np.float32)
    for row in range(n):
        for begin in range(0, k, group_size):
            chunk = values[row, begin : begin + group_size].astype(np.float32)
            amax = np.float32(np.abs(chunk).max())
            scale = np.float16(np.float32(np.float64(amax) / 7.0))
            if scale == 0 and amax > 0:
                scale = np.float16(FP16_MIN_SUBNORMAL)
            if scale == 0:
                continue
            reciprocal = np.float32(1.0 / np.float64(scale))
            codes = np.clip(np.rint(chunk * reciprocal), -8, 7)
            out[row, begin : begin + group_size] = codes * np.float32(scale)
    return out


def _matrix(k: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    values = torch.randn(6, k, generator=generator) * 0.02
    values[0, :] = 0.0  # all-zero row
    values[1, 40:80] = 1e-9  # FP16 scale underflow
    values[2, 5] = 3.0  # outlier inside one coarse group only
    values[3, : min(k, 64)] = 0.0  # zero groups next to live groups
    values[4, -1] = -65504.0 * 7.0  # scale at the FP16 maximum
    return values


@pytest.mark.parametrize("group_size", [32, 64, 128, 256])
@pytest.mark.parametrize("k", [128, 300, 640, 1000])
def test_g32_decode_equals_coarse_group_oracle(group_size, k):
    values = _matrix(k)
    encoded = quantize_q4_virtual_groups(values, group_size, device="cpu")
    payload = encode_row_split(encoded.codes, encoded.scales, "q4_g32_fp16", values.shape)
    decoded = dequantize_row_split(payload, "q4_g32_fp16", values.shape, dtype=torch.float32)
    assert torch.equal(decoded, torch.from_numpy(_oracle(values.numpy(), group_size)))
    ratio = group_size // 32
    scales = encoded.scales.view(torch.int16)
    for first in range(0, scales.shape[1], ratio):
        block = scales[:, first : first + ratio]
        assert bool((block == block[:, :1]).all())


@pytest.mark.parametrize("k", [128, 300])
def test_group_32_is_the_ordinary_absmax_quantizer(k):
    values = _matrix(k)
    virtual = quantize_q4_virtual_groups(values, 32, device="cpu")
    ordinary = quantize_matrix(values, "q4_g32_fp16", device="cpu")
    assert torch.equal(virtual.codes, ordinary.codes)
    assert torch.equal(virtual.scales.view(torch.int16), ordinary.scales.view(torch.int16))


def test_coarser_groups_never_beat_finer_ones_on_reconstruction():
    values = _matrix(1000)[1:]
    errors = []
    for group_size in (32, 64, 128, 256):
        got = _oracle(values.numpy(), group_size)
        errors.append(float(np.square(got - values.numpy()).sum()))
    assert errors == sorted(errors)


def _model(names):
    model = Model({"text": {"config": {}}})
    for name in names:
        values = torch.ones(4, 128, dtype=torch.bfloat16)
        model.add(Parameter(name, (4, 128), array_source(values, name), inputs=("input",)))
    model.packing_groups = [tuple(names)]
    return model


def test_equal_group_size_packs_and_unequal_does_not():
    packed = Recipe(_model(("gate", "up")))
    packed.assign("*", format="q4_g32_fp16", method=grouped_absmax_emulated,
                  parameters={"group_size": 128})
    assert len(packed.prepare(device="cpu").weights) == 1

    split = Recipe(_model(("gate", "up")))
    split.assign("*", format="q4_g32_fp16", method=grouped_absmax_emulated,
                 parameters={"group_size": 128})
    split.assign("up", parameters={"group_size": 256})
    assert len(split.prepare(device="cpu").weights) == 2


@pytest.mark.parametrize("parameters", [None, {}, {"group_size": 48}, {"group_size": "128"}])
def test_missing_or_invalid_group_size_is_rejected(parameters):
    recipe = Recipe(_model(("gate",)))
    recipe.assign("*", format="q4_g32_fp16", method=grouped_absmax_emulated,
                  parameters=parameters)
    with pytest.raises(ValueError, match="group_size"):
        recipe.prepare(device="cpu")
