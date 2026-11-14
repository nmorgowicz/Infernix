"""Native q4_g128/g256 words decode identically to the g32-emulated words and are smaller."""

from __future__ import annotations

import pytest
import torch

from tools.artifact.codecs.row_split import dequantize_row_split, encode_row_split
from tools.artifact.layouts import encoded_size
from tools.convert.methods import grouped_absmax_emulated, grouped_mse_search
from tools.convert.model import Model, Parameter
from tools.convert.quantization.groupwise import (
    QuantizedMatrix,
    quantize_q4_scale_search,
    quantize_q4_virtual_groups,
    to_native_groups,
)
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def _values(k: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(5)
    values = torch.randn(7, k, generator=generator) * 0.03
    values[0] = 0.0
    values[3, 10] = 4.0
    return values


def _decode(codes, scales, format_name, shape):
    payload = encode_row_split(codes, scales, format_name, shape)
    return dequantize_row_split(payload, format_name, shape, dtype=torch.float32)


@pytest.mark.parametrize("group_size", [128, 256])
@pytest.mark.parametrize("k", [256, 512, 1024])
@pytest.mark.parametrize("search", [False, True])
def test_native_decodes_identically_to_the_emulated_g32_words(group_size, k, search):
    values = _values(k)
    emulated = (
        quantize_q4_scale_search(values, group_size, torch.rand(k), device="cpu")
        if search
        else quantize_q4_virtual_groups(values, group_size, device="cpu")
    )
    native = to_native_groups(emulated, group_size)
    assert native.codes.shape == (7, k // group_size, group_size)
    assert native.scales.shape == (7, k // group_size)
    want = _decode(emulated.codes, emulated.scales, "q4_g32_fp16", values.shape)
    got = _decode(native.codes, native.scales, f"q4_g{group_size}_fp16", values.shape)
    assert torch.equal(got, want)


def test_native_artifacts_are_smaller_by_the_dropped_scale_bytes():
    shape = (16384, 5376)
    g32 = encoded_size("row_split_k128_v1", "q4_g32_fp16", shape)
    for group_size in (128, 256):
        native = encoded_size("row_split_k128_v1", f"q4_g{group_size}_fp16", shape)
        saved = shape[0] * shape[1] * (2 / 32 - 2 / group_size)
        assert abs((g32 - native) - saved) < 256 * 2  # plane alignment only


def test_non_constant_emulated_scales_are_refused():
    bad = QuantizedMatrix(
        codes=torch.zeros(1, 4, 32, dtype=torch.int8),
        scales=torch.tensor([[1.0, 2.0, 1.0, 1.0]], dtype=torch.float16),
    )
    with pytest.raises(ValueError, match="not constant"):
        to_native_groups(bad, 128)


def _model(k):
    model = Model({"text": {"config": {}}})
    values = torch.randn(4, k).to(torch.bfloat16)
    model.add(Parameter("w", (4, k), array_source(values, "w"), inputs=("input",)))
    model.packing_groups = []
    return model


@pytest.mark.parametrize("method", [grouped_absmax_emulated, grouped_mse_search])
def test_recipe_builds_native_targets_and_refuses_a_mismatched_group(method):
    recipe = Recipe(_model(512))
    recipe.assign("w", format="q4_g256_fp16", method=method, parameters={"group_size": 256})
    assert recipe.prepare(device="cpu").weights[0].spec.format == "q4_g256_fp16"
    for target, size in (("q4_g256_fp16", 128), ("q4_g128_fp16", 256), ("q4_g128_fp16", 32)):
        recipe = Recipe(_model(512))
        recipe.assign("w", format=target, method=method, parameters={"group_size": size})
        with pytest.raises(ValueError, match="requires target format"):
            recipe.prepare(device="cpu")
