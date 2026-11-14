"""The QAT encoder keeps a group that is already on the 4-bit grid."""

from __future__ import annotations

import torch

from tools.artifact.codecs.row_split import dequantize_row_split, encode_row_split
from tools.convert.quantization.groupwise import quantize_matrix, quantize_q4_g32_qat


def _matrix() -> tuple[torch.Tensor, torch.Tensor]:
    scale = torch.tensor(0.015625, dtype=torch.float16)
    hit_eight = torch.zeros(32, dtype=torch.int8)
    hit_eight[:6] = torch.tensor([-8, -1, 0, 1, 5, 7], dtype=torch.int8)
    hit_seven = torch.zeros(32, dtype=torch.int8)
    hit_seven[:6] = torch.tensor([-7, -1, 0, 1, 4, 7], dtype=torch.int8)
    codes = torch.stack((hit_eight, hit_seven))
    values = codes.float() * scale.float()
    return values, codes


def test_q4_g32_qat_recovers_the_stored_codes():
    values, codes = _matrix()
    encoded = quantize_q4_g32_qat(values, device="cpu")
    assert torch.equal(encoded.codes[:, 0], codes)
    assert torch.all(encoded.codes[:, 1:] == 0)
    rebuilt = encoded.codes[:, 0].float() * encoded.scales[:, :1].float()
    assert torch.equal(rebuilt, values)


def test_fresh_absmax_moves_the_group_that_uses_code_minus_8():
    values, codes = _matrix()
    fresh = quantize_matrix(values, "q4_g32_fp16", device="cpu")
    assert not torch.equal(fresh.codes[:, 0], codes)


def test_q4_g32_row_split_round_trip():
    values, _ = _matrix()
    encoded = quantize_q4_g32_qat(values, device="cpu")
    payload = encode_row_split(encoded.codes, encoded.scales, "q4_g32_fp16", values.shape)
    decoded = dequantize_row_split(payload, "q4_g32_fp16", values.shape, dtype=torch.float32)
    assert torch.equal(decoded, values)
    bf16 = dequantize_row_split(payload, "q4_g32_fp16", values.shape)
    assert torch.equal(bf16, values.to(torch.bfloat16))
