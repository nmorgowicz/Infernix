"""Fused GPTQ solve (tools/gptq/fast_solve.py): the kernel's arithmetic equals LLM Compressor 0.13.0's per-column loop bit for bit.

The CPU tests need no GPU: `block_update_torch` is the Triton kernel's arithmetic written op for op in torch and
`block_update_eager` is the 0.13.0 loop (compressed-tensors `fake_quantize` per column). The GPU test runs the real kernel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("compressed_tensors", reason="fast_solve needs compressed-tensors from the GPTQ venv (tools/gptq/requirements.txt)")
from compressed_tensors.quantization import QuantizationArgs  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gptq"))
import fast_solve  # noqa: E402

ARGS = {
    group: QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=group, actorder=None)
    for group in (32, 256)
}


def problem(rows: int, cols: int, group: int, seed: int, device: str = "cpu"):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(rows, cols, generator=generator) * 0.02
    weight[:, 3] *= 40  # clamped codes
    weight[0, :] = 0  # exact zeros
    samples = torch.randn(cols, 4 * cols, generator=generator)
    H = samples @ samples.T / samples.shape[1] + 0.01 * torch.eye(cols)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    scale = (weight.reshape(rows, cols // group, group).abs().amax(-1) / 7.5).clamp_min(1e-8).to(torch.bfloat16)
    return weight.to(device), Hinv.to(device), scale.to(device)


def eager(group):
    return lambda *a: fast_solve.block_update_eager(*a, ARGS[group])


@pytest.mark.parametrize("group", [32, 256])
@pytest.mark.parametrize("rows", [5, 40])
def test_kernel_semantics_equal_the_eager_column_loop(group, rows):
    weight, Hinv, scale = problem(rows, 256, group, seed=rows + group)
    results = []
    for update in (eager(group), fast_solve.block_update_torch):
        W = weight.clone()
        losses = fast_solve.solve_blocks(W, Hinv, scale, group, 128, -8.0, 7.0, update)
        results.append((W, losses))
    (a, la), (b, lb) = results
    assert torch.equal(a, b)
    torch.testing.assert_close(la, lb, rtol=1e-4, atol=0)
    assert not torch.equal(a, weight)  # the solve did change the weights


def test_unsupported_modules_fall_back_to_the_original_function():
    on_cpu = SimpleNamespace(weight=torch.zeros(4, 128))
    assert not fast_solve.supported(on_cpu, ARGS[32], 128)  # kernel needs CUDA
    if torch.cuda.is_available():
        on_gpu = SimpleNamespace(weight=torch.zeros(4, 128, device="cuda"))
        assert fast_solve.supported(on_gpu, ARGS[32], 128)
        assert not fast_solve.supported(on_gpu, ARGS[32], 96)  # block width must be a power of two
        assert not fast_solve.supported(SimpleNamespace(weight=torch.zeros(4, 200, device="cuda")), ARGS[32], 128)
        asymmetric = QuantizationArgs(num_bits=4, type="int", symmetric=False, strategy="group", group_size=32, actorder=None)
        assert not fast_solve.supported(on_gpu, asymmetric, 128)


def test_modes():
    assert fast_solve.MODES == ("auto", "on", "off")
    assert fast_solve.install("off") is False
    with pytest.raises(ValueError):
        fast_solve.install("sometimes")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and Triton")
@pytest.mark.parametrize("group", [32, 256])
def test_triton_kernel_is_bit_identical_to_the_eager_loop(group):
    assert fast_solve.self_test("cuda") is None
    weight, Hinv, scale = problem(40, 512, group, seed=group, device="cuda")
    results = []
    for update in (eager(group), fast_solve.block_update_triton):
        W = weight.clone()
        losses = fast_solve.solve_blocks(W, Hinv, scale, group, 128, -8.0, 7.0, update)
        results.append((W, losses))
    (a, la), (b, lb) = results
    assert torch.equal(a, b)
    torch.testing.assert_close(la, lb, rtol=1e-4, atol=0)


def test_a_failing_self_test_keeps_the_eager_solver_with_a_warning(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fast_solve, "self_test", lambda device="cuda": "weights differ in 3 of 10240 entries")
    assert fast_solve.install("auto") is False
    assert "WARNING: fused solve unavailable (weights differ in 3 of 10240 entries); using the eager solver" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="fused solve unavailable"):
        fast_solve.install("on")


def test_no_cuda_device_is_a_clean_fallback(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert fast_solve.install("auto") is False
    assert "no CUDA device" in capsys.readouterr().out


@pytest.mark.parametrize("rows", [5, 40])
def test_five_bit_group64_kernel_arithmetic_matches_eager(rows):
    args = QuantizationArgs(num_bits=5, type="int", symmetric=True, strategy="group", group_size=64, actorder=None)
    weight, Hinv, old_scale = problem(rows, 256, 64, seed=rows + 64)
    scale = (old_scale.float() * (7.5 / 15.5)).to(torch.bfloat16)
    results = []
    for update in (lambda *a: fast_solve.block_update_eager(*a, args), fast_solve.block_update_torch):
        W = weight.clone()
        losses = fast_solve.solve_blocks(W, Hinv, scale, 64, 128, -16.0, 15.0, update)
        results.append((W, losses))
    assert torch.equal(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1], rtol=1e-4, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and Triton")
def test_five_bit_group64_triton_self_test():
    assert fast_solve.self_test("cuda", group_size=64, bits=5) is None


def test_five_bit_install_qualifies_the_five_bit_range(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def self_test(device, **kwargs):
        calls.append((device, kwargs))
        return "qualification failed"

    monkeypatch.setattr(fast_solve, "self_test", self_test)
    assert fast_solve.install("auto", bits=5) is False
    assert calls == [("cuda", {"group_size": 64, "bits": 5})]
