"""Fused GPTQ solve for LLM Compressor 0.13.0: one Triton launch per 128-column block instead of 128 Python steps.

`quantize_weight` is LLM Compressor 0.13.0's `modifiers/gptq/gptq_quantize.py:quantize_weight` (lines 84-264) for the one
case this route uses (group strategy, symmetric INT4, no activation order, bf16 scales) with the inner per-column loop
(lines 177-238: ~30 CUDA launches and 2-3 host/device syncs per column, 0.70 ms per column on the author's RTX 5090,
3.42 M columns for Gemma 4 31B) replaced by `fused_gptq_kernel.fused_gptq_block_update`, the kernel LLM Compressor 0.14.0
(PR #3128) ships for the same purpose. Everything else - dead-column masking, damping, Cholesky chain, the error
feedback GEMM between blocks, the RTN fallback - is the 0.13.0 code, so the output is bit-identical to the eager solver
(`tests/convert/test_gptq_fast_solve.py` checks the semantics on the CPU, `self_test` checks the kernel on the GPU at
every start, and the tool README records the full-run comparison). Anything outside that case falls back to the
original function.

`install(mode)` replaces the name `quantize_weight` in `llmcompressor.modifiers.gptq.base`, which `compress_module_list`
resolves at call time; the installed package is not touched.
"""

from __future__ import annotations

import copy
import time

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy, fake_quantize

GPTQ_PRECISION = torch.float32  # llmcompressor.modifiers.gptq.gptq_quantize.GPTQ_PRECISION
FLOAT32_TINY = 1.1754943508222875e-38  # the kernel clamps scales to this (smallest normal float32), like the eager divide does
MODES = ("auto", "on", "off")


def block_update_eager(W1, Hinv1, window, q_min, q_max, Q1, Err1, quant_args: QuantizationArgs):
    """LLM Compressor 0.13.0's per-column loop (`gptq_quantize.py:177-238`), with the group scale pre-gathered per column.

    W1 [rows, width] is updated in place; Q1 and Err1 receive the quantized columns and the scaled errors.
    `window[:, i]` is the scale of the group of column i (what `scale[:, g_idx[i1 + i]]` selects in the original)."""
    channel = copy.copy(quant_args)
    channel.strategy = QuantizationStrategy.CHANNEL
    zero_point = torch.zeros(W1.shape[0], dtype=torch.int8, device=W1.device)
    for i in range(W1.shape[1]):
        w = W1[:, i]
        d = Hinv1[i, i]
        q = fake_quantize(w.clone(), window[:, i], zero_point, channel)
        Q1[:, i] = q
        err = (w - q) / d
        W1[:, i:] -= err.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
        Err1[:, i] = err


def block_update_torch(W1, Hinv1, window, q_min, q_max, Q1, Err1):
    """The kernel's arithmetic in plain torch, op for op (divide, clamp, round half to even, bf16 round trip, error,
    separately rounded multiply and subtract on the tail). Slow; it documents the semantics and runs on the CPU."""
    for i in range(W1.shape[1]):
        w = W1[:, i]
        scale = window[:, i].float().clamp_min(FLOAT32_TINY)
        rounded = torch.round(torch.clamp(w / scale, q_min, q_max))
        q = (rounded.to(torch.bfloat16) * scale.to(torch.bfloat16)).float()
        err = (w - q) / Hinv1[i, i]
        Q1[:, i] = q
        Err1[:, i] = err
        W1[:, i + 1 :] -= err.unsqueeze(1) * Hinv1[i, i + 1 :].unsqueeze(0)


def block_update_triton(W1, Hinv1, window, q_min, q_max, Q1, Err1):
    from fused_gptq_kernel import fused_gptq_block_update

    fused_gptq_block_update(W1[None], Hinv1[None], window[None], None, Q1[None], Err1[None], q_min, q_max, 0)


def solve_blocks(W, Hinv, scale, group_size, blocksize, q_min, q_max, block_update):
    """The block loop of `quantize_weight` (0.13.0 lines 167-245). W is updated in place; returns the per-row loss."""
    rows, cols = W.shape
    g_idx = torch.arange(cols, device=W.device, dtype=torch.int) // group_size
    losses = torch.zeros(rows, device=W.device)
    for i1 in range(0, cols, blocksize):
        i2 = i1 + blocksize
        W1 = W[:, i1:i2].clone()
        Q1 = torch.empty_like(W1)
        Err1 = torch.empty_like(W1)
        window = scale.index_select(1, g_idx[i1:i2].long())
        block_update(W1, Hinv[i1:i2, i1:i2], window, q_min, q_max, Q1, Err1)
        W[:, i1:i2] = Q1
        losses += torch.sum(Err1.square(), 1) / 2
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
    return losses


def supported(module, quant_args: QuantizationArgs, blocksize: int) -> bool:
    weight = module.weight
    return bool(
        weight.is_cuda
        and quant_args.strategy == QuantizationStrategy.GROUP
        and quant_args.symmetric
        and quant_args.type == "int"
        and quant_args.num_bits == 4
        and not quant_args.actorder
        and 0 < blocksize <= 256
        and not blocksize & (blocksize - 1)
        and weight.ndim == 2
        and weight.shape[1] % blocksize == 0
    )


def quantize_weight(module, quant_args, hessian, blocksize=128, percdamp=0.01):
    """Drop-in for llmcompressor 0.13.0 `quantize_weight`; same arguments, same return value."""
    from llmcompressor.modifiers.gptq import gptq_quantize as ref

    if not supported(module, quant_args, blocksize):
        return ref.quantize_weight(module, quant_args, hessian, blocksize, percdamp)
    qparams = module.weight_observer.get_qparams()
    scale, zero_point = qparams["scale"], qparams["zero_point"]
    if scale.dtype != torch.bfloat16 or qparams["global_scale"] is not None or zero_point.any():
        return ref.quantize_weight(module, quant_args, hessian, blocksize, percdamp)

    final_shape, final_dtype = module.weight.shape, module.weight.dtype
    W = module.weight.clone().to(GPTQ_PRECISION)
    H = hessian
    cols = W.shape[1]
    q_min, q_max = -float(2 ** (quant_args.num_bits - 1)), float(2 ** (quant_args.num_bits - 1) - 1)

    dead = torch.diag(H) == 0  # 0.13.0 lines 143-146
    H[dead, dead] = 1
    W[:, dead] = 0
    try:  # 0.13.0 lines 148-164
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(H.shape[0], device=H.device)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        Hinv = torch.linalg.cholesky(H, upper=True)
    except torch._C._LinAlgError:
        ref.logger.warning(
            "Failed to invert hessian due to numerical instability. Consider increasing GPTQModifier.dampening_frac, "
            "increasing the number of calibration samples, or shuffling the calibration dataset. "
            "Falling back to round-to-nearest for this module."
        )
        Hinv = torch.eye(cols, dtype=H.dtype, device=H.device)

    losses = solve_blocks(W, Hinv, scale, quant_args.group_size, blocksize, q_min, q_max, block_update_triton)
    W = W.reshape(final_shape).to(final_dtype)
    return torch.sum(losses).item(), {
        "weight": W,
        "weight_scale": scale.to(dtype=final_dtype),
        "weight_zero_point": zero_point.to(dtype=quant_args.zp_dtype),
    }


def self_test(device: str = "cuda", rows: int = 40, cols: int = 256, group_size: int = 32, seed: int = 7) -> str | None:
    """Run the kernel and the eager loop on a small random problem (two blocks, rows not a multiple of the kernel's row
    tile, dead-looking small weights included). Returns None when they are bit-identical, else a description."""
    try:
        from fused_gptq_kernel import HAS_TRITON

        if not HAS_TRITON:
            return "Triton is not importable"
        generator = torch.Generator(device="cpu").manual_seed(seed)
        weight = (torch.randn(rows, cols, generator=generator) * 0.02).to(device)
        samples = torch.randn(cols, 4 * cols, generator=generator).to(device)
        H = samples @ samples.T / samples.shape[1] + 0.01 * torch.eye(cols, device=device)
        Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
        scale = (weight.reshape(rows, cols // group_size, group_size).abs().amax(-1) / 7.5).clamp_min(1e-8)
        scale = scale.to(torch.bfloat16)
        args = QuantizationArgs(num_bits=4, type="int", symmetric=True, strategy="group", group_size=group_size, actorder=None)
        results = []
        for update in (lambda *a: block_update_eager(*a, args), block_update_triton):
            W = weight.clone()
            losses = solve_blocks(W, Hinv, scale, group_size, 128, -8.0, 7.0, update)
            results.append((W, losses))
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        (a, la), (b, lb) = results
        if not torch.equal(a, b):
            return f"weights differ in {(a != b).sum().item()} of {a.numel()} entries (max {(a - b).abs().max().item():.3e})"
        if not torch.allclose(la, lb, rtol=1e-4, atol=0):
            return "losses differ"
        return None
    except Exception as error:  # compile failure, missing ptxas, unsupported GPU ...
        return f"{type(error).__name__}: {error}"


def install(mode: str = "auto", device: str = "cuda") -> bool:
    """Install the fused solve. `on` raises when it cannot run, `auto` warns and keeps the eager solver, `off` does nothing."""
    if mode not in MODES:
        raise ValueError(f"--fused-solve must be one of {MODES}")
    if mode == "off":
        print("fused solve: off (eager solver, about 0.7 ms per weight column)", flush=True)
        return False
    started = time.time()
    problem = self_test(device) if torch.cuda.is_available() else "no CUDA device"
    if problem is not None:
        message = f"fused solve unavailable ({problem}); using the eager solver, expect a run about 3x slower"
        if mode == "on":
            raise RuntimeError(message)
        print(f"WARNING: {message}", flush=True)
        return False
    from llmcompressor.modifiers.gptq import base

    base.quantize_weight = quantize_weight
    print(f"fused solve: on (kernel self-test bit-identical to the eager loop, {time.time() - started:.1f}s)", flush=True)
    return True
