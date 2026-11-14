# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Vendored, unmodified apart from the import preamble below, from LLM Compressor 0.14.0
# (https://github.com/vllm-project/llm-compressor, Apache-2.0, see https://www.apache.org/licenses/LICENSE-2.0),
# PR #3128 "Add batched GPTQ quantization and triton kernel", file
# src/llmcompressor/modifiers/gptq/gptq_quantize.py, lines 261-516 of the 0.14.0 wheel
# (`fused_gptq_block_update` and `_gptq_block_update_kernel`).
#
# One launch quantizes a whole block of up to 256 columns for a stack of weights, doing for every column exactly what the
# per-column Python loop of LLM Compressor 0.13.0 (`gptq_quantize.py:177-238`) does: divide (round to nearest), clamp,
# rint, round trip through the scale dtype, error against the Hessian diagonal, tail update with separately rounded
# multiply and subtract (no fused multiply-add). The pinned environment (llmcompressor 0.13.0) does not ship it, so
# `fast_solve.py` installs it at run time; nothing in the environment is modified.
#
# Changes against upstream: the first lines (imports) only. `HAS_TRITON`, `tl` and `triton` come from
# compressed-tensors 0.18's `compressed_tensors.utils.triton` (same names as upstream's import) and `_is_fp8_supported`
# (upstream imports it from compressed_tensors 0.19) is a local copy; it is only consulted for the FP8 mode
# (`quant_type == 2`), which GPTQ for INT4 never uses.
import torch
from compressed_tensors.utils.triton import HAS_TRITON, tl, triton


def _is_fp8_supported(device) -> bool:
    return torch.cuda.get_device_capability(device) >= (8, 9)


def fused_gptq_block_update(
    work: torch.Tensor,
    hinv: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor | None,
    quantized: torch.Tensor,
    errors: torch.Tensor,
    q_min: float,
    q_max: float,
    quant_type: int,
) -> None:
    """Run the fused Triton GPTQ block update on FP32 working tensors."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is unavailable")
    if (
        work.dim() != 3
        or work.device.type != "cuda"
        or work.dtype != torch.float32
        or hinv.dtype != torch.float32
        or quantized.dtype != torch.float32
        or errors.dtype != torch.float32
        or scale.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or any(
            tensor.device != work.device for tensor in (hinv, scale, quantized, errors)
        )
        or (zero_point is not None and zero_point.device != work.device)
    ):
        raise ValueError("invalid tensors for fused GPTQ block update")

    batch, out_rows, width = work.shape
    if (
        width <= 0
        or width > 256
        or width & (width - 1)
        or hinv.shape != (batch, width, width)
        or scale.shape != work.shape
        or quantized.shape != work.shape
        or errors.shape != work.shape
    ):
        raise ValueError("invalid shapes for fused GPTQ block update")
    if zero_point is not None and zero_point.shape != work.shape:
        raise ValueError("zero_point must have the same shape as work")

    dequant_dtype = {
        torch.float32: 0,
        torch.bfloat16: 1,
        torch.float16: 2,
    }[scale.dtype]
    has_zp = zero_point is not None
    use_fp8_e4b15 = quant_type == 2 and not _is_fp8_supported(work.device)
    if has_zp:
        zero_point = zero_point.to(torch.float32)

    block_rows = 16
    _gptq_block_update_kernel[(batch, triton.cdiv(out_rows, block_rows))](
        work,
        hinv,
        scale,
        zero_point if has_zp else scale,
        quantized,
        errors,
        out_rows,
        *work.stride(),
        *hinv.stride(),
        *scale.stride(),
        *(zero_point.stride() if has_zp else (0, 0, 0)),
        *quantized.stride(),
        *errors.stride(),
        float(q_min),
        float(q_max),
        WIDTH=width,
        QUANT_TYPE=quant_type,
        DEQUANT_DTYPE=dequant_dtype,
        HAS_ZP=has_zp,
        USE_FP8_E4B15=use_fp8_e4b15,
        BLOCK_ROWS=block_rows,
        num_warps=4,
        num_stages=2,
    )


if HAS_TRITON:

    @triton.jit
    def _gptq_block_update_kernel(
        work_ptr,
        hinv_ptr,
        scale_ptr,
        zp_ptr,
        quant_ptr,
        errors_ptr,
        out_rows,
        stride_w_b,
        stride_w_r,
        stride_w_c,
        stride_h_b,
        stride_h_r,
        stride_h_c,
        stride_s_b,
        stride_s_r,
        stride_s_c,
        stride_z_b,
        stride_z_r,
        stride_z_c,
        stride_q_b,
        stride_q_r,
        stride_q_c,
        stride_e_b,
        stride_e_r,
        stride_e_c,
        q_min,
        q_max,
        WIDTH: tl.constexpr,
        QUANT_TYPE: tl.constexpr,
        DEQUANT_DTYPE: tl.constexpr,
        HAS_ZP: tl.constexpr,
        USE_FP8_E4B15: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        batch = tl.program_id(axis=0)
        row_block = tl.program_id(axis=1)
        rows = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        cols = tl.arange(0, WIDTH)
        batch_i64 = batch.to(tl.int64)
        rows_i64 = rows.to(tl.int64)
        cols_i64 = cols.to(tl.int64)
        row_mask = rows < out_rows
        work_offsets = (
            batch_i64 * stride_w_b
            + rows_i64[:, None] * stride_w_r
            + cols_i64[None, :] * stride_w_c
        )
        work = tl.load(work_ptr + work_offsets, mask=row_mask[:, None], other=0.0).to(
            tl.float32
        )

        for column in range(0, WIDTH):
            selector = cols[None, :] == column
            weight_column = tl.sum(tl.where(selector, work, 0.0), axis=1)
            scale = tl.load(
                scale_ptr
                + batch_i64 * stride_s_b
                + rows_i64 * stride_s_r
                + column * stride_s_c,
                mask=row_mask,
                other=1.0,
            ).to(tl.float32)
            scale = tl.maximum(scale, 1.1754943508222875e-38)
            normalized = tl.extra.cuda.libdevice.div_rn(weight_column, scale)
            if HAS_ZP:
                zp = tl.load(
                    zp_ptr
                    + batch_i64 * stride_z_b
                    + rows_i64 * stride_z_r
                    + column * stride_z_c,
                    mask=row_mask,
                    other=0.0,
                ).to(tl.float32)
                normalized = normalized + zp

            clamped = tl.clamp(normalized, q_min, q_max)
            if QUANT_TYPE == 0:
                rounded = tl.extra.cuda.libdevice.rint(clamped)
            elif QUANT_TYPE == 1:
                absolute = tl.abs(clamped)
                magnitude = tl.where(
                    absolute <= 0.25,
                    0.0,
                    tl.where(
                        absolute < 0.75,
                        0.5,
                        tl.where(
                            absolute <= 1.25,
                            1.0,
                            tl.where(
                                absolute < 1.75,
                                1.5,
                                tl.where(
                                    absolute <= 2.5,
                                    2.0,
                                    tl.where(
                                        absolute < 3.5,
                                        3.0,
                                        tl.where(absolute <= 5.0, 4.0, 6.0),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
                rounded = tl.where(clamped < 0.0, -magnitude, magnitude)
            else:
                # Ampere does not expose the native E4M3FN (``float8e4nv``) conversion
                # in Triton, but it does expose E4B15.  The formats have the same sign,
                # exponent, and mantissa widths and differ only by eight in exponent
                # bias, so scaling by 2**-8 before the cast and 2**8 afterwards gives
                # the E4M3FN rounding operation without requiring Hopper instructions.
                if USE_FP8_E4B15:
                    rounded = (clamped * 0.00390625).to(tl.float8e4b15).to(
                        tl.float32
                    ) * 256.0
                else:
                    rounded = clamped.to(tl.float8e4nv).to(tl.float32)

            if DEQUANT_DTYPE == 1:
                rounded = rounded.to(tl.bfloat16)
                scale_value = scale.to(tl.bfloat16)
                if HAS_ZP:
                    rounded = (rounded - zp.to(tl.bfloat16)).to(tl.bfloat16)
                quantized_column = (
                    (rounded * scale_value).to(tl.bfloat16).to(tl.float32)
                )
            elif DEQUANT_DTYPE == 2:
                rounded = rounded.to(tl.float16)
                scale_value = scale.to(tl.float16)
                if HAS_ZP:
                    rounded = (rounded - zp.to(tl.float16)).to(tl.float16)
                quantized_column = (rounded * scale_value).to(tl.float16).to(tl.float32)
            else:
                if HAS_ZP:
                    quantized_column = tl.extra.cuda.libdevice.mul_rn(
                        tl.extra.cuda.libdevice.sub_rn(rounded, zp), scale
                    )
                else:
                    quantized_column = tl.extra.cuda.libdevice.mul_rn(rounded, scale)

            diagonal = tl.load(
                hinv_ptr
                + batch_i64 * stride_h_b
                + column * stride_h_r
                + column * stride_h_c,
            ).to(tl.float32)
            error = tl.extra.cuda.libdevice.div_rn(
                tl.extra.cuda.libdevice.sub_rn(weight_column, quantized_column),
                diagonal,
            )
            q_offsets = (
                batch_i64 * stride_q_b + rows_i64 * stride_q_r + column * stride_q_c
            )
            e_offsets = (
                batch_i64 * stride_e_b + rows_i64 * stride_e_r + column * stride_e_c
            )
            tl.store(quant_ptr + q_offsets, quantized_column, mask=row_mask)
            tl.store(errors_ptr + e_offsets, error, mask=row_mask)

            hinv_row = tl.load(
                hinv_ptr
                + batch_i64 * stride_h_b
                + column * stride_h_r
                + cols_i64 * stride_h_c,
            ).to(tl.float32)
            tail = cols[None, :] > column
            update = tl.extra.cuda.libdevice.mul_rn(error[:, None], hinv_row[None, :])
            work = tl.where(tail, tl.extra.cuda.libdevice.sub_rn(work, update), work)

        tl.store(work_ptr + work_offsets, work, mask=row_mask[:, None])
