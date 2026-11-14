"""compressed-tensors ``pack-quantized`` INT4 (symmetric, static contiguous groups) as encoded ``q4_g32_fp16`` rows.

The source keeps its codes and scales: they are copied, never re-rounded. Layout accepted, per matrix ``<prefix>``:

- ``<prefix>.weight_packed``: int32 ``[N, K/8]``; eight offset-binary 4-bit codes per word, element ``8*w + j`` in
  bits ``4j..4j+3`` of word ``w`` (``unsigned = code + 8``).
- ``<prefix>.weight_scale``: BF16/FP16/FP32 ``[N, K/G]`` for the group size ``G`` (32, 64, 128 or 256). Every scale
  must convert to binary16 without changing its value; a coarse group repeats its scale over its 32-wide subgroups,
  which decodes to the same values.
- ``<prefix>.weight_shape``: int64 ``[2]`` equal to ``[N, K]``.

Refused: ``weight_zero_point`` and ``weight_g_idx`` (asymmetric or activation-ordered groups do not map to
contiguous symmetric groups), a ``K`` that is not a multiple of ``G``, and a ``G`` that is not one of the above.

Converter-side permutations (rows of global q/k, columns of global o) are applied to the encoded words as a row gather
and a column-group gather (32-wide, or the whole source group for native storage). A column permutation must move
whole groups, otherwise it is refused.
"""

from __future__ import annotations

import torch

from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

GROUP_SIZES = (32, 64, 128, 256)
NATIVE_GROUP_SIZES = (128, 256)
_PADDED_K = 128
_SCALE_DTYPES = ("BF16", "F16", "F32")


def column_group_index(col_index: torch.Tensor, width: int = 32) -> torch.Tensor:
    """Converter ``width``-group -> source ``width``-group for a converter-column -> source-column index; refuse a split group."""
    if col_index.dim() != 1 or col_index.numel() % width:
        raise ValueError(f"column permutation must be a vector whose length is a multiple of {width}")
    blocks = col_index.reshape(-1, width)
    first = blocks[:, 0]
    expected = first[:, None] + torch.arange(width, dtype=col_index.dtype)
    if bool((first % width).any()) or not torch.equal(blocks, expected):
        raise ValueError(
            f"the column permutation does not map every {width}-group onto a whole source {width}-group; "
            "its tensor cannot be imported exactly"
        )
    return first // width


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    """int32 ``[R, K/8]`` offset-binary words -> int8 ``[R, K]`` codes in [-8, 7]."""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32)
    nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
    return (nibbles.reshape(packed.shape[0], -1) - 8).to(torch.int8)


def exact_fp16_scales(scales: torch.Tensor, name: str) -> torch.Tensor:
    """Binary16 words equal in value to ``scales``; any scale that changes value is an error."""
    wide = scales.float()
    narrow = wide.to(torch.float16)
    if not bool(torch.isfinite(wide).all()) or bool((wide < 0).any()):
        raise ValueError(f"{name}: scales must be finite and nonnegative")
    inexact = narrow.float() != wide
    if bool(inexact.any()):
        worst = ((narrow.float() - wide).abs() / wide.clamp_min(1e-30))[inexact].max().item()
        raise ValueError(
            f"{name}: {int(inexact.sum())} scales are not exactly representable in binary16 "
            f"(largest relative change {worst:.3e}); the source cannot be imported exactly"
        )
    return narrow


def packed_int4_matrix_source(
    store: SafetensorsSource,
    prefix: str,
    shape: tuple[int, int],
    group_size: int,
    *,
    row_index: torch.Tensor | None = None,
    col_index: torch.Tensor | None = None,
    native: bool = False,
) -> LogicalSource:
    """Encoded rows of ``shape`` read from the packed matrix at ``prefix``.

    Without ``native`` the format is ``q4_g32_fp16`` (a coarse scale repeated over its 32-wide subgroups). With
    ``native`` and group size 128 or 256 it is the real ``q4_g128_fp16`` / ``q4_g256_fp16``: the source codes with one
    scale per source group; a column permutation must then move whole source groups, otherwise it is refused.

    ``row_index[i]`` is the source row of output row ``i``; ``col_index[j]`` the source column of output column ``j``.
    """
    if group_size not in GROUP_SIZES:
        raise ValueError(f"{prefix}: group size {group_size} is not one of {GROUP_SIZES}")
    if native and group_size not in (32, *NATIVE_GROUP_SIZES):
        raise ValueError(f"{prefix}: native storage exists for group sizes 32, 128 and 256 (no Gemma g64 kernels), not {group_size}")
    width = group_size if native else 32  # code-plane group width of the stored format
    target = f"q4_g{width}_fp16"
    n, k = shape
    if k % group_size:
        raise ValueError(f"{prefix}: K {k} is not a multiple of the group size {group_size}")
    if row_index is not None and (row_index.shape != (n,) or sorted(row_index.tolist()) != list(range(n))):
        raise ValueError(f"{prefix}: the row index must be a permutation of the {n} rows")
    if col_index is not None and (col_index.shape != (k,) or sorted(col_index.tolist()) != list(range(k))):
        raise ValueError(f"{prefix}: the column index must be a permutation of the {k} columns")
    groups = None if col_index is None else column_group_index(col_index, width)
    for extra in ("weight_zero_point", "weight_g_idx"):
        if store.has(f"{prefix}.{extra}"):
            raise ValueError(f"{prefix}: {extra} is present; asymmetric or reordered groups are not importable")
    packed_name, scale_name, shape_name = (f"{prefix}.{part}" for part in ("weight_packed", "weight_scale", "weight_shape"))
    for name, expected, dtype in ((packed_name, (n, k // 8), "I32"), (shape_name, (2,), "I64")):
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}")
    scale_info = store.describe(scale_name)
    if scale_info.shape != (n, k // group_size) or scale_info.dtype not in _SCALE_DTYPES:
        raise ValueError(
            f"{scale_name}: expected BF16/F16/F32 {(n, k // group_size)}, got {scale_info.dtype}{scale_info.shape}"
        )
    if store.read_flat(shape_name).tolist() != [n, k]:
        raise ValueError(f"{shape_name}: stored shape differs from {[n, k]}")
    k_pad = -(-k // _PADDED_K) * _PADDED_K
    if native and k_pad % width:
        raise ValueError(f"{prefix}: K padded to {_PADDED_K} ({k_pad}) must be a multiple of {width} for {target}")
    subgroups = 1 if native else group_size // 32

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        rows = None if row_index is None else row_index[begin:end]
        low, high = (begin, end) if rows is None else (int(rows.min()), int(rows.max()) + 1)
        words = store.read_flat(packed_name, low * (k // 8), high * (k // 8)).reshape(high - low, k // 8)
        raw = store.read_flat(scale_name, low * (k // group_size), high * (k // group_size)).reshape(
            high - low, k // group_size
        )
        if rows is not None:
            words, raw = words[rows - low], raw[rows - low]
        codes = unpack_codes(words).reshape(end - begin, k // width, width)
        scales = exact_fp16_scales(raw, scale_name).repeat_interleave(subgroups, dim=1)
        if groups is not None:
            codes, scales = codes[:, groups], scales[:, groups]
        if k_pad != k:
            codes = torch.cat((codes, torch.zeros(end - begin, (k_pad - k) // width, width, dtype=torch.int8)), dim=1)
            scales = torch.cat((scales, torch.zeros(end - begin, (k_pad - k) // width, dtype=torch.float16)), dim=1)
        return EncodedRows(target, codes.contiguous(), scales.contiguous())

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        values = (words.codes.float() * words.scales.float().unsqueeze(-1)).reshape(last - first, k_pad)[:, :k]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(shape, f"{store.path}:{prefix} (packed int4 g{group_size})", read, encoded)
