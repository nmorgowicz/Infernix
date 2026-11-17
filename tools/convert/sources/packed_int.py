"""Exact compressed-tensors ``pack-quantized`` symmetric INT4/INT5 matrix sources.

INT4 words hold eight offset-binary codes; INT5 uses compressed-tensors 0.18's dense
cross-word packing: element j starts at bit 5*j, with ceil(K*5/32) int32 words per row.
Six-codes-per-word INT5 packing (floor(32/5)) is a different layout and is refused.
Scales must already be exactly binary16-representable; codes and scales are never re-rounded.
INT4 supports source groups 32/64/128/256, with repeated g32 scales or native g128/g256.
INT5 requires group64 and stores the existing native ``q5_g64_fp16`` format.
Row permutations gather complete rows; column permutations must gather complete stored groups.
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


def unpack_codes(packed: torch.Tensor, bits: int = 4, *, columns: int | None = None) -> torch.Tensor:
    """Offset-binary int32 words -> signed int8 codes, preserving the source codes exactly."""
    if bits not in (4, 5):
        raise ValueError(f"packed import supports 4 or 5 bits, not {bits}")
    if packed.dtype != torch.int32 or packed.dim() != 2:
        raise ValueError("packed codes must be a two-dimensional int32 matrix")
    if bits == 4:
        shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
        nibbles = (packed.unsqueeze(-1) >> shifts) & 0xF
        codes = (nibbles.reshape(packed.shape[0], -1) - 8).to(torch.int8)
        return codes if columns is None else codes[:, :columns]
    if columns is None or columns <= 0 or packed.shape[1] != (columns * bits + 31) // 32:
        raise ValueError(
            "INT5 requires dense cross-word packing with ceil(K*5/32) int32 words per row; "
            "six-codes-per-word (floor(32/5)) packing is unsupported"
        )
    positions = torch.arange(columns, dtype=torch.int64, device=packed.device) * bits
    index, shift = positions // 32, positions % 32
    words = packed.to(torch.int64) & 0xFFFFFFFF
    # The next word contributes only for a code straddling a word boundary.
    low = words[:, index] >> shift
    high = words[:, (index + 1).clamp_max(packed.shape[1] - 1)] << (32 - shift)
    return (((low | high) & 0x1F) - 16).to(torch.int8)


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


def packed_matrix_source(
    store: SafetensorsSource,
    prefix: str,
    shape: tuple[int, int],
    group_size: int,
    *,
    row_index: torch.Tensor | None = None,
    bits: int = 4,
    col_index: torch.Tensor | None = None,
    native: bool = False,
) -> LogicalSource:
    """Read exact encoded rows; INT5 requires native group64, INT4 may repeat coarse scales."""
    if bits not in (4, 5):
        raise ValueError(f"{prefix}: supported bit widths are 4 and 5, not {bits}")
    if bits == 5 and (group_size != 64 or not native):
        raise ValueError(f"{prefix}: INT5 requires group64 and native q5_g64_fp16 storage")
    if group_size not in GROUP_SIZES:
        raise ValueError(f"{prefix}: group size {group_size} is not one of {GROUP_SIZES}")
    if bits == 4 and native and group_size not in (32, *NATIVE_GROUP_SIZES):
        raise ValueError(f"{prefix}: native storage exists for group sizes 32, 128 and 256 (no Gemma g64 kernels), not {group_size}")
    width = group_size if native else 32
    target = f"q{bits}_g{width}_fp16"
    n, k = shape
    if k % group_size:
        raise ValueError(f"{prefix}: K {k} is not a multiple of the group size {group_size}")
    packed_k = (k * bits + 31) // 32
    if row_index is not None and (row_index.shape != (n,) or sorted(row_index.tolist()) != list(range(n))):
        raise ValueError(f"{prefix}: the row index must be a permutation of the {n} rows")
    if col_index is not None and (col_index.shape != (k,) or sorted(col_index.tolist()) != list(range(k))):
        raise ValueError(f"{prefix}: the column index must be a permutation of the {k} columns")
    groups = None if col_index is None else column_group_index(col_index, width)
    for extra in ("weight_zero_point", "weight_g_idx"):
        if store.has(f"{prefix}.{extra}"):
            raise ValueError(f"{prefix}: {extra} is present; asymmetric or reordered groups are not importable")
    packed_name, scale_name, shape_name = (f"{prefix}.{part}" for part in ("weight_packed", "weight_scale", "weight_shape"))
    for name, expected, dtype in ((packed_name, (n, packed_k), "I32"), (shape_name, (2,), "I64")):
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            layout = " (INT5 needs dense cross-word ceil(K*5/32) packing)" if bits == 5 else ""
            raise ValueError(f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}{layout}")
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
        words = store.read_flat(packed_name, low * packed_k, high * packed_k).reshape(high - low, packed_k)
        raw = store.read_flat(scale_name, low * (k // group_size), high * (k // group_size)).reshape(
            high - low, k // group_size
        )
        if rows is not None:
            words, raw = words[rows - low], raw[rows - low]
        codes = unpack_codes(words, bits, columns=k).reshape(end - begin, k // width, width)
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

    return LogicalSource(shape, f"{store.path}:{prefix} (packed int{bits} g{group_size})", read, encoded)
