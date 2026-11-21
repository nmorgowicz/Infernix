"""Join two rendered calibration files into one that llama-imatrix reads as consecutive aligned chunks.

``gemma4_q4 calibration-text`` drops the literal BOS of the first chunk of its file, because llama-imatrix adds one BOS to
the whole file. A second file therefore starts one token short; this restores its BOS so every chunk of the joined file
is exactly ``ctx_size`` tokens and starts at a chunk boundary. Both receipts must describe the same tokenizer, chunk
length and prefix, and match the files on disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

RECEIPT_SCHEMA = "infernix.templated-calibration/v1"
MIX_SCHEMA = "infernix.templated-calibration-mix/v1"
GEMMA_BOS_ID, GEMMA_BOS_TEXT = 2, "<bos>"


def _receipt(path: Path) -> dict:
    receipt_path = path.with_suffix(".json")
    if not receipt_path.is_file():
        raise SystemExit(f"{receipt_path} is missing: render {path} with `gemma4_q4 calibration-text`")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise SystemExit(f"{receipt_path} is not a calibration-text receipt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != receipt["out_sha256"]:
        raise SystemExit(f"{path} does not match the SHA-256 in {receipt_path}")
    return receipt


def mix(first: Path, second: Path, out: Path) -> dict:
    """Write ``first`` followed by ``second`` (with its BOS restored) to ``out`` and a receipt beside it."""
    receipt_path = out.with_suffix(".json")
    if out.exists() or receipt_path.exists():
        raise SystemExit(f"{out} or its receipt already exists; use a fresh path")
    a, b = _receipt(first), _receipt(second)
    for key in ("ctx_size", "prefix_ids", "tokenizer_sha256"):
        if a[key] != b[key]:
            raise SystemExit(f"the receipts disagree on {key}: {a[key]!r} against {b[key]!r}")
    if a["prefix_ids"][0] != GEMMA_BOS_ID:
        raise SystemExit(f"the prefix must start with the Gemma BOS id {GEMMA_BOS_ID}, got {a['prefix_ids'][0]}")
    data = first.read_bytes() + GEMMA_BOS_TEXT.encode("utf-8") + second.read_bytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    receipt = {
        "schema": MIX_SCHEMA,
        "out": str(out.resolve()),
        "out_sha256": hashlib.sha256(data).hexdigest(),
        "ctx_size": a["ctx_size"],
        "prefix_ids": a["prefix_ids"],
        "tokenizer_sha256": a["tokenizer_sha256"],
        "chunks": a["chunks"] + b["chunks"],
        "parts": [
            {"path": str(part.resolve()), "sha256": r["out_sha256"], "chunks": r["chunks"], "skipped": r["skipped"]}
            for part, r in ((first, a), (second, b))
        ],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt
