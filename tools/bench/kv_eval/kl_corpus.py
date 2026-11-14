# Ported from ninfer.spo tools/bench/kv_eval; instrument definitions retained.
"""Deterministic head-truncated subset of a perplexity corpus for exact full-vocabulary KL.

The KL reference cache stores one F32 row of vocab floats per scored position (about 1 MiB at
vocab 262144), so the full 1.09M-token corpus would need ~1 TB. This tool cuts the first N
characters of every stream (UTF-8 safe), writes a standard corpus manifest that
``infernix-perplexity`` accepts, and records the source manifest and every output file hash in
``subset.json``. Scored positions per stream are (tokens - 1); the token counts are a runtime
property of the artifact's tokenizer, so size the cut from a ``--dry-run`` of the real run.

    python -m tools.bench.kv_eval.kl_corpus --source <manifest.json> --out <dir> \
        [--chars english_reference=4500 ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# About 1000 tokens per stream for Gemma's tokenizer on these domains (measured on the
# ninfer-ppl-1m-v1 corpus is not available here: these are starting sizes, not guarantees).
DEFAULT_CHARS = {
    "english_reference": 4500,
    "english_long_form": 4500,
    "chinese_reference": 1500,
    "ninfer_code": 3500,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_chars(items: list[str]) -> dict[str, int]:
    chars = dict(DEFAULT_CHARS)
    for item in items:
        domain, separator, value = item.partition("=")
        if not separator or not value.isdigit() or int(value) < 0:
            raise SystemExit(f"--chars expects domain=non-negative-integer, got {item!r}")
        chars[domain] = int(value)
    return chars


def build(source: Path, out: Path, chars: dict[str, int], scale: float, corpus_suffix: str) -> dict:
    manifest = json.loads(source.read_text(encoding="utf-8"))
    base = source.parent
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    (out / "data").mkdir(parents=True, exist_ok=True)
    streams = []
    records = []
    for stream in manifest["streams"]:
        domain = stream["domain"]
        if chars.get(domain, 0) == 0:
            continue
        budget = max(1, round(chars[domain] * scale))
        text = (base / stream["path"]).read_text(encoding="utf-8")[:budget]
        if not text.strip():
            raise SystemExit(f"stream {stream['id']} is empty after truncation")
        relative = f"data/{stream['id']}.txt"
        target = out / relative
        target.write_text(text, encoding="utf-8", newline="")
        streams.append({"id": stream["id"], "domain": domain, "path": relative})
        records.append(
            {
                "id": stream["id"],
                "domain": domain,
                "source_path": stream["path"],
                "source_sha256": sha256_file(base / stream["path"]),
                "characters": len(text),
                "sha256": sha256_file(target),
            }
        )
    ids = [stream["id"] for stream in streams]
    subset_id = f"{manifest['corpus_id']}-kl{corpus_suffix}"
    (out / "manifest.json").write_text(
        json.dumps(
            {"corpus_id": subset_id, "streams": streams, "modes": {"quick": ids, "full": ids}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "ninfer.kl-corpus/1",
        "corpus_id": subset_id,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": sha256_file(source),
        "characters_per_domain": chars,
        "scale": scale,
        "manifest_sha256": sha256_file(out / "manifest.json"),
        "streams": records,
    }
    (out / "subset.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, type=Path, help="source corpus manifest.json")
    parser.add_argument("--out", required=True, type=Path, help="new empty output directory")
    parser.add_argument("--chars", action="append", default=[], metavar="DOMAIN=N",
                        help="characters kept per stream of a domain (defaults in DEFAULT_CHARS)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every budget (e.g. 4 for a ~4x larger subset)")
    parser.add_argument("--suffix", default="", help="appended to the corpus id (e.g. -x4)")
    args = parser.parse_args(argv)
    if args.scale <= 0:
        raise SystemExit("--scale must be positive")
    receipt = build(args.source, args.out, parse_chars(args.chars), args.scale, args.suffix)
    print(json.dumps({"corpus_id": receipt["corpus_id"], "streams": len(receipt["streams"]),
                      "characters": sum(item["characters"] for item in receipt["streams"]),
                      "manifest": str(args.out / "manifest.json")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
