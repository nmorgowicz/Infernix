"""Integrity, shingle disjointness and published-hash comparison of a built corpus."""

from __future__ import annotations

import hashlib
from pathlib import Path

from . import pins
from .corpus import corpus_digest, is_default_selection, load_corpus
from .shingles import SHINGLE, normal, scan, shingles, text_files


def check(corpus: Path, against: list[Path], expected: str | None = None, report=print) -> int:
    """Verify a corpus directory; returns 0 when every check passes, 1 otherwise.

    * every stream file matches the SHA-256 recorded in ``provenance.json`` and the recorded corpus hash;
    * no two streams share a 60-character shingle (lower-cased, whitespace-collapsed);
    * no stream shares one with any text under ``against`` (a directory contributes its ``*.txt`` files);
    * the corpus hash equals ``expected``, or the published hash when the corpus is the default selection.
    """
    manifest, provenance, texts = load_corpus(corpus)
    problems: list[str] = []
    ids = [stream["id"] for stream in manifest["streams"]]
    recorded = {record["id"]: record["sha256"] for record in provenance["streams"]}
    for stream_id, text in zip(ids, texts):
        if hashlib.sha256(text).hexdigest() != recorded.get(stream_id):
            problems.append(f"{stream_id}: file does not match the SHA-256 in provenance.json")
    digest = corpus_digest(texts)
    if digest != provenance["corpus_sha256"]:
        problems.append("corpus SHA-256 differs from provenance.json")
    report(f"{len(ids)} streams, {sum(len(t.decode('utf-8')) for t in texts):,} characters")
    report(f"corpus sha256 {digest}")

    candidates = {stream_id: text.decode("utf-8") for stream_id, text in zip(ids, texts)}
    owner: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    for stream_id, text in candidates.items():
        for shingle in shingles(normal(text)):
            first = owner.setdefault(shingle, stream_id)
            if first != stream_id:
                duplicates.setdefault(stream_id, set()).add(first)
    for stream_id, others in sorted(duplicates.items()):
        problems.append(f"{stream_id} shares {SHINGLE}-character shingles with {', '.join(sorted(others))}")
    report(f"streams sharing a shingle with another stream: {len(duplicates)}")

    files = text_files(against)
    hits, scanned = scan(candidates, files) if files else ({}, {})
    for path in scanned:
        total = sum(h.get(path, 0) for h in hits.values())
        if total:
            report(f"  {path}: {total} hit positions")
    report(f"normalized characters scanned: {sum(scanned.values()):,}")
    hit_streams = {stream_id: h for stream_id, h in hits.items() if h}
    for stream_id, h in sorted(hit_streams.items()):
        problems.append(f"{stream_id} has shingle hits: " + ", ".join(f"{Path(p).name} {n}" for p, n in sorted(h.items())))
    report(f"texts checked: {len(files)}; streams with a shingle hit: {len(hit_streams)}")

    if expected is None and is_default_selection(provenance):
        expected = pins.PUBLISHED_CORPUS_SHA256
        label = "published"
    else:
        label = "expected"
    if expected is None:
        report("no published hash applies to this corpus" if not is_default_selection(provenance) else
               "no published hash is pinned in this checkout")
    elif digest == expected:
        report(f"corpus hash matches the {label} hash")
    else:
        problems.append(f"corpus hash {digest} differs from the {label} hash {expected}")
    for problem in problems:
        report(f"FAIL {problem}")
    report("check passed" if not problems else f"check failed: {len(problems)} problem(s)")
    return 0 if not problems else 1
