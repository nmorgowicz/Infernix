"""Shingle overlap: every common substring of at least ``SHINGLE`` normalized characters is found."""

from __future__ import annotations

import re
from pathlib import Path

SHINGLE = 60


def normal(text: str) -> str:
    """Lower-cased text with every whitespace run collapsed to one space."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str) -> set[str]:
    """All stride-1 shingles of a normalized text."""
    return {text[i : i + SHINGLE] for i in range(len(text) - SHINGLE + 1)}


def text_files(paths: list[Path]) -> list[Path]:
    """Files named by ``paths``; a directory contributes its ``*.txt`` files recursively, sorted."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*.txt") if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise SystemExit(f"no such file or directory: {path}")
    return files


def scan(candidates: dict[str, str], files: list[Path]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    """Hits of every candidate text in every file.

    Returns ``(hits, scanned)``: ``hits[candidate][file]`` counts the shingle positions of the file that occur in
    the candidate, and ``scanned[file]`` is the file's normalized length. Files are decoded as UTF-8 with
    replacement characters.
    """
    owners: dict[str, list[str]] = {}
    for cid, text in candidates.items():
        for shingle in shingles(normal(text)):
            owners.setdefault(shingle, []).append(cid)
    hits: dict[str, dict[str, int]] = {cid: {} for cid in candidates}
    scanned: dict[str, int] = {}
    for path in files:
        blob = normal(path.read_bytes().decode("utf-8", errors="replace"))
        scanned[str(path)] = len(blob)
        for i in range(len(blob) - SHINGLE + 1):
            found = owners.get(blob[i : i + SHINGLE])
            if found:
                for cid in found:
                    hits[cid][str(path)] = hits[cid].get(str(path), 0) + 1
    return hits, scanned
