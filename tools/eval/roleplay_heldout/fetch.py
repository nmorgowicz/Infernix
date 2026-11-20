"""Download the pinned raw parquet files at the pinned dataset revision and verify their SHA-256."""

from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from pathlib import Path

from . import pins

BLOCK = 1 << 22


def url_of(source, endpoint: str, repo: str, revision: str) -> str:
    return f"{endpoint}/datasets/{repo}/resolve/{revision}/{urllib.parse.quote(source.filename)}"


def fetch_file(source, out: Path, endpoint: str, repo: str, revision: str) -> str:
    """Make ``out/<filename>`` match the pinned file; returns "present" or "downloaded"."""
    target = out / source.filename
    if target.is_file() and _matches(target, source):
        return "present"
    out.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url_of(source, endpoint, repo, revision)) as response, part.open("wb") as handle:
        for block in iter(lambda: response.read(BLOCK), b""):
            handle.write(block)
            digest.update(block)
            size += len(block)
    if (source.size is not None and size != source.size) or (source.sha256 is not None and digest.hexdigest() != source.sha256):
        part.unlink()
        raise SystemExit(
            f"{source.filename}: downloaded {size} bytes with sha256 {digest.hexdigest()}, "
            f"pinned {source.size} bytes with sha256 {source.sha256}; the file was discarded"
        )
    part.replace(target)
    return "downloaded"


def _matches(path: Path, source) -> bool:
    if source.size is not None and path.stat().st_size != source.size:
        return False
    if source.sha256 is None:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest() == source.sha256


def fetch(out: Path, sources=pins.SOURCES, endpoint: str = pins.ENDPOINT, repo: str = pins.DATASET,
          revision: str = pins.REVISION) -> dict[str, str]:
    """Fetch every source; returns filename -> "present" | "downloaded"."""
    return {source.filename: fetch_file(source, out, endpoint, repo, revision) for source in sources}
