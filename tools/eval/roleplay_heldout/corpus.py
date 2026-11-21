"""Deterministic selection of the held-out roleplay corpus from the pinned raw forum parquet files.

Per stratum (one parquet file) the threads are ordered by ``sha256(seed + ":" + thread_url)``; each thread in
that order is dropped or kept by the frozen rules of ``cleaning.py`` (excluded thread, title, parse failure,
too few posts, minors-risk screen, no excerpt, duplicate of an earlier excerpt) and the first ``per_stratum``
survivors each contribute one excerpt. No score and no external text is read, so the corpus is a pure
function of the raw files, the seed, the per-stratum count and the exclusion list.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import lxml
import lxml.etree
import pyarrow
import pyarrow.parquet as pq

from . import pins
from .cleaning import MIN_THREAD_POSTS, PARSERS, POST_REASONS, RULES, TITLE_DROP, excerpt_of, minors_risk, post_reason
from .shingles import normal, shingles

EXCLUSIONS = Path(__file__).with_name("excluded_threads.json")
PROVENANCE_SCHEMA = "rp-heldout-provenance/1"
THREAD_TALLY = (
    "excluded",
    "title_drop",
    "parse_error",
    "thread_too_short",
    "minors_filter",
    "no_excerpt",
    "duplicate",
)


@dataclass
class Excerpt:
    stratum: str
    thread_url: str
    thread_title: str
    thread_posts: int
    eligible_posts: int
    first_post_index: int
    posts: int
    authors: int
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_digest(texts: list[bytes]) -> str:
    """SHA-256 over the stream texts in manifest order, each followed by one NUL byte."""
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text)
        digest.update(b"\0")
    return digest.hexdigest()


def load_exclusions(extra: list[str] | None = None) -> dict[str, list[str]]:
    """Thread URLs that must not enter the corpus, by group (the committed list plus optional extra URLs)."""
    groups = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    if extra:
        groups["extra"] = sorted(set(extra))
    return groups


def read_urls(path: Path) -> list[str]:
    """Thread URLs of a text file (one per line) or of a ``calibration_threads.json`` (key ``thread_urls``)."""
    if path.suffix == ".json":
        return list(json.loads(path.read_text(encoding="utf-8"))["thread_urls"])
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_sources(raw: Path, sources) -> list[dict]:
    """Check presence, size and SHA-256 of every pinned raw file; returns their records."""
    records = []
    for source in sources:
        path = raw / source.filename
        if not path.is_file():
            raise SystemExit(f"missing raw file {path}: run `fetch` first")
        size = path.stat().st_size
        digest = sha256_file(path)
        if (source.size is not None and size != source.size) or (source.sha256 is not None and digest != source.sha256):
            raise SystemExit(f"{path} does not match the pinned file (size {size}, sha256 {digest})")
        records.append({"stratum": source.stratum, "name": source.filename, "bytes": size, "sha256": digest})
    return records


def thread_order(path: Path, seed: str) -> tuple[pq.ParquetFile, list[tuple]]:
    """Threads of a raw file as (digest, row group, row, url, title), in selection order."""
    pf = pq.ParquetFile(path)
    locator = []
    for group in range(pf.num_row_groups):
        table = pf.read_row_group(group, columns=["thread_url", "thread_title"])
        for row, (url, title) in enumerate(zip(table["thread_url"].to_pylist(), table["thread_title"].to_pylist())):
            locator.append((hashlib.sha256(f"{seed}:{url}".encode()).hexdigest(), group, row, url, title))
    locator.sort()
    return pf, locator


def select_stratum(source, path: Path, seed: str, per_stratum: int, excluded: set[str], seen: dict[str, str]):
    """The first ``per_stratum`` surviving excerpts of one raw file and its removal tally."""
    pf, locator = thread_order(path, seed)
    tally: dict = {"threads_in_file": len(locator), "examined": 0, **{name: 0 for name in THREAD_TALLY}}
    posts_tally: dict[str, int] = {}
    parse = PARSERS[source.kind]
    kept: list[Excerpt] = []
    contents_group, contents = None, None
    for _, group, row, url, title in locator:
        if len(kept) >= per_stratum:
            break
        tally["examined"] += 1
        if url in excluded:
            tally["excluded"] += 1
            continue
        if TITLE_DROP.search(title or ""):
            tally["title_drop"] += 1
            continue
        if group != contents_group:
            contents, contents_group = pf.read_row_group(group, columns=["contents"]).column("contents"), group
        try:
            posts = parse(contents[row].as_py())
        except Exception:
            tally["parse_error"] += 1
            continue
        eligible = [text for _, text in posts if post_reason(text) is None]
        if len(posts) < MIN_THREAD_POSTS or len(eligible) < MIN_THREAD_POSTS:
            tally["thread_too_short"] += 1
            continue
        if minors_risk("\n".join(text for _, text in posts)):
            tally["minors_filter"] += 1
            continue
        result = excerpt_of(posts, posts_tally)
        if result is None:
            tally["no_excerpt"] += 1
            continue
        text, info = result
        mine = shingles(normal(text))
        if any(s in seen for s in mine):
            tally["duplicate"] += 1
            continue
        for s in mine:
            seen[s] = url
        kept.append(Excerpt(source.stratum, url, title, len(posts), len(eligible), **info, text=text))
    tally["accepted"] = len(kept)
    tally["posts"] = {name: posts_tally.get(name, 0) for name in ("eligible", *POST_REASONS)}
    return kept, tally


def build(
    raw: Path,
    out: Path,
    sources=pins.SOURCES,
    seed: str = pins.SEED,
    per_stratum: int = pins.PER_STRATUM,
    exclusions: dict[str, list[str]] | None = None,
) -> dict:
    """Select, render and write the corpus; returns the provenance record."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    exclusions = load_exclusions() if exclusions is None else exclusions
    excluded = {url for urls in exclusions.values() for url in urls}
    files = verify_sources(raw, sources)
    seen: dict[str, str] = {}
    excerpts: list[Excerpt] = []
    tallies = {}
    for source in sources:
        kept, tallies[source.stratum] = select_stratum(source, raw / source.filename, seed, per_stratum, excluded, seen)
        if len(kept) < per_stratum:
            raise SystemExit(f"stratum {source.stratum} has only {len(kept)} of {per_stratum} eligible threads")
        excerpts.extend(kept)
    (out / "data").mkdir(parents=True, exist_ok=True)
    streams, records, texts = [], [], []
    for index, excerpt in enumerate(excerpts):
        stream_id = f"rpf-{index:02d}"
        data = excerpt.text.encode("utf-8")
        texts.append(data)
        (out / "data" / f"{stream_id}.txt").write_bytes(data)
        streams.append({"id": stream_id, "domain": pins.DOMAIN, "path": f"data/{stream_id}.txt"})
        records.append(
            {
                "id": stream_id,
                "stratum": excerpt.stratum,
                "thread_url": excerpt.thread_url,
                "thread_title": excerpt.thread_title,
                "thread_posts": excerpt.thread_posts,
                "eligible_posts": excerpt.eligible_posts,
                "first_post_index": excerpt.first_post_index,
                "posts": excerpt.posts,
                "authors": excerpt.authors,
                "chars": len(excerpt.text),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    ids = [stream["id"] for stream in streams]
    manifest = {"corpus_id": pins.CORPUS_ID, "streams": streams, "modes": {"quick": ids, "full": ids}}
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "corpus_id": pins.CORPUS_ID,
        "dataset": {
            "repo": pins.DATASET,
            "revision": pins.REVISION,
            "url": pins.DATASET_URL,
            "license": pins.LICENSE,
            "attribution": pins.ATTRIBUTION,
            "files": files,
        },
        "selection": {
            "version": pins.SELECTION_VERSION,
            "seed": seed,
            "per_stratum": per_stratum,
            "order": "threads of each file by sha256(seed + ':' + thread_url); first survivors of the thread rules; "
            "one excerpt per thread; strata in file order; excerpts must share no 60-character shingle",
            "excluded_threads": {group: len(urls) for group, urls in sorted(exclusions.items())},
        },
        "filters": {"version": pins.FILTER_VERSION, "rules": RULES},
        "versions": {
            "python": platform.python_version(),
            "lxml": lxml.__version__,
            "libxml2": ".".join(map(str, lxml.etree.LIBXML_VERSION)),
            "pyarrow": pyarrow.__version__,
        },
        "tallies": tallies,
        "totals": {"streams": len(records), "chars": sum(r["chars"] for r in records)},
        "streams": records,
        "corpus_sha256": corpus_digest(texts),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def load_corpus(corpus: Path) -> tuple[dict, dict, list[bytes]]:
    """Manifest, provenance and the stream texts (manifest order) of a built corpus directory."""
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads((corpus / "provenance.json").read_text(encoding="utf-8"))
    return manifest, provenance, [(corpus / stream["path"]).read_bytes() for stream in manifest["streams"]]


def is_default_selection(provenance: dict) -> bool:
    """True when the corpus was built with the pinned dataset, seed, count, filters and the committed exclusion list only."""
    selection = provenance["selection"]
    return (
        provenance["dataset"]["revision"] == pins.REVISION
        and selection["seed"] == pins.SEED
        and selection["per_stratum"] == pins.PER_STRATUM
        and selection["excluded_threads"] == {group: len(urls) for group, urls in sorted(load_exclusions().items())}
        and provenance["filters"]["version"] == pins.FILTER_VERSION
        and selection["version"] == pins.SELECTION_VERSION
    )


def describe(provenance: dict) -> str:
    """Counts report of a build."""
    lines = [
        f"corpus {provenance['corpus_id']}: {provenance['totals']['streams']} streams, "
        f"{provenance['totals']['chars']:,} characters, sha256 {provenance['corpus_sha256']}",
        f"dataset {provenance['dataset']['repo']}@{provenance['dataset']['revision']} ({provenance['dataset']['license']}); "
        f"seed {provenance['selection']['seed']}, {provenance['selection']['per_stratum']} per stratum; "
        f"excluded threads {provenance['selection']['excluded_threads']}",
        "versions " + ", ".join(f"{k} {v}" for k, v in provenance["versions"].items())
        + f"; filters {provenance['filters']['version']}",
    ]
    for stratum, tally in provenance["tallies"].items():
        removed = ", ".join(f"{name} {tally[name]}" for name in THREAD_TALLY)
        posts = ", ".join(f"{name} {count}" for name, count in tally["posts"].items())
        lines.append(
            f"  {stratum}: {tally['threads_in_file']} threads in file, {tally['examined']} examined, "
            f"{tally['accepted']} kept; removed: {removed}; posts of the threads that reached the excerpt stage: {posts}"
        )
    return "\n".join(lines)
