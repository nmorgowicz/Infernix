"""Deterministic selection of the roleplay calibration text from the pinned raw forum parquet files.

The cleaning rules, the minors-risk screen, the per-thread excerpt cut, the hash order of threads and the duplicate
rule are those of ``tools.eval.roleplay_heldout`` (``select_stratum``): one excerpt per surviving thread, per raw file
a quota of them in ``sha256(seed + ":" + thread_url)`` order. What differs is the seed, the quotas, the exclusion list
(the held-out tool's development and long-context threads plus the fresh held-out threads) and the disjointness
check: a candidate that shares a 60-character shingle with any evaluation or earlier calibration text is dropped and
the next thread in hash order takes its place. The result is a pure function of the raw files, the seed, the quotas,
the reserve, the exclusion list and the texts checked against.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Callable
from pathlib import Path

import lxml
import lxml.etree
import pyarrow

from tools.eval.roleplay_heldout import pins as dataset
from tools.eval.roleplay_heldout.cleaning import RULES
from tools.eval.roleplay_heldout.corpus import (
    THREAD_TALLY,
    corpus_digest,
    load_exclusions,
    select_stratum,
    verify_sources,
)
from tools.eval.roleplay_heldout.shingles import SHINGLE, normal, scan, shingles, text_files

from . import pins

FRESH = Path(__file__).with_name("fresh_heldout_threads.json")
PROVENANCE_SCHEMA = "rp-calibration-provenance/1"
THREADS_SCHEMA = "rp-calibration-threads/1"
SOURCE_NAME = "calibration-source.txt"
THREADS_NAME = "calibration_threads.json"
JOIN = b"\n\n"


def exclusion_groups(extra: list[str] | None = None) -> dict[str, list[str]]:
    """Thread URLs no calibration thread may be: the held-out tool's groups, the fresh held-out set and any extras."""
    groups = load_exclusions(extra)
    groups.update(json.loads(FRESH.read_text(encoding="utf-8")))
    return groups


def build(
    raw: Path,
    out: Path,
    against: list[Path],
    *,
    sources=dataset.SOURCES,
    seed: str = pins.SEED,
    quotas: dict[str, int] = pins.QUOTAS,
    reserve: int = pins.RESERVE,
    exclusions: dict[str, list[str]] | None = None,
    count_tokens: Callable[[str], int] | None = None,
    chunks: int = 400,
    body_tokens: int = 495,
) -> dict:
    """Select, check against the evaluation texts, write the calibration text; returns the provenance record.

    ``count_tokens`` (the target tokenizer, optional) makes the build refuse a text too short for ``chunks`` chunks of
    ``body_tokens`` body tokens each, which is what ``gemma4_q4 calibration-text`` needs.
    """
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory is not empty: {out}")
    files = text_files(against)
    if not files:
        raise SystemExit("pass every evaluation and earlier calibration text with --against (a file, or a directory of *.txt)")
    unknown = {source.stratum for source in sources} ^ set(quotas)
    if unknown:
        raise SystemExit(f"the quotas and the sources disagree on the strata {sorted(unknown)}")
    exclusions = exclusion_groups() if exclusions is None else exclusions
    excluded = {url for urls in exclusions.values() for url in urls}
    dataset_files = verify_sources(raw, sources)

    seen: dict[str, str] = {}
    pool, tallies = {}, {}
    for source in sources:
        pool[source.stratum], tallies[source.stratum] = select_stratum(
            source, raw / source.filename, seed, quotas[source.stratum] + reserve, excluded, seen
        )
    candidates = {e.thread_url: e.text for kept in pool.values() for e in kept}
    hits, scanned = scan(candidates, files)

    excerpts, dropped = [], []
    for source in sources:
        clean = [e for e in pool[source.stratum] if not hits[e.thread_url]]
        dropped += [
            {"stratum": e.stratum, "thread_url": e.thread_url, "files": sorted(Path(p).name for p in hits[e.thread_url])}
            for e in pool[source.stratum]
            if hits[e.thread_url]
        ]
        if len(clean) < quotas[source.stratum]:
            raise SystemExit(
                f"stratum {source.stratum} has only {len(clean)} of {quotas[source.stratum]} threads that pass the rules "
                f"and the shingle check; lower its quota"
            )
        excerpts += clean[: quotas[source.stratum]]

    texts = [e.text.encode("utf-8") for e in excerpts]
    source_text = JOIN.join(texts)
    tokens = None
    if count_tokens is not None:
        total, required = count_tokens(source_text.decode("utf-8")), chunks * body_tokens
        if total < required:
            raise SystemExit(f"the text has {total} tokens; {chunks} chunks of {body_tokens} need {required}: raise a quota")
        tokens = {
            "total": total,
            "required": required,
            "chunks": chunks,
            "body_tokens": body_tokens,
            "stride_tokens": (total - body_tokens) // chunks,
        }
    (out / "data").mkdir(parents=True, exist_ok=True)
    records = []
    for index, (excerpt, data) in enumerate(zip(excerpts, texts)):
        excerpt_id = f"rpc-{index:03d}"
        (out / "data" / f"{excerpt_id}.txt").write_bytes(data)
        records.append(
            {
                "id": excerpt_id,
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
    (out / SOURCE_NAME).write_bytes(source_text)
    digest = corpus_digest(texts)
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "corpus_id": pins.CORPUS_ID,
        "dataset": {
            "repo": dataset.DATASET,
            "revision": dataset.REVISION,
            "url": dataset.DATASET_URL,
            "license": dataset.LICENSE,
            "attribution": dataset.ATTRIBUTION,
            "files": dataset_files,
        },
        "selection": {
            "version": pins.SELECTION_VERSION,
            "seed": seed,
            "quotas": quotas,
            "reserve": reserve,
            "order": "threads of each file by sha256(seed + ':' + thread_url); survivors of the held-out tool's thread "
            "rules; one excerpt per thread; strata in file order; threads whose excerpt shares a "
            f"{SHINGLE}-character shingle with a checked text are replaced by the next survivor",
            "excluded_threads": {group: len(urls) for group, urls in sorted(exclusions.items())},
        },
        "filters": {"version": dataset.FILTER_VERSION, "rules": RULES},
        "versions": {
            "python": platform.python_version(),
            "lxml": lxml.__version__,
            "libxml2": ".".join(map(str, lxml.etree.LIBXML_VERSION)),
            "pyarrow": pyarrow.__version__,
        },
        "tallies": tallies,
        "shingle_check": {
            "texts": {path: size for path, size in sorted(scanned.items())},
            "normalized_characters": sum(scanned.values()),
            "candidates": len(candidates),
            "dropped": dropped,
        },
        "tokens": tokens,
        "totals": {"excerpts": len(records), "chars": sum(r["chars"] for r in records)},
        "excerpts": records,
        "source_sha256": hashlib.sha256(source_text).hexdigest(),
        "corpus_sha256": digest,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (out / THREADS_NAME).write_text(
        json.dumps(
            {"schema": THREADS_SCHEMA, "corpus_sha256": digest, "thread_urls": sorted(r["thread_url"] for r in records)},
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return provenance


def is_default_selection(provenance: dict) -> bool:
    """True when the corpus was built with the pinned dataset, seed, quotas, reserve, filters and exclusion groups only."""
    selection = provenance["selection"]
    return (
        provenance["dataset"]["revision"] == dataset.REVISION
        and selection["seed"] == pins.SEED
        and selection["quotas"] == pins.QUOTAS
        and selection["reserve"] == pins.RESERVE
        and selection["excluded_threads"] == {group: len(urls) for group, urls in sorted(exclusion_groups().items())}
        and provenance["filters"]["version"] == dataset.FILTER_VERSION
        and selection["version"] == pins.SELECTION_VERSION
    )


def describe(provenance: dict) -> str:
    """Counts report of a build."""
    totals, check = provenance["totals"], provenance["shingle_check"]
    lines = [
        f"corpus {provenance['corpus_id']}: {totals['excerpts']} excerpts, {totals['chars']:,} characters, "
        f"corpus sha256 {provenance['corpus_sha256']}, source sha256 {provenance['source_sha256']}",
        f"dataset {provenance['dataset']['repo']}@{provenance['dataset']['revision']} ({provenance['dataset']['license']}); "
        f"seed {provenance['selection']['seed']}, reserve {provenance['selection']['reserve']}; "
        f"excluded threads {provenance['selection']['excluded_threads']}",
        "versions " + ", ".join(f"{k} {v}" for k, v in provenance["versions"].items())
        + f"; filters {provenance['filters']['version']}",
        f"shingle check: {len(check['texts'])} texts, {check['normalized_characters']:,} normalized characters, "
        f"{check['candidates']} candidates, {len(check['dropped'])} dropped",
    ]
    if provenance["tokens"]:
        t = provenance["tokens"]
        lines.append(
            f"tokens: {t['total']:,} (need {t['required']:,} for {t['chunks']} chunks of {t['body_tokens']}); "
            f"chunk stride {t['stride_tokens']} tokens"
        )
    for stratum, tally in provenance["tallies"].items():
        removed = ", ".join(f"{name} {tally[name]}" for name in THREAD_TALLY)
        kept = sum(1 for r in provenance["excerpts"] if r["stratum"] == stratum)
        lines.append(
            f"  {stratum}: {tally['threads_in_file']} threads in file, {tally['examined']} examined, "
            f"{tally['accepted']} survived (quota {provenance['selection']['quotas'][stratum]}, kept {kept}); removed: {removed}"
        )
    return "\n".join(lines)


def check(corpus: Path, against: list[Path], expected: str | None = None, report=print) -> int:
    """Verify a built calibration text; returns 0 when every check passes, 1 otherwise.

    * every excerpt file matches the SHA-256 in ``provenance.json``; the source file is their join; the corpus hash matches;
    * no thread is listed twice or sits in an exclusion group, and ``calibration_threads.json`` lists exactly the threads;
    * no two excerpts share a 60-character shingle, and none shares one with any text under ``against``;
    * the corpus hash equals ``expected``, or the published hash when the corpus is the default selection.
    """
    provenance = json.loads((corpus / "provenance.json").read_text(encoding="utf-8"))
    records = provenance["excerpts"]
    texts = [(corpus / "data" / f"{r['id']}.txt").read_bytes() for r in records]
    problems: list[str] = []
    for record, text in zip(records, texts):
        if hashlib.sha256(text).hexdigest() != record["sha256"]:
            problems.append(f"{record['id']}: file does not match the SHA-256 in provenance.json")
    digest = corpus_digest(texts)
    if digest != provenance["corpus_sha256"]:
        problems.append("corpus SHA-256 differs from provenance.json")
    source = (corpus / SOURCE_NAME).read_bytes()
    if source != JOIN.join(texts) or hashlib.sha256(source).hexdigest() != provenance["source_sha256"]:
        problems.append(f"{SOURCE_NAME} is not the join of the excerpts recorded in provenance.json")
    report(f"{len(records)} excerpts, {sum(len(t.decode('utf-8')) for t in texts):,} characters")
    report(f"corpus sha256 {digest}")

    urls = [r["thread_url"] for r in records]
    if len(set(urls)) != len(urls):
        problems.append("a thread contributes more than one excerpt")
    excluded = {url: group for group, listed in exclusion_groups().items() for url in listed}
    for url in sorted(set(urls) & set(excluded)):
        problems.append(f"{url} is an excluded thread ({excluded[url]})")
    report(f"threads: {len(set(urls))}; in an exclusion group: {len(set(urls) & set(excluded))}")
    listed = json.loads((corpus / THREADS_NAME).read_text(encoding="utf-8"))
    if listed.get("thread_urls") != sorted(urls) or listed.get("corpus_sha256") != provenance["corpus_sha256"]:
        problems.append(f"{THREADS_NAME} does not list the threads and hash of provenance.json")

    candidates = {r["id"]: text.decode("utf-8") for r, text in zip(records, texts)}
    owner: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    for excerpt_id, text in candidates.items():
        for shingle in shingles(normal(text)):
            first = owner.setdefault(shingle, excerpt_id)
            if first != excerpt_id:
                duplicates.setdefault(excerpt_id, set()).add(first)
    for excerpt_id, others in sorted(duplicates.items()):
        problems.append(f"{excerpt_id} shares {SHINGLE}-character shingles with {', '.join(sorted(others))}")
    report(f"excerpts sharing a shingle with another excerpt: {len(duplicates)}")

    files = text_files(against)
    hits, scanned = scan(candidates, files) if files else ({}, {})
    report(f"normalized characters scanned: {sum(scanned.values()):,}")
    hit_excerpts = {excerpt_id: h for excerpt_id, h in hits.items() if h}
    for excerpt_id, h in sorted(hit_excerpts.items()):
        problems.append(f"{excerpt_id} has shingle hits: " + ", ".join(f"{Path(p).name} {n}" for p, n in sorted(h.items())))
    report(f"texts checked: {len(files)}; excerpts with a shingle hit: {len(hit_excerpts)}")

    if expected is None and is_default_selection(provenance):
        expected, label = pins.PUBLISHED_CORPUS_SHA256, "published"
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

