"""tools/eval/roleplay_calibration: quotas, exclusion groups, shingle replacement, corpus hash, check, mix.

Every fixture is synthetic (generated stopword-rich prose in forum-shaped HTML); no forum text is used.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

pytest.importorskip("lxml")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from tools.eval.roleplay_calibration import corpus, mix as mixing, pins  # noqa: E402
from tools.eval.roleplay_calibration.__main__ import main  # noqa: E402
from tools.eval.roleplay_heldout import cleaning  # noqa: E402
from tools.eval.roleplay_heldout import corpus as heldout  # noqa: E402
from tools.eval.roleplay_heldout import pins as dataset  # noqa: E402

CONTENT = (
    "castle river lantern horse silver garden window shadow whisper harbor meadow candle forest mirror "
    "thunder ember violet orchard compass anchor feather"
).split()
QUOTAS = {"forum-x": 3, "forum-e": 3}
PLAN = ["ok"] * 8 + ["short", "short", "title", "minors", "minors", "dup:0", "dup:1", "ok"]


def prose(tag: str, words: int = 70) -> str:
    rng = random.Random(tag)
    fillers = sorted(cleaning.STOPWORDS)
    parts = [rng.choice(fillers) if rng.random() < 0.5 else rng.choice(CONTENT) + rng.choice(["", "s", "ed"]) for _ in range(words)]
    return " ".join(parts).capitalize() + "."


def html(kind: str, posts) -> str:
    if kind == "xenforo":
        body = "".join(
            f'<article class="message message--post" data-author="{a}"><div><article class="message-body">'
            f'<div class="bbWrapper">{t}</div></article></div></article>'
            for a, t in posts
        )
    else:
        body = "".join(
            f'<dl><dt class="postheader"><strong>Re: title</strong><strong>{a}</strong></dt><dd class="postbody">{t}</dd></dl>'
            for a, t in posts
        )
    return f"<html><body>{body}</body></html>"


def thread_posts(tag: str, count: int = 10, minors: bool = False):
    posts = [("Ann" if i % 2 == 0 else "Bob", prose(f"{tag}-{i}")) for i in range(count)]
    if minors:
        posts[4] = (posts[4][0], posts[4][1] + " She was 15 years old.")
    return posts


def make_raw(directory: Path):
    """Two synthetic raw files of 16 threads; returns the sources and the class of every thread URL."""
    directory.mkdir(parents=True, exist_ok=True)
    sources, classes = [], {}
    for stratum, kind in (("forum-x", "xenforo"), ("forum-e", "elliquiy")):
        rows = []
        for index, label in enumerate(PLAN):
            url = f"https://forum.test/{stratum}/threads/{index}/"
            title = "Interest Check: tavern" if label == "title" else f"The {stratum} tale {index}"
            if label.startswith("dup:"):
                posts = thread_posts(f"{stratum}-{label[4:]}")
            else:
                posts = thread_posts(f"{stratum}-{index}", count=5 if label == "short" else 10, minors=label == "minors")
            classes[url] = "ok" if label.startswith("dup") else label
            rows.append({"thread_url": url, "thread_title": title, "contents": html(kind, posts)})
        path = directory / f"{stratum}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        sources.append(dataset.Source(stratum, path.name, kind, path.stat().st_size, heldout.sha256_file(path)))
    return tuple(sources), classes


@pytest.fixture
def raw(tmp_path):
    sources, classes = make_raw(tmp_path / "raw")
    other = tmp_path / "evaluation.txt"
    other.write_text(" ".join(prose(f"unrelated-{i}") for i in range(50)), encoding="utf-8")
    return tmp_path, sources, classes, other


def build(tmp_path, sources, against, name="a", **kwargs):
    kwargs.setdefault("quotas", QUOTAS)
    kwargs.setdefault("reserve", 2)
    kwargs.setdefault("exclusions", {})
    return corpus.build(tmp_path / "raw", tmp_path / name, [against], sources=sources, seed="1", **kwargs)


def urls_of(provenance, stratum=None):
    return [r["thread_url"] for r in provenance["excerpts"] if stratum in (None, r["stratum"])]


# ------------------------------------------------------------------------------- selection


def test_each_stratum_contributes_its_quota_of_threads_that_pass_the_frozen_rules(raw):
    tmp_path, sources, classes, other = raw
    one = build(tmp_path, sources, other, "a", quotas={"forum-x": 3, "forum-e": 5})
    assert [len(urls_of(one, s.stratum)) for s in sources] == [3, 5]
    assert len(set(urls_of(one))) == 8  # one excerpt per thread
    assert {classes[u] for u in urls_of(one)} == {"ok"}  # no short, title-dropped or minors-screened thread
    assert [r["id"] for r in one["excerpts"]] == [f"rpc-{i:03d}" for i in range(8)]
    for r in one["excerpts"]:
        assert 2400 <= r["chars"] <= 3000 and r["authors"] == 2
    tally = one["tallies"]["forum-x"]
    assert tally["accepted"] == 3 + 2  # the quota and the reserve, all survivors of the thread rules
    two = build(tmp_path, sources, other, "b", quotas={"forum-x": 3, "forum-e": 5})
    assert one["corpus_sha256"] == two["corpus_sha256"] and one["source_sha256"] == two["source_sha256"]


def test_the_seed_and_the_exclusions_change_the_selection(raw):
    tmp_path, sources, _, other = raw
    base = build(tmp_path, sources, other, "a")
    reseeded = corpus.build(tmp_path / "raw", tmp_path / "b", [other], sources=sources, seed="2", quotas=QUOTAS, reserve=2, exclusions={})
    assert reseeded["corpus_sha256"] != base["corpus_sha256"]
    dropped = urls_of(base)[0]
    excluded = build(tmp_path, sources, other, "c", exclusions={"earlier": [dropped]})
    assert dropped not in urls_of(excluded) and len(urls_of(excluded)) == 6
    assert excluded["selection"]["excluded_threads"] == {"earlier": 1}
    assert excluded["tallies"][base["excerpts"][0]["stratum"]]["excluded"] == 1


def test_the_default_exclusion_groups_hold_the_development_long_and_fresh_threads():
    groups = corpus.exclusion_groups()
    assert {k: len(v) for k, v in groups.items()} == {"development_short": 42, "long_context": 32, "fresh_heldout": 36}
    every = [u for urls in groups.values() for u in urls]
    assert len(every) == len(set(every)) == 110
    assert corpus.exclusion_groups(["https://x.test/1"])["extra"] == ["https://x.test/1"]


def test_quotas_name_exactly_the_pinned_strata_and_the_default_uses_the_frozen_cleaner():
    assert set(pins.QUOTAS) == {s.stratum for s in dataset.SOURCES}
    assert pins.SEED != dataset.SEED and pins.RESERVE > 0
    assert corpus.select_stratum is heldout.select_stratum and corpus.RULES is cleaning.RULES


# ------------------------------------------------------------------------------- disjointness


def test_a_candidate_sharing_a_shingle_with_an_evaluation_text_is_replaced_by_the_next_thread(raw):
    tmp_path, sources, _, other = raw
    wider = build(tmp_path, sources, other, "wide", quotas={"forum-x": 4, "forum-e": 3})
    base = build(tmp_path, sources, other, "base")
    victim = urls_of(base, "forum-x")[1]
    text = next((tmp_path / "base" / "data" / f"{r['id']}.txt") for r in base["excerpts"] if r["thread_url"] == victim)
    piece = text.read_text(encoding="utf-8")[300:400]
    evaluation = tmp_path / "evaluation-dir"
    evaluation.mkdir()
    (evaluation / "dev.txt").write_text("Opening words. " + piece.upper().replace(" ", "  \n ") + " Closing.", encoding="utf-8")
    (evaluation / "fine.txt").write_text(other.read_text(encoding="utf-8"), encoding="utf-8")
    replaced = corpus.build(tmp_path / "raw", tmp_path / "r", [evaluation], sources=sources, seed="1", quotas=QUOTAS, reserve=2, exclusions={})
    assert victim not in urls_of(replaced) and len(urls_of(replaced, "forum-x")) == 3
    # the reserve thread that takes its place is the fourth survivor of the stratum in hash order
    assert urls_of(replaced, "forum-x") == [u for u in urls_of(wider, "forum-x") if u != victim]
    assert urls_of(replaced, "forum-e") == urls_of(base, "forum-e")
    (dropped,) = replaced["shingle_check"]["dropped"]
    assert dropped["thread_url"] == victim and dropped["files"] == ["dev.txt"]
    assert replaced["shingle_check"]["candidates"] == 10


def test_build_refuses_what_it_cannot_make_disjoint_or_complete(raw):
    tmp_path, sources, _, other = raw
    with pytest.raises(SystemExit, match="--against"):
        corpus.build(tmp_path / "raw", tmp_path / "a", [], sources=sources, quotas=QUOTAS, reserve=2, exclusions={})
    with pytest.raises(SystemExit, match="disagree on the strata"):
        build(tmp_path, sources, other, "b", quotas={"forum-x": 3})
    with pytest.raises(SystemExit, match="lower its quota"):
        build(tmp_path, sources, other, "c", quotas={"forum-x": 3, "forum-e": 12})
    built = build(tmp_path, sources, other, "d")
    assert built
    with pytest.raises(SystemExit, match="not empty"):
        build(tmp_path, sources, other, "d")
    first = tmp_path / "raw" / sources[0].filename
    first.write_bytes(first.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="does not match the pinned file"):
        build(tmp_path, sources, other, "e")


def test_a_tokenizer_count_refuses_a_text_too_short_for_the_chunks_and_reports_the_stride(raw):
    tmp_path, sources, _, other = raw
    words = lambda text: len(text.split())  # noqa: E731
    with pytest.raises(SystemExit, match="raise a quota"):
        build(tmp_path, sources, other, "a", count_tokens=words, chunks=1000, body_tokens=1000)
    built = build(tmp_path, sources, other, "b", count_tokens=words, chunks=4, body_tokens=100)
    tokens = built["tokens"]
    assert tokens["total"] == words((tmp_path / "b" / corpus.SOURCE_NAME).read_text(encoding="utf-8"))
    assert tokens["required"] == 400 and tokens["stride_tokens"] == (tokens["total"] - 100) // 4
    assert build(tmp_path, sources, other, "c")["tokens"] is None


# ------------------------------------------------------------------------------- outputs and check


def test_outputs_join_the_excerpts_hash_them_and_list_the_threads(raw):
    tmp_path, sources, _, other = raw
    built = build(tmp_path, sources, other)
    out = tmp_path / "a"
    texts = [(out / "data" / f"{r['id']}.txt").read_bytes() for r in built["excerpts"]]
    assert (out / corpus.SOURCE_NAME).read_bytes() == b"\n\n".join(texts)
    assert built["corpus_sha256"] == hashlib.sha256(b"".join(t + b"\0" for t in texts)).hexdigest()
    assert built["source_sha256"] == hashlib.sha256(b"\n\n".join(texts)).hexdigest()
    for record, text in zip(built["excerpts"], texts):
        assert record["sha256"] == hashlib.sha256(text).hexdigest() and record["chars"] == len(text.decode("utf-8"))
    listed = json.loads((out / corpus.THREADS_NAME).read_text(encoding="utf-8"))
    assert listed == {"schema": corpus.THREADS_SCHEMA, "corpus_sha256": built["corpus_sha256"], "thread_urls": sorted(urls_of(built))}
    assert built["dataset"]["license"] == "CC-BY-4.0" and built["totals"]["excerpts"] == 6
    assert "6 excerpts" in corpus.describe(built) and built["corpus_sha256"] in corpus.describe(built)


def run_check(corpus_dir, against=(), expected=None):
    lines: list[str] = []
    code = corpus.check(corpus_dir, [Path(p) for p in against], expected, report=lines.append)
    return code, "\n".join(lines)


def test_check_passes_and_finds_modified_excerpts_a_foreign_source_and_shingle_hits(raw):
    tmp_path, sources, _, other = raw
    built = build(tmp_path, sources, other)
    out = tmp_path / "a"
    code, output = run_check(out, [other])
    assert code == 0 and built["corpus_sha256"] in output and "excerpts with a shingle hit: 0" in output
    path = out / "data" / "rpc-001.txt"
    original = path.read_bytes()
    path.write_bytes(original + b" edited")
    code, output = run_check(out)
    assert code == 1 and "rpc-001: file does not match" in output and "corpus SHA-256 differs" in output
    assert "is not the join of the excerpts" in output
    path.write_bytes(original)
    piece = original.decode("utf-8")[200:300]
    (tmp_path / "later.txt").write_text("Words. " + piece + " Words.", encoding="utf-8")
    code, output = run_check(out, [tmp_path / "later.txt"])
    assert code == 1 and "rpc-001 has shingle hits: later.txt" in output
    other_path = out / "data" / "rpc-002.txt"
    kept = other_path.read_bytes()
    other_path.write_bytes(original)
    assert "shares 60-character shingles" in run_check(out)[1]
    other_path.write_bytes(kept)
    assert run_check(out)[0] == 0


def test_check_refuses_an_excluded_thread_and_a_thread_list_that_does_not_match(raw):
    tmp_path, sources, _, other = raw
    build(tmp_path, sources, other)
    out = tmp_path / "a"
    provenance = json.loads((out / "provenance.json").read_text(encoding="utf-8"))
    provenance["excerpts"][0]["thread_url"] = corpus.exclusion_groups()["fresh_heldout"][0]
    (out / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    code, output = run_check(out)
    assert code == 1 and "is an excluded thread (fresh_heldout)" in output and "does not list the threads" in output


def test_check_compares_with_an_expected_or_the_published_hash(raw, monkeypatch):
    tmp_path, sources, _, other = raw
    built = build(tmp_path, sources, other)
    out = tmp_path / "a"
    assert "no published hash applies" in run_check(out)[1]  # a custom selection
    code, output = run_check(out, expected=built["corpus_sha256"])
    assert code == 0 and "matches the expected hash" in output
    assert run_check(out, expected="0" * 64)[0] == 1
    monkeypatch.setattr(pins, "SEED", "1")
    monkeypatch.setattr(pins, "QUOTAS", QUOTAS)
    monkeypatch.setattr(pins, "RESERVE", 2)
    default = corpus.build(
        tmp_path / "raw", tmp_path / "d", [other], sources=sources, seed="1", quotas=QUOTAS, reserve=2,
        exclusions=corpus.exclusion_groups(),
    )
    assert corpus.is_default_selection(default) and not corpus.is_default_selection(built)
    monkeypatch.setattr(pins, "PUBLISHED_CORPUS_SHA256", default["corpus_sha256"])
    assert "matches the published hash" in run_check(tmp_path / "d")[1]
    monkeypatch.setattr(pins, "PUBLISHED_CORPUS_SHA256", "f" * 64)
    assert "differs from the published hash" in run_check(tmp_path / "d")[1]


def test_cli_check_exit_codes_and_later_draws_can_exclude_the_calibration_threads(raw, capsys):
    tmp_path, sources, _, other = raw
    built = build(tmp_path, sources, other)
    out = tmp_path / "a"
    assert main(["check", "--corpus", str(out), "--against", str(other), "--expected", built["corpus_sha256"]]) == 0
    assert main(["check", "--corpus", str(out), "--expected", "0" * 64]) == 1
    assert built["corpus_sha256"] in capsys.readouterr().out
    # the held-out tool reads the thread list as an exclusion file
    assert heldout.read_urls(out / corpus.THREADS_NAME) == sorted(urls_of(built))
    lines = tmp_path / "urls.txt"
    lines.write_text("https://a.test/1\n\n https://a.test/2 \n", encoding="utf-8")
    assert heldout.read_urls(lines) == ["https://a.test/1", "https://a.test/2"]


# ------------------------------------------------------------------------------- mix


def rendered(directory: Path, name: str, text: str, *, chunks=2, ctx=512, tokenizer="t" * 64, prefix=(2, 105)):
    path = directory / f"{name}.txt"
    path.write_text(text, encoding="utf-8", newline="")
    receipt = {
        "schema": mixing.RECEIPT_SCHEMA, "out": str(path), "out_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "tokenizer_sha256": tokenizer, "prefix_ids": list(prefix), "ctx_size": ctx, "chunks": chunks, "skipped": 1,
    }
    path.with_suffix(".json").write_text(json.dumps(receipt), encoding="utf-8")
    return path


def test_mix_restores_the_bos_the_second_file_lost_and_adds_the_chunks(tmp_path):
    first = rendered(tmp_path, "forum", "<bos>x<|turn>a<|turn>b", chunks=3)
    second = rendered(tmp_path, "generic", "y<|turn>c<bos>z<|turn>d", chunks=2)
    out = tmp_path / "mixed.txt"
    receipt = mixing.mix(first, second, out)
    assert out.read_bytes() == b"<bos>x<|turn>a<|turn>b<bos>y<|turn>c<bos>z<|turn>d"
    assert receipt["chunks"] == 5 and receipt["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert [p["chunks"] for p in receipt["parts"]] == [3, 2] and receipt["parts"][0]["skipped"] == 1
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8")) == receipt
    assert main(["mix", "--first", str(first), "--second", str(second), "--out", str(tmp_path / "again.txt")]) == 0


def test_mix_refuses_unequal_renders_changed_files_and_existing_output(tmp_path):
    first = rendered(tmp_path, "a", "one")
    with pytest.raises(SystemExit, match="ctx_size"):
        mixing.mix(first, rendered(tmp_path, "b", "two", ctx=1024), tmp_path / "o1.txt")
    with pytest.raises(SystemExit, match="tokenizer_sha256"):
        mixing.mix(first, rendered(tmp_path, "c", "two", tokenizer="u" * 64), tmp_path / "o2.txt")
    with pytest.raises(SystemExit, match="prefix_ids"):
        mixing.mix(first, rendered(tmp_path, "d", "two", prefix=(2, 106)), tmp_path / "o3.txt")
    with pytest.raises(SystemExit, match="Gemma BOS"):
        mixing.mix(rendered(tmp_path, "e", "x", prefix=(1, 105)), rendered(tmp_path, "f", "y", prefix=(1, 105)), tmp_path / "o4.txt")
    good = rendered(tmp_path, "g", "two")
    good.write_text("tampered", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match the SHA-256"):
        mixing.mix(first, good, tmp_path / "o5.txt")
    (tmp_path / "h.txt").write_text("no receipt", encoding="utf-8")
    with pytest.raises(SystemExit, match="is missing"):
        mixing.mix(first, tmp_path / "h.txt", tmp_path / "o6.txt")
    done = mixing.mix(first, rendered(tmp_path, "i", "ok"), tmp_path / "o7.txt")
    assert done
    with pytest.raises(SystemExit, match="already exists"):
        mixing.mix(first, rendered(tmp_path, "j", "ok"), tmp_path / "o7.txt")
