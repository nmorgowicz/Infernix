"""tools/eval/roleplay_heldout: cleaning rules, deterministic selection, shingle check, corpus hash, fetch.

Every fixture is synthetic (generated stopword-rich prose in forum-shaped HTML); no forum text is used.
"""

from __future__ import annotations

import functools
import hashlib
import http.server
import json
import random
import threading
from pathlib import Path

import pytest

pytest.importorskip("lxml")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from tools.eval.roleplay_heldout import check as check_module  # noqa: E402
from tools.eval.roleplay_heldout import cleaning, corpus, fetch, pins  # noqa: E402
from tools.eval.roleplay_heldout.__main__ import main  # noqa: E402

CONTENT = (
    "castle river lantern horse silver garden window shadow whisper harbor meadow candle forest mirror "
    "thunder ember violet orchard compass anchor feather"
).split()


def prose(tag: str, words: int = 70) -> str:
    """Deterministic stopword-rich text of about 400 characters that passes every post filter."""
    rng = random.Random(tag)
    fillers = sorted(cleaning.STOPWORDS)
    parts = [rng.choice(fillers) if rng.random() < 0.5 else rng.choice(CONTENT) + rng.choice(["", "s", "ed"]) for _ in range(words)]
    return " ".join(parts).capitalize() + "."


def xenforo_html(posts):
    body = "".join(
        f'<article class="message message--post" data-author="{author}"><div><article class="message-body">'
        f'<div class="bbWrapper">{text}</div></article></div></article>'
        for author, text in posts
    )
    return f"<html><body>{body}</body></html>"


def elliquiy_html(posts):
    body = "".join(
        f'<dl><dt class="postheader"><strong>Re: title</strong><strong>{author}</strong></dt>'
        f'<dd class="postbody">{text}</dd></dl>'
        for author, text in posts
    )
    return f"<html><body>{body}</body></html>"


HTML = {"xenforo": xenforo_html, "elliquiy": elliquiy_html}


# --------------------------------------------------------------------------------------- cleaning


def test_inline_markup_whitespace_is_kept_and_quotes_are_dropped():
    html = (
        '<div class="bbWrapper">Glamour is <em>possibility</em> and<br>next line<blockquote class="bbCodeBlock bbCodeBlock--quote">'
        'quoted words</blockquote><div class="bbCodeBlock--spoiler">hidden words</div><img src="x"> tail'
        "<script>alert(1)</script></div>"
    )
    thread = f'<article class="message message--post" data-author="Ann"><article class="message-body">{html}</article></article>'
    (author, text), = cleaning.parse_xenforo(f"<html><body>{thread}</body></html>")
    assert author == "Ann"
    assert text == "Glamour is possibility and\nnext line tail"
    mangled = html.replace("is <em>", "is<em>")
    thread = f'<article class="message message--post" data-author="Ann"><article class="message-body">{mangled}</article></article>'
    (_, text), = cleaning.parse_xenforo(f"<html><body>{thread}</body></html>")
    assert text.startswith("Glamour ispossibility")  # no space is inserted at an inline tag


def test_elliquiy_pages_authors_and_blockquote():
    page_one = elliquiy_html([("Ann", "First post.<blockquote>quoted</blockquote><br>Second line.")])
    page_two = elliquiy_html([("Bob", "<p>Para one.</p><p>Para two.</p>")])
    posts = cleaning.parse_elliquiy(page_one + "\n" + cleaning.PAGE_MARK + "\n" + page_two)
    assert [author for author, _ in posts] == ["Ann", "Bob"]
    assert posts[0][1] == "First post.\nSecond line."
    assert posts[1][1] == "Para one.\n\nPara two."


def test_clean_normalizes_whitespace_and_invisible_characters():
    text = "  A\u00a0b\u200bc\r\n \n\n\n\nd \t e\u0301 \n"
    assert cleaning.clean(text) == "A bc\n\nd \u00e9"


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("short.", "length"),
        (prose("a") + " ((out of character))", "ooc"),
        (prose("b") + " OOC: hello", "ooc"),
        (prose("c") + " see https://example.test/x", "url"),
        (prose("d") + " roll 2d6 now", "dice"),
        (prose("e") + " \u00e9\u00e8\u00ea" * 12, "non_english"),
        (" ".join(["zzz"] * 80), "non_english"),
        ("\n".join(f"k{i}: v" for i in range(7)) + "\n" + prose("f"), "sheet"),
        (prose("g") + " \u201cquoted\u201d \u2014 dash \u2026", None),
    ],
)
def test_post_filter_reasons(text, reason):
    assert cleaning.post_reason(text) == reason


def test_minors_risk_keyword_screen():
    assert cleaning.minors_risk("she is 15 years old and shy")
    assert cleaning.minors_risk("a loli character")
    assert cleaning.minors_risk("the teacher moaned while the classroom emptied")  # school term with a sexual term
    assert not cleaning.minors_risk("the school bell rang over the river")
    assert not cleaning.minors_risk("he is 25 years old and she moaned")
    assert not cleaning.minors_risk("a long night by the fire with fuck all to do")


def test_title_filter():
    for title in ("Interest Check: dragons", "OOC chat", "Looking for a partner", "Character sheets", "Sign-ups open"):
        assert cleaning.TITLE_DROP.search(title)
    assert not cleaning.TITLE_DROP.search("The Lantern Harbor")


def test_excerpt_skips_setup_posts_needs_two_authors_and_cuts_long_runs():
    tally: dict[str, int] = {}
    posts = [("Ann" if i % 2 == 0 else "Bob", prose(f"x{i}")) for i in range(10)]
    text, info = cleaning.excerpt_of(posts, tally)
    assert info["first_post_index"] == 2 and info["authors"] == 2
    assert text.startswith(posts[2][1]) and 2400 <= len(text) <= 3000
    assert tally == {"eligible": 10}
    # one author only: no excerpt
    assert cleaning.excerpt_of([("Ann", p) for _, p in posts], {}) is None
    # an ineligible post resets the run
    broken = list(posts)
    broken[5] = ("Bob", "too short")
    result = cleaning.excerpt_of(broken, {})
    assert result is None or result[1]["first_post_index"] > 5
    # long posts are cut at the last blank line below 3,000 characters
    long_posts = [("Ann" if i % 2 == 0 else "Bob", prose(f"y{i}", 160)) for i in range(8)]
    text, _ = cleaning.excerpt_of(long_posts, {})
    assert 2000 <= len(text) <= 3000 and text.endswith(".") and "\n\n" in text


# --------------------------------------------------------------------------------------- selection


def thread_posts(tag: str, count: int = 10, minors: bool = False):
    posts = [("Ann" if i % 2 == 0 else "Bob", prose(f"{tag}-{i}")) for i in range(count)]
    if minors:
        posts[4] = (posts[4][0], posts[4][1] + " She was 15 years old.")
    return posts


def make_raw(directory: Path, seed_tags: str = "t"):
    """Two synthetic raw files. Returns sources, the thread class of every URL and its content identity."""
    directory.mkdir(parents=True, exist_ok=True)
    sources, classes, content = [], {}, {}
    for stratum, kind in (("forum-x", "xenforo"), ("forum-e", "elliquiy")):
        rows = []
        plan = ["ok"] * 8 + ["short", "short", "title", "minors", "minors", "dup:0", "dup:1", "ok"]
        for index, label in enumerate(plan):
            url = f"https://forum.test/{stratum}/threads/{index}/"
            title = "Interest Check: tavern" if label == "title" else f"The {stratum} tale {index}"
            tag = f"{seed_tags}-{stratum}-{index}"
            if label.startswith("dup:"):
                posts = thread_posts(f"{seed_tags}-{stratum}-{label[4:]}")
                content[url] = f"{stratum}-{label[4:]}"
            else:
                posts = thread_posts(tag, count=5 if label == "short" else 10, minors=label == "minors")
                content[url] = f"{stratum}-{index}"
            classes[url] = "ok" if label.startswith("dup") else label
            rows.append({"thread_url": url, "thread_title": title, "contents": HTML[kind](posts)})
        path = directory / f"{stratum}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=3)
        sources.append(pins.Source(stratum, path.name, kind, path.stat().st_size, corpus.sha256_file(path)))
    return tuple(sources), classes, content


def oracle(urls, classes, content, seed, k, excluded=()):
    """Expected selection by construction: threads in hash order; the class of each decides its fate."""
    order = sorted(urls, key=lambda u: hashlib.sha256(f"{seed}:{u}".encode()).hexdigest())
    kept, tally, taken = [], {"examined": 0, "excluded": 0, "title_drop": 0, "thread_too_short": 0, "minors_filter": 0, "duplicate": 0}, set()
    for url in order:
        if len(kept) == k:
            break
        tally["examined"] += 1
        if url in excluded:
            tally["excluded"] += 1
        elif classes[url] == "title":
            tally["title_drop"] += 1
        elif classes[url] == "short":
            tally["thread_too_short"] += 1
        elif classes[url] == "minors":
            tally["minors_filter"] += 1
        elif content[url] in taken:
            tally["duplicate"] += 1
        else:
            taken.add(content[url])
            kept.append(url)
    return kept, tally


def test_selection_is_deterministic_and_follows_the_hash_order(tmp_path):
    sources, classes, content = make_raw(tmp_path / "raw")
    first = corpus.build(tmp_path / "raw", tmp_path / "a", sources, seed="1", per_stratum=4, exclusions={})
    second = corpus.build(tmp_path / "raw", tmp_path / "b", sources, seed="1", per_stratum=4, exclusions={})
    assert first["corpus_sha256"] == second["corpus_sha256"]
    for stream in first["streams"]:
        assert (tmp_path / "a" / "data" / f"{stream['id']}.txt").read_bytes() == (tmp_path / "b" / "data" / f"{stream['id']}.txt").read_bytes()
    for source in sources:
        urls = [u for u in classes if f"/{source.stratum}/" in u]
        expected, tally = oracle(urls, classes, content, "1", 4)
        got = [s["thread_url"] for s in first["streams"] if s["stratum"] == source.stratum]
        assert got == expected
        for name, value in tally.items():
            assert first["tallies"][source.stratum][name] == value, name
        assert first["tallies"][source.stratum]["accepted"] == 4
    urls = [s["thread_url"] for s in first["streams"]]
    assert len(urls) == len(set(urls)) == 8  # one excerpt per thread
    manifest = json.loads((tmp_path / "a" / "manifest.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in manifest["streams"]] == [f"rpf-{i:02d}" for i in range(8)]
    assert manifest["modes"]["full"] == [s["id"] for s in manifest["streams"]]
    assert {s["domain"] for s in manifest["streams"]} == {pins.DOMAIN}


def test_seed_changes_the_selection(tmp_path):
    sources, *_ = make_raw(tmp_path / "raw")
    one = corpus.build(tmp_path / "raw", tmp_path / "a", sources, seed="1", per_stratum=4, exclusions={})
    two = corpus.build(tmp_path / "raw", tmp_path / "b", sources, seed="2", per_stratum=4, exclusions={})
    assert one["corpus_sha256"] != two["corpus_sha256"]
    assert [s["thread_url"] for s in one["streams"]] != [s["thread_url"] for s in two["streams"]]


def test_excluded_threads_never_enter_and_the_next_thread_takes_the_place(tmp_path):
    sources, classes, content = make_raw(tmp_path / "raw")
    base = corpus.build(tmp_path / "raw", tmp_path / "a", sources, seed="1", per_stratum=4, exclusions={})
    dropped = base["streams"][0]["thread_url"]
    result = corpus.build(tmp_path / "raw", tmp_path / "b", sources, seed="1", per_stratum=4, exclusions={"earlier": [dropped]})
    urls = [s["thread_url"] for s in result["streams"]]
    assert dropped not in urls and len(urls) == 8
    stratum = base["streams"][0]["stratum"]
    expected, tally = oracle([u for u in classes if f"/{stratum}/" in u], classes, content, "1", 4, excluded={dropped})
    assert [s["thread_url"] for s in result["streams"] if s["stratum"] == stratum] == expected
    assert result["tallies"][stratum]["excluded"] == tally["excluded"] == 1
    assert result["selection"]["excluded_threads"] == {"earlier": 1}


def test_build_refuses_a_stratum_without_enough_threads_a_used_directory_and_a_wrong_raw_file(tmp_path):
    sources, *_ = make_raw(tmp_path / "raw")
    with pytest.raises(SystemExit, match="only"):
        corpus.build(tmp_path / "raw", tmp_path / "a", sources, seed="1", per_stratum=12, exclusions={})
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "file").write_text("x")
    with pytest.raises(SystemExit, match="not empty"):
        corpus.build(tmp_path / "raw", tmp_path / "used", sources, seed="1", per_stratum=2, exclusions={})
    first = tmp_path / "raw" / sources[0].filename
    first.write_bytes(first.read_bytes() + b"x")
    with pytest.raises(SystemExit, match="does not match the pinned file"):
        corpus.build(tmp_path / "raw", tmp_path / "b", sources, seed="1", per_stratum=2, exclusions={})
    first.unlink()
    with pytest.raises(SystemExit, match="fetch"):
        corpus.build(tmp_path / "raw", tmp_path / "c", sources, seed="1", per_stratum=2, exclusions={})


def test_corpus_digest_is_the_nul_terminated_concatenation(tmp_path):
    sources, *_ = make_raw(tmp_path / "raw")
    built = corpus.build(tmp_path / "raw", tmp_path / "a", sources, seed="1", per_stratum=2, exclusions={})
    texts = [(tmp_path / "a" / "data" / f"{s['id']}.txt").read_bytes() for s in built["streams"]]
    assert built["corpus_sha256"] == hashlib.sha256(b"".join(t + b"\0" for t in texts)).hexdigest()
    assert corpus.corpus_digest(texts[::-1]) != built["corpus_sha256"]  # order matters
    changed = [texts[0] + b"x"] + texts[1:]
    assert corpus.corpus_digest(changed) != built["corpus_sha256"]
    assert corpus.corpus_digest([b"ab", b"c"]) != corpus.corpus_digest([b"a", b"bc"])  # boundaries matter
    for stream, text in zip(built["streams"], texts):
        assert stream["sha256"] == hashlib.sha256(text).hexdigest() and stream["chars"] == len(text.decode("utf-8"))


# --------------------------------------------------------------------------------------- shingle check


@pytest.fixture
def built(tmp_path):
    sources, *_ = make_raw(tmp_path / "raw")
    provenance = corpus.build(tmp_path / "raw", tmp_path / "corpus", sources, seed="1", per_stratum=3, exclusions={})
    return tmp_path / "corpus", provenance


def run_check(corpus_dir, against=(), expected=None):
    lines: list[str] = []
    code = check_module.check(corpus_dir, [Path(p) for p in against], expected, report=lines.append)
    return code, "\n".join(lines)


def test_check_passes_on_unrelated_text_and_reports_the_hash(built, tmp_path):
    corpus_dir, provenance = built
    unrelated = tmp_path / "calibration.txt"
    unrelated.write_text(" ".join(prose(f"unrelated-{i}") for i in range(50)), encoding="utf-8")
    code, output = run_check(corpus_dir, [unrelated])
    assert code == 0, output
    assert provenance["corpus_sha256"] in output and "streams with a shingle hit: 0" in output


def test_check_finds_a_shared_substring_across_case_and_whitespace(built, tmp_path):
    corpus_dir, _ = built
    stream = (corpus_dir / "data" / "rpf-04.txt").read_text(encoding="utf-8")
    start = len(stream) // 2
    piece = stream[start : start + 90]
    mangled = piece.upper().replace(" ", "  \n ")
    directory = tmp_path / "texts"
    directory.mkdir()
    (directory / "other.txt").write_text("Unrelated opening words. " + mangled + " Closing words.", encoding="utf-8")
    code, output = run_check(corpus_dir, [directory])
    assert code == 1
    assert "rpf-04 has shingle hits: other.txt" in output and "rpf-00 has" not in output
    # a common substring shorter than a shingle is not a hit
    (directory / "other.txt").write_text(piece[:50], encoding="utf-8")
    code, output = run_check(corpus_dir, [directory])
    assert code == 0, output


def test_check_detects_modified_streams_and_streams_sharing_text(built, tmp_path):
    corpus_dir, provenance = built
    path = corpus_dir / "data" / "rpf-01.txt"
    original = path.read_bytes()
    path.write_bytes(original + b" edited")
    code, output = run_check(corpus_dir)
    assert code == 1 and "rpf-01: file does not match" in output and "corpus SHA-256 differs" in output
    path.write_bytes(original)
    # a stream that repeats another stream's text is refused
    other = corpus_dir / "data" / "rpf-02.txt"
    other_original = other.read_bytes()
    other.write_bytes(original)
    code, output = run_check(corpus_dir)
    assert code == 1 and "shares 60-character shingles" in output
    other.write_bytes(other_original)
    assert run_check(corpus_dir)[0] == 0


def test_check_compares_with_an_expected_or_the_published_hash(built, monkeypatch):
    corpus_dir, provenance = built
    digest = provenance["corpus_sha256"]
    code, output = run_check(corpus_dir, expected=digest)
    assert code == 0 and "matches the expected hash" in output
    code, output = run_check(corpus_dir, expected="0" * 64)
    assert code == 1 and "differs from the expected hash" in output
    assert "no published hash applies" in run_check(corpus_dir)[1]  # per-stratum count 3 is a custom selection


def test_published_hash_applies_to_the_default_selection_only(tmp_path, monkeypatch):
    sources, *_ = make_raw(tmp_path / "raw")
    monkeypatch.setattr(pins, "PER_STRATUM", 4)
    provenance = corpus.build(tmp_path / "raw", tmp_path / "corpus", sources, seed=pins.SEED, per_stratum=4)
    assert corpus.is_default_selection(provenance)
    monkeypatch.setattr(pins, "PUBLISHED_CORPUS_SHA256", provenance["corpus_sha256"])
    code, output = run_check(tmp_path / "corpus")
    assert code == 0 and "matches the published hash" in output
    monkeypatch.setattr(pins, "PUBLISHED_CORPUS_SHA256", "f" * 64)
    code, output = run_check(tmp_path / "corpus")
    assert code == 1 and "differs from the published hash" in output
    other = corpus.build(tmp_path / "raw", tmp_path / "other", sources, seed=pins.SEED, per_stratum=4, exclusions={"x": []})
    assert not corpus.is_default_selection(other)


def test_cli_check_exit_codes(tmp_path, capsys):
    sources, *_ = make_raw(tmp_path / "raw")
    provenance = corpus.build(tmp_path / "raw", tmp_path / "c", sources, seed="9", per_stratum=3, exclusions={})
    assert main(["check", "--corpus", str(tmp_path / "c"), "--expected", provenance["corpus_sha256"]]) == 0
    assert main(["check", "--corpus", str(tmp_path / "c"), "--expected", "0" * 64]) == 1
    assert provenance["corpus_sha256"] in capsys.readouterr().out


# --------------------------------------------------------------------------------------- fetch


@pytest.fixture
def server(tmp_path):
    served = tmp_path / "served"
    calls: list[str] = []

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            calls.append(self.path)
            super().do_GET()

    handler = functools.partial(Handler, directory=str(served))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield served, f"http://127.0.0.1:{httpd.server_address[1]}", calls
    httpd.shutdown()
    thread.join()


def test_fetch_downloads_verifies_and_skips_present_files(tmp_path, server):
    served, endpoint, calls = server
    data = b"synthetic parquet bytes" * 1000
    location = served / "datasets" / "owner" / "name" / "resolve" / "rev1"
    location.mkdir(parents=True)
    (location / "part one.parquet").write_bytes(data)
    good = pins.Source("s", "part one.parquet", "xenforo", len(data), hashlib.sha256(data).hexdigest())
    out = tmp_path / "raw"
    assert fetch.fetch(out, (good,), endpoint, "owner/name", "rev1") == {"part one.parquet": "downloaded"}
    assert (out / "part one.parquet").read_bytes() == data and not list(out.glob("*.part"))
    assert fetch.fetch(out, (good,), endpoint, "owner/name", "rev1") == {"part one.parquet": "present"}
    assert len(calls) == 1 and calls[0].endswith("/rev1/part%20one.parquet")
    # a corrupt local copy is replaced
    (out / "part one.parquet").write_bytes(b"damaged")
    assert fetch.fetch(out, (good,), endpoint, "owner/name", "rev1") == {"part one.parquet": "downloaded"}
    assert (out / "part one.parquet").read_bytes() == data


def test_fetch_discards_a_download_that_does_not_match_the_pin(tmp_path, server):
    served, endpoint, _ = server
    location = served / "datasets" / "owner" / "name" / "resolve" / "rev1"
    location.mkdir(parents=True)
    (location / "f.parquet").write_bytes(b"served bytes")
    bad = pins.Source("s", "f.parquet", "xenforo", 12, "0" * 64)
    with pytest.raises(SystemExit, match="discarded"):
        fetch.fetch(tmp_path / "raw", (bad,), endpoint, "owner/name", "rev1")
    assert not list((tmp_path / "raw").glob("f.parquet*"))


def test_pinned_sources_name_the_six_strata_with_hashes():
    assert pins.REVISION and len(pins.REVISION) == 40
    assert [s.stratum for s in pins.SOURCES] == [
        "elliquiy-nonadult", "elliquiy-oneshots", "bluemoon-group", "bluemoon-fanbased-1x1", "rpnation-quests", "rpnation-1x1",
    ]
    assert all(s.sha256 and len(s.sha256) == 64 and s.size for s in pins.SOURCES)
    groups = corpus.load_exclusions()
    assert len(groups["development_short"]) == 42 and len(groups["long_context"]) == 32
    assert not set(groups["development_short"]) & set(groups["long_context"])
