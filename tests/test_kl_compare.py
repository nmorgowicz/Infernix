"""tools/bench/kv_eval/kl_compare.py: window and stream-cluster paired bootstrap (no GPU)."""

import json

import pytest

from tools.bench.kv_eval import kl_compare


def kl(kl_sums, streams, domains=None):
    """A `--full-vocab-reference` comparison report with one scored window per entry (count 1)."""
    windows = [{"stream": streams[i], "domain": (domains or {}).get(streams[i], "d"), "begin": i, "count": 1,
                "kl_sum": value, "delta_nll_sum": 0.0} for i, value in enumerate(kl_sums)]
    tails = {key: 1.0 for key in ("p95_kl", "p99_kl", "p999_kl", "top1_disagreement")}
    return {"protocol": {"context": 512}, "reference_artifact": {"gguf": "ref"}, "reference_cache": {"sha": "c"},
            "statistics": {"windows": windows, "overall": tails}}


STREAMS = ["a"] * 6 + ["b"] * 6


def test_stream_resampling_moves_a_stream_as_one_unit():
    baseline = kl([1.0] * 12, STREAMS)
    candidate = kl([0.0] * 6 + [2.0] * 6, STREAMS)  # stream a drops to zero KL, stream b doubles
    window = kl_compare.compare(baseline, candidate, 2000, 7, None, "window")["overall"]
    stream = kl_compare.compare(baseline, candidate, 2000, 7, None, "stream")["overall"]
    assert window["kl_ratio"] == stream["kl_ratio"] == pytest.approx(1.0)
    # with two independent streams every resample is {a,a}, {a,b} or {b,b}: ratios 0, 1 or 2
    assert stream["clusters"] == 2 and stream["windows"] == 12
    assert stream["kl_ratio_ci95"] == [0.0, 2.0]
    # twelve independent windows almost never draw all six of one kind, so the window interval is narrower
    assert window["clusters"] == 12
    assert 0.0 < window["kl_ratio_ci95"][0] < 1.0 < window["kl_ratio_ci95"][1] < 2.0


def test_constant_ratio_has_a_degenerate_interval_in_both_modes():
    baseline = kl([1.0, 2.0, 3.0, 4.0], ["a", "a", "b", "c"])
    candidate = kl([0.5, 1.0, 1.5, 2.0], ["a", "a", "b", "c"])
    for mode in ("window", "stream"):
        overall = kl_compare.compare(baseline, candidate, 500, 3, None, mode)["overall"]
        assert overall["kl_ratio"] == pytest.approx(0.5)
        assert overall["kl_ratio_ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_domains_are_resampled_within_their_own_streams():
    streams = ["a", "a", "b", "b", "c", "c"]
    domains = {"a": "x", "b": "x", "c": "y"}
    baseline = kl([1.0] * 6, streams, domains)
    candidate = kl([1.0, 1.0, 3.0, 3.0, 2.0, 2.0], streams, domains)
    result = kl_compare.compare(baseline, candidate, 500, 3, None, "stream")
    by_domain = {entry["domain"]: entry for entry in result["domains"]}
    assert by_domain["x"]["clusters"] == 2 and by_domain["y"]["clusters"] == 1
    assert by_domain["x"]["kl_ratio"] == pytest.approx(2.0)
    assert by_domain["y"]["kl_ratio_ci95"] == [pytest.approx(2.0), pytest.approx(2.0)]


def test_default_window_resampling_is_unchanged_by_the_cluster_option():
    baseline = kl([1.0, 2.0, 3.0, 4.0, 5.0], list("abcde"))
    candidate = kl([1.5, 1.0, 4.0, 3.0, 6.0], list("abcde"))
    default = kl_compare.compare(baseline, candidate, 300, 11, None)
    singleton_streams = kl_compare.compare(baseline, candidate, 300, 11, None, "stream")
    assert default["cluster"] == "window"
    # every stream holds one window, so both modes draw the identical resamples
    assert default["overall"] == singleton_streams["overall"]


def test_unknown_cluster_is_refused_and_cli_writes_the_mode(tmp_path):
    baseline, candidate = kl([1.0, 1.0], ["a", "b"]), kl([2.0, 2.0], ["a", "b"])
    with pytest.raises(SystemExit):
        kl_compare.compare(baseline, candidate, 10, 1, None, "thread")
    paths = {}
    for name, report in (("base", baseline), ("cand", candidate)):
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps({"kl": report}), encoding="utf-8")
    out = tmp_path / "paired.json"
    kl_compare.main(["--baseline", str(paths["base"]), "--candidate", str(paths["cand"]), "--cluster", "stream",
                     "--resamples", "50", "--out", str(out)])
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["cluster"] == "stream" and written["overall"]["kl_ratio"] == pytest.approx(2.0)
