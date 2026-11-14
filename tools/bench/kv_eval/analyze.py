# Ported from ninfer.spo tools/bench/kv_eval; instrument definitions retained.
#!/usr/bin/env python3
"""Compare KV/quantization arms from infernix-perplexity reports and --save-top-tokens files.

  python -m tools.bench.kv_eval.analyze --dir <out-dir> --arms-json <arms.json> \
      (--artifact <artifact> | --tokenizer <tokenizer.json>) [--corpus-manifest <manifest>] [--out <dir>] [--arms a,b,c]

Layout under --dir (what run_ppl.py writes):  <arm>/report.json  and  dumps/<arm>.topk.
Writes <out>/analysis.json and <out>/tables.md (also printed). The metric definitions live in metrics.py and
docs/maintainer/gemma4-benchmarks.md. Everything is teacher-forced fidelity against the reference arm on the same artifact, corpus,
context, stride and prefix ids; none of it is task quality or generation behaviour.

Position bins use pos = stream_token_index - window input_begin (the prefix ids are not counted). Per-bin dPPL is NOT
available (the dump holds no per-position NLL), so bins report top-1, truncated KL and target dlogprob only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from tokenizers import Tokenizer

from tools.bench.kv_eval import metrics
from tools.bench.kv_eval.common import (
    Arm, corpus_identity, fail, find_arm, load_arms, read_json, resolve_tokenizer, sha256_file, use_utf8_stdio, utc_now,
)
from tools.bench.kv_eval.metrics import EvalError

DEFAULT_BINS = (0, 4096, 16384, 32768, 49152, 65536)
KL_LABEL = "truncated-distribution diagnostic, not full-vocab KL"
COMPARED_EXECUTION_KEYS = ("context_tokens", "stride_tokens", "prefill_chunk_tokens", "score_tile_tokens", "device",
                           "logits_dtype", "rope_yarn_factor")


def bin_labels(edges):
    def one(lo, hi):
        return f"{lo // 1024}-{hi // 1024}k" if lo % 1024 == 0 and hi % 1024 == 0 else f"{lo}-{hi}"
    return [one(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]


def load_report(root: Path, name: str):
    report_path = root / name / "report.json"
    dump_path = root / "dumps" / f"{name}.topk"
    for path in (report_path, dump_path):
        if not path.is_file():
            raise EvalError(f"arm '{name}': missing {path}; run run_ppl.py for this arm or drop it from --arms")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return report, dump_path


def check_comparable(reports: dict, ref_name: str) -> list[dict]:
    """Every arm must share artifact, corpus, context, stride and window layout with the reference."""
    ref = reports[ref_name]
    ref_streams = {s["id"]: s for s in ref["streams"]}
    fields = []
    for name, r in reports.items():
        def need(cond, what):
            if not cond:
                raise EvalError(f"arm '{name}' differs from reference '{ref_name}' in {what}; arms are only comparable "
                                "with identical artifact, corpus, context and stride - rerun it with matching settings")
        need(r["schema_version"] == ref["schema_version"], "report schema_version")
        need(r["corpus"]["id"] == ref["corpus"]["id"], "corpus id")
        for key in COMPARED_EXECUTION_KEYS:
            need(r["execution"][key] == ref["execution"][key], f"execution.{key}")
        need(r["artifact"]["name"] == ref["artifact"]["name"] and r["artifact"]["prefill_signature"] == ref["artifact"]["prefill_signature"],
             "artifact name or prefill signature")
        need(r["artifact"]["path"] == ref["artifact"]["path"] and r["artifact"]["formats"] == ref["artifact"]["formats"],
             "artifact path or weight formats")
        if r["corpus"]["mode"] == ref["corpus"]["mode"]:
            need(r["corpus"]["stream_count"] == ref["corpus"]["stream_count"], "corpus stream count")
            need(r["overall"]["scored_tokens"] == ref["overall"]["scored_tokens"], "scored-token total")
        for s in r["streams"]:
            sr = ref_streams.get(s["id"])
            need(sr is not None, f"stream '{s['id']}' (not in the reference)")
            need((s["input_tokens"], s["scored_tokens"], s["domain"]) == (sr["input_tokens"], sr["scored_tokens"], sr["domain"]),
                 f"stream '{s['id']}' token counts or domain")
            need([(w["input_begin"], w["input_end"], w["target_begin"], w["target_end"]) for w in s["windows"]] ==
                 [(w["input_begin"], w["input_end"], w["target_begin"], w["target_end"]) for w in sr["windows"]],
                 f"stream '{s['id']}' window ranges")
        fields.append({"arm": name, "mode": r["corpus"]["mode"], "kv_dtype": r["execution"]["kv_dtype"],
                       "qk_requested": r["execution"].get("gemma4_qk"), "qk_resolved": r["execution"].get("gemma4_qk_resolved"),
                       "qk_scope": r["execution"].get("gemma4_qk_scope"), "context": r["execution"]["context_tokens"],
                       "stride": r["execution"]["stride_tokens"], "streams": len(r["streams"]),
                       "windows": sum(len(s["windows"]) for s in r["streams"]), "scored_tokens": r["overall"]["scored_tokens"]})
    return fields


def layout(report, tokenizer: Tokenizer, manifest_dir: Path):
    """Per scored position of `report`: stream, domain, global window, position in window and target id."""
    domains = sorted({s["domain"] for s in report["streams"]})
    cols = {k: [] for k in ("stream", "domain", "window", "pos", "target")}
    fingerprints = {}
    wglobal = 0
    for si, s in enumerate(report["streams"]):
        path = manifest_dir / s["path"].replace("\\", "/")
        if not path.is_file():
            raise EvalError(f"corpus stream file not found: {path}; pass --corpus-manifest <manifest.json> of the corpus the run used")
        ids = tokenizer.encode(path.read_bytes().decode("utf-8"), add_special_tokens=False).ids
        if len(ids) != s["input_tokens"]:
            raise EvalError(f"tokenizer mismatch for stream {s['id']}: {len(ids)} tokens vs the report's {s['input_tokens']}; "
                            "use the tokenizer of the artifact that produced the report (--artifact) and the same corpus files")
        for w in s["windows"]:
            begin, end = w["input_begin"], w["input_end"]
            if not 0 <= begin <= w["target_begin"] < w["target_end"] <= end <= len(ids):
                raise EvalError(f"invalid input/target range in stream {s['id']}")
            key = (s["id"], w["target_begin"], w["target_end"])
            if key in fingerprints:
                raise EvalError(f"duplicate scored window identity: {key}")
            fingerprints[key] = metrics.token_fingerprint(ids[begin:end])
            n = w["target_end"] - w["target_begin"]
            cols["stream"].append(np.full(n, si, np.int32))
            cols["domain"].append(np.full(n, domains.index(s["domain"]), np.int32))
            cols["window"].append(np.full(n, wglobal, np.int32))
            cols["pos"].append(np.arange(w["target_begin"], w["target_end"], dtype=np.int64) - w["input_begin"])
            cols["target"].append(np.asarray(ids[w["target_begin"]:w["target_end"]], np.int32))
            wglobal += 1
    return domains, {k: np.concatenate(v) for k, v in cols.items()}, fingerprints


def analyze(root: Path, arms: list[Arm], tokenizer_path: Path, manifest_dir: Path | None, edges=DEFAULT_BINS) -> dict:
    """Full analysis of the arms found under `root`; returns the analysis.json document."""
    ref_name = next(a.name for a in arms if a.role == "reference")
    names = [a.name for a in arms]
    noise = [a.name for a in arms if a.role == "noise"]
    quant = [a.name for a in arms if a.role == "quant"]
    labels = bin_labels(edges)
    reports, dumps, dump_paths = {}, {}, {}
    for name in names:
        reports[name], dump_paths[name] = load_report(root, name)
    ref = reports[ref_name]
    cmp_fields = check_comparable(reports, ref_name)

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    if manifest_dir is None:
        manifest_dir = Path(ref["corpus"]["source"]).parent
    if not manifest_dir.is_dir():
        raise EvalError(f"corpus directory {manifest_dir} not found (taken from the report); pass --corpus-manifest <manifest.json>")
    domains, lay, fingerprints = layout(ref, tokenizer, manifest_dir)
    for name in names:
        report = reports[name]
        try:
            expected = [fingerprints[(stream["id"], window["target_begin"], window["target_end"])]
                        for stream in report["streams"] for window in stream["windows"]]
        except KeyError as exc:
            raise EvalError(f"arm '{name}': missing reconstructed input identity for {exc.args[0]}") from exc
        dumps[name] = metrics.open_dump(dump_paths[name], report["overall"]["scored_tokens"], name, report,
                                        expected_fingerprints=expected)
    pos_bin = np.full(len(lay["pos"]), -1, np.int8)
    for bi, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        pos_bin[(lay["pos"] >= lo) & (lay["pos"] < hi)] = bi
    max_pos = int(lay["pos"].max())
    if max_pos >= edges[-1]:
        raise EvalError(f"position {max_pos} in a window is beyond the last bin edge {edges[-1]}; pass --bins with a larger final edge")

    # Tokenizer faithfulness: misaligned targets would almost never sit in the top-64; at PPL ~10 a faithful target
    # misses it on a few percent of positions. A one-sided window bound then ties the dump to the report's NLL.
    tlp_ref = metrics.target_lp(dumps[ref_name], lay["target"])
    hit = ~np.isnan(tlp_ref)
    cov_by_domain = {d: float(hit[lay["domain"] == i].mean()) for i, d in enumerate(domains)}
    if min(cov_by_domain.values()) <= 0.5:
        raise EvalError(f"target-in-reference-top64 coverage too low {cov_by_domain}: the tokenizer or corpus does not match the run")
    win_nll = np.array([w["total_nll"] for s in ref["streams"] for w in s["windows"]])
    found = -np.bincount(lay["window"], weights=np.nan_to_num(tlp_ref.astype(np.float64)))
    floor = -np.bincount(lay["window"], weights=np.where(hit, 0.0, dumps[ref_name]["lp"][:, -1].astype(np.float64)))
    violations = int(((found + floor) > win_nll * (1 + 1e-4)).sum())
    if violations:
        raise EvalError(f"{violations} windows violate the one-sided NLL bound: the dump does not belong to the report's run")
    target_check = {"target_in_ref_top64_frac": float(hit.mean()), "target_in_ref_top64_frac_by_domain": cov_by_domain,
                    "windows_total": int(len(win_nll)), "windows_violating_one_sided_nll_bound": violations,
                    "positions_by_bin": {labels[i]: int((pos_bin == i).sum()) for i in range(len(labels))},
                    "max_position_in_window": max_pos}

    # An arm may cover only a subset of streams (e.g. a --quick repeat): pairs use the streams both arms contain.
    stream_ids = [s["id"] for s in ref["streams"]]
    sel_of, tlp_of = {}, {ref_name: tlp_ref}
    for name in names:
        have = {s["id"] for s in reports[name]["streams"]}
        sel_of[name] = np.isin(lay["stream"], [i for i, sid in enumerate(stream_ids) if sid in have])
        if int(sel_of[name].sum()) != len(dumps[name]):
            raise EvalError(f"arm '{name}': dump has {len(dumps[name])} positions but its streams cover {int(sel_of[name].sum())}")
        if name != ref_name:
            tlp_of[name] = metrics.target_lp(dumps[name], lay["target"][sel_of[name]])

    def dump_view(name, common):
        if sel_of[name].all() and common.all():
            return dumps[name]
        rel = common[sel_of[name]]
        return dumps[name][rel] if not rel.all() else dumps[name]

    def tlp_view(name, common):
        full = np.full(len(lay["pos"]), np.nan, np.float32)
        full[sel_of[name]] = tlp_of[name]
        return full[common]

    pair_defs = [(f"noise floor: {a} vs {ref_name}", a, ref_name) for a in noise]
    pair_defs += [(f"{a} vs {ref_name}", a, ref_name) for a in quant]
    for i in range(len(quant)):
        for j in range(i + 1, len(quant)):
            pair_defs.append((f"{quant[j]} vs {quant[i]}", quant[j], quant[i]))

    out = {"comparability": cmp_fields, "target_derivation_check": target_check, "domains": domains,
           "arms": [{"name": a.name, "role": a.role, "flags": list(a.flags)} for a in arms], "reference": ref_name,
           "bins": dict(zip(labels, [list(p) for p in zip(edges[:-1], edges[1:])])), "kl_label": KL_LABEL,
           "ppl_per_arm": {}, "pairs": {}}
    for name in names:
        r = reports[name]
        out["ppl_per_arm"][name] = {"overall": r["overall"], "mode": r["corpus"]["mode"], "streams": len(r["streams"]),
                                    "domains": {d["domain"]: {"scored_tokens": d["scored_tokens"], "mean_nll": d["mean_nll"],
                                                              "perplexity": d["perplexity"]} for d in r["domains"]}}

    per_pos_vs_ref = {}
    n_bins = len(labels)
    for label, a, r in pair_defs:
        common = sel_of[a] & sel_of[r]
        pp = metrics.per_position(dump_view(a, common), dump_view(r, common), tlp_view(a, common), tlp_view(r, common))
        dom, binx, win = lay["domain"][common], pos_bin[common], lay["window"][common]
        entry = {"arm": a, "reference": r, "ppl": metrics.ppl_delta(reports[a], reports[r]),
                 "metrics": {"overall": metrics.agg(pp, np.ones(len(dom), bool))}, "by_domain": {}, "by_bin": {}, "by_bin_window_se": {}}
        for i, d in enumerate(domains):
            if (dom == i).any():
                entry["by_domain"][d] = metrics.agg(pp, dom == i)
        dis = (~pp["top1"]).astype(np.float64)
        everything = np.ones(len(dom), bool)
        kl_wb = metrics.window_bin_stats(win, binx, pp["kl"], everything, n_bins)
        dis_wb = metrics.window_bin_stats(win, binx, dis, everything, n_bins)
        for bi, bl in enumerate(labels):
            m = binx == bi
            if not m.any():
                continue
            entry["by_bin"][bl] = metrics.agg(pp, m)
            for nm, wb in (("kl_top64_truncated", kl_wb), ("top1_disagree", dis_wb)):
                v = wb[:, bi][~np.isnan(wb[:, bi])]
                stat = metrics.paired_t(v)
                entry["by_bin_window_se"].setdefault(bl, {})[nm] = {"windows": int(v.size), "mean_of_window_means": stat["mean"], "se": stat["se"]}
        out["pairs"][label] = entry
        if r == ref_name:
            per_pos_vs_ref[a] = {"kl": pp["kl"], "dis": dis, "win": win, "bin": binx, "common": common}

    out["arm_vs_arm_paired"] = {
        "note": "per-position metric vs the reference averaged within each window (and within each position bin), then paired "
                "difference between two quantized arms: mean +/- SE over windows; negative = first arm closer to reference",
        "rows": {}}
    for i in range(len(quant)):
        for j in range(i + 1, len(quant)):
            x, y = quant[i], quant[j]
            px, py = per_pos_vs_ref[x], per_pos_vs_ref[y]
            if not (px["common"].all() and py["common"].all()):
                raise EvalError("arm-vs-arm paired analysis requires full-corpus quantized arms (do not use --quick for them)")
            row = {}
            for nm, key in (("kl_top64_truncated", "kl"), ("top1_disagree", "dis")):
                everything = np.ones(len(px[key]), bool)
                dd = metrics.window_means(px["win"], px[key]) - metrics.window_means(py["win"], py[key])
                stat = metrics.paired_t(dd)
                row[nm] = {"mean_diff": stat["mean"], "se": stat["se"], "t": stat["t"]}
                bins = {}
                wx = metrics.window_bin_stats(px["win"], px["bin"], px[key], everything, n_bins)
                wy = metrics.window_bin_stats(py["win"], py["bin"], py[key], everything, n_bins)
                for bi, bl in enumerate(labels):
                    d = wx[:, bi] - wy[:, bi]
                    d = d[~np.isnan(d)]
                    if d.size > 1:
                        bs = metrics.paired_t(d)
                        bins[bl] = {"windows": int(d.size), "mean_diff": bs["mean"], "se": bs["se"], "t": bs["t"]}
                row[nm]["by_bin"] = bins
            out["arm_vs_arm_paired"]["rows"][f"{x} - {y}"] = row
    return out


def fmt_t(stat):
    return f"{stat:+.2f}" if stat is not None else "n/a"


def render(out) -> str:
    arms = [a["name"] for a in out["arms"]]
    domains = out["domains"]
    labels = list(out["bins"])
    L = ["### Per-arm PPL (report values, unrounded token-weighted)", "",
         "| arm | mode | streams | scored tokens | PPL overall | " + " | ".join(domains) + " |",
         "| --- | --- | ---: | ---: | ---: | " + " | ".join("---:" for _ in domains) + " |"]
    for a in arms:
        r = out["ppl_per_arm"][a]
        L.append(f"| {a} | {r['mode']} | {r['streams']} | {r['overall']['scored_tokens']:,} | {r['overall']['perplexity']:.5f} | "
                 + " | ".join(f"{r['domains'][d]['perplexity']:.4f}" if d in r["domains"] else "-" for d in domains) + " |")
    L += ["", "### Pairwise vs reference / between arms (overlap of streams both arms contain)", "",
          "| pair | streams-windows | dPPL % (ref PPL -> arm PPL) | window paired dNLL/tok ± SE (t) | top-1 agree | trunc. KL mean / p99 *(truncated diagnostic)* | mean abs dlogprob on both-in-top-64 *(subset)* | bit-identical top-64 |",
          "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |"]
    for label, p in out["pairs"].items():
        o, pp = p["metrics"]["overall"], p["ppl"]["overall"]
        se = f"{pp['se']:.2e}" if pp["se"] is not None else "n/a"
        L.append(f"| {label} | {pp['windows']} win | {pp['dppl_pct']:+.4f} % ({pp['ppl_ref']:.4f} -> {pp['ppl_arm']:.4f}) | "
                 f"{pp['mean_dnll_per_token_window_mean']:+.2e} ± {se} ({fmt_t(pp['t'])}) | {100 * o['top1_agree']:.3f} % | "
                 f"{o['kl_top64_truncated_mean']:.3e} / {o['kl_top64_truncated_p99']:.3f} | "
                 f"{o['target_abs_dlogprob_mean_subset']:.4f} | {100 * o['bit_identical_topk_frac']:.2f} % |")
    L += ["", "### Per domain: dPPL % / top-1 agreement % / truncated KL mean", "",
          "| pair | " + " | ".join(domains) + " |", "| --- | " + " | ".join("---" for _ in domains) + " |"]
    for label, p in out["pairs"].items():
        cells = []
        for d in domains:
            if d in p["by_domain"]:
                m, pp = p["by_domain"][d], p["ppl"]["domains"][d]
                cells.append(f"{pp['dppl_pct']:+.3f} / {100 * m['top1_agree']:.2f} / {m['kl_top64_truncated_mean']:.2e}")
            else:
                cells.append("-")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "### Position-in-window bins (distance from window start; window 0 of each stream supplies the first `stride` positions, later windows only the rest)", "",
          "Cells: top-1 agreement % / truncated KL mean *(truncated diagnostic)* ± SE over windows / mean abs dlogprob on both-in-top-64 subset (positions). "
          "Per-bin dPPL is not available (no per-position NLL in the dump).", "",
          "| pair | " + " | ".join(f"{b}" for b in labels) + " |", "| --- | " + " | ".join("---" for _ in labels) + " |"]
    for label, p in out["pairs"].items():
        cells = []
        for bl in labels:
            if bl in p["by_bin"]:
                m, w = p["by_bin"][bl], p["by_bin_window_se"][bl]["kl_top64_truncated"]
                se = f"{w['se']:.1e}" if w["se"] is not None else "n/a"
                sub = m["target_abs_dlogprob_mean_subset"]
                cells.append(f"{100 * m['top1_agree']:.2f} / {m['kl_top64_truncated_mean']:.2e} ± {se} / "
                             f"{'n/a' if sub is None else f'{sub:.4f}'} ({m['positions']:,})")
            else:
                cells.append("-")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L += ["", "### Arm vs arm: paired difference of divergence-from-reference (negative = first arm closer to reference)", ""]
    for k, row in out["arm_vs_arm_paired"]["rows"].items():
        for nm, v in row.items():
            bins = "; ".join(f"{bl} {b['mean_diff']:+.2e}±{b['se']:.1e} (t {b['t']:+.1f})" for bl, b in v["by_bin"].items() if b["t"] is not None)
            se = f"{v['se']:.2e}" if v["se"] is not None else "n/a"
            L.append(f"* `{k}` {nm}: overall {v['mean_diff']:+.3e} ± {se} (t = {fmt_t(v['t'])}); by bin: {bins}")
    L.append("")
    return "\n".join(L)


def parse_bins(text: str):
    try:
        edges = tuple(int(x) for x in text.split(","))
    except ValueError:
        fail(f"--bins expects comma-separated integers, got '{text}'")
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])) or edges[0] != 0:
        fail("--bins must start at 0 and strictly increase, e.g. 0,4096,16384,32768,49152,65536")
    return edges


def select_arms(args) -> list[Arm]:
    if args.arms_json:
        arms = load_arms(args.arms_json)
        if args.arms:
            wanted = args.arms.split(",")
            arms = [find_arm(arms, name, args.arms_json) for name in wanted]
        if sum(a.role == "reference" for a in arms) != 1:
            fail("the selected arms must include the reference arm exactly once")
        return arms
    if not args.arms:
        fail("pass --arms-json <arms.json> or --arms name1,name2,... (first = reference; names containing 'repeat' = noise floor)")
    names = args.arms.split(",")
    return [Arm(n, (), "reference" if i == 0 else "noise" if ("repeat" in n) else "quant") for i, n in enumerate(names)]


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="directory holding <arm>/report.json and dumps/<arm>.topk")
    ap.add_argument("--arms-json", type=Path, help="arms file (see arms.example.json); roles decide reference/noise/quant")
    ap.add_argument("--arms", help="comma list of arm names; with --arms-json a subset, without it the first is the reference")
    ap.add_argument("--tokenizer", type=Path, help="tokenizer.json of the artifact that produced the reports")
    ap.add_argument("--artifact", type=Path, help="the .infernix artifact; its embedded tokenizer.json is used when --tokenizer is absent")
    ap.add_argument("--corpus-manifest", type=Path, help="manifest.json of the corpus used by the run (default: the path recorded in report.json)")
    ap.add_argument("--bins", default=",".join(str(e) for e in DEFAULT_BINS), help="position-bin edges in tokens (default %(default)s)")
    ap.add_argument("--out", type=Path, help="output directory (default: --dir)")
    ap.add_argument("--no-hash-inputs", action="store_true", help="skip sha256 of reports/dumps in the provenance block")
    args = ap.parse_args()
    root = args.dir.resolve()
    outdir = (args.out or args.dir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    arms = select_arms(args)
    tokenizer_path, tok_prov = resolve_tokenizer(args.tokenizer, args.artifact, outdir / "artifact-resources")
    manifest_dir = args.corpus_manifest.resolve().parent if args.corpus_manifest else None
    try:
        out = analyze(root, arms, tokenizer_path, manifest_dir, parse_bins(args.bins))
    except EvalError as exc:
        fail(str(exc))
    ref_report = read_json(root / out["reference"] / "report.json", "reference report")
    manifest_path = args.corpus_manifest or Path(ref_report["corpus"]["source"])
    provenance = {"analyzed_utc": utc_now(), "tokenizer": tok_prov, "root": str(root)}
    if manifest_path.is_file():
        provenance["corpus"] = corpus_identity(manifest_path)
    if not args.no_hash_inputs:
        provenance["inputs"] = {a.name: {"report_sha256": sha256_file(root / a.name / "report.json"),
                                         "dump_sha256": sha256_file(root / "dumps" / f"{a.name}.topk")} for a in arms}
    out["provenance"] = provenance
    (outdir / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    text = render(out)
    (outdir / "tables.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
