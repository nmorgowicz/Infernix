"""Paired comparison of two ``infernix-perplexity --full-vocab-reference`` reports.

Both reports must have scored the same windows against the same reference cache (identical
``kl.protocol`` and ``kl.reference_cache`` provenance). The comparison resamples whole windows,
the same ones in both arms, and reports the candidate/baseline mean-KL ratio and the paired
change in target NLL with 95% percentile intervals, overall and per domain.

    python -m tools.bench.kv_eval.kl_compare --baseline <g32/report.json> \
        --candidate <g128/report.json> [--delta 0.10]

``--delta`` is the operator's acceptance factor (plan section 3.3): the candidate passes a group
when the upper end of its mean-KL ratio interval is at most ``1 + delta``. There is deliberately
no default; the operator sets it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def load(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if "kl" not in report or "statistics" not in report["kl"]:
        raise SystemExit(f"{path} is not a --full-vocab-reference comparison report")
    return report["kl"]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if ordered[lower] == ordered[upper]:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired(baseline: list[dict], candidate: list[dict], resamples: int, seed: int) -> dict:
    n = len(baseline)
    counts = [w["count"] for w in baseline]
    rng = random.Random(seed)
    ratios: list[float] = []
    deltas: list[float] = []
    for _ in range(resamples):
        picks = [rng.randrange(n) for _ in range(n)]
        total = sum(counts[i] for i in picks)
        base_kl = sum(baseline[i]["kl_sum"] for i in picks)
        cand_kl = sum(candidate[i]["kl_sum"] for i in picks)
        ratios.append(cand_kl / base_kl if base_kl > 0 else float("inf"))
        deltas.append(
            (sum(candidate[i]["delta_nll_sum"] for i in picks)
             - sum(baseline[i]["delta_nll_sum"] for i in picks)) / total
        )
    total = sum(counts)
    base_kl = sum(w["kl_sum"] for w in baseline)
    cand_kl = sum(w["kl_sum"] for w in candidate)
    return {
        "windows": n,
        "positions": total,
        "baseline_mean_kl": base_kl / total,
        "candidate_mean_kl": cand_kl / total,
        "kl_ratio": cand_kl / base_kl if base_kl > 0 else float("inf"),
        "kl_ratio_ci95": [percentile(ratios, 0.025), percentile(ratios, 0.975)],
        "delta_nll_candidate_minus_baseline": (
            sum(w["delta_nll_sum"] for w in candidate) - sum(w["delta_nll_sum"] for w in baseline)
        ) / total,
        "delta_nll_ci95": [percentile(deltas, 0.025), percentile(deltas, 0.975)],
    }


def compare(baseline: dict, candidate: dict, resamples: int, seed: int, delta: float | None) -> dict:
    if resamples <= 0:
        raise SystemExit("resamples must be positive")
    if baseline["protocol"] != candidate["protocol"]:
        raise SystemExit("reports used different protocols; they are not paired")
    if baseline.get("reference_artifact") != candidate.get("reference_artifact"):
        raise SystemExit("reports used different reference artifacts; they are not paired")
    if not baseline.get("reference_cache") or baseline["reference_cache"] != candidate.get("reference_cache"):
        raise SystemExit("reports used different reference caches; they are not paired")
    base_windows = baseline["statistics"]["windows"]
    cand_windows = candidate["statistics"]["windows"]
    if not base_windows or any(w["count"] <= 0 for w in base_windows + cand_windows):
        raise SystemExit("reports must contain nonempty scored windows")
    keys = [(w["stream"], w["domain"], w["begin"], w["count"]) for w in base_windows]
    if keys != [(w["stream"], w["domain"], w["begin"], w["count"]) for w in cand_windows]:
        raise SystemExit("reports scored different windows; they are not paired")
    result = {"overall": paired(base_windows, cand_windows, resamples, seed), "domains": []}
    for domain in sorted({w["domain"] for w in base_windows}):
        members = [i for i, w in enumerate(base_windows) if w["domain"] == domain]
        entry = paired([base_windows[i] for i in members], [cand_windows[i] for i in members],
                       resamples, seed)
        entry["domain"] = domain
        result["domains"].append(entry)
    tails = {}
    for key in ("p95_kl", "p99_kl", "p999_kl", "top1_disagreement"):
        b = baseline["statistics"]["overall"][key]
        c = candidate["statistics"]["overall"][key]
        tails[key] = {"baseline": b, "candidate": c, "ratio": (c / b) if b else None}
    result["tails_overall_point_estimates"] = tails
    if delta is not None:
        limit = 1.0 + delta
        groups = [result["overall"]] + result["domains"]
        result["acceptance"] = {
            "delta": delta,
            "rule": "upper 95% bound of mean-KL ratio <= 1 + delta, overall and every domain",
            "per_group": {g.get("domain", "overall"): g["kl_ratio_ci95"][1] <= limit for g in groups},
            "p999_kl_ratio_within": tails["p999_kl"]["ratio"] is not None
            and tails["p999_kl"]["ratio"] <= limit,
        }
        result["acceptance"]["accepted"] = (
            all(result["acceptance"]["per_group"].values())
            and result["acceptance"]["p999_kl_ratio_within"]
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--delta", type=float, default=None,
                        help="operator acceptance factor; omit to report without a verdict")
    parser.add_argument("--out", type=Path, default=None, help="write the JSON here as well")
    args = parser.parse_args(argv)
    result = compare(load(args.baseline), load(args.candidate), args.resamples, args.seed, args.delta)
    text = json.dumps(result, indent=2)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
