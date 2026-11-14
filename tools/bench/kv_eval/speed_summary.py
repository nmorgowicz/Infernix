# Ported from ninfer.spo tools/bench/kv_eval; instrument definitions retained.
#!/usr/bin/env python3
"""Aggregate <label>.cells.jsonl (+ <label>.serve.jsonl server_start events) into summary.{json,md}.

  python -m tools.bench.kv_eval.speed_summary --dir <speed-dir> --arms <arm1,arm2,...> [--modes mtp,nomtp]

Labels are <arm>-<mode>. Per (label, nominal length): warmup excluded, median and min/max over the timed repetitions of
prefill tok/s (prompt_n / prompt_ms), decode tok/s (server predicted_per_second) and speculative-draft acceptance
(draft_n_accepted / draft_n per request, accepted draft tokens over all drafted tokens, not mean accepted length).
The server_start event (the server's --request-log-jsonl) supplies the KV format, resolved capacity and memory. Cells
whose requests were not 256 new tokens, or reused more than --max-cache-n cached tokens, carry a note.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.kv_eval.common import fail, use_utf8_stdio

GIB = 2 ** 30


def fmt(vals, digits=1):
    if not vals:
        return "n/a"
    return f"{statistics.median(vals):.{digits}f} ({min(vals):.{digits}f}–{max(vals):.{digits}f})"


def load(path: Path):
    if not path.exists():
        return None
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def server_start(path: Path):
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d.get("event") == "server_start":
            m, e = d.get("memory", {}), d.get("engine", {})
            gemma = m.get("gemma4_kv") or {}
            row = {"argv": d.get("argv", [])[2:], "kv_cache": e.get("kv_cache"), "kv_capacity": e.get("kv_capacity"),
                   "max_context": e.get("max_context"), "gemma4_qk_resolved": gemma.get("qk_mode", e.get("gemma4_qk")),
                   "speculative_backend": e.get("speculative_backend"),
                   "kv_payload_gib": m["kv_payload_bytes"] / GIB if "kv_payload_bytes" in m else None,
                   "available_after_startup_gib": m["available_after_startup_bytes"] / GIB if "available_after_startup_bytes" in m else None,
                   "runtime_reservation_gib": m["runtime_reservation_bytes"] / GIB if "runtime_reservation_bytes" in m else None,
                   "environment": d.get("environment")}
            if "global_payload_bytes" in gemma:
                row["global_kv_gib"] = gemma["global_payload_bytes"] / GIB
                row["sliding_kv_gib"] = gemma["sliding_payload_bytes"] / GIB
            return row
    return None


def gib(v, split=None):
    if v is None:
        return "n/a"
    return f"{v:.2f}" if split is None else f"{v:.2f} ({split})"


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True, help="directory with <arm>-<mode>.cells.jsonl and <arm>-<mode>.serve.jsonl")
    ap.add_argument("--arms", required=True)
    ap.add_argument("--modes", default="mtp,nomtp")
    ap.add_argument("--expect-new", type=int, default=256, help="new tokens each request must produce")
    ap.add_argument("--max-cache-n", type=int, default=8, help="cached prompt tokens above this flag cache reuse (chat-template head only)")
    args = ap.parse_args()
    arms, modes = args.arms.split(","), args.modes.split(",")
    summary, lines = {}, []
    lines.append("| arm | spec | prompt (nominal → exact tokens) | prefill tok/s median (min–max) | decode tok/s median (min–max) | draft accept % median (min–max) | n | notes |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- |")
    for arm in arms:
        for mode in modes:
            label = f"{arm}-{mode}"
            rows = load(args.dir / f"{label}.cells.jsonl")
            if rows is None:
                continue
            reqs = [r for r in rows if r["kind"] == "request"]
            for nominal in sorted({r["nominal"] for r in reqs}):
                cell = [r for r in reqs if r["nominal"] == nominal]
                timed = [r for r in cell if not r["warmup"] and not r["error"]]
                errs = [r["error"] for r in cell if r["error"]]
                pt = [r["prompt_tokens"] for r in timed]
                pre = [r["prefill_tok_s"] for r in timed]
                dec = [r["decode_tok_s"] for r in timed]
                acc = [100.0 * r["draft_n_accepted"] / r["draft_n"] for r in timed if r.get("draft_n")]
                notes = []
                if errs:
                    notes.append("ERROR: " + errs[0][:160])
                if any(r["predicted_n"] != args.expect_new for r in timed):
                    notes.append(f"predicted_n≠{args.expect_new}: " + ",".join(str(r["predicted_n"]) for r in timed))
                if any(r["cache_n"] > args.max_cache_n for r in timed):
                    notes.append(f"cache_n>{args.max_cache_n}: " + ",".join(str(r["cache_n"]) for r in timed))
                summary[f"{label}|{nominal}"] = {"n": len(timed), "prompt_tokens": pt, "prefill_tok_s": pre, "decode_tok_s": dec,
                                                 "accept_pct": acc, "errors": errs, "notes": notes,
                                                 "draft_n": [r.get("draft_n") for r in timed],
                                                 "draft_n_accepted": [r.get("draft_n_accepted") for r in timed]}
                lines.append(f"| {arm} | {mode} | {nominal} → {fmt(pt, 0)} | {fmt(pre)} | {fmt(dec)} | "
                             f"{fmt(acc)} | {len(timed)} | {'; '.join(notes)} |")
    if not summary:
        fail(f"no <arm>-<mode>.cells.jsonl for arms {arms} and modes {modes} under {args.dir}")
    starts = {}
    lines += ["", "| label | KV | KV capacity (tokens) | max_context | KV payload GiB (global + sliding where reported) | free after startup GiB | runtime reservation GiB | QK resolved |",
              "| --- | --- | ---: | ---: | --- | ---: | ---: | --- |"]
    for arm in arms:
        for mode in modes:
            label = f"{arm}-{mode}"
            s = server_start(args.dir / f"{label}.serve.jsonl")
            if s:
                starts[label] = s
                split = f"{s['global_kv_gib']:.2f} + {s['sliding_kv_gib']:.2f}" if "global_kv_gib" in s else None
                lines.append(f"| {label} | {s['kv_cache']} | {s['kv_capacity']:,} | {s['max_context']:,} | {gib(s['kv_payload_gib'], split)} | "
                             f"{gib(s['available_after_startup_gib'])} | {gib(s['runtime_reservation_gib'])} | {s['gemma4_qk_resolved']} |")
    (args.dir / "summary.json").write_text(json.dumps({"cells": summary, "server_start": starts}, indent=2), encoding="utf-8")
    (args.dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
