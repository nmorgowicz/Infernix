# Ported from ninfer.spo tools/bench/kv_eval; instrument definitions retained.
#!/usr/bin/env python3
"""Needle-in-haystack retrieval over HTTP for one running infernix-serve instance, plus the cross-arm summary.

  python -m tools.bench.kv_eval.needle run --label <arm> --out <out-dir>/<arm>.jsonl --url <chat-completions-url> \
      --model <model id> (--artifact <artifact> | --tokenizer <tokenizer.json>) --manifest <corpus-manifest> [--dry-run]
  python -m tools.bench.kv_eval.needle summarize --dir <out-dir> --arms <arm1,arm2,...>

Cases are a deterministic function of (master seed, length, depth, needle index, corpus, tokenizer): the haystack is a
natural-text window of the corpus pool (corpus_pool.Pool); one synthetic fact "The access code for vault <Name> is
<6 digits>." is inserted on its own line at the first line break at/after the target depth, then the question is
appended. Names and codes are distinct across all cases. Prompts are byte-identical across arms; prompt_sha256 is
recorded per row and `summarize` refuses arms whose prompts differ. Request: temperature 0, top_k 20, fixed seed,
max_tokens 32, no streaming. Score: the reply contains exactly one distinct 6-digit string and it equals the planted code.
This is task-level single-fact retrieval, a separate instrument from perplexity fidelity: never mix their numbers.
`run` resumes: cases already present in --out are skipped. --dry-run builds the cases and prints their prompt hashes
without contacting a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.gemma4_common import add_thinking_arg, thinking_value
from tools.bench.kv_eval.common import fail, resolve_tokenizer, use_utf8_stdio
from tools.bench.kv_eval.corpus_pool import DEFAULT_DOMAINS, Pool
from tools.bench.kv_eval.metrics import mcnemar_exact, wilson

DEFAULT_LENGTHS = (8000, 32000, 100000)
DEFAULT_DEPTHS = (0.05, 0.25, 0.50, 0.75, 0.95)
NEEDLES_PER_CELL = 8
OVERHEAD_TOKENS = 160
SEED = 1234
TOP_K = 20
MAX_NEW = 32
MASTER_SEED = 20261104
ADJ = ["Amber", "Brisk", "Cobalt", "Dusky", "Ember", "Frosty", "Gilded", "Hollow", "Ivory", "Jade", "Kestrel", "Lunar", "Mossy",
       "Nimble", "Onyx", "Pale", "Quiet", "Russet", "Silver", "Tawny"]
NOUN = ["Anchor", "Beacon", "Canyon", "Delta", "Echo", "Falcon", "Garnet", "Harbor", "Island", "Juniper", "Lantern", "Meadow",
        "Needle", "Orchard", "Prairie", "Quarry", "Ridge", "Summit", "Thicket", "Valley"]


def build_cases(pool, lengths, depths, per_cell, master_seed, overhead):
    rng = random.Random(master_seed)
    names = [f"{a} {n}" for a in ADJ for n in NOUN]
    rng.shuffle(names)
    total = len(lengths) * len(depths) * per_cell
    if total > len(names):
        fail(f"{total} cases need {total} distinct vault names but only {len(names)} exist; reduce --lengths/--depths/--needles-per-cell")
    codes = rng.sample(range(100000, 1000000), total)
    cases, i = [], 0
    for length in lengths:
        for depth in depths:
            for k in range(per_cell):
                crng = random.Random(f"{master_seed}|{length}|{depth}|{k}")
                n_text = length - overhead
                start = crng.randrange(0, pool.n_tokens - n_text - 1000)
                body, first, last = pool.window(start, n_text)
                name, code = names[i], str(codes[i])
                i += 1
                target = int(len(body) * depth)
                nl = body.find("\n", target)
                if nl < 0:
                    nl = body.rfind("\n")
                at = nl + 1
                fact = f"The access code for vault {name} is {code}."
                text = body[:at] + fact + "\n\n" + body[at:]
                prompt = (f"Read the following document carefully.\n\n{text}\n\n"
                          f"What is the access code for vault {name}? Answer with only the 6-digit code.")
                cases.append({"id": f"L{length}-D{int(depth * 100):02d}-N{k}", "length": length, "depth": depth, "needle": k,
                              "name": name, "code": code, "pool_first_token": first, "pool_last_token": last,
                              "actual_depth_tokens": pool.count(body[:at]) / max(1, pool.count(body)),
                              "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()})
    return cases


def score(reply, code):
    found = set(re.findall(r"(?<!\d)\d{6}(?!\d)", reply))
    return found == {code}


def make_pool(args):
    tokenizer, _ = resolve_tokenizer(args.tokenizer, args.artifact, args.out.resolve().parent / "artifact-resources")
    return Pool(args.manifest, tokenizer, tuple(args.pool_domains.split(",")))


def run(args):
    lengths = tuple(int(x) for x in args.lengths.split(","))
    depths = tuple(float(x) for x in args.depths.split(","))
    pool = make_pool(args)
    cases = build_cases(pool, lengths, depths, args.needles_per_cell, args.master_seed, args.overhead_tokens)
    if args.dry_run:
        print(f"pool: {pool.n_tokens} tokens, text sha256 {pool.text_sha256}, sources {pool.sources}")
        print(f"{len(cases)} cases ({len(lengths)} lengths x {len(depths)} depths x {args.needles_per_cell} needles); no request sent")
        for c in cases:
            print(json.dumps({k: c[k] for k in ("id", "prompt_sha256")} | {"chars": len(c["prompt"])}))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    sent = 0
    if args.out.exists():
        done = {json.loads(l)["id"] for l in args.out.read_text(encoding="utf-8").splitlines() if l.strip()}
    for c in cases:
        if c["id"] in done:
            continue
        body = {"model": args.model, "messages": [{"role": "user", "content": c["prompt"]}], "max_tokens": MAX_NEW,
                "temperature": 0, "top_k": TOP_K, "seed": SEED, "stream": False}
        if thinking_value(args) is not None:
            body["enable_thinking"] = thinking_value(args)
        req = urllib.request.Request(args.url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
        row = {k: c[k] for k in ("id", "length", "depth", "needle", "name", "code", "pool_first_token", "pool_last_token",
                                 "actual_depth_tokens", "prompt_sha256")}
        row.update({"label": args.label, "utc": datetime.now(timezone.utc).isoformat()})
        t0 = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            reply = payload["choices"][0]["message"]["content"]
            row.update({"reply": reply, "correct": score(reply, c["code"]), "finish_reason": payload["choices"][0]["finish_reason"],
                        "prompt_tokens": payload["usage"]["prompt_tokens"], "completion_tokens": payload["usage"]["completion_tokens"],
                        "cache_n": payload["timings"]["cache_n"], "prompt_ms": payload["timings"]["prompt_ms"],
                        "request_params": {k: body.get(k) for k in ("temperature", "top_k", "seed", "max_tokens", "enable_thinking")}, "error": None})
        except urllib.error.HTTPError as exc:
            row.update({"reply": None, "correct": False, "error": f"{type(exc).__name__}: {exc} | " + exc.read().decode("utf-8", "replace")[:500]})
        except urllib.error.URLError as exc:
            if sent == 0:
                fail(f"cannot reach {args.url}: {exc.reason}. Start infernix-serve with this arm's flags "
                     f"(--model-id {args.model}) and pass its address with --url.")
            row.update({"reply": None, "correct": False, "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            row.update({"reply": None, "correct": False, "error": f"{type(exc).__name__}: {exc}"})
        sent += 1
        row["wall_s"] = time.monotonic() - t0
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({k: row.get(k) for k in ("id", "prompt_tokens", "cache_n", "correct", "reply", "error", "wall_s")}, ensure_ascii=False), flush=True)


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def summarize(args):
    arms = args.arms.split(",")
    rows = {}
    for a in arms:
        p = args.dir / f"{a}.jsonl"
        if not p.is_file():
            fail(f"{p} not found; run `needle run --label {a}` against that arm's server first, or drop it from --arms")
        rows[a] = {r["id"]: r for r in (json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip())}
    ids = list(rows[arms[0]])
    for a in arms:
        if list(rows[a]) != ids:
            fail(f"{a}: case list differs from {arms[0]} (incomplete or differently configured run)")
        for i in ids:
            if rows[a][i]["prompt_sha256"] != rows[arms[0]][i]["prompt_sha256"] or rows[a][i]["code"] != rows[arms[0]][i]["code"]:
                fail(f"{a}: prompt for case {i} differs from {arms[0]}; arms are only comparable on byte-identical prompts")
    lengths = sorted({rows[arms[0]][i]["length"] for i in ids})
    depths = sorted({rows[arms[0]][i]["depth"] for i in ids})
    per_cell = sum(1 for i in ids if rows[arms[0]][i]["length"] == lengths[0] and rows[arms[0]][i]["depth"] == depths[0])
    L, summary = [], {"arms": arms, "cases": len(ids), "grids": {}, "pairs": {}}
    L.append(f"{len(ids)} cases per arm; prompts byte-identical across arms (sha256 asserted). Exact-match correct / {per_cell} per cell.\n")
    for a in arms:
        L.append(f"**{a}** (correct/{per_cell} per cell; rows = depth, columns = nominal length)\n")
        L.append("| depth | " + " | ".join(
            f"{n // 1000}k (≈{mean([rows[a][i]['prompt_tokens'] for i in ids if rows[a][i]['length'] == n and rows[a][i].get('prompt_tokens')]):.0f} tok)"
            for n in lengths) + " |")
        L.append("| --- | " + " | ".join("---:" for _ in lengths) + " |")
        grid = {}
        for d in depths:
            cells = []
            for n in lengths:
                k = sum(rows[a][i]["correct"] for i in ids if rows[a][i]["length"] == n and rows[a][i]["depth"] == d)
                grid[f"{n}|{d}"] = k
                cells.append(f"{k}/{per_cell}")
            L.append(f"| {int(d * 100)} % | " + " | ".join(cells) + " |")
        tot = []
        for n in lengths:
            k = sum(rows[a][i]["correct"] for i in ids if rows[a][i]["length"] == n)
            nn = sum(1 for i in ids if rows[a][i]["length"] == n)
            lo, hi = wilson(k, nn)
            tot.append(f"{k}/{nn} [{100 * lo:.0f}–{100 * hi:.0f} %]")
        L.append("| **all depths** (Wilson 95 %) | " + " | ".join(tot) + " |")
        k = sum(rows[a][i]["correct"] for i in ids)
        lo, hi = wilson(k, len(ids))
        errs = sum(1 for i in ids if rows[a][i]["error"])
        L.append(f"\nOverall {k}/{len(ids)} = {100 * k / len(ids):.1f} % (Wilson 95 % {100 * lo:.1f}–{100 * hi:.1f} %); request errors: {errs}\n")
        summary["grids"][a] = {"cells": grid, "correct": k, "n": len(ids), "errors": errs}
    L.append("**Paired comparisons (exact McNemar on the same prompts; b = first right & second wrong, c = first wrong & second right)**\n")
    L.append("| pair | scope | both right | b | c | both wrong | p (two-sided exact) |")
    L.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for x in range(len(arms)):
        for y in range(x + 1, len(arms)):
            a1, a2 = arms[x], arms[y]
            for scope, sel in [("all", ids)] + [(f"{n // 1000}k", [i for i in ids if rows[a1][i]["length"] == n]) for n in lengths]:
                both = sum(rows[a1][i]["correct"] and rows[a2][i]["correct"] for i in sel)
                b = sum(rows[a1][i]["correct"] and not rows[a2][i]["correct"] for i in sel)
                c = sum((not rows[a1][i]["correct"]) and rows[a2][i]["correct"] for i in sel)
                neither = len(sel) - both - b - c
                p = mcnemar_exact(b, c)
                summary["pairs"][f"{a1} vs {a2}|{scope}"] = {"both": both, "b": b, "c": c, "neither": neither, "p": p}
                L.append(f"| {a1} vs {a2} | {scope} | {both} | {b} | {c} | {neither} | {p:.3f} |")
    text = "\n".join(L) + "\n"
    (args.dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.md or args.dir / "summary.md").write_text(text, encoding="utf-8")
    print(text)


def main():
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="send every case to the running server")
    r.add_argument("--label", required=True, help="arm name recorded in each row")
    r.add_argument("--out", type=Path, required=True, help="<arm>.jsonl (appended; completed cases are skipped)")
    r.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    r.add_argument("--model", default="kv-eval", help="model id; must equal the server's --model-id")
    r.add_argument("--timeout", type=float, default=1800)
    r.add_argument("--manifest", type=Path, required=True, help="corpus manifest.json (haystack source)")
    r.add_argument("--tokenizer", type=Path)
    r.add_argument("--artifact", type=Path, help="artifact whose embedded tokenizer.json is used when --tokenizer is absent")
    r.add_argument("--pool-domains", default=",".join(DEFAULT_DOMAINS), help="comma list of corpus domains forming the haystack pool")
    r.add_argument("--lengths", default=",".join(map(str, DEFAULT_LENGTHS)), help="nominal prompt lengths in tokens")
    r.add_argument("--depths", default=",".join(map(str, DEFAULT_DEPTHS)), help="needle depths as fractions of the text")
    r.add_argument("--needles-per-cell", type=int, default=NEEDLES_PER_CELL)
    r.add_argument("--overhead-tokens", type=int, default=OVERHEAD_TOKENS, help="template/question allowance subtracted from the nominal length")
    r.add_argument("--master-seed", type=int, default=MASTER_SEED)
    r.add_argument("--dry-run", action="store_true", help="build the cases and print prompt hashes; contact no server")
    add_thinking_arg(r, "false", "Defaults to false: the 32-token answer budget cannot hold a reasoning block, "
                     "so with thinking on the reply comes back empty. ")
    s = sub.add_parser("summarize", help="grids, Wilson intervals and paired exact McNemar across arms")
    s.add_argument("--arms", required=True)
    s.add_argument("--dir", type=Path, required=True, help="directory holding <arm>.jsonl")
    s.add_argument("--md", type=Path)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
