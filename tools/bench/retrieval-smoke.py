#!/usr/bin/env python3
"""Six deterministic long-context retrieval cases; smoke, not a quality benchmark.

It uses the in-tree needle instrument
for corpus selection, prompt construction and exact six-digit answer scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import add_pool_args, add_thinking_arg, chat, make_pool, thinking_value
from tools.bench.kv_eval.needle import build_cases, score, MASTER_SEED, OVERHEAD_TOKENS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_pool_args(parser)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8041/v1/chat/completions")
    parser.add_argument("--model", default="gemma-bench")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    add_thinking_arg(parser, "false", "Defaults to false: the 32-token answer budget cannot hold a reasoning block, "
                     "so with thinking on the reply comes back empty. ")
    args = parser.parse_args()
    pool = make_pool(args, args.out.resolve().parent)
    cases = build_cases(pool, (32000, 100000), (0.05, 0.50, 0.95), 1, MASTER_SEED, OVERHEAD_TOKENS)
    if args.dry_run:
        for case in cases:
            print(json.dumps({"id": case["id"], "prompt_sha256": case["prompt_sha256"]}))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.out.open("x", encoding="utf-8") as handle:
        for case in cases:
            row = {key: value for key, value in case.items() if key != "prompt"}
            row["label"] = args.label
            row["enable_thinking"] = args.enable_thinking
            started = time.monotonic()
            try:
                payload, wall = chat(args.url, args.model, [{"role": "user", "content": case["prompt"]}],
                                     32, timeout=args.timeout, enable_thinking=thinking_value(args))
                reply = payload["choices"][0]["message"]["content"]
                row.update({"reply": reply, "correct": score(reply, case["code"]), "error": None,
                            "timings": payload.get("timings"), "usage": payload.get("usage"), "wall_s": wall})
            except (OSError, ValueError, KeyError, IndexError) as exc:
                row.update({"reply": None, "correct": False, "error": f"{type(exc).__name__}: {exc}",
                            "wall_s": time.monotonic() - started})
            failed |= not row["correct"]
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps(row, ensure_ascii=False), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
