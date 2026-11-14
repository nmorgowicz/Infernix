#!/usr/bin/env python3
"""ABAB serving decode campaign, ported from pp-bench-20261108/decode_abab.py.

Two explicit arms replace the fork-only decode environment selector. --arms-json is an
object with `a` and `b` objects: optional `exe`, `args` argv list, and `env` string map.
Executable paths are relative to that JSON file, or absolute. Every run starts a fresh
server, then measures unique corpus prompts with a discarded warmup at each length.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import add_pool_args, add_server_args, add_thinking_arg, gpu_lock, running_server, server_argv
from tools.bench.kv_eval.common import sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_pool_args(ap)
    add_server_args(ap)
    ap.add_argument("--arms-json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--lengths", default="1000,8000,32000,100000")
    ap.add_argument("--mtp-draft", type=int, choices=range(1, 8))
    ap.add_argument("--tag", default="decode")
    ap.add_argument("--reps", type=int, default=3)
    add_thinking_arg(ap, "default")
    args = ap.parse_args()
    arms = json.loads(args.arms_json.read_text(encoding="utf-8"))
    for key in ("a", "b"):
        arm = arms.get(key)
        if not isinstance(arm, dict) or not isinstance(arm.get("args", []), list) or not all(isinstance(s, str) for s in arm.get("args", [])):
            ap.error(f"arms-json needs {key} with an args string list")
        if not isinstance(arm.get("env", {}), dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in arm.get("env", {}).items()):
            ap.error(f"arm {key} env must be a string map")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    with gpu_lock(args.lock, "decode-abab"):
        for index, name in enumerate(("a", "b", "a", "b"), 1):
            label = f"{args.tag}-{name}-{index}"
            cells = args.out_dir / f"{label}.cells.jsonl"
            if cells.exists():
                ap.error(f"output exists: {cells}; use a fresh output directory")
            arm = arms[name]
            spec = ["--spec", "mtp", "--draft-tokens", str(args.mtp_draft)] if args.mtp_draft else ["--spec", "off"]
            argv = server_argv(args, spec) + arm.get("args", [])
            if arm.get("exe"):
                argv[0] = str((args.arms_json.resolve().parent / arm["exe"]).resolve())
            env = dict(os.environ, **arm.get("env", {}))
            rec = {"arm": name, "argv": argv, "env": arm.get("env", {}), "binary_sha256": sha256_file(Path(argv[0])),
                   "enable_thinking": args.enable_thinking}
            try:
                with running_server(args, argv, args.out_dir / f"{label}.server.log", env) as ready:
                    rec["ready_s"] = ready
                    client = [sys.executable, "-m", "tools.bench.kv_eval.speed_client", "--label", label,
                              "--out", str(cells.resolve()), "--url", f"http://127.0.0.1:{args.port}/v1/chat/completions",
                              "--model", args.model, "--manifest", str(args.manifest.resolve()),
                              "--artifact", str(args.artifact.resolve()), "--pool-domains", args.pool_domains,
                              "--lengths", args.lengths, "--reps", str(args.reps),
                              "--enable-thinking", args.enable_thinking]
                    if args.tokenizer:
                        client += ["--tokenizer", str(args.tokenizer.resolve())]
                    rec["client_argv"] = client
                    with (args.out_dir / f"{label}.client.log").open("xb") as log:
                        completed = subprocess.run(client, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                    rec["client_exit"] = completed.returncode
                    if completed.returncode:
                        raise RuntimeError(f"speed client exited {completed.returncode}; see {label}.client.log")
                    rows = [json.loads(line) for line in cells.read_text(encoding="utf-8").splitlines() if line.strip()]
                    errors = [row["error"] for row in rows if row.get("error")]
                    if errors:
                        raise RuntimeError(f"speed requests failed: {errors}")
                rec["complete"] = True
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                (args.out_dir / f"{label}.receipt.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
