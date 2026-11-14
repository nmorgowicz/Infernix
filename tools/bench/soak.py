"""Near-capacity soak: one production-profile server, repeated large cached-tail conversations, then health checks.

Starts `infernix-serve` with the operator profile (FP8 KV, FP8 QK, chunk auto, host-only prefix cache, MTP when --draft-tokens
is non-zero), runs `bench_tail.py` (a ~HISTORY-token conversation, a 256-token reply, then the same conversation plus a new
~TAIL-token message, REPS unique corpus windows), and records what a long roleplay session depends on:
  - GPU memory after the server is ready and after the last request (growth shows as a difference);
  - whether the server process is still alive and still answers a sampled request (temperature 1, top-p 0.95, top-k 64);
  - the server log's error, fatal, abort and out-of-memory lines.
It is a stress pass, not a qualification: it says nothing about output quality. Run it under the exclusive GPU lease.

usage: python -m tools.bench.soak --manifest M --artifact A --exe infernix-serve --out-dir D [--history 120000 --tail 1200 --reps 3]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

from tools.bench.gemma4_common import (add_pool_args, add_server_args, add_thinking_arg, gpu_lock, gpu_state,
                                         running_server, server_argv)

BENCH_TAIL = Path(__file__).resolve().with_name("bench_tail.py")
ERROR_WORDS = ("error", "fatal", "abort", "out of memory")


def final_request(args):
    body = {"model": args.model, "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
            "max_tokens": 32, "temperature": 1.0, "top_p": 0.95, "top_k": 64}
    request = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                     data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.status, json.loads(response.read())["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pool_args(parser)
    add_server_args(parser)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history", type=int, default=120000)
    parser.add_argument("--tail", type=int, default=1200)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--draft-tokens", type=int, default=1, help="MTP draft tokens (0: speculation off)")
    add_thinking_arg(parser, "default")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = ["--spec", "mtp", "--draft-tokens", str(args.draft_tokens)] if args.draft_tokens else ["--spec", "off"]
    argv = server_argv(args, spec)
    log_path = args.out_dir / "server.log"
    receipt = {"server_argv": argv, "history": args.history, "tail": args.tail, "reps": args.reps,
               "enable_thinking": args.enable_thinking}
    exit_code = 1
    with gpu_lock(args.lock, "soak"):
        with running_server(args, argv, log_path) as ready_s:
            receipt["ready_s"] = ready_s
            receipt["gpu_mib_ready"] = gpu_state()
            client = subprocess.run(
                [sys.executable, str(BENCH_TAIL), "--manifest", str(args.manifest), "--artifact", str(args.artifact),
                 "--pool-domains", args.pool_domains, "--label", "soak", "--out", str(args.out_dir / "cells.jsonl"),
                 "--url", f"http://127.0.0.1:{args.port}/v1/chat/completions", "--model", args.model,
                 "--history", str(args.history), "--tail", str(args.tail), "--reps", str(args.reps),
                 "--enable-thinking", args.enable_thinking]
                + (["--tokenizer", str(args.tokenizer)] if args.tokenizer else []))
            receipt["client_exit"] = client.returncode
            receipt["gpu_mib_after"] = gpu_state()
            status, text = final_request(args)
            receipt["final_request"] = {"status": status, "text": text}
            exit_code = client.returncode or (0 if status == 200 else 1)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    receipt["log_error_lines"] = [line for line in text.splitlines() if any(w in line.lower() for w in ERROR_WORDS)][:20]
    receipt["passed"] = exit_code == 0 and not receipt["log_error_lines"]
    (args.out_dir / "receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("client_exit", "gpu_mib_ready", "gpu_mib_after", "final_request",
                                              "log_error_lines", "passed")}, indent=1))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
