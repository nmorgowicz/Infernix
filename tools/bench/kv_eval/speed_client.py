#!/usr/bin/env python3
"""Speed client for one running infernix-serve instance (HTTP /v1/chat/completions).

  python -m tools.bench.kv_eval.speed_client --label <arm>-<mtp|nomtp> --out <out-dir>/<label>.cells.jsonl --url <url> \
      --model <model id> --manifest <corpus-manifest> (--artifact <artifact> | --tokenizer <tokenizer.json>) [--dry-run]

For each nominal prompt length: rep 0 is a WARMUP (recorded, excluded from statistics) followed by --reps timed
repetitions. Every request has a UNIQUE natural-text prompt (a different corpus window per repetition and a unique
reference id on the first line), so prefix reuse cannot cover more than the chat-template head; summaries flag any
timed request whose cache_n exceeds that head. Sampling is pinned: temperature 0, top_k 20, seed 1234, 256 new
tokens. Per request the server's own `timings` (prompt_n/prompt_ms, predicted_per_second, draft_n/draft_n_accepted) and
`usage.prompt_tokens` (the exact prompt length) are stored, with nvidia-smi readings at the start and end of each cell.
Run on an otherwise idle GPU, one server at a time (see gpu_state.py). --dry-run builds the prompts and prints their
hashes without contacting a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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

OVERHEAD_TOKENS = 75          # template + instruction + reference id, subtracted from the nominal length
WINDOW_STRIDE_TOKENS = 61000  # corpus offset between repetitions
SEED = 1234
TOP_K = 20
MAX_NEW = 256


def gpu_reading() -> str:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def build_prompt(pool, nominal, rep, overhead=OVERHEAD_TOKENS, window_stride=WINDOW_STRIDE_TOKENS):
    body, first, last = pool.window(rep * window_stride, nominal - overhead)
    ref = hashlib.sha256(f"speed|{nominal}|{rep}".encode()).hexdigest()[:16]
    prompt = (f"Reference id {ref}. The document below is background material.\n\n{body}\n\n"
              "Write a detailed, multi-paragraph discussion of the document above, at least 800 words.")
    return prompt, {"ref": ref, "pool_first_token": first, "pool_last_token": last,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}


def request(url, model, prompt, timeout, max_new, enable_thinking=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_new,
            "temperature": 0, "top_k": TOP_K, "seed": SEED, "stream": False}
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload, time.monotonic() - t0, body


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="e.g. fp8-mtp; speed_summary.py expects <arm>-<mode>")
    ap.add_argument("--out", type=Path, required=True, help="<label>.cells.jsonl (appended)")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1/chat/completions")
    ap.add_argument("--model", default="kv-eval", help="model id; must equal the server's --model-id")
    ap.add_argument("--manifest", type=Path, required=True, help="corpus manifest.json (prompt text source)")
    ap.add_argument("--tokenizer", type=Path)
    ap.add_argument("--artifact", type=Path, help="artifact whose embedded tokenizer.json is used when --tokenizer is absent")
    ap.add_argument("--pool-domains", default=",".join(DEFAULT_DOMAINS))
    ap.add_argument("--lengths", default="1000,8000,32000,100000", help="nominal prompt lengths in tokens")
    ap.add_argument("--reps", type=int, default=5, help="timed repetitions per length (plus one warmup)")
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--overhead-tokens", type=int, default=OVERHEAD_TOKENS)
    ap.add_argument("--window-stride-tokens", type=int, default=WINDOW_STRIDE_TOKENS)
    ap.add_argument("--timeout", type=float, default=1800)
    add_thinking_arg(ap, "default")
    ap.add_argument("--dry-run", action="store_true", help="build the prompts and print their hashes; contact no server")
    args = ap.parse_args()

    tokenizer, _ = resolve_tokenizer(args.tokenizer, args.artifact, args.out.resolve().parent / "artifact-resources")
    pool = Pool(args.manifest, tokenizer, tuple(args.pool_domains.split(",")))
    print(f"pool: {pool.n_tokens} tokens, text sha256 {pool.text_sha256}, sources {pool.sources}", flush=True)
    lengths = [int(x) for x in args.lengths.split(",")]
    if args.dry_run:
        for nominal in lengths:
            for rep in range(args.reps + 1):
                _, meta = build_prompt(pool, nominal, rep, args.overhead_tokens, args.window_stride_tokens)
                print(json.dumps({"nominal": nominal, "rep": rep, "warmup": rep == 0, **meta}))
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def emit(row):
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    for nominal in lengths:
        emit({"kind": "gpu", "when": "cell-start", "label": args.label, "nominal": nominal, "gpu": gpu_reading(),
              "utc": datetime.now(timezone.utc).isoformat()})
        for rep in range(args.reps + 1):               # rep 0 is the warmup
            prompt, meta = build_prompt(pool, nominal, rep, args.overhead_tokens, args.window_stride_tokens)
            try:
                payload, wall, body = request(args.url, args.model, prompt, args.timeout, args.max_new,
                                              thinking_value(args))
                err = None
            except urllib.error.HTTPError as exc:      # a cell that does not fit must be reported, not skipped
                payload, wall, body = None, None, None
                err = f"{type(exc).__name__}: {exc} | " + exc.read().decode("utf-8", "replace")[:500]
            except urllib.error.URLError as exc:
                if rep == 0 and nominal == lengths[0]:
                    fail(f"cannot reach {args.url}: {exc.reason}. Start infernix-serve with this arm's flags "
                         f"(--model-id {args.model}) and pass its address with --url.")
                payload, wall, body, err = None, None, None, f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                payload, wall, body, err = None, None, None, f"{type(exc).__name__}: {exc}"
            row = {"kind": "request", "label": args.label, "nominal": nominal, "rep": rep, "warmup": rep == 0,
                   "utc": datetime.now(timezone.utc).isoformat(), "wall_s": wall, "error": err, **meta}
            if payload is not None:
                t, u = payload["timings"], payload["usage"]
                row.update({
                    "prompt_tokens": u["prompt_tokens"], "completion_tokens": u["completion_tokens"],
                    "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens"),
                    "prompt_n": t["prompt_n"], "cache_n": t["cache_n"], "prompt_ms": t["prompt_ms"],
                    "prefill_tok_s": t["prompt_n"] / t["prompt_ms"] * 1000.0 if t["prompt_ms"] > 0 else None,
                    "predicted_n": t["predicted_n"], "predicted_ms": t["predicted_ms"],
                    "decode_tok_s": t["predicted_per_second"],
                    "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
                    "finish_reason": payload["choices"][0]["finish_reason"],
                    "text_head": payload["choices"][0]["message"]["content"][:120],
                    "request_params": {k: body.get(k) for k in ("temperature", "top_k", "seed", "max_tokens", "enable_thinking")},
                    "enable_thinking": args.enable_thinking})
            emit(row)
            print(json.dumps({k: row.get(k) for k in ("label", "nominal", "rep", "prompt_tokens", "prompt_n", "cache_n",
                  "prefill_tok_s", "predicted_n", "decode_tok_s", "draft_n", "draft_n_accepted", "finish_reason",
                  "error")}), flush=True)
            if err:
                break
        emit({"kind": "gpu", "when": "cell-end", "label": args.label, "nominal": nominal, "gpu": gpu_reading(),
              "utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
