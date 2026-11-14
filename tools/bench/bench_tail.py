# Ported from pp-bench-20261108/bench_tail.py (ninfer.spo campaign).
"""SillyTavern-style cached-tail probe against one running OpenAI-compatible server.

Replicates what a chat frontend does: request 1 ingests a large history (unique corpus window per rep,
so the cache starts cold), the model answers 256 tokens; request 2 resends THE SAME conversation plus the
assistant reply and a new user message (~tail tokens). A prefix cache then has to reprocess only the tail.
We record each request's server timings (prompt_n/prompt_ms = actually-processed tokens, predicted_per_second,
usage.prompt_tokens = full length) so the "tail pp" the frontend console shows is measured directly.

Usage: bench_tail.py --label ninfer-mtp --out tail/ninfer.cells.jsonl --url http://127.0.0.1:8031/v1/chat/completions
      --model 131k --history 100000 --tail 1200 --reps 3
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import add_pool_args, add_thinking_arg, make_pool, chat, gpu_state, thinking_value

SEED = 1234
GEN_TOKENS = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_pool_args(ap)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8031/v1/chat/completions")
    ap.add_argument("--model", default="gemma-bench")
    ap.add_argument("--history", type=int, default=100000)
    ap.add_argument("--tail", type=int, default=1200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--stride", type=int, default=61000)
    add_thinking_arg(ap, "default")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    thinking = thinking_value(args)

    pool = make_pool(args, args.out.resolve().parent)
    print(f"pool tokens: {pool.n_tokens} text sha {pool.text_sha256[:16]}", flush=True)
    task = "Here is source material, then instructions. At the end, answer in at least 800 words."
    question = ("Now continue with this additional material, then answer: give a detailed checklist that "
                "follows from everything above, at least 400 words.")

    with open(args.out, "x", encoding="utf-8", newline="\n") as fh:
        for rep in range(args.reps):
            base = rep * args.stride + args.history * rep
            hist, first, last = pool.window(base, args.history - 90)
            tail, tfirst, tlast = pool.window(last + 500, args.tail - 60)

            msgs1 = [{"role": "user", "content": f"Reference id rp{rep:02d}-{first}-{last}\n\n{task}\n\n{hist}"}]
            before = gpu_state()
            r1, w1 = chat(args.url, args.model, msgs1, GEN_TOKENS, enable_thinking=thinking)
            reply = r1["choices"][0]["message"]["content"]

            msgs2 = msgs1 + [{"role": "assistant", "content": reply},
                             {"role": "user", "content": f"Reference id rp{rep:02d}-tail-{tfirst}\n\n{question}\n\n{tail}"}]
            r2, w2 = chat(args.url, args.model, msgs2, GEN_TOKENS, enable_thinking=thinking)

            row = {"label": args.label, "rep": rep, "utc": datetime.now(timezone.utc).isoformat(),
                   "enable_thinking": args.enable_thinking,
                   "gpu_before": before,
                   "t1_wall_s": round(w1, 3), "t1": r1.get("timings"), "t1_usage": r1.get("usage"),
                   "t2_wall_s": round(w2, 3), "t2": r2.get("timings"), "t2_usage": r2.get("usage"),
                   "gpu_after": gpu_state()}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            t1, t2 = row["t1"] or {}, row["t2"] or {}
            print(f"rep{rep}: ingest prompt_n={t1.get('prompt_n')} pp={(t1.get('prompt_n') or 0) / max(t1.get('prompt_ms', 1) / 1000, 1e-9):.0f} tok/s | "
                  f"tail prompt_n={t2.get('prompt_n')} pp={(t2.get('prompt_n') or 0) / max(t2.get('prompt_ms', 1) / 1000, 1e-9):.0f} tok/s | "
                  f"tg1={t1.get('predicted_per_second')} tg2={t2.get('predicted_per_second')}", flush=True)


if __name__ == "__main__":
    main()
