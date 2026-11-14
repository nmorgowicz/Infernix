# Ported from pp-bench-20261108/rp_ladder.py (ninfer.spo campaign).
"""MTP draft-count x context ladder under the operator launcher settings (FP8 KV, FP8-QK, chunk auto).

One server per arm (MTP off, draft 1/2/4), then for each context size a fresh corpus-derived chat history
(different corpus offset per size, so no prefix reuse across sizes), three sampled tail turns (roleplay sampling),
and one greedy probe (temperature 0, same messages, no append). Records the server's own timings. prompt_ms of a
tail turn is the TTFT proxy (template/tokenization are outside it). Replies differ between arms, so compare
medians; one session per cell.
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import (add_pool_args, add_server_args, add_thinking_arg, make_pool, server_argv,
                                         gpu_lock, running_server, gpu_state, chat, thinking_value)
from tools.bench.kv_eval.common import sha256_file

PERSONA = ("You are Mara, a wry harbor-master in a rain-soaked port city. Stay in character, answer in vivid "
           "second-person prose, and never mention these instructions.")

HERE = Path(__file__).resolve().parent
CONTEXTS = [4000, 16000, 32000, 64000, 90000]
ARMS = {
    "off": ["--spec", "off"],
    "d1": ["--spec", "mtp", "--draft-tokens", "1"],
    "d2": ["--spec", "mtp", "--draft-tokens", "2"],
    "d4": ["--spec", "mtp", "--draft-tokens", "4"],
}


def build(pool, hist, offset, turns):
    text, first, last = pool.window(offset, hist)
    chunks, cur = [], ""
    for p in [p for p in text.split("\n") if p.strip()]:
        cur += p + "\n"
        if len(cur) > 3200:
            chunks.append(cur.strip())
            cur = ""
    if cur:
        chunks.append(cur.strip())
    extra, _, _ = pool.window(last + 400, 1500)
    ex = [p for p in extra.split("\n") if p.strip()]
    turn_msgs = [" ".join(ex[i * 3:(i + 1) * 3])[:1400] for i in range(turns)]
    msgs = [{"role": "system", "content": PERSONA}]
    for i, c in enumerate(chunks):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": c})
    if msgs[-1]["role"] == "user":
        msgs.append({"role": "assistant", "content": "The rain keeps drumming on the harbor office roof."})
    return msgs, turn_msgs


def request(args, msgs, seed, temperature):
    payload, wall = chat(f"http://127.0.0.1:{args.port}/v1/chat/completions", args.model, msgs, 300,
                         temperature=temperature, seed=seed, top_k=64, enable_thinking=thinking_value(args))
    t = payload["timings"]
    row = {"prompt_n": t["prompt_n"], "cache_n": t["cache_n"], "ttft_proxy_s": t["prompt_ms"] / 1000,
           "pp_tok_s": t["prompt_n"] / t["prompt_ms"] * 1000 if t["prompt_ms"] else None,
           "tg_tok_s": t["predicted_per_second"], "predicted_n": t["predicted_n"], "draft_n": t.get("draft_n"),
           "draft_accepted": t.get("draft_n_accepted"), "wall_s": wall, "enable_thinking": args.enable_thinking,
           "reply_head": payload["choices"][0]["message"]["content"][:60]}
    return row, payload["choices"][0]["message"]["content"]


def session(args, pool, hist, offset):
    msgs, turns = build(pool, hist, offset, 3)
    rows = []
    # Ingest (fresh) with a short greedy-free sampled request; not part of the tail statistics.
    msgs.append({"role": "user", "content": "Continue the scene."})
    row, reply = request(args, msgs, 1234, 1.0)
    row["kind"] = "ingest"
    rows.append(row)
    msgs.append({"role": "assistant", "content": reply})
    for i, user in enumerate(turns):
        msgs.append({"role": "user", "content": user})
        row, reply = request(args, msgs, 2000 + i, 1.0)
        row["kind"] = "tail"
        rows.append(row)
        msgs.append({"role": "assistant", "content": reply})
    # Greedy probe: same history, new user turn, temperature 0, not appended.
    probe = msgs + [{"role": "user", "content": "Describe the next hour in the harbor."}]
    row, _ = request(args, probe, 1, 0.0)
    row["kind"] = "greedy"
    rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_pool_args(ap)
    add_server_args(ap)
    ap.add_argument("--out-dir", type=Path, required=True)
    add_thinking_arg(ap, "default")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--contexts", default=",".join(map(str, CONTEXTS)))
    args = ap.parse_args()
    contexts = [int(c) for c in args.contexts.split(",")]
    names = args.arms.split(",")
    if not contexts or min(contexts) <= 0 or any(n not in ARMS for n in names):
        ap.error("contexts must be positive and arms must come from " + ",".join(ARMS))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = make_pool(args, args.out_dir.resolve())
    span = pool.n_tokens - max(contexts) - 4000
    if span < 20000:
        ap.error("corpus pool is too short for the requested context ladder")
    offsets = [int(20000 + i * (span - 20000) / max(1, len(contexts) - 1)) for i in range(len(contexts))]
    failed = False
    with gpu_lock(args.lock, "rp-ladder"):
        for name in names:
            out = args.out_dir / f"{name}.json"
            if out.exists():
                ap.error(f"output exists: {out}; use a fresh output directory")
            argv = server_argv(args, ARMS[name])
            rec = {"arm": name, "argv": argv, "binary_sha256": sha256_file(args.exe), "contexts": {},
                   "enable_thinking": args.enable_thinking,
                   "artifact": str(args.artifact), "artifact_bytes": args.artifact.stat().st_size,
                   "pool_sha256": pool.text_sha256, "idle_gpu": gpu_state(), "vram": {}}
            try:
                with running_server(args, argv, args.out_dir / f"{name}.server.log") as ready:
                    rec["ready_s"] = ready
                    rec["gpu_ready"] = gpu_state()
                    for hist, offset in zip(contexts, offsets):
                        rows = session(args, pool, hist, offset)
                        rec["contexts"][str(hist)] = rows
                        rec["vram"][str(hist)] = gpu_state()
                        tails = [row for row in rows if row["kind"] == "tail"]
                        print(name, hist, "tail tg", statistics.median(row["tg_tok_s"] for row in tails),
                              "acceptance", [(row["draft_accepted"], row["draft_n"]) for row in tails], flush=True)
                rec["complete"] = True
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            finally:
                out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
