"""Cached-tail probe with long thinking replies, as roleplay with thinking on produces them (several thousand reasoning
tokens plus the answer), under roleplay sampling (temperature 1, top-p 0.95, top-k 64).

One server per arm. For each context size: a corpus-derived history behind a long system prompt (--system-tokens), an ingest
turn, then --turns sampled tail turns with max_tokens --max-tokens, so the model may think as long as it likes. A client
keeps only `content` in the history, as chat frontends do, so each next prompt re-renders the previous answer without its
reasoning and diverges from the state the previous request left behind. Per turn this records the generated length
(`predicted_n`, thinking plus answer), the computed and cached prompt tokens, the server's prompt time (TTFT proxy), decode
speed, draft acceptance and the server log's line for the request (it names the prefix-reuse path).

With --preset, the system prompt and in-chat insertions come from a SillyTavern preset, as that frontend sends them.

The question it answers: how does the next turn's first-token time depend on how much the previous turn generated?

usage: python -m tools.bench.think_tail --exe <infernix-serve> --artifact A --manifest M --out-dir D --contexts 32000,100000
"""
from __future__ import annotations

import random
import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import (add_pool_args, add_server_args, add_thinking_arg, chat, gpu_lock, gpu_state,
                                        make_pool, running_server, server_argv, thinking_value)
from tools.bench.kv_eval.common import sha256_file
from tools.bench.rp_ladder import ARMS, PERSONA, build
from tools.bench import st_preset



def log_line_for(log: Path, start: int) -> str:
    """The server log's last request line written after byte offset `start` (it names the prefix-reuse path)."""
    try:
        with log.open("rb") as handle:
            handle.seek(start)
            lines = [line for line in handle.read().decode("utf-8", "replace").splitlines() if line.strip()]
    except OSError:
        return ""
    return lines[-1][:400] if lines else ""


def turn(args, url, log: Path, msgs, seed):
    start = log.stat().st_size if log.exists() else 0
    extra = {"min_p": 0.03, "chat_template_kwargs": {"enable_thinking": True, "preserve_reasoning": True,
                                                       "reasoning_effort": "medium"}} if args.layout else None
    payload, wall = chat(url, args.model, msgs, args.max_tokens, temperature=1.0, seed=seed, top_k=64,
                         enable_thinking=thinking_value(args), extra=extra)
    t = payload["timings"]
    choice = payload["choices"][0]
    message = choice["message"]
    row = {"prompt_n": t["prompt_n"], "cache_n": t["cache_n"], "ttft_proxy_s": t["prompt_ms"] / 1000,
           "tg_tok_s": t["predicted_per_second"], "predicted_n": t["predicted_n"], "draft_n": t.get("draft_n"),
           "draft_accepted": t.get("draft_n_accepted"), "wall_s": wall, "finish_reason": choice.get("finish_reason"),
           "content_chars": len(message.get("content") or ""),
           "reasoning_chars": len(message.get("reasoning_content") or ""), "server_log": log_line_for(log, start)}
    return row, message.get("content") or ""


def sent_messages(args, system, history):
    """The request's messages: the clean history behind the system prompt, or the preset's layout around it. The preset's
    macros are drawn afresh for every request (a roll in front of the history changes the prompt there each time)."""
    if args.layout is None:
        return system + history
    args.requests += 1
    before, injections, after = args.layout.render(random.Random(20261115 + args.requests))
    return st_preset.apply(before, history, injections, after)


def session(args, pool, url, log, hist, offset):
    base, turns = build(pool, hist, offset, args.turns)
    system, history = base[:1], base[1:]
    filler, _, _ = pool.window(offset + hist + 6000, args.system_tokens)
    system = [{"role": "system", "content": PERSONA + "\n\nSetting notes:\n" + filler.strip()}]
    rows = []
    history.append({"role": "user", "content": "Continue the scene."})
    row, reply = turn(args, url, log, sent_messages(args, system, history), 1234)
    row["kind"] = "ingest"
    rows.append(row)
    history.append({"role": "assistant", "content": reply})
    for i, user in enumerate(turns):
        history.append({"role": "user", "content": user})
        row, reply = turn(args, url, log, sent_messages(args, system, history), 2000 + i)
        row["kind"] = "tail"
        rows.append(row)
        history.append({"role": "assistant", "content": reply})
        print(f"  tail {i}: generated {row['predicted_n']} (reasoning {row['reasoning_chars']} chars), computed "
              f"{row['prompt_n']}, cached {row['cache_n']}, ttft {row['ttft_proxy_s']:.2f}s, {row['server_log'][-80:]}",
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pool_args(ap)
    add_server_args(ap)
    ap.add_argument("--out-dir", type=Path, required=True)
    add_thinking_arg(ap, "true")
    ap.add_argument("--arms", default="d1")
    ap.add_argument("--contexts", default="32000,100000")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--system-tokens", type=int, default=3000)
    ap.add_argument("--preset", type=Path, help="a SillyTavern chat-completion preset (exported JSON): its prompts become "
                    "the system messages before and after the history and its in-chat prompts are inserted at their depth "
                    "on every request (tools/bench/st_preset.py), and requests carry its sampling (min_p, "
                    "chat_template_kwargs); top_k stays 64, Gemma's value")
    ap.add_argument("--arm-argv-json", type=Path, help="JSON object {arm: complete server argv}; another engine as the "
                    "control. The server log is then that engine's, so `server_log` may not name the reuse path.")
    args = ap.parse_args()
    args.layout = st_preset.Preset(args.preset) if args.preset else None
    args.requests = 0
    overrides = json.loads(args.arm_argv_json.read_text(encoding="utf-8")) if args.arm_argv_json else None
    names = args.arms.split(",")
    contexts = [int(c) for c in args.contexts.split(",")]
    if any(n not in ARMS for n in names) or min(contexts) <= 0:
        ap.error("arms must come from " + ",".join(ARMS) + " and contexts must be positive")
    if overrides is not None and any(n not in overrides for n in names):
        ap.error("--arm-argv-json must give an argv for every selected arm")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = make_pool(args, args.out_dir.resolve())
    span = pool.n_tokens - max(contexts) - args.system_tokens - 12000
    if span < 20000:
        ap.error("corpus pool is too short for the requested contexts")
    offsets = [int(20000 + i * (span - 20000) / max(1, len(contexts) - 1)) for i in range(len(contexts))]
    failed = False
    with gpu_lock(args.lock, "think-tail"):
        for name in names:
            out = args.out_dir / f"{name}.json"
            if out.exists():
                ap.error(f"output exists: {out}; use a fresh output directory")
            argv = overrides[name] if overrides else server_argv(args, ARMS[name])
            log = args.out_dir / f"{name}.server.log"
            rec = {"arm": name, "argv": argv, "binary_sha256": sha256_file(Path(argv[0])), "contexts": {},
                   "enable_thinking": args.enable_thinking, "max_tokens": args.max_tokens,
                   "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}, "system_tokens": args.system_tokens,
                   "artifact": str(args.artifact), "pool_sha256": pool.text_sha256, "idle_gpu": gpu_state(), "vram": {}}
            try:
                with running_server(args, argv, log) as ready:
                    rec["ready_s"] = ready
                    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
                    for hist, offset in zip(contexts, offsets):
                        print(name, hist, flush=True)
                        rows = session(args, pool, url, log, hist, offset)
                        rec["contexts"][str(hist)] = rows
                        rec["vram"][str(hist)] = gpu_state()
                        tails = [r for r in rows if r["kind"] == "tail"]
                        print(name, hist, "median generated", statistics.median(r["predicted_n"] for r in tails),
                              "median ttft", statistics.median(r["ttft_proxy_s"] for r in tails), flush=True)
                rec["complete"] = True
                if args.layout is not None:
                    rec["preset"] = {"path": str(args.preset), "volatile_macros": sorted(args.layout.volatile),
                                     "ignored_macros": sorted(args.layout.ignored)}
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                failed = True
            finally:
                out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
