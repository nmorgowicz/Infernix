"""Sustained roleplay soak against ONE running OpenAI-compatible server (client only; the server is started by the caller,
for example the generated serve launcher, so the exact production command is what is tested).

A conversation starts from a corpus-derived chat history of --start-tokens, then grows turn by turn with sampled replies
(temperature 1, top-p 0.95, top-k 64, thinking as --enable-thinking states) until its prompt nears --context-limit; the next
conversation then starts at a fresh corpus window (a cold ingest), so cold ingest, host prefix-cache restore, decode with MTP and
long-context growth all repeat for --minutes. Every --abort-every turns a streamed request is dropped after its first chunk and
the next turn must still succeed (abort/recovery at depth). GPU memory is sampled in the background every --sample-seconds.

Pass rules (all reported, the exit code is 0 only if every one holds): no failed request; the server answers /health at the end; the
GPU memory of the last quarter of the run exceeds that of the second quarter (after warm-up) by no more than --vram-growth-mib; at
least --min-cached-fraction of the non-first turns of each conversation restored most of their prompt from the cache.
It measures stability, not output quality. Run it under the exclusive GPU lease.

usage: python -m tools.bench.roleplay_soak --manifest M --artifact A --url http://127.0.0.1:8001 --model 131k --out-dir D --minutes 120
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import threading
import time
import urllib.request

from tools.bench.gemma4_common import add_pool_args, add_thinking_arg, chat, gpu_state, make_pool, thinking_value
from tools.bench.rp_ladder import PERSONA, build


def sample_gpu(stop: threading.Event, series: list, every: float) -> None:
    while not stop.is_set():
        used = gpu_state().split(",")[0].strip()
        if used.isdigit():
            series.append((time.time(), int(used)))
        stop.wait(every)


def drop_streamed(base_url: str, model: str, messages: list, thinking) -> None:
    """Open a streamed completion and close the socket after the first data line: the server must survive the abort."""
    body = {"model": model, "messages": messages, "max_tokens": 600, "temperature": 1.0, "top_p": 0.95, "top_k": 64,
            "stream": True}
    if thinking is not None:
        body["enable_thinking"] = thinking
    request = urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    response = urllib.request.urlopen(request, timeout=600)
    try:
        for line in response:
            if line.startswith(b"data:"):
                break
    finally:
        response.close()


def turn_messages(pool, at: int) -> list[str]:
    """Roleplay-sized user turns (three paragraphs each) from a 30,000-token corpus window: hundreds of distinct turns."""
    text, _, _ = pool.window(at, 30000)
    paragraphs = [p for p in text.split("\n") if p.strip()]
    return [" ".join(paragraphs[i * 3:(i + 1) * 3])[:1400] for i in range(len(paragraphs) // 3)] or ["Go on."]


def health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
            return response.status == 200
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pool_args(ap)
    add_thinking_arg(ap, "true")
    ap.add_argument("--url", default="http://127.0.0.1:8001")
    ap.add_argument("--model", default="131k")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--minutes", type=float, default=120.0)
    ap.add_argument("--start-tokens", type=int, default=30000)
    ap.add_argument("--context-limit", type=int, default=118000, help="start a new conversation when the prompt exceeds this")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--abort-every", type=int, default=25)
    ap.add_argument("--sample-seconds", type=float, default=30.0)
    ap.add_argument("--vram-growth-mib", type=int, default=256)
    ap.add_argument("--min-cached-fraction", type=float, default=0.8)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pool = make_pool(args, args.out_dir.resolve())
    thinking = thinking_value(args)
    url = f"{args.url.rstrip('/')}/v1/chat/completions"
    series: list = []
    stop = threading.Event()
    sampler = threading.Thread(target=sample_gpu, args=(stop, series, args.sample_seconds), daemon=True)
    sampler.start()
    started = time.time()
    deadline = started + args.minutes * 60
    rows, errors, conversation, turn = [], [], 0, 0
    span = pool.n_tokens - args.start_tokens - 45000
    if span < 20000:
        ap.error("corpus pool is too short for --start-tokens plus the turn window")
    cursor = 20000
    log = (args.out_dir / "turns.jsonl").open("a", encoding="utf-8")
    try:
        while time.time() < deadline:
            offset = cursor % max(1, span)
            cursor += 70000
            messages, _ = build(pool, args.start_tokens, offset, 0)
            user_turns = turn_messages(pool, (offset + args.start_tokens + 3000) % span + 1000)
            messages.append({"role": "user", "content": "Continue the scene."})
            conversation += 1
            index, previous_prompt = 0, 0
            while time.time() < deadline:
                turn += 1
                try:
                    payload, wall = chat(url, args.model, messages, args.max_tokens, temperature=1.0, seed=turn,
                                         top_k=64, enable_thinking=thinking)
                    t = payload["timings"]
                    message = payload["choices"][0]["message"]
                    row = {"turn": turn, "conversation": conversation, "index": index, "prompt_n": t["prompt_n"],
                           "cache_n": t["cache_n"], "prompt_ms": t["prompt_ms"], "tg_tok_s": t["predicted_per_second"],
                           "predicted_n": t["predicted_n"], "draft_n": t.get("draft_n"),
                           "draft_accepted": t.get("draft_n_accepted"), "wall_s": wall,
                           "finish": payload["choices"][0]["finish_reason"],
                           "reasoning_tokens": (payload.get("usage", {}).get("completion_tokens_details") or {}).get(
                               "reasoning_tokens"), "empty_answer": not (message.get("content") or "").strip(),
                           "prompt_tokens": payload["usage"]["prompt_tokens"], "t": round(time.time() - started, 1)}
                except Exception as exc:  # a failed request is a finding, never silently retried
                    row = {"turn": turn, "conversation": conversation, "index": index, "error": f"{type(exc).__name__}: {exc}",
                           "t": round(time.time() - started, 1)}
                    errors.append(row)
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    break  # a conversation that errored is abandoned; the next one proves recovery
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
                reply = (message.get("content") or "").strip() or "..."
                messages.append({"role": "assistant", "content": reply})
                index += 1
                previous_prompt = row["prompt_tokens"]
                if previous_prompt + args.max_tokens + 400 > args.context_limit:
                    break
                messages.append({"role": "user", "content": user_turns[(index - 1) % len(user_turns)]})
                if args.abort_every and turn % args.abort_every == 0:
                    try:
                        drop_streamed(args.url.rstrip("/"), args.model, messages, thinking)
                    except Exception as exc:
                        errors.append({"turn": turn, "abort_probe_error": f"{type(exc).__name__}: {exc}"})
    finally:
        stop.set()
        sampler.join(timeout=5)
        log.close()

    elapsed = time.time() - started
    quarter = len(series) // 4
    second = [mib for _, mib in series[quarter:2 * quarter]]
    last = [mib for _, mib in series[3 * quarter:]]
    growth = (statistics.median(last) - statistics.median(second)) if second and last else None
    later = [r for r in rows if r["index"] > 0 and r["prompt_n"] is not None]
    cached = [r for r in later if r["cache_n"] >= 0.5 * (r["cache_n"] + r["prompt_n"])]
    cached_fraction = len(cached) / len(later) if later else None
    tails = [r for r in later if r["index"] > 0]
    summary = {
        "minutes": round(elapsed / 60, 1), "turns": len(rows), "conversations": conversation, "errors": errors,
        "alive_at_end": health(args.url.rstrip("/")), "enable_thinking": args.enable_thinking,
        "vram_mib_samples": len(series), "vram_mib_min": min((m for _, m in series), default=None),
        "vram_mib_max": max((m for _, m in series), default=None), "vram_growth_mib": growth,
        "cached_turn_fraction": cached_fraction,
        "empty_answer_turns": sum(1 for r in rows if r.get("empty_answer")),
        "median_tail_ms": statistics.median(r["prompt_ms"] for r in tails) if tails else None,
        "median_tg_tok_s": statistics.median(r["tg_tok_s"] for r in rows) if rows else None,
        "acceptance": (sum(r["draft_accepted"] or 0 for r in rows) / max(1, sum(r["draft_n"] or 0 for r in rows))),
        "max_prompt_tokens": max((r["prompt_tokens"] for r in rows), default=None),
    }
    summary["passed"] = bool(rows) and not errors and summary["alive_at_end"] \
        and (growth is None or growth <= args.vram_growth_mib) \
        and (cached_fraction is None or cached_fraction >= args.min_cached_fraction)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    (args.out_dir / "vram.json").write_text(json.dumps(series), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
