"""Controlled scripted-divergence tail probe; does not use generated replies as benchmark input.

Request 1 generates up to 256 tokens to exercise sliding-ring movement. Request 2 replaces that
reply with a fixed corpus-derived assistant message plus a fixed user tail. All request message
bytes are identical across configurations for a given rep. This intentionally measures divergence
from the generated reply, not a claim that the live generated continuation is reusable.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bench.gemma4_common import add_pool_args, add_thinking_arg, make_pool, chat, gpu_state, thinking_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_pool_args(ap)
    ap.add_argument('--url', default='http://127.0.0.1:8041/v1/chat/completions')
    ap.add_argument('--model', default='gemma-bench')
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--history', type=int, default=100000)
    ap.add_argument('--reps', type=int, default=3)
    add_thinking_arg(ap, 'default')
    args = ap.parse_args()
    thinking = thinking_value(args)
    pool = make_pool(args, args.out.resolve().parent)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as fh:
        for rep in range(args.reps):
            hist, first, last = pool.window(rep * 100000, args.history - 100)
            reply, _, end_reply = pool.window(last + 400, 256)
            tail, _, _ = pool.window(end_reply + 400, 1150)
            m1 = [{'role': 'user', 'content': f'Fixed-tail reference {rep}-{first}\nSummarize the material in detail.\n{hist}'}]
            m2 = m1 + [{'role': 'assistant', 'content': reply},
                       {'role': 'user', 'content': f'{tail}\nContinue the summary using this additional material.'}]
            before = gpu_state()
            r1, w1 = chat(args.url, args.model, m1, 256, enable_thinking=thinking)
            r2, w2 = chat(args.url, args.model, m2, 64, enable_thinking=thinking)
            row = {'label': args.label, 'rep': rep, 'scenario': 'scripted-reply-divergence',
                   'enable_thinking': args.enable_thinking,
                   'messages1_sha256': hashlib.sha256(json.dumps(m1, ensure_ascii=False).encode()).hexdigest(),
                   'messages2_sha256': hashlib.sha256(json.dumps(m2, ensure_ascii=False).encode()).hexdigest(),
                   'gpu_before': before, 'gpu_after': gpu_state(),
                   't1': r1.get('timings'), 't2': r2.get('timings'),
                   't1_usage': r1.get('usage'), 't2_usage': r2.get('usage'),
                   't1_wall_s': w1, 't2_wall_s': w2}
            fh.write(json.dumps(row) + '\n')
            fh.flush()
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
