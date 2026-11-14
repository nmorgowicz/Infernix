#!/usr/bin/env python3
"""Run one native infernix-perplexity arm, retaining argv, logs, hashes and GPU receipts.

Default: --save-top-tokens diagnostic consumed by analyze. Native full-vocabulary modes:
--full-vocab-reference-write CACHE or --full-vocab-reference CACHE. These are not the
existing target --kl-reference (top-token lower-bound) mode. Native prefix ids must match the cache.
Ported from ninfer.spo tools/bench/kv_eval/run_ppl.py.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.gemma4_common import gpu_lock
from tools.bench.kv_eval import gpu_state
from tools.bench.kv_eval.common import fail, find_arm, load_arms, read_json, sha256_file, use_utf8_stdio, utc_now
from tools.bench.kv_eval.metrics import open_dump

PROFILES = {"short": (4096, 2048), "long": (65536, 32768)}


def main():
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exe", type=Path, required=True)
    ap.add_argument("--artifact", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--arms-json", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=sorted(PROFILES), default="short")
    ap.add_argument("--context", type=int)
    ap.add_argument("--stride", type=int)
    ap.add_argument("--quick", action="store_true")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full-vocab-reference-write", type=Path)
    mode.add_argument("--full-vocab-reference", type=Path)
    ap.add_argument("--full-vocab-seed", type=int, default=20261105)
    ap.add_argument("--full-vocab-resamples", type=int, default=1000)
    ap.add_argument("--full-vocab-reference-kv-dtype",
                    choices=("bf16", "bf16p", "int8", "fp8", "nvfp4", "k8v4", "vq2", "k4v2"))
    ap.add_argument("--full-vocab-reference-gemma4-qk", choices=("bf16", "fp8"))
    prefix = ap.add_mutually_exclusive_group()
    prefix.add_argument("--prefix-ids-file", type=Path, help="derive_prefix_ids JSON or comma-separated ids; native modes only")
    prefix.add_argument("--prefix-ids", help="comma-separated ids passed as --full-vocab-prefix-ids")
    ap.add_argument("--save-tokens", type=Path, help="native window-token evidence for adapter parity")
    ap.add_argument("--artifact-sha256", help="explicit precomputed hash, recorded as supplied")
    ap.add_argument("--path-prepend", type=Path, action="append", default=[])
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--lock", type=Path, default=Path(tempfile.gettempdir()) / "infernix-bench-gpu.lock")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    arm = find_arm(load_arms(args.arms_json), args.arm, args.arms_json)
    if Path(arm.name).name != arm.name or arm.name in (".", ".."):
        fail("arm name must be a single filename component")
    context, stride = PROFILES[args.profile]
    context = args.context if args.context is not None else context
    stride = args.stride if args.stride is not None else stride
    if context < 2 or not 1 <= stride < context:
        fail("context/stride must satisfy context>=2 and 1<=stride<context")
    exe, artifact, corpus, out = (p.resolve() for p in (args.exe, args.artifact, args.corpus, args.out_dir))
    for path in (exe, artifact, corpus):
        if not path.is_file():
            fail(f"input not found: {path}")
    native = bool(args.full_vocab_reference or args.full_vocab_reference_write)
    if (args.full_vocab_reference_kv_dtype or args.full_vocab_reference_gemma4_qk) and not args.full_vocab_reference:
        fail("expected-reference configuration flags require --full-vocab-reference")
    prefix_ids = args.prefix_ids
    if args.prefix_ids_file:
        text = args.prefix_ids_file.read_text(encoding="utf-8").strip()
        if text.startswith("{"):
            doc = json.loads(text)
            prefix_ids = ",".join(map(str, doc["ids"])) if "ids" in doc else doc["ids_csv"]
        else:
            prefix_ids = text
    if prefix_ids is not None:
        if not native:
            fail("prefix ids require a native full-vocabulary mode")
        try:
            ids = [int(value.strip()) for value in prefix_ids.split(",")]
        except ValueError:
            fail("prefix ids must be comma-separated nonnegative integers")
        if not ids or min(ids) < 0:
            fail("prefix ids must be comma-separated nonnegative integers")
        prefix_ids = ",".join(map(str, ids))
    owned = {"--output", "--corpus", "--context", "--stride", "--save-top-tokens", "--dump-topk", "--prefix-ids",
             "--full-vocab-reference", "--full-vocab-reference-write", "--full-vocab-seed", "--full-vocab-resamples", "--kl-reference"}
    owned.update(("--full-vocab-reference-kv-dtype", "--full-vocab-reference-gemma4-qk", "--full-vocab-prefix-ids", "--save-tokens"))
    if any(token.split("=", 1)[0] in owned for token in arm.flags):
        fail("arms may set execution flags, not runner-owned inputs, outputs or reference modes")
    if native and "--logits-dtype" in arm.flags:
        index = arm.flags.index("--logits-dtype")
        if index + 1 >= len(arm.flags) or arm.flags[index + 1] != "bf16":
            fail("native full-vocabulary references require --logits-dtype bf16")
    arm_dir = out / arm.name
    dump = out / "dumps" / f"{arm.name}.topk"
    argv = [str(exe), str(artifact), "--corpus", str(corpus), "--context", str(context), "--stride", str(stride)]
    argv += ["--quick"] if args.quick else []
    argv += list(arm.flags)
    if native:
        argv += ["--logits-dtype", "bf16", "--full-vocab-seed", str(args.full_vocab_seed),
                 "--full-vocab-resamples", str(args.full_vocab_resamples)]
        reference = args.full_vocab_reference or args.full_vocab_reference_write
        flag = "--full-vocab-reference" if args.full_vocab_reference else "--full-vocab-reference-write"
        argv += [flag, str(reference.resolve())]
        if prefix_ids is not None:
            argv += ["--full-vocab-prefix-ids", prefix_ids]
        if args.full_vocab_reference_kv_dtype:
            argv += ["--full-vocab-reference-kv-dtype", args.full_vocab_reference_kv_dtype]
        if args.full_vocab_reference_gemma4_qk:
            argv += ["--full-vocab-reference-gemma4-qk", args.full_vocab_reference_gemma4_qk]
        if args.full_vocab_reference and not reference.is_file():
            fail(f"reference cache not found: {reference}")
        if args.full_vocab_reference_write and reference.exists():
            fail(f"reference cache exists: {reference}; preserve it and choose a fresh path")
    else:
        argv += ["--save-top-tokens", str(dump)]
    if args.save_tokens:
        argv += ["--save-tokens", str(args.save_tokens.resolve())]
    argv += ["--output", str(arm_dir)]
    if args.dry_run:
        print(json.dumps({"argv": argv, "cwd": str(exe.parent), "native_full_vocab": native}, indent=2))
        return
    if arm_dir.exists() or (not native and dump.exists()):
        fail("arm output already exists; choose a fresh --out-dir")
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if not native:
        dump.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock(args.lock, f"ppl-{arm.name}"):
        before = gpu_state.snapshot(f"ppl-{arm.name}-before", logs / "gpu-states.jsonl")
        if before["infernix_processes"] and not args.allow_busy_gpu:
            fail(f"another infernix process is running: {before['infernix_processes']}")
        identity = {"exe_sha256": sha256_file(exe), "artifact_sha256": args.artifact_sha256 or sha256_file(artifact),
                    "artifact_hash_supplied": bool(args.artifact_sha256), "corpus_manifest_sha256": sha256_file(corpus)}
        (logs / f"ppl-{arm.name}.argv.json").write_text(json.dumps(argv, indent=2), encoding="utf-8")
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(p.resolve()) for p in args.path_prepend] + [env.get("PATH", "")])
        started, t0 = utc_now(), time.monotonic()
        with (logs / f"ppl-{arm.name}.stdout.txt").open("wb") as so, (logs / f"ppl-{arm.name}.stderr.txt").open("wb") as se:
            rc = subprocess.run(argv, cwd=exe.parent, env=env, stdout=so, stderr=se).returncode
        receipt = {"arm": arm.name, "argv": argv, "exit": rc, "start_utc": started, "end_utc": utc_now(),
                   "wall_seconds": time.monotonic() - t0, "identity": identity, "gpu_before": before,
                   "gpu_after": gpu_state.snapshot(f"ppl-{arm.name}-after", logs / "gpu-states.jsonl")}
        if rc == 0:
            report_path = arm_dir / "report.json"
            report = read_json(report_path, "report")
            scored = report["overall"]["scored_tokens"]
            receipt.update({"scored_tokens": scored, "report_sha256": sha256_file(report_path)})
            if not native:
                positions = len(open_dump(dump, scored, arm.name, report))
                receipt.update({"dump_positions": positions, "dump_sha256": sha256_file(dump)})
            elif args.full_vocab_reference:
                sidecar = arm_dir / "kl_positions.f32"
                if not sidecar.is_file() or sidecar.stat().st_size != scored * 16:
                    fail("native full-vocabulary sidecar is missing or has the wrong position count")
                receipt["kl_positions_sha256"] = sha256_file(sidecar)
        (logs / f"ppl-{arm.name}.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"{arm.name}: exit={rc}")
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
