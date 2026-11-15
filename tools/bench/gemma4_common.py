"""Standalone corpus, HTTP and process helpers for the Gemma benchmark campaigns.

Campaign workloads run from this checkout; no external tooling checkout is used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request

from tools.bench.kv_eval.common import resolve_tokenizer
from tools.bench.kv_eval.corpus_pool import DEFAULT_DOMAINS, Pool


def add_pool_args(parser):
    parser.add_argument("--manifest", type=Path, required=True, help="local corpus manifest.json")
    parser.add_argument("--tokenizer", type=Path, help="local tokenizer.json; otherwise extract from --artifact")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--pool-domains", default=",".join(DEFAULT_DOMAINS))


def make_pool(args, scratch):
    tokenizer, _ = resolve_tokenizer(args.tokenizer, args.artifact, scratch / "artifact-resources")
    return Pool(args.manifest, tokenizer, tuple(args.pool_domains.split(",")))


def add_server_args(parser):
    parser.add_argument("--exe", type=Path, required=True, help="infernix-serve executable")
    parser.add_argument("--port", type=int, default=8041)
    parser.add_argument("--model", default="gemma-bench")
    parser.add_argument("--max-context", type=int, default=131072)
    parser.add_argument("--prefill-chunk", default="auto")
    parser.add_argument("--kv-dtype", default="fp8", choices=("bf16", "fp8", "int8"))
    parser.add_argument("--gemma4-qk", default="fp8", choices=("bf16", "fp8"))
    parser.add_argument("--host-context-mib", default="8192")
    parser.add_argument("--vram-headroom-mib", default="1024")
    parser.add_argument("--no-prefix-reuse", action="store_true")
    parser.add_argument("--serve-arg", action="append", default=[], help="extra argv token; use --serve-arg=--flag")
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--lock", type=Path, default=Path(tempfile.gettempdir()) / "infernix-bench-gpu.lock",
                        help="exclusive campaign lock shared across output directories")


def server_argv(args, spec):
    argv = [str(args.exe.resolve()), str(args.artifact.resolve()), "--host", "127.0.0.1", "--port", str(args.port),
            "--model-id", args.model, "--max-context", str(args.max_context), "--kv-capacity", "auto",
            "--vram-headroom-mib", str(args.vram_headroom_mib), "--kv-dtype", args.kv_dtype,
            "--gemma4-qk", args.gemma4_qk, "--max-concurrency", "1", "--prefill-chunk", str(args.prefill_chunk),
            "--logits-dtype", "bf16"]
    argv += ["--no-prefix-reuse"] if args.no_prefix_reuse else ["--host-context-mib", str(args.host_context_mib)]
    return argv + list(spec) + args.serve_arg


@contextmanager
def gpu_lock(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise RuntimeError(f"benchmark lock exists: {path}; check its owner before removing a stale lock") from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "label": label}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def stop(process):
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def running_server(args, argv, log_path, env=None):
    # Never mistake a pre-existing server for this process becoming ready.
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", args.port))
        except OSError as exc:
            raise RuntimeError(f"port {args.port} is already in use") from exc
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("xb") as log:
        process = subprocess.Popen(argv, cwd=Path(argv[0]).parent, stdout=log, stderr=subprocess.STDOUT,
                                   env=env, start_new_session=os.name != "nt")
        started = time.monotonic()
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with {process.returncode}; see {log_path}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as response:
                        if response.status == 200 and json.loads(response.read()) == {"status": "ok"}:
                            break
                except (OSError, ValueError):
                    pass
                if time.monotonic() - started > args.startup_timeout:
                    raise RuntimeError(f"server startup timed out; see {log_path}")
                time.sleep(0.25)
            yield time.monotonic() - started
        finally:
            stop(process)


def add_thinking_arg(parser, default, note=""):
    """--enable-thinking {true,false,default}; `default` omits the field so the server decides."""
    parser.add_argument("--enable-thinking", choices=("true", "false", "default"), default=default,
                        help="send enable_thinking in every chat request; 'default' omits the field and the server "
                             f"decides (infernix resolves an absent value to true on Gemma 4). {note}(default: {default})")


def thinking_value(args):
    """Map the parsed --enable-thinking choice to True, False or None (omit)."""
    return {"true": True, "false": False, "default": None}[args.enable_thinking]


def chat(url, model, messages, max_tokens, *, temperature=0, seed=1234, timeout=3600, top_k=20, enable_thinking=None,
         extra=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": 0.95, "top_k": top_k, "seed": seed, "stream": False}
    if enable_thinking is not None:
        body["enable_thinking"] = enable_thinking
    body.update(extra or {})
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload, time.monotonic() - started


def gpu_state():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
