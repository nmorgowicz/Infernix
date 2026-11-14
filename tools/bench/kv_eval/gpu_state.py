# Ported from ninfer.spo tools/bench/kv_eval; instrument definitions retained.
#!/usr/bin/env python3
"""Record one GPU/process idle-state snapshot as a receipt: gpu_state.py <label> --log <file.jsonl>.

A row holds nvidia-smi memory/temperature/clock/power/utilization, every running process whose name contains
"infernix" and (Windows only) any "System Error" window. Run it before and after every model process: an idle-state
claim in a report must be backed by these rows. Works without nvidia-smi (the field then says so).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.kv_eval.common import use_utf8_stdio, utc_now

SMI_QUERY = "name,driver_version,memory.used,memory.total,temperature.gpu,clocks.current.sm,power.draw,utilization.gpu"


def _run(cmd: list[str]) -> str:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return done.stdout or ""


def infernix_processes() -> list[str]:
    if sys.platform == "win32":
        lines = _run(["tasklist"]).splitlines()
    else:
        lines = _run(["ps", "-eo", "pid,comm"]).splitlines()
    return [line.strip() for line in lines if "infernix" in line.lower()]


def snapshot(label: str, log: Path | None) -> dict:
    smi = _run(["nvidia-smi", f"--query-gpu={SMI_QUERY}", "--format=csv,noheader"]).strip()
    row = {"utc": utc_now(), "label": label, "nvidia_smi_fields": SMI_QUERY,
           "nvidia_smi": smi or "unavailable (nvidia-smi not found or failed)", "infernix_processes": infernix_processes(),
           "system_error_windows": ([l.strip() for l in _run(["tasklist", "/v"]).splitlines() if "system error" in l.lower()]
                                    if sys.platform == "win32" else [])}
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    return row


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label")
    ap.add_argument("--log", type=Path, help="append the row to this JSONL file (default: print only)")
    args = ap.parse_args()
    print(json.dumps(snapshot(args.label, args.log)))


if __name__ == "__main__":
    main()
