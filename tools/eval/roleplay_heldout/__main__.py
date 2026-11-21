"""CLI: ``python -m tools.eval.roleplay_heldout {fetch,build,check}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pins
from .check import check
from .corpus import build, describe, load_exclusions, read_urls
from .fetch import fetch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.eval.roleplay_heldout", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fetch", help="download the pinned raw parquet files and verify their SHA-256")
    p.add_argument("--out", type=Path, required=True, help="directory for the raw files")
    p.add_argument("--endpoint", default=pins.ENDPOINT, help=argparse.SUPPRESS)

    p = sub.add_parser("build", help="clean, select and write the corpus; prints the removal counts and the corpus SHA-256")
    p.add_argument("--raw", type=Path, required=True, help="directory with the raw files (from `fetch`)")
    p.add_argument("--out", type=Path, required=True, help="empty or new output directory")
    p.add_argument("--seed", default=pins.SEED, help=f"selection seed (default {pins.SEED}; any other value is a different corpus)")
    p.add_argument("--per-stratum", type=int, default=pins.PER_STRATUM,
                   help=f"threads per file (default {pins.PER_STRATUM}; any other value is a different corpus)")
    p.add_argument("--also-exclude", type=Path,
                   help="further thread URLs to exclude: a text file, one per line, or a calibration_threads.json")
    p.add_argument("--no-default-exclusions", action="store_true",
                   help="do not apply the committed exclusion list of earlier evaluation sets")

    p = sub.add_parser("check", help="verify a built corpus: file hashes, shingle disjointness, published SHA-256")
    p.add_argument("--corpus", type=Path, required=True, help="directory written by `build`")
    p.add_argument("--against", type=Path, action="append", default=[],
                   help="calibration or earlier evaluation text (file, or directory of *.txt); repeatable")
    p.add_argument("--expected", help="expected corpus SHA-256 (default: the published hash of the default selection)")

    args = parser.parse_args(argv)
    if args.command == "fetch":
        for name, state in fetch(args.out, endpoint=args.endpoint).items():
            print(f"{state:10s} {name}")
        return 0
    if args.command == "build":
        if args.no_default_exclusions:
            exclusions = {}
        else:
            exclusions = load_exclusions()
        if args.also_exclude:
            exclusions["extra"] = sorted(set(read_urls(args.also_exclude)))
        provenance = build(args.raw, args.out, seed=args.seed, per_stratum=args.per_stratum, exclusions=exclusions)
        print(describe(provenance))
        return 0
    return check(args.corpus, args.against, args.expected)


if __name__ == "__main__":
    sys.exit(main())
