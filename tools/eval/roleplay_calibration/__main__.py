"""CLI: ``python -m tools.eval.roleplay_calibration {build,check,mix}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.eval.roleplay_heldout.corpus import read_urls

from . import pins
from .corpus import build, check, describe, exclusion_groups
from .mix import mix


def tokenizer_counter(path: Path):
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise SystemExit("--tokenizer needs the tokenizers package: pip install tokenizers") from error
    tokenizer = Tokenizer.from_file(str(path))
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.eval.roleplay_calibration", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="select the calibration threads and write the text, provenance and thread list")
    p.add_argument("--raw", type=Path, required=True, help="directory with the six raw files (`roleplay_heldout fetch`)")
    p.add_argument("--out", type=Path, required=True, help="empty or new output directory (a camp, never a repository)")
    p.add_argument("--against", type=Path, action="append", default=[],
                   help="evaluation or earlier calibration text (file, or directory of *.txt); repeatable, at least one")
    p.add_argument("--seed", default=pins.SEED, help=f"selection seed (default {pins.SEED}; any other value is a different corpus)")
    p.add_argument("--also-exclude", type=Path,
                   help="further thread URLs to exclude: a text file, one per line, or a calibration_threads.json")
    p.add_argument("--tokenizer", type=Path, help="tokenizer.json of the target model: refuse a text too short for the chunks")
    p.add_argument("--chunks", type=int, default=400, help="chunks `calibration-text` will render (with --tokenizer)")
    p.add_argument("--ctx-size", type=int, default=512, help="chunk length in tokens (with --tokenizer)")
    p.add_argument("--prefix-tokens", type=int, default=17, help="chat-prefix tokens per chunk (with --tokenizer)")

    p = sub.add_parser("check", help="verify a built corpus: hashes, exclusion groups, shingle disjointness")
    p.add_argument("--corpus", type=Path, required=True, help="directory written by `build`")
    p.add_argument("--against", type=Path, action="append", default=[], help="text to check against; repeatable")
    p.add_argument("--expected", help="expected corpus SHA-256 (default: the published hash of the default selection)")

    p = sub.add_parser("mix", help="join two `calibration-text` files into one aligned file with a receipt")
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="new text file; a .json receipt is written beside it")

    args = parser.parse_args(argv)
    if args.command == "build":
        exclusions = exclusion_groups()
        if args.also_exclude:
            exclusions["extra"] = sorted(set(read_urls(args.also_exclude)))
        extra = {}
        if args.tokenizer:
            extra = {
                "count_tokens": tokenizer_counter(args.tokenizer),
                "chunks": args.chunks,
                "body_tokens": args.ctx_size - args.prefix_tokens,
            }
        provenance = build(args.raw, args.out, args.against, seed=args.seed, exclusions=exclusions, **extra)
        print(describe(provenance))
        return 0
    if args.command == "mix":
        receipt = mix(args.first, args.second, args.out)
        print(f"{receipt['chunks']} chunks, sha256 {receipt['out_sha256']}")
        return 0
    return check(args.corpus, args.against, args.expected)


if __name__ == "__main__":
    sys.exit(main())
