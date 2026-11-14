"""Tokenize the chat-prefixed calibration text into a [chunks, context] int32 tensor for run_gptq.py.

The input is the file written by `python -m tools.convert.gemma4_q4 calibration-text`: every chunk is
`<bos>` + chat prefix + body, except the first, whose literal BOS was dropped for llama-imatrix. Every chunk must
encode to exactly --context tokens and start with --prefix-ids; anything else is an error, not a skip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

GEMMA4_CHAT_PREFIX = "2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101"


def calibration_rows(text: str, tokenizer, context: int, prefix: list[int]) -> list[list[int]]:
    """Recover exact logical sequences shared by GPTQ and native activation calibration."""
    if not prefix or context <= len(prefix):
        raise ValueError("calibration context must exceed the nonempty prefix")
    bos = tokenizer.id_to_token(prefix[0])
    if not bos:
        raise ValueError("prefix BOS token is absent from tokenizer")
    chunks = [piece for piece in text.split(bos) if piece]
    rows = []
    for index, piece in enumerate(chunks):
        ids = tokenizer.encode(bos + piece, add_special_tokens=False).ids
        if len(ids) != context or ids[: len(prefix)] != prefix:
            raise ValueError(f"chunk {index} has {len(ids)} tokens or does not start with the chat prefix")
        rows.append(ids)
    if not rows:
        raise ValueError("no chunks found; is this a calibration-text file?")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, type=Path, help="calibration-text output")
    parser.add_argument("--tokenizer", required=True, type=Path, help="tokenizer.json of the model")
    parser.add_argument("--out", required=True, type=Path, help="output .pt file")
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--prefix-ids", default=GEMMA4_CHAT_PREFIX, help="comma separated chat prefix token ids")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    prefix = [int(x) for x in args.prefix_ids.split(",")]
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    rows = calibration_rows(args.text.read_text(encoding="utf-8"), tokenizer, args.context, prefix)
    tensor = torch.tensor(rows, dtype=torch.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, args.out)
    print(f"wrote {args.out}: {tuple(tensor.shape)} tokens, every row starts with the {len(prefix)}-token chat prefix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
