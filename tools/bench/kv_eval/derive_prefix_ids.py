#!/usr/bin/env python3
"""Derive native full-vocabulary scoring prefix ids from an artifact's own template and tokenizer.

  python -m tools.bench.kv_eval.derive_prefix_ids --artifact <artifact> --out <prefix.json>
  python -m tools.bench.kv_eval.derive_prefix_ids --source-dir <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja> --out <prefix.json>

Instruction-tuned models score raw prose far from their training distribution. A short single-user-turn conversation is
rendered through the artifact's chat template with the generation prompt appended (the model turn is left open), then
tokenized with the artifact's tokenizer.json without adding special tokens (the template already carries them).
Native full-vocabulary scoring accepts these IDs as context-only --full-vocab-prefix-ids.
Pinned references require their exact recorded prefix, not a newly derived replacement.

--artifact reads the embedded resource/text/{tokenizer.json,tokenizer_config.json,chat_template.jinja} (so the ids come
from exactly what the engine will serve) and records their sha256. Needs `transformers` (Jinja2 template rendering)
and `tokenizers`; the two tokenizations must agree or the script stops.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.bench.kv_eval.common import (
    extract_artifact_resources, fail, sha256_file, use_utf8_stdio,
)

DEFAULT_MESSAGE = "Continue the document."


def main() -> None:
    use_utf8_stdio()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--artifact", type=Path, help=".infernix artifact whose embedded template/tokenizer are used")
    src.add_argument("--source-dir", type=Path, help="directory holding tokenizer.json, tokenizer_config.json and chat_template.jinja")
    ap.add_argument("--out", type=Path, required=True, help="prefix.json for run_ppl.py --prefix-ids-file")
    ap.add_argument("--message", default=DEFAULT_MESSAGE, help="content of the single user message (default: %(default)r)")
    ap.add_argument("--template-kwargs", default="{}", help='JSON object passed to apply_chat_template, e.g. \'{"enable_thinking": false}\'')
    args = ap.parse_args()

    try:
        extra = json.loads(args.template_kwargs)
        assert isinstance(extra, dict)
    except (json.JSONDecodeError, AssertionError):
        fail("--template-kwargs must be a JSON object")
    try:
        from tokenizers import Tokenizer
        from transformers import AutoTokenizer
    except ImportError as exc:
        fail(f"{exc}. Install the kit dependencies: pip install -r tools/bench/kv_eval/requirements.txt")

    if args.artifact:
        src_dir = args.out.resolve().parent / "artifact-resources"
        extract_artifact_resources(args.artifact, src_dir)
        origin = {"artifact": str(args.artifact), "extracted_to": str(src_dir)}
    else:
        src_dir = args.source_dir
        origin = {"source_dir": str(src_dir)}
    needed = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    missing = [n for n in needed if not (src_dir / n).is_file()]
    if missing:
        fail(f"{src_dir} lacks {missing}; the artifact does not embed them or the source directory is incomplete")

    messages = [{"role": "user", "content": args.message}]
    hf = AutoTokenizer.from_pretrained(str(src_dir), local_files_only=True)
    text = hf.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **extra)
    raw = Tokenizer.from_file(str(src_dir / "tokenizer.json"))
    ids = raw.encode(text, add_special_tokens=False).ids
    ids_hf = hf(text, add_special_tokens=False)["input_ids"]
    if ids != ids_hf:
        fail("the tokenizers library and transformers disagree on the rendered prefix; inspect the tokenizer files before using these ids")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "messages": messages,
        "template_kwargs": extra,
        "rendered_text": text,
        "ids": ids,
        "ids_csv": ",".join(map(str, ids)),
        "tokens": [raw.id_to_token(i) for i in ids],
        "origin": origin,
        "source_sha256": {n: sha256_file(src_dir / n) for n in needed},
    }, indent=2, ensure_ascii=True), encoding="utf-8")
    print(repr(text))
    print(",".join(map(str, ids)))


if __name__ == "__main__":
    main()
