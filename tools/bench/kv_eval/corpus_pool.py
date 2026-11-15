"""Natural filler text drawn from a perplexity corpus, shared by the needle and speed harnesses.

The pool is the streams of the chosen domains (default: the English ones) of the corpus manifest joined with a blank
line, in manifest order for each domain in the given order. It is tokenized once with the artifact's tokenizer.json
(with character offsets), so a window of N pool tokens maps back to an exact text substring. Re-tokenizing that
substring inside a chat prompt gives a token count that differs from N by a few tokens; callers must use the
server's reported prompt_tokens as the true length.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tools.bench.kv_eval.common import fail, sha256_bytes

DEFAULT_DOMAINS = ("english_reference", "english_long_form")


class Pool:
    def __init__(self, manifest: Path, tokenizer_json: Path, domains=DEFAULT_DOMAINS):
        if not manifest.is_file():
            fail(f"corpus manifest not found: {manifest}")
        spec = json.loads(manifest.read_text(encoding="utf-8"))
        parts, self.sources = [], []
        for domain in domains:
            for s in spec["streams"]:
                if s["domain"] == domain:
                    parts.append((manifest.parent / s["path"].replace("\\", "/")).read_bytes().decode("utf-8"))
                    self.sources.append(s["id"])
        if not parts:
            have = sorted({s["domain"] for s in spec["streams"]})
            fail(f"no stream of domains {list(domains)} in {manifest}; available domains: {have} (use --pool-domains)")
        self.text = "\n\n".join(parts)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        enc = self.tokenizer.encode(self.text, add_special_tokens=False)
        self.starts = np.asarray([o[0] for o in enc.offsets], dtype=np.int64)
        self.ends = np.asarray([o[1] for o in enc.offsets], dtype=np.int64)
        self.n_tokens = len(enc.ids)
        self.text_sha256 = sha256_bytes(self.text.encode("utf-8"))

    def window(self, start_token, n_tokens):
        """Exact text of ~n_tokens pool tokens beginning at the first line start at/after start_token."""
        nl = self.text.find("\n", int(self.starts[start_token]))
        if nl < 0:
            raise ValueError(f"no line break after pool token {start_token}")
        first = int(np.searchsorted(self.starts, nl + 1))
        last = first + n_tokens
        if last > self.n_tokens:
            fail(f"pool too small: window of {n_tokens} tokens from {start_token} needs {last} but the pool has {self.n_tokens}; "
                 "use a longer corpus, more --pool-domains, or shorter --lengths")
        return self.text[int(self.starts[first]):int(self.ends[last - 1])], first, last

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
