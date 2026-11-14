"""Each rendered chunk re-encodes to exactly the chunk length and starts with the chat prefix."""

from __future__ import annotations

import hashlib
import json

import pytest

pytest.importorskip("tokenizers")
from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from tools.convert import gemma4_q4_cli as cli  # noqa: E402
from tools.convert.gemma4_q4 import render  # noqa: E402

WORDS = [f"w{i}" for i in range(50)]
SPECIAL = ["<bos>", "<turn>", "user", "<end>"]


def tokenizer():
    vocab = {token: index for index, token in enumerate(SPECIAL + WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.add_special_tokens(["<bos>", "<turn>", "<end>"])
    return tok


PREFIX = [0, 1, 2, 3]
TEXT = " ".join(WORDS[i % 50] for i in range(5000))


def test_every_chunk_is_exactly_ctx_long_and_starts_with_the_prefix_and_the_first_drops_its_bos():
    tok = tokenizer()
    segments, skipped, stride = render(tok, TEXT, PREFIX, 64, 20)
    assert skipped == 0 and len(segments) == 20 and stride > 0
    assert not segments[0].startswith("<bos>") and all(s.startswith("<bos>") for s in segments[1:])
    for index, segment in enumerate(segments):
        ids = tok.encode(segment, add_special_tokens=False).ids
        assert len(ids) == (63 if index == 0 else 64)
    joined = tok.encode("".join(segments), add_special_tokens=False).ids
    assert len(joined) == 20 * 64 - 1  # llama-imatrix adds the single missing BOS
    # With that BOS prepended, chunk k starts at 64*k and begins with the BOS id and the prefix.
    shifted = [0] + joined
    assert all(shifted[64 * k : 64 * k + 4] == PREFIX for k in range(20))


def test_source_too_short_for_the_requested_chunks_is_refused():
    with pytest.raises(ValueError, match="need"):
        render(tokenizer(), " ".join(WORDS), PREFIX, 64, 20)


def test_prefix_without_bos_is_refused():
    with pytest.raises(ValueError, match="bos"):
        render(tokenizer(), TEXT, [1, 2, 3], 64, 5)


def test_ctx_size_must_exceed_the_prefix():
    with pytest.raises(ValueError, match="exceed the prefix"):
        render(tokenizer(), TEXT, PREFIX, 4, 5)


def calibration_argv(tmp_path):
    tokenizer().save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "source.txt").write_text(TEXT, encoding="utf-8")
    return [
        "calibration-text",
        "--tokenizer", str(tmp_path / "tokenizer.json"),
        "--source", str(tmp_path / "source.txt"),
        "--prefix-ids", "0,1,2,3",
        "--chunks", "20",
        "--ctx-size", "64",
        "--out", str(tmp_path / "out" / "cal.txt"),
    ]  # fmt: skip


def test_calibration_text_command_writes_the_text_and_a_receipt(tmp_path, capsys):
    assert cli.main(calibration_argv(tmp_path)) == 0
    printed = json.loads(capsys.readouterr().out)
    out = tmp_path / "out" / "cal.txt"
    receipt = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert receipt == printed
    assert receipt["schema"] == cli.CALIBRATION_SCHEMA
    assert receipt["out_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert receipt["source_sha256"] == hashlib.sha256((tmp_path / "source.txt").read_bytes()).hexdigest()
    assert receipt["tokenizer_sha256"] == hashlib.sha256((tmp_path / "tokenizer.json").read_bytes()).hexdigest()
    assert (receipt["prefix_ids"], receipt["ctx_size"], receipt["chunks"], receipt["skipped"]) == (
        [0, 1, 2, 3],
        64,
        20,
        0,
    )
    assert set(receipt) == {
        "schema", "out", "out_sha256", "source", "source_sha256", "tokenizer_sha256",
        "prefix_ids", "ctx_size", "chunks", "skipped", "stride_tokens",
    }  # fmt: skip


def test_calibration_text_refuses_existing_output_and_short_sources(tmp_path, capsys):
    argv = calibration_argv(tmp_path)
    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main(argv) == 2
    assert "already exists" in capsys.readouterr().err
    short = list(argv)
    short[short.index("--out") + 1] = str(tmp_path / "short.txt")
    short[short.index("--chunks") + 1] = "400"
    assert cli.main(short) == 2
    assert "need" in capsys.readouterr().err
    assert not (tmp_path / "short.txt").exists()
