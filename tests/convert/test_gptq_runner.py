"""GPTQ argument defaults and real compressor5 packing; CPU-only, in the GPTQ environment."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("llmcompressor", reason="needs tools/gptq/requirements.txt")
pytest.importorskip("datasets")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "gptq"))
import run_gptq  # noqa: E402


def arguments(tmp_path):
    return ["--recipe", "gemma4_31b", "--tree", str(tmp_path / "tree"), "--calib", str(tmp_path / "calib.pt"),
            "--out", str(tmp_path / "out")]


def test_four_bit_runner_defaults_and_explicit_five_bit_selection(tmp_path):
    parser = run_gptq.build_parser()
    args = parser.parse_args(arguments(tmp_path))
    assert args.bits == 4 and args.group_size is None
    assert run_gptq.recipes.get(args.recipe).group_size == 32
    args = parser.parse_args(arguments(tmp_path) + ["--bits", "5", "--group-size", "64"])
    assert args.bits == 5 and args.group_size == 64


@pytest.mark.parametrize("group", [[], ["--group-size", "32"], ["--group-size", "128"]])
def test_five_bit_rejects_wrong_groups_before_loading_inputs(tmp_path, group, capsys):
    with pytest.raises(SystemExit) as error:
        run_gptq.main(arguments(tmp_path) + ["--bits", "5"] + group)
    assert error.value.code == 2
    assert "--bits 5 requires --group-size 64" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_native_compressor_five_bit_dense_cross_word_proof():
    facts = run_gptq.validate_compressor(5)
    assert facts["num_bits"] == 5 and facts["packing"] == "dense_cross_word_int32"
    assert facts["compressed_tensors_version"]


def test_unsupported_compressor_is_reported_not_worked_around(monkeypatch):
    def unsupported(*args):
        raise ValueError("5-bit packing is unsupported")

    monkeypatch.setattr(run_gptq.PackedQuantizationCompressor, "compress", unsupported)
    with pytest.raises(RuntimeError, match="cannot produce native packed INT5 group64.*unsupported"):
        run_gptq.validate_compressor(5)


def test_wrong_native_packing_is_refused(monkeypatch):
    original = run_gptq.PackedQuantizationCompressor.compress

    def wrong_layout(state, scheme):
        result = original(state, scheme)
        result["weight_packed"] = result["weight_packed"][:, :-1]
        return result

    monkeypatch.setattr(run_gptq.PackedQuantizationCompressor, "compress", wrong_layout)
    with pytest.raises(RuntimeError, match="unsupported INT5 packing"):
        run_gptq.validate_compressor(5)
