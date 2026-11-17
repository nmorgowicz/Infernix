"""Argument validation and subcommand wiring of `python -m tools.convert.gemma4_q4`; converts nothing."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from tools.convert import gemma4_q4_cli as cli
from tools.convert.gemma4_q4 import Q4Options

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_checkout(monkeypatch):
    monkeypatch.setattr(cli, "checkout_revision", lambda: "test-revision")


def convert_args(tmp_path, *extra, no_mtp=True):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    argv = ["convert", "--model", str(model), "--out", str(tmp_path / "out" / "g.ninfer")]
    return argv + (["--no-mtp"] if no_mtp else []) + list(extra)


def run(argv, capsys):
    code = cli.main(argv)
    return code, capsys.readouterr().err


@pytest.mark.parametrize(
    "extra,message",
    [
        (["--weights", "fp8_row", "--native-groups"], "apply only to the q4_g32"),
        (["--weights", "fp8_row", "--region-group", "mlp_down=128"], "apply only to the q4_g32"),
        (["--weights", "fp8_row", "--group-size", "128"], "apply only to the q4_g32"),
        (["--weights", "fp8_row", "--scale-search", "mse"], "apply only to the q4_g32"),
        (["--scale-search", "weighted"], "weighted requires --imatrix"),
        (["--imatrix", "m.gguf"], "weighted requires --imatrix"),
        (["--region-group", "bogus=128"], "region-group"),
        (["--region-group", "mlp_down=48"], "region-group"),
        (["--rows-per-chunk", "0"], "--rows-per-chunk must be positive"),
    ],
)
def test_argument_mistakes_are_refused_before_any_checkpoint_is_opened(
    tmp_path, capsys, clean_checkout, extra, message
):
    code, err = run(convert_args(tmp_path, *extra), capsys)
    assert code == 1 and message in err and err.startswith("gemma4_q4: ")
    assert not (tmp_path / "out").exists()


def with_out(argv, path):
    changed = list(argv)
    changed[changed.index("--out") + 1] = str(path)
    return changed


def test_output_must_be_a_new_ninfer_file_outside_the_source(tmp_path, capsys, clean_checkout):
    argv = convert_args(tmp_path)
    assert "outside the source" in run(with_out(argv, tmp_path / "model" / "g.ninfer"), capsys)[1]
    assert "--out must end in .ninfer" in run(with_out(argv, tmp_path / "g.bin"), capsys)[1]
    for suffix in ("", ".build.json", ".conversion.json"):
        existing = tmp_path / ("g.ninfer" + suffix)
        existing.write_text("x")
        code, err = run(with_out(argv, tmp_path / "g.ninfer"), capsys)
        assert code == 1 and "Output already exists" in err
        existing.unlink()


def test_mtp_assistant_and_no_mtp_are_mutually_exclusive_and_one_is_required(tmp_path, capsys, clean_checkout):
    neither = convert_args(tmp_path, no_mtp=False)
    code, err = run(neither, capsys)
    assert code == 1 and "hf download google/gemma-4-31B-it-assistant" in err and "--no-mtp" in err
    both = convert_args(tmp_path, "--mtp-assistant", str(tmp_path / "assistant"))
    code, err = run(both, capsys)
    assert code == 1 and "--no-mtp" in err
    missing = convert_args(tmp_path, "--mtp-assistant", str(tmp_path / "assistant"), no_mtp=False)
    code, err = run(missing, capsys)
    assert code == 1 and "MTP assistant not found" in err


@pytest.mark.parametrize("mode", ["fp8", "q4"])
def test_mtp_weights_cannot_be_combined_with_no_mtp(tmp_path, capsys, clean_checkout, mode):
    code, err = run(convert_args(tmp_path, "--mtp-weights", mode), capsys)
    assert code == 1 and "needs the MTP assistant" in err
    assert not (tmp_path / "out").exists()


def test_valid_arguments_become_options_without_touching_the_filesystem(tmp_path):
    args = cli.build_parser().parse_args(
        convert_args(
            tmp_path,
            "--weights", "q4_g32_a16",
            "--group-size", "256",
            "--native-groups",
            "--region-group", "attn_out=32",
            "--scale-search", "mse",
        )
    )
    options, output = cli.check_convert_arguments(args)
    assert options == Q4Options(
        weights="q4_g32_a16",
        group_size=256,
        region_groups=("attn_out=32",),
        scale_search="mse",
        native_groups=True,
    )
    assert output == (tmp_path / "out" / "g.ninfer").resolve()
    assert not output.parent.exists()


def test_resource_overrides_are_validated_before_any_checkpoint_is_opened(tmp_path, capsys, clean_checkout):
    template = tmp_path / "template.jinja"
    template.write_text("{{ messages }}", encoding="utf-8")
    args = cli.build_parser().parse_args(
        convert_args(tmp_path, "--resource", f"chat_template.jinja={template}")
    )
    assert cli.resource_overrides(args) == {"chat_template.jinja": template}
    for item, message in (
        ("chat_template.jinja", "ROLE=PATH"),
        ("vocab.txt=" + str(template), "ROLE=PATH"),
        ("chat_template.jinja=" + str(tmp_path / "missing.jinja"), "file not found"),
    ):
        code, err = run(convert_args(tmp_path, "--resource", item), capsys)
        assert code == 1 and message in err
    code, err = run(
        convert_args(tmp_path, "--resource", f"chat_template.jinja={template}", "--resource", f"chat_template.jinja={template}"),
        capsys,
    )
    assert code == 1 and "ROLE=PATH" in err
    assert not (tmp_path / "out").exists()


def test_dirty_checkout_is_refused_for_convert(tmp_path, capsys, monkeypatch):
    def dirty():
        raise ValueError("this infernix checkout has tracked modifications; reconcile them before conversion")

    monkeypatch.setattr(cli, "checkout_revision", dirty)
    code, err = run(convert_args(tmp_path), capsys)
    assert code == 1 and "tracked modifications" in err


@pytest.mark.parametrize("command", [["convert"], ["verify"], ["calibration-text"], []])
def test_help_works_for_the_tool_and_every_subcommand(command, capsys):
    with pytest.raises(SystemExit) as stop:
        cli.build_parser().parse_args(command + ["--help"])
    assert stop.value.code == 0
    assert "usage: python -m tools.convert.gemma4_q4" in capsys.readouterr().out


def test_module_entry_point_runs_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "tools.convert.gemma4_q4", "convert", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert result.returncode == 0, result.stderr
    assert "--native-groups" in result.stdout and "--imatrix" in result.stdout


def test_verify_refuses_a_missing_artifact(tmp_path, capsys):
    code, err = run(["verify", "--artifact", str(tmp_path / "none.ninfer")], capsys)
    assert code == 1 and err.startswith("gemma4_q4: ")



@pytest.mark.parametrize("weights", ["q5_g64", "q5_g64_a16"])
def test_explicit_q5_recipe_and_unchanged_four_bit_cli_defaults(tmp_path, weights):
    parser = cli.build_parser()
    defaults = cli.options_from(parser.parse_args(convert_args(tmp_path)))
    assert defaults == Q4Options()
    args = parser.parse_args(convert_args(
        tmp_path, "--weights", weights, "--gptq-source", str(tmp_path / "gptq5")
    ))
    options, _ = cli.check_convert_arguments(args)
    assert options.bits == 5 and options.gptq_source == tmp_path / "gptq5"
    assert not (tmp_path / "out").exists()
