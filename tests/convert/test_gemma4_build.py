"""`python -m tools.convert.gemma4_build`: preset commands, preflight, resume, launcher, serve-check; runs nothing heavy."""

from __future__ import annotations

from functools import partial
import hashlib
import io
import json
import os
import socket
from pathlib import Path
import struct
import subprocess
import sys
import time

import pytest

from tools.convert import gemma4_build as build
from tools.convert import gemma4_build_checks as checks
from tools.convert import gemma4_build_cli as cli
from tools.convert import gemma4_build_launcher as launcher

PY = "PY"
PREFIX = "2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101"
PINS = checks.parse_pins(build.REPO_ROOT / "tools" / "gptq" / "requirements.txt", checks.GPTQ_PINS)


# ---------------------------------------------------------------- fixtures


def write_safetensors(path: Path, dtype: str = "BF16") -> None:
    header = json.dumps({"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 2]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def gemma_config(hidden: int = 5376, layers: int = 60) -> dict:
    types = (["sliding_attention"] * 5 + ["full_attention"]) * 10
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": hidden, "vocab_size": 8, "num_hidden_layers": layers, "max_position_embeddings": 262144,
            "intermediate_size": 21504, "num_attention_heads": 32, "num_key_value_heads": 16,
            "num_global_key_value_heads": 4, "head_dim": 256, "global_head_dim": 512,
            "layer_types": types[:layers], "sliding_window": 1024, "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0, "tie_word_embeddings": True,
            "rope_parameters": {
                "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0, "rope_type": "proportional"},
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    }


def make_model(path: Path, *, dtype: str = "BF16", config: dict | None = None) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config or gemma_config()))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.norm.weight": "model-00001-of-00001.safetensors"}})
    )
    write_safetensors(path / "model-00001-of-00001.safetensors", dtype)
    (path / "tokenizer.json").write_text("{}")
    return path


def make_assistant(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps({"architectures": ["Gemma4AssistantForCausalLM"],
                                                  "backbone_hidden_size": 5376}))
    (path / "model.safetensors").write_bytes(b"x")
    return path


class FakeProbes:
    """Machine probes with healthy defaults; tests override one fact at a time."""

    def __init__(self, **over):
        self.facts = {
            "gpu": checks.GpuInfo("NVIDIA GeForce RTX 5090", 32607, 30000, "12.0"),
            "ram": (64 * 2**30, 60 * 2**30),
            "disk": 500e9,
            "git": ("a" * 40, []),
            "gptq_versions": dict(PINS),
            "cuda": True,
            "formats": True,
            "dlls": [],
            "missing_packages": (),
        }
        self.facts.update(over)

    def gpu(self):
        return self.facts["gpu"]

    def ram(self):
        return self.facts["ram"]

    def disk_free(self, path):
        return self.facts["disk"]

    def device_of(self, path):
        return 1

    def python_info(self, python, packages, cuda=False):
        versions = {name: self.facts["gptq_versions"].get(name, "1.0") for name in packages}
        for name in self.facts["missing_packages"]:
            if name in versions:
                versions[name] = None
        return {"python": "3.13.1", "packages": versions, "cuda": self.facts["cuda"]}

    def git_state(self):
        return self.facts["git"]

    def tool_version(self, exe):
        return "version: test"

    def scan_binary(self, path, needles):
        return {needle: self.facts["formats"] for needle in needles}

    def missing_dlls(self, exe):
        return self.facts["dlls"]


class Env:
    """A tiny but valid set of inputs and fake tools under `tmp_path`."""

    def __init__(self, tmp_path: Path):
        self.root = tmp_path
        self.model = make_model(tmp_path / "model")
        self.assistant = make_assistant(tmp_path / "assistant")
        self.out = tmp_path / "out"
        self.server = tmp_path / "bin" / launcher.exe_name("infernix-serve")
        self.server.parent.mkdir()
        self.server.write_bytes(b"q4_g32_fp16 q4_g256_fp16")
        self.perplexity = self.server.with_name(launcher.exe_name("infernix-perplexity"))
        self.perplexity.write_bytes(b"x")
        self.gptq_python = tmp_path / "gptq-venv" / "python.exe"
        self.gptq_python.parent.mkdir()
        self.gptq_python.write_bytes(b"x")
        self.calibration = tmp_path / "calibration.txt"
        self.calibration.write_text("public text")
        self.llama_cpp = tmp_path / "llama-bin"
        self.llama_cpp.mkdir()
        for name in ("llama-quantize", "llama-imatrix"):
            (self.llama_cpp / launcher.exe_name(name)).write_bytes(b"tool")
        self.llama_source = tmp_path / "llama-src"
        self.llama_source.mkdir()
        (self.llama_source / "convert_hf_to_gguf.py").write_text("# converter")

    def argv(self, preset: str, *extra: str, mtp: bool = True) -> list[str]:
        argv = ["--preset", preset, "--model", str(self.model), "--out-dir", str(self.out),
                "--serve", str(self.server), "--gptq-python", str(self.gptq_python),
                "--calibration-text", str(self.calibration), "--llama-cpp", str(self.llama_cpp),
                "--llama-cpp-source", str(self.llama_source)]
        argv += ["--assistant", str(self.assistant)] if mtp else ["--no-mtp"]
        return argv + list(extra)

    def config(self, preset: str, **over) -> build.BuildConfig:
        values = dict(
            preset=build.PRESETS[preset], model=self.model, assistant=self.assistant, with_vision=False,
            out_dir=self.out, work_dir=self.out / "work", name="m-" + preset, python=PY,
            gptq_python=self.gptq_python, calibration_text=self.calibration, perplexity_exe=self.perplexity,
            llama_cpp=self.llama_cpp, llama_cpp_source=self.llama_source,
        )
        values.update(over)
        return build.BuildConfig(**values)


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


def serve_settings(env: Env, **over) -> build.ServeSettings:
    return build.ServeSettings(server_exe=env.server, **over)


def preflight(env: Env, cfg: build.BuildConfig, probes=None, **over) -> checks.Report:
    identity = build.build_identity(cfg)
    return checks.preflight(cfg, serve_settings(env, **over), env.server, probes or FakeProbes(),
                            build.fingerprint_of(identity), identity)


def finding(report: checks.Report, what: str) -> checks.Finding:
    matches = [item for item in report.findings if item.what == what]
    assert matches, f"no finding {what!r}: {[item.what for item in report.findings]}"
    return matches[0]


# ---------------------------------------------------------------- preset -> commands


def w(cfg: build.BuildConfig, *parts: str) -> str:
    return str(cfg.work(*parts))


def convert_expected(cfg: build.BuildConfig, *recipe: str, mtp: bool = True, text_only: bool = True) -> list[str]:
    argv = [PY, "-m", "tools.convert.gemma4_q4", "convert", "--model", str(cfg.model), "--weights", "q4_g32", *recipe]
    argv += ["--text-only"] if text_only else []
    argv += ["--mtp-assistant", str(cfg.assistant)] if mtp else ["--no-mtp"]
    return argv + ["--name", cfg.name, "--out", str(cfg.artifact)]


def plan_by_id(cfg: build.BuildConfig) -> dict[str, build.Step]:
    return {step.id: step for step in build.build_plan(cfg)}


def test_every_preset_has_the_documented_recipe_flags(env):
    expected = {
        "best-small": ("--gptq-source", str(env.out / "work" / "gptq"), "--native-groups"),
        "best-medium": ("--gptq-source", str(env.out / "work" / "gptq")),
        "draft": (),
    }
    for preset, recipe in expected.items():
        cfg = env.config(preset, calibration_text=None)
        assert plan_by_id(cfg)["convert"].argv == convert_expected(cfg, *recipe), preset


def test_gptq_and_draft_preserve_their_recipes():
    small, medium = build.PRESETS["best-small"], build.PRESETS["best-medium"]
    assert (small.format_name, small.group_size, small.native_groups, small.method) == ("q4_g256_fp16", 256, True, "gptq")
    assert (medium.format_name, medium.group_size, medium.native_groups, medium.method) == (
        "q4_g32_fp16", 32, False, "gptq")
    assert build.PRESETS["draft"].method == "plain"
    assert round(small.file_gb * 1e9, -7) == 16.85e9 and round(medium.file_gb * 1e9, -7) == 18.41e9
    assert small.kl > medium.kl
    assert build.PRESETS["draft"].group_size == 32


def test_gptq_steps_are_the_documented_commands(env):
    for preset, group in (("best-small", "256"), ("best-medium", "32")):
        cfg = env.config(preset)
        steps = plan_by_id(cfg)
        gptq_dir = build.REPO_ROOT / "tools" / "gptq"
        gpy = str(env.gptq_python)
        tok = str(env.model / "tokenizer.json")
        assert list(steps) == ["calibration-render", "calibration-tokens", "text-tree", "gptq", "gptq-check",
                               "convert", "receipt-check", "finalize"]
        assert steps["calibration-render"].argv == [
            PY, "-m", "tools.convert.gemma4_q4", "calibration-text", "--tokenizer", tok, "--source",
            str(env.calibration), "--prefix-ids", PREFIX, "--chunks", "400", "--ctx-size", "512", "--out",
            w(cfg, "calibration", "templated.txt")]
        # The GPTQ scripts run in the GPTQ interpreter, the converter in the builder's own.
        assert steps["calibration-render"].argv[0] == PY
        assert steps["calibration-tokens"].argv == [
            gpy, str(gptq_dir / "make_calib.py"), "--text", w(cfg, "calibration", "templated.txt"),
            "--tokenizer", tok, "--out", w(cfg, "calib.pt"), "--context", "512", "--prefix-ids", PREFIX]
        assert steps["text-tree"].argv == [
            gpy, str(gptq_dir / "make_text_tree.py"), "--model", str(env.model), "--out", w(cfg, "text60")]
        assert steps["gptq"].argv == [
            gpy, str(gptq_dir / "run_gptq.py"), "--tree", w(cfg, "text60"), "--calib", w(cfg, "calib.pt"),
            "--out", w(cfg, "gptq"), "--recipe", "gemma4_31b", "--observer", "imatrix_mse", "--group-size", group,
            "--max-cpu-gb", "32", "--offload-dir", w(cfg, "offload"), "--peak-log", w(cfg, "gptq.peak.jsonl")]
        assert steps["gptq-check"].argv == [
            gpy, str(gptq_dir / "check_checkpoint.py"), "--gptq", w(cfg, "gptq"), "--tree", w(cfg, "text60")]
        assert steps["gptq"].prune_always == (cfg.work("offload"),)
        assert steps["gptq-check"].prune_if_short == (cfg.work("text60"),)


@pytest.mark.parametrize("preset, quant", [("iq3-xs", "IQ3_XS"), ("iq3-xxs", "IQ3_XXS")])
def test_iq3_uses_own_q8_imatrix_and_imports_blocks_with_assistant(env, preset, quant):
    cfg = env.config(preset)
    steps = plan_by_id(cfg)
    assert list(steps) == ["calibration-render", "gguf-bf16", "gguf-q8", "imatrix", "gguf-quantize",
                           "convert", "receipt-check", "finalize"]
    assert steps["gguf-bf16"].argv == [
        PY, str(env.llama_source / "convert_hf_to_gguf.py"), str(env.model), "--outtype", "bf16",
        "--outfile", w(cfg, "model-bf16.gguf")]
    tool = str(env.llama_cpp / launcher.exe_name("llama-quantize"))
    assert steps["gguf-q8"].argv == [tool, w(cfg, "model-bf16.gguf"), w(cfg, "model-q8_0.gguf"), "Q8_0"]
    assert steps["imatrix"].argv == [
        str(env.llama_cpp / launcher.exe_name("llama-imatrix")), "-m", w(cfg, "model-q8_0.gguf"),
        "-o", w(cfg, "imatrix.gguf"), "--output-format", "gguf", "--no-ppl", "--ctx-size", "512",
        "--gpu-layers", "999", "--file", w(cfg, "calibration", "templated.txt"), "--chunks", "400",
        "--parse-special"]
    argv = steps["gguf-quantize"].argv
    assert argv[:3] == [tool, "--imatrix", w(cfg, "imatrix.gguf")]
    assert argv[-3:] == [w(cfg, "model-bf16.gguf"), w(cfg, f"model-{quant}.gguf"), quant]
    pins = argv[3:-3]
    assert pins[::2] == ["--tensor-type"] * len(pins[1::2])
    assert "attn_output=IQ3_S" in pins
    if quant == "IQ3_XS":
        assert "attn_v=IQ3_S" in pins and "ffn_down=IQ3_S" in pins
        assert r"blk\.([0-6]|5[2-9])\.ffn_(gate|up)=IQ3_S" in pins
    else:
        assert "attn_q=IQ2_S" in pins and "attn_k=IQ2_S" in pins
    assert steps["convert"].argv == convert_expected(cfg, "--gguf-source", w(cfg, f"model-{quant}.gguf"))
    assert not any("gptq" in step for step in steps)
    assert steps["imatrix"].prune_always == (cfg.work("model-q8_0.gguf"),)
    assert steps["gguf-quantize"].prune_always == (cfg.work("model-bf16.gguf"),)


@pytest.mark.parametrize("preset", ["iq3-xs", "iq3-xxs"])
def test_iq3_score_uses_default_policy_not_the_q4_experiment_switch(env, preset):
    cfg = env.config(preset, score=True)
    steps = plan_by_id(cfg)
    assert steps["score-run"].env == {}
    assert "GGUF AllowA8" in steps["score-run"].title
    twin = steps["score-candidate"].argv
    assert "--gguf-source" in twin and "--no-mtp" in twin and "--mtp-assistant" not in twin


def test_a_gptq_build_downloads_the_pinned_corpus_without_a_text_file(env):
    cfg = env.config("best-small", calibration_text=None)
    steps = plan_by_id(cfg)
    assert list(steps)[:2] == ["calibration-source", "calibration-render"]
    assert build.CALIBRATION_URL in steps["calibration-source"].note
    assert build.CALIBRATION_REVISION in build.CALIBRATION_URL


def test_draft_is_a_plain_convert_and_no_mtp_vision_variants(env):
    cfg = env.config("draft", assistant=None, with_vision=True)
    steps = plan_by_id(cfg)
    assert list(steps) == ["convert", "receipt-check", "finalize"]
    assert steps["convert"].argv == convert_expected(cfg, mtp=False, text_only=False)


def test_score_steps_use_the_native_full_vocabulary_kl_and_explicit_exact_route(env):
    cfg = env.config("best-small", score=True)
    steps = plan_by_id(cfg)
    assert list(steps)[-6:] == ["score-corpus", "score-reference", "score-reference-write", "score-candidate",
                                "score-run", "score-report"]
    root = cfg.work("score")
    common = ["--corpus", str(root / "corpus" / "manifest.json"), "--context", "512", "--stride", "256",
              "--prefill-chunk", "128", "--kv-dtype", "fp8", "--logits-dtype", "bf16",
              "--full-vocab-prefix-ids", PREFIX]
    assert steps["score-reference-write"].argv == [
        str(env.perplexity), str(root / "ref-fp8row.ninfer")] + common + [
        "--output", str(root / "ref-run"), "--full-vocab-reference-write", str(root / "ref.nklref")]
    assert steps["score-run"].argv == [str(env.perplexity), str(root / "candidate.ninfer")] + common + [
        "--full-vocab-reference", str(root / "ref.nklref"), "--output", str(root / "run")]
    assert steps["score-run"].env == {build.EXACT_ROUTE_ENV: "1"} and "NINFER" not in build.EXACT_ROUTE_ENV
    assert not steps["score-reference-write"].env
    twin = steps["score-candidate"].argv
    assert "--no-mtp" in twin and "--text-only" in twin and "--mtp-assistant" not in twin
    assert steps["score-reference"].argv[-4:-2] == ["--weights", "fp8_row"]
    assert steps["gptq-check"].prune_if_short == (cfg.work("text60"),)


def test_score_report_reads_the_native_report_structure(env):
    cfg = env.config("best-small", score=True)
    builder, _, _ = make_builder(env, cfg)
    builder.open_work_dir()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifact.write_bytes(b"abc")
    write_receipt(cfg)
    run = cfg.work("score", "run")
    run.mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({
        "kl": {"statistics": {"overall": {"mean_kl": 0.12, "mean_delta_nll": 0.03, "top1_disagreement": 0.2}}},
        "overall": {"perplexity": 7.5},
    }))
    build.score_report(builder)
    assert builder.score == {"mean_kl": 0.12, "mean_delta_nll": 0.03, "top1_disagreement": 0.2, "perplexity": 7.5}
    assert "Mean KL vs FP8-row reference: 0.1200" in (env.out / "BUILD-SUMMARY.md").read_text()


# ---------------------------------------------------------------- dry run through the command line


@pytest.mark.parametrize("preset", list(build.PRESETS))
def test_dry_run_prints_the_plan_for_every_preset_and_writes_nothing(env, capsys, preset):
    before = sorted(path for path in env.root.rglob("*"))
    code = cli.main(env.argv(preset, "--dry-run", "--yes"), probes=FakeProbes())
    out = capsys.readouterr().out
    assert code == 0
    assert "Dry run: nothing was written." in out and "[FAIL]" not in out
    assert build.PRESETS[preset].summary in out and "tools.convert.gemma4_q4 convert" in out
    assert " min  " in out
    assert sorted(path for path in env.root.rglob("*")) == before
    assert not env.out.exists()


@pytest.mark.parametrize("preset, reason", [("lite-medium", "q4_g32_s6_fp16"), ("lite-small", "llama.cpp"),
                                             ("nonsense", "choose one of")])
def test_unavailable_and_unknown_presets_are_refused_with_the_reason(env, capsys, preset, reason):
    assert cli.main(env.argv(preset, "--dry-run"), probes=FakeProbes()) == 1
    assert reason in capsys.readouterr().err
    assert not env.out.exists()


def test_dry_run_exits_zero_even_when_checks_would_stop_a_real_run(env, capsys):
    probes = FakeProbes(gpu=None)
    assert cli.main(env.argv("draft", "--dry-run"), probes=probes) == 0
    out = capsys.readouterr().out
    assert "[FAIL] GPU" in out and "would stop a real run" in out


def test_check_exit_code_follows_the_checks(env, capsys):
    assert cli.main(env.argv("best-small", "--check"), probes=FakeProbes()) == 0
    assert "All checks passed." in capsys.readouterr().out
    assert cli.main(env.argv("best-small", "--check"), probes=FakeProbes(gpu=None)) == 1
    assert "1 check(s) failed." in capsys.readouterr().out


def test_real_run_stops_on_failed_checks_without_writing(env, capsys):
    assert cli.main(env.argv("draft", "--yes"), probes=FakeProbes(ram=(1, 1), gpu=None)) == 1
    assert not env.out.exists()
    assert "Not starting" in capsys.readouterr().err


def test_confirmation_is_required_without_yes(env, capsys, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert cli.main(env.argv("draft"), probes=FakeProbes()) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert not env.out.exists()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert cli.main(env.argv("draft"), probes=FakeProbes()) == 1
    assert not env.out.exists()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--alias", "bad alias"], "--alias"),
        (["--port", "70000"], "--port"),
        (["--host", 'a"b'], "--host"),
        (["--top-k", "65"], "--top-k"),
        (["--host-context-mib", "0"], "--host-context-mib"),
        (["--draft-tokens", "9"], "--draft-tokens"),
    ],
)
def test_argument_mistakes_are_refused_before_any_probe(env, capsys, extra, message):
    assert cli.main(env.argv("draft", "--dry-run", *extra), probes=FakeProbes()) == 1
    assert message in capsys.readouterr().err


def test_mtp_choice_is_required_and_exclusive(env, capsys):
    argv = env.argv("draft", "--no-mtp")  # assistant and --no-mtp together
    assert cli.main(argv, probes=FakeProbes()) == 1
    assert "--assistant" in capsys.readouterr().err
    neither = [a for a in env.argv("draft", mtp=False) if a != "--no-mtp"]
    assert cli.main(neither, probes=FakeProbes()) == 1
    assert "explicitly --no-mtp" in capsys.readouterr().err


def test_help_names_infernix_the_presets_and_the_fp8_profile(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    for token in (*build.PRESETS, "infernix-serve", "FP8", "--assistant", "--no-mtp", "--out-dir", "--work-dir",
                  "--calibration-text", "--gptq-python", "--score", "--serve", "setup-gptq", "launcher",
                  "serve-check"):
        assert token in text


def test_module_entry_point_runs_from_the_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "tools.convert.gemma4_build", "--list-presets"], cwd=build.REPO_ROOT,
        capture_output=True, text=True, check=True,
    )
    assert "best-small" in result.stdout and "best-medium" in result.stdout and "lite-" not in result.stdout


def test_the_gptq_interpreter_comes_from_the_argument_then_the_environment(env, monkeypatch):
    monkeypatch.delenv(build.GPTQ_PYTHON_ENV, raising=False)
    default = build.default_gptq_python()
    assert default.parent.parent == build.REPO_ROOT / ".venv-gptq"
    monkeypatch.setenv(build.GPTQ_PYTHON_ENV, str(env.gptq_python))
    assert build.default_gptq_python() == env.gptq_python
    args = cli.build_parser().parse_args([a for a in env.argv("best-small") if a not in
                                          ("--gptq-python", str(env.gptq_python))])
    assert cli.resolve_config(args)[0].gptq_python == env.gptq_python.absolute()
    explicit = cli.build_parser().parse_args(env.argv("best-small", "--gptq-python", str(env.root / "other.exe")))
    assert cli.resolve_config(explicit)[0].gptq_python == (env.root / "other.exe").absolute()


def test_llama_cpp_paths_use_explicit_arguments_then_environment(env, monkeypatch):
    monkeypatch.setenv(build.LLAMA_CPP_ENV, str(env.root / "env-bin"))
    monkeypatch.setenv(build.LLAMA_CPP_SOURCE_ENV, str(env.root / "env-src"))
    cfg = cli.resolve_config(cli.build_parser().parse_args(env.argv("iq3-xs")))[0]
    assert cfg.llama_cpp == env.llama_cpp.resolve() and cfg.llama_cpp_source == env.llama_source.resolve()
    argv = env.argv("iq3-xs")
    for flag in ("--llama-cpp", "--llama-cpp-source"):
        index = argv.index(flag)
        del argv[index:index + 2]
    cfg = cli.resolve_config(cli.build_parser().parse_args(argv))[0]
    assert cfg.llama_cpp == (env.root / "env-bin").resolve()
    assert cfg.llama_cpp_source == (env.root / "env-src").resolve()
    monkeypatch.delenv(build.LLAMA_CPP_ENV)
    monkeypatch.delenv(build.LLAMA_CPP_SOURCE_ENV)
    cfg = cli.resolve_config(cli.build_parser().parse_args(argv))[0]
    assert cfg.llama_cpp is None and cfg.llama_cpp_source is None


# ---------------------------------------------------------------- preflight with fake probes


def test_healthy_machine_passes_every_preset(env):
    for preset in build.PRESETS:
        report = preflight(env, env.config(preset))
        assert not report.failed, [item.render() for item in report.failed]


@pytest.mark.parametrize(
    "preset, probes, what, cause, action",
    [
        ("best-small", FakeProbes(gpu=checks.GpuInfo("RTX", 32607, 8000, "12.0")), "GPU memory", "needs 12 GiB",
         "nvidia-smi"),
        ("best-small", FakeProbes(gpu=None), "GPU", "no NVIDIA GPU", "driver"),
        ("best-small", FakeProbes(ram=(64 * 2**30, 20 * 2**30)), "RAM", "20 GiB available", "close other programs"),
        ("best-small", FakeProbes(disk=30e9), "disk, work dir and output (same drive)", "30 GB free", "--work-dir"),
        ("best-small", FakeProbes(git=("b" * 40, [" M tools/x.py"])), "infernix checkout", "tracked file",
         "commit or stash"),
        ("best-small", FakeProbes(formats=False), "infernix-serve", "does not know the q4_g256_fp16", "rebuild"),
        ("best-small", FakeProbes(cuda=False), "converter Python", "CUDA", "CUDA build of torch"),
        ("best-small", FakeProbes(missing_packages=("tokenizers",)), "converter Python packages", "tokenizers",
         "pip install tokenizers"),
        ("best-small", FakeProbes(gptq_versions={"transformers": "99.0.0"}), "GPTQ environment",
         f"transformers 99.0.0 (pinned {PINS['transformers']})", "setup-gptq"),
    ],
)
def test_preflight_failures_name_the_cause_and_the_next_action(env, preset, probes, what, cause, action):
    item = finding(preflight(env, env.config(preset), probes), what)
    assert item.level == "fail" and cause in item.detail + item.what and action in item.action


def test_comfortable_disk_threshold_warns_instead_of_failing(env):
    item = finding(preflight(env, env.config("best-small"), FakeProbes(disk=90e9)), "disk, work dir and output (same drive)")
    assert item.level == "warn" and "103 GB" in item.detail


def test_missing_gptq_environment_points_at_setup_gptq(env, tmp_path):
    cfg = env.config("best-medium", gptq_python=tmp_path / "missing" / "python.exe")
    item = finding(preflight(env, cfg), "GPTQ environment")
    assert item.level == "fail" and "setup-gptq --venv" in item.action and build.GPTQ_PYTHON_ENV in item.action


def test_iq3_missing_tools_are_actionable_and_do_not_require_gptq(env):
    cfg = env.config("iq3-xs", llama_cpp=None, llama_cpp_source=None, gptq_python=None)
    report = preflight(env, cfg)
    assert "--llama-cpp" in finding(report, "llama-quantize").action
    assert "--llama-cpp" in finding(report, "llama-imatrix").action
    assert "--llama-cpp-source" in finding(report, "llama.cpp source").action
    assert not any(item.what == "GPTQ environment" for item in report.findings)


def test_iq3_needs_gguf_conversion_dependencies(env):
    item = finding(preflight(env, env.config("iq3-xs"), FakeProbes(missing_packages=("gguf",))),
                   "converter Python packages")
    assert item.level == "fail" and "pip install gguf" in item.action


def test_model_checks_reuse_the_converter_and_name_the_fix(tmp_path):
    for label, kwargs in (
        ("fp16", dict(dtype="F16")),
        ("moe", dict(config=gemma_config(hidden=2816))),
        ("short", dict(config=gemma_config(layers=30))),
    ):
        folder = make_model(tmp_path / label, **kwargs)
        with pytest.raises(ValueError):
            checks.inspect_model(folder)
    folder = make_model(tmp_path / "ok")
    facts = checks.inspect_model(folder)
    assert facts["shards"] == 1 and facts["vision"] is False
    (folder / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(ValueError, match="missing weight files"):
        checks.inspect_model(folder)


def test_bad_model_and_assistant_report_actions(env):
    (env.model / "config.json").write_text(json.dumps({"architectures": ["LlamaForCausalLM"]}))
    (env.assistant / "config.json").write_text(json.dumps({"architectures": ["Other"]}))
    report = preflight(env, env.config("draft"))
    assert "Hugging Face directory" in finding(report, "model").action
    assert "--no-mtp" in finding(report, "MTP assistant").action


def test_server_missing_dlls_and_missing_binary(env):
    warn = finding(preflight(env, env.config("draft"), FakeProbes(dlls=["avcodec-*.dll"])), "server DLLs")
    assert warn.level == "warn" and "--cuda-bin" in warn.action
    report = checks.preflight(env.config("draft"), build.ServeSettings(), None, FakeProbes(), "x", {})
    assert "--serve" in finding(report, "infernix-serve").action
    skipped = checks.preflight(env.config("draft"), build.ServeSettings(launcher=False), None, FakeProbes(), "x", {})
    assert finding(skipped, "launcher").level == "ok"


def test_score_prerequisites_are_checked(env):
    env.perplexity.unlink()
    cfg = env.config("draft", score=True)
    item = finding(preflight(env, cfg), "--score prerequisites")
    assert item.level == "fail" and "infernix-perplexity" in item.detail and "drop --score" in item.action


def test_out_and_work_dirs_must_stay_outside_the_source(env):
    cfg = env.config("draft", out_dir=env.model / "out", work_dir=env.model / "work")
    report = preflight(env, cfg)
    assert finding(report, "--out-dir").level == "fail" and finding(report, "--work-dir").level == "fail"


# ---------------------------------------------------------------- runner, resume, receipts


def write_receipt(cfg: build.BuildConfig, status: str = "storage-verified; runtime-and-quality-unverified") -> None:
    data = cfg.artifact.read_bytes()
    receipt = {"status": status, "revision": "r" * 40, "recipe": "test", "seconds": 1.0,
               "artifact": {"file_bytes": len(data), "file_sha256": {cfg.artifact.name: hashlib.sha256(data).hexdigest()}}}
    Path(str(cfg.artifact) + ".build.json").write_text(json.dumps(receipt))


class FakeRunner:
    """Stands in for the child processes: records commands and creates each step's declared outputs."""

    def __init__(self, cfg: build.BuildConfig, steps: list[build.Step], fail: set[str] = frozenset()):
        self.cfg, self.fail, self.calls = cfg, set(fail), []
        self.by_argv = {tuple(step.argv): step for step in steps if step.argv}

    def __call__(self, argv, log_path, env, cwd, status):
        step = self.by_argv[tuple(argv)]
        self.calls.append(step.id)
        assert cwd == build.REPO_ROOT
        log_path.write_text(f"ran {step.id}\n")
        if step.id in self.fail:
            return 3
        for path in step.outputs:
            if path.suffix in (".ninfer",):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"artifact bytes")
                write_receipt(self.cfg)
            elif path.name.endswith((".json", ".jsonl")) or path.suffix in (".txt", ".pt", ".gguf", ".nklref"):
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text("x")
            elif not str(path).endswith((".build.json", ".conversion.json")):
                path.mkdir(parents=True, exist_ok=True)
                (path / "blob").write_bytes(b"x" * 10)
        return 0


def make_builder(env, cfg, *, fail=(), disk_free=None, out=None, **serve):
    steps = build.build_plan(cfg)
    runner = FakeRunner(cfg, steps, set(fail))
    identity = build.build_identity(cfg)
    messages = [] if out is None else out
    builder = build.Builder(
        cfg, steps, identity, serve_settings(env, **serve), server_exe=env.server, runner=runner,
        out=messages.append, disk_free=disk_free or (lambda path: 10**13),
        artifact_info=launcher.ArtifactInfo(True, False, 262144),
    )
    return builder, runner, messages


def test_failure_names_the_log_and_a_rerun_skips_finished_steps(env):
    cfg = env.config("best-small")
    builder, runner, messages = make_builder(env, cfg, fail={"gptq"})
    with pytest.raises(build.BuildError) as error:
        builder.run()
    assert "step gptq failed (exit code 3)" in str(error.value)
    assert str(cfg.work("logs", "04-gptq.log")) in str(error.value) and "rerun the same command" in str(error.value)
    assert runner.calls == ["calibration-render", "calibration-tokens", "text-tree", "gptq"]
    state = json.loads(cfg.work("build-state.json").read_text())
    assert state["steps"]["gptq"]["status"] == "failed" and state["steps"]["text-tree"]["status"] == "done"

    again, second, messages = make_builder(env, cfg)
    again.run()
    assert second.calls == ["gptq", "gptq-check", "convert"]
    assert sum("already done, skipped" in line for line in messages) == 3
    assert not cfg.work("offload").exists()  # scratch is removed as soon as GPTQ is done
    assert cfg.work("text60").exists()  # the tree stays while disk is plentiful


def test_a_third_run_after_success_skips_everything(env):
    cfg = env.config("draft")
    first, runner, _ = make_builder(env, cfg)
    first.run()
    second, rerun, messages = make_builder(env, cfg)
    second.run()
    assert runner.calls == ["convert"] and rerun.calls == []
    assert sum("already done" in line for line in messages) == 3


def test_deliverables_summary_and_the_launcher_carry_the_operator_profile(env):
    cfg = env.config("best-small")
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    assert builder.cleanup() is True and not cfg.work_dir.exists()
    names = sorted(path.name for path in env.out.iterdir())
    assert names == sorted([cfg.artifact.name, cfg.artifact.name + ".build.json",
                            cfg.artifact.name + ".conversion.json", f"serve-{cfg.name}.cmd", "BUILD-SUMMARY.md",
                            f"{cfg.name}.gemma4-build.json"])
    summary = (env.out / "BUILD-SUMMARY.md").read_text()
    digest = hashlib.sha256(b"artifact bytes").hexdigest()
    assert digest in summary and "best-small" in summary and "infernix-serve" in summary
    assert "FP8 KV cache and FP8 attention QK" in summary and "--any-model-name" in summary
    assert "`131k`" in summary and "mean KL 0.1213" in summary
    receipt = json.loads((env.out / f"{cfg.name}.gemma4-build.json").read_text())
    assert receipt["schema"] == build.BUILD_SCHEMA and receipt["sha256"] == digest and receipt["mtp"] is True
    assert receipt["serve"] == {"kv": "fp8", "qk": "fp8"}
    plan = build.parse_launcher(env.out / f"serve-{cfg.name}.cmd")
    assert plan.server == env.server.resolve() and plan.artifact == cfg.artifact.resolve()
    assert plan.effective_profile() == {
        "host": "0.0.0.0", "port": 8001, "model_id": "131k", "model_match": "any", "max_context": 131072,
        "kv_dtype": "fp8", "gemma4_qk": "fp8", "prefill_chunk": "auto", "draft_tokens": "1",
        "logits_dtype": "bf16", "host_context_mib": 8192, "device_snapshot_slots": 0,
    }
    flags = list(plan.flags)
    assert flags[flags.index("--spec") + 1] == "mtp" and flags[flags.index("--max-concurrency") + 1] == "1"
    assert "--no-prefix-reuse" not in flags and "--vision" not in flags


def test_keep_work_and_protected_directories(env):
    cfg = env.config("draft", keep_work=True)
    builder, _, _ = make_builder(env, cfg)
    builder.run()
    assert builder.cleanup() is False and cfg.work_dir.exists()
    for bad in (env.out, env.out.parent, env.model, build.REPO_ROOT):
        guarded = env.config("draft", work_dir=bad)
        other, _, _ = make_builder(env, guarded)
        other.state = {}
        with pytest.raises(build.BuildError, match="refusing to delete"):
            other.cleanup()


def test_a_work_dir_made_with_other_arguments_is_refused(env):
    cfg = env.config("best-small")
    builder, _, _ = make_builder(env, cfg, fail={"gptq"})
    with pytest.raises(build.BuildError):
        builder.run()
    other = env.config("best-medium", name=cfg.name)
    again, _, _ = make_builder(env, other)
    with pytest.raises(build.BuildError, match=r"different arguments \(.*preset"):
        again.run()
    assert "different arguments" in finding(preflight(env, other), "work dir").detail


def test_calibration_file_change_changes_the_identity(env):
    cfg = env.config("best-small")
    before = build.fingerprint_of(build.build_identity(cfg))
    env.calibration.write_text("other text")
    assert build.fingerprint_of(build.build_identity(cfg)) != before


@pytest.mark.parametrize("preset", ["iq3-xs", "iq3-xxs"])
def test_iq3_runner_resumes_after_quantizer_failure_and_records_its_representation(env, preset):
    cfg = env.config(preset)
    builder, first, _ = make_builder(env, cfg, fail={"gguf-quantize"})
    with pytest.raises(build.BuildError, match="step gguf-quantize failed"):
        builder.run()
    assert first.calls == ["calibration-render", "gguf-bf16", "gguf-q8", "imatrix", "gguf-quantize"]
    assert not cfg.work("model-q8_0.gguf").exists()
    assert cfg.work("model-bf16.gguf").exists()
    again, resumed, _ = make_builder(env, cfg)
    again.run()
    assert resumed.calls == ["gguf-quantize", "convert"]
    assert not cfg.work("model-bf16.gguf").exists()
    summary = (cfg.out_dir / "BUILD-SUMMARY.md").read_text()
    assert "IQ3 GGUF blocks, AllowA8 for prefill and decode" in summary
    assert "own Q8_0 reference" in summary
    assert "4-bit weights" not in summary
    profile = build.parse_launcher(cfg.out_dir / f"serve-{cfg.name}.cmd")
    assert profile.draft_tokens == "1" and profile.prefill_chunk == "auto"
    third, rerun, _ = make_builder(env, cfg)
    third.run()
    assert not rerun.calls


def test_iq3_build_identity_pins_llama_source_and_allocation(env):
    before = build.build_identity(env.config("iq3-xs"))
    assert before["gguf_recipe"]["type"] == "IQ3_XS"
    after = build.build_identity(env.config("iq3-xs", llama_cpp_source=env.root / "other-source"))
    assert build.fingerprint_of(before) != build.fingerprint_of(after)


def test_a_foreign_non_empty_work_dir_is_refused(env):
    cfg = env.config("draft")
    cfg.work_dir.mkdir(parents=True)
    (cfg.work_dir / "precious.txt").write_text("mine")
    builder, runner, _ = make_builder(env, cfg)
    with pytest.raises(build.BuildError, match="not a gemma4_build work dir"):
        builder.run()
    assert runner.calls == [] and (cfg.work_dir / "precious.txt").exists()


def test_an_existing_output_is_never_overwritten(env):
    cfg = env.config("draft")
    cfg.out_dir.mkdir(parents=True)
    cfg.artifact.write_bytes(b"someone else's file")
    builder, runner, _ = make_builder(env, cfg)
    with pytest.raises(build.BuildError, match="Output already exists"):
        builder.run()
    assert runner.calls == [] and cfg.artifact.read_bytes() == b"someone else's file"
    assert finding(preflight(env, cfg), "output").level == "fail"


def test_big_intermediates_are_removed_only_when_disk_is_short(env):
    cfg = env.config("best-small")
    short, _, messages = make_builder(env, cfg, disk_free=lambda path: 1e9)
    short.run()
    assert not cfg.work("text60").exists() and any("disk is short" in line for line in messages)
    roomy = env.config("best-small", out_dir=env.root / "out2", work_dir=env.root / "out2" / "work", name="roomy")
    ample, _, _ = make_builder(env, roomy, disk_free=lambda path: 10**13)
    ample.run()
    assert roomy.work("text60").exists()
    # a pruned tree does not make the next run redo the step that wrote it
    rerun, runner, _ = make_builder(env, cfg, disk_free=lambda path: 1e9)
    rerun.run()
    assert runner.calls == []


def test_receipt_check_refuses_an_unverified_or_mismatched_receipt(env):
    cfg = env.config("draft")
    builder, _, _ = make_builder(env, cfg)
    cfg.out_dir.mkdir(parents=True)
    cfg.artifact.write_bytes(b"abc")
    builder.open_work_dir()
    builder._log_path = None
    with pytest.raises(RuntimeError, match="receipt missing"):
        build.check_receipt(builder)
    write_receipt(cfg, status="failed")
    with pytest.raises(RuntimeError, match="status is 'failed'"):
        build.check_receipt(builder)
    write_receipt(cfg)
    cfg.artifact.write_bytes(b"abcd")
    with pytest.raises(RuntimeError, match="4 bytes"):
        build.check_receipt(builder)


def test_main_builds_end_to_end_with_a_fake_runner(env, capsys, monkeypatch):
    cfg = env.config("best-medium", name="m-medium", work_dir=env.out / "work-m-medium", python=sys.executable)
    plan = build.build_plan(cfg)
    runner = FakeRunner(cfg, plan)
    monkeypatch.setattr(cli, "Builder", partial(build.Builder, runner=runner, disk_free=lambda path: 10**13))
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144))
    argv = env.argv("best-medium", "--name", "m-medium", "--yes")
    assert cli.main(argv, probes=FakeProbes()) == 0
    out = capsys.readouterr().out
    assert "Done. Artifact:" in out and "Work dir removed." in out
    assert runner.calls == ["calibration-render", "calibration-tokens", "text-tree", "gptq", "gptq-check", "convert"]
    assert (env.out / "BUILD-SUMMARY.md").is_file() and not (env.out / "work-m-medium").exists()


# ---------------------------------------------------------------- child processes


def test_run_child_streams_to_the_log_and_returns_the_exit_code(tmp_path):
    log = tmp_path / "child.log"
    seen = []
    code = build.run_child([sys.executable, "-c", "print('hello'); import sys; print('oops', file=sys.stderr)"],
                           log, {}, tmp_path, seen.append)
    assert code == 0 and "hello" in log.read_text() and "oops" in log.read_text()
    assert build.run_child([sys.executable, "-c", "raise SystemExit(7)"], log, {}, tmp_path, seen.append) == 7
    assert build.tail_line(log).startswith("$ ")


def test_batch_files_run_through_cmd_on_windows(monkeypatch):
    monkeypatch.setattr(build.os, "name", "nt")
    assert build.popen_argv(["C:/x/tool.cmd", "a"]) == ["cmd", "/c", "C:/x/tool.cmd", "a"]
    assert build.popen_argv(["C:/x/tool.exe", "a"]) == ["C:/x/tool.exe", "a"]


# ---------------------------------------------------------------- calibration text


def test_calibration_export_reproduces_the_authors_file_layout(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    source = tmp_path / "c.parquet"
    content = "Don\u2019t panic.\nSecond line \u00e9\n"
    pq.write_table(pa.table({"content": [content]}), source)
    out = tmp_path / "source.txt"
    digest = build.export_author_text(source, out, expected_sha256=None)
    data = out.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and b"\r\n" in data and data.endswith(b"\r\n")
    assert data.decode("utf-8-sig").replace("\r\n", "\n") == content.encode("utf-8").decode("cp437") + "\n"
    assert digest == hashlib.sha256(data).hexdigest()
    with pytest.raises(RuntimeError, match="expected"):
        build.export_author_text(source, tmp_path / "other.txt", expected_sha256="0" * 64)
    assert not (tmp_path / "other.txt").exists()


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_is_hash_pinned_and_never_silent(tmp_path, monkeypatch):
    payload = b"parquet bytes"
    monkeypatch.setattr(build.urllib.request, "urlopen", lambda request, timeout: FakeResponse(payload))
    out = tmp_path / "f.parquet"
    messages = []
    build.download_pinned("https://example.invalid/f", out, hashlib.sha256(payload).hexdigest(), messages.append)
    assert out.read_bytes() == payload and messages
    with pytest.raises(RuntimeError, match="refusing the download"):
        build.download_pinned("https://example.invalid/f", tmp_path / "bad", "0" * 64, messages.append)
    assert not (tmp_path / "bad").exists() and not (tmp_path / "bad.part").exists()

    def broken(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr(build.urllib.request, "urlopen", broken)
    with pytest.raises(RuntimeError, match="--calibration-text"):
        build.download_pinned("https://example.invalid/f", tmp_path / "x", "0" * 64, messages.append)


# ---------------------------------------------------------------- launcher


def lines_profile(lines: list[str], tmp_path: Path) -> build.LauncherCommand:
    """Write generated launcher lines to a file and parse them back, as serve-check does."""
    cmd = tmp_path / "serve.cmd"
    cmd.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    return build.parse_launcher(cmd)


def test_the_default_launcher_is_the_operator_roleplay_profile(tmp_path):
    lines = launcher.launcher_lines("C:/bin/infernix-serve.exe", "D:/models/a.ninfer", launcher.ServeProfile(),
                                    mtp=True, vision=False)
    plan = lines_profile(lines, tmp_path)
    assert plan.server == Path("C:/bin/infernix-serve.exe") and plan.artifact == Path("D:/models/a.ninfer")
    assert plan.effective_profile() == {
        "host": "0.0.0.0", "port": 8001, "model_id": "131k", "model_match": "any", "max_context": 131072,
        "kv_dtype": "fp8", "gemma4_qk": "fp8", "prefill_chunk": "auto", "draft_tokens": "1",
        "logits_dtype": "bf16", "host_context_mib": 8192, "device_snapshot_slots": 0,
    }
    flags = list(plan.flags)
    assert flags[flags.index("--kv-capacity") + 1] == "auto" and flags[flags.index("--vram-headroom-mib") + 1] == "1024"
    assert flags[flags.index("--max-concurrency") + 1] == "1"
    for fork_only in ("--model-match", "--kv-headroom-mib", "--device-state-slots", "--no-prefix-reuse", "--top-k",
                      "--chat-template"):
        assert fork_only not in flags
    assert "infernix-serve" in lines[2] and "FP8 KV cache, FP8 attention QK" in lines[2]


def test_model_match_exact_emits_no_flag_and_any_emits_any_model_name():
    exact = "\n".join(launcher.launcher_lines("S", "A", launcher.ServeProfile(model_match="exact"), mtp=True,
                                             vision=False))
    assert "--any-model-name" not in exact and "--model-match" not in exact
    anyname = "\n".join(launcher.launcher_lines("S", "A", launcher.ServeProfile(), mtp=True, vision=False))
    assert "--any-model-name" in anyname
    with pytest.raises(ValueError, match="--model-match"):
        launcher.check_serve_settings(launcher.ServeProfile(model_match="none"))


def test_launcher_settings_flow_into_the_parsed_profile(tmp_path):
    profile = launcher.ServeProfile(host="127.0.0.1", port=8080, alias="160k", context=65536, kv_dtype="bf16",
                                    gemma4_qk="bf16", prefill_chunk="2048", draft_tokens="2", logits_dtype="fp32",
                                    host_context_mib=4096, vram_headroom_mib=512, top_k=64)
    plan = lines_profile(launcher.launcher_lines("S", "A", profile, mtp=True, vision=True, cuda_bin="C:\\cuda\\bin"),
                         tmp_path)
    assert plan.effective_profile() == {
        "host": "127.0.0.1", "port": 8080, "model_id": "160k", "model_match": "any", "max_context": 65536,
        "kv_dtype": "bf16", "gemma4_qk": "bf16", "prefill_chunk": "2048", "draft_tokens": "2",
        "logits_dtype": "fp32", "host_context_mib": 4096, "device_snapshot_slots": 0,
    }
    assert "--gemma4-qk" not in plan.flags and "--vision" in plan.flags
    assert plan.flags[plan.flags.index("--top-k") + 1] == "64"
    assert plan.env == {"PATH": "C:\\cuda\\bin"}
    assert launcher.default_alias(131072) == "131k" and launcher.default_alias(262144) == "262k"


def test_without_an_assistant_the_launcher_does_not_speculate():
    lines = launcher.launcher_lines("S", "A", launcher.ServeProfile(), mtp=False, vision=False)
    assert not any("--spec" in line or "--draft-tokens" in line for line in lines)


def test_serve_argv_matches_the_launcher_flags():
    profile = launcher.ServeProfile(draft_tokens="3")
    argv = launcher.serve_argv(Path("S"), Path("A"), profile, mtp=True, vision=True)
    assert argv[:2] == ["S", "A"] and argv[argv.index("--draft-tokens") + 1] == "3" and "--vision" in argv
    plain = launcher.serve_argv(Path("S"), Path("A"), profile, mtp=False, vision=False)
    assert "--spec" not in plain and "--vision" not in plain


@pytest.mark.parametrize("draft", ["1", "2", "7", "auto"])
def test_launcher_accepts_valid_drafts(draft):
    launcher.check_serve_settings(launcher.ServeProfile(draft_tokens=draft))


@pytest.mark.parametrize("draft", ["0", "8", "-1", "six", ""])
def test_launcher_rejects_invalid_drafts(draft):
    with pytest.raises(ValueError, match="--draft-tokens must be"):
        launcher.launcher_lines("S", "A", launcher.ServeProfile(draft_tokens=draft), mtp=True, vision=False)


def test_gemma4_qk_fp8_is_refused_without_fp8_kv_and_bad_values_are_named():
    with pytest.raises(ValueError, match=r"--gemma4-qk fp8 needs --kv-dtype fp8 .*bf16 KV"):
        launcher.launcher_lines("S", "A", launcher.ServeProfile(kv_dtype="bf16", gemma4_qk="fp8"), mtp=False,
                                vision=False)
    bf16 = launcher.launcher_lines("S", "A", launcher.ServeProfile(kv_dtype="bf16", gemma4_qk="bf16"), mtp=False,
                                   vision=False)
    assert "--gemma4-qk" not in "\n".join(bf16)
    with pytest.raises(ValueError, match="--gemma4-qk must be bf16 or fp8"):
        launcher.check_serve_settings(launcher.ServeProfile(gemma4_qk="int8"))


@pytest.mark.parametrize("chunk", ["128", "256", "2048", "16384", "auto"])
def test_prefill_chunk_accepts_multiples_of_128_and_auto(chunk):
    launcher.check_serve_settings(launcher.ServeProfile(prefill_chunk=chunk))


@pytest.mark.parametrize("chunk", ["", "0", "64", "100", "129", "16512", "-128", "Auto", "2048k"])
def test_prefill_chunk_refuses_everything_else(chunk):
    with pytest.raises(ValueError, match="--prefill-chunk must be 'auto' or a multiple of 128 in 128..16384"):
        launcher.check_serve_settings(launcher.ServeProfile(prefill_chunk=chunk))


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(top_k=65), "--top-k"), (dict(top_k=-1), "--top-k"), (dict(host_context_mib=0), "--host-context-mib"),
        (dict(vram_headroom_mib=-1), "--vram-headroom-mib"), (dict(logits_dtype="fp16"), "--logits-dtype"),
        (dict(kv_dtype="int8"), "--kv-dtype"), (dict(port=0), "--port"), (dict(context=0), "--context"),
    ],
)
def test_other_settings_are_refused_with_the_flag_name(bad, message):
    with pytest.raises(ValueError, match=message):
        launcher.check_serve_settings(launcher.ServeProfile(**bad))


def test_thinking_control_flags_are_optional_and_flow_into_the_launcher(tmp_path):
    default = "\n".join(launcher.launcher_lines("S", "A", launcher.ServeProfile(), mtp=True, vision=False))
    for flag in ("--default-thinking-budget", "--thinking-budget-message", "--reasoning-loop"):
        assert flag not in default  # the operator profile is unchanged unless asked
    profile = launcher.ServeProfile(default_thinking_budget=6000, thinking_budget_message="Time to act now: <channel|>",
                                    reasoning_loop="conclude")
    plan = lines_profile(launcher.launcher_lines("S", "A", profile, mtp=True, vision=False), tmp_path)
    assert plan.flags[plan.flags.index("--default-thinking-budget") + 1] == "6000"
    # A message with spaces and CMD metacharacters is one quoted word and reads back whole.
    assert plan.flags[plan.flags.index("--thinking-budget-message") + 1] == "Time to act now: <channel|>"
    assert plan.flags[plan.flags.index("--reasoning-loop") + 1] == "conclude"
    assert plan.effective_profile()["draft_tokens"] == "1"
    only_loop = lines_profile(
        launcher.launcher_lines("S", "A", launcher.ServeProfile(reasoning_loop="stop"), mtp=False, vision=False),
        tmp_path)
    assert only_loop.flags[only_loop.flags.index("--reasoning-loop") + 1] == "stop"
    assert "--default-thinking-budget" not in only_loop.flags


@pytest.mark.parametrize(
    "bad, message",
    [
        (dict(default_thinking_budget=0), "--default-thinking-budget"),
        (dict(thinking_budget_message=""), "--thinking-budget-message"),
        (dict(thinking_budget_message='say "go"'), "--thinking-budget-message"),
        (dict(thinking_budget_message="100% done"), "--thinking-budget-message"),
        (dict(thinking_budget_message="two\nlines"), "--thinking-budget-message"),
        (dict(reasoning_loop="off"), "--reasoning-loop"),
    ],
)
def test_thinking_control_settings_are_refused_with_the_flag_name(bad, message):
    with pytest.raises(ValueError, match=message):
        launcher.check_serve_settings(launcher.ServeProfile(**bad))


def test_launcher_refuses_quotes_and_neutralizes_percent():
    with pytest.raises(ValueError, match="quotes"):
        launcher.cmd_path(Path('C:/a"b'))
    with pytest.raises(ValueError, match="quotes"):
        launcher.cmd_path(Path("C:/a\nb"))
    assert "%%PATH%%" in launcher.cmd_path(Path("C:/a%PATH%b"))
    for bad in ("bad alias", 'a"b', "%x%", "-lead"):
        with pytest.raises(ValueError, match="--alias"):
            launcher.check_serve_settings(launcher.ServeProfile(alias=bad))
    with pytest.raises(ValueError, match="--host"):
        launcher.check_serve_settings(launcher.ServeProfile(host="a b"))


def test_write_launcher_detects_components_and_never_overwrites(tmp_path):
    server = tmp_path / "s.exe"
    artifact = tmp_path / "a.ninfer"
    template = tmp_path / "my template.jinja"
    for path in (server, artifact, template):
        path.write_bytes(b"x")
    out = tmp_path / "serve.cmd"
    info = launcher.ArtifactInfo(mtp=True, vision=True, max_context=262144)
    launcher.write_launcher(out, artifact, server, launcher.ServeProfile(chat_template=template), info=info)
    data = out.read_bytes()
    assert b"\r\n" in data and data.rstrip().endswith(b"%errorlevel%")
    plan = build.parse_launcher(out)
    assert "--vision" in plan.flags and plan.draft_tokens == "1"
    assert Path(plan.flags[plan.flags.index("--chat-template") + 1]) == template.resolve()  # a path with a space
    with pytest.raises(FileExistsError):
        launcher.write_launcher(out, artifact, server, info=info)
    with pytest.raises(ValueError, match="position limit"):
        launcher.write_launcher(tmp_path / "x.cmd", artifact, server, launcher.ServeProfile(context=300000), info=info)
    with pytest.raises(ValueError, match="--chat-template file not found"):
        launcher.write_launcher(tmp_path / "y.cmd", artifact, server,
                                launcher.ServeProfile(chat_template=tmp_path / "none.jinja"), info=info)
    off = tmp_path / "nomtp.cmd"
    launcher.write_launcher(off, artifact, server, info=info, no_mtp=True)
    assert b"--spec" not in off.read_bytes()


def test_launcher_subcommand(tmp_path, monkeypatch, capsys):
    server, artifact = tmp_path / "s.exe", tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(False, False, 262144))
    out = tmp_path / "serve.cmd"
    code = cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out),
                     "--port", "8002", "--alias", "262k", "--context", "262144", "--host", "127.0.0.1"])
    assert code == 0 and "MTP off" in capsys.readouterr().out
    plan = build.parse_launcher(out)
    assert (plan.host, plan.port, plan.alias, plan.context) == ("127.0.0.1", 8002, "262k", 262144)
    assert "--spec" not in plan.flags
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out)]) == 1


def test_launcher_subcommand_defaults_and_overrides(tmp_path, monkeypatch, capsys):
    server, artifact = tmp_path / "s.exe", tmp_path / "a.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144))
    base = ["launcher", "--artifact", str(artifact), "--serve", str(server)]
    default = tmp_path / "default.cmd"
    assert cli.main(base + ["--out", str(default)]) == 0
    profile = build.parse_launcher(default).effective_profile()
    assert (profile["host"], profile["port"], profile["gemma4_qk"], profile["prefill_chunk"]) == (
        "0.0.0.0", 8001, "fp8", "auto")
    exact = tmp_path / "exact.cmd"
    bf16 = tmp_path / "bf16.cmd"
    assert cli.main(base + ["--out", str(exact), "--model-match", "exact"]) == 0
    assert cli.main(base + ["--out", str(bf16), "--gemma4-qk", "bf16", "--prefill-chunk", "512",
                            "--draft-tokens", "3", "--top-k", "64"]) == 0
    assert build.parse_launcher(exact).model_match == "exact"
    other = build.parse_launcher(bf16)
    assert (other.gemma4_qk, other.prefill_chunk, other.draft_tokens) == ("bf16", "512", "3")
    refused = tmp_path / "refused.cmd"
    capsys.readouterr()
    assert cli.main(base + ["--out", str(refused), "--gemma4-qk", "fp8", "--kv-dtype", "bf16"]) == 1
    assert "--gemma4-qk fp8 needs --kv-dtype fp8" in capsys.readouterr().err and not refused.exists()
    assert cli.main(base + ["--out", str(refused), "--prefill-chunk", "100"]) == 1
    assert "multiple of 128" in capsys.readouterr().err and not refused.exists()
    with pytest.raises(SystemExit):
        cli.main(base + ["--out", str(refused), "--model-match", "some"])


def test_build_summary_and_launcher_follow_the_serve_settings(env):
    cfg = env.config("draft", out_dir=env.root / "o2", work_dir=env.root / "o2" / "w", name="ex")
    builder, _, _ = make_builder(env, cfg, model_match="exact", gemma4_qk="bf16", prefill_chunk="1024",
                                 host="127.0.0.1", port=8080)
    builder.run()
    summary = (env.root / "o2" / "BUILD-SUMMARY.md").read_text()
    assert "--any-model-name" not in summary and "BF16 attention QK" in summary and "--prefill-chunk 1024" in summary
    plan = build.parse_launcher(env.root / "o2" / "serve-ex.cmd")
    assert (plan.model_match, plan.gemma4_qk, plan.prefill_chunk, plan.host, plan.port) == (
        "exact", "bf16", "1024", "127.0.0.1", 8080)


def test_one_command_build_passes_the_serve_arguments_to_the_settings(env, capsys):
    args = cli.build_parser().parse_args(env.argv("draft", "--gemma4-qk", "bf16", "--prefill-chunk", "2048",
                                                  "--host-context-mib", "4096", "--top-k", "64"))
    serve = cli.resolve_config(args)[1]
    assert (serve.gemma4_qk, serve.prefill_chunk, serve.host_context_mib, serve.top_k) == ("bf16", "2048", 4096, 64)
    assert cli.resolve_config(cli.build_parser().parse_args(env.argv("draft")))[1].gemma4_qk == "fp8"
    for extra, message in ((("--gemma4-qk", "fp8", "--kv-dtype", "bf16"), "--gemma4-qk fp8 needs --kv-dtype fp8"),
                           (("--prefill-chunk", "100"), "--prefill-chunk must be")):
        assert cli.main(env.argv("draft", *extra, "--dry-run", "--yes"), probes=FakeProbes()) == 1
        assert message in capsys.readouterr().err


def test_find_server_exe_looks_in_apps_release_and_prefers_a_build_that_knows_the_format(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    exe = launcher.exe_name("infernix-serve")
    old, new = checkout / "build-old" / "apps" / "Release", checkout / "apps" / "Release"
    for folder, payload in ((old, b"q4_g256_fp16"), (new, b"q4_g32_fp16 only")):
        folder.mkdir(parents=True)
        (folder / exe).write_bytes(payload)
    os.utime(old / exe, (1, 1))
    assert launcher.find_server_exe(checkout) == new / exe
    assert launcher.find_server_exe(checkout, ("q4_g256_fp16",)) == old / exe
    assert launcher.find_server_exe(tmp_path / "nowhere") is None


def test_missing_dlls_lists_patterns_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    exe = tmp_path / "s.exe"
    exe.write_bytes(b"x")
    (tmp_path / "avcodec-63.dll").write_bytes(b"x")
    missing = launcher.missing_dlls(exe)
    assert "avcodec-*.dll" not in missing and "cublas64_*.dll" in missing


# ---------------------------------------------------------------- setup-gptq


def test_setup_gptq_dry_run_prints_the_documented_commands(tmp_path, capsys):
    code = cli.main(["setup-gptq", "--venv", str(tmp_path / "v"), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0 and not (tmp_path / "v").exists()
    assert "venv --python 3.13" in out and "https://download.pytorch.org/whl/cu130" in out
    assert "--extra-index-url https://pypi.org/simple" in out and "--index-strategy unsafe-best-match" in out
    assert str(Path("tools") / "gptq" / "requirements.txt") in out and "-r" in out


def test_setup_gptq_needs_uv_and_verifies_versions(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")]) == 1
    assert "uv is not installed" in capsys.readouterr().err
    monkeypatch.setattr(cli.shutil, "which", lambda name: "uv")
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kw: calls.append(command) or subprocess.CompletedProcess(command, 0))
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=FakeProbes()) == 0
    assert calls[0][:2] == ["uv", "venv"] and calls[1][:3] == ["uv", "pip", "install"] and "--python" in calls[1]
    assert "Ready." in capsys.readouterr().out
    wrong = FakeProbes(gptq_versions={"transformers": "99.0.0"})
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=wrong) == 1
    assert f"transformers 99.0.0 (pinned {PINS['transformers']})" in capsys.readouterr().err
    assert cli.main(["setup-gptq", "--venv", str(tmp_path / "v")], probes=FakeProbes(cuda=False)) == 1
    assert "CUDA" in capsys.readouterr().err


def test_requirements_pin_every_package_the_environment_check_compares():
    assert set(PINS) == set(checks.GPTQ_PINS)


# ---------------------------------------------------------------- serve-check


FAKE_SERVER_SCRIPT = '''
"""Fake infernix-serve for the serve-check tests: the OpenAI endpoints only, no GPU, no model."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model-id", required=True)
parser.add_argument("--any-model-name", action="store_true")
parser.add_argument("--max-context", type=int)
parser.add_argument("--break-check", action="append", default=[])
args, _ = parser.parse_known_args()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            data = [] if "models" in args.break_check else [
                {"id": args.model_id, "object": "model", "max_model_len": args.max_context}]
            self._send(200, {"object": "list", "data": data})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        except ValueError:
            body = {}
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": "not found"})
        elif "chat" in args.break_check:
            self._send(500, {"error": "broken on purpose"})
        elif "bearer" in args.break_check and self.headers.get("Authorization"):
            self._send(401, {"error": "unauthorized"})
        elif "model" not in body:
            self._send(200 if "missing" in args.break_check else 400, {"error": "model is required"})
        elif not args.any_model_name and body["model"] != args.model_id:
            self._send(404, {"error": {"message": "model '" + str(body["model"]) + "' not found",
                                       "code": "model_not_found"}})
        else:
            self._send(200, {"object": "chat.completion", "model": body["model"],
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                                          "finish_reason": "stop"}]})

    def log_message(self, *parts):
        pass


print("fake KV server on {}:{} model {} any {}".format(args.host, args.port, args.model_id, args.any_model_name),
      flush=True)
ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
'''


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def write_fake_launcher(tmp_path: Path, *, alias: str = "131k", match: str = "any", breaks: tuple = (),
                        host: str = "127.0.0.1", server: Path | None = None) -> Path:
    """A .cmd shaped like a generated launcher whose server is the fake script under this Python."""
    script = tmp_path / "fake_server.py"
    script.write_text(FAKE_SERVER_SCRIPT, encoding="utf-8")
    break_flags = [word for name in breaks for word in ("--break-check", name)]
    lines = [
        "@echo off",
        "rem fake launcher for the serve-check tests",
        f'"{server or sys.executable}" "{script}" ^',
        "  --host {} --port {} --model-id {}{} ^".format(host, free_port(), alias,
                                                          " --any-model-name" if match == "any" else ""),
        "  --max-context 131072 --kv-capacity auto --kv-dtype fp8 --gemma4-qk fp8 ^",
        "  --max-concurrency 1 --prefill-chunk auto --logits-dtype bf16 ^",
        "  --spec mtp --draft-tokens 1 ^",
        "  --device-snapshot-slots 0 --host-context-mib 8192 " + " ".join(break_flags),
        "exit /b %errorlevel%",
        "",
    ]
    cmd = tmp_path / "serve-m.cmd"
    cmd.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    return cmd


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def test_serve_check_help_prints_usage(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["serve-check", "--help"])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert text.startswith("usage:") and "--launcher" in text and "--serve" in text and "--out" in text


def test_serve_check_passes_every_check_and_writes_a_receipt(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path)
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert text.count("PASS") == 6 and "FAIL" not in text and f"Receipt: {out}" in text
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == build.SERVE_CHECK_SCHEMA
    assert [item["status"] for item in data["checks"]] == ["pass"] * 6
    assert data["checks"][0]["name"] == "ready" and data["checks"][0]["http_code"] == 200
    names = {item["name"]: item for item in data["checks"]}
    assert names["chat with another model name"]["http_code"] == 200
    assert names["chat without a model field"]["http_code"] == 400
    assert names["chat with an arbitrary bearer token"]["http_code"] == 200
    assert data["launcher"].endswith("serve-m.cmd") and data["url"].startswith("http://127.0.0.1:")
    assert data["artifact"]["size"] == (tmp_path / "fake_server.py").stat().st_size
    assert data["started"].endswith("Z") and data["finished"] >= data["started"]
    engine = data["engine"]
    assert (engine["model_id"], engine["model_match"], engine["context"], engine["kv_dtype"]) == (
        "131k", "any", 131072, "fp8")
    assert engine["kv_line"].startswith("fake KV server")
    assert engine["profile"]["gemma4_qk"] == "fp8" and engine["profile"]["host_context_mib"] == 8192
    assert engine["profile"]["device_snapshot_slots"] == 0 and engine["profile"]["draft_tokens"] == "1"


def test_serve_check_probes_loopback_when_the_launcher_binds_all_interfaces(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path, host="0.0.0.0")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    assert "FAIL" not in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["url"].startswith("http://127.0.0.1:")


def test_serve_check_exact_match_wants_404_for_another_model_name(tmp_path):
    cmd = write_fake_launcher(tmp_path, match="exact")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    result = {item["name"]: item for item in json.loads(out.read_text(encoding="utf-8"))["checks"]}
    assert result["chat with another model name"]["http_code"] == 404
    assert result["chat with another model name"]["status"] == "pass"


def test_serve_check_failure_exits_nonzero_and_names_the_check(tmp_path, capsys):
    cmd = write_fake_launcher(tmp_path, breaks=("chat",))
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 1
    captured = capsys.readouterr()
    assert "FAIL chat with the launcher model id" in captured.out
    assert "-> the server refuses its own --model-id" in captured.out
    assert "serve-check failed: chat with the launcher model id" in captured.err
    assert json.loads(out.read_text(encoding="utf-8"))["checks"][2]["status"] == "fail"


def test_serve_check_takes_the_server_from_serve_when_given(tmp_path):
    cmd = write_fake_launcher(tmp_path, server=tmp_path / "missing" / "infernix-serve.exe")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--serve", sys.executable, "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["server_exe"] == str(Path(sys.executable).resolve())
    assert cli.main(["serve-check", "--launcher", str(cmd), "--serve", str(tmp_path / "nope.exe")]) == 1


def test_serve_check_refuses_another_engines_launcher_and_a_missing_server(tmp_path, capsys, monkeypatch):
    other = write_fake_launcher(tmp_path, server=tmp_path / "ninfer-spo-serve.exe")
    assert cli.main(["serve-check", "--launcher", str(other)]) == 1
    assert "another engine's server" in capsys.readouterr().err
    monkeypatch.setattr(build.launcher_tools, "find_server_exe", lambda root, required=(): None)
    gone = write_fake_launcher(tmp_path, server=tmp_path / "missing" / "infernix-serve.exe")
    assert cli.main(["serve-check", "--launcher", str(gone)]) == 1
    assert "--serve" in capsys.readouterr().err


def test_serve_check_discovers_the_checkout_build_when_the_launcher_server_is_gone(tmp_path, monkeypatch):
    found = Path(sys.executable)
    monkeypatch.setattr(build.launcher_tools, "find_server_exe", lambda root, required=(): found)
    cmd = write_fake_launcher(tmp_path, server=tmp_path / "missing" / "infernix-serve.exe")
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["server_exe"] == str(found)


def test_serve_check_refuses_a_missing_launcher(tmp_path, capsys):
    assert cli.main(["serve-check", "--launcher", str(tmp_path / "nope.cmd")]) == 1
    assert "launcher not found" in capsys.readouterr().err


def test_serve_check_stops_the_child_process(tmp_path):
    cmd = write_fake_launcher(tmp_path)
    out = tmp_path / "receipt.json"
    assert cli.main(["serve-check", "--launcher", str(cmd), "--out", str(out)]) == 0
    pid = json.loads(out.read_text(encoding="utf-8"))["pid"]
    deadline = time.monotonic() + 10
    while process_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not process_alive(pid), f"child {pid} still runs after serve-check"


def test_memory_profile_reuses_iq3_xxs_builder_and_does_not_change_default(env):
    argv = env.argv("iq3-xxs", "--profile", "memory-16gb")
    index = argv.index("--preset")
    del argv[index:index + 2]
    cfg, profile, _ = cli.resolve_config(cli.build_parser().parse_args(argv))
    assert cfg.preset is build.PRESETS["iq3-xxs"]
    assert profile.context == "auto" and profile.prefill_chunk == "128"
    assert profile.gemma4_key_only_kv and profile.model_id == "gemma4-16gb"
    assert profile.vram_headroom_mib == 64 and profile.mtp_fallback
    default = cli.resolve_config(cli.build_parser().parse_args(env.argv("iq3-xxs")))[1]
    assert default.context == 131072 and default.prefill_chunk == "auto"
    assert not default.gemma4_key_only_kv and default.vram_headroom_mib == 1024 and not default.mtp_fallback


@pytest.mark.parametrize("extra, message", [
    (["--preset", "best-small"], "iq3-xxs only"),
    (["--with-vision"], "text-only"),
    (["--prefill-chunk", "auto"], "--prefill-chunk"),
    (["--prefill-chunk", "1024"], "--prefill-chunk"),
    (["--context", "8192"], "at least 9216"),
    (["--kv-dtype", "bf16"], "--kv-dtype fp8"),
    (["--draft-tokens", "3"], "--draft-tokens 1"),
])
def test_memory_profile_refuses_outside_qualified_settings(env, extra, message):
    args = cli.build_parser().parse_args(env.argv("iq3-xxs", "--profile", "memory-16gb", *extra))
    with pytest.raises(ValueError, match=message):
        cli.resolve_config(args)


def test_memory_profile_generated_launcher_roundtrip(tmp_path, monkeypatch):
    server, artifact = tmp_path / "s.exe", tmp_path / "iq3.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, True))
    out = tmp_path / "memory.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out),
                     "--profile", "memory-16gb", "--no-mtp"]) == 0
    plan = build.parse_launcher(out)
    assert plan.context == "auto" and plan.prefill_chunk == "128"
    assert plan.gemma4_key_only_kv and plan.effective_profile()["gemma4_key_only_kv"]
    assert "--gemma4-key-only-kv" in plan.flags and "--spec" not in plan.flags
    assert not plan.env
    wrong = tmp_path / "wrong.cmd"
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, False))
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(wrong),
                     "--profile", "memory-16gb"]) == 1
    assert not wrong.exists()


def test_memory_profile_launcher_plans_with_mtp_then_falls_back_without_it(tmp_path, monkeypatch):
    server, artifact = tmp_path / "s.exe", tmp_path / "iq3.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, True))
    out = tmp_path / "memory.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out),
                     "--profile", "memory-16gb"]) == 0
    text = out.read_text(encoding="utf-8")
    lines = text.splitlines()
    plan_at = next(i for i, line in enumerate(lines) if " --plan " in line)
    assert lines[plan_at].endswith(">nul 2>nul") and "--spec" in " ".join(lines[:plan_at + 1])
    assert "if errorlevel 1 goto :no_mtp" in lines and ":no_mtp" in lines
    after = " ".join(lines[lines.index(":no_mtp"):])
    assert "--spec" not in after and "--max-context auto" in after and "--vram-headroom-mib 64" in after
    parsed = build.parse_launcher(out)
    assert "--plan" not in parsed.flags and parsed.draft_tokens == "1" and not parsed.env
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(tmp_path / "off.cmd"),
                     "--profile", "memory-16gb", "--no-mtp"]) == 0
    assert ":no_mtp" not in (tmp_path / "off.cmd").read_text(encoding="utf-8")


def test_memory_profile_shared_memory_option_is_the_last_resort_stage(tmp_path, monkeypatch):
    server, artifact = tmp_path / "s.exe", tmp_path / "iq3.ninfer"
    server.write_bytes(b"x")
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, True))
    out = tmp_path / "spill.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out),
                     "--profile", "memory-16gb", "--allow-shared-memory"]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    shared_at = lines.index(":shared_memory")
    before, after = " ".join(lines[:shared_at]), " ".join(lines[shared_at:])
    assert "--allow-shared-memory" not in before and "--allow-shared-memory" in after
    assert "--spec" in before.split(":no_mtp")[0] and "--spec" not in after
    assert before.count(" --plan ") == 2 and "WARNING" in after
    parsed = build.parse_launcher(out)
    assert "--allow-shared-memory" not in parsed.flags and parsed.draft_tokens == "1"
    plain = tmp_path / "plain.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(plain),
                     "--profile", "memory-16gb"]) == 0
    assert "--allow-shared-memory" not in plain.read_text(encoding="utf-8")


@pytest.mark.parametrize("profile", ["default", "memory-16gb"])
@pytest.mark.parametrize("selection", [None, "bf16", "fp8"])
def test_sliding_ring_profile_selection_survives_all_stages(env, tmp_path, monkeypatch, profile, selection):
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, True))
    artifact = tmp_path / "iq3.ninfer"
    artifact.write_bytes(b"x")
    out = tmp_path / "ring.cmd"
    option = [] if selection is None else ["--gemma4-sliding-kv", selection]
    expected = selection or ("fp8" if profile == "memory-16gb" else "bf16")
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(env.server), "--out", str(out),
                     "--profile", profile, *option, "--allow-shared-memory"]) == 0
    parsed = build.parse_launcher(out)
    assert parsed.gemma4_sliding_kv == expected
    assert parsed.effective_profile().get("gemma4_sliding_kv", "bf16") == expected
    text = out.read_text(encoding="utf-8")
    for command in text.split(f'"{env.server.resolve()}" "{artifact.resolve()}"')[1:]:
        assert ("--gemma4-sliding-kv fp8" in command) == (expected == "fp8")
    args = cli.build_parser().parse_args(env.argv("iq3-xxs", "--profile", profile, *option))
    assert cli.resolve_config(args)[1].gemma4_sliding_kv == expected


def test_fp8_sliding_ring_launcher_refuses_bf16_globals(env, capsys):
    assert cli.main(env.argv("draft", "--dry-run", "--gemma4-qk", "bf16", "--kv-dtype", "bf16",
                             "--gemma4-sliding-kv", "fp8"), probes=FakeProbes()) == 1
    assert "--gemma4-sliding-kv fp8 needs --kv-dtype fp8" in capsys.readouterr().err
    with pytest.raises(ValueError, match="--gemma4-sliding-kv must be"):
        launcher.check_serve_settings(launcher.ServeProfile(gemma4_sliding_kv="int8"))


@pytest.mark.parametrize("profile", ["default", "memory-16gb"])
@pytest.mark.parametrize("selection", [None, "a16", "fp8", "int8-g32", "int8-g64"])
def test_q5_prefill_selection_is_opt_in_and_in_every_serve_command(env, tmp_path, monkeypatch, profile, selection):
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, True))
    artifact = tmp_path / "iq3.ninfer"
    artifact.write_bytes(b"x")
    out = tmp_path / "q5.cmd"
    option = [] if selection is None else ["--gemma4-q5-prefill", selection]
    expected = selection or "a16"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(env.server), "--out", str(out),
                     "--profile", profile, *option, "--allow-shared-memory"]) == 0
    parsed = build.parse_launcher(out)
    assert parsed.gemma4_q5_prefill == expected
    assert parsed.effective_profile().get("gemma4_q5_prefill", "a16") == expected
    assert ("gemma4_q5_prefill" in parsed.effective_profile()) == (expected != "a16")
    text = out.read_text(encoding="utf-8")
    commands = text.split(f'"{env.server.resolve()}" "{artifact.resolve()}"')[1:]
    assert commands
    for command in commands:
        assert ("--gemma4-q5-prefill " + expected in command) == (expected != "a16")
        assert (" --gemma4-q5-prefill" in command) == (expected != "a16")
    args = cli.build_parser().parse_args(env.argv("iq3-xxs", "--profile", profile, *option))
    assert cli.resolve_config(args)[1].gemma4_q5_prefill == expected


def test_q5_prefill_refuses_unknown_values(env):
    with pytest.raises(ValueError, match="--gemma4-q5-prefill must be a16, fp8, int8-g32 or int8-g64"):
        launcher.check_serve_settings(launcher.ServeProfile(gemma4_q5_prefill="int8"))
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(env.argv("iq3-xxs", "--gemma4-q5-prefill", "int8"))

def test_q5_preset_builds_own_hessians_and_one_a16_artifact(env):
    cfg = env.config("q5-g64", score=True)
    steps = plan_by_id(cfg)
    quant = steps["gptq"].argv
    assert quant[quant.index("--tree") + 1] == w(cfg, "text60")
    assert quant[quant.index("--bits") + 1] == "5"
    assert quant[quant.index("--group-size") + 1] == "64"
    assert quant[quant.index("--observer") + 1] == "imatrix_mse"
    assert steps["gptq-check"].argv[-2:] == ["--bits", "5"]
    convert = steps["convert"].argv
    assert convert[convert.index("--weights") + 1] == "q5_g64_a16"
    assert convert[convert.index("--mtp-assistant") + 1] == str(cfg.assistant)
    assert "--native-groups" not in convert
    assert steps["score-run"].env == {}
    assert "Q5 A16" in steps["score-run"].title


@pytest.mark.parametrize("profile, ring", [("q5-quality", "bf16"), ("q5-long-context", "fp8")])
def test_q5_profiles_recommend_int8_and_preserve_explicit_ring_choices(env, profile, ring):
    args = cli.build_parser().parse_args(env.argv("q5-g64", "--profile", profile))
    cfg, serve, _ = cli.resolve_config(args)
    assert cfg.preset.bits == 5 and serve.gemma4_sliding_kv == ring
    assert serve.gemma4_q5_prefill == "int8-g64" and not serve.gemma4_key_only_kv
    assert serve.context == "auto" and serve.prefill_chunk == "2048"
    assert serve.draft_tokens == "1" and not serve.mtp_fallback
    args = cli.build_parser().parse_args(env.argv(
        "q5-g64", "--profile", profile, "--gemma4-sliding-kv", "bf16",
        "--gemma4-q5-prefill", "a16", "--alias", "own", "--top-k", "64"))
    _, serve, _ = cli.resolve_config(args)
    assert serve.gemma4_sliding_kv == "bf16" and serve.gemma4_q5_prefill == "a16"
    assert serve.alias == "own" and serve.top_k == 64 and not serve.gemma4_key_only_kv


@pytest.mark.parametrize("profile", [None, "default", "q5-quality", "q5-long-context"])
@pytest.mark.parametrize("compute", [None, "a16", "fp8", "int8-g32", "int8-g64"])
def test_q5_defaults_and_overrides_agree_for_build_api_and_launcher(env, tmp_path, monkeypatch, profile, compute):
    options = ([] if profile is None else ["--profile", profile])
    options += [] if compute is None else ["--gemma4-q5-prefill", compute]
    _, settings, _ = cli.resolve_config(cli.build_parser().parse_args(env.argv("q5-g64", *options)))
    selected = profile or "q5-quality"
    direct = build.ServeSettings(profile=selected, gemma4_q5_prefill=compute, server_exe=env.server)
    assert settings == direct
    expected = compute or ("a16" if selected == "default" else "int8-g64")
    assert settings.gemma4_q5_prefill == expected
    assert settings.top_k == (None if selected == "default" else 64)
    assert settings.context == (131072 if selected == "default" else "auto")
    assert settings.prefill_chunk == ("auto" if selected == "default" else "2048")
    assert settings.vram_headroom_mib == 1024 and not settings.gemma4_key_only_kv
    artifact = tmp_path / "arbitrary-name.ninfer"
    artifact.write_bytes(b"x")
    monkeypatch.setattr(launcher, "artifact_info", lambda path: launcher.ArtifactInfo(True, False, 262144, q5=True))
    out = tmp_path / "serve-q5.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(env.server),
                     "--out", str(out), *options]) == 0
    plan = build.parse_launcher(out)
    assert plan.gemma4_q5_prefill == expected and plan.context == settings.context
    assert plan.gemma4_sliding_kv == settings.gemma4_sliding_kv
    assert list(plan.flags) == launcher.serve_argv(
        env.server, artifact, direct, mtp=True, vision=False)[2:]
    api_out = tmp_path / "api.cmd"
    launcher.write_launcher(api_out, artifact, env.server)
    assert build.parse_launcher(api_out).gemma4_q5_prefill == "int8-g64"


@pytest.mark.parametrize("profile", ["q5-quality", "q5-long-context"])
def test_q5_profile_explicit_fidelity_and_memory_choices_survive_generation(env, tmp_path, profile):
    args = cli.build_parser().parse_args(env.argv(
        "q5-g64", "--profile", profile, "--gemma4-qk", "bf16", "--gemma4-sliding-kv", "bf16",
        "--gemma4-q5-prefill", "a16", "--context", "16384", "--prefill-chunk", "1536",
        "--vram-headroom-mib", "512", "--top-k", "20", "--gemma4-key-only-kv",
        "--allow-shared-memory", "--alias", "coding"))
    _, settings, _ = cli.resolve_config(args)
    direct = build.ServeSettings(
        profile=profile, server_exe=env.server, gemma4_qk="bf16", gemma4_sliding_kv="bf16",
        gemma4_q5_prefill="a16", context=16384, prefill_chunk="1536", vram_headroom_mib=512,
        top_k=20, gemma4_key_only_kv=True, allow_shared_memory=True, alias="coding")
    assert settings == direct
    plan = lines_profile(launcher.launcher_lines("S", "A", settings, mtp=False, vision=False), tmp_path)
    assert plan.gemma4_qk == "bf16" and plan.gemma4_sliding_kv == "bf16"
    assert plan.gemma4_q5_prefill == "a16" and plan.context == 16384 and plan.alias == "coding"
    assert plan.prefill_chunk == "1536" and "--spec" not in plan.flags
    assert "--gemma4-key-only-kv" in plan.flags and "--allow-shared-memory" in plan.flags
    assert plan.flags[plan.flags.index("--top-k") + 1] == "20"
    assert plan.flags[plan.flags.index("--vram-headroom-mib") + 1] == "512"


@pytest.mark.parametrize("preset", ["best-small", "best-medium", "draft", "iq3-xs", "iq3-xxs"])
def test_non_q5_presets_keep_the_general_profile(env, preset):
    settings = cli.resolve_config(cli.build_parser().parse_args(env.argv(preset)))[1]
    assert settings == build.ServeSettings(server_exe=env.server)
    assert settings.gemma4_q5_prefill == "a16" and settings.gemma4_sliding_kv == "bf16"


@pytest.mark.parametrize("parameter, q5", [
    ("text/layers/0/mlp/down", True),
    ("mtp/layers/0/mlp/down", False),
    ("text/output_head", False),
])
@pytest.mark.parametrize("split", [False, True])
def test_launcher_detects_native_q5_trunk_from_directory_not_names_or_receipts(tmp_path, parameter, q5, split):
    from tools.artifact.schema import TensorSpec, plan_objects
    from tools.artifact.writer import ArtifactWriter

    artifact = tmp_path / "looks-like-iq3.ninfer"
    spec = TensorSpec("unrelated-object-name", (1, 128), "q5_g64_fp16", "row_split_k128_v1")
    binding = {"parts": [{"object": spec.id, "range": [0, 64]}, {"object": spec.id, "range": [64, 128]}]} if split else {
        "object": spec.id}
    components = {"text": {"config": {"architectures": ["Gemma4ForCausalLM"], "max_position_embeddings": 262144}}}
    if parameter.startswith("mtp/"):
        components["mtp"] = {"config": {}}
    with ArtifactWriter(artifact, [spec], components=components, bindings={parameter: binding}, uses=[]) as writer:
        writer.write_object(spec.id, bytes(plan_objects([spec])[0].bytes))
    info = launcher.artifact_info(artifact)
    assert info.q5 == q5
    server = tmp_path / "infernix-serve.exe"
    server.write_bytes(b"x")
    out = tmp_path / "detected.cmd"
    assert cli.main(["launcher", "--artifact", str(artifact), "--serve", str(server), "--out", str(out)]) == 0
    plan = build.parse_launcher(out)
    assert plan.gemma4_q5_prefill == ("int8-g64" if q5 else "a16")
    assert plan.context == ("auto" if q5 else 131072)


MIXED_RECIPE = ("--q5-gptq-source", "gptq-q5", "--q5-unit", "o", "--q5-unit", "down")


def mixed_convert(cfg: build.BuildConfig, *, mtp: bool = True) -> list[str]:
    q5 = [w(cfg, part) if part == "gptq-q5" else part for part in MIXED_RECIPE]
    return convert_expected(cfg, "--gptq-source", w(cfg, "gptq"), *q5, mtp=mtp)


def test_mixed_preset_is_best_medium_plus_q5_o_and_down_and_marked_experimental():
    mixed, medium = build.PRESETS["q4q5-mixed-a"], build.PRESETS["best-medium"]
    assert mixed.experimental and not medium.experimental
    assert (mixed.method, mixed.bits, mixed.group_size, mixed.native_groups) == ("gptq", 4, 32, False)
    assert mixed.q5_units == ("o", "down") and mixed.formats == ("q4_g32_fp16", "q5_g64_fp16")
    assert medium.formats == ("q4_g32_fp16",)
    # Q5 over Q4 costs 0.75 bit/weight: o 289,017,856 B + down 650,285,056 B over the 60 layers.
    assert abs(mixed.file_gb - (medium.file_gb + (289_017_856 + 650_285_056) / 1e9)) < 0.02
    assert medium.file_gb < mixed.file_gb < build.PRESETS["q5-g64"].file_gb
    assert medium.kl > mixed.kl > build.PRESETS["q5-g64"].kl


def test_mixed_preset_runs_two_gptq_passes_on_one_tree_and_converts_with_q5_units(env):
    cfg = env.config("q4q5-mixed-a", score=True)
    steps = plan_by_id(cfg)
    assert list(steps)[:9] == ["calibration-render", "calibration-tokens", "text-tree", "gptq", "gptq-check",
                               "gptq-q5", "gptq-q5-check", "convert", "receipt-check"]
    gptq_dir = build.REPO_ROOT / "tools" / "gptq"
    gpy = str(env.gptq_python)
    assert steps["gptq"].argv == [
        gpy, str(gptq_dir / "run_gptq.py"), "--tree", w(cfg, "text60"), "--calib", w(cfg, "calib.pt"),
        "--out", w(cfg, "gptq"), "--recipe", "gemma4_31b", "--observer", "imatrix_mse", "--group-size", "32",
        "--max-cpu-gb", "32", "--offload-dir", w(cfg, "offload"), "--peak-log", w(cfg, "gptq.peak.jsonl")]
    assert steps["gptq-q5"].argv == [
        gpy, str(gptq_dir / "run_gptq.py"), "--tree", w(cfg, "text60"), "--calib", w(cfg, "calib.pt"),
        "--out", w(cfg, "gptq-q5"), "--recipe", "gemma4_31b", "--observer", "imatrix_mse", "--group-size", "64",
        "--max-cpu-gb", "32", "--offload-dir", w(cfg, "offloadq5"), "--peak-log", w(cfg, "gptq.q5.peak.jsonl"),
        "--bits", "5"]
    assert steps["gptq-check"].argv == [
        gpy, str(gptq_dir / "check_checkpoint.py"), "--gptq", w(cfg, "gptq"), "--tree", w(cfg, "text60")]
    assert steps["gptq-q5-check"].argv == [
        gpy, str(gptq_dir / "check_checkpoint.py"), "--gptq", w(cfg, "gptq-q5"), "--tree", w(cfg, "text60"),
        "--bits", "5"]
    # The shared tree is only ever freed after the second pass has read it.
    assert steps["gptq-check"].prune_if_short == () and steps["gptq-q5-check"].prune_if_short == (cfg.work("text60"),)
    assert steps["gptq"].prune_always == (cfg.work("offload"),)
    assert steps["gptq-q5"].prune_always == (cfg.work("offloadq5"),)
    assert steps["convert"].argv == mixed_convert(cfg)
    assert "--native-groups" not in steps["convert"].argv and "q5_g64_a16" not in steps["convert"].argv
    assert steps["score-candidate"].argv[-2:] == ["--out", w(cfg, "score", "candidate.ninfer")]
    twin = steps["score-candidate"].argv
    assert twin[twin.index("--q5-gptq-source") + 1] == w(cfg, "gptq-q5")
    assert [twin[i + 1] for i, item in enumerate(twin) if item == "--q5-unit"] == ["o", "down"]
    assert steps["score-run"].env == {build.EXACT_ROUTE_ENV: "1"} and "Q5 cells A16" in steps["score-run"].title


def test_other_gptq_presets_plan_no_second_pass(env):
    for preset in ("best-small", "best-medium", "q5-g64"):
        steps = plan_by_id(env.config(preset))
        assert "gptq-q5" not in steps and "--q5-unit" not in steps["convert"].argv
        assert steps["gptq-check"].prune_if_short == (env.out / "work" / "text60",)


def test_mixed_preset_keeps_the_general_profile_and_refuses_q5_profiles(env):
    args = cli.build_parser().parse_args(env.argv("q4q5-mixed-a"))
    cfg, serve, _ = cli.resolve_config(args)
    assert cfg.preset.id == "q4q5-mixed-a" and serve == build.ServeSettings(server_exe=env.server)
    assert serve.gemma4_q5_prefill == "a16" and serve.context == 131072
    args = cli.build_parser().parse_args(env.argv("q4q5-mixed-a", "--gemma4-q5-prefill", "int8-g64"))
    assert cli.resolve_config(args)[1].gemma4_q5_prefill == "int8-g64"
    for profile in ("q5-quality", "q5-long-context"):
        args = cli.build_parser().parse_args(env.argv("q4q5-mixed-a", "--profile", profile))
        with pytest.raises(ValueError, match="builds --preset q5-g64 only"):
            cli.resolve_config(args)


def test_mixed_preset_needs_a_server_that_knows_both_tensor_formats(env):
    class Probes(FakeProbes):
        def scan_binary(self, path, needles):
            return {needle: needle == "q4_g32_fp16" for needle in needles}

    item = finding(preflight(env, env.config("q4q5-mixed-a"), Probes()), launcher.SERVER_NAME)
    assert item.level == "fail" and "q5_g64_fp16" in item.detail and "q4_g32_fp16" not in item.detail
    assert finding(preflight(env, env.config("best-medium"), Probes()), launcher.SERVER_NAME).level == "ok"
    ok = finding(preflight(env, env.config("q4q5-mixed-a")), launcher.SERVER_NAME)
    assert ok.level == "ok" and "q4_g32_fp16, q5_g64_fp16" in ok.detail


def test_mixed_preset_is_labelled_experimental_in_every_user_facing_text(env):
    table = cli.preset_table().splitlines()
    row = next(line for line in table if line.lstrip().startswith("q4q5-mixed-a"))
    assert "experimental, unverified on diode" in row and "19.34 GB" in row
    assert not any("experimental" in line for line in table if "q4q5-mixed-a" not in line)
    assert "unverified on diode variants" in cli.EPILOG and "EXPERIMENTAL" in cli.EPILOG
    cfg = env.config("q4q5-mixed-a")
    plan = "\n".join(build.describe_plan(cfg, build.build_plan(cfg)))
    assert "EXPERIMENTAL" in plan and "gptq-q5" in plan
    plain = env.config("best-medium")
    assert "EXPERIMENTAL" not in "\n".join(build.describe_plan(plain, build.build_plan(plain)))


def test_mixed_build_runs_both_passes_and_the_summary_says_experimental(env):
    cfg = env.config("q4q5-mixed-a")
    builder, runner, _ = make_builder(env, cfg)
    builder.run()
    assert runner.calls == ["calibration-render", "calibration-tokens", "text-tree", "gptq", "gptq-check", "gptq-q5",
                            "gptq-q5-check", "convert"]
    assert not cfg.work("offload").exists() and not cfg.work("offloadq5").exists()
    summary = (env.out / "BUILD-SUMMARY.md").read_text()
    assert "EXPERIMENTAL" in summary and "unverified on diode variants" in summary
    assert "GPTQ5 g64 Q5 (A16Only" in summary and "o, down" in summary and "mean KL 0.0458" in summary
    assert "default route with int8-g64 Q5 prefill" in summary
