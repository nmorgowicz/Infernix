"""One-command Gemma 4 31B `.ninfer` builder: presets, step plan, resumable runner, serve-check.

`python -m tools.convert.gemma4_build --preset best-small --model <BF16 dir> --assistant <dir> --out-dir <dir>`
chains the steps of the conversion guide (calibration text, GPTQ, `gemma4_q4 convert`) and writes the artifact, its
receipts, a Windows `infernix-serve` launcher and BUILD-SUMMARY.md. The converter and the GPTQ scripts keep doing
the work; this module only builds their command lines, runs them with logs, and records which step finished. The
GPTQ scripts run in their own interpreter (`tools/gptq/requirements.txt`), never in the converter's. Source weights
are only read.

Presets: best-small (GPTQ group 256, `q4_g256_fp16`), best-medium (GPTQ group 32, `q4_g32_fp16`),
q5-g64 (GPTQ5 group 64, A16Only), q4q5-mixed-a (best-medium with Q5 `o` and `down` in every layer, a supported option),
draft (uncalibrated absmax, group 32), and iq3-xs / iq3-xxs
(llama.cpp IQ3_XS / IQ3_XXS blocks quantized with the model's own
importance matrix and imported verbatim; they need a llama.cpp build and its source tree for `convert_hf_to_gguf.py`).
The two-level 6-bit scale format `q4_g32_s6_fp16` and calibrated round-to-nearest presets are not built here.

Calibration text. Without `--calibration-text` the builder downloads the pinned public English corpus the measured
quality numbers used (eaddario/imatrix-calibration, combined_en_large.parquet) and exports it byte for byte as the
author's text file: a UTF-8 BOM, CRLF line ends, and the CP437 round trip that Windows PowerShell applied to the
export (it affects the 18,228 non-ASCII characters of the 20 MB). The result is checked against its sha256.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from . import gemma4_build_launcher as launcher_tools

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_SCHEMA = "infernix.gemma4-build-state/v1"
BUILD_SCHEMA = "infernix.gemma4-build/v1"
CHAT_PREFIX_IDS = "2,105,2364,107,34230,506,2354,236761,106,107,105,4368,107,100,45518,107,101"
CALIBRATION_CHUNKS = 400
CALIBRATION_CONTEXT = 512
CALIBRATION_REPO = "eaddario/imatrix-calibration"
CALIBRATION_REVISION = "e87ed55dcba9d9c3a3e41539f3e728e981b1daa4"
CALIBRATION_FILE = "combined_en_large.parquet"
CALIBRATION_URL = (
    f"https://huggingface.co/datasets/{CALIBRATION_REPO}/resolve/{CALIBRATION_REVISION}/{CALIBRATION_FILE}"
)
CALIBRATION_PARQUET_SHA256 = "8a804421c074713aaa537a35da8bdb12e93b94b44344c09a9942d047c25d6e03"
CALIBRATION_TEXT_SHA256 = "82b7e7804f2486a34c23feef43c5356e542724102321cd53c52e2c4d0c316c8a"
GPTQ_RECIPE = "gemma4_31b"  # tools/gptq/recipes: sequential targets, ignore patterns and Hessian leaders
GPTQ_PYTHON_ENV = "INFERNIX_GPTQ_PYTHON"
LLAMA_CPP_ENV = "INFERNIX_LLAMA_CPP"  # llama.cpp binaries (llama-quantize, llama-imatrix)
LLAMA_CPP_SOURCE_ENV = "INFERNIX_LLAMA_CPP_SOURCE"  # llama.cpp source tree (convert_hf_to_gguf.py, gguf-py)
EXACT_ROUTE_ENV = "INFERNIX_TMP_Q4_EXACT"  # Gemma scope temporary switch: exact W4A8 prefill for scoring
Q5_FORMAT = "q5_g64_fp16"  # tensor format of the Q5 cells of a `q5_units` preset
Q5_GROUP_SIZE = 64
Q5_CHECKPOINT = "gptq-q5"  # work-dir name of the 5-bit GPTQ checkpoint a `q5_units` preset reads
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
MIXED_A_EVIDENCE = (
    "on fresh held-out roleplay text (36 forum threads, 72 windows, disjoint from the development windows and every "
    "calibration text; block-prefill KL at context 512 against each model's own Q8_0) the diode build on its default "
    "route scored a KL ratio to best-medium of 0.740 (95 % thread-cluster interval 0.722-0.759), lower in 36 of 36 "
    "threads; the plain model on A16 compute scored 0.713, and the earlier development windows gave 0.76 (A16) and 0.75 "
    "(default route) on the diode, 0.72 on the plain model. Costs against best-medium: +0.94 GB, decode about -11 %, "
    "prefill at 65k about -5 %, capacity 131,072 tokens with MTP off and 122,432 with MTP d1. Limits: context-512 block KL "
    "only (long-context and incremental-decode KL are not measured for this allocation; the Q5 INT8-g64 prefill route has "
    "a measured long-context cost, see llama-local-tooling docs/plans/20261124-gemma4-long-context-roleplay.md), a single GPTQ pass per width, one GPU, roleplay text "
    "from three forums that the finetune may have seen, a keyword-only content screen. Measure your own build before relying on it."
)


@dataclass(frozen=True, slots=True)
class Preset:
    """A conversion recipe with model/variant-specific historical quality hints, not a cross-preset ranking."""

    id: str
    method: str  # "gptq", "plain" (uncalibrated absmax), or "gguf" (own-imatrix IQ3)
    summary: str
    meaning: str
    size_class: str
    file_gb: float  # text + MTP
    kl: float  # historical English hints: diode / FP8-row reference for q4; plain / own-Q8_0 for IQ3
    wait: str
    group_size: int
    native_groups: bool
    format_name: str  # tensor format the serving binary must know
    work_hard_gb: float  # free space the work drive needs at the peak
    work_soft_gb: float  # comfortable free space (adds estimated scratch)
    residual_gb: float  # what stays in the work dir once big intermediates are pruned
    minutes: dict[str, float]  # per step, typical RTX 5090 + NVMe
    gguf_type: str = ""  # method "gguf": the llama-quantize type (IQ3_XS, IQ3_XXS)
    bits: int = 4
    q5_units: tuple[str, ...] = ()  # `--q5-unit` selectors: fused groups read from a second, 5-bit GPTQ checkpoint
    evidence: str = ""  # a supported non-default option: the measured evidence and its limits, shown in the plan and summary

    @property
    def formats(self) -> tuple[str, ...]:
        """Tensor formats the serving binary must know."""
        return (self.format_name, Q5_FORMAT) if self.q5_units else (self.format_name,)


PRESETS: dict[str, Preset] = {
    "best-small": Preset(
        "best-small", "gptq", "GPTQ, group 256", "Best quality per byte: the recommended default for most people.",
        "small", 16.85, 0.1213, "about 30 min", 256, True, "q4_g256_fp16", 75, 103, 17,
        {"calibration-source": 1, "calibration-render": 1, "calibration-tokens": 1, "text-tree": 2, "gptq": 18,
         "gptq-check": 1, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
    ),
    "best-medium": Preset(
        "best-medium", "gptq", "GPTQ, group 32", "Higher-quality Q4; about 1.6 GB larger than best-small.",
        "medium", 18.41, 0.0845, "about 30 min", 32, False, "q4_g32_fp16", 75, 103, 17,
        {"calibration-source": 1, "calibration-render": 1, "calibration-tokens": 1, "text-tree": 2, "gptq": 18,
         "gptq-check": 1, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
    ),
    "q5-g64": Preset(
        "q5-g64", "gptq", "GPTQ 5-bit, group 64, A16Only",
        "Higher-quality Q5 tier; prefill compute (a16, fp8, int8-g32, int8-g64) is chosen at launch on one artifact. "
        "Historical KL is plain-model A16 versus own Q8_0, not a diode qualification.",
        "quality", 21.15, 0.0199, "about 40 min", 64, False, "q5_g64_fp16", 85, 115, 22,
        {"calibration-source": 1, "calibration-render": 1, "calibration-tokens": 1, "text-tree": 2, "gptq": 25,
         "gptq-check": 1, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        bits=5,
    ),
    "q4q5-mixed-a": Preset(
        "q4q5-mixed-a", "gptq", "GPTQ g32 Q4 with GPTQ5 g64 Q5 `o` and `down` in all layers (supported option)",
        "Best-medium with the attention output and MLP down projections of every layer at Q5, about 0.94 GB "
        "larger; both GPTQ checkpoints use your own model's Hessians. A supported option, not the default: on fresh "
        "held-out roleplay text its KL is 0.74x that of best-medium (diode, default route; context-512 block KL).",
        "medium+", 19.34, 0.0458, "about 70 min", 32, False, "q4_g32_fp16", 102, 132, 39,
        {"calibration-source": 1, "calibration-render": 1, "calibration-tokens": 1, "text-tree": 2, "gptq": 18,
         "gptq-q5": 25, "gptq-check": 1, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        q5_units=("o", "down"), evidence=MIXED_A_EVIDENCE,
    ),
    "draft": Preset(
        "draft", "plain", "uncalibrated absmax, group 32 (the old default)",
        "A quick test build only: nothing to install, but lower quality than the calibrated q4 presets.",
        "medium", 18.41, 0.2654, "about 15 min", 32, False, "q4_g32_fp16", 0, 0, 0,
        {"convert": 15, "receipt-check": 0.5, "finalize": 0.5},
    ),
    "iq3-xs": Preset(
        "iq3-xs", "gguf", "llama.cpp IQ3_XS, own imatrix",
        "About 3 GB smaller than best-small: IQ3_XS blocks with 8-bit activations; check runtime VRAM capacity.",
        "smaller", 13.91, 0.1762, "about 2 h", 0, False, "ggml_iq3_s", 100, 120, 14,
        {"calibration-source": 1, "calibration-render": 1, "gguf-bf16": 15, "gguf-q8": 8, "imatrix": 25,
         "gguf-quantize": 45, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        "IQ3_XS",
    ),
    "iq3-xxs": Preset(
        "iq3-xxs", "gguf", "llama.cpp IQ3_XXS, own imatrix",
        "The smallest, about 1 GB below iq3-xs, at a visible quality cost.",
        "smallest", 13.03, 0.2273, "about 2 h", 0, False, "ggml_iq3_xxs", 100, 120, 13,
        {"calibration-source": 1, "calibration-render": 1, "gguf-bf16": 15, "gguf-q8": 8, "imatrix": 25,
         "gguf-quantize": 45, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        "IQ3_XXS",
    ),
}
# Presets that are not built by infernix, with the reason a caller gets.
UNAVAILABLE_PRESETS = {
    "lite-medium": "it needs the q4_g32_s6_fp16 format (two-level 6-bit scales), which infernix has not ported yet; "
                   "use best-small or best-medium",
    "lite-small": "calibrated round-to-nearest from llama.cpp statistics is not part of the infernix builder; "
                  "use best-small, or iq3-xs for a smaller file",
}
SCORE_MINUTES = {
    "score-corpus": 8, "score-reference": 41, "score-reference-write": 5, "score-candidate": 8,
    "score-run": 2, "score-report": 0.5,
}
SCORE_EXTRA_GB = 62  # reference artifact 31 + corpus cache 14 + candidate 17


@dataclass(frozen=True)
class BuildConfig:
    """Everything that decides the artifact; launcher settings live in `ServeSettings`."""

    preset: Preset
    model: Path
    assistant: Path | None
    with_vision: bool
    out_dir: Path
    work_dir: Path
    name: str
    python: str = sys.executable
    gptq_python: Path | None = None
    gptq_max_cpu_gb: int = 32
    calibration_text: Path | None = None
    score: bool = False
    perplexity_exe: Path | None = None
    keep_work: bool = False
    llama_cpp: Path | None = None  # method "gguf": directory with llama-quantize and llama-imatrix
    llama_cpp_source: Path | None = None  # method "gguf": llama.cpp source tree (convert_hf_to_gguf.py)

    @property
    def artifact(self) -> Path:
        return self.out_dir / f"{self.name}.ninfer"

    @property
    def receipts(self) -> tuple[Path, ...]:
        base = str(self.artifact)
        return (self.artifact, Path(base + ".build.json"), Path(base + ".conversion.json"))

    def work(self, *parts: str) -> Path:
        return self.work_dir.joinpath(*parts)


@dataclass(frozen=True, slots=True)
class ServeSettings(launcher_tools.ServeProfile):
    """The launcher's `ServeProfile` plus where the server binary is and whether to write a launcher at all."""

    server_exe: Path | None = None
    cuda_bin: Path | None = None
    launcher: bool = True


@dataclass
class Step:
    id: str
    title: str
    minutes: float
    argv: list[str] | None = None
    call: Callable[["Builder"], None] | None = None
    note: str = ""
    env: dict[str, str] = field(default_factory=dict)
    outputs: tuple[Path, ...] = ()
    external: bool = False  # outputs live outside the work dir (only ever written, never overwritten)
    prune_always: tuple[Path, ...] = ()
    prune_if_short: tuple[Path, ...] = ()


def safe_name(text: str) -> str:
    return SAFE_NAME.sub("-", text).strip("-.") or "gemma4-31b"


def default_name(model: Path, preset: str) -> str:
    return safe_name(f"{model.resolve().name}-{preset}")


def format_command(argv: list[str]) -> str:
    return subprocess.list2cmdline(argv) if os.name == "nt" else shlex.join(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def gb(value: float) -> str:
    return f"{value:g} GB"


def default_gptq_python() -> Path:
    """The GPTQ environment's interpreter: `$INFERNIX_GPTQ_PYTHON`, else `<checkout>/.venv-gptq`."""
    given = os.environ.get(GPTQ_PYTHON_ENV)
    if given:
        return Path(given)
    return REPO_ROOT / ".venv-gptq" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


# ---------------------------------------------------------------- plan


def q4_convert_flags(cfg: BuildConfig, checkpoint: Path | None) -> list[str]:
    """Storage recipe; Q5 is A16Only, independently selectable at serving time. `q5_units` presets keep the Q4 g32
    import and move the named fused groups to the second checkpoint."""
    preset = cfg.preset
    flags = ["--weights", "q5_g64_a16" if preset.bits == 5 else "q4_g32"]
    if preset.method == "gptq":
        flags += ["--gptq-source", str(checkpoint)]
        if preset.native_groups:
            flags.append("--native-groups")
        if preset.q5_units:
            flags += ["--q5-gptq-source", str(cfg.work(Q5_CHECKPOINT))]
            for unit in preset.q5_units:
                flags += ["--q5-unit", unit]
    elif preset.method == "gguf":
        flags += ["--gguf-source", str(checkpoint)]
    return flags


def gptq_steps(cfg: BuildConfig, tree: Path, calib: Path, checkpoint: Path, *, bits: int, group_size: int,
               suffix: str, free_tree: bool) -> list[Step]:
    """GPTQ and its importability check; `suffix` names the second (Q5) pass of a mixed preset. Only the last pass
    may free the shared text tree."""
    gpy = str(cfg.gptq_python)
    gptq_dir = REPO_ROOT / "tools" / "gptq"
    offload = cfg.work(f"offload{suffix.replace('-', '')}")
    peak = cfg.work(f"gptq{suffix.replace('-', '.')}.peak.jsonl" if suffix else "gptq.peak.jsonl")
    bit_flags = ["--bits", "5"] if bits == 5 else []
    return [
        Step(f"gptq{suffix}", f"GPTQ, {bits}-bit, group {group_size}, imatrix_mse observer (the long step)",
             cfg.preset.minutes[f"gptq{suffix}"],
             [gpy, str(gptq_dir / "run_gptq.py"), "--tree", str(tree), "--calib", str(calib), "--out",
              str(checkpoint), "--recipe", GPTQ_RECIPE, "--observer", "imatrix_mse", "--group-size",
              str(group_size), "--max-cpu-gb", str(cfg.gptq_max_cpu_gb), "--offload-dir", str(offload), "--peak-log",
              str(peak)] + bit_flags,
             outputs=(checkpoint, offload, peak), prune_always=(offload,)),
        Step(f"gptq{suffix}-check", f"Check the {bits}-bit GPTQ checkpoint is importable (exact binary16 scales)",
             cfg.preset.minutes["gptq-check"],
             [gpy, str(gptq_dir / "check_checkpoint.py"), "--gptq", str(checkpoint), "--tree", str(tree)] + bit_flags,
             prune_if_short=(tree,) if free_tree else ()),
    ]


# llama-quantize --tensor-type pins (regex searched in the tensor name) of the measured IQ3 recipes: the per-tensor
# types a published imatrix IQ3_XS / IQ3_XXS of Gemma 4 31B uses where they differ from llama.cpp's own mix.
GGUF_TENSOR_TYPES = {
    "IQ3_XS": ("attn_output=IQ3_S", "attn_v=IQ3_S", "ffn_down=IQ3_S", r"blk\.([0-6]|5[2-9])\.ffn_(gate|up)=IQ3_S"),
    "IQ3_XXS": ("attn_q=IQ2_S", "attn_k=IQ2_S", "attn_output=IQ3_S"),
}


def llama_tool(cfg: BuildConfig, name: str) -> str:
    return str(Path(cfg.llama_cpp or ".") / (f"{name}.exe" if os.name == "nt" else name))


def gguf_steps(cfg: BuildConfig, templated: Path, quantized: Path) -> list[Step]:
    """BF16 GGUF, Q8_0 copy for the importance matrix, own imatrix on the rendered calibration text, quantize."""
    preset = cfg.preset
    minutes = preset.minutes
    bf16, q8, imatrix = cfg.work("model-bf16.gguf"), cfg.work("model-q8_0.gguf"), cfg.work("imatrix.gguf")
    quantize = llama_tool(cfg, "llama-quantize")
    pins = [arg for pin in GGUF_TENSOR_TYPES[preset.gguf_type] for arg in ("--tensor-type", pin)]
    return [
        Step("gguf-bf16", "Write a BF16 GGUF of the model with llama.cpp's convert_hf_to_gguf.py (about 62 GB)",
             minutes["gguf-bf16"],
             [cfg.python, str(Path(cfg.llama_cpp_source or ".") / "convert_hf_to_gguf.py"), str(cfg.model),
              "--outtype", "bf16", "--outfile", str(bf16)],
             outputs=(bf16,)),
        Step("gguf-q8", "Quantize a Q8_0 copy for the importance matrix (about 33 GB)", minutes["gguf-q8"],
             [quantize, str(bf16), str(q8), "Q8_0"], outputs=(q8,)),
        Step("imatrix", f"Collect the importance matrix on the {CALIBRATION_CHUNKS} rendered chunks (GPU)",
             minutes["imatrix"],
             [llama_tool(cfg, "llama-imatrix"), "-m", str(q8), "-o", str(imatrix), "--output-format", "gguf",
              "--no-ppl", "--ctx-size", str(CALIBRATION_CONTEXT), "--gpu-layers", "999", "--file", str(templated),
              "--chunks", str(CALIBRATION_CHUNKS), "--parse-special"],
             outputs=(imatrix,), prune_always=(q8,)),
        Step("gguf-quantize", f"Quantize to {preset.gguf_type} with the importance matrix (CPU, the long step)",
             minutes["gguf-quantize"],
             [quantize, "--imatrix", str(imatrix), *pins, str(bf16), str(quantized), preset.gguf_type],
             outputs=(quantized,), prune_always=(bf16,)),
    ]


def convert_argv(cfg: BuildConfig, out: Path, *, scoring: bool, checkpoint: Path | None) -> list[str]:
    """`gemma4_q4 convert` command; the scoring twin is always text-only and without MTP."""
    argv = [cfg.python, "-m", "tools.convert.gemma4_q4", "convert", "--model", str(cfg.model)]
    argv += q4_convert_flags(cfg, checkpoint)
    if scoring or not cfg.with_vision:
        argv.append("--text-only")
    if not scoring and cfg.assistant is not None:
        argv += ["--mtp-assistant", str(cfg.assistant)]
    else:
        argv.append("--no-mtp")
    argv += ["--name", cfg.name, "--out", str(out)]
    return argv


def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered steps with their exact commands. Pure: touches no file."""
    preset = cfg.preset
    minutes = preset.minutes
    templated = cfg.work("calibration", "templated.txt")
    source = cfg.calibration_text or cfg.work("calibration", "source.txt")
    tokenizer = cfg.model / "tokenizer.json"
    checkpoint = cfg.work(f"model-{preset.gguf_type}.gguf") if preset.method == "gguf" else cfg.work("gptq")
    tree = cfg.work("text60")
    steps: list[Step] = []

    if preset.method != "plain":
        if cfg.calibration_text is None:
            steps.append(Step(
                "calibration-source", f"Download the pinned public English calibration text ({CALIBRATION_REPO})",
                minutes["calibration-source"], call=fetch_calibration, outputs=(source,),
                note=f"{CALIBRATION_URL} (sha256 {CALIBRATION_PARQUET_SHA256[:12]}...) exported to {source}",
            ))
        steps.append(Step(
            "calibration-render", f"Render {CALIBRATION_CHUNKS} chat-prefixed chunks of {CALIBRATION_CONTEXT} tokens",
            minutes["calibration-render"],
            [cfg.python, "-m", "tools.convert.gemma4_q4", "calibration-text", "--tokenizer", str(tokenizer),
             "--source", str(source), "--prefix-ids", CHAT_PREFIX_IDS, "--chunks", str(CALIBRATION_CHUNKS),
             "--ctx-size", str(CALIBRATION_CONTEXT), "--out", str(templated)],
            outputs=(templated, templated.with_suffix(".json")),
        ))
    if preset.method == "gptq":
        calib = cfg.work("calib.pt")
        gptq_dir = REPO_ROOT / "tools" / "gptq"
        steps += [
            Step("calibration-tokens", "Tokenize the calibration text for GPTQ", minutes["calibration-tokens"],
                 [str(cfg.gptq_python), str(gptq_dir / "make_calib.py"), "--text", str(templated), "--tokenizer",
                  str(tokenizer), "--out", str(calib), "--context", str(CALIBRATION_CONTEXT), "--prefix-ids",
                  CHAT_PREFIX_IDS],
                 outputs=(calib,)),
            Step("text-tree", "Write the text-only 60-layer tree GPTQ loads (about 58 GB)", minutes["text-tree"],
                 [str(cfg.gptq_python), str(gptq_dir / "make_text_tree.py"), "--model", str(cfg.model), "--out",
                  str(tree)],
                 outputs=(tree,)),
        ]
        steps += gptq_steps(cfg, tree, calib, checkpoint, bits=preset.bits, group_size=preset.group_size, suffix="",
                            free_tree=not preset.q5_units)
        if preset.q5_units:
            steps += gptq_steps(cfg, tree, calib, cfg.work(Q5_CHECKPOINT), bits=5, group_size=Q5_GROUP_SIZE,
                                suffix="-q5", free_tree=True)
    if preset.method == "gguf":
        steps += gguf_steps(cfg, templated, checkpoint)
    steps.append(Step(
        "convert", f"Convert to .ninfer ({preset.id}: {preset.summary}; the converter verifies storage itself)",
        minutes["convert"], convert_argv(cfg, cfg.artifact, scoring=False, checkpoint=checkpoint),
        outputs=cfg.receipts, external=True,
    ))
    steps.append(Step("receipt-check", "Check the converter's receipt (status, size, sha256)", minutes["receipt-check"],
                      call=check_receipt))
    steps.append(Step("finalize", "Write the launcher, BUILD-SUMMARY.md and the build receipt", minutes["finalize"],
                      call=finalize, outputs=finalize_outputs(cfg), external=True))
    if cfg.score:
        steps += score_steps(cfg, checkpoint)
    return steps


def finalize_outputs(cfg: BuildConfig) -> tuple[Path, ...]:
    return (
        cfg.out_dir / "BUILD-SUMMARY.md",
        cfg.out_dir / f"{cfg.name}.gemma4-build.json",
        cfg.out_dir / f"serve-{cfg.name}.cmd",
    )


def score_steps(cfg: BuildConfig, checkpoint: Path) -> list[Step]:
    """FP8-row reference and no-MTP text-only twin, scored with native full-vocabulary KL.
    Q4 uses its exact W4A8 scoring route; IQ3 honors its stored AllowA8 Uses at every width."""
    root = cfg.work("score")
    corpus, reference, nklref = root / "corpus", root / "ref-fp8row.ninfer", root / "ref.nklref"
    candidate, run, reference_run = root / "candidate.ninfer", root / "run", root / "ref-run"
    manifest = corpus / "manifest.json"
    exe = str(cfg.perplexity_exe)
    common = ["--corpus", str(manifest), "--context", "512", "--stride", "256", "--prefill-chunk", "128",
              "--kv-dtype", "fp8", "--logits-dtype", "bf16", "--full-vocab-prefix-ids", CHAT_PREFIX_IDS]
    minutes = SCORE_MINUTES
    return [
        Step("score-corpus", "Build the English KL corpus from the bundled corpus (about 14 GB of cache)",
             minutes["score-corpus"],
             [cfg.python, "-m", "tools.bench.kv_eval.kl_corpus", "--source",
              str(REPO_ROOT / "eval" / "corpora" / "perplexity-1m" / "manifest.json"), "--out", str(corpus),
              "--chars", "chinese_reference=0"], outputs=(corpus,)),
        Step("score-reference", "Convert the near-lossless FP8-row reference (31 GB; the slow scoring step)",
             minutes["score-reference"],
             [cfg.python, "-m", "tools.convert.gemma4_q4", "convert", "--model", str(cfg.model), "--text-only",
              "--no-mtp", "--weights", "fp8_row", "--out", str(reference)],
             outputs=(reference, Path(str(reference) + ".build.json"), Path(str(reference) + ".conversion.json"))),
        Step("score-reference-write", "Score the reference once and keep its logits for pairing",
             minutes["score-reference-write"],
             [exe, str(reference)] + common + ["--output", str(reference_run), "--full-vocab-reference-write",
                                               str(nklref)],
             outputs=(nklref, reference_run), prune_if_short=(reference,)),
        Step("score-candidate", "Convert a no-MTP text-only twin of your recipe for scoring", minutes["score-candidate"],
             convert_argv(cfg, candidate, scoring=True, checkpoint=checkpoint),
             outputs=(candidate, Path(str(candidate) + ".build.json"), Path(str(candidate) + ".conversion.json"))),
        Step("score-run", "Paired KL against the reference"
             + (f" (W4A8 exact route, {EXACT_ROUTE_ENV}=1{'; Q5 cells A16' if cfg.preset.q5_units else ''})"
                if cfg.preset.bits == 4 and cfg.preset.method != "gguf"
                else " (Q5 A16)" if cfg.preset.bits == 5 else " (GGUF AllowA8)"),
             minutes["score-run"],
             [exe, str(candidate)] + common + ["--full-vocab-reference", str(nklref), "--output", str(run)],
             env={EXACT_ROUTE_ENV: "1"} if cfg.preset.bits == 4 and cfg.preset.method != "gguf" else {}, outputs=(run,)),
        Step("score-report", "Record the score in the build receipt and BUILD-SUMMARY.md", minutes["score-report"],
             call=score_report),
    ]


def plan_minutes(steps: list[Step]) -> float:
    return sum(step.minutes for step in steps)


def describe_plan(cfg: BuildConfig, steps: list[Step]) -> list[str]:
    """Human-readable plan: one block per step with its time estimate and exact command."""
    preset = cfg.preset
    lines = [
        f"Preset      {preset.id}: {preset.summary} ({preset.size_class}; {gb(preset.file_gb)} with MTP; "
        f"KL {preset.kl:.4f}, lower is better)"
        + ("; supported option, not the default: roleplay KL 0.74x best-medium on fresh held-out text" if preset.evidence else ""),
        f"            {preset.meaning}",
        f"Model       {cfg.model}",
        f"MTP         {cfg.assistant if cfg.assistant else 'none (--no-mtp)'}",
        f"Vision      {'kept (--with-vision)' if cfg.with_vision else 'left out (text only)'}",
        f"Output      {cfg.artifact}",
        f"Work dir    {cfg.work_dir}",
        f"Estimate    about {plan_minutes(steps):.0f} min in total (typical RTX 5090 + NVMe; your times vary)",
        "",
    ]
    for index, step in enumerate(steps, 1):
        lines.append(f"{index:>2}. {step.id:<22} ~{step.minutes:g} min  {step.title}")
        if step.argv:
            lines.append(("    " + (" ".join(f"{k}={v}" for k, v in step.env.items()) + " " if step.env else ""))
                         + format_command(step.argv))
        elif step.note:
            lines.append(f"    {step.note}")
    return lines


# ---------------------------------------------------------------- calibration text


def export_author_text(parquet: Path, out: Path, *, expected_sha256: str | None = CALIBRATION_TEXT_SHA256) -> str:
    """Parquet `content` column -> the author's calibration text file; returns its sha256."""
    try:
        import pyarrow.parquet as parquet_reader
    except ImportError as error:
        raise RuntimeError(
            f"{sys.executable} needs pyarrow to read the calibration corpus: {sys.executable} -m pip install pyarrow "
            "(or pass --calibration-text FILE)"
        ) from error
    rows = parquet_reader.read_table(parquet, columns=["content"]).column("content").to_pylist()
    text = "\n".join(rows).encode("utf-8").decode("cp437")
    data = ("\ufeff" + text.replace("\r\n", "\n").replace("\n", "\r\n") + "\r\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(
            f"exported calibration text has sha256 {digest}, expected {expected_sha256}; pass --calibration-text FILE "
            "to use your own text"
        )
    part = out.with_name(out.name + ".part")
    part.write_bytes(data)
    os.replace(part, out)
    return digest


def download_pinned(url: str, out: Path, sha256: str, log: Callable[[str], None]) -> None:
    """Download `url` to `out`, refusing any size or hash surprise."""
    part = out.with_name(out.name + ".part")
    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "infernix-gemma4-build"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as stream:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                stream.write(chunk)
    except OSError as error:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"download of {url} failed: {error}. Fix the connection and rerun (the build resumes), or pass "
            "--calibration-text FILE with public English text"
        ) from error
    if digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"{url} has sha256 {digest.hexdigest()}, expected {sha256}; refusing the download")
    os.replace(part, out)
    log(f"downloaded {out.name}: {out.stat().st_size} bytes, sha256 {sha256}")


def fetch_calibration(builder: "Builder") -> None:
    cfg = builder.cfg
    folder = cfg.work("calibration")
    folder.mkdir(parents=True, exist_ok=True)
    parquet = folder / CALIBRATION_FILE
    if not parquet.exists():
        builder.log(f"downloading {CALIBRATION_URL}")
        download_pinned(CALIBRATION_URL, parquet, CALIBRATION_PARQUET_SHA256, builder.log)
    digest = export_author_text(parquet, folder / "source.txt")
    builder.log(f"exported calibration text, sha256 {digest}")
    parquet.unlink(missing_ok=True)


# ---------------------------------------------------------------- receipts


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


def check_receipt(builder: "Builder") -> None:
    """The converter verifies storage before it returns; confirm its receipt says so for the file on disk."""
    cfg = builder.cfg
    receipt_path = Path(str(cfg.artifact) + ".build.json")
    if not receipt_path.is_file():
        raise RuntimeError(f"converter receipt missing: {receipt_path}")
    receipt = read_json(receipt_path)
    status = str(receipt.get("status", ""))
    if not status.startswith("storage-verified"):
        raise RuntimeError(f"converter receipt status is {status!r}: {receipt.get('error', 'no error recorded')}")
    size = cfg.artifact.stat().st_size
    if receipt["artifact"]["file_bytes"] != size:
        raise RuntimeError(f"{cfg.artifact} is {size} bytes, its receipt says {receipt['artifact']['file_bytes']}")
    if not receipt["artifact"].get("file_sha256", {}).get(cfg.artifact.name):
        raise RuntimeError("converter receipt has no sha256 for the artifact")
    builder.log(f"receipt ok: {status}; {size} bytes; revision {receipt.get('revision')}")


def render_summary(info: dict) -> str:
    """BUILD-SUMMARY.md text from the facts `build_info` collected."""
    p = info["preset"]
    serve = info["serve"]
    steps = "\n".join(f"| {s['id']} | {s['status']} | {s['seconds']:.0f} |" for s in info["steps"])
    parts = [
        f"# {info['name']}",
        "",
        f"Built by `python -m tools.convert.gemma4_build` (preset `{p['id']}`) for `infernix-serve`.",
        "",
        "## What you built",
        "",
        f"- Preset: **{p['id']}**: {p['summary']} ({p['size_class']} file).",
        f"- In plain words: {p['meaning']}",
        ("- Weights: IQ3 GGUF blocks, AllowA8 for prefill and decode; "
         if p["method"] == "gguf" else "- Weights: native GPTQ5 group 64, A16Only; "
         if p["bits"] == 5 else "- Weights: GPTQ g32 Q4 (W4A8, FP8 activations while reading the prompt) with "
         f"GPTQ5 g64 Q5 (A16Only, choose the Q5 prefill route at launch) for {', '.join(p['q5_units'])} in every layer; "
         if p.get("q5_units") else "- Weights: W4A8 (4-bit weights, FP8 activations while reading the prompt); ")
        + f"{'with the MTP draft assistant (speculative decoding)' if info['mtp'] else 'no MTP assistant'}; "
        f"{'with the vision tower' if info['vision'] else 'text only'}.",
        f"- Historical hint (English, text only, RTX 5090): mean KL {p['kl']:.4f} from "
        + ("the plain model's own Q8_0 reference, A16 compute" if p["bits"] == 5 else
           "the plain model's own Q8_0 reference, default route with int8-g64 Q5 prefill" if p.get("q5_units") else
           "the model's own Q8_0 reference" if p["method"] == "gguf" else "the near-lossless FP8-row reference")
        + ". This is not a quality measurement of your build.",
    ] + ([
        f"- Supported option, not the default (best-medium stays the general default): {p['evidence']}",
    ] if p.get("evidence") else []) + [
        "",
        "## Files",
        "",
        f"- Artifact: `{info['artifact']}`",
        f"- Size: {info['file_bytes']:,} bytes ({info['file_bytes'] / 1e9:.2f} GB)",
        f"- sha256: `{info['sha256']}`",
        f"- Converter receipt: `{info['artifact']}.build.json`",
        f"- Launcher: `{info['launcher']}`" if info["launcher"] else "- Launcher: not generated (--no-launcher)",
        "",
        "## Serve it",
        "",
        f"The launcher runs `infernix-serve` with {serve['kv'].upper()} KV cache and {serve['qk'].upper()} attention "
        "QK, one lane, a host-only prefix cache and BF16-rounded logits:",
        "",
        "```bat",
        info["serve_command"] if info["serve_command"] else "REM no launcher; see docs/serving.md",
        "```",
        "",
        "Check the real KV capacity in the server's startup log: the model list can overstate an auto-sized pool. "
        f"`/v1/models` reports `{info['alias']}`; "
        + (
            "OpenAI-style requests may send any model name (`--any-model-name`). "
            if "--any-model-name" in (info["serve_command"] or "")
            else "OpenAI-style requests must send that exact model name. "
        )
        + "No API key is required.",
        "",
        "## Versions",
        "",
        f"- infernix revision: `{info['revision']}`",
        f"- Python: {info['python']}",
    ]
    parts += [f"- {k}: {v}" for k, v in info["tools"].items()]
    parts += ["", "## Steps", "", "| step | status | seconds |", "|---|---|---|", steps]
    if info.get("score"):
        s = info["score"]
        parts += [
            "", "## Quality check (--score)", "",
            f"- Mean KL vs FP8-row reference: {s['mean_kl']:.4f}; delta NLL {s['mean_delta_nll']:+.3f}; "
            f"top-1 disagreement {s['top1_disagreement']:.4f}; perplexity {s['perplexity']:.3f}",
            "- Instrument: windows of 512 tokens (English subset), FP8 KV, "
            + ("GGUF AllowA8" if p["method"] == "gguf" else "Q5 A16" if p["bits"] == 5
               else "W4A8 exact route, Q5 cells A16" if p.get("q5_units") else "W4A8 exact route")
            + ", no MTP, text only, native full-vocabulary KL.",
        ]
    return "\n".join(parts) + "\n"


def build_info(builder: "Builder") -> dict:
    cfg, settings = builder.cfg, builder.serve
    receipt = read_json(Path(str(cfg.artifact) + ".build.json"))
    launcher_path = cfg.out_dir / f"serve-{cfg.name}.cmd"
    info_artifact = builder.read_artifact_info()
    steps = [
        {"id": step_id, "status": rec["status"], "seconds": rec.get("seconds", 0.0)}
        for step_id, rec in builder.state["steps"].items()
        if rec["status"] == "done"
    ]
    server = builder.server_exe
    serve = None
    if settings.launcher and server is not None:
        serve = format_command(
            launcher_tools.serve_argv(
                server, cfg.artifact, settings, mtp=info_artifact.mtp, vision=info_artifact.vision
            )
        )
    return {
        "name": cfg.name,
        "preset": {k: getattr(cfg.preset, k) for k in ("id", "summary", "meaning", "size_class", "kl", "method", "bits",
                                                        "q5_units", "evidence")},
        "mtp": info_artifact.mtp,
        "vision": info_artifact.vision,
        "artifact": str(cfg.artifact),
        "file_bytes": receipt["artifact"]["file_bytes"],
        "sha256": receipt["artifact"]["file_sha256"][cfg.artifact.name],
        "launcher": str(launcher_path) if settings.launcher and server is not None else None,
        "serve_command": serve,
        "serve": {"kv": settings.kv_dtype, "qk": settings.gemma4_qk},
        "alias": settings.model_id,
        "revision": receipt.get("revision"),
        "python": sys.version.split()[0],
        "tools": builder.tool_versions,
        "steps": steps,
        "recipe": receipt.get("recipe"),
        "converter_seconds": receipt.get("seconds"),
        "score": builder.score,
    }


def finalize(builder: "Builder") -> None:
    cfg, settings = builder.cfg, builder.serve
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if settings.launcher and builder.server_exe is not None:
        launcher_tools.write_launcher(
            cfg.out_dir / f"serve-{cfg.name}.cmd", cfg.artifact, builder.server_exe, settings,
            cuda_bin=settings.cuda_bin, info=builder.read_artifact_info(), overwrite=True,
        )
    write_summary(builder)


def write_summary(builder: "Builder") -> None:
    info = build_info(builder)
    cfg = builder.cfg
    (cfg.out_dir / "BUILD-SUMMARY.md").write_text(render_summary(info), encoding="utf-8")
    write_json(cfg.out_dir / f"{cfg.name}.gemma4-build.json", {"schema": BUILD_SCHEMA, **info})
    builder.log("wrote BUILD-SUMMARY.md and the build receipt")


def score_report(builder: "Builder") -> None:
    report = read_json(builder.cfg.work("score", "run", "report.json"))
    overall = report["kl"]["statistics"]["overall"]
    builder.score = {
        "mean_kl": overall["mean_kl"],
        "mean_delta_nll": overall["mean_delta_nll"],
        "top1_disagreement": overall["top1_disagreement"],
        "perplexity": report["overall"]["perplexity"],
    }
    write_summary(builder)


# ---------------------------------------------------------------- state and runner


def popen_argv(argv: list[str]) -> list[str]:
    """Batch files need cmd /c on Windows; executables are started directly."""
    if os.name == "nt" and Path(argv[0]).suffix.lower() in (".cmd", ".bat"):
        return ["cmd", "/c", *argv]
    return list(argv)


def tail_line(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 4096))
            text = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    return lines[-1] if lines else ""


def run_child(argv: list[str], log_path: Path, env: dict[str, str], cwd: Path, status: Callable[[str], None]) -> int:
    """Run `argv` with stdout+stderr in `log_path`; call `status(last log line)` every few seconds."""
    child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", **env}
    with log_path.open("ab") as log:
        log.write(f"$ {format_command(argv)}\n".encode("utf-8"))
        log.flush()
        process = subprocess.Popen(popen_argv(argv), cwd=cwd, env=child_env, stdout=log, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        try:
            while True:
                try:
                    return process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    status(tail_line(log_path))
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=False)
    elif path.exists() or path.is_symlink():
        path.unlink()


def device_of(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return os.stat(probe).st_dev


class BuildError(RuntimeError):
    pass


def fingerprint_of(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _hash_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def build_identity(cfg: BuildConfig) -> dict:
    """Facts that decide the artifact; a work dir only resumes under the same identity. Reads inputs, writes nothing."""
    model, files = cfg.model, {}
    index = model / "model.safetensors.index.json"
    shards = sorted(set(read_json(index)["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        files[name] = _hash_or_none(model / name)
    files["shards"] = [[name, (model / name).stat().st_size if (model / name).is_file() else None] for name in shards]
    assistant = None
    if cfg.assistant is not None:
        assistant = {name: _hash_or_none(cfg.assistant / name) for name in ("config.json", "model.safetensors.index.json")}
        assistant["weights"] = sorted(
            [path.name, path.stat().st_size] for path in cfg.assistant.glob("*.safetensors")
        )
    if cfg.preset.method == "plain":
        calibration = None
    elif cfg.calibration_text is not None:
        calibration = {"file": str(cfg.calibration_text.resolve()), "sha256": _hash_or_none(cfg.calibration_text)}
    else:
        calibration = {"source": f"{CALIBRATION_REPO}@{CALIBRATION_REVISION}", "text_sha256": CALIBRATION_TEXT_SHA256}
    return {
        "preset": cfg.preset.id,
        "model": str(model.resolve()),
        "model_files": files,
        "assistant": str(cfg.assistant.resolve()) if cfg.assistant is not None else None,
        "assistant_files": assistant,
        "with_vision": cfg.with_vision,
        "name": cfg.name,
        "out_dir": str(cfg.out_dir.resolve()),
        "calibration": calibration,
        **({"gguf_recipe": {"type": cfg.preset.gguf_type, "tensor_types": GGUF_TENSOR_TYPES[cfg.preset.gguf_type]},
            "llama_cpp": str(cfg.llama_cpp.resolve()) if cfg.llama_cpp else None,
            "llama_cpp_source": str(cfg.llama_cpp_source.resolve()) if cfg.llama_cpp_source else None}
           if cfg.preset.method == "gguf" else {}),
    }


def load_state(work: Path, fingerprint: str, identity: dict) -> dict | None:
    """The saved state of a work dir that matches, None for a missing or empty dir; refuses anything else."""
    if not work.exists() or not any(work.iterdir()):
        return None
    path = work / "build-state.json"
    if not path.is_file():
        raise BuildError(
            f"{work} is not empty and has no build-state.json, so it is not a gemma4_build work dir. "
            "Choose another --work-dir or empty this one yourself."
        )
    state = read_json(path)
    if state.get("fingerprint") != fingerprint:
        saved = state.get("identity", {})
        changed = sorted(key for key in set(identity) | set(saved) if identity.get(key) != saved.get(key))
        raise BuildError(
            f"{work} was created with different arguments ({', '.join(changed) or 'identity'}). Rerun with the "
            "original arguments to resume, pick a new --work-dir, or delete this one."
        )
    return state


class Builder:
    """Runs a plan in order, skipping steps whose done-marker matches, and records every step in the work dir."""

    def __init__(
        self,
        cfg: BuildConfig,
        steps: list[Step],
        identity: dict,
        serve: ServeSettings,
        *,
        server_exe: Path | None = None,
        tool_versions: dict[str, str] | None = None,
        disk_free: Callable[[Path], int] | None = None,
        runner: Callable[..., int] = run_child,
        out: Callable[[str], None] = print,
        artifact_info: launcher_tools.ArtifactInfo | None = None,
    ):
        self.cfg, self.steps, self.identity, self.serve = cfg, steps, identity, serve
        self.server_exe = server_exe
        self.tool_versions = tool_versions or {}
        self.disk_free = disk_free or (lambda path: shutil.disk_usage(path).free)
        self.runner, self.out = runner, out
        self.artifact_info = artifact_info
        self.fingerprint = fingerprint_of(identity)
        self.state: dict = {}
        self.score: dict | None = None
        self._log_path: Path | None = None
        self._last_status = 0.0

    # -- state
    @property
    def state_path(self) -> Path:
        return self.cfg.work("build-state.json")

    def save(self) -> None:
        write_json(self.state_path, self.state)

    def read_artifact_info(self) -> launcher_tools.ArtifactInfo:
        if self.artifact_info is None:
            self.artifact_info = launcher_tools.artifact_info(self.cfg.artifact)
        return self.artifact_info

    def open_work_dir(self) -> None:
        """Create or resume the work dir; refuse one that is foreign or was made with other arguments."""
        state = load_state(self.cfg.work_dir, self.fingerprint, self.identity)
        if state is not None:
            self.state = state
            return
        work = self.cfg.work_dir
        work.mkdir(parents=True, exist_ok=True)
        (work / "logs").mkdir(exist_ok=True)
        self.state = {"schema": STATE_SCHEMA, "fingerprint": self.fingerprint, "identity": self.identity,
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "steps": {}}
        self.save()

    def check_outputs_free(self) -> None:
        """Never overwrite: an external output may pre-exist only if this work dir already started its step."""
        for step in self.steps:
            if not step.external or step.id in self.state["steps"]:
                continue
            for path in step.outputs:
                if path.exists():
                    raise BuildError(
                        f"Output already exists: {path}. This build never overwrites; choose another --name or "
                        "--out-dir, or move the file away."
                    )

    def step_key(self, step: Step) -> str:
        args = step.argv[1:] if step.argv else [step.id]
        return hashlib.sha256(json.dumps([self.fingerprint, step.id, args, step.env]).encode()).hexdigest()

    # -- output
    def log(self, message: str) -> None:
        if self._log_path is not None:
            with self._log_path.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    def status(self, index: int, step: Step, started: float, last: str) -> None:
        now = time.time()
        if sys.stdout.isatty():
            line = f"[{index}/{len(self.steps)}] {step.id} {int(now - started) // 60}:{int(now - started) % 60:02d} {last}"
            sys.stdout.write("\r" + line[:110].ljust(110))
            sys.stdout.flush()
        elif now - self._last_status >= 60:
            self._last_status = now
            self.out(f"[{index}/{len(self.steps)}] {step.id} running {int(now - started) // 60} min: {last[:100]}")

    # -- pruning
    def disk_short(self) -> bool:
        cfg = self.cfg
        need = 10.0 * 2**30
        if device_of(cfg.out_dir) == device_of(cfg.work_dir):
            need += cfg.preset.file_gb * 1e9
        if cfg.score:
            need += SCORE_EXTRA_GB * 1e9
        return self.disk_free(cfg.work_dir) < need

    def prune(self, step: Step) -> None:
        targets = list(step.prune_always)
        if step.prune_if_short and self.disk_short():
            targets += step.prune_if_short
        for path in targets:
            if path.exists():
                remove_path(path)
                self.out(f"    removed {path} (no longer needed{'; disk is short' if path in step.prune_if_short else ''})")
                self.state["steps"][step.id].setdefault("pruned", []).append(str(path))

    # -- run
    def finished(self, step: Step) -> bool:
        record = self.state["steps"].get(step.id)
        if not record or record.get("status") != "done" or record.get("key") != self.step_key(step):
            return False
        pruned = set(record.get("pruned", []))
        return all(path.exists() or str(path) in pruned or self.pruned_later(path) for path in step.outputs)

    def pruned_later(self, path: Path) -> bool:
        return any(str(path) in rec.get("pruned", []) for rec in self.state["steps"].values())

    def run_step(self, index: int, step: Step) -> None:
        total = len(self.steps)
        if self.finished(step):
            self.out(f"[{index}/{total}] {step.id}: already done, skipped")
            return
        for path in step.outputs:
            if path.exists() and (not step.external or step.id in self.state["steps"]):
                remove_path(path)
        self._log_path = self.cfg.work("logs", f"{index:02d}-{step.id}.log")
        started = time.time()
        self.out(f"[{index}/{total}] {step.id}: {step.title} (about {step.minutes:g} min); log {self._log_path}")
        self.state["steps"][step.id] = {"status": "running", "key": self.step_key(step), "log": str(self._log_path),
                                         "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.save()
        try:
            if step.call is not None:
                step.call(self)
                code = 0
            else:
                code = self.runner(step.argv, self._log_path, step.env, REPO_ROOT,
                                   lambda last: self.status(index, step, started, last))
        except BaseException as error:
            self.finish_failed(step, started, str(error))
            raise
        if code != 0:
            self.finish_failed(step, started, f"exit code {code}")
            raise BuildError(
                f"step {step.id} failed (exit code {code}). Log: {self._log_path}\n    last line: {tail_line(self._log_path)}\n"
                "    Fix the cause and rerun the same command: finished steps are skipped."
            )
        record = self.state["steps"][step.id]
        record.update(status="done", seconds=time.time() - started, finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.save()
        if sys.stdout.isatty():
            sys.stdout.write("\r" + " " * 110 + "\r")
        self.out(f"[{index}/{total}] {step.id}: done in {record['seconds'] / 60:.1f} min")
        self.prune(step)
        self.save()

    def finish_failed(self, step: Step, started: float, reason: str) -> None:
        self.state["steps"][step.id].update(status="failed", seconds=time.time() - started, error=reason)
        self.save()

    def run(self) -> None:
        self.open_work_dir()
        self.check_outputs_free()
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        for index, step in enumerate(self.steps, 1):
            try:
                self.run_step(index, step)
            except Exception as error:
                message = str(error) if isinstance(error, BuildError) else (
                    f"step {step.id} failed: {error}\n    Log: {self._log_path}"
                )
                if step.id.startswith("score-"):
                    message = (
                        f"The build itself is complete and usable ({self.cfg.artifact}); only the quality check "
                        f"failed.\n    {message}"
                    )
                raise BuildError(message) from error

    def cleanup(self) -> bool:
        """Delete this build's own work dir after success; returns whether it did."""
        work = self.cfg.work_dir.resolve()
        protected = [self.cfg.out_dir.resolve(), self.cfg.model.resolve(), REPO_ROOT.resolve()]
        if self.cfg.assistant:
            protected.append(self.cfg.assistant.resolve())
        if any(work == item or work in item.parents for item in protected):
            raise BuildError(f"refusing to delete {work}: it contains the output, a source or the checkout")
        if self.cfg.keep_work or not self.state_path.is_file():
            return False
        shutil.rmtree(work)
        return True


# ---------------------------------------------------------------- serve-check

SERVE_CHECK_SCHEMA = "infernix.serve-check/v1"
SERVE_CHECK_TIMEOUT = 300.0  # seconds until /v1/models must answer
SERVE_CHECK_ACTIONS = {
    "ready": "read the server's startup output (is the port free, is the artifact readable, is there VRAM for the "
             "context?) and start it by hand",
    "models list": "write the launcher again with the intended --alias",
    "chat with the launcher model id": "the server refuses its own --model-id; rebuild infernix-serve from this "
                                      "checkout (build_native.bat)",
    "chat with another model name": "the server does not honor --any-model-name; rebuild infernix-serve from this "
                                    "checkout (build_native.bat) or write the launcher with --model-match exact",
    "chat without a model field": "the server must answer 400 to a request without a model field; rebuild it from "
                                  "this checkout",
    "chat with an arbitrary bearer token": "start the server without --api-key so it accepts any key a client sends",
}


@dataclass(frozen=True)
class LauncherCommand:
    """What a generated serve launcher runs: the server exe, the artifact and the flat flag list."""

    server: Path
    artifact: Path
    flags: tuple[str, ...]
    host: str
    port: int
    alias: str | None  # --model-id; None when the launcher omits it (the server names the model itself)
    model_match: str  # "any" only when the launcher passes --any-model-name; the server default is exact
    kv_dtype: str | None
    context: int | str | None  # --max-context, including explicit "auto"
    gemma4_qk: str  # "bf16" when the launcher omits --gemma4-qk (the engine default)
    prefill_chunk: str | None
    draft_tokens: str | None  # None without --spec mtp
    logits_dtype: str | None
    host_context_mib: int | None
    device_snapshot_slots: int | None
    gemma4_key_only_kv: bool = False
    gemma4_sliding_kv: str = "bf16"
    gemma4_q5_prefill: str = "a16"
    env: dict[str, str] = field(default_factory=dict)  # `set NAME=VALUE` lines; PATH is prepended, not replaced

    def effective_profile(self) -> dict[str, object]:
        """The serving profile this launcher selects, as recorded in the serve-check receipt."""
        return {
            "host": self.host, "port": self.port, "model_id": self.alias, "model_match": self.model_match,
            "max_context": self.context, "kv_dtype": self.kv_dtype, "gemma4_qk": self.gemma4_qk,
            "prefill_chunk": self.prefill_chunk, "draft_tokens": self.draft_tokens,
            "logits_dtype": self.logits_dtype, "host_context_mib": self.host_context_mib,
            "device_snapshot_slots": self.device_snapshot_slots,
            **({"gemma4_key_only_kv": True} if self.gemma4_key_only_kv else {}),
            **({"gemma4_sliding_kv": self.gemma4_sliding_kv} if self.gemma4_sliding_kv != "bf16" else {}),
            **({"gemma4_q5_prefill": self.gemma4_q5_prefill} if self.gemma4_q5_prefill != "a16" else {}),
        }


def _launcher_words(line: str) -> list[str]:
    """Split a launcher command line into words, keeping quoted paths (Windows backslashes) whole."""
    words: list[str] = []
    for piece in re.split(r'("[^"]*")', line):
        if piece.startswith('"') and piece.endswith('"') and len(piece) >= 2:
            words.append(piece[1:-1].replace("%%", "%"))
        else:
            words.extend(piece.replace("%%", "%").split())
    return words


def parse_launcher(path: Path) -> LauncherCommand:
    """Read a generated serve `.cmd` and return the server command it runs; refuses anything unparsable."""
    if not path.is_file():
        raise ValueError(f"launcher not found: {path}")
    logical: list[str] = []
    env: dict[str, str] = {}
    pending = ""
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if line.lower().startswith("set "):
            name, sep, text = line[4:].partition("=")
            if sep and name.strip():
                key, text = name.strip().strip('"'), text.strip().strip('"')
                if key.upper() == "PATH" and text.upper().endswith("%PATH%"):
                    text = text[: -len("%PATH%")].rstrip(";")  # the launcher prepends to the inherited PATH
                env[key] = text
            continue
        if line == "exit /b %errorlevel%":
            continue
        if line.endswith("^"):
            pending += " " + line[:-1].strip()
            continue
        logical.append((pending + " " + line).strip())
        pending = ""
    if pending:
        logical.append(pending.strip())
    # The memory-16gb launcher plans first (`--plan` to a temp file); the command after it is the one that serves.
    command = next((line for line in logical if line.startswith('"') and " --plan " not in line), "")
    words = _launcher_words(command)
    if len(words) < 2:
        raise ValueError(f"no quoted \"<server>\" \"<artifact>\" command line found in {path}")
    flags = words[2:]

    def value(name: str, default=None):
        return flags[flags.index(name) + 1] if name in flags else default

    def number(name: str) -> int | None:
        text = value(name)
        return int(text) if text is not None and text.isdigit() else None

    return LauncherCommand(
        server=Path(words[0]), artifact=Path(words[1]), flags=tuple(flags),
        host=value("--host", launcher_tools.SERVER_DEFAULT_HOST),
        port=int(value("--port", launcher_tools.SERVER_DEFAULT_PORT)),
        alias=value("--model-id"), model_match="any" if "--any-model-name" in flags else "exact",
        kv_dtype=value("--kv-dtype"), context="auto" if value("--max-context") == "auto" else number("--max-context"),
        gemma4_qk=value("--gemma4-qk", "bf16"), prefill_chunk=value("--prefill-chunk"),
        draft_tokens=value("--draft-tokens") if value("--spec") == "mtp" else None,
        logits_dtype=value("--logits-dtype"), host_context_mib=number("--host-context-mib"),
        gemma4_key_only_kv="--gemma4-key-only-kv" in flags,
        gemma4_sliding_kv=value("--gemma4-sliding-kv", "bf16"),
        gemma4_q5_prefill=value("--gemma4-q5-prefill", "a16"),
        device_snapshot_slots=number("--device-snapshot-slots"), env=env,
    )


def _request(method: str, url: str, body: dict | None = None, headers: dict | None = None,
             timeout: float = 30.0) -> tuple[int, bytes]:
    """One HTTP request; returns (0, b"") on connection errors instead of raising, so polling stays simple."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Content-Type", "application/json")
    for name, text in (headers or {}).items():
        request.add_header(name, text)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:  # a real HTTP answer, just not 2xx
        return error.code, error.read()
    except (urllib.error.URLError, OSError):
        return 0, b""


def _utcstamp(moment: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(moment))


def _stop_child(child: subprocess.Popen) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait(timeout=10)


def _chat(url: str, headers: dict | None = None, **fields) -> tuple[int, bytes]:
    return _request("POST", url + "/v1/chat/completions", body=dict(fields), headers=headers)


def resolve_serve_exe(plan: LauncherCommand, override: Path | None) -> Path:
    """The `infernix-serve` to start: `--serve` if given, else the launcher's own, never another engine's binary."""
    if override is not None:
        if not override.is_file():
            raise ValueError(f"--serve binary not found: {override}")
        return override.resolve()
    if "ninfer-spo" in plan.server.name.lower():
        raise ValueError(
            f"the launcher runs another engine's server {plan.server}; write it again with "
            "`python -m tools.convert.gemma4_build launcher` or pass --serve <infernix-serve>"
        )
    if plan.server.is_file():
        return plan.server
    found = launcher_tools.find_server_exe(REPO_ROOT)
    if found is None:
        raise ValueError(
            f"server binary {plan.server} not found and no {launcher_tools.exe_name(launcher_tools.SERVER_NAME)} "
            "under apps/Release or build*/apps/Release of this checkout; build it (build_native.bat) or pass --serve"
        )
    return found


def serve_check(launcher: Path, out: Path | None = None, *, host: str | None = None, port: int | None = None,
                timeout: float = SERVE_CHECK_TIMEOUT, serve: Path | None = None) -> tuple[list[dict], Path]:
    """Start what `launcher` runs, probe the OpenAI endpoints, stop the child and write a JSON receipt.

    Returns (checks, receipt path); each check is {name, status, http_code, detail}. Raises ValueError/OSError for a
    missing or unparsable launcher, which the CLI turns into a one-line error. `serve` replaces the launcher's
    server binary (otherwise the launcher's own, or the checkout's `apps/Release/infernix-serve`, is used).
    """
    plan = parse_launcher(launcher)
    server = resolve_serve_exe(plan, serve)
    host, port = host or plan.host, port or plan.port
    flags = list(plan.flags)
    if "--host" in flags:
        flags[flags.index("--host") + 1] = host
    if "--port" in flags:
        flags[flags.index("--port") + 1] = str(port)
    started = time.time()
    # The server may bind 0.0.0.0 (all interfaces), which Windows refuses as a connect target; probe loopback.
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    base = f"http://{probe_host}:{port}"
    checks: list[dict] = []

    def record(name: str, ok: bool, http_code: int, detail: str) -> bool:
        checks.append({"name": name, "status": "pass" if ok else "fail", "http_code": http_code, "detail": detail})
        return ok
    with tempfile.TemporaryFile() as log:
        child_env = {**os.environ, **plan.env}
        if "PATH" in plan.env:
            child_env["PATH"] = plan.env["PATH"] + os.pathsep + os.environ.get("PATH", "")
        child = subprocess.Popen(popen_argv([str(server), str(plan.artifact), *flags]),
                                 stdout=log, stderr=subprocess.STDOUT, env=child_env)
        deadline = time.monotonic() + timeout
        code, body = 0, b""
        while time.monotonic() < deadline:
            if child.poll() is not None:
                break
            code, body = _request("GET", base + "/v1/models", timeout=5.0)
            if code == 200:
                break
            time.sleep(0.25)
        ready = code == 200
        record("ready", ready, code, "" if ready else
               f"GET /v1/models did not answer 200 within {timeout:g} s (child exit code {child.poll()})")
        models: dict = {}
        if ready:
            try:
                models = json.loads(body)
            except ValueError:
                models = {}
        listed = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
        expected_id = plan.alias or (listed[0] if listed else None)
        if ready:
            record("models list", expected_id is not None and expected_id in listed, code,
                   f"/v1/models lists {listed}" + (f", expected {expected_id}" if expected_id else ""))
            other = "160k" if expected_id != "160k" else "262k"
            wanted = 200 if plan.model_match == "any" else 404
            ok_code, _ = _chat(base, model=expected_id,
                               messages=[{"role": "user", "content": "serve-check ping"}], max_tokens=1)
            record("chat with the launcher model id", ok_code == 200, ok_code,
                   f"model {expected_id!r} -> HTTP {ok_code}")
            other_code, _ = _chat(base, model=other,
                                  messages=[{"role": "user", "content": "serve-check ping"}], max_tokens=1)
            record("chat with another model name", other_code == wanted, other_code,
                   f"model {other!r} with {'--any-model-name' if plan.model_match == 'any' else 'exact names'}: "
                   f"got HTTP {other_code}, want {wanted}")
            missing_code, _ = _chat(base, messages=[{"role": "user", "content": "serve-check ping"}],
                                    max_tokens=1)
            record("chat without a model field", missing_code == 400, missing_code,
                   f"no model field -> HTTP {missing_code}, want 400")
            token_code, _ = _chat(base, headers={"Authorization": "Bearer sk-serve-check"}, model=expected_id,
                                  messages=[{"role": "user", "content": "serve-check ping"}], max_tokens=1)
            record("chat with an arbitrary bearer token", token_code == 200, token_code,
                   f"Bearer sk-serve-check -> HTTP {token_code}")
        else:
            for name in list(SERVE_CHECK_ACTIONS)[1:]:
                record(name, False, 0, "skipped: the server never became ready")
        _stop_child(child)
        log.seek(0)
        stdout = log.read().decode("utf-8", errors="replace")

    entry = next((item for item in models.get("data", []) if isinstance(item, dict)), {})
    engine = {
        "model_id": expected_id,
        "model_match": plan.model_match,
        "context": entry.get("max_model_len", plan.context),
        "kv_dtype": plan.kv_dtype,
        "kv_line": next((line for line in stdout.splitlines() if "KV" in line), ""),
        "profile": plan.effective_profile(),
    }
    receipt = {
        "schema": SERVE_CHECK_SCHEMA,
        "launcher": str(launcher.resolve()),
        "server_exe": str(server),
        "artifact": {"path": str(plan.artifact), "size": plan.artifact.stat().st_size
                     if plan.artifact.is_file() else None},
        "url": base,
        "pid": child.pid,
        "started": _utcstamp(started),
        "finished": _utcstamp(time.time()),
        "engine": engine,
        "server_startup_errors": [line for line in stdout.splitlines() if " ERROR " in line or " FATAL " in line][-6:],
        "checks": checks,
    }
    out = out or launcher.resolve().parent / f"serve-check-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(started))}.json"
    write_json(Path(out), receipt)
    return checks, Path(out)


if __name__ == "__main__":
    from .gemma4_build_cli import main

    raise SystemExit(main())
