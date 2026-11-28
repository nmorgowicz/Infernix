"""Command line of `python -m tools.convert.gemma4_build`: build, check, setup-gptq, launcher, serve-check."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from . import gemma4_build_launcher as launcher_tools
from .gemma4_build import (
    BUNDLE_NAMES,
    CALIBRATION_REPO,
    CALIBRATION_REVISION,
    CARD_CLASSES,
    FEEDBACK_IMATRIX_CORPORA,
    FEEDBACK_TOOLING_ENV,
    GPTQ_PYTHON_ENV,
    LLAMA_CPP_ENV,
    LLAMA_CPP_SOURCE_ENV,
    PRESETS,
    PROFILE_BUNDLES,
    REPO_ROOT,
    SERVE_CHECK_ACTIONS,
    SERVE_CHECK_TIMEOUT,
    UNAVAILABLE_PRESETS,
    BuildConfig,
    BuildError,
    Builder,
    ServeSettings,
    build_identity,
    build_plan,
    default_gptq_python,
    default_name,
    describe_plan,
    fingerprint_of,
    preset_file_bytes,
    resolve_bundle,
    safe_name,
    serve_check,
)
from .gemma4_build_checks import GPTQ_PINS, Probes, parse_pins, preflight, versions_text

METHODS = {
    "best-small": "GPTQ g256",
    "best-medium": "GPTQ g32",
    "q5-g64": "GPTQ5 g64, A16Only",
    "q4q5-mixed-a": "GPTQ g32 + Q5 o,down",
    "q4q5-mixed-b": "GPTQ g32 + Q5 gate_up,down@20-59",
    "draft": "absmax g32, no calibration",
    "iq3-xs": "IQ3_XS, own imatrix",
    "iq3-xxs": "IQ3_XXS, own imatrix",
    "forum-iq3-16gb": "IQ3_XXS base + forum allocation, RP imatrix",
    "forum-iq3-16gb-quality": "IQ3_XXS base + F1 allocation, RP imatrix",
    "iq3-xxs-feedback": "IQ3_XXS + Hessian feedback, frozen X5 damping",
}
NEEDS = {
    "gptq": "GPU with 12 GiB free, 64 GiB RAM, about 75 GB disk, the GPTQ environment (setup-gptq)",
    "plain": "nothing extra",
    "gguf": "llama.cpp binaries and source (--llama-cpp, --llama-cpp-source); own-model imatrix; "
            "iq3-xxs-feedback additionally needs the GPTQ Python environment (the research tool's interpreter), "
            "the research tooling checkout (--feedback-tooling) and about 170 GB disk",
}
EPILOG = f"""\
presets (text only, sizes include MTP; quality hints are historical, NOT a matched ranking):
q4 hints use the diode variant / FP8-row reference; IQ3 hints use the plain variant / own-Q8_0 reference.
See docs/maintainer/gemma4-benchmarks.md#four-tier-comparison for matched diode quality and thinking-enabled speed.

{{table}}

Recommended: best-small for most people (best quality per byte); best-medium for higher-quality Q4.
q5-g64 is the explicit higher-quality Q5 tier (plain-model own-Q8 historical hint, not a diode claim).
q4q5-mixed-a is a supported option, not the default: best-medium with Q5 attention-output and down projections in every
layer (+0.94 GB, two GPTQ passes, decode about -11 %). On fresh held-out roleplay text (36 forum threads, context-512 block
KL against each model's own Q8_0, diode build, default route) its KL is 0.74x that of best-medium (interval 0.72-0.76,
lower in 36 of 36 threads). Not measured: long-context or incremental-decode KL, more than one GPTQ pass, other GPUs.
iq3-xs and iq3-xxs trade quality for smaller files and collect their own importance matrix; draft is a quick
test build only. The calibrated-RTN presets and the two-level 6-bit scale format (q4_g32_s6_fp16) are not built here.

Research-only presets (labels, never defaults; every number is measured on the diode finetune campaign, not on
your build): q4q5-mixed-b (Q5 gate_up all layers + down@20-59, two GPTQ passes, fresh KL 0.62x best-medium) and
forum-iq3-16gb / forum-iq3-16gb-quality (the measured 16 GB IQ3 recipes: pinned per-tensor allocation, tied
q4_g32 vocabulary, and an importance matrix that MUST come from the pinned forum roleplay calibration corpus of
YOUR OWN model — pass --rp-raw (raw files from `roleplay_heldout fetch`) and --rp-against (your evaluation texts);
the public and diode matrices are never reused). forum-iq3-16gb defaults to the FP8 assistant; --assistant-encoding
q4-g32 (balanced, better observed acceptance) or q4-g256 (smallest, higher tested context) stores the draft smaller.
forum-iq3-16gb-quality embeds the FP8 assistant but generates its launcher without MTP; the serve-<name>-mtp.cmd
launcher beside it is the explicit opt-in override (native --spec mtp --draft-tokens 1 --allow-shared-memory).
iq3-xxs-feedback (campaign X5 stage 1) re-solves all 410 XXS-role projections with Hessian error feedback at the
same encoded bytes: KL ratios 0.44-0.53 of the deployed XXS (confirmed at the historical strength on an untouched
confirmation draw). Its imatrix defaults to the campaign-faithful generic templated English corpus;
--imatrix-corpus roleplay swaps in the pinned forum corpus of your own model (an UNMEASURED stack). It needs
--feedback-tooling <llama-local-tooling> (the research tool checkout), --rp-raw/--rp-against (the registration
corpus), and a GPU lease for the feedback pass; the damping policy is frozen and never re-selected.

Bundled recommendations (quality / balanced / context per card class; labels resolved to concrete combinations,
not new CLI profiles; explicit flags always win; key-only KV stays explicit; MTP embedding and MTP enable are
separate; targets assume about 1.9 GiB desktop overhead and are planning targets, not capacity promises):

{{bundles}}

other commands:
  python -m tools.convert.gemma4_build setup-gptq --venv DIR     create the pinned GPTQ environment
  python -m tools.convert.gemma4_build launcher --artifact F --out FILE.cmd   write a launcher for an existing file
  python -m tools.convert.gemma4_build serve-check --launcher F.cmd   start the launcher's infernix-serve, probe the
                                       OpenAI endpoints, stop it again and write a serving receipt JSON

Calibration text defaults to the pinned public corpus {CALIBRATION_REPO}@{CALIBRATION_REVISION[:12]}.
The GPTQ scripts run in their own Python (--gptq-python, ${GPTQ_PYTHON_ENV}, or <checkout>/.venv-gptq); the converter
runs in the interpreter that started this command.
The serve launcher runs infernix-serve with the operator profile: 131072 context (alias 131k), FP8 global KV and
sliding K/V storage, FP8 attention QK, one lane, --prefill-chunk auto, MTP with one draft token when the file has it, BF16 logits, a host-only
prefix cache of 8192 MiB, listening on 0.0.0.0:8001. The server accepts any API key. With --model-match any (the
default) it also accepts any model name; with exact, OpenAI-style requests must use the --alias model name.
For Q5 builds/native-Q5 artifacts, omitted --profile selects q5-quality instead: int8-g64 prefill,
BF16 sliding ring, auto context, chunk 2048, top-k 64, alias gemma4-q5. Explicit --profile default keeps
the general settings above. Mixed-A builds also default to auto context when neither --profile nor --context
is supplied; all other general settings remain unchanged. See docs/weight-conversion.md for the lower-VRAM option ladder.
Omitted ring storage follows --kv-dtype (fp8 by default, bf16 with global BF16 KV), except q5-quality,
which pins BF16 ring storage regardless of global KV (fidelity-first; the FP8 ring measurably costs KL, and the pin
costs plain Q5 capacity: no measured 32 GB MTP d1 start reached 131k with either ring). `launcher` on a mixed Q4/Q5 artifact (mixed-A)
with --profile omitted selects q5-quality but keeps the ring following global KV: at class B the BF16 ring would cost
it about 13k tokens (64,256 FP8 vs 51,264 BF16 at 1.9 GiB desktop overhead).
Explicit --gemma4-sliding-kv always wins; --gemma4-qk bf16 changes compute, not storage.
"""


def preset_table() -> str:
    rows = [f"  {'id':<12} {'method':<28} {'size':<7} {'file':<9} {'KL':<7} {'wait':<13} needs"]
    for preset in PRESETS.values():
        rows.append(
            f"  {preset.id:<12} {METHODS[preset.id]:<28} {preset.size_class:<7} {f'{preset.file_gb:g} GB':<9} "
            f"{preset.kl:<7.4f} {preset.wait:<13} {NEEDS[preset.method]}"
            + ("; supported option, not the default; roleplay KL 0.74x best-medium (held-out, diode)"
               if preset.id == "q4q5-mixed-a" else "")
            + (f"; {preset.label}" if preset.label else "")
        )
    return "\n".join(rows)


def bundle_table() -> str:
    """The card-class recommendation labels resolved to concrete combinations (labels, not CLI profiles)."""
    rows = [f"  {'card':<6} {'bundle':<9} {'preset':<24} {'serve profile':<14} {'ctx':<8} flags / measured"]
    for card in CARD_CLASSES:
        for name in BUNDLE_NAMES:
            combo = PROFILE_BUNDLES[card][name]
            rows.append(
                f"  {card:<6} {name:<9} {combo['preset']:<24} --profile {combo['profile']:<6} "
                f"{combo['context']:<8} {' '.join(combo['flags']) or '(no extra flags)'}"
            )
            rows.append(f"  {'':<6} {'':<9} measured: {combo['measured']}"
                        + (f" [{combo['note']}]" if combo.get("note") else ""))
    return "\n".join(rows)


def default_venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def context_value(text: str) -> int | str:
    if text == "auto":
        return text
    try:
        return int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("context must be a positive token count or 'auto'") from error


def add_serve_arguments(parser: argparse.ArgumentParser, *, prefix: str) -> None:
    """The launcher flags shared by a build and the `launcher` subcommand; `prefix` names the command in help."""
    add = parser.add_argument
    add("--profile", choices=launcher_tools.SERVE_PROFILES, default=None,
        help="omitted: q5-quality for q5-g64 builds/native Q5 artifacts, default for other tiers. "
        "Explicit default keeps the general 131k roleplay settings. Q5 profiles default to int8-g64 "
        "prefill, auto context, chunk 2048, top-k 64 and no key-only KV; q5-quality pins the BF16 "
        "sliding ring (an omitted-profile launcher for a mixed Q4/Q5 artifact keeps the FP8 ring), "
        "q5-long-context follows global KV (FP8 with global FP8). "
        "memory-16gb: IQ3_XXS, key-only FP8 globals/ring, chunk 128")
    add("--context", type=context_value,
        help=f"{prefix} --max-context: positive tokens or auto (general default 131072; implicit mixed-A/Q5/memory-16gb: auto)")
    add("--alias", help=f"{prefix} --model-id, the name /v1/models reports and requests must use unless "
        "--model-match any (general default <context/1000>k; Q5: gemma4-q5; memory-16gb: gemma4-16gb)")
    add("--host", default=launcher_tools.DEFAULT_HOST, help=f"{prefix} --host (default {launcher_tools.DEFAULT_HOST}, "
        "all interfaces; 127.0.0.1 for this machine only)")
    add("--port", type=int, default=launcher_tools.DEFAULT_PORT,
        help=f"{prefix} --port (default {launcher_tools.DEFAULT_PORT})")
    add("--kv-dtype", choices=launcher_tools.KV_DTYPES, default="fp8",
        help=f"{prefix} global KV storage (default fp8, qualified; an omitted sliding ring follows this choice, "
        "except q5-quality, which always pins BF16)")
    add("--model-match", choices=launcher_tools.MODEL_MATCHES, default=launcher_tools.DEFAULT_MODEL_MATCH,
        help="any (default) passes --any-model-name so OpenAI requests may send any model name, as a gateway that "
        "forwards several names needs; exact accepts only --alias and emits no flag")
    add("--gemma4-qk", choices=launcher_tools.GEMMA4_QKS, default=launcher_tools.DEFAULT_GEMMA4_QK,
        help="Gemma attention QK precision: fp8 (default: faster long-prompt prefill, measurably lower fidelity; "
        "needs --kv-dtype fp8) or bf16 (no flag, the engine default)")
    add("--gemma4-key-only-kv", action="store_true",
        help="experimental global key-only FP8 KV (default off; memory-16gb enables it)")
    add("--gemma4-sliding-kv", choices=("bf16", "fp8"), default=None,
        help="sliding K/V ring storage: omitted follows --kv-dtype (fp8 by default, bf16 with global BF16 KV), "
        "except the q5-quality fidelity profile, which defaults to bf16 regardless of global KV (a launcher "
        "for a mixed Q4/Q5 artifact with --profile omitted still follows global KV); fp8 uses lower "
        "memory with lossy K/V storage and requires --kv-dtype fp8. Explicit bf16 preserves BF16 storage for "
        "every profile, independently of --gemma4-qk")
    add("--gemma4-q5-prefill", choices=("a16", "fp8", "int8-g32", "int8-g64"), default=None,
        help="Gemma4 native Q5 trunk calls of 64+ columns: fp8 row-E4M3 bake or int8-g32/g64 groupwise INT8 "
        "activations; Q5 profiles default to int8-g64 (lossy speed/fidelity balance), others to a16 "
        "(preserves artifact policy, no flag). Explicit choices override profiles; narrow calls stay A16")
    add("--allow-shared-memory", action="store_true",
        help="opt in to starting when the model does not fit the free VRAM: the shortfall runs in shared GPU memory "
        "(system RAM over PCIe, much slower; default off = the server refuses). memory-16gb tries resident starts first")
    add("--prefill-chunk",
        help="prefill chunk: Q5 default 2048; memory-16gb 128; general default auto (the server picks the largest "
        "of 2048/1024/512/256/128 that fits full context after weights load), or a multiple of 128 up to 16384")
    add("--draft-tokens", default=launcher_tools.DEFAULT_DRAFT_TOKENS,
        help="MTP drafts per round: 1 (default, best for growing roleplay context), 2..7, or auto (6). Ignored "
        "without an MTP assistant")
    add("--logits-dtype", choices=launcher_tools.LOGITS_DTYPES, default=launcher_tools.DEFAULT_LOGITS_DTYPE,
        help="the head's logits the sampler reads: bf16 (default, rounded once as Transformers returns them) or fp32")
    add("--host-context-mib", type=int, default=launcher_tools.DEFAULT_HOST_CONTEXT_MIB,
        help="pinned host prefix-cache pool in MiB (default %(default)s); the launcher gives Gemma no device "
        "snapshot slots")
    add("--vram-headroom-mib", type=int, default=None,
        help="VRAM in MiB the KV sizing leaves free for the display and other programs (default 1024; memory-16gb: 64)")
    add("--top-k", type=int, help="process-level top-k, 0..64 for Gemma (Q5 default 64; otherwise no flag, "
        "the artifact's sampling preset applies)")
    add("--chat-template", type=Path, help="Jinja chat template file that replaces the artifact's (--chat-template)")
    add("--default-thinking-budget", type=int, help="process-level thinking cap for requests that set none, in tokens "
        "(default: no flag; the server's answer room still ends a long thought). A thought that reaches it is closed "
        "with a short notice and the model answers")
    add("--thinking-budget-message", help="text that replaces the built-in notice a reached thinking budget (or "
        "--reasoning-loop conclude) commits before the thought close (default: no flag, the built-in notice)")
    add("--reasoning-loop", choices=launcher_tools.REASONING_LOOPS, help="end (stop) or close (conclude) a thought "
        "that keeps repeating whole passages (default: no flag, off)")
    add("--cuda-bin", type=Path, help="directory the launcher puts on PATH (if the DLLs are not beside the exe)")


def serve_kwargs(args: argparse.Namespace, *, default_profile: str | None = "default") -> dict:
    """Explicit arguments plus profile selection; ServeProfile owns unspecified-field defaults."""
    return {
        "profile": args.profile if args.profile is not None else default_profile,
        "host": args.host, "port": args.port, "alias": args.alias, "context": args.context,
        "kv_dtype": args.kv_dtype, "model_match": args.model_match, "gemma4_qk": args.gemma4_qk,
        "prefill_chunk": args.prefill_chunk, "draft_tokens": args.draft_tokens, "logits_dtype": args.logits_dtype,
        "host_context_mib": args.host_context_mib, "vram_headroom_mib": args.vram_headroom_mib,
        "top_k": args.top_k, "chat_template": args.chat_template.resolve() if args.chat_template else None,
        "default_thinking_budget": args.default_thinking_budget,
        "thinking_budget_message": args.thinking_budget_message, "reasoning_loop": args.reasoning_loop,
        "gemma4_key_only_kv": args.gemma4_key_only_kv, "allow_shared_memory": args.allow_shared_memory,
        "gemma4_sliding_kv": args.gemma4_sliding_kv,
        "gemma4_q5_prefill": args.gemma4_q5_prefill,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build",
        description="Build a Gemma 4 31B .ninfer for infernix-serve from a BF16 checkpoint in one command "
        "(Q4, Q5 or IQ3 weights; FP8 KV cache and attention QK in the launcher): checks, plan, "
        "calibration, quantization, conversion, receipts, launcher.",
        epilog=EPILOG.format(table=preset_table(), bundles=bundle_table()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add = parser.add_argument
    add("--preset", help="what to build (see the table below); required except with --profile memory-16gb (iq3-xxs)")
    add("--list-presets", action="store_true", help="print the preset table and exit")
    add("--model", type=Path, help="BF16 Hugging Face directory of Gemma 4 31B dense or a fine-tune (only read)")
    add("--assistant", type=Path, help="google/gemma-4-31B-it-assistant directory, embedded as the MTP component")
    add("--no-mtp", action="store_true", help="build without the MTP assistant (pass this or --assistant)")
    add("--out-dir", type=Path, help="where the deliverables go; the artifact is never overwritten")
    add("--work-dir", type=Path, help="scratch and resume state (default: <out-dir>/work-<name>); put it on a "
        "big fast drive; deleted after success unless --keep-work")
    add("--name", help="artifact name, used for the file names (default: <model dir>-<preset>)")
    add("--with-vision", action="store_true", help="keep the vision tower (default: text only, as measured)")
    add("--calibration-text", type=Path, help="your own public UTF-8 calibration text, disjoint from anything you "
        "evaluate on (default: download the pinned public English corpus)")
    add("--assistant-encoding", choices=("fp8", "q4-g32", "q4-g256"), default=None,
        help="forum-iq3-16gb only: the MTP assistant's stored encoding: fp8 (default), q4-g32 (balanced, better "
        "observed acceptance, 24,772,608 B larger than g256) or q4-g256 (smallest, higher tested context)")
    add("--rp-raw", type=Path, help="forum presets: directory with the six raw forum parquet files "
        "(python -m tools.eval.roleplay_heldout fetch --out DIR); required, the imatrix comes from your own model "
        "on this corpus")
    add("--rp-against", type=Path, action="append", default=[], help="forum presets: evaluation text (file or "
        "directory of *.txt) the calibration corpus must be shingle-disjoint from; repeatable, at least one")
    add("--imatrix-corpus", choices=FEEDBACK_IMATRIX_CORPORA, default=None,
        help="iq3-xxs-feedback only: the importance-matrix corpus: public (default, the generic templated English "
        "corpus the campaign measured — the campaign-faithful stack) or roleplay (the pinned forum calibration "
        "corpus of your own model; UNMEASURED stack)")
    add("--feedback-tooling", type=Path, help=f"iq3-xxs-feedback only: llama-local-tooling checkout with "
        f"converter/gguf/iq_hessian_feedback.py (default: ${FEEDBACK_TOOLING_ENV})")
    add("--gptq-python", type=Path, help=f"python.exe of the GPTQ environment (default: ${GPTQ_PYTHON_ENV}, else "
        "<checkout>/.venv-gptq); never the converter's interpreter")
    add("--llama-cpp", type=Path, help=f"IQ3 presets: directory containing llama-quantize and llama-imatrix "
        f"(default: ${LLAMA_CPP_ENV})")
    add("--llama-cpp-source", type=Path, help=f"IQ3 presets: llama.cpp source with convert_hf_to_gguf.py "
        f"(default: ${LLAMA_CPP_SOURCE_ENV})")
    add("--gptq-max-cpu-gb", type=int, default=32, help="GPTQ host-RAM budget for offloaded weights (default 32, the "
        "measured route; host peak is about this plus 8 GiB; lower it on a machine with less free RAM, the rest "
        "goes to disk and the step runs slower)")
    add_serve_arguments(parser, prefix="launcher")
    add("--serve", "--server-exe", dest="serve", type=Path, help="infernix-serve executable (default: the newest "
        "apps/Release or build*/apps/Release of the checkout that knows the preset's format)")
    add("--no-launcher", action="store_true", help="do not generate a launcher (and do not need the server binary)")
    add("--score", action="store_true", help="after the build, run the paired-KL quality check with "
        "infernix-perplexity (native full-vocabulary KL; about 1 h and 60 GB extra; needs infernix-perplexity "
        "beside the server exe)")
    add("--keep-work", action="store_true", help="keep the work dir (logs, intermediates) after success")
    add("--yes", action="store_true", help="do not ask for confirmation")
    add("--dry-run", action="store_true", help="print the checks and the exact plan, write nothing, exit 0")
    add("--check", action="store_true", help="run only the preflight checks; exit 1 if any fails")
    return parser


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build setup-gptq",
        description="Create the pinned GPTQ environment (uv venv + tools/gptq/requirements.txt, torch cu130) and "
        "verify its versions. Needs uv (https://docs.astral.sh/uv/) and about 10 GB.",
    )
    parser.add_argument("--venv", type=Path, default=REPO_ROOT / ".venv-gptq", help="environment directory "
                        "(default: <checkout>/.venv-gptq); an existing environment is reused")
    parser.add_argument("--python", default="3.13", help="Python version for uv venv (default 3.13, the measured one)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    return parser


def launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build launcher",
        description="Write a Windows .cmd that starts infernix-serve on a .ninfer with the operator profile: FP8 KV "
        "cache and sliding K/V storage, FP8 attention QK, --prefill-chunk auto, MTP with one draft token, BF16 logits, a host-only prefix "
        "cache, 0.0.0.0:8001, alias 131k. Native-Q5 artifacts instead default to q5-quality: int8-g64 prefill, "
        "BF16 ring, auto context, chunk 2048, top-k 64, alias gemma4-q5; a mixed Q4/Q5 artifact keeps the ring "
        "following global KV (FP8). Explicit --profile default keeps "
        "the general profile. Explicit --gemma4-sliding-kv bf16 preserves BF16 storage; an omitted ring follows "
        "global KV, except q5-quality, which pins BF16. "
        "MTP and vision are detected; every setting can be changed.",
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="new .cmd file (never overwritten)")
    parser.add_argument("--serve", "--server-exe", dest="serve", type=Path, help="infernix-serve executable "
                        "(default: newest build found in apps/Release or build*/apps/Release)")
    add_serve_arguments(parser, prefix="infernix-serve")
    parser.add_argument("--no-mtp", action="store_true", help="leave speculative decoding off even if the file has MTP")
    return parser


def serve_check_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.convert.gemma4_build serve-check",
        description="Start what a generated serve launcher runs (infernix-serve), probe the OpenAI endpoints "
        "(/v1/models and chat completions: the launcher model id, another name under the launcher's model-name "
        "policy, a missing model field, an arbitrary bearer key), stop the child again and write a serving receipt "
        "JSON with the launcher's effective profile. Exit 0 only if every check passes.",
    )
    parser.add_argument("--launcher", required=True, type=Path, help="serve-<name>.cmd written by the build or the "
                        "launcher command")
    parser.add_argument("--serve", "--server-exe", dest="serve", type=Path, help="infernix-serve executable to run "
                        "instead of the launcher's (default: the launcher's, else apps/Release/infernix-serve of "
                        "the checkout)")
    parser.add_argument("--out", type=Path, help="receipt JSON path (default: serve-check-<utcstamp>.json beside "
                        "the launcher)")
    parser.add_argument("--host", help="override the launcher --host for the probe (default: the launcher's)")
    parser.add_argument("--port", type=int, help="override the launcher --port for the probe (default: the launcher's)")
    parser.add_argument("--timeout", type=float, default=SERVE_CHECK_TIMEOUT,
                        help="seconds the server may take to answer /v1/models (default %(default)s)")
    return parser


# ---------------------------------------------------------------- build


def resolve_config(args: argparse.Namespace) -> tuple[BuildConfig, ServeSettings, Path | None]:
    """Turn flags into a config; every argument-only mistake is refused here, before any probe."""
    if args.preset is None and args.profile == "memory-16gb":
        args.preset = "iq3-xxs"
    if args.profile in ("q5-quality", "q5-long-context"):
        if args.preset is None:
            args.preset = "q5-g64"
        elif args.preset != "q5-g64":
            raise ValueError(f"--profile {args.profile} builds --preset q5-g64 only")
    if args.profile == "memory-16gb" and args.preset != "iq3-xxs":
        raise ValueError("--profile memory-16gb builds --preset iq3-xxs only")
    if args.profile == "memory-16gb" and args.with_vision:
        raise ValueError("--profile memory-16gb is text-only; omit --with-vision")
    if args.preset is None:
        raise ValueError("--preset is required (see --list-presets)")
    if args.preset in UNAVAILABLE_PRESETS:
        raise ValueError(f"preset {args.preset!r} is not available in infernix: {UNAVAILABLE_PRESETS[args.preset]}")
    if args.preset not in PRESETS:
        raise ValueError(f"unknown preset {args.preset!r}; choose one of {', '.join(PRESETS)}")
    preset = PRESETS[args.preset]
    if preset.feedback:
        if not args.rp_raw or not args.rp_against:
            raise ValueError(f"preset {preset.id} registers its run against the pinned forum roleplay corpus: "
                "pass --rp-raw (raw files from `roleplay_heldout fetch`) and --rp-against (your evaluation "
                "texts); the registration corpus, validation windows and untouched confirmation draw never reuse "
                "public or diode data")
        if args.imatrix_corpus == "roleplay" and args.calibration_text is not None:
            raise ValueError(f"preset {preset.id} with --imatrix-corpus roleplay quantizes with the forum "
                "calibration corpus; --calibration-text would replace it and is refused")
        if args.assistant_encoding is None:
            args.assistant_encoding = "fp8"
    elif preset.assistant_encodings:
        if not args.rp_raw or not args.rp_against:
            raise ValueError(f"preset {preset.id} quantizes with an importance matrix from your own model on the "
                "pinned forum roleplay calibration corpus: pass --rp-raw (raw files from `roleplay_heldout fetch`) "
                "and --rp-against (your evaluation texts); public and diode matrices are never reused")
        if args.calibration_text is not None:
            raise ValueError(f"preset {preset.id} uses the roleplay calibration corpus; --calibration-text would "
                "replace it and is refused")
        if args.assistant_encoding is None:
            args.assistant_encoding = "fp8"
    elif args.assistant_encoding is not None:
        raise ValueError("--assistant-encoding applies to forum-iq3-16gb only")
    if not preset.feedback and (args.imatrix_corpus is not None or args.feedback_tooling is not None):
        raise ValueError("--imatrix-corpus and --feedback-tooling apply to iq3-xxs-feedback only")
    feedback_tooling = args.feedback_tooling or os.environ.get(FEEDBACK_TOOLING_ENV)
    if preset.feedback and not feedback_tooling:
        raise ValueError(f"preset {args.preset} drives the registered X5 research tool in a llama-local-tooling "
                         f"checkout: pass --feedback-tooling <checkout> or set {FEEDBACK_TOOLING_ENV}")
    for flag in ("model", "out_dir"):
        if getattr(args, flag) is None:
            raise ValueError(f"--{flag.replace('_', '-')} is required")
    if args.no_mtp == (args.assistant is not None):
        raise ValueError("Pass --assistant <dir> (hf download google/gemma-4-31B-it-assistant --local-dir <DIR>) "
                         "or explicitly --no-mtp")
    model, out_dir = args.model.resolve(), args.out_dir.resolve()
    name = safe_name(args.name) if args.name else default_name(model, preset.id)
    work_dir = (args.work_dir or out_dir / f"work-{name}").resolve()
    if work_dir == out_dir:
        raise ValueError("--work-dir must not be the output directory itself")
    kwargs = serve_kwargs(args, default_profile="q5-quality" if preset.bits == 5 else "default")
    if preset.id in ("q4q5-mixed-a", "q4q5-mixed-b", "iq3-xxs-feedback") and args.profile is None and args.context is None:
        kwargs["context"] = "auto"
    launcher_tools.check_serve_settings(launcher_tools.ServeProfile(**kwargs))
    if args.score and args.no_launcher and args.serve is None:
        raise ValueError("--score needs infernix-perplexity beside the server: pass --serve")
    server = args.serve.resolve() if args.serve else (
        None if args.no_launcher and not args.score
        else launcher_tools.find_server_exe(REPO_ROOT, preset.formats)
    )
    gptq_python = (args.gptq_python or default_gptq_python()).absolute() \
        if (preset.method == "gptq" or preset.feedback) else None
    llama_cpp = args.llama_cpp or os.environ.get(LLAMA_CPP_ENV)
    llama_source = args.llama_cpp_source or os.environ.get(LLAMA_CPP_SOURCE_ENV)
    cfg = BuildConfig(
        preset=preset, model=model, assistant=args.assistant.resolve() if args.assistant else None,
        with_vision=args.with_vision, out_dir=out_dir, work_dir=work_dir, name=name, gptq_python=gptq_python,
        gptq_max_cpu_gb=args.gptq_max_cpu_gb,
        calibration_text=args.calibration_text.resolve() if args.calibration_text else None,
        score=args.score, perplexity_exe=launcher_tools.sibling_binary(server, launcher_tools.PERPLEXITY_NAME)
        if args.score and server else None, keep_work=args.keep_work,
        llama_cpp=Path(llama_cpp).resolve() if llama_cpp else None,
        llama_cpp_source=Path(llama_source).resolve() if llama_source else None,
        assistant_encoding=args.assistant_encoding or "fp8",
        rp_raw=args.rp_raw.resolve() if args.rp_raw else None,
        rp_against=tuple(path.resolve() for path in args.rp_against),
        feedback_tooling=Path(feedback_tooling).resolve() if feedback_tooling else None,
        imatrix_corpus=args.imatrix_corpus or ("public" if preset.feedback else ""),
    )
    serve = ServeSettings(**kwargs, server_exe=server, cuda_bin=args.cuda_bin, launcher=not args.no_launcher)
    return cfg, serve, server


def confirm(args: argparse.Namespace) -> bool:
    if args.yes:
        return True
    if not sys.stdin.isatty():
        print("infernix: not a terminal; pass --yes to start without a prompt", file=sys.stderr)
        return False
    return input("Start the build? [y/N] ").strip().lower() in ("y", "yes")


def run_build(args: argparse.Namespace, probes: Probes) -> int:
    cfg, serve, server = resolve_config(args)
    identity = build_identity(cfg)
    fingerprint = fingerprint_of(identity)
    report = preflight(cfg, serve, server, probes, fingerprint, identity)
    print("Preflight")
    print("\n".join(report.render()))
    print()
    if args.check:
        print(f"{len(report.failed)} check(s) failed." if report.failed else "All checks passed.")
        return 1 if report.failed else 0
    steps = build_plan(cfg)
    print("\n".join(describe_plan(cfg, steps)))
    print()
    if args.dry_run:
        if report.failed:
            print(f"Dry run: nothing was written. {len(report.failed)} preflight check(s) above would stop a real run.")
        else:
            print("Dry run: nothing was written.")
        return 0
    if report.failed:
        print(f"Not starting: {len(report.failed)} preflight check(s) failed (see the lines marked [FAIL]).",
              file=sys.stderr)
        return 1
    if not confirm(args):
        print("Cancelled; nothing was written.")
        return 1
    builder = Builder(cfg, steps, identity, serve, server_exe=server, tool_versions=report.tool_versions)
    builder.run()
    removed = builder.cleanup()
    print()
    print(f"Done. Artifact: {cfg.artifact}")
    print(f"      Summary:  {cfg.out_dir / 'BUILD-SUMMARY.md'}")
    if serve.launcher and server is not None:
        print(f"      Launcher: {cfg.out_dir / f'serve-{cfg.name}.cmd'}")
    print("      Work dir removed." if removed else f"      Work dir kept: {cfg.work_dir}")
    return 0


# ---------------------------------------------------------------- setup-gptq, launcher, serve-check


def setup_gptq(args: argparse.Namespace, probes: Probes) -> int:
    uv = shutil.which("uv")
    venv = args.venv.resolve()
    python = default_venv_python(venv)
    requirements = REPO_ROOT / "tools" / "gptq" / "requirements.txt"
    create = [uv or "uv", "venv", "--python", args.python, str(venv)]
    install = [uv or "uv", "pip", "install", "--python", str(python), "--index-url",
               "https://download.pytorch.org/whl/cu130", "--extra-index-url", "https://pypi.org/simple",
               "--index-strategy", "unsafe-best-match", "-r", str(requirements)]
    reuse = (venv / "pyvenv.cfg").is_file()
    commands = ([] if reuse else [create]) + [install]
    if args.dry_run:
        for command in commands:
            print(subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command))
        print("Dry run: nothing was changed.")
        return 0
    if uv is None:
        raise ValueError("uv is not installed: see https://docs.astral.sh/uv/getting-started/installation/ "
                         "(for example: winget install astral-sh.uv), then rerun")
    if venv.exists() and any(venv.iterdir()) and not reuse:
        raise ValueError(f"{venv} exists and is not a virtual environment; choose another --venv")
    for command in commands:
        print("+ " + (subprocess.list2cmdline(command) if os.name == "nt" else " ".join(command)), flush=True)
        if subprocess.run(command).returncode != 0:
            raise ValueError(f"{command[1]} failed; fix the error above and rerun (an existing venv is reused)")
    pins = parse_pins(requirements, GPTQ_PINS)
    info = probes.python_info(str(python), GPTQ_PINS, cuda=True)
    if info is None or "error" in info:
        raise ValueError(f"the new environment's interpreter {python} does not run: {(info or {}).get('error')}")
    wrong = [f"{name} {info['packages'].get(name)} (pinned {pins[name]})" for name in pins
             if (info["packages"].get(name) or "").split("+")[0] != pins[name].split("+")[0]]
    print(f"{python}: Python {info['python']}; {versions_text(info['packages'])}; CUDA "
          f"{'visible' if info.get('cuda') else 'NOT visible'}")
    if wrong:
        raise ValueError("pinned versions differ: " + "; ".join(wrong))
    if not info.get("cuda"):
        raise ValueError("the environment installed but torch cannot see CUDA: update the NVIDIA driver (CUDA 13 "
                         "builds need a recent one) and rerun")
    print(f"Ready. GPTQ presets pick it up automatically when it is {default_venv_python(REPO_ROOT / '.venv-gptq')}; "
          f"otherwise pass --gptq-python {python} or set {GPTQ_PYTHON_ENV}")
    return 0


def run_launcher(args: argparse.Namespace) -> int:
    server = args.serve.resolve() if args.serve else launcher_tools.find_server_exe(REPO_ROOT)
    if server is None:
        raise ValueError("no infernix-serve found in apps/Release or build*/apps/Release; build it "
                         "(build_native.bat) or pass --serve")
    info = launcher_tools.artifact_info(args.artifact.resolve())
    profile = launcher_tools.artifact_profile(info, **serve_kwargs(args, default_profile=None))
    if args.profile == "memory-16gb" and (not info.iq3_xxs or info.vision):
        raise ValueError("--profile memory-16gb requires a text-only IQ3_XXS artifact; build with "
                         "--profile memory-16gb or select its IQ3_XXS .ninfer")
    info = launcher_tools.write_launcher(
        args.out.resolve(), args.artifact.resolve(), server, profile, no_mtp=args.no_mtp, cuda_bin=args.cuda_bin, info=info,
    )
    print(f"Wrote {args.out}: server {server}; MTP {'on' if info.mtp and not args.no_mtp else 'off'}; "
          f"vision {'on' if info.vision else 'off'}; --model-id {profile.model_id}; context {profile.context}; "
          f"KV {profile.kv_dtype}; QK {profile.gemma4_qk}; Q5 prefill {profile.gemma4_q5_prefill}; "
          f"ring {profile.gemma4_sliding_kv}; http://{args.host}:{args.port}")
    return 0


def run_serve_check(args: argparse.Namespace) -> int:
    checks, receipt = serve_check(args.launcher, args.out, host=args.host, port=args.port, timeout=args.timeout,
                                  serve=args.serve)
    failed = [item for item in checks if item["status"] != "pass"]
    for item in checks:
        head = "PASS" if item["status"] == "pass" else "FAIL"
        line = f"{head} {item['name']}" + (f": {item['detail']}" if item["detail"] else "")
        if item["status"] != "pass":
            action = SERVE_CHECK_ACTIONS.get(item["name"], "rerun serve-check after fixing the above")
            line += f"\n     -> {action}"
        print(line)
    print(f"Receipt: {receipt}")
    if failed:
        action = SERVE_CHECK_ACTIONS.get(failed[0]["name"], "rerun serve-check after fixing the above")
        print(f"infernix: serve-check failed: {failed[0]['name']}; {action}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None, probes: Probes | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    argv = list(sys.argv[1:] if argv is None else argv)
    probes = probes or Probes()
    try:
        if argv and argv[0] == "setup-gptq":
            return setup_gptq(setup_parser().parse_args(argv[1:]), probes)
        if argv and argv[0] == "serve-check":
            return run_serve_check(serve_check_parser().parse_args(argv[1:]))
        if argv and argv[0] == "launcher":
            return run_launcher(launcher_parser().parse_args(argv[1:]))
        args = build_parser().parse_args(argv)
        if args.list_presets:
            print(preset_table())
            print()
            print("bundled recommendations (labels, not CLI profiles):")
            print(bundle_table())
            return 0
        return run_build(args, probes)
    except (ValueError, OSError, BuildError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"infernix: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("infernix: interrupted; rerun the same command to resume", file=sys.stderr)
        return 130
