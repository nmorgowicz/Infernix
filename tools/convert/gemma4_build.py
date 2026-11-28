"""One-command Gemma 4 31B `.ninfer` builder: presets, step plan, resumable runner, serve-check.

`python -m tools.convert.gemma4_build --preset best-small --model <BF16 dir> --assistant <dir> --out-dir <dir>`
chains the steps of the conversion guide (calibration text, GPTQ, `gemma4_q4 convert`) and writes the artifact, its
receipts, a Windows `infernix-serve` launcher and BUILD-SUMMARY.md. The converter and the GPTQ scripts keep doing
the work; this module only builds their command lines, runs them with logs, and records which step finished. The
GPTQ scripts run in their own interpreter (`tools/gptq/requirements.txt`), never in the converter's. Source weights
are only read.

Presets: best-small (GPTQ group 256, `q4_g256_fp16`), best-medium (GPTQ group 32, `q4_g32_fp16`),
q5-g64 (GPTQ5 group 64, A16Only), q4q5-mixed-a (best-medium with Q5 `o` and `down` in every layer, a supported option),
q4q5-mixed-b (best-medium with Q5 `gate_up` in every layer and `down` in layers 20-59, a research-only option),
draft (uncalibrated absmax, group 32), iq3-xs / iq3-xxs
(llama.cpp IQ3_XS / IQ3_XXS blocks quantized with the model's own
importance matrix and imported verbatim; they need a llama.cpp build and its source tree for `convert_hf_to_gguf.py`),
and the research-only forum-iq3-16gb / forum-iq3-16gb-quality IQ3 recipes: the measured X2/F allocations
(sliding K IQ3_XXS or all Q/K IQ3_XXS, V IQ3_S, staged down projections) with the tied q4_g32 vocabulary, an
importance matrix from the pinned forum roleplay calibration corpus (`tools/eval/roleplay_calibration`) of the user's
own model, and their measured numbers from the diode finetune campaign (research-only, never defaults).
Also the research-only `iq3-xxs-feedback` Hessian-feedback recipe: the deployed XXS role allocation whose 410
projections are re-solved with iterative Hessian error feedback (registered X5 recipe; the frozen rolewise damping
policy and the FP64 covariance capture of the campaign stage 1), driven through the separate research tooling
checkout's `converter/gguf/iq_hessian_feedback.py` (prepare, full-baseline stock parity anchor, full-feedback,
full-pack), then imported with the ordinary IQ3 route. The campaign stage 1 collected its importance matrix from
the generic templated public corpus; that campaign-faithful choice is this preset's default, and the forum
roleplay corpus is an explicit option labelled unmeasured-stack.
The two-level 6-bit scale format `q4_g32_s6_fp16` and calibrated round-to-nearest presets are not built here.

Calibration text. Without `--calibration-text` the builder downloads the pinned public English corpus the measured
quality numbers used (eaddario/imatrix-calibration, combined_en_large.parquet) and exports it byte for byte as the
author's text file: a UTF-8 BOM, CRLF line ends, and the CP437 round trip that Windows PowerShell applied to the
export (it affects the 18,228 non-ASCII characters of the 20 MB). The result is checked against its sha256.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace as dataclasses_replace
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
RP_SOURCE_NAME = "calibration-source.txt"  # the corpus text file `roleplay_calibration build` writes
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
FORUM_F2_EVIDENCE = (
    "research-only (campaign F2, measured on the diode finetune; never a default). Exact artifact bytes: FP8 assistant "
    "12,630,891,524; q4_g32 assistant 12,431,956,996; q4_g256 assistant 12,407,184,388. On the combined 65k/16 "
    "instrument (forum fresh windows, own-diode Q8) its fresh KL ratio to the deployed XXS control is 0.9707 "
    "[0.9415, 1.0018]: unresolved, superiority not demonstrated; the long-window ratio 0.8790 [0.8557, 0.9027] is "
    "resolved. The historical <0.85 fresh-superiority gates stay failed where registered; the utility offered is "
    "approximately-current-quality 32k+MTP on 16 GB (explicit 32k+MTP passed under active ballast: dedicated peaks "
    "13,815.95 / 13,791.95 MiB for the g32 / g256 assistants; highest passing contexts 37,888 (fp8, next refusal "
    "38,912) and 47,104 (g256, next refusal 49,152), not exact maxima). Deployed with key-only FP8 global KV, FP8 "
    "ring, chunk 128, --draft-tokens 1. Not measured: your finetune (never reuse the diode matrices), untouched "
    "confirmation, human-rated dialogue, physical 16 GB throughput."
)
FORUM_F1_EVIDENCE = (
    "research-only (campaign F1, measured on the diode finetune; never a default). The FP8 assistant is embedded in "
    "the artifact but the generated launcher starts without MTP; the serve-<name>-mtp.cmd launcher beside it is the "
    "explicit opt-in override (native --spec mtp --draft-tokens 1 --allow-shared-memory: shared memory is system RAM "
    "over PCIe, not a resident fit). On the fresh forum instrument its KL ratio 0.847 missed the registered 0.85 "
    "gate on the plain control. Key-only/FP8-ring/chunk-128 serving reached the low-40k contexts under ballast "
    "(32768/36864/38912 passed, 39936 refused, MTP off). Not measured: your finetune, untouched confirmation, "
    "physical 16 GB throughput."
)
MIXED_B_EVIDENCE = (
    "research-only (campaign mixed-B, measured on the diode finetune; never a default). Exact artifact bytes "
    "20,139,576,324 (about 20.14 GB). On fresh held-out roleplay text its KL ratio to best-medium is 0.6196 "
    "[0.6061, 0.6324] (diode, default route; context-512 block KL), lower than mixed-A's 0.740; at the 65k "
    "instrument its KL is 0.06673 [0.06181, 0.07352] against best-medium's 0.08497. Clean 65k timing medians "
    "3109.60 PP / 76.14 TG tok/s (best-medium: 3380.24 / 95.01; mixed-A: 3233.83 / 83.49). Costs: +1.73 GB, two "
    "GPTQ passes, decode about -20 %. Not measured: long-context incremental-decode KL, more than one GPTQ pass "
    "per width, other GPUs."
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
    allocation: str = ""  # method "gguf": key into GGUF_ALLOCATIONS / GGUF_TENSOR_TYPES (the pinned per-tensor map)
    imatrix_corpus: str = "public"  # "roleplay": the pinned forum roleplay calibration corpus (required, never public)
    vocab_weights: str = "fp8"  # tied-vocabulary storage the converter is asked for (forum presets: q4_g32, absmax)
    assistant_encodings: tuple[str, ...] = ()  # nonempty: the preset's `--assistant-encoding` choices (fp8 default)
    launcher_no_mtp: bool = False  # embed the assistant but generate the launcher without MTP (+ an override launcher)
    extra_formats: tuple[str, ...] = ()  # further tensor formats the serving binary must know
    label: str = ""  # provenance label shown in the plan, table and summary (e.g. research-only, measured source)
    feedback: bool = False  # method "gguf": re-solve the projections with the registered X5 Hessian-feedback tool

    @property
    def formats(self) -> tuple[str, ...]:
        """Tensor formats the serving binary must know."""
        formats = (self.format_name,) + self.extra_formats
        return formats + (Q5_FORMAT,) if self.q5_units else formats


FEEDBACK_TOOLING_ENV = "INFERNIX_FEEDBACK_TOOLING"  # llama-local-tooling checkout with the X5 research tool
FEEDBACK_TOOL = Path("converter") / "gguf" / "iq_hessian_feedback.py"
FEEDBACK_SOLVER = Path("lib") / "iq_hessian_feedback.py"
FEEDBACK_STAGE0_EVIDENCE = Path("evidence") / "gemma4-x5-20261123" / "stage0-result.json"
# The registered stage1 policy: the deployed XXS role types and the damping each role's validation-only trials
# selected (2026-11-23). Frozen as measured; a user build never re-selects damping.
FEEDBACK_ROLE_TYPES = {"attn_q": "IQ2_S", "attn_k": "IQ2_S", "attn_v": "IQ3_XXS", "attn_output": "IQ3_S",
                       "ffn_gate": "IQ3_XXS", "ffn_up": "IQ3_XXS", "ffn_down": "IQ3_XXS"}
FEEDBACK_ROLE_DAMPING = {"attn_q": 0.01, "attn_k": 0.01, "attn_v": 0.03, "attn_output": 0.03,
                         "ffn_gate": 0.03, "ffn_up": 0.03, "ffn_down": 0.1}
# Same-bytes facts of the registered stage1 candidate (verified in the tests): the 410 projection payloads encode
# to exactly the deployed XXS trunk; the stock/candidate GGUF container adds the template's metadata overhead.
FEEDBACK_PROJECTION_BYTES = 11_087_462_400
FEEDBACK_REGISTERED_CONTAINER_BYTES = 11_103_270_944  # the diode campaign's observed container (metadata varies)
FEEDBACK_STAGE0_RESULT_SHA256 = "34be4a02d8479f5f0d63fd5bdbfd1364a2dece75f5a263637278ebe0b99e54d9"
FEEDBACK_SEED = 2026112305  # the campaign's registration seed: validation split, confirmation draw, torch seed
FEEDBACK_CONFIRMATION_SEED = "x5-untouched-confirmation-20261123-05"
FEEDBACK_CAPTURE_BATCH = 2
FEEDBACK_DAMPINGS = (0.01, 0.03, 0.1)  # the original screen grid; stage 1 froze one damping per role from it
FEEDBACK_IMATRIX_CORPORA = ("public", "roleplay")
X5_EVIDENCE = (
    "research-only (campaign X5 stage 1, measured on the diode finetune; never a default). The 410 projections "
    "carry the deployed XXS role types (Q/K IQ2_S, V/gate/up/down IQ3_XXS, O IQ3_S) and encode to exactly the "
    "same bytes (11,087,462,400 projection bytes; stock container 11,103,270,944 B on the campaign inputs); only "
    "the projection payload bytes differ from a stock own-imatrix XXS build, all formats/sizes/inventory "
    "identical, native A8 scoring, speed unchanged by construction (no timing claim). Paired thread-cluster 95 % "
    "intervals, candidate/deployed-KL ratio: RP-dev 0.4388 [0.4248, 0.4551], fresh 0.4463 [0.4225, 0.4714], "
    "65k-16 long 0.5263 [0.5024, 0.5502], untouched new short confirmation 0.4455 [0.4239, 0.4720] (upper < 0.85: "
    "confirmed at the historical strength). Measured with the generic templated English imatrix; the roleplay "
    "imatrix option of this preset is an UNMEASURED stack. The 32k+MTP memory anchors are measured on the F2 "
    "forum allocation, NOT on this container. Not measured: your finetune, human-rated dialogue, physical 16 GB "
    "throughput, an untouched long-context confirmation."
)
FEEDBACK_ROLEPLAY_NOTE = (
    "imatrix corpus roleplay: UNMEASURED stack — the campaign stage 1 measured the generic templated English "
    "imatrix; this option swaps only the importance matrix, everything else stays the registered recipe."
)
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
    "forum-iq3-16gb": Preset(
        "forum-iq3-16gb", "gguf", "IQ3_XXS base, measured forum allocation, roleplay imatrix (research-only)",
        "The measured 16 GB research recipe (campaign F2): deployed IQ3_XXS base with sliding-layer K at IQ3_XXS, "
        "V at IQ3_S and ffn_down of layers 20-39 at IQ3_S, tied q4_g32 (absmax) vocabulary, and an importance matrix "
        "from the pinned forum roleplay calibration corpus of your own model (required; never a public or diode "
        "matrix). FP8 assistant by default; --assistant-encoding q4-g32 or q4-g256 stores the draft smaller. "
        "Research-only, measured on the diode finetune, never a default.",
        "smallest", 12.63, 0.2273, "about 2 h", 0, False, "ggml_iq3_xxs", 100, 120, 13,
        {"rp-corpus": 4, "calibration-render": 1, "gguf-bf16": 15, "gguf-q8": 8, "imatrix": 25,
         "gguf-quantize": 45, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        "IQ3_XXS", allocation="forum-iq3-16gb", imatrix_corpus="roleplay", vocab_weights="q4_g32",
        assistant_encodings=("fp8", "q4-g32", "q4-g256"), extra_formats=("ggml_iq2_s", "ggml_iq3_s"),
        evidence=FORUM_F2_EVIDENCE, label="research-only; measured on the diode finetune; not a default",
    ),
    "forum-iq3-16gb-quality": Preset(
        "forum-iq3-16gb-quality", "gguf", "IQ3_XXS base, Q/K IQ3_XXS, V IQ3_S, staged down (research-only)",
        "The measured 16 GB quality research recipe (campaign F1): the same roleplay-imatrix forum family with V at "
        "IQ3_S, Q/K at IQ3_XXS in every layer, ffn_down of layers 20-39 at IQ3_S and of layers 40-59 at IQ4_XS, tied "
        "q4_g32 vocabulary, FP8 assistant embedded but the launcher generated without MTP (the documented override "
        "launcher beside it re-enables it with --spec mtp --draft-tokens 1 --allow-shared-memory). "
        "Research-only, measured on the diode finetune, never a default.",
        "smallest", 13.17, 0.2273, "about 2 h", 0, False, "ggml_iq3_xxs", 100, 120, 14,
        {"rp-corpus": 4, "calibration-render": 1, "gguf-bf16": 15, "gguf-q8": 8, "imatrix": 25,
         "gguf-quantize": 45, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        "IQ3_XXS", allocation="forum-iq3-16gb-quality", imatrix_corpus="roleplay", vocab_weights="q4_g32",
        launcher_no_mtp=True, extra_formats=("ggml_iq2_s", "ggml_iq3_s", "ggml_iq4_xs"),
        evidence=FORUM_F1_EVIDENCE, label="research-only; measured on the diode finetune; not a default",
    ),
    "iq3-xxs-feedback": Preset(
        "iq3-xxs-feedback", "gguf", "IQ3_XXS base, Hessian-feedback projections, frozen X5 damping (research-only)",
        "The registered X5 stage-1 recipe: the deployed IQ3_XXS role allocation with every one of the 410 "
        "projections re-solved by decoder-layer-sequential Hessian error feedback (FP64 covariance capture, the "
        "frozen per-role damping 0.01/0.03/0.1, exact IQ decode rounded BF16), packed same-bytes over the stock "
        "own-imatrix XXS container and imported with the ordinary IQ3 route. FP8 vocabulary and FP8 assistant by "
        "default; --assistant-encoding q4-g32 or q4-g256 stores the draft smaller. The importance matrix defaults "
        "to the campaign-faithful generic templated English corpus; --imatrix-corpus roleplay swaps in the pinned "
        "forum corpus of your own model (unmeasured stack). Needs the research tooling checkout "
        "(--feedback-tooling) and a GPU lease for the feedback pass. Research-only, measured on the diode "
        "finetune at XXS bytes, never a default.",
        "smallest", 13.02, 0.0933, "about 4 h", 0, False, "ggml_iq3_xxs", 170, 200, 55,
        {"calibration-source": 1, "rp-corpus": 4, "calibration-render": 1, "gguf-bf16": 15, "gguf-q8": 8,
         "imatrix": 25, "feedback-config": 1,
         "feedback-prepare": 10, "feedback-register": 0.5, "feedback-baseline": 60, "feedback-capture": 150,
         "feedback-pack": 10, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        "IQ3_XXS", allocation="xxs-feedback", vocab_weights="fp8",
        assistant_encodings=("fp8", "q4-g32", "q4-g256"), extra_formats=("ggml_iq2_s", "ggml_iq3_s"),
        evidence=X5_EVIDENCE,
        label="research-only; measured on the diode finetune at XXS bytes; confirmed at historical strength on "
              "the untouched confirmation draw; not a default",
        feedback=True,
    ),
    "q4q5-mixed-b": Preset(
        "q4q5-mixed-b", "gptq", "GPTQ g32 Q4 with GPTQ5 g64 Q5 `gate_up` (all layers) and `down@20-59` (research-only)",
        "Best-medium with the fused gate/up projections of every layer and the down projections of layers 20-59 at Q5 "
        "(about 1.73 GB larger); both GPTQ checkpoints use your own model's Hessians from the same calibration. "
        "A measured research option, never a default: on fresh held-out roleplay text its KL is 0.62x that of "
        "best-medium (diode, default route; context-512 block KL).",
        "medium+", 20.14, 0.0524, "about 70 min", 32, False, "q4_g32_fp16", 110, 140, 45,
        {"calibration-source": 1, "calibration-render": 1, "calibration-tokens": 1, "text-tree": 2, "gptq": 18,
         "gptq-q5": 25, "gptq-check": 1, "convert": 6, "receipt-check": 0.5, "finalize": 0.5},
        q5_units=("gate_up", "down@20-59"), evidence=MIXED_B_EVIDENCE,
        label="research-only; measured on the diode finetune; not a default",
    ),
}
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
    assistant_encoding: str = "fp8"  # presets with assistant_encodings: the assistant's stored encoding
    rp_raw: Path | None = None  # roleplay-imatrix presets: raw forum parquet directory (roleplay_heldout fetch)
    rp_against: tuple[Path, ...] = ()  # roleplay-imatrix presets: evaluation texts the corpus must be disjoint from
    feedback_tooling: Path | None = None  # iq3-xxs-feedback: llama-local-tooling checkout with the X5 research tool
    imatrix_corpus: str = ""  # iq3-xxs-feedback: "public" (campaign-faithful default) or "roleplay" (unmeasured)

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


# ---------------------------------------------------------------- allocation plan (exact byte arithmetic)

# Text-trunk shape of Gemma 4 31B (converter geometry): layer i is GLOBAL (full attention, K stored with Q, no V)
# when i % period == period - 1.
GGUF_ROLES = ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down")
GGUF_BLOCK_BYTES = {"IQ2_S": 82, "IQ3_XXS": 98, "IQ3_S": 110, "IQ4_XS": 136}  # bytes per 256 weights, QK columns 256


@dataclass(frozen=True)
class TextGeometry:
    n_layers: int
    hidden: int
    inter: int
    sliding: dict
    glob: dict
    period: int = 6

    def is_global(self, layer: int) -> bool:
        return layer % self.period == self.period - 1

    def shape(self, layer: int, role: str) -> tuple[int, int] | None:
        """(rows, cols) of the layer's tensor for `role`, or None when the layer has no such tensor."""
        rows = self.glob if self.is_global(layer) else self.sliding
        if role in ("attn_q", "attn_k"):
            return rows[role], self.hidden
        if role == "attn_v":
            return None if self.is_global(layer) else (rows[role], self.hidden)
        if role == "attn_output":
            return self.hidden, rows["attn_q"]
        if role in ("ffn_gate", "ffn_up"):
            return self.inter, self.hidden
        return self.hidden, self.inter


TEXT_GEOMETRY = TextGeometry(60, 5376, 21504, {"attn_q": 8192, "attn_k": 4096, "attn_v": 4096}, {"attn_q": 16384, "attn_k": 2048})
SLIDING_LAYERS = tuple(i for i in range(TEXT_GEOMETRY.n_layers) if not TEXT_GEOMETRY.is_global(i))
GLOBAL_LAYERS = tuple(i for i in range(TEXT_GEOMETRY.n_layers) if TEXT_GEOMETRY.is_global(i))

# llama-quantize --tensor-type pins (regex searched in the tensor name) of the measured IQ3 recipes: the per-tensor
# types a published imatrix IQ3_XS / IQ3_XXS of Gemma 4 31B uses where they differ from llama.cpp's own mix, plus the
# pinned per-tensor maps of the measured forum research allocations (the pins never overlap one another).
GGUF_TENSOR_TYPES = {
    "IQ3_XS": ("attn_output=IQ3_S", "attn_v=IQ3_S", "ffn_down=IQ3_S", r"blk\.([0-6]|5[2-9])\.ffn_(gate|up)=IQ3_S"),
    "IQ3_XXS": ("attn_q=IQ2_S", "attn_k=IQ2_S", "attn_output=IQ3_S"),
    "xxs-feedback": (  # the registered X5 stock pins: the deployed XXS mix, feedback re-solves the payloads
        "attn_q=IQ2_S", "attn_k=IQ2_S", "attn_output=IQ3_S",
    ),
    "forum-iq3-16gb": (
        "attn_q=IQ2_S",
        f"blk\\.({'|'.join(str(i) for i in GLOBAL_LAYERS)})\\.attn_k=IQ2_S",
        "attn_output=IQ3_S",
        f"blk\\.({'|'.join(str(i) for i in SLIDING_LAYERS)})\\.attn_k=IQ3_XXS",
        "attn_v=IQ3_S",
        r"blk\.(2[0-9]|3[0-9])\.ffn_down=IQ3_S",
    ),
    "forum-iq3-16gb-quality": (
        "attn_output=IQ3_S",
        "attn_q=IQ3_XXS",
        "attn_k=IQ3_XXS",
        "attn_v=IQ3_S",
        r"blk\.(2[0-9]|3[0-9])\.ffn_down=IQ3_S",
        r"blk\.(4[0-9]|5[0-9])\.ffn_down=IQ4_XS",
    ),
}

# The measured X2/F allocation specs (campaign registration): base recipe plus per-role/layer overrides.
ALLOCATION_BASE_TYPES = {  # the deployed IQ3_XXS recipe's own mix
    "attn_q": "IQ2_S", "attn_k": "IQ2_S", "attn_v": "IQ3_XXS", "attn_output": "IQ3_S",
    "ffn_gate": "IQ3_XXS", "ffn_up": "IQ3_XXS", "ffn_down": "IQ3_XXS",
}
GGUF_ALLOCATIONS = {
    "xxs-feedback": {  # the deployed IQ3_XXS recipe's own mix (the registered X5 stock allocation)
        "id": "xxs-feedback", "base": "xxs", "rules": [],
    },
    "forum-iq3-16gb": {
        "id": "forum-iq3-16gb", "base": "xxs",
        "rules": [
            {"roles": ["attn_k"], "layers": "sliding", "type": "IQ3_XXS"},
            {"roles": ["attn_v"], "layers": "all", "type": "IQ3_S"},
            {"roles": ["ffn_down"], "layers": "20-39", "type": "IQ3_S"},
        ],
    },
    "forum-iq3-16gb-quality": {
        "id": "forum-iq3-16gb-quality", "base": "xxs",
        "rules": [
            {"roles": ["attn_v"], "layers": "all", "type": "IQ3_S"},
            {"roles": ["attn_q", "attn_k"], "layers": "all", "type": "IQ3_XXS"},
            {"roles": ["ffn_down"], "layers": "20-39", "type": "IQ3_S"},
            {"roles": ["ffn_down"], "layers": "40-59", "type": "IQ4_XS"},
        ],
    },
}


def _parse_allocation_layers(selection: str) -> list[int]:
    allv = list(range(TEXT_GEOMETRY.n_layers))
    if selection == "all":
        return allv
    if selection == "sliding":
        return [i for i in allv if not TEXT_GEOMETRY.is_global(i)]
    if selection == "global":
        return [i for i in allv if TEXT_GEOMETRY.is_global(i)]
    out = []
    for part in selection.split(","):
        lo, _, hi = part.strip().partition("-")
        try:
            out += list(range(int(lo), int(hi or lo) + 1))
        except ValueError as error:
            raise ValueError(f"allocation layer selector {selection!r}: use all, sliding, global or 20-39 ranges") from error
    if any(not 0 <= i < TEXT_GEOMETRY.n_layers for i in out):
        raise ValueError(f"allocation layer selector {selection!r} names layers outside 0-{TEXT_GEOMETRY.n_layers - 1}")
    return out


def resolve_allocation(spec: dict) -> dict[tuple[int, str], str]:
    """Allocation spec -> {(layer, role): type}, validated (global layers have no V; fused objects share a type)."""
    alloc = {
        (i, r): ALLOCATION_BASE_TYPES[r]
        for i in range(TEXT_GEOMETRY.n_layers) for r in GGUF_ROLES if TEXT_GEOMETRY.shape(i, r)
    }
    for rule in spec.get("rules", []):
        typ, roles, layers = rule["type"], rule["roles"], _parse_allocation_layers(rule.get("layers", "all"))
        if typ not in GGUF_BLOCK_BYTES:
            raise ValueError(f"allocation {spec['id']}: type {typ!r} is not one of {tuple(GGUF_BLOCK_BYTES)}")
        for i in layers:
            for role in roles:
                if TEXT_GEOMETRY.shape(i, role) is None:
                    if rule.get("layers", "all") not in ("all", "sliding", "global"):
                        raise ValueError(f"allocation {spec['id']}: layer {i} has no {role} tensor")
                    continue
                alloc[(i, role)] = typ
    for i in range(TEXT_GEOMETRY.n_layers):
        if alloc[(i, "ffn_gate")] != alloc[(i, "ffn_up")]:
            raise ValueError(f"allocation {spec['id']}: layer {i} gate/up are one fused object and must share a type")
        if TEXT_GEOMETRY.is_global(i) and alloc[(i, "attn_q")] != alloc[(i, "attn_k")]:
            raise ValueError(f"allocation {spec['id']}: global layer {i} Q/K are one fused object and must share a type")
    return alloc


def _tensor_bytes(shape: tuple[int, int], typ: str) -> int:
    rows, cols = shape
    return rows * (cols // 256) * GGUF_BLOCK_BYTES[typ]


def allocation_trunk_bytes(spec: dict) -> int:
    """Exact trunk bytes of an allocation (the compose plan arithmetic, verified against the campaign artifacts)."""
    alloc = resolve_allocation(spec)
    return sum(_tensor_bytes(TEXT_GEOMETRY.shape(i, r), t) for (i, r), t in alloc.items())


TRUNK_PARAMS = sum(
    shape[0] * shape[1]
    for i in range(TEXT_GEOMETRY.n_layers) for role in GGUF_ROLES
    if (shape := TEXT_GEOMETRY.shape(i, role))
)
Q4_G32_BYTES_PER_256, Q5_G64_BYTES_PER_256 = 144, 168  # native groupwise storage, exact binary16 scales
VOCAB_BYTES = {"fp8": 1_409_810_432, "q4_g32": 792_723_456}  # the tied 262,144 x 5,376 table
ASSISTANT_BYTES = {"fp8": 486_806_032, "q4_g32": 287_871_504, "q4_g256": 263_098_896}  # whole assistant payload
ASSISTANT_ENCODING_ALIASES = {"fp8": "fp8", "q4-g32": "q4_g32", "q4-g256": "q4_g256"}
ASSISTANT_ENCODING_FLAGS = {
    "fp8": (),
    "q4-g32": ("--mtp-weights", "q4", "--mtp-head", "q4_g32"),
    "q4-g256": ("--mtp-weights", "q4_g256", "--mtp-head", "q4_g256"),
}
FIXED_PAYLOAD_BYTES = 2_663_152 + 32_192_698 + 225_280 + 15_946  # norms, tokenizer, header

_Q5_UNIT_SHAPES = {"o": ("attn_output",), "gate_up": ("ffn_gate", "ffn_up"), "down": ("ffn_down",)}


def _q5_unit_params(unit: str) -> int:
    """Weight count of one `--q5-unit` selector such as `gate_up` or `down@20-59` (what moves from Q4 to Q5)."""
    role, _, selection = unit.partition("@")
    if role not in _Q5_UNIT_SHAPES:
        raise ValueError(f"q5 unit {unit!r}: use o, gate_up or down[@LAYERS]")
    layers = _parse_allocation_layers(selection) if selection else list(range(TEXT_GEOMETRY.n_layers))
    total = 0
    for i in layers:
        for name in _Q5_UNIT_SHAPES[role]:
            shape = TEXT_GEOMETRY.shape(i, name)
            if shape:
                total += shape[0] * shape[1]
    return total


def preset_file_bytes(preset: Preset, assistant_encoding: str = "fp8") -> int:
    """Exact planned artifact bytes (text + MTP) of a preset with an analytic plan; None-safe plans only."""
    if preset.method == "gguf" and preset.allocation:
        assistant = ASSISTANT_BYTES[ASSISTANT_ENCODING_ALIASES[assistant_encoding]] if assistant_encoding else 0
        return allocation_trunk_bytes(GGUF_ALLOCATIONS[preset.allocation]) + VOCAB_BYTES[preset.vocab_weights] \
            + FIXED_PAYLOAD_BYTES + assistant
    if preset.method == "gptq" and preset.q5_units:
        moved = sum(_q5_unit_params(unit) for unit in preset.q5_units)
        return TRUNK_PARAMS * Q4_G32_BYTES_PER_256 // 256 + moved * (Q5_G64_BYTES_PER_256 - Q4_G32_BYTES_PER_256) // 256 \
            + VOCAB_BYTES["fp8"] + ASSISTANT_BYTES["fp8"] + FIXED_PAYLOAD_BYTES
    if preset.method == "gptq" and preset.bits == 5:
        return TRUNK_PARAMS * Q5_G64_BYTES_PER_256 // 256 + VOCAB_BYTES["fp8"] + ASSISTANT_BYTES["fp8"] + FIXED_PAYLOAD_BYTES
    if preset.method == "gptq":
        return TRUNK_PARAMS * Q4_G32_BYTES_PER_256 // 256 + VOCAB_BYTES["fp8"] + ASSISTANT_BYTES["fp8"] + FIXED_PAYLOAD_BYTES
    raise ValueError(f"preset {preset.id} has no analytic byte plan (the quantizer's own mix decides the bytes)")


# ---------------------------------------------------------------- bundled recommendations
# Design labels (quality / balanced / context per card class), NOT new CLI profiles: each bundle resolves to a
# concrete preset + serve profile + explicit flag combination with its measured numbers. Explicit operator flags
# always win. Key-only KV stays explicit (the memory-16gb profile owns it); MTP embedding (assistant encoding) is
# separate from MTP enable (the launcher/runtime). Numbers are measured campaign anchors, not capacity promises.
PROFILE_BUNDLES = {
    "16gb": {
        "quality": {
            "preset": "forum-iq3-16gb-quality", "profile": "memory-16gb", "assistant_encoding": None,
            "flags": ["--text-only"], "context": 32768, "max_context": 38912,
            "measured": "F1: fresh forum KL 0.847 vs plain control (0.85 gate missed); key-only/FP8-ring/chunk-128 "
                        "passed 32768/36864/38912, refused 39936 (MTP off) under ballast; launcher starts without "
                        "MTP, override launcher enables it",
            "note": "research-only; measured on the diode finetune",
        },
        "balanced": {
            "preset": "forum-iq3-16gb", "profile": "memory-16gb", "assistant_encoding": "q4-g32",
            "flags": ["--assistant-encoding", "q4-g32", "--text-only"], "context": 32768, "max_context": 32768,
            "measured": "F2 + Q4g32 assistant: explicit 32k+MTP passed under ballast (dedicated peak 13,815.95 MiB); "
                        "fresh forum KL ratio 0.9707 [0.9415, 1.0018] unresolved vs deployed XXS; g32 maximum "
                        "context unmeasured",
            "note": "research-only; measured on the diode finetune",
        },
        "context": {
            "preset": "forum-iq3-16gb", "profile": "memory-16gb", "assistant_encoding": "q4-g256",
            "flags": ["--assistant-encoding", "q4-g256", "--text-only"], "context": 32768, "max_context": 47104,
            "measured": "F2 + Q4g256 assistant: explicit 32k+MTP passed under ballast (dedicated peak 13,791.95 "
                        "MiB); highest passing context 47,104, next refusal 49,152 (not an exact maximum); "
                        "acceptance trade versus g32 (-3.24 pp)",
            "note": "research-only; measured on the diode finetune",
        },
    },
    "24gb": {
        "context": {
            "preset": "best-small", "profile": "default", "assistant_encoding": None, "flags": [],
            "context": 65536, "max_context": 65536,
            "measured": "ballast: best-small FP8 KV/BF16 ring 91.4k with MTP; the 65k target fits with margin",
        },
        "balanced": {
            "preset": "best-medium", "profile": "default", "assistant_encoding": None, "flags": [],
            "context": 65536, "max_context": 65536,
            "measured": "ballast: best-medium FP8 ring 71.2k with MTP",
        },
        "quality": {
            "preset": "q4q5-mixed-a", "profile": "default", "assistant_encoding": None,
            "flags": ["--gemma4-key-only-kv"], "context": 65536, "max_context": 65536,
            "measured": "ballast: mixed-A key-only/BF16 ring 67.6k; an explicit capacity trade, never a default",
        },
    },
    "32gb": {
        "context": {
            "preset": "best-medium", "profile": "default", "assistant_encoding": None, "flags": [],
            "context": 131072, "max_context": 131072,
            "measured": "physical 5090: best-medium FP8 KV/BF16 ring 131k with MTP",
        },
        "balanced": {
            "preset": "best-medium", "profile": "default", "assistant_encoding": None, "flags": [],
            "context": 131072, "max_context": 131072,
            "measured": "the medium preset at the full 131k target (historical BF16-ring capacity anchor)",
        },
        "quality": {
            "preset": "q4q5-mixed-a", "profile": "default", "assistant_encoding": None,
            "flags": ["--gemma4-key-only-kv"], "context": 131072, "max_context": 131072,
            "measured": "mixed-A historically reaches 131k with FP8 ring (now the launcher default) or key-only "
                        "globals (explicit); q5-g64 used key-only to meet 131k with the study margin",
        },
    },
}
CARD_CLASSES = ("16gb", "24gb", "32gb")
BUNDLE_NAMES = ("quality", "balanced", "context")


def resolve_bundle(card: str, bundle: str, requested_context: int | str | None = None) -> dict:
    """Resolve a recommendation label to its concrete combination; refuses impossible explicit contexts."""
    combos = PROFILE_BUNDLES.get(card)
    if combos is None:
        raise ValueError(f"card class {card!r}: use one of {CARD_CLASSES} (16/24/32 GB, 1.9 GiB overhead assumed)")
    combo = combos.get(bundle)
    if combo is None:
        raise ValueError(f"bundle {bundle!r}: use one of {BUNDLE_NAMES}")
    requested = combo["context"] if requested_context is None else requested_context
    if requested != "auto" and requested > combo["max_context"]:
        raise ValueError(
            f"{card} {bundle} bundle: --context {requested} exceeds the measured anchor {combo['max_context']} "
            f"({combo['measured']}). The server would refuse or silently spill; choose a smaller explicit context "
            "or 'auto' and read the resolved limit from /v1/models."
        )
    return {**combo, "card": card, "bundle": bundle,
            "requested_context": requested_context, "resolved_context": requested}


def llama_tool(cfg: BuildConfig, name: str) -> str:
    return str(Path(cfg.llama_cpp or ".") / (f"{name}.exe" if os.name == "nt" else name))


def gguf_source_steps(cfg: BuildConfig, templated: Path, bf16: Path, q8: Path, imatrix: Path,
                      *, prune_q8: bool) -> list[Step]:
    """BF16 GGUF, Q8_0 copy for the importance matrix, own imatrix on the rendered calibration text."""
    minutes = cfg.preset.minutes
    return [
        Step("gguf-bf16", "Write a BF16 GGUF of the model with llama.cpp's convert_hf_to_gguf.py (about 62 GB)",
             minutes["gguf-bf16"],
             [cfg.python, str(Path(cfg.llama_cpp_source or ".") / "convert_hf_to_gguf.py"), str(cfg.model),
              "--outtype", "bf16", "--outfile", str(bf16)],
             outputs=(bf16,)),
        Step("gguf-q8", "Quantize a Q8_0 copy for the importance matrix (about 33 GB)", minutes["gguf-q8"],
             [llama_tool(cfg, "llama-quantize"), str(bf16), str(q8), "Q8_0"], outputs=(q8,)),
        Step("imatrix", f"Collect the importance matrix on the {CALIBRATION_CHUNKS} rendered chunks (GPU)",
             minutes["imatrix"],
             [llama_tool(cfg, "llama-imatrix"), "-m", str(q8), "-o", str(imatrix), "--output-format", "gguf",
              "--no-ppl", "--ctx-size", str(CALIBRATION_CONTEXT), "--gpu-layers", "999", "--file", str(templated),
              "--chunks", str(CALIBRATION_CHUNKS), "--parse-special"],
             outputs=(imatrix,), prune_always=(q8,) if prune_q8 else ()),
    ]


def feedback_tooling_paths(cfg: BuildConfig) -> tuple[Path, Path, Path]:
    """(tool script, solver module, registered stage0 result) of the research tooling checkout."""
    root = cfg.feedback_tooling
    if root is None:
        raise ValueError(f"preset {cfg.preset.id} needs the research tooling checkout: pass --feedback-tooling "
                         f"<llama-local-tooling> or set {FEEDBACK_TOOLING_ENV}")
    return root / FEEDBACK_TOOL, root / FEEDBACK_SOLVER, root / FEEDBACK_STAGE0_EVIDENCE


def codec_library_path(cfg: BuildConfig) -> Path:
    """The ggml library the research solver binds ggml_quantize_chunk from (beside the llama.cpp binaries)."""
    name = "ggml-base.dll" if os.name == "nt" else "libggml-base.so"
    return Path(cfg.llama_cpp or ".") / name


def feedback_steps(cfg: BuildConfig, templated: Path) -> list[Step]:
    """The registered X5 chain: config/registration, stock parity baseline, feedback, same-byte pack.

    The research tool (`<tooling>/converter/gguf/iq_hessian_feedback.py`, explicit JSON config) keeps doing the
    work: full-baseline writes the lossless BF16 projection GGUF and the stock own-imatrix XXS container with
    whole/slab parity on all 410 tensors, full-feedback re-solves every projection with FP64 Hessians collected
    decoder-layer-sequentially under the frozen per-role damping (this is the preset's Hessian-capture stage), and
    full-pack copies the stock container replacing only the same-type/shape/byte projection payloads.
    """
    preset, minutes = cfg.preset, cfg.preset.minutes
    root = cfg.work("feedback")
    stage1 = root / "stage1"
    config = root / "config.json"
    quantized = stage1 / "feedback-XXS.gguf"
    raw = stage1 / "raw"
    tool, solver, stage0 = feedback_tooling_paths(cfg)
    gpy = str(cfg.gptq_python) if cfg.gptq_python else sys.executable
    tool_argv = lambda command: [gpy, str(tool), command, "--config", str(config)]  # noqa: E731
    return gguf_source_steps(cfg, templated, cfg.work("model-bf16.gguf"), cfg.work("model-q8_0.gguf"),
                             cfg.work("imatrix.gguf"), prune_q8=False) + [
        Step("feedback-config", "Write the research tool's config, the frozen stage1 policy and the registration "
             "corpus manifest (the registered stage0 gate is copied unchanged)", minutes["feedback-config"],
             call=write_feedback_config, outputs=(config, root / "stage1-policy.json", root / "stage0-result.json",
                                                  root / "dev-corpus" / "manifest.json",
                                                  root / "calibration-metadata.json"),
             prune_always=(cfg.work("model-bf16.gguf"),),
             note="the roleplay validation/confirmation corpus is built from --rp-raw; the damping policy is the "
                  "campaign-frozen stage1 policy and is never re-selected"),
        Step("feedback-prepare", "Register the full-model run (identities, tokens, untouched confirmation draw)",
             minutes["feedback-prepare"], tool_argv("prepare"),
             outputs=(root / "registration.json", root / "tokens.safetensors")),
        Step("feedback-register", "Pin the execution layout: tool/solver SHAs, metadata template and config hash",
             minutes["feedback-register"], call=register_feedback_amendment,
             outputs=(root / "implementation-amendment.json",)),
        Step("feedback-baseline", "Stock own-imatrix XXS quantize with whole/slab parity on all 410 tensors "
             "(CPU, the parity anchor)", minutes["feedback-baseline"], tool_argv("full-baseline"),
             outputs=(stage1 / "baseline.json", stage1 / "source-BF16.gguf", stage1 / "stock-XXS.gguf"),
             prune_always=(stage1 / "source-BF16.gguf", cfg.work("model-q8_0.gguf"))),
        Step("feedback-capture", "Hessian feedback: FP64 covariance capture and re-solve of every projection, "
             "decoder-layer-sequential, frozen damping (GPU, the long step)", minutes["feedback-capture"],
             tool_argv("full-feedback"), outputs=(stage1 / "feedback.json", raw),
             env={"PYTHONUTF8": "1"}),
        Step("feedback-pack", "Pack the feedback payloads into a byte-identical copy of the stock container",
             minutes["feedback-pack"], tool_argv("full-pack"),
             outputs=(quantized, stage1 / "candidate.json"), prune_always=(raw,)),
    ]


def gguf_steps(cfg: BuildConfig, templated: Path, quantized: Path) -> list[Step]:
    """BF16 GGUF, Q8_0 copy for the importance matrix, own imatrix on the rendered calibration text, quantize."""
    preset = cfg.preset
    bf16, q8, imatrix = cfg.work("model-bf16.gguf"), cfg.work("model-q8_0.gguf"), cfg.work("imatrix.gguf")
    quantize = llama_tool(cfg, "llama-quantize")
    pins = [arg for pin in GGUF_TENSOR_TYPES[cfg.preset.allocation or cfg.preset.gguf_type] for arg in ("--tensor-type", pin)]
    steps = gguf_source_steps(cfg, templated, bf16, q8, imatrix, prune_q8=True)
    steps.append(
        Step("gguf-quantize", f"Quantize to {preset.gguf_type} with the importance matrix (CPU, the long step)",
             preset.minutes["gguf-quantize"],
             [quantize, "--imatrix", str(imatrix), *pins, str(bf16), str(quantized), preset.gguf_type],
             outputs=(quantized,), prune_always=(bf16,)))
    return steps


def write_feedback_config(builder: "Builder") -> None:
    """Write the research tool's explicit JSON config and every input it registers from."""
    cfg = builder.cfg
    tool, solver, stage0 = feedback_tooling_paths(cfg)
    for path in (tool, solver, stage0):
        if not path.is_file():
            raise RuntimeError(f"research tooling file missing: {path} (--feedback-tooling {cfg.feedback_tooling})")
    root = cfg.work("feedback")
    (root / "stage1").mkdir(parents=True, exist_ok=True)
    templated = cfg.work("calibration", "templated.txt")
    rendered = read_json(templated.with_suffix(".json"))  # the calibration-render receipt: prefix ids and stride
    shutil.copyfile(stage0, root / "stage0-result.json")
    if sha256_file(root / "stage0-result.json") != FEEDBACK_STAGE0_RESULT_SHA256:
        raise RuntimeError(f"the registered stage0 gate changed: {stage0} does not match the frozen sha256")
    write_json(root / "stage1-policy.json", {
        "schema": "infernix.gemma4-feedback-policy/v1",
        "role_types": dict(FEEDBACK_ROLE_TYPES), "role_damping": dict(FEEDBACK_ROLE_DAMPING),
        "stage0_result_sha256": FEEDBACK_STAGE0_RESULT_SHA256,
        "damping_source": "X5 stage0 validation-only selection, frozen 2026-11-23; a build never re-selects damping",
    })
    rp_dir = cfg.work("calibration", "rp-corpus")
    provenance = read_json(rp_dir / "provenance.json")
    streams = [{"id": record["id"], "path": str(rp_dir / "data" / f"{record['id']}.txt")}
               for record in provenance["excerpts"]]
    if len(streams) < 32:
        raise RuntimeError(f"the roleplay corpus has {len(streams)} threads; registration needs at least 32 "
                           "(raise the corpus quotas or add raw forum files)")
    dev = root / "dev-corpus"
    dev.mkdir(exist_ok=True)
    write_json(dev / "manifest.json", {"schema": "infernix.gemma4-feedback-devcorpus/v1", "streams": streams})
    write_json(root / "calibration-metadata.json", {
        "source": str(templated), "prefix_ids": rendered["prefix_ids"], "stride_tokens": rendered["stride_tokens"],
    })
    write_json(root / "config.json", {
        "out": str(root), "source": str(cfg.model), "infernix": str(REPO_ROOT),
        "imatrix": str(cfg.work("imatrix.gguf")), "codec_library": str(codec_library_path(cfg)),
        "quantizer": llama_tool(cfg, "llama-quantize"),
        "calibration_metadata": str(root / "calibration-metadata.json"),
        "development_corpus": str(dev), "raw": str(cfg.rp_raw),
        "against": [str(path) for path in cfg.rp_against],
        "seed": FEEDBACK_SEED, "confirmation_seed": FEEDBACK_CONFIRMATION_SEED,
        "capture_batch": FEEDBACK_CAPTURE_BATCH, "dampings": list(FEEDBACK_DAMPINGS),
        "hessian_dtype": "float64", "prior_thread_records": [],
        "minimum_free_gib": {cfg.work_dir.anchor: 100, "C:/": 10, "D:/": 100},
        "stage0_result": "stage0-result.json", "stage1_policy": str(root / "stage1-policy.json"),
        "stage1_directory": "stage1",
    })
    builder.log(f"wrote the feedback config ({cfg.imatrix_corpus or 'public'} imatrix; frozen stage1 policy, "
                f"FP64 Hessians, seed {FEEDBACK_SEED})")


def register_feedback_amendment(builder: "Builder") -> None:
    """Pin the execution layout the tool's registration check enforces: tool/solver SHAs, template, config hash."""
    cfg = builder.cfg
    tool, solver, _ = feedback_tooling_paths(cfg)
    template = cfg.work("model-q8_0.gguf")
    if not template.is_file():
        raise RuntimeError(f"the Q8_0 metadata template is missing: {template}")
    config = cfg.work("feedback", "config.json")
    write_json(cfg.work("feedback", "implementation-amendment.json"), {
        "schema": "infernix.gemma4-feedback-amendment/v1",
        "reason": "builder preset iq3-xxs-feedback pins the execution layout of the registered X5 tool chain",
        "driver_sha256": sha256_file(tool), "solver_sha256": sha256_file(solver),
        "metadata_template": str(template),
        "active_config_sha256": sha256_file(config),
        "additional_input_sha256": {str(template): sha256_file(template)},
    })
    builder.log(f"pinned the feedback execution: tool {sha256_file(tool)[:12]}..., config "
                f"{sha256_file(config)[:12]}..., template {template.name}")


def convert_argv(cfg: BuildConfig, out: Path, *, scoring: bool, checkpoint: Path | None) -> list[str]:
    """`gemma4_q4 convert` command; the scoring twin is always text-only and without MTP."""
    argv = [cfg.python, "-m", "tools.convert.gemma4_q4", "convert", "--model", str(cfg.model)]
    argv += q4_convert_flags(cfg, checkpoint)
    if scoring or not cfg.with_vision:
        argv.append("--text-only")
    if not scoring and cfg.assistant is not None:
        argv += ["--mtp-assistant", str(cfg.assistant)]
        if cfg.preset.assistant_encodings:
            argv += list(ASSISTANT_ENCODING_FLAGS[cfg.assistant_encoding])
    else:
        argv.append("--no-mtp")
    if cfg.preset.vocab_weights != "fp8":
        argv += ["--vocab-weights", cfg.preset.vocab_weights, "--vocab-scale-search", "absmax"]
    argv += ["--name", cfg.name, "--out", str(out)]
    return argv


def build_plan(cfg: BuildConfig) -> list[Step]:
    """The ordered steps with their exact commands. Pure: touches no file."""
    preset = cfg.preset
    minutes = preset.minutes
    templated = cfg.work("calibration", "templated.txt")
    source = cfg.calibration_text or cfg.work("calibration", "source.txt")
    tokenizer = cfg.model / "tokenizer.json"
    checkpoint = (cfg.work("feedback", "stage1", "feedback-XXS.gguf") if preset.feedback
                  else cfg.work(f"model-{preset.gguf_type}.gguf")) if preset.method == "gguf" else cfg.work("gptq")
    tree = cfg.work("text60")
    steps: list[Step] = []

    if preset.method != "plain":
        roleplay = cfg.imatrix_corpus == "roleplay" if preset.feedback else preset.imatrix_corpus == "roleplay"
        # The feedback preset always builds the registration corpus: validation windows and the untouched
        # confirmation draw never reuse public or diode data, whatever the imatrix corpus is.
        if roleplay or preset.feedback:
            corpus = cfg.work("calibration", "rp-corpus")
            source = corpus / RP_SOURCE_NAME
            steps.append(Step(
                "rp-corpus", "Build the pinned roleplay calibration corpus from the raw forum files (required: "
                "never a public or diode matrix)", minutes["rp-corpus"],
                [cfg.python, "-m", "tools.eval.roleplay_calibration", "build", "--raw", str(cfg.rp_raw), "--out",
                 str(corpus)]
                + [flag for path in cfg.rp_against for flag in ("--against", str(path))]
                + ["--tokenizer", str(tokenizer), "--chunks", str(CALIBRATION_CHUNKS), "--ctx-size",
                   str(CALIBRATION_CONTEXT), "--prefix-tokens", "17"],
                outputs=(source, corpus / "provenance.json"),
                note=f"raw forum parquet files must exist in {cfg.rp_raw} (`roleplay_heldout fetch`); the corpus is "
                     "shingle-checked disjoint from every --rp-against evaluation text"
                     + ("; it is also the imatrix corpus" if roleplay else
                        "; the imatrix itself comes from the corpus chosen by --imatrix-corpus (public default)"),
            ))
        if not roleplay and cfg.calibration_text is None:
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
        steps += feedback_steps(cfg, templated) if preset.feedback else gguf_steps(cfg, templated, checkpoint)
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
    outputs = (
        cfg.out_dir / "BUILD-SUMMARY.md",
        cfg.out_dir / f"{cfg.name}.gemma4-build.json",
        cfg.out_dir / f"serve-{cfg.name}.cmd",
    )
    if cfg.preset.launcher_no_mtp:
        outputs += (cfg.out_dir / f"serve-{cfg.name}-mtp.cmd",)
    return outputs


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
        + ("; supported option, not the default: roleplay KL 0.74x best-medium on fresh held-out text" if preset.id == "q4q5-mixed-a" else ""),
        f"            {preset.meaning}",
        *((f"            {preset.label}",) if preset.label else ()),
        f"Model       {cfg.model}",
        f"MTP         {cfg.assistant if cfg.assistant else 'none (--no-mtp)'}",
        f"Vision      {'kept (--with-vision)' if cfg.with_vision else 'left out (text only)'}",
        *((f"Imatrix     {cfg.imatrix_corpus or 'public'} corpus — "
           + (FEEDBACK_ROLEPLAY_NOTE if cfg.imatrix_corpus == "roleplay" else
              "the campaign-faithful default (generic templated English imatrix)"),)
          if preset.feedback else ()),
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
    if cfg.preset.feedback:
        check_feedback_receipt(builder, size)


def check_feedback_receipt(builder: "Builder", artifact_size: int) -> None:
    """The registered same-bytes facts: 410 projections at exactly the deployed XXS trunk bytes, planned size."""
    cfg = builder.cfg
    feedback = read_json(cfg.work("feedback", "stage1", "feedback.json"))
    candidate = read_json(cfg.work("feedback", "stage1", "candidate.json"))
    policy = read_json(cfg.work("feedback", "stage1-policy.json"))
    if len(feedback["tensors"]) != 410 or feedback["encoded_bytes"] != FEEDBACK_PROJECTION_BYTES:
        raise RuntimeError(f"the feedback payload inventory is not the registered same-bytes XXS trunk "
                           f"({len(feedback['tensors'])} tensors, {feedback['encoded_bytes']} bytes; want 410 and "
                           f"{FEEDBACK_PROJECTION_BYTES})")
    stock = cfg.work("feedback", "stage1", "stock-XXS.gguf")
    if not candidate.get("same_types_shapes_bytes") or candidate["bytes"] != stock.stat().st_size:
        raise RuntimeError("the packed candidate is not byte-identical to the stock container")
    planned = preset_file_bytes(cfg.preset, cfg.assistant_encoding)
    if artifact_size != planned:
        raise RuntimeError(f"{cfg.artifact} is {artifact_size} bytes; the preset plan says {planned}")
    if any(policy["role_types"][role] != feedback_role for role, feedback_role in FEEDBACK_ROLE_TYPES.items()):
        raise RuntimeError("the packed role types differ from the deployed XXS allocation")
    builder.log(f"feedback receipt ok: 410 projections, {FEEDBACK_PROJECTION_BYTES} projection bytes, candidate "
                f"{candidate['bytes']} bytes (registered container {FEEDBACK_REGISTERED_CONTAINER_BYTES} on the "
                "campaign inputs)")


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
         f"GPTQ5 g64 Q5 (A16Only, choose the Q5 prefill route at launch) for {', '.join(p['q5_units'])}"
         + (" in the named layers; " if any("@" in u for u in p["q5_units"]) else " in every layer; ")
         if p.get("q5_units") else "- Weights: W4A8 (4-bit weights, FP8 activations while reading the prompt); ")
        + f"{'with the MTP draft assistant (speculative decoding)' if info['mtp'] else 'no MTP assistant'}; "
        f"{'with the vision tower' if info['vision'] else 'text only'}.",
        (f"- Assistant encoding: {info['assistant_encoding']} (the converter's --mtp-weights/--mtp-head recipe); "
         "FP8 is the default." if info.get("assistant_encoding") else ""),
        (f"- Tied vocabulary: {p['vocab_weights']} with absmax scale search (the converter's --vocab-weights "
         "--vocab-scale-search recipe)." if p.get("vocab_weights", "fp8") != "fp8" else ""),
        (f"- {p['label']}" if p.get("label") else ""),
        f"- Historical hint (English, text only, RTX 5090): mean KL {p['kl']:.4f} from "
        + ("the plain model's own Q8_0 reference, A16 compute" if p["bits"] == 5 else
           "the plain model's own Q8_0 reference, default route with int8-g64 Q5 prefill" if p.get("q5_units") else
           "the model's own Q8_0 reference" if p["method"] == "gguf" else "the near-lossless FP8-row reference")
        + ". This is not a quality measurement of your build.",
    ] + ([
        f"- Supported option, not the default (best-medium stays the general default): {p['evidence']}",
    ] if p.get("evidence") and not p.get("label") else []) + ([
        f"- {p['evidence']}",
    ] if p.get("evidence") and p.get("label") else []) + ([
        "",
        "## Roleplay calibration provenance",
        "",
        f"- Corpus: `{info['roleplay_calibration']['corpus_id']}` built by `tools/eval.roleplay_calibration` from "
        "your own model's raw forum files (the diode matrices are never reused).",
        f"- Corpus sha256: `{info['roleplay_calibration']['corpus_sha256']}`; selection seed "
        f"`{info['roleplay_calibration']['selection']['seed']}`; "
        f"{info['roleplay_calibration']['totals']['excerpts']} excerpts.",
    ] if info.get("roleplay_calibration") else []) + ([
        "",
        "## Hessian-feedback provenance (research-only)",
        "",
        f"- Registered X5 recipe; imatrix corpus: `{info['hessian_feedback']['imatrix_corpus']}`"
        + (" (UNMEASURED stack — the campaign measured the generic templated English imatrix)."
           if info['hessian_feedback']['imatrix_corpus'] == "roleplay" else " (the campaign-faithful default)."),
        f"- Importance matrix sha256: `{info['hessian_feedback']['imatrix_sha256']}`.",
        f"- Frozen role damping: {info['hessian_feedback']['role_damping']} (validation-only selection, never "
        "re-selected); stage0 gate sha256 `" + info['hessian_feedback']['stage0_result_sha256'][:12] + "...`.",
        f"- Hessians: FP64 covariance capture, decoder-layer-sequential; projection bytes "
        f"{info['hessian_feedback']['projection_bytes']:,} (same-bytes XXS trunk, 410 projections).",
        f"- Candidate container sha256: `{info['hessian_feedback']['candidate_sha256']}` "
        f"({info['hessian_feedback']['candidate_bytes']:,} bytes; the campaign's registered container was "
        f"{info['hessian_feedback']['registered_container_bytes']:,} B on the campaign inputs).",
        f"- Research tool: `{info['hessian_feedback']['tool']}` (driver sha256 "
        f"`{info['hessian_feedback']['tool_sha256'][:12]}...`, solver `{info['hessian_feedback']['solver_sha256'][:12]}...`).",
    ] if info.get("hessian_feedback") else []) + [
        "",
        "## Files",
        "",
        f"- Artifact: `{info['artifact']}`",
        f"- Size: {info['file_bytes']:,} bytes ({info['file_bytes'] / 1e9:.2f} GB)",
        f"- sha256: `{info['sha256']}`",
        f"- Converter receipt: `{info['artifact']}.build.json`",
        f"- Launcher: `{info['launcher']}`" if info["launcher"] else "- Launcher: not generated (--no-launcher)",
        (f"- MTP override launcher: `{info['launchers']['mtp_override']}` (starts with `--spec mtp --draft-tokens 1 "
         "--allow-shared-memory`; the main launcher starts without MTP)" if info["launchers"].get("mtp_override")
         else ""),
        "",
        "## Serve it",
        "",
        f"The launcher runs `infernix-serve` with {serve['kv'].upper()} KV cache and {serve['qk'].upper()} attention "
        f"QK, {serve['sliding_kv'].upper()} sliding K/V storage, one lane, a host-only prefix cache and BF16-rounded logits:",
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
                server, cfg.artifact, settings, mtp=info_artifact.mtp and not cfg.preset.launcher_no_mtp,
                vision=info_artifact.vision,
            )
        )
    roleplay = None
    if (cfg.imatrix_corpus or cfg.preset.imatrix_corpus) == "roleplay":
        provenance_path = cfg.work("calibration", "rp-corpus", "provenance.json")
        if provenance_path.is_file():
            try:
                roleplay = read_json(provenance_path)
            except ValueError:
                roleplay = None
    launchers = {"main": str(launcher_path) if settings.launcher and server is not None else None}
    if cfg.preset.launcher_no_mtp and settings.launcher and server is not None:
        launchers["mtp_override"] = str(cfg.out_dir / f"serve-{cfg.name}-mtp.cmd")
    hessian = None
    if cfg.preset.feedback:
        root = cfg.work("feedback")
        candidate = read_json(root / "stage1" / "candidate.json")
        registration = read_json(root / "registration.json")
        policy = read_json(root / "stage1-policy.json")
        amendment = read_json(root / "implementation-amendment.json")
        hessian = {
            "provenance": "registered X5 stage-1 recipe (research-only; measured on the diode finetune at XXS "
                          "bytes; not a default)",
            "imatrix_corpus": cfg.imatrix_corpus or "public",
            "imatrix_sha256": registration["input_sha256"].get(str(cfg.work("imatrix.gguf"))),
            "role_types": policy["role_types"], "role_damping": policy["role_damping"],
            "damping_source": policy["damping_source"], "stage0_result_sha256": policy["stage0_result_sha256"],
            "hessian_dtype": "float64", "seed": registration["config"]["seed"],
            "confirmation_seed": registration["config"]["confirmation_seed"],
            "tool": str(feedback_tooling_paths(cfg)[0]), "tool_sha256": amendment["driver_sha256"],
            "solver_sha256": amendment["solver_sha256"], "config_sha256": amendment["active_config_sha256"],
            "candidate_sha256": candidate["sha256"], "candidate_bytes": candidate["bytes"],
            "stock_sha256": candidate["stock_sha256"], "projections": candidate["projections"],
            "projection_bytes": candidate["encoded_bytes"],
            "registered_container_bytes": FEEDBACK_REGISTERED_CONTAINER_BYTES,
        }
    return {
        "name": cfg.name,
        "preset": {k: getattr(cfg.preset, k)
                   for k in ("id", "summary", "meaning", "size_class", "kl", "method", "bits", "q5_units", "evidence",
                             "label", "allocation", "vocab_weights", "assistant_encodings", "imatrix_corpus",
                             "feedback")},
        "assistant_encoding": cfg.assistant_encoding if cfg.preset.assistant_encodings else None,
        "roleplay_calibration": roleplay,
        "hessian_feedback": hessian,
        "mtp": info_artifact.mtp,
        "vision": info_artifact.vision,
        "artifact": str(cfg.artifact),
        "file_bytes": receipt["artifact"]["file_bytes"],
        "sha256": receipt["artifact"]["file_sha256"][cfg.artifact.name],
        "launcher": launchers["main"],
        "launchers": launchers,
        "serve_command": serve,
        "serve": {"kv": settings.kv_dtype, "qk": settings.gemma4_qk, "sliding_kv": settings.gemma4_sliding_kv},
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
        info = launcher_tools.write_launcher(
            cfg.out_dir / f"serve-{cfg.name}.cmd", cfg.artifact, builder.server_exe, settings,
            cuda_bin=settings.cuda_bin, info=builder.read_artifact_info(), overwrite=True,
            no_mtp=cfg.preset.launcher_no_mtp,
        )
        if cfg.preset.launcher_no_mtp and info.mtp:
            # The explicit opt-in override: the same settings with the assistant enabled and the shortfall allowed
            # in shared GPU memory (native --spec mtp --draft-tokens 1 --allow-shared-memory).
            launcher_tools.write_launcher(
                cfg.out_dir / f"serve-{cfg.name}-mtp.cmd", cfg.artifact, builder.server_exe,
                dataclasses_replace(settings, allow_shared_memory=True),
                cuda_bin=settings.cuda_bin, info=info, overwrite=True,
            )
            builder.log("wrote the MTP-off launcher and the explicit serve-<name>-mtp.cmd override launcher")
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
    elif (cfg.imatrix_corpus or cfg.preset.imatrix_corpus) == "roleplay":
        corpus_text = cfg.work("calibration", "rp-corpus", RP_SOURCE_NAME)
        calibration = {
            "corpus": "tools.eval.roleplay_calibration (the pinned forum roleplay calibration; never a public or "
                      "diode matrix)",
            "raw": str(cfg.rp_raw.resolve()) if cfg.rp_raw else None,
            "against": sorted(str(path.resolve()) for path in cfg.rp_against),
            "text_sha256": _hash_or_none(corpus_text),
        }
    elif cfg.calibration_text is not None:
        calibration = {"file": str(cfg.calibration_text.resolve()), "sha256": _hash_or_none(cfg.calibration_text)}
    else:
        calibration = {"source": f"{CALIBRATION_REPO}@{CALIBRATION_REVISION}", "text_sha256": CALIBRATION_TEXT_SHA256}
    identity = {
        "preset": cfg.preset.id,
        "model": str(model.resolve()),
        "model_files": files,
        "assistant": str(cfg.assistant.resolve()) if cfg.assistant is not None else None,
        "assistant_files": assistant,
        "with_vision": cfg.with_vision,
        "name": cfg.name,
        "out_dir": str(cfg.out_dir.resolve()),
        "calibration": calibration,
        **({"assistant_encoding": cfg.assistant_encoding} if cfg.preset.assistant_encodings else {}),
        **({"vocab_weights": cfg.preset.vocab_weights} if cfg.preset.vocab_weights != "fp8" else {}),
        **({"gguf_recipe": {"type": cfg.preset.gguf_type,
                            "allocation": cfg.preset.allocation or None,
                            "tensor_types": GGUF_TENSOR_TYPES[cfg.preset.allocation or cfg.preset.gguf_type]},
            "llama_cpp": str(cfg.llama_cpp.resolve()) if cfg.llama_cpp else None,
            "llama_cpp_source": str(cfg.llama_cpp_source.resolve()) if cfg.llama_cpp_source else None}
           if cfg.preset.method == "gguf" else {}),
        **({"feedback_tooling": str(cfg.feedback_tooling.resolve()) if cfg.feedback_tooling else None,
            "feedback_tool_sha256": _hash_or_none(feedback_tooling_paths(cfg)[0]) if cfg.feedback_tooling else None,
            "feedback_solver_sha256": _hash_or_none(feedback_tooling_paths(cfg)[1]) if cfg.feedback_tooling else None,
            "imatrix_corpus": cfg.imatrix_corpus or "public",
            "registration_corpus": {"raw": str(cfg.rp_raw.resolve()) if cfg.rp_raw else None,
                                    "against": sorted(str(path.resolve()) for path in cfg.rp_against)}}
           if cfg.preset.feedback else {}),
    }
    return identity


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
        if step.id == "finalize":
            args = [*args, "--gemma4-sliding-kv", self.serve.gemma4_sliding_kv]
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
            "gemma4_sliding_kv": self.gemma4_sliding_kv,
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
