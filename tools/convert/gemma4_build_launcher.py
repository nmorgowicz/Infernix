"""Windows launcher for `infernix-serve` on a Gemma 4 31B artifact, the server binary search, and binary checks.

The default `ServeProfile` is the operator roleplay profile: 131,072 context (alias `131k`), FP8 KV cache, FP8
attention QK, one lane, `--prefill-chunk auto`, MTP with one draft token when the artifact carries the assistant,
BF16-rounded logits, a host-only prefix cache (no device snapshot slots, 8192 MiB host context), any model name
accepted, listening on `0.0.0.0:8001`. Every field can be changed; `--gemma4-qk bf16` omits the QK flag so the
engine default applies. This module imports nothing from the other `gemma4_build*` modules.
"""

from __future__ import annotations

from dataclasses import dataclass
import glob
import mmap
import os
from pathlib import Path
import re

SERVER_NAME = "infernix-serve"
PERPLEXITY_NAME = "infernix-perplexity"
DEFAULT_CONTEXT = 131072
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
SERVER_DEFAULT_HOST = "127.0.0.1"  # what infernix-serve binds when a launcher passes no --host
SERVER_DEFAULT_PORT = 8080
KV_DTYPES = ("fp8", "bf16")
MODEL_MATCHES = ("any", "exact")  # "any" emits --any-model-name; "exact" emits nothing (the server default)
DEFAULT_MODEL_MATCH = "any"
GEMMA4_QKS = ("bf16", "fp8")
DEFAULT_GEMMA4_QK = "fp8"
DEFAULT_PREFILL_CHUNK = "auto"
LOGITS_DTYPES = ("bf16", "fp32")
DEFAULT_LOGITS_DTYPE = "bf16"
DEFAULT_DRAFT_TOKENS = "1"
DEFAULT_HOST_CONTEXT_MIB = 8192
DEFAULT_VRAM_HEADROOM_MIB = 1024
SMALL_VRAM_HEADROOM_MIB = 64  # memory-16gb default; any value >= 0 (docs/weight-conversion.md, 16 GB profile)
SERVE_PROFILES = ("default", "memory-16gb")
MAX_TOP_K = 64  # Gemma4's sampler ceiling (Qwen stays at 20)
REASONING_LOOPS = ("stop", "conclude")  # --reasoning-loop; "off" is the server default and emits no flag
# Libraries infernix binaries load from their own directory: FFmpeg (vcpkg) and the staged cuBLAS.
REQUIRED_DLLS = (
    "avcodec-*.dll",
    "avformat-*.dll",
    "avutil-*.dll",
    "swscale-*.dll",
    "swresample-*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
)
_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.:-]*")


@dataclass(frozen=True, slots=True)
class ArtifactInfo:
    """What the launcher needs to know about a `.ninfer` file."""

    mtp: bool
    vision: bool
    max_context: int
    iq3_xxs: bool = False


@dataclass(frozen=True, slots=True)
class ServeProfile:
    """The `infernix-serve` settings a launcher decides; `alias` None means `<context/1000>k`."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    alias: str | None = None
    context: int | str = DEFAULT_CONTEXT
    kv_dtype: str = "fp8"
    model_match: str = DEFAULT_MODEL_MATCH
    gemma4_qk: str = DEFAULT_GEMMA4_QK
    prefill_chunk: str = DEFAULT_PREFILL_CHUNK
    draft_tokens: str = DEFAULT_DRAFT_TOKENS
    logits_dtype: str = DEFAULT_LOGITS_DTYPE
    host_context_mib: int = DEFAULT_HOST_CONTEXT_MIB
    vram_headroom_mib: int = DEFAULT_VRAM_HEADROOM_MIB
    top_k: int | None = None  # None: no flag, the artifact's sampling preset applies
    chat_template: Path | None = None  # runtime override of the artifact's template (--chat-template)
    # Thinking control (none by default: the server's answer room already ends a long thought, and the loop guard is
    # off): --default-thinking-budget N, --thinking-budget-message TEXT, --reasoning-loop stop|conclude.
    default_thinking_budget: int | None = None
    thinking_budget_message: str | None = None
    reasoning_loop: str | None = None
    gemma4_key_only_kv: bool = False
    gemma4_sliding_kv: str = "bf16"
    gemma4_q5_prefill: str = "a16"
    mtp_fallback: bool = False  # memory-16gb: plan with MTP first, start without it when only that does not fit
    allow_shared_memory: bool = False  # --allow-shared-memory: start with the shortfall in shared GPU memory (slow)

    @property
    def model_id(self) -> str:
        return self.alias or ("gemma4-16gb" if self.context == "auto" else default_alias(self.context))


def exe_name(base: str) -> str:
    return base + (".exe" if os.name == "nt" else "")


def cmd_path(value: Path) -> str:
    """Absolute path safe inside a double-quoted CMD word; refuses quotes and line breaks like the other tools."""
    text = str(Path(value).resolve())
    if any(char in text for char in ('"', "\r", "\n")):
        raise ValueError(f"CMD paths cannot contain quotes or line breaks: {text!r}")
    return text.replace("%", "%%")


def default_alias(context: int) -> str:
    return f"{round(context / 1000)}k"


def check_serve_settings(profile: ServeProfile) -> None:
    """Refuse settings the server would reject or a CMD line cannot carry; every message names the flag."""
    if not _HOST.fullmatch(profile.host):
        raise ValueError(
            f"--host {profile.host!r}: use an IPv4/IPv6 address or host name (letters, digits, '.', ':', '-')"
        )
    if not 1 <= profile.port <= 65535:
        raise ValueError("--port must be 1..65535")
    if not _ALIAS.fullmatch(profile.model_id):
        raise ValueError(
            f"--alias {profile.model_id!r}: use letters, digits, '.', '_' and '-' only (no spaces, quotes or %)"
        )
    if profile.context != "auto" and (not isinstance(profile.context, int) or profile.context < 1):
        raise ValueError("--context must be positive or 'auto'")
    if profile.kv_dtype not in KV_DTYPES:
        raise ValueError(f"--kv-dtype must be one of {KV_DTYPES} (fp8 is the qualified setting)")
    if profile.model_match not in MODEL_MATCHES:
        raise ValueError(f"--model-match must be one of {MODEL_MATCHES}")
    if profile.gemma4_key_only_kv and profile.kv_dtype != "fp8":
        raise ValueError("--gemma4-key-only-kv needs --kv-dtype fp8")
    if profile.gemma4_sliding_kv not in ("bf16", "fp8"):
        raise ValueError("--gemma4-sliding-kv must be bf16 or fp8")
    if profile.gemma4_q5_prefill not in ("a16", "fp8", "int8-g32", "int8-g64"):
        raise ValueError("--gemma4-q5-prefill must be a16, fp8, int8-g32 or int8-g64")
    if profile.gemma4_sliding_kv == "fp8" and profile.kv_dtype != "fp8":
        raise ValueError("--gemma4-sliding-kv fp8 needs --kv-dtype fp8")
    if profile.gemma4_qk not in GEMMA4_QKS:
        raise ValueError(f"--gemma4-qk must be bf16 or fp8, got {profile.gemma4_qk!r}")
    if profile.gemma4_qk == "fp8" and profile.kv_dtype != "fp8":
        raise ValueError("--gemma4-qk fp8 needs --kv-dtype fp8 (the server rejects it with "
                         f"{profile.kv_dtype} KV); drop --gemma4-qk or use --kv-dtype fp8")
    chunk = profile.prefill_chunk
    if chunk != "auto" and not (chunk.isdigit() and 128 <= int(chunk) <= 16384 and int(chunk) % 128 == 0):
        raise ValueError(f"--prefill-chunk must be 'auto' or a multiple of 128 in 128..16384, got {chunk!r}")
    if profile.draft_tokens != "auto" and profile.draft_tokens not in tuple(str(n) for n in range(1, 8)):
        raise ValueError(f"--draft-tokens must be 'auto' or 1..7, got {profile.draft_tokens!r}")
    if profile.logits_dtype not in LOGITS_DTYPES:
        raise ValueError(f"--logits-dtype must be one of {LOGITS_DTYPES}")
    if profile.host_context_mib < 1:
        raise ValueError("--host-context-mib must be positive (the Gemma prefix cache is host-only)")
    if profile.vram_headroom_mib < 0:
        raise ValueError("--vram-headroom-mib must not be negative")
    if profile.top_k is not None and not 0 <= profile.top_k <= MAX_TOP_K:
        raise ValueError(f"--top-k must be 0..{MAX_TOP_K} for Gemma, got {profile.top_k}")
    if profile.default_thinking_budget is not None and profile.default_thinking_budget < 1:
        raise ValueError("--default-thinking-budget must be positive")
    message = profile.thinking_budget_message
    if message is not None and (not message.strip() or any(char in message for char in ('"', "%", "\r", "\n"))):
        raise ValueError("--thinking-budget-message must be non-empty text without quotes, '%' or line breaks")
    if profile.reasoning_loop is not None and profile.reasoning_loop not in REASONING_LOOPS:
        raise ValueError(f"--reasoning-loop must be one of {REASONING_LOOPS} (omit it for off)")


def serve_groups(
    profile: ServeProfile, *, mtp: bool, vision: bool, chat_template: str | None = None, shared_memory: bool = False
) -> list[list[str]]:
    """The `infernix-serve` flags (after the artifact path), grouped one launcher line each.

    `chat_template` is the template path as it must appear in the command (a `cmd_path` string for CMD lines);
    it defaults to `profile.chat_template`.
    """
    if chat_template is None and profile.chat_template is not None:
        chat_template = str(profile.chat_template)
    groups = [
        ["--host", profile.host, "--port", str(profile.port), "--model-id", profile.model_id]
        + (["--any-model-name"] if profile.model_match == "any" else []),
        ["--max-context", str(profile.context), "--kv-capacity", "auto", "--vram-headroom-mib",
         str(profile.vram_headroom_mib), "--kv-dtype", profile.kv_dtype]
        + (["--gemma4-key-only-kv"] if profile.gemma4_key_only_kv else [])
        + (["--gemma4-sliding-kv", "fp8"] if profile.gemma4_sliding_kv == "fp8" else [])
        + (["--gemma4-q5-prefill", profile.gemma4_q5_prefill] if profile.gemma4_q5_prefill != "a16" else [])
        + (["--gemma4-qk", "fp8"] if profile.gemma4_qk == "fp8" else []),
        ["--max-concurrency", "1", "--prefill-chunk", profile.prefill_chunk, "--logits-dtype", profile.logits_dtype]
        + (["--top-k", str(profile.top_k)] if profile.top_k is not None else [])
        + (["--chat-template", chat_template] if chat_template else []),
    ]
    thinking = (
        (["--default-thinking-budget", str(profile.default_thinking_budget)]
         if profile.default_thinking_budget is not None else [])
        + (["--thinking-budget-message", profile.thinking_budget_message]
           if profile.thinking_budget_message is not None else [])
        + (["--reasoning-loop", profile.reasoning_loop] if profile.reasoning_loop is not None else [])
    )
    if thinking:
        groups.append(thinking)
    if mtp:
        groups.append(["--spec", "mtp", "--draft-tokens", profile.draft_tokens])
    groups.append(["--device-snapshot-slots", "0", "--host-context-mib", str(profile.host_context_mib)]
                  + (["--allow-shared-memory"] if shared_memory else [])
                  + (["--vision"] if vision else []))
    return groups


def serve_argv(server: Path, artifact: Path, profile: ServeProfile, *, mtp: bool, vision: bool) -> list[str]:
    """Flat argv of the server command, for the build summary."""
    return [str(server), str(artifact)] + [
        flag for group in serve_groups(profile, mtp=mtp, vision=vision) for flag in group
    ]


def _cmd_word(word: str) -> str:
    return f'"{word}"' if any(char.isspace() or char in "&|<>^()" for char in word) else word


def _command_lines(server: str, artifact: str, groups: list[list[str]], suffix: str = "") -> list[str]:
    lines = [f'"{server}" "{artifact}" ^']
    for index, group in enumerate(groups):
        last = index == len(groups) - 1
        lines.append("  " + " ".join(_cmd_word(word) for word in group) + (suffix if last else " ^"))
    return lines


def launcher_lines(
    server: str,
    artifact: str,
    profile: ServeProfile,
    *,
    mtp: bool,
    vision: bool,
    cuda_bin: str | None = None,
    chat_template: str | None = None,
) -> list[str]:
    """CMD lines; `server`, `artifact`, `cuda_bin` and `chat_template` are already `cmd_path` strings."""
    check_serve_settings(profile)
    # The memory profile asks for shared GPU memory only after the resident starts (below) do not fit.
    staged = profile.allow_shared_memory and profile.mtp_fallback
    groups = serve_groups(profile, mtp=mtp, vision=vision, chat_template=chat_template,
                          shared_memory=profile.allow_shared_memory and not staged)
    kv = "FP8" if profile.kv_dtype == "fp8" else "BF16"
    qk = "FP8" if profile.gemma4_qk == "fp8" else "BF16"
    lines = [
        "@echo off",
        "setlocal DisableDelayedExpansion",
        f"rem Generated by python -m tools.convert.gemma4_build for infernix-serve; Gemma 4 31B profile: {kv} KV "
        f"cache, {qk} attention QK" + ("; MTP assistant enabled." if mtp else "; no MTP assistant in this artifact."),
        "rem The server accepts any API key (none is configured). "
        + (
            "--any-model-name: OpenAI requests may send any model name."
            if profile.model_match == "any"
            else "OpenAI endpoints answer only to --model-id."
        ),
        "rem Prefix cache is host-only: no device snapshot slots, "
        f"{profile.host_context_mib} MiB of pinned host context.",
    ]
    if profile.gemma4_key_only_kv:
        lines.append("rem Experimental key-only global FP8 KV. Default launches do not enable it.")
    if profile.gemma4_sliding_kv == "fp8":
        lines.append("rem Experimental FP8 sliding K/V ring: lower memory, measurable fidelity and prefill cost; memory-16gb default, explicit bf16 overrides it.")
    if profile.gemma4_q5_prefill != "a16":
        lines.append(f"rem Experimental Q5 prefill {profile.gemma4_q5_prefill} (lossy, opt-in): calls of 64+ columns; narrow decode/MTP verification stays A16. Artifact unchanged.")
    if profile.context == "auto":
        lines.append("rem Context is fitted at startup to available VRAM; refuses below 9216 tokens. Read /v1/models for the resolved limit.")
    if profile.gemma4_qk == "fp8":
        lines.append("rem Gemma attention QK in FP8: faster long prompts, measurably lower fidelity "
                     "(infernix docs/serving.md, --gemma4-qk).")
    if cuda_bin:
        lines.append(f'set "PATH={cuda_bin};%PATH%"')
    if staged:
        plan = '"%TEMP%\\infernix-memory-16gb-plan.json"'
        lines.append("rem Shared GPU memory was allowed when this launcher was generated: a start that fits resident is never spilled. The server plans (no GPU work) with")
        lines.append("rem the MTP assistant, then without it, and only when neither fits starts without it WITH shared GPU memory")
        lines.append("rem (system RAM over PCIe: much slower). Without the option those cases are refused.")
        if mtp:
            lines.extend(_command_lines(server, artifact, groups, f" --plan {plan} >nul 2>nul"))
            lines.append("if errorlevel 1 goto :no_mtp")
            lines.extend(_command_lines(server, artifact, groups))
            lines.append("exit /b %errorlevel%")
            lines.append(":no_mtp")
            groups = serve_groups(profile, mtp=False, vision=vision, chat_template=chat_template)
        lines.extend(_command_lines(server, artifact, groups, f" --plan {plan} >nul 2>nul"))
        lines.append("if errorlevel 1 goto :shared_memory")
        lines.extend(_command_lines(server, artifact, groups))
        lines.append("exit /b %errorlevel%")
        lines.append(":shared_memory")
        lines.append("echo WARNING: the model does not fit the free VRAM; starting with shared GPU memory (much slower). 1>&2")
        groups = serve_groups(profile, mtp=False, vision=vision, chat_template=chat_template, shared_memory=True)
    elif mtp and profile.mtp_fallback:
        plan = f'"%TEMP%\\infernix-memory-16gb-plan.json"'
        lines.append("rem The MTP assistant needs about 464 MiB more VRAM: the server plans with it first (no GPU work) and")
        lines.append("rem starts without it when only that fits. Neither fitting is the server's refusal; nothing is paged.")
        lines.extend(_command_lines(server, artifact, groups, f" --plan {plan} >nul 2>nul"))
        lines.append("if errorlevel 1 goto :no_mtp")
        lines.extend(_command_lines(server, artifact, groups))
        lines.append("exit /b %errorlevel%")
        lines.append(":no_mtp")
        lines.append("echo The MTP assistant does not fit in the free VRAM; starting without it. 1>&2")
        groups = serve_groups(profile, mtp=False, vision=vision, chat_template=chat_template)
    lines.extend(_command_lines(server, artifact, groups))
    lines.extend(["exit /b %errorlevel%", ""])
    return lines


def artifact_info(path: Path) -> ArtifactInfo:
    """Components and context limit of a Gemma 4 artifact (reads the directory only)."""
    from tools.artifact.reader import Artifact

    with Artifact(path) as artifact:
        components = artifact.directory.components
        config = components["text"]["config"]
        if config.get("architectures") != ["Gemma4ForCausalLM"]:
            raise ValueError(f"{path} is not a Gemma 4 text artifact")
        return ArtifactInfo(
            mtp="mtp" in components,
            vision="vision" in components,
            iq3_xxs=sum(obj.bytes for obj in artifact.objects if getattr(obj, "format", None) == "ggml_iq3_xxs")
            > sum(obj.bytes for obj in artifact.objects if getattr(obj, "format", None) == "ggml_iq3_s"),
            max_context=int(config["max_position_embeddings"]),
        )


def write_launcher(
    out: Path,
    artifact: Path,
    server: Path,
    profile: ServeProfile | None = None,
    *,
    no_mtp: bool = False,
    cuda_bin: Path | None = None,
    info: ArtifactInfo | None = None,
    overwrite: bool = False,
) -> ArtifactInfo:
    """Write `out` (CRLF, refuses an existing file unless `overwrite`) and return what it detected."""
    profile = profile or ServeProfile()
    if not server.is_file():
        raise ValueError(f"server binary not found: {server}")
    if not artifact.is_file():
        raise ValueError(f"artifact not found: {artifact}")
    if profile.chat_template is not None and not profile.chat_template.is_file():
        raise ValueError(f"--chat-template file not found: {profile.chat_template}")
    if out.exists() and not overwrite:
        raise FileExistsError(f"Launcher already exists: {out}")
    info = info or artifact_info(artifact)
    if profile.context != "auto" and profile.context > info.max_context:
        raise ValueError(f"--context {profile.context} exceeds the artifact's position limit {info.max_context}")
    lines = launcher_lines(
        cmd_path(server),
        cmd_path(artifact),
        profile,
        mtp=info.mtp and not no_mtp,
        vision=info.vision,
        cuda_bin=cmd_path(cuda_bin) if cuda_bin else None,
        chat_template=cmd_path(profile.chat_template) if profile.chat_template else None,
    )
    if any(ord(char) > 127 for line in lines for char in line):
        lines.insert(1, "chcp 65001 >nul")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\r\n".join(lines), encoding="utf-8", newline="")
    return info


def server_candidates(repo_root: Path) -> list[Path]:
    """`infernix-serve` builds of a checkout, newest first: `apps/Release` and `build*/apps/Release`."""
    exe = exe_name(SERVER_NAME)
    patterns = [str(repo_root / "apps" / "Release" / exe), str(repo_root / "build*" / "apps" / "Release" / exe)]
    found = {Path(item) for pattern in patterns for item in glob.glob(pattern)}
    return sorted(found, key=lambda path: path.stat().st_mtime, reverse=True)


def scan_binary(path: Path, needles: tuple[str, ...]) -> dict[str, bool]:
    """Whether each ASCII needle occurs in the binary (memory-mapped, read-only)."""
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return {needle: data.find(needle.encode("ascii")) >= 0 for needle in needles}


def find_server_exe(repo_root: Path, required: tuple[str, ...] = ()) -> Path | None:
    """The newest server build that contains every `required` format name; else the newest build; else None."""
    candidates = server_candidates(repo_root)
    for path in candidates:
        if all(scan_binary(path, required).values()):
            return path
    return candidates[0] if candidates else None


def missing_dlls(exe: Path) -> list[str]:
    """DLL patterns found neither next to `exe` nor in a PATH directory."""
    folders = [exe.parent] + [Path(item) for item in os.environ.get("PATH", "").split(os.pathsep) if item]
    missing = []
    for pattern in REQUIRED_DLLS:
        if not any(glob.glob(str(folder / pattern)) for folder in folders):
            missing.append(pattern)
    return missing


def sibling_binary(server: Path, base: str) -> Path | None:
    path = server.parent / exe_name(base)
    return path if path.is_file() else None
