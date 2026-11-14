"""Preflight for `gemma4_build`: probes of the machine and checks of every input, each failure with its next action.

`Probes` is the only code that looks at the machine (nvidia-smi, RAM, disk, other Python environments, git, server
binaries), so the tests replace it with a fake. Nothing here writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

from . import gemma4_build_launcher as launcher_tools
from .gemma4_build import (
    CALIBRATION_CHUNKS,
    GPTQ_PYTHON_ENV,
    REPO_ROOT,
    SCORE_EXTRA_GB,
    BuildConfig,
    ServeSettings,
    device_of,
    gb,
    load_state,
)

GIB = 2**30
GPTQ_PINS = ("torch", "transformers", "llmcompressor", "compressed-tensors")
GPTQ_VRAM_GIB = 12
GPTQ_RAM_AVAILABLE_GIB = 40  # measured host RSS peak with --max-cpu-gb 32
GPTQ_RAM_TOTAL_GIB = 64
EXPECTED_TEXT = {"hidden_size": 5376, "num_hidden_layers": 60}
SETUP_HINT = "python -m tools.convert.gemma4_build setup-gptq"


@dataclass(frozen=True, slots=True)
class Finding:
    level: str  # "ok", "warn" or "fail"
    what: str
    detail: str = ""
    action: str = ""

    def render(self) -> str:
        head = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}[self.level]
        text = f"{head} {self.what}" + (f": {self.detail}" if self.detail else "")
        return text + (f"\n         -> {self.action}" if self.action else "")


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    total_mib: int
    free_mib: int
    compute_cap: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    tool_versions: dict[str, str] = field(default_factory=dict)

    def add(self, level: str, what: str, detail: str = "", action: str = "") -> None:
        self.findings.append(Finding(level, what, detail, action))

    @property
    def failed(self) -> list[Finding]:
        return [item for item in self.findings if item.level == "fail"]

    def render(self) -> list[str]:
        return [item.render() for item in self.findings]


PROBE_CODE = (
    "import importlib.metadata as m, json, sys\n"
    "names, cuda = sys.argv[2:], sys.argv[1] == '1'\n"
    "out = {}\n"
    "for n in names:\n"
    "    try: out[n] = m.version(n)\n"
    "    except m.PackageNotFoundError: out[n] = None\n"
    "info = {'python': '.'.join(map(str, sys.version_info[:3])), 'packages': out}\n"
    "if cuda:\n"
    "    import torch\n"
    "    info['cuda'] = bool(torch.cuda.is_available())\n"
    "print(json.dumps(info))\n"
)


class Probes:
    """Real probes; every method is read-only."""

    def gpu(self) -> GpuInfo | None:
        exe = shutil.which("nvidia-smi")
        if exe is None:
            return None
        query = "name,memory.total,memory.free,compute_cap"
        try:
            result = subprocess.run([exe, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                                    capture_output=True, text=True, timeout=30, check=True)
            name, total, free, cap = [item.strip() for item in result.stdout.splitlines()[0].split(",")]
            return GpuInfo(name, int(float(total)), int(float(free)), cap)
        except (subprocess.SubprocessError, OSError, ValueError, IndexError):
            return None

    def ram(self) -> tuple[int, int]:
        """(total, available) bytes."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            return memory.total, memory.available
        except ImportError:
            pass
        if os.name == "nt":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong), ("page_total", ctypes.c_ulonglong),
                            ("page_avail", ctypes.c_ulonglong), ("virt_total", ctypes.c_ulonglong),
                            ("virt_avail", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]

            status = Status()
            status.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return status.total, status.avail
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        return int(info["MemTotal"].split()[0]) * 1024, int(info["MemAvailable"].split()[0]) * 1024

    def disk_free(self, path: Path) -> int:
        probe = path
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        return shutil.disk_usage(probe).free

    def device_of(self, path: Path) -> int:
        return device_of(path)

    def python_info(self, python: str, packages: tuple[str, ...], cuda: bool = False) -> dict | None:
        try:
            result = subprocess.run([python, "-c", PROBE_CODE, "1" if cuda else "0", *packages],
                                    capture_output=True, text=True, timeout=180)
            if result.returncode != 0:
                return {"error": (result.stderr.strip().splitlines() or ["failed"])[-1]}
            return json.loads(result.stdout.strip().splitlines()[-1])
        except (subprocess.SubprocessError, OSError, ValueError):
            return None

    def git_state(self) -> tuple[str, list[str]]:
        env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # `git status` must not touch the index in a dry run
        revision = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True, env=env
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked-files=no"], text=True, env=env
        )
        return revision, [line for line in dirty.splitlines() if line.strip()]

    def tool_version(self, exe: Path) -> str:
        try:
            result = subprocess.run([str(exe), "--version"], capture_output=True, text=True, timeout=60)
            lines = (result.stdout + result.stderr).splitlines()
            return next((line.strip() for line in lines if "version" in line.lower()), "unknown")
        except (subprocess.SubprocessError, OSError):
            return "unknown"

    def scan_binary(self, path: Path, needles: tuple[str, ...]) -> dict[str, bool]:
        return launcher_tools.scan_binary(path, needles)

    def missing_dlls(self, exe: Path) -> list[str]:
        return launcher_tools.missing_dlls(exe)


# ---------------------------------------------------------------- inputs


def safetensors_dtypes(path: Path) -> set[str]:
    """Tensor dtypes named in a safetensors header (reads the header only)."""
    with path.open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 64 * 2**20:
            raise ValueError(f"{path.name}: implausible safetensors header")
        header = json.loads(stream.read(length))
    return {item["dtype"] for name, item in header.items() if name != "__metadata__"}


def inspect_model(model: Path) -> dict:
    """Refuse anything but a BF16 Gemma 4 31B dense checkpoint; returns facts for the report."""
    config_path = model / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{model} has no config.json")
    from .gemma4_q4 import normalize_config

    config = normalize_config(json.loads(config_path.read_text(encoding="utf-8")))
    text = config.get("text_config", {})
    if config.get("architectures") != ["Gemma4ForConditionalGeneration"]:
        raise ValueError(f"architectures is {config.get('architectures')!r}, expected Gemma4ForConditionalGeneration")
    for key, expected in EXPECTED_TEXT.items():
        if text.get(key) != expected:
            raise ValueError(f"text_config.{key} is {text.get(key)!r}; Gemma 4 31B has {expected}")
    if text.get("enable_moe_block"):
        raise ValueError("this is a mixture-of-experts Gemma 4 (26B-A4B); only the dense 31B is supported")
    try:
        from .gemma4 import text_config
    except ImportError as error:
        raise ValueError(f"cannot import the converter's checks ({error}); install torch for {sys.executable}") from error
    text_config(config)
    index_path = model / "model.safetensors.index.json"
    if index_path.is_file():
        shards = sorted(set(json.loads(index_path.read_text(encoding="utf-8"))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    paths = [model / name for name in shards]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"missing weight files: {', '.join(missing[:3])}{' ...' if len(missing) > 3 else ''}")
    dtypes = set().union(*(safetensors_dtypes(path) for path in paths))
    if not dtypes <= {"BF16", "F32"}:
        raise ValueError(f"weights are {sorted(dtypes)}; the converter needs BF16 (F32 scalars allowed)")
    if "BF16" not in dtypes:
        raise ValueError("no BF16 weights found")
    return {"shards": len(paths), "bytes": sum(path.stat().st_size for path in paths),
            "vision": bool(config.get("vision_config"))}


def versions_text(packages: dict[str, str | None]) -> str:
    return ", ".join(f"{name} {version or 'MISSING'}" for name, version in packages.items())


def parse_pins(requirements: Path, names: tuple[str, ...]) -> dict[str, str]:
    pins = {}
    for line in requirements.read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match and match.group(1).lower() in names:
            pins[match.group(1).lower()] = match.group(2)
    return pins


# ---------------------------------------------------------------- the checks


def check_disk(report: Report, cfg: BuildConfig, probes) -> None:
    preset = cfg.preset
    extra = SCORE_EXTRA_GB if cfg.score else 0
    out = preset.file_gb + 2
    residual = preset.residual_gb
    same = probes.device_of(cfg.out_dir) == probes.device_of(cfg.work_dir)
    work_hard, work_soft = preset.work_hard_gb + extra, preset.work_soft_gb + extra
    rows = []
    if same:
        hard = max(work_hard, residual + out + extra)
        soft = max(work_soft, residual + out + extra)
        rows.append(("work dir and output (same drive)", cfg.work_dir, hard, soft))
    else:
        rows.append(("work dir", cfg.work_dir, work_hard, work_soft))
        rows.append(("output dir", cfg.out_dir, out, out))
    for label, path, hard, soft in rows:
        free = probes.disk_free(path) / 1e9
        if free < hard:
            report.add("fail", f"disk, {label}", f"{free:.0f} GB free at {path}, needs at least {gb(hard)}",
                       "free space or pass a --work-dir / --out-dir on a bigger drive")
        elif free < soft:
            report.add("warn", f"disk, {label}", f"{free:.0f} GB free at {path}; {gb(soft)} is comfortable "
                       f"(minimum {gb(hard)}); big intermediates are removed as soon as they are done when space is short")
        else:
            report.add("ok", f"disk, {label}", f"{free:.0f} GB free at {path} (needs {gb(hard)})")


def check_machine(report: Report, cfg: BuildConfig, probes) -> None:
    gptq = cfg.preset.method == "gptq"
    gpu = probes.gpu()
    if gpu is None:
        report.add("fail", "GPU", "nvidia-smi found no NVIDIA GPU",
                   "install the NVIDIA driver; the converter runs its kernels on CUDA and GPTQ needs a GPU with "
                   f"{GPTQ_VRAM_GIB} GiB free")
    else:
        free = gpu.free_mib / 1024
        detail = f"{gpu.name}, {free:.1f} GiB free of {gpu.total_mib / 1024:.1f} GiB"
        if gptq and free < GPTQ_VRAM_GIB:
            report.add("fail", "GPU memory", f"{detail}; GPTQ peaks near 10.3 GiB and needs {GPTQ_VRAM_GIB} GiB free",
                       "close whatever uses the GPU (nvidia-smi lists it) and rerun")
        elif free < 4:
            report.add("warn", "GPU memory", detail, "free some GPU memory before the convert step")
        else:
            report.add("ok", "GPU", detail)
        if not gpu.compute_cap.startswith("12."):
            report.add("warn", "GPU architecture", f"compute capability {gpu.compute_cap}",
                       "infernix is built for Blackwell (sm_120a) only; you can still build the file here and "
                       "serve it on another machine")
    total, available = probes.ram()
    detail = f"{available / GIB:.0f} GiB available of {total / GIB:.0f} GiB"
    if gptq and available < GPTQ_RAM_AVAILABLE_GIB * GIB:
        report.add("fail", "RAM", f"{detail}; GPTQ's host peak is about {GPTQ_RAM_AVAILABLE_GIB} GiB with --max-cpu-gb 32",
                   "close other programs (the biggest memory users first) and rerun")
    elif gptq and total < GPTQ_RAM_TOTAL_GIB * GIB:
        report.add("warn", "RAM", f"{detail}; the measured route used a 64 GiB machine")
    else:
        report.add("ok", "RAM", detail)


def check_python(report: Report, cfg: BuildConfig, probes) -> None:
    needed = ["torch", "numpy", "safetensors"]
    if cfg.preset.method != "plain":
        needed.append("tokenizers")
    if cfg.preset.method != "plain" and cfg.calibration_text is None:
        needed.append("pyarrow")
    info = probes.python_info(cfg.python, tuple(needed), cuda=True)
    if info is None or "error" in (info or {}):
        report.add("fail", "converter Python", f"{cfg.python} did not run ({(info or {}).get('error', 'no output')})",
                   "run this tool with a Python 3.11+ environment that has torch (see docs/weight-conversion.md)")
        return
    missing = [name for name, version in info["packages"].items() if version is None]
    version = tuple(int(part) for part in info["python"].split(".")[:2])
    if version < (3, 11):
        report.add("fail", "converter Python", f"Python {info['python']}", "use Python 3.11 or newer")
    elif missing:
        report.add("fail", "converter Python packages", f"missing {', '.join(missing)} in {cfg.python}",
                   f"{cfg.python} -m pip install {' '.join(missing)}")
    elif not info.get("cuda"):
        report.add("fail", "converter Python", "torch cannot see a CUDA device",
                   "install a CUDA build of torch for this environment")
    else:
        report.add("ok", "converter Python", f"{cfg.python} (Python {info['python']}; "
                   f"{versions_text(info['packages'])}; CUDA visible)")
        report.tool_versions.update({name: str(v) for name, v in info["packages"].items() if v})


def check_gptq_env(report: Report, cfg: BuildConfig, probes) -> None:
    python = cfg.gptq_python
    if python is None or not Path(python).is_file():
        report.add("fail", "GPTQ environment", f"no interpreter at {python}" if python else "no GPTQ interpreter given",
                   f"{SETUP_HINT} --venv {REPO_ROOT / '.venv-gptq'}  (then pass --gptq-python <venv>/Scripts/python.exe "
                   f"or set {GPTQ_PYTHON_ENV}, or leave it to the default)")
        return
    pins = parse_pins(REPO_ROOT / "tools" / "gptq" / "requirements.txt", GPTQ_PINS)
    info = probes.python_info(str(python), GPTQ_PINS, cuda=True)
    if info is None or "error" in info:
        report.add("fail", "GPTQ environment", f"{python} did not run ({(info or {}).get('error', 'no output')})",
                   f"rebuild it: {SETUP_HINT} --venv <new dir>")
        return
    wrong = [f"{name} {info['packages'].get(name) or 'MISSING'} (pinned {pins[name]})" for name in pins
             if (info["packages"].get(name) or "").split("+")[0] != pins[name].split("+")[0]]
    if wrong:
        report.add("fail", "GPTQ environment", "; ".join(wrong),
                   f"the route is only measured with the pinned versions: {SETUP_HINT} --venv <new dir>")
    elif not info.get("cuda"):
        report.add("fail", "GPTQ environment", f"torch in {python} cannot see CUDA",
                   f"recreate the environment with the cu130 torch build: {SETUP_HINT} --venv <new dir>")
    else:
        report.add("ok", "GPTQ environment", f"{python} ({versions_text(info['packages'])}; CUDA visible)")
        report.tool_versions.update({f"gptq env {name}": str(v) for name, v in info["packages"].items() if v})


def check_server(report: Report, cfg: BuildConfig, serve: ServeSettings, server: Path | None, probes) -> None:
    name = launcher_tools.SERVER_NAME
    if not serve.launcher:
        report.add("ok", "launcher", "skipped (--no-launcher)")
        return
    if server is None or not server.is_file():
        report.add("fail", name, "no server binary found (looked in apps/Release and build*/apps/Release of the "
                   "checkout)", f"build it (build_native.bat) or pass --serve <path to {name}>, or use "
                   "--no-launcher and serve elsewhere")
        return
    found = probes.scan_binary(server, (cfg.preset.format_name,))
    if not found[cfg.preset.format_name]:
        report.add("fail", name, f"{server} does not know the {cfg.preset.format_name} tensor format "
                   f"needed by {cfg.preset.id}", "rebuild the server from this checkout (build_native.bat) "
                   "or pass --serve of a newer build")
    else:
        report.add("ok", name, f"{server} (knows {cfg.preset.format_name})")
    missing = probes.missing_dlls(server)
    if missing:
        report.add("warn", "server DLLs", f"not next to the exe or on PATH: {', '.join(missing)}",
                   "build with build_native.bat (it stages the DLLs beside the exe), or pass --cuda-bin <dir> so "
                   "the launcher adds it to PATH")
    else:
        report.add("ok", "server DLLs", "FFmpeg and cuBLAS libraries found")


def check_score(report: Report, cfg: BuildConfig, server: Path | None) -> None:
    missing = []
    if cfg.perplexity_exe is None or not cfg.perplexity_exe.is_file():
        missing.append(f"{launcher_tools.exe_name(launcher_tools.PERPLEXITY_NAME)} next to the server exe "
                       "(build the release targets)")
    for path in (REPO_ROOT / "eval" / "corpora" / "perplexity-1m" / "manifest.json",
                 REPO_ROOT / "tools" / "bench" / "kv_eval" / "kl_corpus.py"):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        report.add("fail", "--score prerequisites", "missing " + "; ".join(missing),
                   "restore the missing files, or drop --score (the build does not need it)")
    else:
        report.add("ok", "--score prerequisites", f"{cfg.perplexity_exe}; bundled corpus present; "
                   f"adds about {SCORE_EXTRA_GB} GB of scratch and about 1 h")


def check_inputs(report: Report, cfg: BuildConfig, probes) -> None:
    try:
        facts = inspect_model(cfg.model)
        report.add("ok", "model", f"{cfg.model}: BF16 Gemma 4 31B dense, {facts['shards']} weight files, "
                   f"{facts['bytes'] / 1e9:.1f} GB, vision tower {'present' if facts['vision'] else 'absent'}")
        if cfg.with_vision and not facts["vision"]:
            report.add("fail", "--with-vision", "the model has no vision tower", "drop --with-vision")
    except (ValueError, OSError, KeyError) as error:
        report.add("fail", "model", f"{cfg.model}: {error}",
                   "--model must be the Hugging Face directory (safetensors, BF16) of Gemma 4 31B or a fine-tune of it")
    if cfg.preset.method != "plain" and not (cfg.model / "tokenizer.json").is_file():
        report.add("fail", "tokenizer", f"{cfg.model / 'tokenizer.json'} is missing", "restore tokenizer.json in the model")
    if cfg.assistant is not None:
        from .gemma4_q4 import check_assistant

        try:
            check_assistant(cfg.assistant)
            report.add("ok", "MTP assistant", str(cfg.assistant))
        except (ValueError, OSError) as error:
            report.add("fail", "MTP assistant", str(error),
                       "hf download google/gemma-4-31B-it-assistant --local-dir <DIR>, or pass --no-mtp")
    model = cfg.model.resolve()
    for label, path in (("--out-dir", cfg.out_dir), ("--work-dir", cfg.work_dir)):
        resolved = path.resolve()
        if resolved == model or model in resolved.parents:
            report.add("fail", label, f"{path} is inside the source model directory",
                       "the source is only ever read; choose a directory outside it")
    try:
        revision, dirty = probes.git_state()
    except (subprocess.SubprocessError, OSError) as error:
        report.add("fail", "infernix checkout", f"cannot read the git revision ({error})",
                   "run from a git checkout of infernix (the converter records the revision)")
        return
    if dirty:
        report.add("fail", "infernix checkout", f"{len(dirty)} tracked file(s) modified (for example {dirty[0].strip()})",
                   f"commit or stash them: git -C {REPO_ROOT} status (the converter refuses a dirty checkout so the "
                   "receipt names the code that ran)")
    else:
        report.add("ok", "infernix checkout", f"revision {revision[:12]}, no tracked modifications")


def check_work_dir(report: Report, cfg: BuildConfig, fingerprint: str, identity: dict) -> None:
    try:
        state = load_state(cfg.work_dir, fingerprint, identity)
    except RuntimeError as error:
        report.add("fail", "work dir", str(error), "")
        return
    if state is not None:
        done = sum(1 for rec in state["steps"].values() if rec.get("status") == "done")
        report.add("ok", "work dir", f"{cfg.work_dir} holds this build's progress ({done} step(s) done): it resumes")
    for path in cfg.receipts:
        if path.exists() and (state is None or "convert" not in state["steps"]):
            report.add("fail", "output", f"{path} already exists",
                       "this tool never overwrites: choose another --name or --out-dir, or move the file away")
            return
    report.add("ok", "output", f"{cfg.artifact} is free")


def preflight(
    cfg: BuildConfig, serve: ServeSettings, server: Path | None, probes, fingerprint: str, identity: dict
) -> Report:
    report = Report()
    check_inputs(report, cfg, probes)
    check_work_dir(report, cfg, fingerprint, identity)
    check_machine(report, cfg, probes)
    check_python(report, cfg, probes)
    if cfg.preset.method == "gptq":
        check_gptq_env(report, cfg, probes)
    check_disk(report, cfg, probes)
    check_server(report, cfg, serve, server, probes)
    if cfg.score:
        check_score(report, cfg, server)
    if cfg.preset.method != "plain" and cfg.calibration_text is not None and not cfg.calibration_text.is_file():
        report.add("fail", "calibration text", f"{cfg.calibration_text} not found", "pass an existing UTF-8 text file")
    elif cfg.preset.method != "plain" and cfg.calibration_text is not None:
        report.add("ok", "calibration text", f"{cfg.calibration_text} (must be disjoint from any text you evaluate on)")
    elif cfg.preset.method != "plain":
        report.add("ok", "calibration text", f"pinned public corpus, downloaded at step 1 ({CALIBRATION_CHUNKS} chunks)")
    return report
