"""Shared helpers of the KV-format evaluation kit: stdio, hashing, arms files, tokenizer and corpus identity."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

REPO_ROOT = Path(__file__).resolve().parents[3]

# Resources every infernix text artifact carries under resource/text/ (see tools/artifact/inspect.py --objects).
ARTIFACT_TEXT_RESOURCES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")

ROLES = ("reference", "noise", "quant")


def use_utf8_stdio() -> None:
    """Report text contains non-ASCII symbols; never let a legacy console code page break a run."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def fail(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(2)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        fail(f"{what} not found: {path}")
    except json.JSONDecodeError as exc:
        fail(f"{what} is not valid JSON ({path}): {exc}")


@dataclass(frozen=True)
class Arm:
    name: str
    flags: tuple[str, ...]
    role: str


def load_arms(path: Path) -> list[Arm]:
    """Read an arms file: a JSON list of {"name", "flags", optional "role"}.

    Roles: "reference" (exactly one), "noise" (same-flags repeat of the reference) and "quant" (compared arms).
    Without explicit roles the first arm is the reference and a name containing "repeat" is a noise arm.
    """
    spec = read_json(path, "arms file")
    items = spec.get("arms") if isinstance(spec, dict) else spec
    if not isinstance(items, list) or not items:
        fail(f"{path}: expected a non-empty JSON list of arms (or an object with an 'arms' list)")
    arms: list[Arm] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
            fail(f"{path}: arm #{index} needs a non-empty string 'name'")
        flags = item.get("flags", [])
        if not isinstance(flags, list) or not all(isinstance(f, str) for f in flags):
            fail(f"{path}: arm '{item['name']}' 'flags' must be a list of strings (one argv token per element)")
        name = item["name"]
        if name in seen:
            fail(f"{path}: duplicate arm name '{name}'")
        seen.add(name)
        role = item.get("role") or ("reference" if index == 0 else "noise" if "repeat" in name else "quant")
        if role not in ROLES:
            fail(f"{path}: arm '{name}' has role '{role}'; expected one of {ROLES}")
        arms.append(Arm(name, tuple(flags), role))
    if sum(a.role == "reference" for a in arms) != 1:
        fail(f"{path}: exactly one arm must have role 'reference' (default: the first arm)")
    return arms


def find_arm(arms: list[Arm], name: str, source: Path) -> Arm:
    for arm in arms:
        if arm.name == name:
            return arm
    fail(f"arm '{name}' is not in {source}; available: {', '.join(a.name for a in arms)}")


def extract_artifact_resources(artifact: Path, dest: Path) -> dict[str, dict]:
    """Write the artifact's embedded resource/text/* files to `dest`; return name -> {path, bytes, sha256}."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tools.artifact.reader import Artifact  # stdlib-only v3 reader of this repository

    dest.mkdir(parents=True, exist_ok=True)
    found: dict[str, dict] = {}
    try:
        with Artifact(artifact) as art:
            for name in ARTIFACT_TEXT_RESOURCES:
                object_id = f"resource/text/{name}"
                if object_id not in art.by_id:
                    continue
                data = art.read_object(object_id)
                target = dest / name
                target.write_bytes(data)
                found[name] = {"path": str(target), "bytes": len(data), "sha256": sha256_bytes(data)}
    except (OSError, ValueError) as exc:
        fail(f"cannot read artifact {artifact}: {exc}. Check the path, or pass --tokenizer with a tokenizer.json file instead.")
    return found


def resolve_tokenizer(tokenizer: Path | None, artifact: Path | None, scratch: Path) -> tuple[Path, dict]:
    """tokenizer.json to use plus its provenance record: an explicit file, else the artifact's embedded one."""
    if tokenizer is not None:
        if not tokenizer.is_file():
            fail(f"tokenizer file not found: {tokenizer}")
        return tokenizer, {"source": "file", "path": str(tokenizer), "sha256": sha256_file(tokenizer)}
    if artifact is None:
        fail("pass --tokenizer <tokenizer.json> or --artifact <file.infernix> (its embedded tokenizer is used)")
    found = extract_artifact_resources(artifact, scratch)
    if "tokenizer.json" not in found:
        fail(f"{artifact} has no resource/text/tokenizer.json; pass --tokenizer <tokenizer.json> explicitly")
    return Path(found["tokenizer.json"]["path"]), {"source": "artifact-embedded", "artifact": str(artifact), **found["tokenizer.json"]}


def corpus_identity(manifest: Path) -> dict:
    """Hashes of the corpus manifest and of every stream file it lists (manifest.json alone does not cover the data)."""
    spec = read_json(manifest, "corpus manifest")
    streams = []
    for entry in spec.get("streams", []):
        path = manifest.parent / entry["path"]
        if not path.is_file():
            fail(f"corpus stream file missing: {path} (listed in {manifest})")
        streams.append({"id": entry["id"], "domain": entry["domain"], "path": entry["path"], "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    combined = sha256_bytes("".join(f"{s['id']}\0{s['sha256']}\n" for s in streams).encode("utf-8"))
    return {"corpus_id": spec.get("corpus_id"), "manifest_sha256": sha256_file(manifest), "streams_combined_sha256": combined,
            "streams": streams}
