"""Gemma 4 31B dense BF16 -> Q4/FP8 recipe composition and its pure helpers.

`build_recipe` layers q4 projections (logical absmax group 32/64/128/256, native or emulated storage,
absmax / MSE / imatrix-weighted scale search) on the `gemma4_31b_fp8` base. The command line lives in
`gemma4_q4_cli` (`python -m tools.convert.gemma4_q4`). Reading an imatrix needs the optional `gguf`
package (`pip install gguf`).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

Q4_REGIONS = (
    "mlp_gate_up",
    "mlp_down",
    "attn_qkv",
    "attn_out",
    "global_layers",
    "first4_layers",
    "last4_layers",
)
Q4_WEIGHTS = ("q4_g32", "q4_g32_a16", "fp8_row")
SCALE_SEARCHES = ("absmax", "mse", "weighted")
MTP_WEIGHTS = ("bf16", "fp8", "q4")
IMATRIX_ROLES = {
    "attention/query": "attn_q",
    "attention/key": "attn_k",
    "attention/value": "attn_v",
    "attention/output": "attn_output",
    "mlp/gate": "ffn_gate",
    "mlp/up": "ffn_up",
    "mlp/down": "ffn_down",
}
ASSISTANT_REPO = "google/gemma-4-31B-it-assistant"
ASSISTANT_HINT = f"download it with: hf download {ASSISTANT_REPO} --local-dir <DIR>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parse_region_groups(items: list[str]) -> list[tuple[str, int]]:
    """Parse REGION=G flags, keeping command order (the first match wins)."""
    parsed = []
    for item in items:
        region, _, size = item.partition("=")
        if region not in Q4_REGIONS or size not in ("32", "64", "128", "256"):
            raise ValueError(
                f"--region-group expects REGION=G with REGION in {Q4_REGIONS} and G in 32/64/128/256, got {item!r}"
            )
        parsed.append((region, int(size)))
    return parsed


def q4_group_for(
    name: str, layer_types: list[str], regions: list[tuple[str, int]], default: int
) -> int:
    """Logical Q4 group for a text-layer projection id such as text/layers/7/mlp/gate."""
    parts = name.split("/")
    layer, tail = int(parts[2]), "/".join(parts[3:])
    last = len(layer_types)
    member = {
        "mlp_gate_up": tail in ("mlp/gate", "mlp/up"),
        "mlp_down": tail == "mlp/down",
        "attn_qkv": tail in ("attention/query", "attention/key", "attention/value"),
        "attn_out": tail == "attention/output",
        "global_layers": layer_types[layer] == "full_attention",
        "first4_layers": layer < 4,
        "last4_layers": layer >= last - 4,
    }
    for region, size in regions:
        if member[region]:
            return size
    return default


def read_imatrix(path: Path) -> dict[str, tuple]:
    """GGUF imatrix -> {weight name: (in_sum2, counts)}; both as flat float arrays."""
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise ValueError("reading an imatrix requires the gguf package: pip install gguf") from error
    import numpy as np

    reader = GGUFReader(str(path))
    if "imatrix" not in str(reader.fields["general.type"].contents()):
        raise ValueError(f"{path} is not a GGUF imatrix file")
    tensors = {
        tensor.name: np.asarray(tensor.data, dtype=np.float64).reshape(-1) for tensor in reader.tensors
    }
    stats = {}
    for name, sums in tensors.items():
        if name.endswith(".in_sum2"):
            base = name[: -len(".in_sum2")]
            if base + ".counts" not in tensors:
                raise ValueError(f"{base}: in_sum2 without counts")
            stats[base] = (sums, tensors[base + ".counts"])
    return stats


def importance_vectors(
    stats: dict[str, tuple], widths: dict[str, int], layer_types: list[str], heads: int, perm
) -> dict:
    """Per-input-column importance sum(in_sum2)/sum(counts) in converter column order, one vector per projection.

    `widths` maps a logical projection id to its input width. Global attention/output columns are permuted
    exactly as the converter permutes the weight (heads x head_dim, columns indexed by `perm`).
    Missing, mis-sized, non-finite or negative statistics are refused, never replaced.
    """
    import numpy as np

    vectors = {}
    for name, width in widths.items():
        parts = name.split("/")
        layer, role = int(parts[2]), "/".join(parts[3:])
        weight = f"blk.{layer}.{IMATRIX_ROLES[role]}.weight"
        if weight not in stats:
            raise ValueError(f"imatrix has no statistics for {weight} (needed by {name})")
        sums, counts = stats[weight]
        if sums.size != width or counts.size != 1 or counts[0] <= 0:
            raise ValueError(
                f"{weight}: expected {width} sums and one positive count, got {sums.size} and {counts.size}"
                + (f" (count {counts[0]})" if counts.size == 1 else "")
            )
        vector = sums / counts[0]
        if not np.isfinite(vector).all() or (vector < 0).any():
            raise ValueError(f"{weight}: statistics are not finite and non-negative")
        if role == "attention/output" and layer_types[layer] == "full_attention":
            vector = vector.reshape(heads, -1)[:, np.asarray(perm)].reshape(-1)
        vectors[name] = vector.astype(np.float32)
    return vectors


def normalize_config(config: dict) -> dict:
    """Translate uniform per-kind geometry without changing the HF checkpoint."""
    result = deepcopy(config)
    text = result.get("text_config", {})
    if (
        result.get("architectures") != ["Gemma4ForConditionalGeneration"]
        or text.get("enable_moe_block")
        or (text.get("hidden_size"), text.get("num_hidden_layers"), text.get("intermediate_size"))
        != (5376, 60, 21504)
    ):
        raise ValueError("Gemma 4 conversion requires 31B dense BF16; 26B-A4B MoE is unsupported")
    if result.get("quantization_config") or text.get("quantization_config"):
        raise ValueError("Use the unquantized BF16 checkpoint, not QAT/compressed-tensors inputs")
    kinds = text.get("layer_types", [])
    if len(kinds) != 60 or set(kinds) != {"sliding_attention", "full_attention"}:
        raise ValueError("Gemma layer_types must describe all 60 sliding/full layers")
    overrides = text.get("per_layer_config") or {}
    parsed = {}
    for key, value in overrides.items():
        index = int(key)
        if not 0 <= index < len(kinds) or index in parsed or not isinstance(value, dict):
            raise ValueError(f"Invalid per_layer_config entry {key!r}")
        if set(value) - {"head_dim", "num_key_value_heads"}:
            raise ValueError(f"Unsupported per-layer fields at layer {key}: {sorted(value)}")
        parsed[index] = value
    for kind, head_key, kv_key in (
        ("sliding_attention", "head_dim", "num_key_value_heads"),
        ("full_attention", "global_head_dim", "num_global_key_value_heads"),
    ):
        geometries = set()
        for index, layer_kind in enumerate(kinds):
            if layer_kind != kind:
                continue
            layer = parsed.get(index, {})
            head = layer.get("head_dim", text.get(head_key, text.get("head_dim")))
            kv = layer.get("num_key_value_heads", text.get(kv_key, text.get("num_key_value_heads")))
            if type(head) is not int or type(kv) is not int or head <= 0 or kv <= 0:
                raise ValueError(f"Invalid attention geometry at layer {index}")
            geometries.add((head, kv))
        if len(geometries) != 1:
            raise ValueError(f"Gemma 4 conversion requires uniform geometry within {kind}")
        head, kv = geometries.pop()
        for key, value in ((head_key, head), (kv_key, kv)):
            if key in text and text[key] != value:
                raise ValueError(f"Conflicting {key} and per_layer_config")
            text[key] = value
    return result


def check_assistant(path: Path) -> dict:
    """Fail fast on a missing or mismatched MTP assistant; the converter checks full geometry."""
    config_path = path / "config.json"
    if not config_path.is_file() or not (
        (path / "model.safetensors").is_file() or (path / "model.safetensors.index.json").is_file()
    ):
        raise ValueError(f"MTP assistant not found at {path}; {ASSISTANT_HINT}, or pass --no-mtp")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("architectures") != ["Gemma4AssistantForCausalLM"]:
        raise ValueError(f"{path} is not a Gemma 4 MTP assistant ({config.get('architectures')!r})")
    if config.get("backbone_hidden_size") != 5376:
        raise ValueError("MTP assistant backbone_hidden_size must be 5376 (Gemma 4 31B)")
    return config


def select_components(has_vision: bool, text_only: bool, mtp: bool) -> tuple[str, ...]:
    components = ["text"]
    if mtp:
        components.append("mtp")
    if has_vision and not text_only:
        components.append("vision")
    return tuple(components)


@dataclass(frozen=True, slots=True)
class Q4Options:
    """Recipe choices; construction refuses every inconsistent combination."""

    weights: str = "q4_g32"
    group_size: int = 32
    region_groups: tuple[str, ...] = ()
    scale_search: str = "absmax"
    native_groups: bool = False
    imatrix: Path | None = None
    mtp_weights: str | None = None
    gptq_source: Path | None = None

    def __post_init__(self):
        if self.weights not in Q4_WEIGHTS:
            raise ValueError(f"--weights must be one of {Q4_WEIGHTS}, got {self.weights!r}")
        if self.scale_search not in SCALE_SEARCHES:
            raise ValueError(f"--scale-search must be one of {SCALE_SEARCHES}, got {self.scale_search!r}")
        regions = parse_region_groups(list(self.region_groups))
        if self.quantized:
            if (self.scale_search == "weighted") != (self.imatrix is not None):
                raise ValueError("--scale-search weighted requires --imatrix, and --imatrix requires weighted")
        elif self.scale_search != "absmax" or self.imatrix:
            raise ValueError("--scale-search/--imatrix apply only to the q4_g32 and q4_g32_a16 weights")
        elif regions or self.group_size != 32 or self.native_groups:
            raise ValueError(
                "--group-size/--region-group/--native-groups apply only to the q4_g32 and q4_g32_a16 weights"
            )
        if self.mtp_weights is not None and self.mtp_weights not in MTP_WEIGHTS:
            raise ValueError(f"--mtp-weights must be one of {MTP_WEIGHTS}, got {self.mtp_weights!r}")
        if self.mtp_weights not in (None, "bf16") and not self.quantized:
            raise ValueError("--mtp-weights fp8|q4 applies only to the q4_g32 and q4_g32_a16 weights")
        if self.gptq_source is not None:
            if not self.quantized:
                raise ValueError("--gptq-source applies only to the q4_g32 and q4_g32_a16 weights")
            if (
                self.scale_search != "absmax"
                or self.imatrix
                or regions
                or self.group_size != 32
            ):
                raise ValueError(
                    "--gptq-source is mutually exclusive with --scale-search/--imatrix/--group-size/"
                    "--region-group (the source decides codes and scales; --native-groups stores a "
                    "128/256 source in the real q4_g128_fp16/q4_g256_fp16 formats)"
                )

    @property
    def quantized(self) -> bool:
        return self.weights in ("q4_g32", "q4_g32_a16")

    @property
    def regions(self) -> list[tuple[str, int]]:
        return parse_region_groups(list(self.region_groups))


@dataclass(frozen=True, slots=True)
class Q4Build:
    """A composed recipe and the receipt facts derived while composing it."""

    recipe: object
    name: str
    groups: dict[str, int]
    scale_search: dict | None
    q4_groups: dict | None
    payload: int


def build_recipe(model, source, options: Q4Options, *, device: str, rows_per_chunk: int) -> Q4Build:
    """Compose the recipe for `model` (BF16 `source`) and project its encoded size.

    Every text-layer projection starts on the FP8 base; for the q4 weights each one is reassigned to
    its logical group's format and method. Emulated groups keep g32 storage (`q4_g32_fp16`) with the
    coarse scale repeated; `native_groups` stores `q4_g128_fp16`/`q4_g256_fp16`. `q4_g32` additionally
    applies the W4A8 override. Weighted search registers per-column importance from the imatrix.
    """
    from tools.artifact.layouts import encoded_size

    from .official_recipes import FP8, fp8_row_maxabs, gemma4_31b_fp8, gemma4_q4_a8
    from .recipe import Recipe

    recipe = Recipe(model)
    gemma4_31b_fp8(model, recipe, {"base": source})
    regions = options.regions
    layer_types = model.config["layer_types"]
    scale_search = {"method": options.scale_search}
    groups = {}
    if options.quantized:
        projections = {
            name: parameter
            for name, parameter in model.parameters.items()
            if name.startswith("text/layers/") and parameter.projection
        }
        if options.scale_search == "weighted":
            from .gemma4 import prope_perm
            from .quantization.importance import set_importance

            perm = prope_perm(model.config["global_head_dim"], model.config["partial_rotary_factor"])
            vectors = importance_vectors(
                read_imatrix(options.imatrix),
                {name: parameter.shape[1] for name, parameter in projections.items()},
                layer_types,
                model.config["num_attention_heads"],
                perm.tolist(),
            )
            set_importance(vectors)
            scale_search["imatrix"] = {
                "path": str(options.imatrix.resolve()),
                "sha256": sha256_file(options.imatrix),
                "role": "sensitivity unless collected from the target model",
            }
        imported, imported_group, imported_formats = {}, None, {}
        if options.gptq_source is not None:
            from .gemma4_gptq import gptq_projection_sources

            imported, imported_group, imported_formats = gptq_projection_sources(
                model, options.gptq_source, native=options.native_groups
            )
            if set(imported) != set(projections):
                raise ValueError("the GPTQ source does not cover exactly the model's text-layer projections")
            scale_search = {
                "method": "external_gptq_import",
                "gptq_source": str(options.gptq_source.resolve()),
                "group_size": imported_group,
                "native": options.native_groups and imported_group > 32,
                "storage_formats": {
                    fmt: sum(1 for value in imported_formats.values() if value == fmt)
                    for fmt in sorted(set(imported_formats.values()))
                },
                "kept_g32_storage": sorted(
                    name for name, fmt in imported_formats.items() if fmt == "q4_g32_fp16" and imported_group > 32
                )
                if options.native_groups and imported_group > 32
                else [],
                "note": "codes and binary16 scales copied exactly from the external checkpoint; nothing re-rounded",
            }
        for name in projections:
            if name in imported:
                target = imported_formats[name]
                groups[name] = imported_group if target != "q4_g32_fp16" or not options.native_groups else 32
                recipe.assign(name, format=target, layout="auto", method="import_encoded", source=imported[name])
                continue
            groups[name] = q4_group_for(name, layer_types, regions, options.group_size)
            if options.native_groups and groups[name] == 64:
                raise ValueError("--native-groups supports logical groups 32, 128 and 256 (no Gemma g64 kernels)")
            target = f"q4_g{groups[name]}_fp16" if options.native_groups else "q4_g32_fp16"
            if options.scale_search != "absmax":
                parameters = {"group_size": groups[name], "weighted": options.scale_search == "weighted"}
                recipe.assign(
                    name,
                    format=target,
                    layout="auto",
                    method="grouped_mse_search",
                    parameters=parameters,
                )
            elif groups[name] == 32:
                recipe.assign(name, format="q4_g32_fp16", layout="auto", method="grouped_absmax")
            else:
                recipe.assign(
                    name,
                    format=target,
                    layout="auto",
                    method="grouped_absmax_emulated",
                    parameters={"group_size": groups[name]},
                )
    mtp_weights = options.mtp_weights or ("fp8" if options.quantized else "bf16")
    if "mtp" not in model.components:
        if options.mtp_weights is not None:
            raise ValueError("--mtp-weights needs the mtp component; pass --mtp-assistant")
        mtp_weights = "bf16"
    if mtp_weights != "bf16":
        # The assistant's layer projections are BF16 in the FP8 base. They keep A16Only (the assistant has no
        # A8 permission); the outer projections and the FP8 head are left alone.
        for name, parameter in model.parameters.items():
            if not name.startswith("mtp/layers/") or not parameter.projection:
                continue
            if mtp_weights == "fp8":
                recipe.assign(name, format=FP8, method=fp8_row_maxabs)
            else:
                recipe.assign(name, format="q4_g32_fp16", layout="auto", method="grouped_absmax")
    if options.weights == "q4_g32":
        gemma4_q4_a8(model, recipe, {"base": source})
    name = {
        "q4_g32": "q4_g32_absmax_fp8_vocab_a8",
        "q4_g32_a16": "q4_g32_absmax_fp8_vocab_a16",
        "fp8_row": "fp8_row_maxabs_a16_reference",
    }[options.weights]
    if options.gptq_source is not None:
        name = name.replace("absmax", "gptq_import")
    if any(size != 32 for size in groups.values()):
        name += "_native_groups" if options.native_groups else "_virtual_groups"
    if options.scale_search != "absmax":
        name = name.replace("absmax", options.scale_search + "_search")
    if mtp_weights != "bf16":
        name += "_mtp_" + mtp_weights
    prepared = recipe.prepare(device=device, rows_per_chunk=rows_per_chunk)
    payload = sum(encoded_size(job.spec.layout, job.spec.format, job.spec.shape) for job in prepared.weights)
    # Emulated groups keep g32 storage; a real coarse format would drop (G-32)/G of the scale bytes.
    saved = 0.0
    for job in prepared.weights:
        size = groups.get(job.parameters[0], 32)
        if job.spec.format == "q4_g32_fp16" and size != 32:
            n, k = job.spec.shape
            saved += n * k * (2 / 32 - 2 / size)
    histogram: dict[int, int] = {}
    for size in groups.values():
        histogram[size] = histogram.get(size, 0) + 1
    q4_groups = {
        "default": options.group_size,
        "regions": list(options.region_groups),
        "native": options.native_groups,
        "tensors_by_group": {str(size): count for size, count in sorted(histogram.items())},
        "projected_real_weight_bytes": round(payload - saved),
        "projection_note": (
            "weight_bytes is the real native size"
            if options.native_groups
            else "scale-plane bytes only; the artifact itself is physically g32"
        ),
    }
    return Q4Build(
        recipe=recipe,
        name=name,
        groups=groups,
        scale_search=scale_search if groups else None,
        q4_groups=q4_groups if groups else None,
        payload=payload,
    )


def render(tokenizer, text: str, prefix_ids: list[int], ctx_size: int, chunks: int) -> tuple[list[str], int, int]:
    """Return (segments, skipped, stride_tokens); each segment re-encodes to exactly ctx_size tokens.

    Every segment is `prefix + body` for a chat prefix (starting with BOS) and a body spread evenly
    across the source. llama-imatrix adds one BOS to the whole file, so the first segment's literal
    BOS is dropped. A chunk that does not re-encode to exactly `ctx_size` tokens is skipped and counted.
    """
    prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    if tokenizer.encode(prefix, add_special_tokens=False).ids != prefix_ids:
        raise ValueError("the prefix does not round-trip through decode/encode; use ids whose text re-encodes exactly")
    if prefix_ids[0] != tokenizer.token_to_id("<bos>"):
        raise ValueError("prefix must begin with the <bos> token")
    body_tokens = ctx_size - len(prefix_ids)
    if body_tokens <= 0:
        raise ValueError("--ctx-size must exceed the prefix length")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if len(ids) < chunks * body_tokens:
        raise ValueError(f"source has {len(ids)} tokens; {chunks} chunks of {body_tokens} need {chunks * body_tokens}")
    stride = (len(ids) - body_tokens) // chunks if chunks > 1 else 0
    segments, skipped = [], 0
    for index in range(chunks):
        body = tokenizer.decode(ids[index * stride : index * stride + body_tokens], skip_special_tokens=False)
        encoded = tokenizer.encode(prefix + body, add_special_tokens=False).ids
        if len(encoded) != ctx_size or encoded[: len(prefix_ids)] != prefix_ids:
            skipped += 1
            continue
        segments.append(prefix + body)
    if not segments:
        raise ValueError("no chunk re-encoded to the requested length")
    segments[0] = segments[0][len(tokenizer.id_to_token(prefix_ids[0])) :]
    return segments, skipped, stride


if __name__ == "__main__":
    from .gemma4_q4_cli import main

    raise SystemExit(main())
