"""Write a text-only `Gemma4ForCausalLM` Hugging Face tree (optionally the first N layers) from the BF16 checkpoint.

LLM Compressor 0.13 needs a tree that transformers can load and trace. The multimodal checkpoint's
`per_layer_config` is not readable by transformers 5.14.1, so the config is normalized exactly as the converter does
(`normalize_config`: global_head_dim / num_global_key_value_heads). Weights are copied bit for bit, with the keys
`model.language_model.*` renamed to `model.*`; vision and audio weights are left out.

Run from the repository root: python tools/gptq/make_text_tree.py --model DIR --out DIR [--layers N]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from tools.convert.gemma4_q4 import normalize_config  # noqa: E402

PREFIX = "model.language_model."
SHARD_BYTES = 4 * 2**30
SIDE_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")


def build_tree(model: Path, out: Path, layers: int | None) -> int:
    config = normalize_config(json.loads((model / "config.json").read_text(encoding="utf-8")))
    text = config["text_config"]
    count = text["num_hidden_layers"] if layers is None else layers
    if not 1 <= count <= text["num_hidden_layers"]:
        raise ValueError(f"--layers must be in 1..{text['num_hidden_layers']}")
    if text["layer_types"][count - 1] != "full_attention":
        # transformers rewrites a truncated model's last layer to full attention, which silently reshapes it.
        raise ValueError(f"--layers {count} ends on a sliding layer; use a count whose last layer is global (6, 12, ... 60)")
    text.pop("per_layer_config", None)
    text["num_hidden_layers"] = count
    text["layer_types"] = text["layer_types"][:count]
    text["architectures"] = ["Gemma4ForCausalLM"]
    text["tie_word_embeddings"] = True
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(text, indent=2), encoding="utf-8")
    for name in SIDE_FILES:
        shutil.copy(model / name, out / name)
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    by_shard: dict[str, list[str]] = {}
    for key, shard in index.items():
        if not key.startswith(PREFIX):
            continue
        rest = key[len(PREFIX):]
        if rest.startswith("layers.") and int(rest.split(".")[1]) >= count:
            continue
        by_shard.setdefault(shard, []).append(key)
    weight_map: dict[str, str] = {}
    bucket: dict = {}
    size = 0
    shards = 0

    def flush() -> None:
        nonlocal bucket, size, shards
        if not bucket:
            return
        name = f"model-{shards:05d}.safetensors"
        save_file(bucket, str(out / name), metadata={"format": "pt"})
        weight_map.update({key: name for key in bucket})
        shards += 1
        bucket, size = {}, 0

    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(model / shard), "pt") as handle:
            for key in keys:
                tensor = handle.get_tensor(key)
                bucket["model." + key[len(PREFIX):]] = tensor
                size += tensor.numel() * tensor.element_size()
                if size > SHARD_BYTES:
                    flush()
    flush()
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1), encoding="utf-8"
    )
    return len(weight_map)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="BF16 Gemma 4 31B checkpoint directory")
    parser.add_argument("--out", required=True, type=Path, help="new tree directory (must not exist)")
    parser.add_argument("--layers", type=int, help="keep only the first N layers (default: all 60)")
    args = parser.parse_args(argv)
    tensors = build_tree(args.model, args.out, args.layers)
    print(f"wrote {args.out}: {tensors} tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
