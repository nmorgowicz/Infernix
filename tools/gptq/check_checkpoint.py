"""Check a run_gptq.py checkpoint against what `gemma4_q4 convert --gptq-source` can import, before converting.

Reads every packed projection: codes must be 4-bit offset-binary in [-8, 7] (any nibble is), every scale must be finite,
positive and exactly representable in binary16, and the layout must be the supported one. With --tree it also prints the
relative weight error of a few projections against the BF16 tree, a sanity check that the codes and scales belong to it.

Run from the repository root (it uses the converter's reader): python tools/gptq/check_checkpoint.py --gptq DIR [--tree DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from tools.convert.gemma4_gptq import checkpoint_group_size  # noqa: E402
from tools.convert.sources.packed_int4 import unpack_codes  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptq", required=True, type=Path)
    parser.add_argument("--tree", type=Path, help="text-only tree the checkpoint was made from")
    parser.add_argument("--error-samples", type=int, default=3, help="projections to compare with --tree")
    args = parser.parse_args(argv)
    config = json.loads((args.gptq / "config.json").read_text())
    group_size = checkpoint_group_size(config)
    index = json.loads((args.gptq / "model.safetensors.index.json").read_text())["weight_map"]
    files: dict[str, list[str]] = {}
    for key, file in index.items():
        files.setdefault(file, []).append(key)
    packed = scales = inexact = zero = 0
    low, high = 15, 0
    smallest = float("inf")
    shapes = {}
    for file, keys in sorted(files.items()):
        with safe_open(str(args.gptq / file), "pt") as handle:
            for key in keys:
                if key.endswith(".weight_scale"):
                    scale = handle.get_tensor(key).float()
                    scales += scale.numel()
                    inexact += int((scale.to(torch.float16).float() != scale).sum())
                    zero += int((scale <= 0).sum()) + int((~torch.isfinite(scale)).sum())
                    smallest = min(smallest, float(scale[scale > 0].min()))
                elif key.endswith(".weight_packed"):
                    packed += 1
                    words = handle.get_tensor(key)
                    nibbles = (words.unsqueeze(-1) >> torch.arange(0, 32, 4, dtype=torch.int32)) & 0xF
                    low, high = min(low, int(nibbles.min())), max(high, int(nibbles.max()))
                    shapes[key] = tuple(words.shape)
                elif not key.endswith(".weight_shape"):
                    raise ValueError(f"unexpected tensor {key}: only weight_packed/weight_scale/weight_shape are importable")
    print(json.dumps({"group_size": group_size, "projections": packed, "scale_values": scales,
                      "scales_not_exact_in_fp16": inexact, "nonpositive_or_nonfinite_scales": zero,
                      "smallest_scale": smallest, "fp16_min_normal": 2.0**-14, "code_range": [low - 8, high - 8]}))
    if inexact or zero:
        print("NOT IMPORTABLE: scales must be positive and exactly representable in binary16", file=sys.stderr)
        return 1
    if args.tree is not None:
        tree_index = json.loads((args.tree / "model.safetensors.index.json").read_text())["weight_map"]
        for key in sorted(shapes)[: args.error_samples]:
            stem = key[: -len(".weight_packed")]
            with safe_open(str(args.gptq / index[key]), "pt") as handle:
                words = handle.get_tensor(key)
                scale = handle.get_tensor(stem + ".weight_scale").float()
            n = words.shape[0]
            codes = unpack_codes(words).float().reshape(n, -1, group_size)
            dequantized = (codes * scale.unsqueeze(-1)).reshape(n, -1)
            with safe_open(str(args.tree / tree_index[stem + ".weight"]), "pt") as handle:
                reference = handle.get_tensor(stem + ".weight").float()
            print(f"{stem}: relative error {float((dequantized - reference).norm() / reference.norm()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
