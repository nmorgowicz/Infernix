"""Check an INT4/INT5 run_gptq.py checkpoint against the converter's exact import contract.

Use --bits 5 for native dense-cross-word INT5 group64; the default remains INT4.
With --tree, also report sampled projection error against the matching BF16 text tree.
Run from the repository root: python tools/gptq/check_checkpoint.py --gptq DIR [--bits 5] [--tree DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from tools.convert.gemma4_gptq import checkpoint_group_size  # noqa: E402
from tools.convert.sources.packed_int import packed_matrix_source  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gptq", required=True, type=Path)
    parser.add_argument("--bits", type=int, choices=(4, 5), default=4)
    parser.add_argument("--tree", type=Path, help="matching text-only BF16 tree")
    parser.add_argument("--error-samples", type=int, default=3)
    args = parser.parse_args(argv)
    scales = nonpositive = 0
    low, high = 127, -128
    smallest = float("inf")
    with SafetensorsSource(args.gptq) as store:
        group_size = checkpoint_group_size(store.config, bits=args.bits)
        suffixes = (".weight_packed", ".weight_scale", ".weight_shape")
        for key in store.weight_map:
            if not key.endswith(suffixes):
                raise ValueError(f"unexpected tensor {key}: only weight_packed/weight_scale/weight_shape are importable")
        prefixes = sorted({key.rsplit(".", 1)[0] for key in store.weight_map})
        if not prefixes:
            raise ValueError("checkpoint has no packed projections")
        sources = {}
        for prefix in prefixes:
            shape = tuple(store.read_flat(prefix + ".weight_shape").tolist())
            if len(shape) != 2 or any(type(value) is not int or value <= 0 for value in shape):
                raise ValueError(f"{prefix}: expected a positive two-dimensional stored shape, got {shape}")
            source = packed_matrix_source(store, prefix, shape, group_size, bits=args.bits, native=args.bits == 5)
            sources[prefix] = source
            for begin in range(0, shape[0], 512):
                words = source.read_encoded(begin, min(shape[0], begin + 512))
                low, high = min(low, int(words.codes.min())), max(high, int(words.codes.max()))
            raw = store.read_flat(prefix + ".weight_scale").float()
            scales += raw.numel()
            nonpositive += int((raw <= 0).sum())
            if bool((raw > 0).any()):
                smallest = min(smallest, float(raw[raw > 0].min()))
        print(json.dumps({
            "bits": args.bits, "group_size": group_size, "projections": len(sources),
            "scale_values": scales, "nonpositive_scales": nonpositive, "smallest_scale": smallest,
            "scales_not_exact_in_fp16": 0, "code_range": [low, high],
            "packing": "dense_cross_word_int32" if args.bits == 5 else "eight_codes_per_int32",
            "compressor": store.config.get("infernix_gptq", {}),
        }))
        if nonpositive:
            print("NOT IMPORTABLE: checkpoint inspection requires positive scales", file=sys.stderr)
            return 1
        if args.tree is not None:
            with SafetensorsSource(args.tree) as tree:
                for prefix in prefixes[:args.error_samples]:
                    dequantized = sources[prefix].values().reshape(sources[prefix].shape)
                    reference = tree.read_flat(prefix + ".weight").float().reshape(dequantized.shape)
                    print(f"{prefix}: relative error {float((dequantized - reference).norm() / reference.norm()):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
