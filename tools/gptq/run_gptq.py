"""GPTQ W4A16 (symmetric INT4, static contiguous groups, no activation order) of a text tree with LLM Compressor.

Needs the environment of tools/gptq/requirements.txt (not the converter's own). Run from anywhere:

    python tools/gptq/run_gptq.py --recipe gemma4_31b --tree TEXT_TREE --calib calib.pt --out GPTQ_DIR \
        [--group-size 32] [--max-cpu-gb 32 --offload-dir DIR] [--fused-solve {auto,on,off}]

The quantized projections are written layer by layer, as soon as LLM Compressor has solved them, by calling
compressed-tensors' own `PackedQuantizationCompressor.compress` (the function its save path uses). `save_pretrained`
is not used: with disk-offloaded weights it dies silently on a 20 GB shard or raises "could not revert some weight
conversions", and either loses a 50 minute run. Output: part-NNNN.safetensors, model.safetensors.index.json and
config.json (the tree's config plus the quantization_config of the live scheme) - exactly what
`gemma4_q4 convert --gptq-source` reads.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import psutil
import torch
from compressed_tensors.compressors.pack_quantized.base import PackedQuantizationCompressor
from compressed_tensors.offload import load_offloaded_model
from compressed_tensors.utils import align_module_device
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import fast_solve
import recipes

OBSERVERS = ("imatrix_mse", "memoryless_mse", "memoryless_minmax")


class DumpingGPTQ(GPTQModifier):
    """GPTQModifier that persists each layer's packed tensors right after it is solved."""

    def calibrate_module(self, module, args, _output):
        """LLM Compressor's hook, except that a k/v/up projection that receives the very tensor its leader (q/gate)
        just accumulated takes over the leader's Hessian instead of recomputing the identical one. A follower that
        receives a different tensor accumulates its own, so sharing can never change a result."""
        if DumpingGPTQ.share_hessians:
            parent, _, leaf = self._module_names[module].rpartition(".")
            if leaf in DumpingGPTQ.leaders:
                leader = DumpingGPTQ.by_name(self).get(f"{parent}.{DumpingGPTQ.leaders[leaf]}")
                fresh = module not in self._num_samples or module in DumpingGPTQ.aliased
                if fresh and leader in self._num_samples and DumpingGPTQ.last_input.get(leader) is args[0]:
                    self._hessians[module] = self._hessians[leader]
                    self._num_samples[module] = self._num_samples[leader]
                    DumpingGPTQ.aliased.add(module)
                    DumpingGPTQ.shared += 1
                    return
                if module in DumpingGPTQ.aliased:
                    raise RuntimeError(f"{self._module_names[module]} stopped receiving its leader's input; Hessian would be wrong")
            elif leaf in DumpingGPTQ.leaders.values():
                DumpingGPTQ.last_input[module] = args[0]
        super().calibrate_module(module, args, _output)

    @staticmethod
    def by_name(modifier) -> dict:
        if DumpingGPTQ.names is None:
            DumpingGPTQ.names = {name: module for module, name in modifier._module_names.items()}
        return DumpingGPTQ.names

    def compress_module_list(self, module_list):
        started = time.time()
        for module in module_list:
            DumpingGPTQ.last_input.pop(module, None)
            DumpingGPTQ.aliased.discard(module)
        super().compress_module_list(module_list)
        solved = time.time()
        tensors = {}
        for module in module_list:
            name = self._module_names[module]
            with align_module_device(module):
                state = {"weight": module.weight.detach().cpu(), "weight_scale": module.weight_scale.detach().cpu()}
                if getattr(module, "weight_zero_point", None) is not None:
                    state["weight_zero_point"] = module.weight_zero_point.detach().cpu()
            packed = PackedQuantizationCompressor.compress(state, module.quantization_scheme)
            unexpected = set(packed) - {"weight_packed", "weight_scale", "weight_shape"}
            if unexpected:
                raise RuntimeError(f"{name}: unexpected compressed tensors {sorted(unexpected)}")
            for key, value in packed.items():
                tensors[f"{name}.{key}"] = value.contiguous()
            DumpingGPTQ.scheme.setdefault("weights", module.quantization_scheme.weights.model_dump(mode="json"))
        if not tensors:
            return
        file = f"part-{DumpingGPTQ.parts:04d}.safetensors"
        DumpingGPTQ.parts += 1
        save_file(tensors, str(DumpingGPTQ.out / file), metadata={"format": "pt"})
        DumpingGPTQ.weight_map.update({key: file for key in tensors})
        print(f"dumped {len(tensors)} tensors -> {file} (solve {solved - started:.1f}s, pack+save {time.time() - solved:.1f}s)", flush=True)


DumpingGPTQ.out = Path(".")
DumpingGPTQ.weight_map = {}
DumpingGPTQ.scheme = {}
DumpingGPTQ.parts = 0
DumpingGPTQ.share_hessians = True
DumpingGPTQ.last_input = {}
DumpingGPTQ.aliased = set()
DumpingGPTQ.shared = 0
DumpingGPTQ.names = None
DumpingGPTQ.leaders = {}


def sample_resources(peak: dict, log: Path | None, stop: threading.Event) -> None:
    process = psutil.Process()
    while not stop.wait(30):
        peak["rss_gib"] = max(peak["rss_gib"], process.memory_info().rss / 2**30)
        peak["avail_min_gib"] = min(peak["avail_min_gib"], psutil.virtual_memory().available / 2**30)
        peak["gpu_torch_peak_gib"] = max(peak["gpu_torch_peak_gib"], torch.cuda.max_memory_allocated() / 2**30)
        if log is not None:
            with log.open("a") as handle:
                handle.write(json.dumps({"t": time.time(), **peak}) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", required=True, type=Path, help="text-only tree from make_text_tree.py")
    parser.add_argument("--calib", required=True, type=Path, help="token tensor from make_calib.py")
    parser.add_argument("--out", required=True, type=Path, help="GPTQ checkpoint directory (must not exist)")
    parser.add_argument("--recipe", required=True, choices=recipes.names(),
                        help="model-family recipe: sequential targets, ignore patterns, Hessian leaders, defaults")
    parser.add_argument("--observer", choices=OBSERVERS, default=None,
                        help="scale search (default: the recipe's; imatrix_mse, the importance-weighted MSE, is the recommended one)")
    parser.add_argument("--group-size", type=int, default=None, choices=(32, 64, 128, 256),
                        help="quantization group size (default: the recipe's)")
    parser.add_argument("--samples", type=int, help="use only the first N calibration chunks (default: all)")
    parser.add_argument("--dampening", type=float, default=None, help="Hessian dampening fraction (default: the recipe's)")
    parser.add_argument("--block-size", type=int, default=None, help="GPTQ block width (default: the recipe's)")
    parser.add_argument("--max-cpu-gb", type=float,
                        help="keep at most this many GiB of weights in RAM and offload the rest to --offload-dir "
                             "(needed for the full 31B on a 64 GiB machine; 32 was used). Default: all weights on the CPU")
    parser.add_argument("--offload-dir", type=Path, help="scratch directory for offloaded weights (with --max-cpu-gb)")
    parser.add_argument("--peak-log", type=Path, help="append a JSON line of resource peaks every 30 s")
    parser.add_argument("--fused-solve", choices=fast_solve.MODES, default="auto",
                        help="solve a 128-column block per Triton launch instead of 128 Python steps (bit-identical output, "
                             "about 10x faster solves). auto: use it when a start-up self-test of the kernel against the "
                             "eager loop passes on this machine, else warn and use the eager solver; on: fail instead of "
                             "falling back; off: eager solver")
    parser.add_argument("--share-hessians", choices=("on", "off"), default="on",
                        help="let k/v and up take over the Hessian that q and gate just accumulated from the same input "
                             "tensor (bit-identical; measured 0.8 s of 7.6 s per layer on the calibration pass)")
    args = parser.parse_args(argv)
    recipe = recipes.get(args.recipe)
    for option in ("observer", "group_size", "dampening", "block_size"):
        if getattr(args, option) is None:
            setattr(args, option, getattr(recipe, option))
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists; use a fresh directory")
    if (args.max_cpu_gb is None) != (args.offload_dir is None):
        parser.error("--max-cpu-gb and --offload-dir go together")

    calib = torch.load(args.calib)
    if args.samples:
        calib = calib[: args.samples]
    context = calib.shape[1]
    dataset = Dataset.from_dict({"input_ids": calib.long().tolist(), "attention_mask": [[1] * context] * len(calib)})
    args.out.mkdir(parents=True)
    DumpingGPTQ.out = args.out

    peak = {"rss_gib": 0.0, "avail_min_gib": 1e9, "gpu_torch_peak_gib": 0.0}
    stop = threading.Event()
    threading.Thread(target=sample_resources, args=(peak, args.peak_log, stop), daemon=True).start()

    gptq = DumpingGPTQ(
        config_groups={"group_0": {
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group",
                        "group_size": args.group_size, "actorder": None, "observer": args.observer},
        }},
        ignore=list(recipe.ignore),
        block_size=args.block_size,
        dampening_frac=args.dampening,
        actorder=None,
        offload_hessians=False,
    )
    fast_solve.install(args.fused_solve)
    DumpingGPTQ.share_hessians = args.share_hessians == "on"
    DumpingGPTQ.leaders = recipe.hessian_leaders
    load = {"device_map": "cpu"}
    if args.max_cpu_gb is not None:
        load = {"device_map": "auto_offload", "max_memory": {"cpu": int(args.max_cpu_gb * 2**30)},
                "offload_folder": str(args.offload_dir)}
    started = time.time()
    with load_offloaded_model():
        model = AutoModelForCausalLM.from_pretrained(str(args.tree), dtype=torch.bfloat16, **load)
    print(f"loaded {type(model).__name__} in {time.time() - started:.1f}s", flush=True)
    # A tree with wrong key names loads "successfully" with randomly initialized weights; compare with the files.
    index = json.loads((args.tree / "model.safetensors.index.json").read_text())["weight_map"]
    for key, parameter in recipe.load_check(model, args.tree):
        with safe_open(str(args.tree / index[key]), "pt") as handle:
            if not torch.equal(handle.get_tensor(key), parameter.detach().cpu()):
                raise RuntimeError(f"{key}: loaded weights differ from the tree files (randomly initialized?)")
    print("load check ok: weights equal the tree files", flush=True)

    calibrating = time.time()
    oneshot(
        model=model,
        dataset=dataset,
        processor=AutoTokenizer.from_pretrained(str(args.tree)),  # the Gemma4 processor needs torchvision; a tokenizer suffices
        recipe=gptq,
        max_seq_length=context,
        num_calibration_samples=len(calib),
        batch_size=1,
        shuffle_calibration_samples=False,
        pipeline="sequential",
        sequential_targets=list(recipe.sequential_targets),
    )
    seconds = time.time() - calibrating
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": DumpingGPTQ.weight_map}, indent=1)
    )
    config = json.loads((args.tree / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "quantization_status": "compressed",
        "ignore": list(recipe.saved_ignore),
        "config_groups": {"group_0": {"targets": ["Linear"], "format": "pack-quantized", "input_activations": None,
                                       "output_activations": None, "weights": DumpingGPTQ.scheme["weights"]}},
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    stop.set()
    peak["rss_gib"] = max(peak["rss_gib"], psutil.Process().memory_info().rss / 2**30)
    peak["gpu_torch_peak_gib"] = max(peak["gpu_torch_peak_gib"], torch.cuda.max_memory_allocated() / 2**30)
    print(json.dumps({"calibration_s": round(seconds, 1), "projections": len(DumpingGPTQ.weight_map) // 3,
                      "observer": args.observer, "group_size": args.group_size,
                      "shared_hessians": DumpingGPTQ.shared, **peak}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
