"""Gemma 4 text projections imported from a llama.cpp GGUF (tools/convert/gemma4_gguf.py).

A toy two-layer Gemma 4 (one sliding, one global layer with the real global head geometry: 512-wide heads, a quarter
rotated) gets random GGUF blocks for every projection. The artifact must hold exactly the GGUF's stored rows (global
query/key rows in the converter's prope order, everything else verbatim), and the global output projection's input
permutation must reproduce the source-order product from the converted attention output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

from tools.artifact.codecs import ggml
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.gemma4 import prope_perm
from tools.convert.gemma4_gguf import PERM_BLOCK, head_row_ranges, output_input_perm
from tools.convert.gemma4_q4 import Q4Options, build_recipe
from tools.convert.model import Model, Parameter
from tools.convert.sources.logical import array_source

HIDDEN, INTER, HEADS, KV, SLIDING_DIM, GLOBAL_DIM, FACTOR = 256, 512, 2, 1, 256, 512, 0.25
KINDS = ["sliding_attention", "full_attention"]
CONFIG = {
    "model_type": "gemma4_text",
    "num_hidden_layers": len(KINDS),
    "hidden_size": HIDDEN,
    "intermediate_size": INTER,
    "layer_types": KINDS,
    "num_attention_heads": HEADS,
    "global_head_dim": GLOBAL_DIM,
    "num_global_key_value_heads": KV,
    "partial_rotary_factor": FACTOR,
}
GGUF_ROLE = {
    "attention/query": "attn_q",
    "attention/key": "attn_k",
    "attention/value": "attn_v",
    "attention/output": "attn_output",
    "mlp/gate": "ffn_gate",
    "mlp/up": "ffn_up",
    "mlp/down": "ffn_down",
}
FORMATS = (ggml.IQ3_S, ggml.IQ3_XXS, ggml.IQ2_S, ggml.Q6_K, ggml.IQ4_XS, ggml.Q4_K, ggml.Q5_K)


def shapes(layer: int) -> dict[str, tuple[int, int]]:
    glob = KINDS[layer] == "full_attention"
    dim, kv = (GLOBAL_DIM, KV) if glob else (SLIDING_DIM, 2)
    result = {
        "attention/query": (HEADS * dim, HIDDEN),
        "attention/key": (kv * dim, HIDDEN),
        "attention/output": (HIDDEN, HEADS * dim),
        "mlp/gate": (INTER, HIDDEN),
        "mlp/up": (INTER, HIDDEN),
        "mlp/down": (HIDDEN, INTER),
    }
    if not glob:
        result["attention/value"] = (kv * dim, HIDDEN)
    return result


def toy_model() -> Model:
    model = Model({"text": {"config": dict(CONFIG)}})

    def add(name, shape, inputs=()):
        model.add(Parameter(name, shape, array_source(torch.zeros(*shape, dtype=torch.bfloat16), name), inputs=inputs))

    add("text/token_embedding", (8, HIDDEN))
    add("text/output_head", (8, HIDDEN))
    add("text/final_norm", (HIDDEN,))
    for layer in range(len(KINDS)):
        for role, shape in shapes(layer).items():
            add(f"text/layers/{layer}/{role}", shape, inputs=("input",))
        add(f"text/layers/{layer}/norm", (HIDDEN,))
    model.packing_groups = [
        ("text/layers/0/attention/query", "text/layers/0/attention/key", "text/layers/0/attention/value"),
        ("text/layers/0/mlp/gate", "text/layers/0/mlp/up"),
        ("text/layers/1/attention/query", "text/layers/1/attention/key"),
        ("text/layers/1/mlp/gate", "text/layers/1/mlp/up"),
    ]
    return model


def random_blocks(fmt: ggml.GgmlType, rows: int, k: int, rng: np.random.Generator) -> np.ndarray:
    blocks = rows * (k // fmt.block_elems)
    data = rng.integers(0, 256, size=(blocks, fmt.block_bytes), dtype=np.uint8)
    words = (rng.integers(0, 1 << 10, size=(blocks, 2)) | (rng.integers(5, 21, size=(blocks, 2)) << 10)).astype("<u2")
    offsets = [208] if fmt is ggml.Q6_K else [0, 2] if fmt.has_min else [0]
    for i, offset in enumerate(offsets):
        data[:, offset : offset + 2] = words[:, i : i + 1].view(np.uint8).reshape(blocks, 2)
    return data.reshape(rows, -1)


def write_gguf(path, model: Model, formats=None) -> dict[str, tuple[ggml.GgmlType, np.ndarray]]:
    """A gemma4 GGUF with random blocks for every projection; returns {parameter: (format, stored rows)}."""
    rng = np.random.default_rng(5)
    writer = gguf.GGUFWriter(str(path), "gemma4")
    writer.add_uint32("gemma4.block_count", len(KINDS))
    writer.add_uint32("gemma4.embedding_length", HIDDEN)
    writer.add_uint32("gemma4.feed_forward_length", INTER)
    writer.add_uint32("gemma4.attention.head_count", HEADS)
    stored = {}
    names = [n for n, p in model.parameters.items() if n.startswith("text/layers/") and p.projection]
    for i, name in enumerate(names):
        _, _, layer, role = name.split("/", 3)
        n, k = model.parameters[name].shape
        fmt = (formats or {}).get(name, FORMATS[i % len(FORMATS)])
        rows = random_blocks(fmt, n, k, rng)
        writer.add_tensor(f"blk.{layer}.{GGUF_ROLE[role]}.weight", rows,
                          raw_dtype=gguf.GGMLQuantizationType(fmt.ggml_id))
        stored[name] = (fmt, rows)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return stored


def written_rows(tmp_path, built, model) -> dict[str, tuple[str, str, np.ndarray]]:
    """{member parameter: (format, layout, stored rows)} of the written artifact."""
    prepared = built.recipe.prepare(device="cpu", rows_per_chunk=5)
    path = tmp_path / "toy.ninfer"
    specs = [job.spec for job in prepared.weights]
    with ArtifactWriter(path, specs, components=model.components, bindings=prepared.bindings,
                        uses=prepared.uses) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    result = {}
    with Artifact(path) as artifact:
        for job in prepared.weights:
            if not job.spec.format.startswith("ggml_"):
                continue
            n = job.spec.shape[0]
            payload = np.frombuffer(bytes(artifact.read_object(job.spec.id)), dtype=np.uint8).reshape(n, -1)
            row = 0
            for member in job.parameters:
                rows = model.parameters[member].shape[0]
                result[member] = (job.spec.format, job.spec.layout, payload[row : row + rows])
                row += rows
    return result


def converted_rows(name: str, rows: np.ndarray) -> np.ndarray:
    _, _, layer, role = name.split("/", 3)
    if KINDS[int(layer)] != "full_attention" or role not in ("attention/query", "attention/key"):
        return rows
    perm = prope_perm(GLOBAL_DIM, FACTOR)
    heads = HEADS if role == "attention/query" else KV
    order = [h * GLOBAL_DIM + j for h in range(heads) for j in perm.tolist()]
    return rows[order]


@pytest.mark.parametrize("weights", ["q4_g32_a16", "q4_g32"])
def test_gguf_blocks_reach_the_artifact_verbatim(tmp_path, weights):
    model = toy_model()
    stored = write_gguf(tmp_path / "toy.gguf", model)
    built = build_recipe(model, None, Q4Options(weights=weights, gguf_source=tmp_path / "toy.gguf"), device="cpu",
                         rows_per_chunk=5)
    got = written_rows(tmp_path, built, model)
    assert set(got) == set(stored)
    for name, (fmt, rows) in stored.items():
        got_format, layout, got_rows = got[name]
        assert got_format == fmt.name and layout == "ggml_rows_v1", name
        assert np.array_equal(got_rows, converted_rows(name, rows)), name
    assert model.config["global_output_order"] == "source"
    assert built.scale_search["method"] == "gguf_import"
    assert sum(built.scale_search["storage_formats"].values()) == len(stored)
    assert "gguf_import" in built.name
    # q4_g32 (the default weights) allows 8-bit activations on every GGUF projection, as on q4 trunks; q4_g32_a16 keeps
    # them on the exact 16-bit routes.
    policies = {name: policy for (name, _), policy in built.recipe.policies.items() if name in stored}
    assert set(policies) == set(stored)
    assert set(policies.values()) == {"AllowA8" if weights == "q4_g32" else "A16Only"}


def test_same_format_groups_share_one_parent_and_mixed_ones_split(tmp_path):
    model = toy_model()
    same = {n: ggml.IQ3_S for n in model.parameters if n.startswith("text/layers/")}
    same["text/layers/1/attention/key"] = ggml.Q6_K
    write_gguf(tmp_path / "toy.gguf", model, same)
    built = build_recipe(model, None, Q4Options(weights="q4_g32_a16", gguf_source=tmp_path / "toy.gguf"),
                         device="cpu", rows_per_chunk=5)
    parents = {tuple(job.parameters) for job in built.recipe.prepare(device="cpu", rows_per_chunk=5).weights}
    assert ("text/layers/0/attention/query", "text/layers/0/attention/key", "text/layers/0/attention/value") in parents
    assert ("text/layers/1/attention/query",) in parents and ("text/layers/1/attention/key",) in parents


def test_head_row_ranges_are_the_converter_row_order():
    perm = prope_perm(GLOBAL_DIM, FACTOR)
    ranges = head_row_ranges(perm, 3)
    rows = [r for a, b in ranges for r in range(a, b)]
    assert rows == [h * GLOBAL_DIM + j for h in range(3) for j in perm.tolist()]
    assert all(b - a >= PERM_BLOCK for a, b in ranges)  # whole 64-row runs, merged


def test_output_input_perm_restores_the_source_order_product():
    perm = prope_perm(GLOBAL_DIM, FACTOR)
    generator = torch.Generator().manual_seed(3)
    w = torch.randn(16, HEADS * GLOBAL_DIM, generator=generator, dtype=torch.float64)  # source (HF) column order
    source = torch.randn(HEADS, GLOBAL_DIM, generator=generator, dtype=torch.float64)  # attention output, HF order
    converted = source[:, perm].reshape(-1)  # what the runtime's attention writes (converted position p = perm[p])
    blocks = output_input_perm(perm, HEADS)
    assert len(blocks) == HEADS * GLOBAL_DIM // PERM_BLOCK and sorted(blocks) == list(range(len(blocks)))
    gathered = converted.reshape(-1, PERM_BLOCK)[torch.tensor(blocks)].reshape(-1)  # input block j = block perm[j]
    assert torch.equal(gathered, source.reshape(-1))
    assert torch.equal(w @ gathered, w @ source.reshape(-1))


def test_a_permutation_that_splits_a_block_is_refused():
    with pytest.raises(ValueError, match="whole 64-element blocks"):
        output_input_perm(prope_perm(256, 0.125), 1)  # 16 rotated pairs: runs of 16


def test_options_exclude_every_other_source_of_codes(tmp_path):
    for extra in ({"gptq_source": tmp_path}, {"scale_search": "mse"}, {"group_size": 128}, {"native_groups": True}):
        with pytest.raises(ValueError, match="--gguf-source is mutually exclusive"):
            Q4Options(weights="q4_g32_a16", gguf_source=tmp_path, **extra)
    with pytest.raises(ValueError, match="--gguf-source applies only"):
        Q4Options(weights="fp8_row", gguf_source=tmp_path)


def test_a_foreign_architecture_is_refused(tmp_path):
    model = toy_model()
    path = tmp_path / "other.gguf"
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    with pytest.raises(ValueError, match="is not gemma4"):
        build_recipe(model, None, Q4Options(weights="q4_g32_a16", gguf_source=path), device="cpu", rows_per_chunk=5)
