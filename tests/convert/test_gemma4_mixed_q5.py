"""Mixed Q4/Q5 builds: named fused projection groups come from the INT5 group64 checkpoint, the rest from the INT4 g32 one."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from tools.artifact.codecs.row_split import decode_row_split_codes, dequantize_row_split
from tools.artifact.layouts import encoded_size
from tools.artifact.reader import Artifact
from tools.convert import gemma4_q4_cli as cli
from tools.convert.gemma4_q4 import Q4Options, build_recipe, parse_q5_units, q5_unit_of, resolve_q5_units

from .test_packed_int4 import HF_ROLE, dense, pack, quantization_config, random_matrix, written
from .test_packed_int5 import expected_words, model_and_checkpoint


def write_q4(path, model):
    """HF-order INT4 g32 checkpoint for every projection of the INT5 fixture's model; {name: (codes, scales)}."""
    generator = torch.Generator().manual_seed(12)
    tensors, values = {}, {}
    for name, parameter in model.parameters.items():
        if not (name.startswith("text/layers/") and parameter.projection):
            continue
        _, _, layer, role = name.split("/", 3)
        n, k = parameter.shape
        codes, scales = random_matrix(generator, n, k)
        prefix = f"model.layers.{layer}.{HF_ROLE[role]}"
        tensors[prefix + ".weight_packed"] = pack(codes)
        tensors[prefix + ".weight_scale"] = scales
        tensors[prefix + ".weight_shape"] = torch.tensor([n, k], dtype=torch.int64)
        values[name] = (codes, scales)
    path.mkdir()
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps({"quantization_config": quantization_config(group_size=32)}))
    return values


def mixed_fixture(tmp_path):
    """A two-layer Gemma model (layer 0 sliding with q,k,v; layer 1 global with q,k) with real packing groups."""
    model, q5_values = model_and_checkpoint(tmp_path / "q5")
    q4_values = write_q4(tmp_path / "q4", model)
    model.packing_groups = [
        tuple(f"text/layers/0/attention/{role}" for role in ("query", "key", "value")),
        tuple(f"text/layers/1/attention/{role}" for role in ("query", "key")),
        *(tuple(f"text/layers/{layer}/mlp/{role}" for role in ("gate", "up")) for layer in range(2)),
    ]
    return model, q4_values, q5_values


def options(tmp_path, *units, weights="q4_g32_a16", **overrides):
    return Q4Options(
        weights=weights, gptq_source=tmp_path / "q4", q5_gptq_source=tmp_path / "q5", q5_units=tuple(units), **overrides
    )


def expected_format(name, q5_cells):
    return "q5_g64_fp16" if q5_unit_of(name) in q5_cells else "q4_g32_fp16"


def test_named_fused_groups_become_q5_and_every_other_group_stays_q4(tmp_path):
    model, _, _ = mixed_fixture(tmp_path)
    built = build_recipe(model, None, options(tmp_path, "down", "qkv@1"), device="cpu", rows_per_chunk=3)
    cells = {("down", 0), ("down", 1), ("qkv", 1)}
    prepared = built.recipe.prepare(device="cpu", rows_per_chunk=3)
    projection_jobs = {
        job.parameters: job for job in prepared.weights if model.parameters[job.parameters[0]].projection
        and job.parameters[0].startswith("text/layers/")
    }
    # The fused groups stay fused, and each is one stored object of exactly one format.
    assert set(projection_jobs) == {
        tuple(f"text/layers/0/attention/{role}" for role in ("query", "key", "value")),
        tuple(f"text/layers/1/attention/{role}" for role in ("query", "key")),
        ("text/layers/0/attention/output",),
        ("text/layers/1/attention/output",),
        ("text/layers/0/mlp/gate", "text/layers/0/mlp/up"),
        ("text/layers/1/mlp/gate", "text/layers/1/mlp/up"),
        ("text/layers/0/mlp/down",),
        ("text/layers/1/mlp/down",),
    }
    for members, job in projection_jobs.items():
        assert job.spec.format == expected_format(members[0], cells), members
        assert {expected_format(member, cells) for member in members} == {job.spec.format}
        assert job.method_name == "import_encoded"
    # Uses: Q5 cells are A16Only, Q4 cells keep the A16 build's A16Only, nothing is AllowA8.
    assert {policy for policy in built.recipe.policies.values()} == {"A16Only"}
    assert built.name == "q4_g32_gptq_import_fp8_vocab_a16_q5mix"
    assert built.scale_search["storage_formats"] == {"q4_g32_fp16": 9, "q5_g64_fp16": 4}
    assert built.scale_search["q5_source"]["cells"] == {"qkv": [1], "o": [], "gate_up": [], "down": [0, 1]}
    assert built.scale_search["q5_source"]["projections"] == 4
    assert built.q4_groups["q5_mixed_projections"] == 4
    assert built.groups["text/layers/0/mlp/down"] == 64 and built.groups["text/layers/0/mlp/gate"] == 32
    # Bytes: the Q4 build plus the exact per-object Q4 -> Q5 difference of the moved groups.
    q4_only = build_recipe(
        model, None, Q4Options(weights="q4_g32_a16", gptq_source=tmp_path / "q4"), device="cpu", rows_per_chunk=3
    )
    moved = sum(
        encoded_size("row_split_k128_v1", "q5_g64_fp16", job.spec.shape)
        - encoded_size("row_split_k128_v1", "q4_g32_fp16", job.spec.shape)
        for job in projection_jobs.values()
        if job.spec.format == "q5_g64_fp16"
    )
    assert moved > 0 and built.payload == q4_only.payload + moved


def test_q5_cells_carry_the_q5_checkpoints_codes_and_q4_cells_the_q4_checkpoints(tmp_path):
    model, q4_values, q5_values = mixed_fixture(tmp_path)
    built = build_recipe(model, None, options(tmp_path, "down", "qkv@0"), device="cpu", rows_per_chunk=3)
    path, prepared = written(tmp_path, built, model)
    jobs = {
        job.parameters: job for job in prepared.weights
        if model.parameters[job.parameters[0]].projection and job.parameters[0].startswith("text/layers/0/")
    }
    qkv = tuple(f"text/layers/0/attention/{role}" for role in ("query", "key", "value"))
    with Artifact(path) as artifact:
        for members in (qkv, ("text/layers/0/mlp/down",)):
            job = jobs[members]
            assert job.spec.format == "q5_g64_fp16"
            scales, codes = decode_row_split_codes(artifact.read_object(job.spec.id), "q5_g64_fp16", job.spec.shape)
            row = 0
            for member in members:
                want_codes, want_scales = expected_words(member, *q5_values[member])  # layer 0 is sliding: unpermuted
                n = want_codes.shape[0]
                assert torch.equal(codes[row : row + n], want_codes), member
                assert torch.equal(scales[row : row + n].view(torch.int16), want_scales.view(torch.int16)), member
                row += n
        for members in (("text/layers/0/attention/output",), ("text/layers/0/mlp/gate", "text/layers/0/mlp/up")):
            job = jobs[members]
            assert job.spec.format == "q4_g32_fp16"
            decoded = dequantize_row_split(artifact.read_object(job.spec.id), "q4_g32_fp16", job.spec.shape, dtype=torch.float32)
            row = 0
            for member in members:
                codes, scales = q4_values[member]
                n = codes.shape[0]
                assert torch.equal(decoded[row : row + n], dense(codes, scales)), member
                row += n
    # Q5 groups differ from what a Q4 import of the same name would store: not the Q4 checkpoint's words.
    assert not torch.equal(q4_values["text/layers/0/mlp/down"][0], q5_values["text/layers/0/mlp/down"][0])


def test_the_a8_override_skips_q5_cells_and_keeps_q4_cells_allow_a8(tmp_path):
    model, _, _ = mixed_fixture(tmp_path)
    built = build_recipe(
        model, None, options(tmp_path, "gate_up@0", "o", weights="q4_g32"), device="cpu", rows_per_chunk=3
    )
    cells = {("gate_up", 0), ("o", 0), ("o", 1)}
    for name, parameter in model.parameters.items():
        if not (name.startswith("text/layers/") and parameter.projection):
            continue
        want = "A16Only" if q5_unit_of(name) in cells else "AllowA8"
        assert built.recipe.policies[(name, "input")] == want, name
        assert built.recipe.selections[name][0].format == expected_format(name, cells)
    assert built.name == "q4_g32_gptq_import_fp8_vocab_a8_q5mix"


def test_without_units_the_build_is_the_unchanged_q4_import(tmp_path):
    model, _, _ = mixed_fixture(tmp_path)
    plain = build_recipe(
        model, None, Q4Options(weights="q4_g32_a16", gptq_source=tmp_path / "q4"), device="cpu", rows_per_chunk=3
    )
    assert plain.name == "q4_g32_gptq_import_fp8_vocab_a16"
    assert "q5_source" not in plain.scale_search and plain.q4_groups["q5_mixed_projections"] == 0
    assert plain.scale_search["storage_formats"] == {"q4_g32_fp16": 13}


def test_every_cell_to_q5_equals_the_all_q5_build_bytes(tmp_path):
    model, _, _ = mixed_fixture(tmp_path)
    everything = build_recipe(
        model, None, options(tmp_path, "qkv", "o", "gate_up", "down"), device="cpu", rows_per_chunk=3
    )
    q5_only = build_recipe(
        model, None, Q4Options(weights="q5_g64_a16", gptq_source=tmp_path / "q5"), device="cpu", rows_per_chunk=3
    )
    assert everything.payload == q5_only.payload
    assert everything.scale_search["storage_formats"] == {"q5_g64_fp16": 13}


def test_selector_grammar_and_resolution():
    assert parse_q5_units(["down", "qkv@0-2,7", "o@5"]) == [("down", None), ("qkv", (0, 1, 2, 7)), ("o", (5,))]
    assert resolve_q5_units(["down@58-59"], 60) == {("down", 58), ("down", 59)}
    assert len(resolve_q5_units(["gate_up"], 60)) == 60
    assert resolve_q5_units(["o@0-1", "o@1-2"], 60) == {("o", 0), ("o", 1), ("o", 2)}
    assert q5_unit_of("text/layers/12/attention/value") == ("qkv", 12)
    assert q5_unit_of("text/layers/3/mlp/up") == ("gate_up", 3)
    with pytest.raises(ValueError, match="outside the model's 60 text layers"):
        resolve_q5_units(["down@59-60"], 60)


@pytest.mark.parametrize("item", ["bogus", "down@", "down@3-1", "down@x", "qkv@0-", "mlp_down", "down@1,,2"])
def test_malformed_selectors_are_refused(item):
    with pytest.raises(ValueError, match="--q5-unit"):
        parse_q5_units([item])


def test_inconsistent_options_are_refused_before_any_checkpoint_is_read(tmp_path):
    with pytest.raises(ValueError, match="come together"):
        Q4Options(weights="q4_g32_a16", gptq_source=tmp_path, q5_units=("down",))
    with pytest.raises(ValueError, match="come together"):
        Q4Options(weights="q4_g32_a16", gptq_source=tmp_path, q5_gptq_source=tmp_path)
    for kwargs in (
        {"weights": "q5_g64_a16"},
        {"weights": "fp8_row"},
        {"gptq_source": None},
        {"native_groups": True, "group_size": 128},
    ):
        base = {"weights": "q4_g32_a16", "gptq_source": tmp_path, "q5_gptq_source": tmp_path, "q5_units": ("down",)}
        base.update(kwargs)
        with pytest.raises(ValueError):
            Q4Options(**base)


def test_wrong_layers_and_wrong_bit_width_sources_are_refused(tmp_path):
    model, _, _ = mixed_fixture(tmp_path)
    with pytest.raises(ValueError, match="outside the model's 2 text layers"):
        build_recipe(model, None, options(tmp_path, "down@2"), device="cpu", rows_per_chunk=3)
    # An INT4 g32 tree is never accepted as the INT5 source (no silent re-quantization).
    swapped = Q4Options(
        weights="q4_g32_a16", gptq_source=tmp_path / "q4", q5_gptq_source=tmp_path / "q4", q5_units=("down",)
    )
    with pytest.raises(ValueError):
        build_recipe(model, None, swapped, device="cpu", rows_per_chunk=3)


def test_command_line_carries_the_units_into_the_options(tmp_path):
    args = cli.build_parser().parse_args(
        [
            "convert", "--model", str(tmp_path), "--out", str(tmp_path / "m.ninfer"), "--no-mtp",
            "--weights", "q4_g32_a16", "--gptq-source", str(tmp_path / "q4"),
            "--q5-gptq-source", str(tmp_path / "q5"), "--q5-unit", "down", "--q5-unit", "qkv@0-19",
        ]
    )
    parsed = cli.options_from(args)
    assert parsed.q5_units == ("down", "qkv@0-19") and parsed.q5_gptq_source == tmp_path / "q5"
    defaults = cli.options_from(
        cli.build_parser().parse_args(["convert", "--model", str(tmp_path), "--out", str(tmp_path / "x.ninfer"), "--no-mtp"])
    )
    assert defaults.q5_units == () and defaults.q5_gptq_source is None
