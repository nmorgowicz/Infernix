"""External compressed-tensors INT4 (GPTQ) import: codes and scales reach the artifact unchanged, permutations included."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.artifact.codecs.row_split import decode_row_split_codes, dequantize_row_split
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.gemma4 import _permute_cols, _permute_rows, prope_perm
from tools.convert.gemma4_gptq import checkpoint_group_size, gptq_projection_sources
from tools.convert.gemma4_q4 import Q4Options, build_recipe
from tools.convert.model import Model, Parameter
from tools.convert.sources.logical import array_source
from tools.convert.sources.packed_int4 import column_group_index, packed_int4_matrix_source, unpack_codes
from tools.convert.sources.safetensors import SafetensorsSource

HIDDEN, INTER, HEADS, KV, SLIDING_DIM, GLOBAL_DIM = 256, 512, 4, 1, 64, 128
KINDS = ["sliding_attention", "full_attention"]
CONFIG = {
    "model_type": "gemma4_text",
    "num_hidden_layers": len(KINDS),
    "layer_types": KINDS,
    "num_attention_heads": HEADS,
    "global_head_dim": GLOBAL_DIM,
    "num_global_key_value_heads": KV,
    "partial_rotary_factor": 0.5,  # rotated pairs 32 -> every permuted block is a whole 32-group
}
HF_ROLE = {
    "attention/query": "self_attn.q_proj",
    "attention/key": "self_attn.k_proj",
    "attention/value": "self_attn.v_proj",
    "attention/output": "self_attn.o_proj",
    "mlp/gate": "mlp.gate_proj",
    "mlp/up": "mlp.up_proj",
    "mlp/down": "mlp.down_proj",
}


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
    model.packing_groups = []
    return model


def pack(codes: torch.Tensor) -> torch.Tensor:
    """Independent packer: offset-binary nibbles, element 8w+j in bits 4j..4j+3 of int32 word w."""
    unsigned = (codes.numpy().astype(np.int64) + 8).reshape(codes.shape[0], -1, 8)
    words = np.zeros(unsigned.shape[:2], dtype=np.uint32)
    for j in range(8):
        words |= (unsigned[:, :, j].astype(np.uint32) << np.uint32(4 * j))
    return torch.from_numpy(words.view(np.int32).copy())


def quantization_config(**weights):
    scheme = {"num_bits": 4, "type": "int", "symmetric": True, "strategy": "group", "group_size": 32, "actorder": None}
    scheme.update(weights)
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {"group_0": {"targets": ["Linear"], "weights": scheme}},
    }


def random_matrix(generator, n, k, group_size=32):
    codes = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    # integers of at most 6 bits times 2**-12: exact in bf16 and binary16
    scales = (torch.randint(1, 64, (n, k // group_size), generator=generator).float() * 2.0**-12).to(torch.bfloat16)
    return codes, scales


def dense(codes: torch.Tensor, scales: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    return codes.float() * scales.float().repeat_interleave(group_size, dim=1)


def write_checkpoint(path: Path, model: Model, group_size=32, config=None):
    """HF-order packed checkpoint for every projection; returns {fork name: dense HF-order values}."""
    generator = torch.Generator().manual_seed(11)
    tensors, values = {}, {}
    for name, parameter in model.parameters.items():
        if not (name.startswith("text/layers/") and parameter.projection):
            continue
        _, _, layer, role = name.split("/", 3)
        n, k = parameter.shape
        codes, scales = random_matrix(generator, n, k, group_size)
        prefix = f"model.layers.{layer}.{HF_ROLE[role]}"
        tensors[prefix + ".weight_packed"] = pack(codes)
        tensors[prefix + ".weight_scale"] = scales
        tensors[prefix + ".weight_shape"] = torch.tensor([n, k], dtype=torch.int64)
        values[name] = (codes, scales, dense(codes, scales, group_size))
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(config or {"quantization_config": quantization_config(group_size=group_size)}))
    return values


def fork_order(name: str, hf: torch.Tensor) -> torch.Tensor:
    """What the converter's own value path (`_permute_rows/_permute_cols`) produces for the same HF-order values."""
    _, _, layer, role = name.split("/", 3)
    if KINDS[int(layer)] != "full_attention":
        return hf
    perm = prope_perm(GLOBAL_DIM, CONFIG["partial_rotary_factor"])
    source = array_source(hf, name)
    if role == "attention/query":
        return _permute_rows(source, HEADS, perm).values().reshape(hf.shape)
    if role == "attention/key":
        return _permute_rows(source, KV, perm).values().reshape(hf.shape)
    if role == "attention/output":
        return _permute_cols(source, HEADS, perm).values().reshape(hf.shape)
    return hf


def written(tmp_path: Path, built, model: Model, name: str = "toy.ninfer"):
    prepared = built.recipe.prepare(device="cpu", rows_per_chunk=3)
    path = tmp_path / name
    specs = [job.spec for job in prepared.weights]
    with ArtifactWriter(
        path, specs, components=model.components, bindings=prepared.bindings, uses=prepared.uses
    ) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    return path, prepared


@pytest.mark.parametrize("weights", ["q4_g32_a16", "q4_g32"])
def test_import_reaches_the_artifact_unchanged_with_global_permutations(tmp_path, weights):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model)
    built = build_recipe(
        model, None, Q4Options(weights=weights, gptq_source=tmp_path / "gptq"), device="cpu", rows_per_chunk=3
    )
    assert built.scale_search["method"] == "external_gptq_import" and built.scale_search["group_size"] == 32
    path, prepared = written(tmp_path, built, model)
    seen = set()
    with Artifact(path) as artifact:
        for job in prepared.weights:
            name = job.parameters[0]
            if job.spec.format != "q4_g32_fp16":
                continue
            assert job.method_name == "import_encoded"
            for member in job.parameters:
                seen.add(member)
            payload = artifact.read_object(job.spec.id)
            decoded = dequantize_row_split(payload, "q4_g32_fp16", job.spec.shape, dtype=torch.float32)
            scales, codes = decode_row_split_codes(payload, "q4_g32_fp16", job.spec.shape)
            row = 0
            for member in job.parameters:
                n = model.parameters[member].shape[0]
                want = fork_order(member, values[member][2])
                assert torch.equal(decoded[row : row + n], want), member
                row += n
            assert codes.dtype == torch.int8
    assert seen == set(values)


def test_codes_and_scales_are_the_checkpoints_own_words(tmp_path):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model)
    sources, group, _ = gptq_projection_sources(model, tmp_path / "gptq")
    assert group == 32
    name = "text/layers/1/attention/output"  # global o: columns permuted in 32-groups, scales follow the codes
    codes, scales, _ = values[name]
    perm = prope_perm(GLOBAL_DIM, CONFIG["partial_rotary_factor"])
    groups = (torch.arange(HEADS)[:, None] * GLOBAL_DIM + perm[None, ::32]) // 32
    words = sources[name].read_encoded(0, HIDDEN)
    assert words.format == "q4_g32_fp16" and words.weight_divisor is None
    assert words.codes.dtype == torch.int8 and words.scales.dtype == torch.float16
    assert torch.equal(words.codes, codes.reshape(HIDDEN, -1, 32)[:, groups.reshape(-1)])
    assert torch.equal(words.scales.float(), scales.float()[:, groups.reshape(-1)])
    # a plain sliding projection is copied verbatim
    codes, scales, _ = values["text/layers/0/mlp/down"]
    words = sources["text/layers/0/mlp/down"].read_encoded(5, 9)
    assert torch.equal(words.codes, codes.reshape(HIDDEN, -1, 32)[5:9])
    assert torch.equal(words.scales.float(), scales.float()[5:9])


def test_global_rows_follow_the_fork_row_permutation(tmp_path):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model)
    sources, _, _ = gptq_projection_sources(model, tmp_path / "gptq")
    for name in ("text/layers/1/attention/query", "text/layers/1/attention/key"):
        words = sources[name].read_encoded(0, model.parameters[name].shape[0])
        got = words.codes.float() * words.scales.float().unsqueeze(-1)
        assert torch.equal(got.reshape(got.shape[0], -1), fork_order(name, values[name][2]))
        assert torch.equal(sources[name].values().reshape(got.shape[0], -1), fork_order(name, values[name][2]))


@pytest.mark.parametrize("group_size", [64, 128])
def test_coarse_source_groups_repeat_their_scale_exactly(tmp_path, group_size):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model, group_size=group_size)
    sources, group, _ = gptq_projection_sources(model, tmp_path / "gptq")
    assert group == group_size
    for name, source in sources.items():
        words = source.read_encoded(0, source.shape[0])
        got = words.codes.float() * words.scales.float().unsqueeze(-1)
        assert torch.equal(got.reshape(got.shape[0], -1), fork_order(name, values[name][2]))
    built = build_recipe(
        model, None, Q4Options(weights="q4_g32_a16", gptq_source=tmp_path / "gptq"), device="cpu", rows_per_chunk=3
    )
    assert set(built.groups.values()) == {group_size}


def test_k_is_padded_to_128_with_zero_groups(tmp_path):
    generator = torch.Generator().manual_seed(3)
    codes, scales = random_matrix(generator, 5, 96)
    save_file(
        {
            "m.weight_packed": pack(codes),
            "m.weight_scale": scales,
            "m.weight_shape": torch.tensor([5, 96], dtype=torch.int64),
        },
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        source = packed_int4_matrix_source(store, "m", (5, 96), 32)
        words = source.read_encoded(0, 5)
        assert words.codes.shape == (5, 4, 32) and words.scales.shape == (5, 4)
        assert not bool(words.codes[:, 3].any()) and not bool(words.scales[:, 3].view(torch.int16).any())
        assert torch.equal(source.values().reshape(5, 96), dense(codes, scales))
        assert torch.equal(source.values(7, 7), torch.empty(0))
        assert torch.equal(source.values(100, 200), dense(codes, scales).reshape(-1)[100:200])


def test_unpack_matches_a_direct_nibble_reading():
    codes = torch.arange(-8, 8, dtype=torch.int8).repeat(2).reshape(1, 32)
    words = pack(codes)
    # word 0 holds codes -8..-1 as unsigned 0..7
    assert (words.view(torch.uint8)[0, :4].tolist()) == [0x10, 0x32, 0x54, 0x76]
    assert torch.equal(unpack_codes(words), codes)


def refusal(tmp_path, extra=None, scales=None, config=None, match=""):
    model = toy_model()
    write_checkpoint(tmp_path / "gptq", model, config=config)
    if extra or scales is not None:
        file = tmp_path / "gptq" / "model.safetensors"
        tensors = load_file(file)
        tensors.update(extra or {})
        if scales is not None:
            tensors["model.layers.0.mlp.down_proj.weight_scale"] = scales
        save_file(tensors, str(file))
    with pytest.raises(ValueError, match=match):
        sources, _, _ = gptq_projection_sources(model, tmp_path / "gptq")
        for source in sources.values():
            source.read_encoded(0, 1)


def test_zero_points_and_group_index_are_refused(tmp_path):
    prefix = "model.layers.0.mlp.down_proj"
    refusal(tmp_path / "a", extra={prefix + ".weight_zero_point": torch.zeros(HIDDEN, 8, dtype=torch.int32)}, match="weight_zero_point")
    refusal(tmp_path / "b", extra={prefix + ".weight_g_idx": torch.zeros(INTER, dtype=torch.int32)}, match="weight_g_idx")


def test_scale_that_binary16_cannot_hold_is_an_error_not_a_rounding(tmp_path):
    scales = torch.full((HIDDEN, INTER // 32), 2.0**-30, dtype=torch.bfloat16)  # underflows binary16
    refusal(tmp_path, scales=scales, match="not exactly representable in binary16")
    big = torch.full((HIDDEN, INTER // 32), 2.0**20, dtype=torch.bfloat16)  # overflows binary16
    refusal(tmp_path / "big", scales=big, match="not exactly representable|finite")
    negative = torch.full((HIDDEN, INTER // 32), -0.5, dtype=torch.bfloat16)
    refusal(tmp_path / "neg", scales=negative, match="finite and nonnegative")


def test_wrong_shapes_and_dtypes_are_refused(tmp_path):
    refusal(tmp_path / "a", scales=torch.ones(HIDDEN, INTER // 32 + 1, dtype=torch.bfloat16), match="weight_scale")
    refusal(tmp_path / "b", scales=torch.ones(HIDDEN, INTER // 32, dtype=torch.int32), match="weight_scale")
    refusal(
        tmp_path / "c",
        extra={"model.layers.0.mlp.down_proj.weight_shape": torch.tensor([HIDDEN, 99], dtype=torch.int64)},
        match="weight_shape",
    )


@pytest.mark.parametrize(
    "weights",
    [
        {"num_bits": 8},
        {"symmetric": False},
        {"strategy": "channel"},
        {"actorder": "weight"},
        {"actorder": "group"},
        {"group_size": 48},
        {"type": "float"},
    ],
)
def test_unsupported_quantization_schemes_are_refused(weights):
    with pytest.raises(ValueError, match="unsupported weight scheme"):
        checkpoint_group_size({"quantization_config": quantization_config(**weights)})
    assert checkpoint_group_size({"quantization_config": quantization_config(group_size=128)}) == 128


def test_other_formats_are_refused():
    config = quantization_config()
    config["format"] = "int-quantized"
    with pytest.raises(ValueError, match="pack-quantized"):
        checkpoint_group_size({"quantization_config": config})
    with pytest.raises(ValueError, match="pack-quantized"):
        checkpoint_group_size({})


def test_a_column_permutation_that_splits_a_32_group_is_refused():
    # Gemma 4 31B: head 512, rotary pairs 64 -> rotated blocks [0,64) and [256,320): every boundary is a multiple of 32.
    column_group_index((torch.arange(2)[:, None] * 512 + prope_perm(512, 0.25)[None, :]).reshape(-1))
    # head 64 with 8 rotary pairs would split groups at 8 and 32+8
    with pytest.raises(ValueError, match="whole source 32-group"):
        column_group_index(prope_perm(64, 0.25))


def test_options_exclude_every_other_scale_choice(tmp_path):
    ok = Q4Options(weights="q4_g32_a16", gptq_source=tmp_path)
    assert ok.gptq_source == tmp_path
    for bad in (
        {"scale_search": "mse"},
        {"group_size": 128},
        {"region_groups": ("attn_out=32",)},
        {"weights": "fp8_row"},
    ):
        with pytest.raises(ValueError, match="gptq-source|only to the q4_g32"):
            Q4Options(**{"weights": "q4_g32_a16", "gptq_source": tmp_path, **bad})


def test_import_encoded_requires_complete_source_coverage(tmp_path):
    model = toy_model()
    write_checkpoint(tmp_path / "gptq", model)
    # Dropping one layer's tensors from the checkpoint is a loud error, not a silent fallback to another recipe.
    file = tmp_path / "gptq" / "model.safetensors"
    tensors = {k: v for k, v in load_file(file).items() if ".layers.1.mlp.up_proj." not in k}
    save_file(tensors, str(file))
    with pytest.raises(ValueError, match="missing source tensor"):
        gptq_projection_sources(model, tmp_path / "gptq")


def test_the_global_permutations_are_not_the_identity_on_the_toy_geometry():
    generator = torch.Generator().manual_seed(5)
    for role, shape in (("attention/query", (HEADS * GLOBAL_DIM, HIDDEN)), ("attention/key", (KV * GLOBAL_DIM, HIDDEN)), ("attention/output", (HIDDEN, HEADS * GLOBAL_DIM))):
        hf = torch.randn(*shape, generator=generator)
        assert not torch.equal(fork_order(f"text/layers/1/{role}", hf), hf)
        assert torch.equal(fork_order(f"text/layers/0/{role}", hf), hf)


# --- native coarse groups (q4_g128_fp16 / q4_g256_fp16) ---------------------------------------------------------


def decoded_by_member(tmp_path, built, model, name):
    """{member parameter: (format, decoded float32 rows)} of a written artifact."""
    path, prepared = written(tmp_path, built, model, name)
    result = {}
    with Artifact(path) as artifact:
        for job in prepared.weights:
            if not job.spec.format.startswith("q4_g"):
                continue
            payload = artifact.read_object(job.spec.id)
            decoded = dequantize_row_split(payload, job.spec.format, job.spec.shape, dtype=torch.float32)
            row = 0
            for member in job.parameters:
                n = model.parameters[member].shape[0]
                result[member] = (job.spec.format, decoded[row : row + n], job.spec.shape)
                row += n
    return result


@pytest.mark.parametrize("group_size", [128, 256])
@pytest.mark.parametrize("weights", ["q4_g32_a16", "q4_g32"])
def test_native_import_stores_real_coarse_formats_and_decodes_like_the_emulation(tmp_path, group_size, weights):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model, group_size=group_size)
    options = {"weights": weights, "gptq_source": tmp_path / "gptq"}
    native = build_recipe(model, None, Q4Options(native_groups=True, **options), device="cpu", rows_per_chunk=3)
    emulated = build_recipe(model, None, Q4Options(**options), device="cpu", rows_per_chunk=3)
    got_native = decoded_by_member(tmp_path, native, model, "native.ninfer")
    got_emulated = decoded_by_member(tmp_path, emulated, model, "emulated.ninfer")
    keep = {"text/layers/1/attention/output"}  # the only tensor whose column permutation splits a source group
    for member, (fmt, decoded, _) in got_native.items():
        want = fork_order(member, values[member][2])
        assert torch.equal(decoded, want), member  # bit identity with the checkpoint, and with the emulation:
        assert torch.equal(decoded, got_emulated[member][1]), member
        assert fmt == ("q4_g32_fp16" if member in keep else f"q4_g{group_size}_fp16"), member
        assert got_emulated[member][0] == "q4_g32_fp16"
    assert native.scale_search["native"] is True and native.scale_search["kept_g32_storage"] == sorted(keep)
    assert native.scale_search["storage_formats"] == {
        f"q4_g{group_size}_fp16": len(values) - 1,
        "q4_g32_fp16": 1,
    }
    assert native.payload < emulated.payload  # the scale planes really shrank
    # W4A8 permission reaches every projection, native ones included
    allowed = {name for (name, _), policy in native.recipe.policies.items() if policy == "AllowA8"}
    assert allowed == (set(values) if weights == "q4_g32" else set())


@pytest.mark.parametrize("group_size", [128, 256])
def test_native_source_words_are_the_checkpoints_own(tmp_path, group_size):
    model = toy_model()
    values = write_checkpoint(tmp_path / "gptq", model, group_size=group_size)
    sources, group, formats = gptq_projection_sources(model, tmp_path / "gptq", native=True)
    assert group == group_size
    name = "text/layers/0/mlp/down"
    codes, scales, _ = values[name]
    words = sources[name].read_encoded(2, 7)
    assert words.format == f"q4_g{group_size}_fp16" == formats[name]
    assert words.codes.shape == (5, INTER // group_size, group_size) and words.scales.shape == (5, INTER // group_size)
    assert torch.equal(words.codes, codes.reshape(HIDDEN, -1, group_size)[2:7])
    assert torch.equal(words.scales.float(), scales.float()[2:7])
    # global q/k: rows permuted, groups untouched
    q = "text/layers/1/attention/query"
    words = sources[q].read_encoded(0, model.parameters[q].shape[0])
    got = (words.codes.float() * words.scales.float().unsqueeze(-1)).reshape(words.codes.shape[0], -1)
    assert torch.equal(got, fork_order(q, values[q][2]))
    assert formats[q] == f"q4_g{group_size}_fp16"
    assert formats["text/layers/1/attention/output"] == "q4_g32_fp16"


def test_native_refusals(tmp_path):
    model = toy_model()
    write_checkpoint(tmp_path / "g64", model, group_size=64)
    with pytest.raises(ValueError, match="no Gemma g64 kernels"):
        gptq_projection_sources(model, tmp_path / "g64", native=True)
    # K must be a multiple of the group size (384 is not a multiple of 256) ...
    generator = torch.Generator().manual_seed(4)
    codes, scales = random_matrix(generator, 4, 384, 128)
    save_file(
        {"m.weight_packed": pack(codes), "m.weight_scale": scales, "m.weight_shape": torch.tensor([4, 384], dtype=torch.int64)},
        str(tmp_path / "model.safetensors"),
    )
    with SafetensorsSource(tmp_path) as store:
        with pytest.raises(ValueError, match="not a multiple of the group size 256"):
            packed_int4_matrix_source(store, "m", (4, 384), 256, native=True)
        # a g128 source of the same K is fine natively
        source = packed_int4_matrix_source(store, "m", (4, 384), 128, native=True)
        assert source.read_encoded(0, 4).format == "q4_g128_fp16"
    # the tensors a column permutation splits cannot be native; the importer refuses to store them so
    perm_cols = (torch.arange(2)[:, None] * 512 + prope_perm(512, 0.25)[None, :]).reshape(-1)
    for width in (128, 256):
        with pytest.raises(ValueError, match="whole source"):
            column_group_index(perm_cols, width)
    column_group_index(perm_cols, 32)


def test_native_options_are_allowed_with_a_gptq_source(tmp_path):
    assert Q4Options(weights="q4_g32_a16", native_groups=True, gptq_source=tmp_path).native_groups
    with pytest.raises(ValueError, match="mutually exclusive"):
        Q4Options(weights="q4_g32_a16", native_groups=True, group_size=256, gptq_source=tmp_path)
