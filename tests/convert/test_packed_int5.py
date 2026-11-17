"""Native GPTQ5: independent dense-bitstream oracle, exact scales, permutations and artifact Uses."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.artifact.codecs.row_split import decode_row_split_codes, dequantize_row_split
from tools.artifact.reader import Artifact
from tools.artifact.tensor_output import TensorOutput
from tools.artifact.writer import ArtifactWriter
from tools.convert.gemma4 import prope_perm
from tools.convert.gemma4_gptq import checkpoint_group_size, gptq_projection_sources
from tools.convert.gemma4_q4 import Q4Options, build_recipe
from tools.convert.model import Model, Parameter
from tools.convert.sources.logical import array_source
from tools.convert.sources.packed_int import column_group_index, packed_matrix_source, unpack_codes
from tools.convert.sources.safetensors import SafetensorsSource

ROLES = {
    "attention/query": "self_attn.q_proj", "attention/key": "self_attn.k_proj",
    "attention/value": "self_attn.v_proj", "attention/output": "self_attn.o_proj",
    "mlp/gate": "mlp.gate_proj", "mlp/up": "mlp.up_proj", "mlp/down": "mlp.down_proj",
}
CONFIG = {
    "model_type": "gemma4_text", "num_hidden_layers": 2,
    "layer_types": ["sliding_attention", "full_attention"], "num_attention_heads": 2,
    "global_head_dim": 512, "num_global_key_value_heads": 1, "partial_rotary_factor": 0.25,
}


def pack_dense(codes):
    """Independent scalar oracle: one Python integer bitstream per row, no word-index gather."""
    rows = []
    width = (codes.shape[1] * 5 + 31) // 32
    for row in codes.tolist():
        stream = sum((code + 16) << (5 * index) for index, code in enumerate(row))
        raw = stream.to_bytes(width * 4, "little")
        rows.append([int.from_bytes(raw[i:i + 4], "little", signed=True) for i in range(0, len(raw), 4)])
    return torch.tensor(rows, dtype=torch.int32)


def model_and_checkpoint(path):
    model = Model({"text": {"config": dict(CONFIG)}})

    def add(name, shape, inputs=()):
        model.add(Parameter(name, shape, array_source(torch.zeros(shape, dtype=torch.bfloat16), name), inputs=inputs))

    add("text/token_embedding", (8, 256))
    add("text/output_head", (8, 256))
    add("text/final_norm", (256,))
    tensors, values = {}, {}
    generator = torch.Generator().manual_seed(51)
    for layer in range(2):
        for role, hf in ROLES.items():
            if layer == 1 and role == "attention/value":
                continue
            n, k = (256, 256)
            if layer == 1:
                if role == "attention/query":
                    n = 1024
                elif role == "attention/key":
                    n = 512
                elif role == "attention/output":
                    k = 1024
            name = f"text/layers/{layer}/{role}"
            add(name, (n, k), inputs=("input",))
            codes = torch.randint(-16, 16, (n, k), dtype=torch.int8, generator=generator)
            # Includes subnormal, non-BF16-exact and largest finite binary16 scales.
            scale_bits = torch.tensor([1, 0x03FF, 0x0400, 0x3555, 0x3C01, 0x7BFF], dtype=torch.int16)
            scales = scale_bits.repeat((n * (k // 64) + 5) // 6)[:n * (k // 64)].reshape(n, k // 64).view(torch.float16)
            prefix = f"model.layers.{layer}.{hf}"
            tensors[prefix + ".weight_packed"] = pack_dense(codes)
            tensors[prefix + ".weight_scale"] = scales
            tensors[prefix + ".weight_shape"] = torch.tensor([n, k], dtype=torch.int64)
            values[name] = (codes, scales)
        add(f"text/layers/{layer}/norm", (256,))
    model.packing_groups = []
    path.mkdir()
    save_file(tensors, str(path / "model.safetensors"))
    config = {
        "quantization_config": {
            "quant_method": "compressed-tensors", "format": "pack-quantized",
            "config_groups": {"group_0": {"targets": ["Linear"], "weights": {
                "num_bits": 5, "type": "int", "symmetric": True, "strategy": "group",
                "group_size": 64, "actorder": None,
            }}},
        },
        "infernix_gptq": {"compressed_tensors_version": "0.18.0", "packing": "dense_cross_word_int32"},
    }
    (path / "config.json").write_text(json.dumps(config))
    return model, values


def expected_words(name, codes, scales):
    perm = prope_perm(512, 0.25)
    role = name.split("/", 3)[3]
    if name.startswith("text/layers/1/"):
        if role in ("attention/query", "attention/key"):
            index = (torch.arange(codes.shape[0] // 512)[:, None] * 512 + perm).reshape(-1)
            codes, scales = codes[index], scales[index]
        elif role == "attention/output":
            index = (torch.arange(2)[:, None] * 512 + perm).reshape(-1)
            codes = codes[:, index]
            scales = scales[:, index[::64] // 64]
    return codes.reshape(codes.shape[0], -1, 64), scales


def test_fixed_cross_word_fixture_and_independent_full_code_range():
    words = torch.tensor([[-1073741824, -268435449, 1, 0, 0, 0, 0, 0, 0, -134217728]], dtype=torch.int32)
    expected = torch.full((1, 64), -16, dtype=torch.int8)
    expected[0, [6, 12, 63]] = 15
    assert torch.equal(unpack_codes(words, 5, columns=64), expected)
    codes = torch.arange(-16, 16, dtype=torch.int8).repeat(4).reshape(2, 64)
    assert torch.equal(unpack_codes(pack_dense(codes), 5, columns=64), codes)


@pytest.mark.parametrize("weights", ["q5_g64", "q5_g64_a16"])
def test_native_q5_exact_words_and_artifact_for_every_projection(tmp_path, weights):
    root = tmp_path / "gptq"
    model, values = model_and_checkpoint(root)
    sources, group, formats = gptq_projection_sources(model, root, bits=5)
    assert group == 64 and set(formats.values()) == {"q5_g64_fp16"}
    for name, (codes, scales) in values.items():
        want_codes, want_scales = expected_words(name, codes, scales)
        words = sources[name].read_encoded(0, codes.shape[0])
        assert torch.equal(words.codes, want_codes), name
        assert torch.equal(words.scales.view(torch.int16), want_scales.view(torch.int16)), name
    built = build_recipe(model, None, Q4Options(weights=weights, gptq_source=root), device="cpu", rows_per_chunk=7)
    activation = "a8" if weights == "q5_g64" else "a16"
    assert built.name == f"q5_g64_gptq_import_fp8_vocab_{activation}"
    assert built.q4_groups["native"] and built.q4_groups["default"] == 64
    assert built.scale_search["packing"] == "dense_cross_word_int32"
    assert built.scale_search["compressor"]["compressed_tensors_version"] == "0.18.0"
    prepared = built.recipe.prepare(device="cpu", rows_per_chunk=7)
    artifact_path = tmp_path / "q5.ninfer"
    with ArtifactWriter(artifact_path, [job.spec for job in prepared.weights], components=model.components,
                        bindings=prepared.bindings, uses=prepared.uses) as writer:
        for job in prepared.weights:
            job.prepared.produce(TensorOutput(writer, job.spec.id))
    seen = set()
    with Artifact(artifact_path) as artifact:
        for job in prepared.weights:
            if not any(member in values for member in job.parameters):
                continue
            assert job.spec.format == "q5_g64_fp16" and job.method_name == "import_encoded"
            payload = artifact.read_object(job.spec.id)
            scales, codes = decode_row_split_codes(payload, job.spec.format, job.spec.shape)
            decoded = dequantize_row_split(payload, job.spec.format, job.spec.shape, dtype=torch.float32)
            row = 0
            for member in job.parameters:
                want_codes, want_scales = expected_words(member, *values[member])
                n = want_codes.shape[0]
                assert torch.equal(codes[row:row + n], want_codes)
                assert torch.equal(scales[row:row + n].view(torch.int16), want_scales.view(torch.int16))
                want = (want_codes.float() * want_scales.float().unsqueeze(-1)).reshape(n, -1)
                assert torch.equal(decoded[row:row + n], want)
                row += n
                seen.add(member)
    assert seen == set(values)
    assert all(
        use["activation_policy"] == (
            "AllowA8" if weights == "q5_g64" and use["parameter"] in values else "A16Only"
        )
        for use in prepared.uses
    )
    for name in ("text/token_embedding", "text/output_head"):
        assert built.recipe.selections[name][0].format == "fp8_e4m3fn_row_bf16"


@pytest.mark.parametrize("part,replacement,match", [
    ("weight_packed", torch.zeros(256, 256 // 6 + 1, dtype=torch.int32), "dense cross-word"),
    ("weight_packed", torch.zeros(256, 40, dtype=torch.int64), "I32"),
    ("weight_shape", torch.tensor([256, 255], dtype=torch.int64), "stored shape"),
    ("weight_scale", torch.ones(256, 5, dtype=torch.float16), "weight_scale"),
    ("weight_scale", torch.full((256, 4), 2.0**-30, dtype=torch.float32), "not exactly representable"),
    ("weight_scale", torch.full((256, 4), float("nan")), "finite"),
    ("weight_zero_point", torch.zeros(256, 4, dtype=torch.int8), "weight_zero_point"),
    ("weight_g_idx", torch.arange(256), "weight_g_idx"),
])
def test_q5_refuses_incompatible_shapes_scales_and_layouts(tmp_path, part, replacement, match):
    root = tmp_path / "gptq"
    model, _ = model_and_checkpoint(root)
    path = root / "model.safetensors"
    tensors = load_file(path)
    tensors["model.layers.0.mlp.gate_proj." + part] = replacement
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match=match):
        sources, _, _ = gptq_projection_sources(model, root, bits=5)
        for source in sources.values():
            source.read_encoded(0, 1)


def test_wrong_bit_width_and_group_are_never_silently_requantized(tmp_path):
    root = tmp_path / "gptq"
    model, _ = model_and_checkpoint(root)
    with pytest.raises(ValueError, match="need 4-bit"):
        build_recipe(model, None, Q4Options(gptq_source=root), device="cpu", rows_per_chunk=7)
    config = json.loads((root / "config.json").read_text())
    weights = config["quantization_config"]["config_groups"]["group_0"]["weights"]
    weights["num_bits"] = 4
    with pytest.raises(ValueError, match="need 5-bit"):
        checkpoint_group_size(config, bits=5)
    weights.update(num_bits=5, group_size=32)
    with pytest.raises(ValueError, match="group_size in \\(64,\\)"):
        checkpoint_group_size(config, bits=5)


def test_q5_padding_and_group_preserving_permutation_contract(tmp_path):
    codes = torch.arange(-16, 16, dtype=torch.int8).repeat(2).reshape(1, 64)
    scales = torch.tensor([[0.125]], dtype=torch.float16)
    save_file({"m.weight_packed": pack_dense(codes), "m.weight_scale": scales,
               "m.weight_shape": torch.tensor([1, 64], dtype=torch.int64)}, str(tmp_path / "model.safetensors"))
    with SafetensorsSource(tmp_path) as store:
        source = packed_matrix_source(store, "m", (1, 64), 64, bits=5, native=True)
        words = source.read_encoded(0, 1)
        assert words.codes.shape == (1, 2, 64) and words.scales.shape == (1, 2)
        assert torch.equal(words.codes[:, 0], codes)
        assert not bool(words.codes[:, 1].any()) and not bool(words.scales[:, 1].view(torch.int16).any())
        assert torch.equal(source.values(), codes.float().reshape(-1) * 0.125)
        with pytest.raises(ValueError, match="multiple of the group size"):
            packed_matrix_source(store, "m", (1, 96), 64, bits=5, native=True)
    permutation = (torch.arange(2)[:, None] * 512 + prope_perm(512, 0.25)).reshape(-1)
    assert column_group_index(permutation, 64).numel() == 16
    with pytest.raises(ValueError, match="whole source 64-group"):
        column_group_index(prope_perm(128, 0.5), 64)


def test_checkpoint_inspection_uses_the_same_exact_q5_contract(tmp_path, capsys):
    from tools.gptq import check_checkpoint

    root = tmp_path / "gptq"
    model_and_checkpoint(root)
    assert check_checkpoint.main(["--gptq", str(root), "--bits", "5"]) == 0
    facts = json.loads(capsys.readouterr().out)
    assert facts["bits"] == 5 and facts["group_size"] == 64
    assert facts["packing"] == "dense_cross_word_int32"
    assert facts["scales_not_exact_in_fp16"] == 0
    with pytest.raises(ValueError, match="need 4-bit"):
        check_checkpoint.main(["--gptq", str(root)])
