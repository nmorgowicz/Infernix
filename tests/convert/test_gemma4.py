from __future__ import annotations

import json

import pytest
from safetensors.torch import save_file
import torch

from tools.artifact.reader import Artifact
from tools.convert.gemma4 import build_model, embed_normalizer, prope_perm
from tools.convert.official_recipes import gemma4_31b_bf16
from tools.convert.pipeline import convert
from tools.convert.recipe import Recipe
from tools.convert.sources.safetensors import SafetensorsSource


# Every packed parent needs N % 128 and K % 128 (row_split_k128_v1 pads K to 128).
DIMS = {"hidden": 128, "inter": 64, "dim": 128}


def _config(hidden=16, inter=32, dim=8, vision=False):
    config = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": hidden,
            "vocab_size": 8,
            "num_hidden_layers": 2,
            "max_position_embeddings": 128,
            "intermediate_size": inter,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "head_dim": dim,
            "global_head_dim": dim,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": 30.0,
            "tie_word_embeddings": True,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    }
    if vision:
        config.update(
            {
                "boi_token_id": 5,
                "eoi_token_id": 6,
                "image_token_id": 7,
                "vision_soft_tokens_per_image": 4,
                "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "patch_size": 2,
                    "pooling_kernel_size": 3,
                    "position_embedding_size": 8,
                    "hidden_activation": "gelu_pytorch_tanh",
                    "rms_norm_eps": 1e-6,
                    "standardize": True,
                    "rope_parameters": {"rope_theta": 100.0, "rope_type": "default"},
                },
            }
        )
    return config


def _checkpoint(path, hidden=16, inter=32, dim=8, vision=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(_config(hidden, inter, dim, vision=vision)))
    vocab = 8
    tensors = {
        "model.language_model.embed_tokens.weight": torch.arange(vocab * hidden)
        .reshape(vocab, hidden)
        .to(torch.bfloat16),
        "model.language_model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
    }
    for index, global_layer in ((0, False), (1, True)):
        src = f"model.language_model.layers.{index}."
        kv = 1 if global_layer else 2
        q_rows, k_rows = 2 * dim, kv * dim
        tensors[src + "self_attn.q_proj.weight"] = torch.arange(q_rows * hidden).reshape(
            q_rows, hidden
        ).to(torch.bfloat16)
        tensors[src + "self_attn.k_proj.weight"] = (
            torch.arange(k_rows * hidden).reshape(k_rows, hidden) + 3
        ).to(torch.bfloat16)
        if not global_layer:
            tensors[src + "self_attn.v_proj.weight"] = torch.arange(
                k_rows * hidden
            ).reshape(k_rows, hidden).to(torch.bfloat16)
        tensors[src + "self_attn.o_proj.weight"] = torch.arange(hidden * q_rows).reshape(
            hidden, q_rows
        ).to(torch.bfloat16)
        tensors[src + "self_attn.q_norm.weight"] = torch.arange(dim).to(torch.bfloat16)
        key_norm = (
            torch.full((dim,), 0.5, dtype=torch.bfloat16)
            if global_layer
            else torch.arange(dim).to(torch.bfloat16)
        )
        tensors[src + "self_attn.k_norm.weight"] = key_norm
        for role in ("gate", "up"):
            tensors[src + f"mlp.{role}_proj.weight"] = torch.ones(
                inter, hidden, dtype=torch.bfloat16
            )
        tensors[src + "mlp.down_proj.weight"] = torch.ones(
            hidden, inter, dtype=torch.bfloat16
        )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[src + name + ".weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        tensors[src + "layer_scalar"] = torch.tensor([0.25], dtype=torch.float32)
    if vision:
        v_hidden, v_inter, v_dim, patch, pos = 8, 16, 4, 2, 8
        pixels = 3 * patch * patch
        q_rows = 2 * v_dim
        tensors["model.embed_vision.embedding_projection.weight"] = torch.arange(
            hidden * v_hidden
        ).reshape(hidden, v_hidden).to(torch.bfloat16)
        tensors["model.vision_tower.patch_embedder.input_proj.weight"] = torch.arange(
            v_hidden * pixels
        ).reshape(v_hidden, pixels).to(torch.bfloat16)
        tensors["model.vision_tower.patch_embedder.position_embedding_table"] = torch.arange(
            2 * pos * v_hidden
        ).reshape(2, pos, v_hidden).to(torch.bfloat16)
        tensors["model.vision_tower.std_bias"] = torch.arange(v_hidden).to(torch.bfloat16)
        tensors["model.vision_tower.std_scale"] = (torch.arange(v_hidden) + 1).to(
            torch.bfloat16
        )
        src = "model.vision_tower.encoder.layers.0."
        tensors[src + "self_attn.q_proj.linear.weight"] = torch.arange(
            q_rows * v_hidden
        ).reshape(q_rows, v_hidden).to(torch.bfloat16)
        tensors[src + "self_attn.k_proj.linear.weight"] = (
            torch.arange(q_rows * v_hidden).reshape(q_rows, v_hidden) + 1
        ).to(torch.bfloat16)
        tensors[src + "self_attn.v_proj.linear.weight"] = (
            torch.arange(q_rows * v_hidden).reshape(q_rows, v_hidden) + 2
        ).to(torch.bfloat16)
        tensors[src + "self_attn.o_proj.linear.weight"] = torch.arange(
            v_hidden * q_rows
        ).reshape(v_hidden, q_rows).to(torch.bfloat16)
        tensors[src + "self_attn.q_norm.weight"] = torch.arange(v_dim).to(torch.bfloat16)
        tensors[src + "self_attn.k_norm.weight"] = torch.arange(v_dim).to(torch.bfloat16)
        for role in ("gate", "up"):
            tensors[src + f"mlp.{role}_proj.linear.weight"] = torch.ones(
                v_inter, v_hidden, dtype=torch.bfloat16
            )
        tensors[src + "mlp.down_proj.linear.weight"] = torch.ones(
            v_hidden, v_inter, dtype=torch.bfloat16
        )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[src + name + ".weight"] = torch.ones(v_hidden, dtype=torch.bfloat16)
    save_file(tensors, path / "model.safetensors")
    for role, value in {
        "tokenizer.json": {"model": {"vocab": {str(i): i for i in range(8)}}},
        "tokenizer_config.json": {},
        "generation_config.json": {},
    }.items():
        (path / role).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    if vision:
        (path / "processor_config.json").write_text(
            json.dumps(
                {
                    "image_processor": {
                        "patch_size": 2,
                        "pooling_kernel_size": 3,
                        "image_seq_length": 4,
                        "image_processor_type": "Gemma4ImageProcessor",
                    }
                }
            )
        )
    return SafetensorsSource(path)


def _values(model, name):
    parameter = model.parameters[name]
    return parameter.source.values().reshape(parameter.shape)


def test_mapping_scales_embedding_and_permutes_global_rope(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        scale = torch.tensor(embed_normalizer(16), dtype=torch.float32).to(
            torch.bfloat16
        )
        raw = source.read_flat(
            "model.language_model.embed_tokens.weight", 0, 8 * 16
        ).reshape(8, 16)
        assert torch.equal(_values(model, "text/token_embedding"), raw.to(torch.bfloat16) * scale)
        assert torch.equal(_values(model, "text/output_head"), raw)
        perm = prope_perm(8, 0.25)
        q = source.read_flat(
            "model.language_model.layers.1.self_attn.q_proj.weight", 0, 16 * 16
        ).reshape(2, 8, 16)
        got = _values(model, "text/layers/1/attention/query").reshape(2, 8, 16)
        assert torch.equal(got, q[:, perm, :])
        restored = got[:, torch.argsort(perm), :].reshape(16, 16)
        assert torch.equal(restored, q.reshape(16, 16))
        o = source.read_flat(
            "model.language_model.layers.1.self_attn.o_proj.weight", 0, 16 * 16
        ).reshape(16, 2, 8)
        got_o = _values(model, "text/layers/1/attention/output").reshape(16, 2, 8)
        assert torch.equal(got_o, o[:, :, perm])
        sliding = source.read_flat(
            "model.language_model.layers.0.self_attn.q_proj.weight", 0, 16 * 16
        ).reshape(16, 16)
        assert torch.equal(_values(model, "text/layers/0/attention/query"), sliding)
        assert "text/layers/1/attention/value" not in model.parameters
        assert model.config["global_k_norm_scalar"] == {"1": 0.5}
        assert model.config["prope_rotary_dim"] == 2
        assert model.parameters["text/layers/0/layer_scalar"].direct_format == "fp32"


def test_bf16_recipe_rereads_scaled_embedding(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        recipe = Recipe(model)
        gemma4_31b_bf16(model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=4)
    with Artifact(out) as artifact:
        assert artifact.directory.components["text"]["config"]["embed_normalizer"] == (
            embed_normalizer(16)
        )
        binding = artifact.directory.bindings["text/token_embedding"]
        obj = artifact.by_id[binding["object"]]
        raw = artifact.read_object(obj.id)
        stored = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(
            obj.shape
        )
        scale = torch.tensor(embed_normalizer(16), dtype=torch.float32).to(
            torch.bfloat16
        )
        expect = (
            torch.arange(8 * 16).reshape(8, 16).to(torch.bfloat16) * scale
        )
        assert torch.equal(stored, expect)


def _objects(artifact, name):
    binding = artifact.directory.bindings[name]
    if "object" in binding:
        return [(binding["object"], None)]
    return [(part["object"], tuple(part["range"])) for part in binding["parts"]]


def _packed(artifact, names):
    """Assert `names` fill one stored object in order and return that object."""
    parts = [_objects(artifact, name) for name in names]
    assert all(len(part) == 1 for part in parts)
    objects = {part[0][0] for part in parts}
    assert len(objects) == 1
    cursor = 0
    for part in parts:
        begin, end = part[0][1]
        assert begin == cursor
        cursor = end
    obj = artifact.by_id[objects.pop()]
    assert cursor == obj.shape[0] * obj.shape[1]
    return obj


def test_fp8_recipe_packs_qkv_and_gate_up_and_shares_the_vocabulary(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_fp8

    trunk = "fp8_e4m3fn_row_bf16"
    with _checkpoint(tmp_path / "source", **DIMS) as source:
        model = build_model(source)
        recipe = Recipe(model)
        gemma4_31b_fp8(model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        sliding = "text/layers/0/"
        qkv = _packed(artifact, [sliding + "attention/" + r for r in ("query", "key", "value")])
        assert qkv.format == trunk and tuple(qkv.shape) == (768, 128)
        global_qk = _packed(artifact, ["text/layers/1/attention/query", "text/layers/1/attention/key"])
        assert global_qk.format == trunk and tuple(global_qk.shape) == (384, 128)
        for layer in (0, 1):
            gate_up = _packed(artifact, [f"text/layers/{layer}/mlp/gate", f"text/layers/{layer}/mlp/up"])
            assert gate_up.format == trunk and tuple(gate_up.shape) == (128, 128)
            down = artifact.by_id[_objects(artifact, f"text/layers/{layer}/mlp/down")[0][0]]
            assert down.format == trunk
        for edge in ("text/token_embedding", "text/output_head"):
            assert artifact.by_id[_objects(artifact, edge)[0][0]].format == "fp8_e4m3fn_row_bf16"
        # One vocabulary table: the embedding binds the head's object.
        assert _objects(artifact, "text/token_embedding") == _objects(artifact, "text/output_head")
        norm = artifact.by_id[_objects(artifact, "text/final_norm")[0][0]]
        assert norm.format == "bf16"
        scalar = artifact.by_id[_objects(artifact, "text/layers/0/layer_scalar")[0][0]]
        assert scalar.format == "fp32" and tuple(scalar.shape) == (1,)


def test_the_chat_template_is_embedded_by_default_and_overridable(tmp_path):
    from tools.convert.gemma4 import DEFAULT_CHAT_TEMPLATE

    with _checkpoint(tmp_path / "source") as source:
        model = build_model(source)
        default = model.resources["resource/text/chat_template.jinja"]
        assert default == DEFAULT_CHAT_TEMPLATE.read_bytes()
        assert default != (source.root / "chat_template.jinja").read_bytes()
        assert model.components["text"]["resources"]["chat_template.jinja"] == "resource/text/chat_template.jinja"
        assert model.resources["resource/text/generation_config.json"] == b"{}"
        own = tmp_path / "own.jinja"
        own.write_bytes(b"{{ bos_token }}")
        model = build_model(source, resource_overrides={"chat_template.jinja": own})
        assert model.resources["resource/text/chat_template.jinja"] == b"{{ bos_token }}"
        with pytest.raises(ValueError, match="chat template not found"):
            build_model(source, resource_overrides={"chat_template.jinja": tmp_path / "missing.jinja"})


def test_a8_override_allows_a8_on_fp8_layer_projections(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_fp8, gemma4_fp8_a8

    with _checkpoint(tmp_path / "source", **DIMS) as source:
        model = build_model(source)
        recipe = Recipe(model)
        sources = {"base": source}
        gemma4_31b_fp8(model, recipe, sources)
        gemma4_fp8_a8(model, recipe, sources)
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        uses = artifact.directory.uses
        for use in uses:
            a8 = use["parameter"].startswith("text/layers/")
            assert use["activation_policy"] == ("AllowA8" if a8 else "A16Only")
            assert not use.get("auxiliaries")
        assert any(use["parameter"] == "text/output_head" for use in uses)


def _assistant_config(hidden=8, backbone=16, dim=8, window=32):
    return {
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": backbone,
        "use_ordered_embeddings": False,
        "tie_word_embeddings": True,
        "text_config": {
            "hidden_size": hidden,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "num_kv_shared_layers": 2,
            "head_dim": dim,
            "global_head_dim": dim,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": window,
            "vocab_size": 8,
            "hidden_activation": "gelu_pytorch_tanh",
            "rms_norm_eps": 1e-6,
            "final_logit_softcapping": None,
            "rope_parameters": {
                "full_attention": {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 1000000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
            },
        },
    }


def _assistant(path, **config):
    path.mkdir()
    settings = _assistant_config(**config)
    (path / "config.json").write_text(json.dumps(settings))
    text = settings["text_config"]
    hidden, dim = text["hidden_size"], text["head_dim"]
    backbone = settings["backbone_hidden_size"]
    tensors = {
        "model.embed_tokens.weight": torch.arange(8 * hidden).reshape(8, hidden).to(torch.bfloat16),
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "pre_projection.weight": torch.ones(hidden, 2 * backbone, dtype=torch.bfloat16),
        "post_projection.weight": torch.ones(backbone, hidden, dtype=torch.bfloat16),
    }
    for index in range(2):
        src = f"model.layers.{index}."
        q_rows = 2 * dim
        tensors[src + "self_attn.q_proj.weight"] = torch.arange(q_rows * hidden).reshape(
            q_rows, hidden
        ).to(torch.bfloat16)
        tensors[src + "self_attn.q_norm.weight"] = torch.arange(dim).to(torch.bfloat16)
        tensors[src + "self_attn.o_proj.weight"] = torch.arange(hidden * q_rows).reshape(
            hidden, q_rows
        ).to(torch.bfloat16)
        for role in ("gate", "up"):
            tensors[src + f"mlp.{role}_proj.weight"] = torch.ones(16, hidden, dtype=torch.bfloat16)
        tensors[src + "mlp.down_proj.weight"] = torch.ones(hidden, 16, dtype=torch.bfloat16)
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            tensors[src + name + ".weight"] = torch.ones(hidden, dtype=torch.bfloat16)
        tensors[src + "layer_scalar"] = torch.tensor([0.5], dtype=torch.bfloat16)
    save_file(tensors, path / "model.safetensors")
    return SafetensorsSource(path)


def test_assistant_reads_target_kv_and_permutes_global_queries(tmp_path):
    from tools.convert.official_recipes import gemma4_31b_fp8, gemma4_fp8_a8

    with _checkpoint(tmp_path / "source", **DIMS) as source, _assistant(
        tmp_path / "assistant", dim=128, backbone=128
    ) as assistant:
        model = build_model(source, components=("text", "mtp"), companions={"mtp": assistant})
        config = model.components["mtp"]["config"]
        assert config["kv_source_layers"] == [0, 1]
        assert config["backbone_hidden_size"] == 128
        assert config["prope_rotary_dim"] == model.config["prope_rotary_dim"]
        perm = prope_perm(128, 0.25)
        q = assistant.read_flat("model.layers.1.self_attn.q_proj.weight", 0, 256 * 8).reshape(
            2, 128, 8
        )
        got = _values(model, "mtp/layers/1/attention/query").reshape(2, 128, 8)
        assert torch.equal(got, q[:, perm, :])
        o = assistant.read_flat("model.layers.1.self_attn.o_proj.weight", 0, 8 * 256).reshape(
            8, 2, 128
        )
        got_o = _values(model, "mtp/layers/1/attention/output").reshape(8, 2, 128)
        assert torch.equal(got_o, o[:, :, perm])
        sliding = assistant.read_flat("model.layers.0.self_attn.q_proj.weight", 0, 256 * 8)
        assert torch.equal(_values(model, "mtp/layers/0/attention/query").reshape(-1), sliding)
        head = assistant.read_flat("model.embed_tokens.weight", 0, 64)
        assert torch.equal(_values(model, "mtp/output_head").reshape(-1), head)
        recipe = Recipe(model)
        sources = {"base": source, "mtp": assistant}
        gemma4_31b_fp8(model, recipe, sources)
        gemma4_fp8_a8(model, recipe, sources)
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=128)
    with Artifact(out) as artifact:
        assert artifact.directory.components["mtp"]["target"] == "text"
        assert artifact.by_id[_objects(artifact, "mtp/input_projection")[0][0]].format == "bf16"
        assert artifact.by_id[_objects(artifact, "mtp/output_head")[0][0]].format == (
            "fp8_e4m3fn_row_bf16"
        )
        assert artifact.by_id[_objects(artifact, "mtp/layers/1/attention/query")[0][0]].format == (
            "bf16"
        )
        _packed(artifact, ["mtp/layers/0/mlp/gate", "mtp/layers/0/mlp/up"])
        for use in artifact.directory.uses:
            if use["parameter"].startswith("mtp/"):
                assert use["activation_policy"] == "A16Only"


def test_assistant_must_match_target_attention(tmp_path):
    with _checkpoint(tmp_path / "source") as source, _assistant(
        tmp_path / "assistant", window=16
    ) as assistant:
        with pytest.raises(ValueError, match="sliding_window"):
            build_model(source, components=("text", "mtp"), companions={"mtp": assistant})


def test_mtp_component_requires_assistant_source(tmp_path):
    with _checkpoint(tmp_path / "source") as source:
        with pytest.raises(ValueError, match="source mtp"):
            build_model(source, components=("text", "mtp"))


def test_vision_maps_clippable_linears_and_constant_scalar(tmp_path):
    with _checkpoint(tmp_path / "source", vision=True) as source:
        model = build_model(source, components=("text", "vision"))
        cfg = model.components["vision"]["config"]
        assert cfg["patch_pixels"] == 12
        assert cfg["image_token_id"] == 7
        assert cfg["boi_token_id"] == 5
        assert cfg["eoi_token_id"] == 6
        assert cfg["attention_scale"] == 1.0
        assert cfg["text_hidden_size"] == 16
        raw = source.read_flat(
            "model.vision_tower.patch_embedder.input_proj.weight", 0, 8 * 12
        )
        assert torch.equal(_values(model, "vision/patch_projection").reshape(-1), raw)
        table = source.read_flat(
            "model.vision_tower.patch_embedder.position_embedding_table", 0, 2 * 8 * 8
        )
        assert torch.equal(_values(model, "vision/position_table").reshape(-1), table)
        q = source.read_flat(
            "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight",
            0,
            8 * 8,
        )
        assert torch.equal(_values(model, "vision/layers/0/attention/query").reshape(-1), q)
        proj = source.read_flat("model.embed_vision.embedding_projection.weight", 0, 16 * 8)
        assert torch.equal(_values(model, "vision/embedding_projection").reshape(-1), proj)
        scale = source.read_flat("model.vision_tower.std_scale", 0, 8)
        assert torch.equal(_values(model, "vision/std_scale").reshape(-1), scale)
        assert torch.equal(_values(model, "vision/layers/0/layer_scalar"), torch.ones(1))
        assert model.parameters["vision/layers/0/layer_scalar"].direct_format == "fp32"
        assert "vision/layers/0/attention/value" in model.parameters
        assert (
            "vision/layers/0/attention/query",
            "vision/layers/0/attention/key",
            "vision/layers/0/attention/value",
        ) in model.packing_groups
        assert (
            model.components["vision"]["resources"]["processor_config.json"]
            == "resource/vision/processor_config.json"
        )
        recipe = Recipe(model)
        gemma4_31b_bf16(model, recipe, {"base": source})
        out = tmp_path / "gemma.ninfer"
        convert(model, recipe, out, device="cpu", rows_per_chunk=4)
    with Artifact(out) as artifact:
        assert artifact.directory.components["vision"]["config"]["head_dim"] == 4
        stored = torch.frombuffer(
            bytearray(artifact.read_object(artifact.directory.bindings["vision/patch_projection"]["object"])),
            dtype=torch.bfloat16,
        ).reshape(8, 12)
        assert torch.equal(stored, torch.arange(8 * 12).reshape(8, 12).to(torch.bfloat16))
        scalar = torch.frombuffer(
            bytearray(artifact.read_object(artifact.directory.bindings["vision/layers/0/layer_scalar"]["object"])),
            dtype=torch.float32,
        )
        assert torch.equal(scalar, torch.ones(1))
        _packed(
            artifact,
            [
                "vision/layers/0/attention/query",
                "vision/layers/0/attention/key",
                "vision/layers/0/attention/value",
            ],
        )
        _packed(artifact, ["vision/layers/0/mlp/gate", "vision/layers/0/mlp/up"])


def test_vision_component_requires_processor(tmp_path):
    with _checkpoint(tmp_path / "source", vision=True) as source:
        (source.root / "processor_config.json").unlink()
        with pytest.raises(FileNotFoundError):
            build_model(source, components=("text", "vision"))


def test_vision_without_text_is_rejected(tmp_path):
    with _checkpoint(tmp_path / "source", vision=True) as source:
        with pytest.raises(ValueError, match="text, mtp, and vision"):
            build_model(source, components=("vision",))
