"""Gemma 4 31B Q4 conversion boundaries; no GPU and no checkpoint required."""

from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from tools.convert import gemma4_q4
from tools.convert.gemma4 import prope_perm
from tools.convert.gemma4_q4 import (
    IMATRIX_ROLES,
    Q4_REGIONS,
    Q4Options,
    build_recipe,
    check_assistant,
    importance_vectors,
    normalize_config,
    parse_region_groups,
    q4_group_for,
    read_imatrix,
    select_components,
)
from tools.convert.methods import (
    grouped_absmax,
    grouped_absmax_emulated,
    grouped_mse_search,
)
from tools.convert.model import Model, Parameter
from tools.convert.quantization.importance import clear_importance, importance_for
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source


def checkpoint():
    kinds = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
    return {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {
            "hidden_size": 5376,
            "num_hidden_layers": 60,
            "intermediate_size": 21504,
            "head_dim": 256,
            "num_key_value_heads": 16,
            "layer_types": kinds,
            "per_layer_config": {
                f"{i:02}": {"head_dim": 512, "num_key_value_heads": 4}
                for i, kind in enumerate(kinds)
                if kind == "full_attention"
            },
        },
    }


def test_per_layer_global_geometry_preserves_source_and_sliding_geometry():
    source = checkpoint()
    before = deepcopy(source)
    normalized = normalize_config(source)["text_config"]
    assert (normalized["global_head_dim"], normalized["num_global_key_value_heads"]) == (512, 4)
    assert (normalized["head_dim"], normalized["num_key_value_heads"]) == (256, 16)
    assert source == before


def test_explicit_global_geometry_needs_no_per_layer_overrides():
    source = checkpoint()
    text = source["text_config"]
    del text["per_layer_config"]
    text.update(global_head_dim=512, num_global_key_value_heads=4)
    assert normalize_config(source)["text_config"] == text


def test_nonuniform_global_geometry_refuses_instead_of_mispacking_weights():
    source = checkpoint()
    source["text_config"]["per_layer_config"]["11"]["head_dim"] = 256
    with pytest.raises(ValueError, match="uniform geometry"):
        normalize_config(source)


def test_conflicting_global_fields_are_not_silently_overwritten():
    source = checkpoint()
    source["text_config"]["global_head_dim"] = 256
    with pytest.raises(ValueError, match="Conflicting global_head_dim"):
        normalize_config(source)


@pytest.mark.parametrize("change", ["moe", "wrong-size", "quantized"])
def test_unsupported_source_refuses_before_conversion(change):
    source = checkpoint()
    if change == "moe":
        source["text_config"]["enable_moe_block"] = True
    elif change == "wrong-size":
        source["text_config"]["hidden_size"] = 3840
    else:
        source["quantization_config"] = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="unsupported|requires|unquantized"):
        normalize_config(source)


def assistant_dir(tmp_path, **overrides):
    config = {"architectures": ["Gemma4AssistantForCausalLM"], "backbone_hidden_size": 5376, **overrides}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


def test_good_assistant_is_accepted(tmp_path):
    assert check_assistant(assistant_dir(tmp_path))["backbone_hidden_size"] == 5376


def test_missing_assistant_names_the_download_and_opt_out(tmp_path):
    with pytest.raises(ValueError, match="hf download google/gemma-4-31B-it-assistant.*--no-mtp"):
        check_assistant(tmp_path)


@pytest.mark.parametrize(
    "override", [{"architectures": ["Gemma4ForCausalLM"]}, {"backbone_hidden_size": 3840}]
)
def test_wrong_assistant_refuses_before_conversion(tmp_path, override):
    with pytest.raises(ValueError, match="assistant"):
        check_assistant(assistant_dir(tmp_path, **override))


def test_text_only_build_keeps_mtp_but_drops_vision():
    assert select_components(True, True, True) == ("text", "mtp")
    assert select_components(True, False, False) == ("text", "vision")
    assert select_components(False, False, False) == ("text",)


KINDS = checkpoint()["text_config"]["layer_types"]


def group(name, regions, default=32):
    return q4_group_for(name, KINDS, parse_region_groups(regions), default)


def test_region_membership_and_first_match_wins():
    assert group("text/layers/5/mlp/gate", ["mlp_gate_up=128"]) == 128
    assert group("text/layers/5/mlp/down", ["mlp_gate_up=128"]) == 32
    assert group("text/layers/5/attention/key", ["global_layers=256", "attn_qkv=64"]) == 256  # layer 5 is global
    assert group("text/layers/4/attention/key", ["global_layers=256", "attn_qkv=64"]) == 64
    assert group("text/layers/3/mlp/up", ["first4_layers=128"]) == 128
    assert group("text/layers/4/mlp/up", ["first4_layers=128"]) == 32
    assert group("text/layers/56/attention/output", ["last4_layers=128"]) == 128
    assert group("text/layers/55/attention/output", ["last4_layers=128"]) == 32
    assert group("text/layers/59/attention/output", ["last4_layers=128"], default=64) == 128
    assert group("text/layers/20/attention/output", ["last4_layers=128"], default=64) == 64


def test_packed_parents_always_share_a_group():
    """Gate/up and q/k/v are packed into one parent, so every region must give each group one size."""
    for region in Q4_REGIONS:
        for layer in range(60):
            sizes = {
                role: group(f"text/layers/{layer}/{role}", [f"{region}=128"])
                for role in ("mlp/gate", "mlp/up", "attention/query", "attention/key", "attention/value")
            }
            assert sizes["mlp/gate"] == sizes["mlp/up"]
            assert sizes["attention/query"] == sizes["attention/key"] == sizes["attention/value"]


@pytest.mark.parametrize("item", ["mlp_down", "mlp_down=48", "bogus=128", "mlp_down=128x"])
def test_malformed_region_group_is_rejected(item):
    with pytest.raises(ValueError, match="region-group"):
        parse_region_groups([item])


def _stats(layer, role, width, value=1.0, count=4.0):
    return {
        f"blk.{layer}.{IMATRIX_ROLES[role]}.weight": (
            np.full(width, value) * np.arange(1, width + 1),
            np.array([count]),
        )
    }


def test_importance_is_sum2_over_counts_and_global_output_columns_follow_the_converter_permutation():
    perm = np.array([2, 0, 1, 3])  # heads=2, head_dim=4 -> 8 columns
    stats = {**_stats(0, "mlp/down", 8), **_stats(5, "attention/output", 8)}
    widths = {"text/layers/0/mlp/down": 8, "text/layers/5/attention/output": 8}
    got = importance_vectors(stats, widths, KINDS, heads=2, perm=perm)
    base = np.arange(1, 9) / 4.0
    assert np.array_equal(got["text/layers/0/mlp/down"], base.astype(np.float32))  # layer 0 is sliding: identity
    assert np.array_equal(
        got["text/layers/5/attention/output"],
        base.reshape(2, 4)[:, perm].reshape(-1).astype(np.float32),
    )  # layer 5 is global
    assert not np.array_equal(got["text/layers/5/attention/output"], base.astype(np.float32))


@pytest.mark.parametrize(
    "stats,message",
    [
        ({}, "no statistics"),
        (_stats(0, "mlp/down", 7), "expected 8"),
        (_stats(0, "mlp/down", 8, count=0.0), "positive count"),
        ({"blk.0.ffn_down.weight": (np.array([1.0, -1.0] + [0.0] * 6), np.array([1.0]))}, "non-negative"),
        ({"blk.0.ffn_down.weight": (np.array([np.nan] + [0.0] * 7), np.array([1.0]))}, "finite"),
    ],
)
def test_unusable_statistics_are_refused_not_replaced(stats, message):
    with pytest.raises(ValueError, match=message):
        importance_vectors(stats, {"text/layers/0/mlp/down": 8}, KINDS, heads=2, perm=np.arange(4))


def _write_imatrix(path, tensors, kind="imatrix"):
    gguf = pytest.importorskip("gguf")
    writer = gguf.GGUFWriter(str(path), "imatrix")
    writer.add_type(kind)
    for name, values in tensors.items():
        writer.add_tensor(name, np.asarray(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_read_imatrix_pairs_sums_with_counts(tmp_path):
    path = tmp_path / "m.imatrix.gguf"
    _write_imatrix(
        path, {"blk.0.ffn_down.weight.in_sum2": [1, 2, 3, 4], "blk.0.ffn_down.weight.counts": [2]}
    )
    sums, counts = read_imatrix(path)["blk.0.ffn_down.weight"]
    assert sums.tolist() == [1, 2, 3, 4] and counts.tolist() == [2]


def test_read_imatrix_refuses_sums_without_counts_and_foreign_gguf(tmp_path):
    lone = tmp_path / "lone.gguf"
    _write_imatrix(lone, {"blk.0.ffn_down.weight.in_sum2": [1, 2]})
    with pytest.raises(ValueError, match="in_sum2 without counts"):
        read_imatrix(lone)
    foreign = tmp_path / "foreign.gguf"
    _write_imatrix(foreign, {"x.in_sum2": [1], "x.counts": [1]}, kind="model")
    with pytest.raises(ValueError, match="not a GGUF imatrix"):
        read_imatrix(foreign)


def test_read_imatrix_without_gguf_says_how_to_install(tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "gguf", None)
    with pytest.raises(ValueError, match="pip install gguf"):
        read_imatrix(tmp_path / "missing.gguf")


# --- recipe composition on a small Gemma-like model -------------------------------------------------

LAYERS = 12
WIDTH = 512  # every projection input; K padded to 128 must be a multiple of 256 for q4_g256_fp16
HEADS, GLOBAL_HEAD_DIM = 64, 8  # heads * global_head_dim == attention/output input width
PROJECTIONS = (
    "attention/query",
    "attention/key",
    "attention/value",
    "attention/output",
    "mlp/gate",
    "mlp/up",
    "mlp/down",
)


def gemma_model():
    kinds = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(LAYERS)]
    config = {
        "model_type": "gemma4_text",
        "num_hidden_layers": LAYERS,
        "layer_types": kinds,
        "num_attention_heads": HEADS,
        "global_head_dim": GLOBAL_HEAD_DIM,
        "num_global_key_value_heads": 4,
        "partial_rotary_factor": 0.5,
    }
    model = Model({"text": {"config": config}})

    def add(name, shape, inputs=()):
        values = torch.randn(*shape).to(torch.bfloat16)
        model.add(Parameter(name, shape, array_source(values, name), inputs=inputs))

    add("text/token_embedding", (8, WIDTH))
    add("text/output_head", (8, WIDTH))
    add("text/final_norm", (WIDTH,))
    for layer in range(LAYERS):
        for role in PROJECTIONS:
            add(f"text/layers/{layer}/{role}", (4, WIDTH), inputs=("input",))
        add(f"text/layers/{layer}/norm", (WIDTH,))
    model.packing_groups = []
    return model, kinds


def projections(model):
    return [name for name, p in model.parameters.items() if name.startswith("text/layers/") and p.projection]


def selection(build, name):
    (only,) = build.recipe.selections[name]
    return only


def build(options, **kwargs):
    model, kinds = gemma_model()
    return model, kinds, build_recipe(model, None, options, device="cpu", rows_per_chunk=512, **kwargs)


@pytest.fixture(autouse=True)
def _no_importance_leak():
    yield
    clear_importance()


def test_emulated_groups_keep_g32_storage_with_region_overrides_in_command_order():
    options = Q4Options(
        weights="q4_g32_a16", group_size=128, region_groups=("attn_out=32", "global_layers=256")
    )
    model, kinds, built = build(options)
    assert built.name == "q4_g32_absmax_fp8_vocab_a16_virtual_groups"
    for name in projections(model):
        layer, role = int(name.split("/")[2]), name.split("/", 3)[3]
        want = 32 if role == "attention/output" else 256 if kinds[layer] == "full_attention" else 128
        assert built.groups[name] == want
        chosen = selection(built, name)
        assert chosen.format == "q4_g32_fp16" and chosen.layout is None
        if want == 32:
            assert chosen.method is grouped_absmax and chosen.parameters == {}
        else:
            assert chosen.method is grouped_absmax_emulated
            assert chosen.parameters == {"group_size": want}
    sizes = {str(size): list(built.groups.values()).count(size) for size in (32, 128, 256)}
    assert built.q4_groups["tensors_by_group"] == sizes
    assert built.q4_groups["default"] == 128 and built.q4_groups["native"] is False
    assert built.q4_groups["regions"] == ["attn_out=32", "global_layers=256"]
    assert built.q4_groups["projection_note"].startswith("scale-plane bytes only")
    assert built.q4_groups["projected_real_weight_bytes"] < built.payload
    assert built.scale_search == {"method": "absmax"}


def test_vocabulary_and_non_projections_stay_on_the_fp8_base():
    _, _, built = build(Q4Options(weights="q4_g32_a16", group_size=128))
    assert selection(built, "text/output_head").format == "fp8_e4m3fn_row_bf16"
    assert selection(built, "text/token_embedding").format == "fp8_e4m3fn_row_bf16"
    assert selection(built, "text/layers/0/norm").format == "bf16"


def test_native_groups_store_the_real_coarse_format_and_project_the_real_size():
    options = Q4Options(
        weights="q4_g32_a16", group_size=256, region_groups=("mlp_down=128", "attn_out=32"), native_groups=True
    )
    model, _, built = build(options)
    assert built.name == "q4_g32_absmax_fp8_vocab_a16_native_groups"
    for name in projections(model):
        role = name.split("/", 3)[3]
        want = {"mlp/down": 128, "attention/output": 32}.get(role, 256)
        chosen = selection(built, name)
        assert built.groups[name] == want
        assert chosen.format == f"q4_g{want}_fp16"
        if want == 32:
            assert chosen.method is grouped_absmax
        else:
            assert chosen.method is grouped_absmax_emulated and chosen.parameters == {"group_size": want}
    assert built.q4_groups["native"] is True
    assert built.q4_groups["projection_note"] == "weight_bytes is the real native size"
    assert built.q4_groups["projected_real_weight_bytes"] == built.payload


def test_native_groups_refuse_group_64():
    with pytest.raises(ValueError, match="no Gemma g64 kernels"):
        build(Q4Options(weights="q4_g32_a16", group_size=64, native_groups=True))
    build(Q4Options(weights="q4_g32_a16", group_size=64))  # emulated g64 is allowed


def allow_a8(built):
    return {name for (name, _), policy in built.recipe.policies.items() if policy == "AllowA8"}


@pytest.mark.parametrize("native", [False, True])
def test_w4a8_override_applies_only_to_q4_g32_weights_and_only_to_projections(native):
    options = {"group_size": 256, "native_groups": native}
    model, _, w4a8 = build(Q4Options(weights="q4_g32", **options))
    assert allow_a8(w4a8) == set(projections(model))
    assert w4a8.name.startswith("q4_g32_absmax_fp8_vocab_a8")
    _, _, a16 = build(Q4Options(weights="q4_g32_a16", **options))
    assert allow_a8(a16) == set()
    assert a16.name.startswith("q4_g32_absmax_fp8_vocab_a16")
    _, _, fp8 = build(Q4Options(weights="fp8_row"))
    assert allow_a8(fp8) == set()


def test_fp8_row_is_the_untouched_reference_recipe():
    model, _, built = build(Q4Options(weights="fp8_row"))
    assert built.name == "fp8_row_maxabs_a16_reference"
    assert built.groups == {} and built.q4_groups is None and built.scale_search is None
    assert {selection(built, name).format for name in projections(model)} == {"fp8_e4m3fn_row_bf16"}


def test_mse_search_is_unweighted_and_renames_the_recipe():
    options = Q4Options(
        weights="q4_g32_a16", group_size=128, region_groups=("attn_out=32",), scale_search="mse"
    )
    model, _, built = build(options)
    assert built.name == "q4_g32_mse_search_fp8_vocab_a16_virtual_groups"
    for name in projections(model):
        chosen = selection(built, name)
        assert chosen.method is grouped_mse_search and chosen.format == "q4_g32_fp16"
        assert chosen.parameters == {"group_size": built.groups[name], "weighted": False}
    assert built.scale_search == {"method": "mse"}


def test_weighted_search_registers_converter_ordered_importance_and_records_the_imatrix(
    tmp_path, monkeypatch
):
    imatrix = tmp_path / "m.imatrix.gguf"
    imatrix.write_bytes(b"imatrix bytes")
    model, kinds = gemma_model()
    columns = np.arange(1, WIDTH + 1, dtype=np.float64)
    stats = {}
    for layer in range(LAYERS):
        for role in PROJECTIONS:
            stats[f"blk.{layer}.{IMATRIX_ROLES[role]}.weight"] = (columns.copy(), np.array([4.0]))
    monkeypatch.setattr(gemma4_q4, "read_imatrix", lambda path: stats)
    options = Q4Options(
        weights="q4_g32_a16", group_size=256, native_groups=True, scale_search="weighted", imatrix=imatrix
    )
    built = build_recipe(model, None, options, device="cpu", rows_per_chunk=512)
    assert built.name == "q4_g32_weighted_search_fp8_vocab_a16_native_groups"
    for name in projections(model):
        chosen = selection(built, name)
        assert chosen.method is grouped_mse_search and chosen.format == "q4_g256_fp16"
        assert chosen.parameters == {"group_size": 256, "weighted": True}
    record = built.scale_search["imatrix"]
    assert record["path"] == str(imatrix.resolve())
    assert record["sha256"] == gemma4_q4.sha256_file(imatrix)
    base = (columns / 4.0).astype(np.float32)
    perm = prope_perm(GLOBAL_HEAD_DIM, 0.5).numpy()
    sliding = importance_for(("text/layers/0/attention/output",), WIDTH)
    glob = importance_for(("text/layers/5/attention/output",), WIDTH)
    assert torch.equal(sliding, torch.from_numpy(base))
    assert torch.equal(glob, torch.from_numpy(base.reshape(HEADS, -1)[:, perm].reshape(-1)))
    assert not torch.equal(glob, sliding)


def test_weighted_search_refuses_missing_statistics_for_a_projection(tmp_path, monkeypatch):
    imatrix = tmp_path / "m.imatrix.gguf"
    imatrix.write_bytes(b"x")
    monkeypatch.setattr(gemma4_q4, "read_imatrix", lambda path: {})
    options = Q4Options(weights="q4_g32_a16", scale_search="weighted", imatrix=imatrix)
    with pytest.raises(ValueError, match="no statistics"):
        build(options)


def test_imatrix_requires_weighted_and_weighted_requires_imatrix(tmp_path):
    with pytest.raises(ValueError, match="weighted requires --imatrix"):
        Q4Options(weights="q4_g32_a16", scale_search="weighted")
    for search in ("absmax", "mse"):
        with pytest.raises(ValueError, match="weighted requires --imatrix"):
            Q4Options(weights="q4_g32", scale_search=search, imatrix=tmp_path / "m.gguf")
    Q4Options(weights="q4_g32", scale_search="weighted", imatrix=tmp_path / "m.gguf")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"scale_search": "mse"}, "apply only to the q4_g32"),
        ({"imatrix": "m.gguf"}, "apply only to the q4_g32"),
        ({"native_groups": True}, "apply only to the q4_g32"),
        ({"group_size": 128}, "apply only to the q4_g32"),
        ({"region_groups": ("mlp_down=128",)}, "apply only to the q4_g32"),
    ],
)
def test_fp8_row_refuses_every_q4_option(kwargs, message):
    with pytest.raises(ValueError, match=message):
        Q4Options(weights="fp8_row", **kwargs)


def test_options_refuse_unknown_choices_and_malformed_regions():
    with pytest.raises(ValueError, match="--weights"):
        Q4Options(weights="q8")
    with pytest.raises(ValueError, match="--scale-search"):
        Q4Options(scale_search="entropy")
    with pytest.raises(ValueError, match="region-group"):
        Q4Options(region_groups=("mlp_down=48",))


def mtp_model():
    model, kinds = gemma_model()
    model.components["mtp"] = {"config": {}}

    def add(name, shape, inputs=()):
        values = torch.randn(*shape).to(torch.bfloat16)
        model.add(Parameter(name, shape, array_source(values, name), inputs=inputs))

    add("mtp/input_projection", (4, WIDTH), inputs=("stem",))
    add("mtp/output_head", (8, WIDTH), inputs=("final",))
    for role in PROJECTIONS:
        add(f"mtp/layers/0/{role}", (4, WIDTH), inputs=("input",))
    add("mtp/layers/0/input_norm", (WIDTH,))
    return model, kinds


def mtp_build(**options):
    model, _ = mtp_model()
    return model, build_recipe(model, None, Q4Options(weights="q4_g32", **options), device="cpu", rows_per_chunk=512)


@pytest.mark.parametrize(
    ("mode", "layer_format", "suffix"),
    [("bf16", "bf16", ""), ("fp8", "fp8_e4m3fn_row_bf16", "_mtp_fp8"), ("q4", "q4_g32_fp16", "_mtp_q4")],
)
def test_mtp_weights_change_only_the_assistant_layer_projections(mode, layer_format, suffix):
    model, built = mtp_build(mtp_weights=mode)
    layers = [name for name in model.parameters if name.startswith("mtp/layers/0/") and model.parameters[name].projection]
    assert {selection(built, name).format for name in layers} == {layer_format}
    # Head stays FP8, the outer projection and the norm stay direct BF16, whatever the mode.
    assert selection(built, "mtp/output_head").format == "fp8_e4m3fn_row_bf16"
    assert selection(built, "mtp/input_projection").format == "bf16"
    assert selection(built, "mtp/layers/0/input_norm").format == "bf16"
    assert built.name == "q4_g32_absmax_fp8_vocab_a8" + suffix
    # The assistant never gets A8; only text projections do.
    assert not any(name.startswith("mtp/") for name in allow_a8(built))
    assert allow_a8(built)


def test_mtp_weights_leave_the_text_recipe_identical():
    model, plain = mtp_build(mtp_weights="bf16")
    _, fp8 = mtp_build(mtp_weights="fp8")
    for name in projections(model):
        a, b = selection(plain, name), selection(fp8, name)
        assert (a.format, a.method, a.parameters) == (b.format, b.method, b.parameters)
    assert plain.payload > fp8.payload


def test_fp8_is_the_default_when_the_assistant_is_present_and_nothing_changes_without_it():
    model, default = mtp_build()
    layer = "mtp/layers/0/mlp/gate"
    assert selection(default, layer).format == "fp8_e4m3fn_row_bf16"
    assert default.name.endswith("_mtp_fp8")
    # No assistant: the default is silently inert (the recipe name and payload are the plain ones).
    _, _, no_mtp = build(Q4Options(weights="q4_g32"))
    assert "_mtp_" not in no_mtp.name
    # The reference recipe never touches the assistant by default.
    model, reference = mtp_model()[0], build_recipe(mtp_model()[0], None, Q4Options(weights="fp8_row"), device="cpu", rows_per_chunk=512)
    assert "_mtp_" not in reference.name and selection(reference, layer).format == "bf16"


def test_mtp_weights_need_the_assistant_and_a_q4_recipe():
    with pytest.raises(ValueError, match="needs the mtp component"):
        build(Q4Options(weights="q4_g32", mtp_weights="fp8"))
    with pytest.raises(ValueError, match="applies only to the q4_g32"):
        Q4Options(weights="fp8_row", mtp_weights="q4")
    with pytest.raises(ValueError, match="--mtp-weights"):
        Q4Options(mtp_weights="q8")


def test_two_level_scales_and_the_s6_format_are_refused():
    model, _ = gemma_model()
    name = "text/layers/0/mlp/down"
    recipe = Recipe(model)
    recipe.assign(
        name,
        format="q4_g32_fp16",
        layout="auto",
        method="grouped_mse_search",
        parameters={"group_size": 32, "scale_bits": 6},
    )
    with pytest.raises(ValueError, match="two-level"):
        recipe.prepare(device="cpu")
    recipe = Recipe(model)
    recipe.assign(
        name,
        format="q4_g32_s6_fp16",
        layout="auto",
        method="grouped_mse_search",
        parameters={"group_size": 32},
    )
    with pytest.raises(ValueError, match="q4_g32_s6_fp16"):
        recipe.prepare(device="cpu")
