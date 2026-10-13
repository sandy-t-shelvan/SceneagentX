"""Block D writers. A missing ply, key, or checkpoint stays unstarted."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.block_d import (
    PhysicsPrior,
    apply_scene_mean,
    build_prior_input,
    fuse_physics,
    parse_d2_response,
    record_isaac_drop_slide,
    reduce_prior,
    run_learned_prior,
    start_d2,
    write_d2_record,
    write_gaussian_physics,
    write_physics_material_usda,
    write_prior_input,
    write_scene_mean_labels,
    refresh_block_d,
)
from tools.block_gates import GateError, check_mass, lock_learned_prior


def test_d12_outside_gets_a_different_scene_mean():
    inside = np.array([True, False, True])
    obj = np.array([1.0, 2.0])
    scene = np.array([3.0, 4.0])
    result = apply_scene_mean(inside, obj, scene)
    assert result["scene_mean_measured"] is True
    assert np.allclose(result["assigned"][1], scene)
    assert result["index"].tolist() == [0, 1, 0]
    assert result["learned_prior"] is False


def test_d12_identical_vectors_stay_unmeasured():
    inside = np.array([True, False, True])
    vec = np.array([1.0, 2.0])
    result = apply_scene_mean(inside, vec, vec)
    assert result["scene_mean_measured"] is False
    assert result["index"].tolist() == [0, -1, 0]


def test_d12_write_labels_only_when_measured(tmp_path: Path):
    inside = np.array([True, False, False, True])
    written = write_scene_mean_labels(tmp_path, inside, np.array([0.1, 0.2]), np.array([0.4, 0.5]))
    index = np.load(tmp_path / "semantics" / "gaussian_feature_idx.npy")
    assert written["scene_mean_measured"] is True
    assert set(index.tolist()) == {0, 1}
    same = write_scene_mean_labels(tmp_path / "same", inside, np.array([0.1, 0.2]), np.array([0.1, 0.2]))
    same_index = np.load(tmp_path / "same" / "semantics" / "gaussian_feature_idx.npy")
    assert same["scene_mean_measured"] is False
    assert same_index.tolist() == [0, -1, -1, 0]


def test_d13_concat_width_and_row_count():
    vectors = np.ones((4, 32))
    xyz = np.arange(12, dtype=float).reshape(4, 3)
    size = np.array([0.1, 0.2, 0.3, 0.4])
    opacity = np.array([0.9, 0.8, 0.7, 0.6])
    colour = np.zeros((4, 3))
    stacked = build_prior_input(vectors, xyz, size, opacity, colour)
    assert stacked.shape == (4, 40)
    assert np.allclose(stacked[:, :32], 1.0)
    assert np.allclose(stacked[:, 32:35], xyz)


def test_d13_missing_ply_does_not_invent_a_file(tmp_path: Path):
    status = write_prior_input("can", tmp_path, ply=tmp_path / "missing.ply", vectors=np.ones((2, 32)))
    assert status["state"] == "unstarted"
    assert not (tmp_path / "data" / "can" / "semantics" / "prior_input.npy").is_file()


def test_d13_writes_only_when_ply_exists(tmp_path: Path):
    ply = tmp_path / "splat.ply"
    _write_splat(ply, n=3)
    vectors = np.arange(96, dtype=float).reshape(3, 32)
    status = write_prior_input("can", tmp_path, ply=ply, vectors=vectors)
    path = tmp_path / "data" / "can" / "semantics" / "prior_input.npy"
    assert status["state"] == "written"
    assert path.is_file()
    assert np.load(path).shape == (3, 40)


def test_d14_missing_checkpoint_raises(tmp_path: Path):
    x = np.ones((2, 40))
    with pytest.raises(GateError, match="checkpoint"):
        run_learned_prior(x, tmp_path / "missing.npz")
    with pytest.raises(GateError, match="checkpoint"):
        run_learned_prior(x, None)


def test_d14_fixture_forward_does_not_set_learned_prior(tmp_path: Path):
    x = np.ones((5, 40))
    ckpt = tmp_path / "prior.npz"
    PhysicsPrior(40).save(ckpt)
    out = run_learned_prior(x, ckpt)
    assert out["learned_prior"] is False
    assert out["p_i"]["material_logits"].shape == (5, 8)
    assert out["p_i"]["mass"].shape == (5,)
    assert out["p_i"]["friction"].shape == (5,)
    assert out["p_i"]["rigidity"].shape == (5,)
    assert np.all((out["p_i"]["mass"] >= 0.0) & (out["p_i"]["mass"] <= 1.0))
    locked = lock_learned_prior({"learned_prior": False, "label": "fixture forward"})
    assert locked["learned_prior"] is False
    with pytest.raises(GateError, match="learned_prior"):
        lock_learned_prior({"learned_prior": True})


def test_d15_object_mean_inherit_and_field_shapes():
    p_i = {
        "mass": np.array([1.0, 3.0, 5.0, 7.0]),
        "friction": np.array([0.2, 0.4, 0.6, 0.8]),
        "rigidity": np.array([0.9, 0.9, 0.1, 0.1]),
        "material_logits": np.zeros((4, 8)),
    }
    inside = np.array([True, True, False, False])
    reduced = reduce_prior(p_i, inside)
    assert reduced["object"]["mass"] == pytest.approx(2.0)
    assert reduced["parts"][0]["id"] == "body"
    assert reduced["parts"][0]["inherit"] is True
    assert reduced["parts"][0]["mass"] == pytest.approx(2.0)
    assert reduced["field"]["mass"].shape == (4,)
    assert reduced["learned_prior"] is False


def test_d2_no_key_does_not_write_invented_mass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    frames = [_tiny_png(tmp_path / f"frame_{index}.png") for index in range(4)]
    bbox = {"span_inside": 0.55, "span_scene": 9.2, "metres_per_unit": None}
    status = start_d2("can", frames, "can", bbox, root=tmp_path, api_key="")
    assert status == "model not asked"
    assert not (tmp_path / "data" / "can" / "physics.json").is_file()


def test_d2_parsed_fixture_schema_mass_unchecked(tmp_path: Path):
    frames = [_tiny_png(tmp_path / f"view_{index}.png") for index in range(4)]
    parsed = parse_d2_response(_fixture_d2())
    assert parsed["material"] == "aluminum"
    assert parsed["deformable_type"] == "rigid"
    bbox = {"span_inside": 0.555993, "span_scene": 9.292794, "metres_per_unit": None, "units": "colmap"}
    record = write_d2_record(parsed, "can", bbox, frames)
    assert record["learned_prior"] is False
    assert record["mass_checked"] is False
    assert record["mass_check"]["checked"] is False
    assert record["metres_per_unit"] is None
    assert check_mass(record["mass_kg"], None)["checked"] is False
    status = start_d2("can", frames, "can", bbox, root=tmp_path, ask=lambda *_args: _fixture_d2())
    assert status == "filled"
    saved = json.loads((tmp_path / "data" / "can" / "physics.json").read_text(encoding="utf-8"))
    assert saved["learned_prior"] is False
    assert saved["mass_check"]["checked"] is False
    assert saved["mass_kg"] != 0.015


def test_d2_unusable_response_does_not_write(tmp_path: Path):
    frames = [_tiny_png(tmp_path / f"bad_{index}.png") for index in range(4)]
    bbox = {"metres_per_unit": None}
    status = start_d2("can", frames, "can", bbox, root=tmp_path, ask=lambda *_args: {"material": "aluminum"})
    assert status == "response not usable"
    assert not (tmp_path / "data" / "can" / "physics.json").is_file()


def test_d3_fuse_without_prior_uses_vlm_and_missing_mass_stays_missing():
    vlm = {
        "material": "aluminum",
        "mass_kg": 0.42,
        "static_friction": 0.5,
        "rigidity": 0.95,
        "deformable_type": "rigid",
    }
    fused = fuse_physics(None, vlm, {"static_friction": 0.9, "dynamic_friction": 0.3, "restitution": 0.1})
    assert fused["mass_kg"] == 0.42
    assert fused["provenance"]["mass_kg"] == "vlm"
    assert fused["static_friction"] == 0.5
    assert fused["dynamic_friction"] == 0.3
    assert fused["provenance"]["dynamic_friction"] == "class_default"
    assert fused["learned_prior"] is False
    missing = fuse_physics(None, {"material": "aluminum"}, {"mass_kg": 0.015, "static_friction": 0.4})
    assert "mass_kg" not in missing
    assert missing["provenance"].get("mass_kg") == "class_default_refused"
    assert missing["static_friction"] == 0.4
    assert missing["learned_prior"] is False


def test_d3_usda_contains_physics_apis(tmp_path: Path):
    path = write_physics_material_usda(
        tmp_path / "physics_material.usda",
        {"static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.2, "mass_kg": 0.42},
        prim="Can",
    )
    text = path.read_text(encoding="utf-8")
    assert "PhysicsMaterialAPI" in text
    assert "PhysicsMassAPI" in text
    assert "physics:friction" not in text
    no_mass = write_physics_material_usda(
        tmp_path / "no_mass.usda",
        {"static_friction": 0.5, "dynamic_friction": 0.4, "restitution": 0.2},
        prim="Can",
    )
    assert "PhysicsMassAPI" not in no_mass.read_text(encoding="utf-8")


def test_d3_gaussian_physics_skips_without_vlm(tmp_path: Path):
    assert write_gaussian_physics(tmp_path / "gaussian_physics.npy", 4, None, None) is None
    assert not (tmp_path / "gaussian_physics.npy").is_file()
    path = write_gaussian_physics(tmp_path / "gaussian_physics.npy", 4, None, {"mass_kg": 0.42, "static_friction": 0.5})
    field = np.load(path)
    assert field.shape == (4, 3)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["copied_from"] == "vlm"
    assert meta["learned_prior"] is False


def test_d3_drop_without_telemetry_is_not_run():
    record = record_isaac_drop_slide(None)
    assert record["drop_test"] == "not_run"
    assert record["slide_test"] == "not_run"
    assert record["settle_within_2s"] is not True
    numbered = record_isaac_drop_slide({"settle_s": 1.2, "drop_ok": True, "slide_ok": True})
    assert numbered["settle_within_2s"] is True
    assert numbered["settle_s"] == pytest.approx(1.2)


def test_refresh_block_d_stays_not_started_without_physics(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    semantics = folder / "semantics"
    semantics.mkdir(parents=True)
    vector = [float(index) for index in range(32)]
    (folder / "codebook.json").write_text(
        json.dumps(
            {
                "vector": vector,
                "scene_mean": [item + 0.5 for item in vector],
                "scene_mean_measured": True,
                "learned_prior": False,
                "dinov2": True,
                "openclip": True,
                "n_copied": 1,
                "n_gaussians": 2,
                "copied_onto_gaussians": True,
                "encoder": "dinov2+openclip",
            }
        ),
        encoding="utf-8",
    )
    np.save(semantics / "gaussian_feature_idx.npy", np.array([0, 1], dtype=np.int8))
    record = refresh_block_d("can", tmp_path)
    assert record["state"] == "not_started"
    assert record["achieved"] is False
    assert record["learned_prior"] is False
    assert record["physics_json"] is False
    by_id = {part["id"]: part for part in record["parts"]}
    assert by_id["D1.1"]["state"] == "measured"
    assert by_id["D1.1"]["achieved"] is False
    assert by_id["D1.1"]["empirical"]["dim"] == 32
    assert by_id["D1.1"]["learned_prior"] is False
    assert by_id["D1.2"]["state"] == "measured"
    assert by_id["D1.2"]["empirical"]["scene_mean_measured"] is True
    assert by_id["D1.3"]["state"] == "unstarted"
    assert by_id["D1.4"]["state"] == "unstarted"
    assert by_id["D1.5"]["state"] == "unstarted"
    assert by_id["D2"]["state"] == "unstarted"
    assert by_id["D3"]["state"] == "unstarted"
    (folder / "physics.json").write_text(json.dumps({"material": "aluminum", "learned_prior": False}), encoding="utf-8")
    assert refresh_block_d("can", tmp_path)["state"] == "not_started"
    (folder / "physics.json").write_text(
        json.dumps({"material": "aluminum", "mass_kg": 0.33, "learned_prior": False}),
        encoding="utf-8",
    )
    partial = refresh_block_d("can", tmp_path)
    assert partial["state"] == "partial"
    assert partial["achieved"] is False
    assert partial["learned_prior"] is False


def _fixture_d2() -> dict:
    return {
        "material": "aluminum",
        "mass_kg": 0.42,
        "density": 2700.0,
        "static_friction": 0.5,
        "dynamic_friction": 0.4,
        "restitution": 0.2,
        "rigidity": 0.95,
        "deformable_type": "rigid",
        "ranges": {
            "mass_kg": [0.3, 0.6],
            "static_friction": [0.3, 0.7],
        },
        "provenance": {"source": "fixture"},
    }


def _tiny_png(path: Path) -> Path:
    path.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
            "0000000a49444154789c6360000002000154a24f9d0000000049454e44ae426082"
        )
    )
    return path


def _write_splat(path: Path, n: int = 3) -> None:
    names = [
        "x",
        "y",
        "z",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
    ]
    rows = np.zeros((n, len(names)), dtype=np.float32)
    rows[:, 0] = np.arange(n, dtype=np.float32)
    rows[:, 3] = 2.0
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + rows.tobytes())
