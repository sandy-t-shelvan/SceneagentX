"""Orbit chain: D then E then F on Block B files. A skip is not a pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools import orbit_chain as oc


def test_runner_calls_d_then_e_then_f(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    calls: list[tuple] = []
    real_d = oc.run_block_d
    real_e = oc.run_block_e
    real_f = oc.run_block_f

    def wrap_d(folder, orbit, b_inputs, chain):
        calls.append(("d", Path(b_inputs["mask"]), Path(b_inputs["codebook"]), Path(b_inputs["splat"])))
        return real_d(folder, orbit, b_inputs, chain)

    def wrap_e(folder, orbit, d_output):
        calls.append(("e", Path(d_output)))
        return real_e(folder, orbit, d_output)

    def wrap_f(folder, orbit, e_output):
        calls.append(("f", Path(e_output)))
        return real_f(folder, orbit, e_output)

    monkeypatch.setattr(oc, "run_block_d", wrap_d)
    monkeypatch.setattr(oc, "run_block_e", wrap_e)
    monkeypatch.setattr(oc, "run_block_f", wrap_f)
    result = oc.run_orbit(can, "orbit1", tmp_path / "orbit_chain.md")
    assert [item[0] for item in calls] == ["d", "e", "f"]
    assert calls[0][1] == can / "masks" / "orbit1" / "gaussian_in_mask.npy"
    assert calls[0][2] == can / "codebook.json"
    assert calls[0][3] == can / "colmap" / "orbit1" / "splat.ply"
    assert calls[1][1] == result["d_output"]
    assert calls[2][1] == result["e_output"]
    assert result["e_input"] == result["d_output"]
    assert result["f_input"] == result["e_output"]
    assert result["d_output"] == (can / "chain" / "orbit1").resolve()
    text = (tmp_path / "orbit_chain.md").read_text(encoding="utf-8")
    assert text.index("### B") < text.index("### C") < text.index("### D") < text.index("### E") < text.index("### F")
    index = np.load(can / "chain" / "orbit1" / "semantics" / "gaussian_feature_idx.npy")
    assert index.tolist() == [0, -1, 0, -1]


def test_c_is_skipped_and_not_achieved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    md = tmp_path / "orbit_chain.md"
    result = oc.run_orbit(can, "orbit1", md)
    assert result["c"]["state"] == "skipped"
    assert result["c"]["achieved"] is False
    saved = json.loads((can / "chain" / "orbit1" / "ingest" / "block_c.json").read_text(encoding="utf-8"))
    assert saved["state"] == "skipped"
    assert saved["achieved"] is False
    text = md.read_text(encoding="utf-8")
    assert "| C | skipped | achieved: false |" in text
    assert "achieved: true" not in text


def test_no_api_key_does_not_write_physics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    seen: dict = {}
    real = oc.start_d2

    def wrap(*args, **kwargs):
        seen["api_key"] = kwargs.get("api_key")
        seen["api_key_passed"] = "api_key" in kwargs
        seen["root"] = kwargs.get("root")
        return real(*args, **kwargs)

    monkeypatch.setattr(oc, "start_d2", wrap)
    result = oc.run_orbit(can, "orbit1", tmp_path / "orbit_chain.md")
    assert seen["api_key_passed"] is False
    assert seen["api_key"] is None
    root = Path(seen["root"]).resolve()
    chain = (can / "chain").resolve()
    assert chain == root or chain in root.parents
    assert result["d"]["d2_status"] == "model not asked"
    assert list(can.rglob("physics.json")) == []
    assert list(can.rglob("physics_material.usda")) == []
    assert list(can.rglob("gaussian_physics.npy")) == []
    assert (can / "ingest" / "block_d.json").read_text(encoding="utf-8") == '{"achieved": false, "sentinel": true}\n'
    assert (can / "asset.urdf").read_text(encoding="utf-8") == "sentinel-urdf\n"


def test_present_key_is_not_discarded_by_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    token = "present-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", token)
    can = _can(tmp_path)
    for parts in oc._D2_FRAMES:
        frame = can.joinpath(*parts)
        frame.parent.mkdir(parents=True, exist_ok=True)
        frame.write_bytes(b"frame")
    seen: dict = {}
    real = oc.start_d2

    def wrap(*args, **kwargs):
        seen["api_key"] = kwargs.get("api_key")
        seen["api_key_passed"] = "api_key" in kwargs
        return real(*args, **kwargs)

    def fake_ask(frames, name, bbox, api_key):
        seen["asked_key"] = api_key
        seen["name"] = name
        seen["frames"] = len(frames)
        raise OSError("stop before a network write")

    def block_network(*_args, **_kwargs):
        raise AssertionError("D2 test must not open a network request")

    monkeypatch.setattr(oc, "start_d2", wrap)
    monkeypatch.setattr("tools.block_d._ask_start_d2", fake_ask)
    monkeypatch.setattr("urllib.request.urlopen", block_network)
    result = oc.run_orbit(can, "orbit1", tmp_path / "present.md")
    assert seen["api_key_passed"] is False
    assert seen["api_key"] != ""
    assert seen["asked_key"] == token
    assert seen["name"] == "can"
    assert seen["frames"] == 4
    assert result["d"]["d2_status"] == "response not usable"
    assert list(can.rglob("physics.json")) == []


def test_no_splat_does_not_write_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    result = oc.run_orbit(can, "orbit1", tmp_path / "orbit_chain.md")
    assert not (can / "colmap" / "orbit1" / "splat.ply").exists()
    assert (can / "masks" / "orbit1" / "gaussian_in_mask.npy").is_file()
    assert list(can.rglob("mesh.ply")) == []
    assert not (can / "mesh").exists()
    assert result["e"]["state"] == "not_started"
    assert result["e"]["achieved"] is False
    parts = {part["id"]: part for part in result["e"]["parts"]}
    assert parts["E1"]["state"] == "unstarted"
    assert "splat.ply is missing" in parts["E1"]["reason"]
    assert "gaussian_in_mask.npy is on disk" in parts["E1"]["reason"]
    assert parts["E2"]["state"] == "unstarted"
    assert parts["E2"]["numbers"]["mesh"] is False
    assert parts["E4"]["state"] == "unstarted"
    assert parts["E4"]["numbers"]["route"] is None
    assert not (can / "chain" / "orbit1" / "deformable.json").exists()


def test_missing_world_t_leaves_f1_unachieved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    result = oc.run_orbit(can, "orbit1", tmp_path / "orbit_chain.md")
    f1 = next(part for part in result["f"]["parts"] if part["id"] == "F1")
    f3 = next(part for part in result["f"]["parts"] if part["id"] == "F3")
    f4 = next(part for part in result["f"]["parts"] if part["id"] == "F4")
    assert f1["state"] == "unstarted"
    assert f1["achieved"] is False
    assert "Block C" in f1["numbers"]["world_T_object"]
    assert "mesh" in f1["numbers"]["missing"]
    assert "intrinsics" in f1["numbers"]["missing"]
    assert "mesh.ply was not written by Block E" in f1["reason"]
    assert "world_T_object is missing because Block C was skipped" in f1["reason"]
    assert "sim.reset" not in f1["reason"]
    assert f3["numbers"]["mesh"] is False
    assert f3["numbers"]["physics"] is False
    assert "Missing mesh, asset, physics" in f3["reason"]
    assert not (can / "chain" / "orbit1" / "world_T_object.json").exists()
    assert not (can / "chain" / "orbit1" / "align.json").exists()
    report = json.loads((can / "chain" / "orbit1" / "inspection_report.json").read_text(encoding="utf-8"))
    assert report["state"] == "not_run"
    assert report["reason"] == "Isaac was not run. No telemetry."
    assert report.get("settle_within_2s") is not True
    assert f4["reason"] == "Isaac was not run. No telemetry."
    assert "sim.reset" not in f4["reason"]
    assert result["f"]["state"] == "not_started"
    assert result["f"]["achieved"] is False


def test_achieved_false_is_not_rendered_as_achieved(tmp_path: Path):
    path = tmp_path / "block_f.json"
    path.write_text(
        json.dumps(
            {
                "block": "F",
                "orbit": "orbit1",
                "state": "measured",
                "achieved": False,
                "parts": [
                    {
                        "id": "F1",
                        "state": "measured",
                        "achieved": False,
                        "numbers": {"iou": 0.4},
                        "reason": "did not converge",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    text = oc.render_block_file(path)
    assert "achieved: false" in text
    assert "achieved: true" not in text
    assert "iou=0.4" in text
    copied = tmp_path / "true.json"
    copied.write_text(
        json.dumps(
            {
                "block": "F",
                "state": "measured",
                "achieved": True,
                "parts": [
                    {"id": "F2", "state": "measured", "achieved": True, "numbers": {"table_height_error_mm": 1.0}, "reason": ""}
                ],
            }
        ),
        encoding="utf-8",
    )
    assert "achieved: true" in oc.render_block_file(copied)


def test_hulls_and_deformable_type_are_required_before_partial(tmp_path: Path):
    folder = tmp_path / "can" / "chain" / "orbit1"
    mesh = folder / "mesh" / "mesh.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_text("ply\n", encoding="utf-8")
    ingest = folder / "ingest"
    ingest.mkdir()
    (ingest / "hulls.json").write_text(
        json.dumps({"convex_hull": "not_built", "coacd": False, "n_hulls": 0}),
        encoding="utf-8",
    )
    e2 = oc._e2_part(folder, [{"id": "body", "joint": None}], {"convex_hull": "not_built"})
    assert e2["state"] == "unstarted"
    assert e2["numbers"]["mesh"] is True
    assert oc._e_block_state(folder) == "partial"
    bare = tmp_path / "empty"
    bare.mkdir()
    assert oc._e_block_state(bare) == "not_started"
    e4 = oc._e4_part(folder, "rigid")
    assert e4["state"] == "unstarted"
    assert e4["numbers"]["route"] is None
    assert "not set to rigid" in e4["reason"]
    (folder / "physics.json").write_text(
        json.dumps({"material": "aluminum", "mass_kg": 0.012, "deformable_type": "rigid", "learned_prior": False}),
        encoding="utf-8",
    )
    oc.write_deformable(folder)
    recorded = oc._e4_part(folder, "rigid")
    assert recorded["state"] == "partial"
    assert recorded["numbers"]["route"] == "rigid"
    assert oc._f_block_state(folder, [oc._step("F1", "partial", False, {}, "looks partial")]) == "not_started"
    (folder / "world_T_object.json").write_text("{}", encoding="utf-8")
    assert oc._f_block_state(folder, [oc._step("F1", "partial", False, {}, "looks partial")]) == "partial"


def test_measured_d1_does_not_start_block_d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    md = tmp_path / "orbit_chain.md"
    result = oc.run_orbit(can, "orbit1", md)
    assert result["d"]["state"] == "not_started"
    assert result["d"]["achieved"] is False
    assert result["d"]["learned_prior"] is False
    parts = {part["id"]: part for part in result["d"]["parts"]}
    assert parts["D1.1"]["state"] == "measured"
    assert parts["D1.1"]["achieved"] is False
    assert parts["D1.1"]["numbers"]["learned_prior"] is False
    assert parts["D1.2"]["state"] == "partial"
    assert parts["D1.2"]["numbers"]["scene_mean_measured"] is False
    assert parts["D1.2"]["numbers"]["n_scene"] == 0
    assert parts["D1.2"]["numbers"]["n_unassigned"] == 2
    assert parts["D1.3"]["state"] == "unstarted"
    assert parts["D1.3"]["numbers"]["prior_input"] is False
    assert parts["D1.4"]["state"] == "unstarted"
    assert parts["D1.4"]["numbers"]["checkpoint"] is False
    assert parts["D1.5"]["state"] == "unstarted"
    assert parts["D1.5"]["numbers"]["reduced"] is False
    assert parts["D2"]["state"] == "unstarted"
    assert parts["D2"]["numbers"]["d2_status"] == "model not asked"
    assert parts["D2"]["numbers"]["physics_json"] is False
    assert parts["D3"]["state"] == "unstarted"
    assert parts["D3"]["numbers"]["physics_material_usda"] is False
    assert parts["D3"]["numbers"]["gaussian_physics_npy"] is False
    chain = can / "chain" / "orbit1"
    assert not (chain / "physics.json").exists()
    assert not (chain / "physics_material.usda").exists()
    assert not (chain / "gaussian_physics.npy").exists()
    assert not (chain / "semantics" / "prior_input.npy").exists()
    assert not (chain / "semantics" / "prior_reduced.json").exists()
    index = np.load(chain / "semantics" / "gaussian_feature_idx.npy")
    assert index.tolist() == [0, -1, 0, -1]
    text = md.read_text(encoding="utf-8")
    assert "Block D state `not_started` achieved: false." in text
    assert "Block D state `partial`" not in text


def test_d12_labels_follow_a_separate_scene_mean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    vector = [float(i) for i in range(32)]
    _write_codebook(can, vector, scene=[item + 0.5 for item in vector], measured=True)
    distinct = oc.run_orbit(can, "orbit1", tmp_path / "distinct.md")
    parts = {part["id"]: part for part in distinct["d"]["parts"]}
    assert parts["D1.1"]["numbers"]["dim"] == 32
    assert parts["D1.1"]["numbers"]["learned_prior"] is False
    assert parts["D1.1"]["achieved"] is False
    assert parts["D1.2"]["state"] == "measured"
    assert parts["D1.2"]["numbers"]["scene_mean_measured"] is True
    assert parts["D1.2"]["numbers"]["n_scene"] == 2
    assert parts["D1.2"]["numbers"]["n_unassigned"] == 0
    assert distinct["d"]["state"] == "not_started"
    assert distinct["d"]["achieved"] is False
    labels = np.load(can / "chain" / "orbit1" / "semantics" / "gaussian_feature_idx.npy")
    assert labels.tolist() == [0, 1, 0, 1]
    _write_codebook(can, vector, scene=list(vector), measured=True)
    same = oc.run_orbit(can, "orbit1", tmp_path / "same.md")
    same_parts = {part["id"]: part for part in same["d"]["parts"]}
    assert same_parts["D1.2"]["state"] == "partial"
    assert same_parts["D1.2"]["numbers"]["scene_mean_measured"] is False
    assert same_parts["D1.2"]["numbers"]["n_scene"] == 0
    assert same["d"]["state"] == "not_started"
    stuck = np.load(can / "chain" / "orbit1" / "semantics" / "gaussian_feature_idx.npy")
    assert stuck.tolist() == [0, -1, 0, -1]


def test_block_d_partial_only_with_material_and_mass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    monkeypatch.setattr(oc, "start_d2", _fill_physics({"material": "aluminum", "learned_prior": False}))
    material_only = oc.run_orbit(can, "orbit1", tmp_path / "material.md")
    assert material_only["d"]["state"] == "not_started"
    assert material_only["d"]["achieved"] is False
    saved = json.loads((can / "chain" / "orbit1" / "physics.json").read_text(encoding="utf-8"))
    assert saved.get("mass_kg") is None
    monkeypatch.setattr(oc, "start_d2", _fill_physics({"mass_kg": 0.33, "learned_prior": False}))
    mass_only = oc.run_orbit(can, "orbit1", tmp_path / "mass.md")
    assert mass_only["d"]["state"] == "not_started"
    assert mass_only["d"]["achieved"] is False
    monkeypatch.setattr(
        oc,
        "start_d2",
        _fill_physics({"material": "aluminum", "mass_kg": 0.33, "learned_prior": False, "drop_test": "not_run"}),
    )
    filled = oc.run_orbit(can, "orbit1", tmp_path / "filled.md")
    assert filled["d"]["state"] == "partial"
    assert filled["d"]["achieved"] is False
    assert filled["d"]["learned_prior"] is False
    physics = json.loads((can / "chain" / "orbit1" / "physics.json").read_text(encoding="utf-8"))
    assert physics["material"] == "aluminum"
    assert physics["mass_kg"] == 0.33
    assert physics["learned_prior"] is False
    text = (tmp_path / "filled.md").read_text(encoding="utf-8")
    assert "Block D state `partial` achieved: false." in text
    assert "achieved: true" not in text


def test_refuses_bottle_and_vaseline(tmp_path: Path):
    bottle = tmp_path / "bottle"
    bottle.mkdir()
    vaseline = tmp_path / "vaseline"
    vaseline.mkdir()
    with pytest.raises(ValueError):
        oc.run_chain(bottle)
    with pytest.raises(ValueError):
        oc.run_chain(vaseline)


def _write_codebook(can: Path, vector: list[float], scene: list[float], measured: bool) -> None:
    path = can / "codebook.json"
    book = json.loads(path.read_text(encoding="utf-8"))
    book["vector"] = vector
    book["scene_mean"] = scene
    book["scene_mean_measured"] = measured
    book["learned_prior"] = False
    path.write_text(json.dumps(book), encoding="utf-8")


def _fill_physics(payload: dict):
    def fake(obj, frames, name, bbox, *, root=None, ask=None, api_key=None):
        from tools.common.paths import object_dir

        folder = object_dir(obj, root)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "physics.json").write_text(json.dumps(payload), encoding="utf-8")
        return "filled"

    return fake


def _can(tmp_path: Path) -> Path:
    can = tmp_path / "can"
    masks = can / "masks" / "orbit1"
    masks.mkdir(parents=True)
    (masks / "object.json").write_text(
        json.dumps(
            {
                "n_views": 4,
                "n_views_with_detection": 1,
                "largest_foreground_pixels": 10,
                "n_inside": 2,
                "n_outside": 2,
                "n_gaussians": 4,
                "projected": True,
                "span_inside": 0.2,
                "span_scene": 2.0,
                "covers_room": False,
            }
        ),
        encoding="utf-8",
    )
    np.save(masks / "gaussian_in_mask.npy", np.array([True, False, True, False]))
    (can / "codebook.json").write_text(
        json.dumps(
            {
                "vector": [1.0, 2.0],
                "learned_prior": False,
                "dinov2": True,
                "openclip": True,
                "n_copied": 2,
                "n_gaussians": 4,
            }
        ),
        encoding="utf-8",
    )
    instances = can / "semantics" / "orbit1"
    instances.mkdir(parents=True)
    (instances / "instances.json").write_text(
        json.dumps(
            {
                "instance_count": 1,
                "n_object": 2,
                "n_unassigned": 2,
                "miou": None,
                "hand_labels": 0,
                "learned_prior": False,
            }
        ),
        encoding="utf-8",
    )
    ingest = can / "ingest"
    ingest.mkdir()
    (ingest / "block_b_orbit1.json").write_text(
        json.dumps(
            {
                "block": "B",
                "state": "partial",
                "achieved": False,
                "m1_met": False,
                "m1_achieved": False,
                "m2_met": False,
                "m2_achieved": False,
                "parts": [
                    {"id": "B1", "state": "measured", "achieved": False, "reason": "detections measured"},
                    {"id": "B2", "state": "measured", "achieved": False, "reason": "vector copied"},
                    {"id": "B3", "state": "partial", "achieved": False, "reason": "mask lift measured"},
                    {"id": "B4", "state": "partial", "achieved": False, "reason": "mIoU unmeasured"},
                ],
            }
        ),
        encoding="utf-8",
    )
    (ingest / "block_d.json").write_text('{"achieved": false, "sentinel": true}\n', encoding="utf-8")
    (can / "asset.urdf").write_text("sentinel-urdf\n", encoding="utf-8")
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
        "0000000a49444154789c6360000002000154a24f9d0000000049454e44ae426082"
    )
    for parts in (
        ("iou", "orbit1", "frame_00167.png"),
        ("iou", "orbit2", "frame_00001.png"),
        ("iou", "orbit3", "frame_00125.png"),
        ("reviews", "frame_00001.png"),
    ):
        dest = can.joinpath(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(png)
    return can
