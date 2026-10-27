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


def test_refused_crop_is_measured_and_not_achieved():
    record = oc._c_from_row(
        "orbit2",
        {
            "state": "measured",
            "crop": "refused",
            "infill": "not_run",
            "m3_met": False,
            "n_inside": 32436,
            "n_outside": 117427,
            "covers_room": True,
            "feature_field": "refused",
            "note": "Mask covers the room. The crop was not written. It is not a pass.",
        },
    )
    assert record["state"] == "measured"
    assert record["achieved"] is False
    part = record["parts"][0]
    assert part["state"] == "measured"
    assert part["achieved"] is False
    assert part["numbers"]["n_inside"] == 32436
    assert part["numbers"]["n_outside"] == 117427
    assert part["numbers"]["crop"] == "refused"
    assert "not a pass" in part["reason"]


def test_object_only_skipped_infill_is_not_a_pass():
    record = oc._c_from_row(
        "orbit1",
        {
            "state": "measured",
            "crop": "object_only",
            "infill": "skipped",
            "object_ply": "objects/bottle/orbit1/object.ply",
            "m3_met": False,
            "note": "Infill skipped. The object is the scene. It is not a pass.",
        },
    )
    assert record["state"] == "measured"
    assert record["achieved"] is False
    assert record["parts"][0]["achieved"] is False
    assert "not a pass" in record["parts"][0]["reason"]


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
    seen: dict = {"called": False}

    def wrap(*args, **kwargs):
        seen["called"] = True
        raise AssertionError("the chain must not call start_d2")

    monkeypatch.setattr(oc, "start_d2", wrap)
    result = oc.run_orbit(can, "orbit1", tmp_path / "orbit_chain.md")
    assert seen["called"] is False
    assert result["d"]["d2_status"] == "missing"
    assert "start_d2 returned" not in next(part["reason"] for part in result["d"]["parts"] if part["id"] == "D2")
    assert list(can.rglob("physics.json")) == []
    assert list(can.rglob("physics_material.usda")) == []
    assert list(can.rglob("gaussian_physics.npy")) == []
    assert (can / "ingest" / "block_d.json").read_text(encoding="utf-8") == '{"achieved": false, "sentinel": true}\n'
    assert (can / "asset.urdf").read_text(encoding="utf-8") == "sentinel-urdf\n"


def test_present_key_is_not_discarded_by_empty_string(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    token = "present-test-key"
    monkeypatch.setenv("OPENAI_API_KEY", token)
    can = _can(tmp_path)
    seen: dict = {"called": False}

    def wrap(*_args, **_kwargs):
        seen["called"] = True
        raise AssertionError("the chain must not call start_d2")

    def block_network(*_args, **_kwargs):
        raise AssertionError("D2 test must not open a network request")

    monkeypatch.setattr(oc, "start_d2", wrap)
    monkeypatch.setattr("urllib.request.urlopen", block_network)
    result = oc.run_orbit(can, "orbit1", tmp_path / "present.md")
    assert seen["called"] is False
    assert result["d"]["d2_status"] == "missing"
    d2 = next(part for part in result["d"]["parts"] if part["id"] == "D2")
    assert "not asked" in d2["reason"]
    assert "start_d2 returned" not in d2["reason"]
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


def test_mesh_and_physics_write_one_body_scene(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    chain = can / "chain" / "orbit1"
    _chain_mesh_physics(chain)
    result = oc.run_block_f(can, "orbit1", chain)
    urdf = (chain / "asset.urdf").read_text(encoding="utf-8")
    usda = (chain / "asset.usda").read_text(encoding="utf-8")
    scene = (chain / "scene.usda").read_text(encoding="utf-8")
    assert "<joint" not in urdf
    assert "PhysicsRevoluteJoint" not in usda
    assert "PhysicsPrismaticJoint" not in usda
    assert "PhysicsDriveAPI" not in usda
    assert "@asset.usda@" in scene
    assert "Franka" not in scene
    assert (chain / "scene.usdz").is_file()
    f3 = next(part for part in result["record"]["parts"] if part["id"] == "F3")
    assert f3["numbers"]["scene_usdz"] is True
    assert f3["achieved"] is False
    assert result["record"]["achieved"] is False
    assert result["record"]["learned_prior"] is False
    m7 = next(part for part in result["record"]["parts"] if part["id"] == "M7")
    assert m7["state"] == "unmeasured"
    assert m7["achieved"] is False
    report = json.loads((chain / "inspection_report.json").read_text(encoding="utf-8"))
    assert report["state"] == "not_run"
    assert report.get("settle_within_2s") is not True
    assert (can / "asset.urdf").read_text(encoding="utf-8") == "sentinel-urdf\n"


def test_missing_parts_does_not_invent_asset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    chain = can / "chain" / "orbit1"
    _chain_mesh_physics(chain, parts=False)
    result = oc.run_block_f(can, "orbit1", chain)
    assert not (chain / "asset.usda").exists()
    assert not (chain / "asset.urdf").exists()
    assert not (chain / "scene.usdz").exists()
    f3 = next(part for part in result["record"]["parts"] if part["id"] == "F3")
    assert f3["achieved"] is False
    assert "ingest/parts.json is absent" in f3["reason"]
    assert result["record"]["achieved"] is False


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


def test_d13_missing_vectors_does_not_say_no_splat(tmp_path: Path):
    chain = tmp_path / "orbit2"
    chain.mkdir()
    part = oc._d13_part(chain, {"state": "unstarted", "reason": "assigned vectors are missing"})
    assert part["state"] == "unstarted"
    assert part["achieved"] is False
    assert part["numbers"]["prior_input"] is False
    assert "No splat" not in part["reason"]
    assert "scene-mean" in part["reason"]
    assert "assigned vectors are missing" in part["reason"]
    missing = oc._d13_part(
        chain,
        {"state": "unstarted", "reason": "splat.ply is missing; geometry was not invented"},
    )
    assert missing["reason"] == "splat.ply is missing; geometry was not invented"
    assert missing["reason"] != "No splat, so prior_input.npy was not written."


def test_f1_copies_c1_world_pose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = _can(tmp_path)
    source = can / "objects" / "can" / "orbit1" / "world_T_object.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "translation": [1.0, 2.0, 3.0],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": 1.0,
                "achieved": True,
                "iou": 0.99,
            }
        ),
        encoding="utf-8",
    )
    orbit_background = can / "background" / "orbit2" / "background.ply"
    orbit_background.parent.mkdir(parents=True)
    orbit_background.write_bytes(b"ply")
    chain = can / "chain" / "orbit1"
    chain.mkdir(parents=True)
    assert oc._orbit_background(can, "orbit2", chain) == orbit_background
    fresh = can / "chain" / "orbit3"
    fresh.mkdir(parents=True)
    other = can / "objects" / "can" / "orbit3" / "world_T_object.json"
    other.parent.mkdir(parents=True)
    other.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    found = oc.copy_c1_pose(can, "orbit3", fresh)
    assert found == fresh / "world_T_object.json"
    assert not (fresh / "align.json").exists()
    assert not (chain / "world_T_object.json").exists()
    result = oc.run_block_f(can, "orbit1", chain)
    f1 = next(part for part in result["record"]["parts"] if part["id"] == "F1")
    assert f1["numbers"]["world_T_object"] == "present"
    assert "world_T_object" not in f1["numbers"]["missing"].split(",")
    assert f1["achieved"] is False
    assert f1["numbers"]["iou"] is None
    assert "world_T_object is missing" not in f1["reason"]
    assert not (chain / "align.json").exists()
    saved = json.loads((chain / "world_T_object.json").read_text(encoding="utf-8"))
    assert saved["translation"] == [1.0, 2.0, 3.0]
    assert result["record"]["achieved"] is False
    assert oc.copy_c1_pose(can, "orbit1", chain) == chain / "world_T_object.json"


def _square_ply(path: Path, x: float, z: float) -> None:
    verts = (
        (x - 0.5, -0.5, z),
        (x + 0.5, -0.5, z),
        (x + 0.5, 0.5, z),
        (x - 0.5, 0.5, z),
    )
    lines = [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 2",
        "property list uchar int vertex_indices",
        "end_header",
        *(f"{vx:.8g} {vy:.8g} {vz:.8g} 204 204 204" for vx, vy, vz in verts),
        "3 0 1 2",
        "3 0 2 3",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_block_f_uses_the_review_camera_on_a_world_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cv2

    from tools.block_f import render_view

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = tmp_path / "can"
    chain = can / "chain" / "orbit1"
    mesh = chain / "mesh" / "mesh.ply"
    _square_ply(mesh, 1.2, 0.3)
    (chain / "world_T_object.json").write_text(
        json.dumps(
            {
                "translation": [1.2, 0.0, 0.3],
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": 1.0,
            }
        ),
        encoding="utf-8",
    )
    ingest = chain / "ingest"
    ingest.mkdir(parents=True)
    (ingest / "block_e.json").write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "id": "E1",
                        "reason": (
                            "mesh written from the existing splat; "
                            "scale is still unmet until metres_per_unit is measured"
                        ),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.2, 0.0, -1.0]
    intrinsics = {"fx": 40.0, "fy": 40.0, "cx": 24.0, "cy": 24.0, "width": 48, "height": 48}
    view = render_view(mesh, c2w, intrinsics)
    reviews = can / "reviews"
    reviews.mkdir(parents=True)
    rgb = np.clip(view["rgb"] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(reviews / "frame_00167.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(reviews / "frame_00167_mask.png"), view["mask"].astype(np.uint8) * 255)
    opengl = c2w @ np.diag([1.0, -1.0, -1.0, 1.0])
    transforms = can / "colmap" / "orbit1" / "transforms.json"
    transforms.parent.mkdir(parents=True)
    transforms.write_text(
        json.dumps(
            {
                "w": 48,
                "h": 48,
                "fl_x": 40.0,
                "fl_y": 40.0,
                "cx": 24.0,
                "cy": 24.0,
                "camera_model": "OPENCV",
                "frames": [
                    {"file_path": "images/frame_00167.png", "transform_matrix": opengl.tolist()},
                ],
            }
        ),
        encoding="utf-8",
    )
    pose_text = (chain / "world_T_object.json").read_text(encoding="utf-8")
    block_e_text = (ingest / "block_e.json").read_text(encoding="utf-8")
    result = oc.run_block_f(can, "orbit1", chain)
    f1 = next(part for part in result["record"]["parts"] if part["id"] == "F1")
    assert result["record"]["achieved"] is False
    assert result["record"]["learned_prior"] is False
    assert f1["achieved"] is False
    assert f1["numbers"]["metres_per_unit"] is None
    assert f1["numbers"]["iou"] > 0.95
    assert "metres_per_unit is missing" in f1["reason"]
    assert "Triangles were drawn" in f1["reason"]
    assert "The render is blank" not in f1["reason"]
    saved = json.loads((chain / "align.json").read_text(encoding="utf-8"))
    assert saved["crop_translation_applied"] is False
    assert saved["mesh_frame"] == "world"
    assert saved["render_blank"] is False
    assert saved["triangles_in_front"] > 0
    assert max(abs(float(value)) for value in saved["translation"]) < 0.06
    assert abs(float(saved["translation"][0]) - 1.2) > 0.5
    assert (chain / "world_T_object.json").read_text(encoding="utf-8") == pose_text
    assert (ingest / "block_e.json").read_text(encoding="utf-8") == block_e_text


def test_missing_dataparser_still_uses_the_training_camera(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import cv2

    from tools.block_f import opencv_c2w, render_view

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    can = tmp_path / "can"
    chain = can / "chain" / "orbit1"
    opengl = np.array(
        [
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            np.array(
                [
                    [1.0, 0.0, 0.0, 1.5],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        ]
    )
    # OpenGL identity looks down -Z. Flip the stored rotation into that convention.
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    stored = np.stack([pose @ flip for pose in opengl])
    training = oc._training_opengl_c2w(stored[0], stored)
    camera = opencv_c2w(training)
    assert not np.allclose(camera, opencv_c2w(stored[0]))
    center = camera[:3, 3]
    right = camera[:3, 0]
    down = camera[:3, 1]
    forward = camera[:3, 2]
    origin = center + forward * 1.6
    corners = [
        origin - 0.35 * right - 0.35 * down,
        origin + 0.35 * right - 0.35 * down,
        origin + 0.35 * right + 0.35 * down,
        origin - 0.35 * right + 0.35 * down,
    ]
    mesh = chain / "mesh" / "mesh.ply"
    mesh.parent.mkdir(parents=True)
    body = [
        "ply",
        "format ascii 1.0",
        "element vertex 4",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "element face 2",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for corner in corners:
        body.append(f"{corner[0]:.8g} {corner[1]:.8g} {corner[2]:.8g} 204 204 204")
    body.extend(["3 0 1 2", "3 0 2 3"])
    mesh.write_text("\n".join(body) + "\n", encoding="utf-8")
    shift = [float(value) for value in origin]
    (chain / "world_T_object.json").write_text(
        json.dumps({"translation": shift, "rotation_wxyz": [1.0, 0.0, 0.0, 0.0], "scale": 1.0}),
        encoding="utf-8",
    )
    ingest = chain / "ingest"
    ingest.mkdir()
    (ingest / "block_e.json").write_text(
        json.dumps(
            {
                "parts": [
                    {
                        "id": "E1",
                        "reason": "mesh written from the existing splat; scale is still unmet until metres_per_unit is measured",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    intrinsics = {"fx": 40.0, "fy": 40.0, "cx": 24.0, "cy": 24.0, "width": 48, "height": 48}
    view = render_view(mesh, camera, intrinsics)
    reviews = can / "reviews"
    reviews.mkdir()
    rgb = np.clip(view["rgb"] * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(reviews / "frame_00167.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(reviews / "frame_00167_mask.png"), view["mask"].astype(np.uint8) * 255)
    transforms = can / "colmap" / "orbit1" / "transforms.json"
    transforms.parent.mkdir(parents=True)
    transforms.write_text(
        json.dumps(
            {
                "w": 48,
                "h": 48,
                "fl_x": 40.0,
                "fl_y": 40.0,
                "cx": 24.0,
                "cy": 24.0,
                "camera_model": "OPENCV",
                "applied_transform": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, -1.0, 0.0, 0.0]],
                "frames": [
                    {"file_path": "images/frame_00167.png", "transform_matrix": stored[0].tolist()},
                    {"file_path": "images/frame_00002.png", "transform_matrix": stored[1].tolist()},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = oc.run_block_f(can, "orbit1", chain)
    f1 = next(part for part in result["record"]["parts"] if part["id"] == "F1")
    saved = json.loads((chain / "align.json").read_text(encoding="utf-8"))
    assert f1["achieved"] is False
    assert f1["numbers"]["metres_per_unit"] is None
    assert f1["numbers"]["iou"] > 0.95
    assert saved["metres_per_unit"] is None
    assert saved["render_blank"] is False
    assert saved["crop_translation_applied"] is False
    assert "metres_per_unit is missing" in f1["reason"]
    assert "Triangles were drawn" in f1["reason"]


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
    assert parts["D1.4"]["state"] == "not_scheduled"
    assert parts["D1.4"]["achieved"] is False
    assert parts["D1.4"]["numbers"]["checkpoint"] is False
    assert parts["D1.4"]["numbers"]["prior_input"] is False
    assert parts["D1.4"]["numbers"]["learned_prior"] is False
    assert parts["D1.4"]["reason"] == oc._CHECKPOINT_REASON
    assert parts["D1.5"]["state"] == "not_scheduled"
    assert parts["D1.5"]["achieved"] is False
    assert parts["D1.5"]["numbers"]["reduced"] is False
    assert parts["M4"]["state"] == "not_scheduled"
    assert parts["M4"]["achieved"] is False
    assert parts["M4b"]["state"] == "not_scheduled"
    assert parts["M4b"]["achieved"] is False
    assert parts["D2"]["state"] == "unstarted"
    assert parts["D2"]["numbers"]["d2_status"] == "missing"
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


def test_d14_keeps_a_present_prior_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    can = _can(tmp_path)
    chain = can / "chain" / "orbit1"
    chain.mkdir(parents=True)
    prior = chain / "semantics" / "prior_input.npy"
    prior.parent.mkdir(parents=True)
    np.save(prior, np.zeros((2, 40), dtype=np.float64))

    def fail(*_args, **_kwargs):
        raise AssertionError("run_learned_prior must not run without a checkpoint")

    monkeypatch.setattr(oc, "run_learned_prior", fail)
    part = oc._run_d14(can, chain, None)["part"]
    assert part["state"] == "not_scheduled"
    assert part["achieved"] is False
    assert part["numbers"]["prior_input"] is True
    assert part["numbers"]["checkpoint"] is False
    assert part["numbers"]["learned_prior"] is False
    assert "no prior_input" not in part["reason"]
    assert part["reason"] == oc._CHECKPOINT_REASON
    assert not (can / "semantics" / "physics_prior.npz").exists()
    assert not (chain / "semantics" / "physics_prior.npz").exists()


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
    chain = can / "chain" / "orbit1"
    chain.mkdir(parents=True)
    inputs = oc._b_inputs(can, "orbit1")

    def refuse(*_args, **_kwargs):
        raise AssertionError("the chain must not call start_d2")

    monkeypatch.setattr(oc, "start_d2", refuse)
    (chain / "physics.json").write_text(
        json.dumps({"material": "aluminum", "learned_prior": False}),
        encoding="utf-8",
    )
    material_only = oc.run_block_d(can, "orbit1", inputs, chain)
    assert material_only["record"]["state"] == "not_started"
    assert material_only["record"]["achieved"] is False
    saved = json.loads((chain / "physics.json").read_text(encoding="utf-8"))
    assert saved.get("mass_kg") is None
    (chain / "physics.json").write_text(json.dumps({"mass_kg": 0.33, "learned_prior": False}), encoding="utf-8")
    mass_only = oc.run_block_d(can, "orbit1", inputs, chain)
    assert mass_only["record"]["state"] == "not_started"
    assert mass_only["record"]["achieved"] is False
    (chain / "physics.json").write_text(
        json.dumps({"material": "aluminum", "mass_kg": 0.33, "learned_prior": False, "drop_test": "not_run"}),
        encoding="utf-8",
    )
    filled = oc.run_block_d(can, "orbit1", inputs, chain)
    assert filled["record"]["state"] == "partial"
    assert filled["record"]["achieved"] is False
    assert filled["record"]["learned_prior"] is False
    physics = json.loads((chain / "physics.json").read_text(encoding="utf-8"))
    assert physics["material"] == "aluminum"
    assert physics["mass_kg"] == 0.33
    assert physics["learned_prior"] is False
    assert physics["drop_test"] == "not_run"
    assert physics["drop_test_reason"] == "drop not scheduled for this video"
    assert physics.get("settle_within_2s") is not True
    text = oc.render_block(filled["record"])
    assert "Block D state `partial` achieved: false." in text
    assert "achieved: true" not in text


def test_relaxed_bake_passes_when_ply_rows_match(tmp_path: Path):
    can = _can(tmp_path)
    orbit = can / "objects" / "can" / "orbit1"
    orbit.mkdir(parents=True)
    (orbit / "object.ply").write_text("ply\nformat ascii 1.0\nelement vertex 4\nend_header\n", encoding="utf-8")
    chain = can / "chain" / "orbit1"
    (chain / "semantics").mkdir(parents=True)
    np.save(chain / "semantics" / "prior_input.npy", np.zeros((4, 40), dtype=np.float64))
    (chain / "physics.json").write_text(
        json.dumps(
            {
                "material": "aluminum",
                "mass_kg": 0.016,
                "static_friction": 0.3,
                "rigidity": 0.85,
                "learned_prior": False,
                "mass_check": {"checked": False, "reason": "not weighed"},
            }
        ),
        encoding="utf-8",
    )
    records = oc.run_block_d_from_block_c(can, ("orbit1",))
    record = records[0]
    assert record["achieved"] is True
    assert record["m4_met"] is False
    assert record["learned_prior"] is False
    assert record["bake"]["ply_vertices"] == 4
    assert record["bake"]["prior_rows"] == 4
    assert record["bake"]["mass_checked"] is False
    assert record["bake"]["settle_within_2s"] is None
    parts = {part["id"]: part for part in record["parts"]}
    assert parts["D1.3"]["achieved"] is True
    assert "Block C object ply" in parts["D1.3"]["reason"]
    assert "written from the splat" not in parts["D1.3"]["reason"]
    assert parts["D2"]["achieved"] is True
    assert "reused" in parts["D2"]["reason"]
    assert "not asked" in parts["D2"]["reason"]
    assert "start_d2 returned" not in parts["D2"]["reason"]
    assert parts["D3"]["achieved"] is True
    assert parts["D3"]["numbers"]["drop_test"] == "not_run"
    assert parts["D3"]["numbers"]["drop_test_reason"] == "drop not scheduled for this video"
    assert parts["D3"]["numbers"]["settle_within_2s"] is None
    assert parts["D1.4"]["state"] == "not_scheduled"
    assert parts["D1.4"]["achieved"] is False
    assert parts["D1.5"]["state"] == "not_scheduled"
    assert parts["D1.5"]["achieved"] is False
    assert parts["M4"]["state"] == "not_scheduled"
    assert parts["M4"]["achieved"] is False
    assert parts["M4b"]["state"] == "not_scheduled"
    assert parts["M4b"]["achieved"] is False
    short = can / "objects" / "can" / "orbit1" / "object.ply"
    short.write_text("ply\nformat ascii 1.0\nelement vertex 2\nend_header\n", encoding="utf-8")
    missed = oc.run_block_d_from_block_c(can, ("orbit1",))[0]
    assert missed["achieved"] is False
    assert missed["m4_met"] is False


def test_block_c_rerun_uses_this_runs_labels_and_does_not_invent_scene_mean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    can = _can(tmp_path)
    book = json.loads((can / "codebook.json").read_text(encoding="utf-8"))
    book["n_copied"] = 99
    (can / "codebook.json").write_text(json.dumps(book), encoding="utf-8")
    orbit = can / "objects" / "can" / "orbit1"
    orbit.mkdir(parents=True)
    (orbit / "object.ply").write_text("ply\nformat ascii 1.0\nelement vertex 4\nend_header\n", encoding="utf-8")
    chain = can / "chain" / "orbit1"
    (chain / "semantics").mkdir(parents=True)
    np.save(chain / "semantics" / "prior_input.npy", np.zeros((4, 40), dtype=np.float64))
    (chain / "physics.json").write_text(
        json.dumps(
            {
                "material": "aluminum",
                "mass_kg": 0.016,
                "static_friction": 0.3,
                "rigidity": 0.85,
                "learned_prior": False,
                "drop_test": "not_run",
                "drop_test_reason": "Isaac sim.reset is known to hang; the drop test was not run",
                "mass_check": {"checked": False, "reason": "not weighed"},
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        can / "codebook_gaussians_orbit1.npz",
        object_vector=np.array([1.0, 2.0], dtype=np.float32),
        scene_mean=np.array([], dtype=np.float32),
    )

    def refuse(*_args, **_kwargs):
        raise AssertionError("start_d2 and run_learned_prior must not run")

    monkeypatch.setattr(oc, "start_d2", refuse)
    monkeypatch.setattr(oc, "run_learned_prior", refuse)
    record = oc.run_block_d_from_block_c(can, ("orbit1",))[0]
    parts = {part["id"]: part for part in record["parts"]}
    assert record["achieved"] is True
    assert record["m4_met"] is False
    assert record["learned_prior"] is False
    assert parts["D1.1"]["numbers"]["n_copied"] == 2
    assert parts["D1.1"]["numbers"]["n_copied"] != 99
    assert parts["D1.2"]["numbers"]["n_object"] == 2
    assert parts["D1.2"]["numbers"]["n_scene"] == 0
    assert parts["D1.2"]["numbers"]["n_unassigned"] == 2
    assert parts["D1.2"]["numbers"]["scene_mean_measured"] is False
    assert parts["D1.4"]["state"] == "not_scheduled"
    assert parts["D1.5"]["state"] == "not_scheduled"
    assert parts["M4"]["state"] == "not_scheduled"
    assert parts["M4"]["reason"] == "a vision mass is an unchecked estimate."
    assert parts["M4b"]["state"] == "not_scheduled"
    assert parts["D3"]["numbers"]["drop_test_reason"] == "drop not scheduled for this video"
    saved = json.loads((chain / "physics.json").read_text(encoding="utf-8"))
    assert saved["mass_kg"] == 0.016
    assert saved["drop_test_reason"] == "drop not scheduled for this video"
    assert "sim.reset" not in (chain / "physics.json").read_text(encoding="utf-8")
    assert saved.get("settle_within_2s") is not True
    assert not (chain / "semantics" / "physics_prior.npz").exists()
    np.savez(
        can / "codebook_gaussians_orbit1.npz",
        object_vector=np.array([1.0, 2.0], dtype=np.float32),
        scene_mean=np.array([1.0, 2.0], dtype=np.float32),
    )
    copied = oc.run_block_d_from_block_c(can, ("orbit1",))[0]
    copied_d12 = next(part for part in copied["parts"] if part["id"] == "D1.2")
    assert copied_d12["numbers"]["scene_mean_measured"] is False
    assert copied_d12["numbers"]["n_scene"] == 0
    np.savez(
        can / "codebook_gaussians_orbit1.npz",
        object_vector=np.array([1.0, 2.0], dtype=np.float32),
        scene_mean=np.array([0.4, 0.5], dtype=np.float32),
    )
    separate = oc.run_block_d_from_block_c(can, ("orbit1",))[0]
    separate_d12 = next(part for part in separate["parts"] if part["id"] == "D1.2")
    assert separate_d12["numbers"]["scene_mean_measured"] is True
    assert separate_d12["numbers"]["n_scene"] == 2
    assert separate_d12["numbers"]["n_unassigned"] == 0


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


def _chain_mesh_physics(chain: Path, parts: bool = True) -> None:
    mesh = chain / "mesh" / "mesh.ply"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_text("ply\n", encoding="utf-8")
    chain.joinpath("physics.json").write_text(
        json.dumps(
            {
                "material": "aluminum",
                "mass_kg": 0.016,
                "static_friction": 0.3,
                "dynamic_friction": 0.22,
                "restitution": 0.25,
                "learned_prior": False,
            }
        ),
        encoding="utf-8",
    )
    ingest = chain / "ingest"
    ingest.mkdir(parents=True, exist_ok=True)
    (ingest / "hulls.json").write_text(
        json.dumps({"coacd": False, "convex_hull": "not_built", "n_hulls": 0, "method": None}),
        encoding="utf-8",
    )
    if parts:
        (ingest / "parts.json").write_text(
            json.dumps(
                {"parts": [{"id": "body", "joint": None, "generated": False, "partfield": False}], "learned_prior": False}
            ),
            encoding="utf-8",
        )


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
