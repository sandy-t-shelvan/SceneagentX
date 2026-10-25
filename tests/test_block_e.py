"""Block E: mesh score, one body, joints, and a rigid can. A skip is not a pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.block_e import (
    attach_chamfer_block,
    cloth_sheet,
    collision_hulls,
    contact_points,
    extract_object_mesh,
    filter_object_gaussians,
    publish_can,
    record_joint_sweep,
    refresh_block_e,
    require_press_picture,
    resolve_parts,
    rope_capsules,
    route_deformable,
    run_generative_parts,
    score_mesh,
    soft_volume,
    write_planar_uv,
    write_vertex_colors,
    split_mesh,
    write_can_asset,
    write_deformable,
    write_physics_joints_usda,
    write_poisson_mesh,
    write_vertex_colors,
    write_xyz_ply,
)
from tools.block_gates import GateError, cube_corners
from tools.partfield_parts import choose_k
from tools.sceneagent_blocks import _block_e


def test_filter_drops_low_opacity_and_points_outside_two_sigma():
    cluster = np.zeros((40, 3), dtype=np.float64)
    cluster[:, 0] = np.linspace(-0.02, 0.02, 40)
    far = np.array([[30.0, 30.0, 30.0]])
    low = np.array([[0.0, 0.0, 0.0]])
    outsider = np.array([[0.0, 0.01, 0.0]])
    xyz = np.vstack((cluster, far, low, outsider))
    opacity = np.concatenate((np.full(40, 0.8), np.array([0.9, 0.04, 0.9])))
    opacity[0] = 0.05
    inside = np.ones(len(xyz), dtype=bool)
    inside[-1] = False
    result = filter_object_gaussians(xyz, opacity, inside)
    keep = result["keep"]
    assert keep[:40].all()
    assert not keep[40]
    assert not keep[41]
    assert not keep[42]
    assert result["opacity_floor"] == 0.05
    assert result["sigma"] == 2.0
    assert result["n_kept"] == 40
    with pytest.raises(GateError, match="lengths"):
        filter_object_gaussians(xyz, opacity[:-1], inside)


def test_missing_splat_does_not_write_mesh(tmp_path: Path):
    dest = tmp_path / "mesh" / "mesh.ply"
    status = extract_object_mesh(tmp_path / "splat.ply", tmp_path / "gaussian_in_mask.npy", dest)
    assert status["state"] == "unstarted"
    assert status["achieved"] is False
    assert status["mesh"] is None
    assert not dest.exists()
    assert write_poisson_mesh(np.zeros((3, 3)), None, tmp_path / "poisson.ply") is None
    assert not (tmp_path / "poisson.ply").exists()


def test_cube_chamfer_is_not_achieved_without_scale(tmp_path: Path):
    cube = cube_corners(0.02)
    mesh = write_xyz_ply(tmp_path / "cube.ply", cube)
    unmet = score_mesh(mesh, cube, None)
    assert unmet["chamfer_cm"] == pytest.approx(0.0, abs=1e-6)
    assert unmet["f1_at_1cm"] == pytest.approx(1.0)
    assert unmet["achieved"] is False
    assert unmet["state"] != "pass"
    assert unmet["metres_per_unit"] is None
    assert unmet["uv_atlas"] == "unstarted"
    measured = score_mesh(mesh, cube, 1.0)
    assert measured["achieved"] is True
    assert measured["chamfer_cm"] < 3.0
    assert measured["f1_at_1cm"] > 0.20
    missing = score_mesh(tmp_path / "no-mesh.ply", cube, 1.0)
    assert missing["state"] == "unstarted"
    assert missing["achieved"] is False
    assert missing["chamfer_cm"] is None


def test_vertex_colors_do_not_start_a_uv_atlas():
    mesh = {"vertices": np.zeros((2, 3))}
    coloured = write_vertex_colors(mesh, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    assert coloured["uv_atlas"] == "unstarted"
    assert coloured["texture"] is None
    assert coloured["n_colored"] == 2
    assert mesh["vertex_colors"].shape == (2, 3)
    short = write_vertex_colors({"vertices": np.zeros((2, 3))}, np.ones((1, 3)))
    assert short["colors"] is None
    assert short["uv_atlas"] == "unstarted"


def test_chamfer_block_keeps_silhouette_orbits(tmp_path: Path):
    cube = cube_corners(0.02)
    mesh = write_xyz_ply(tmp_path / "cube.ply", cube)
    path = tmp_path / "mesh_score.json"
    path.write_text(
        json.dumps(
            {
                "orbits": [{"orbit": "orbit1", "iou": 0.8356}],
                "achieved": False,
                "scale": "model_knowledge",
            }
        ),
        encoding="utf-8",
    )
    attach_chamfer_block(path, score_mesh(mesh, cube, None))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["orbits"] == [{"orbit": "orbit1", "iou": 0.8356}]
    assert saved["achieved"] is False
    assert saved["scale"] == "model_knowledge"
    assert saved["chamfer"]["achieved"] is False
    assert saved["chamfer"]["metres_per_unit"] is None


def test_no_part_masks_stay_one_body_and_choose_k_is_unchanged():
    assert choose_k({1: 0.2, 3: 0.9}) == 3
    assert choose_k({2: 0.9, 3: 0.8}) is None
    assert choose_k(None) is None
    body = resolve_parts(None, {1: 0.2, 3: 0.9})
    assert body == [{"id": "body", "joint": None, "generated": False, "partfield": False}]
    masks = {"door": [1], "knob": [1], "dial": [1]}
    named = resolve_parts(masks, {1: 0.1, 2: 0.9})
    assert [part["id"] for part in named] == ["door", "knob"]
    assert all(part["joint"] is None and part["generated"] is False for part in named)
    assert resolve_parts(masks, None)[0]["id"] == "body"


def test_split_mesh_writes_ply_only_when_labels_match(tmp_path: Path):
    vertices = np.eye(3)
    dest = tmp_path / "parts"
    assert split_mesh(vertices, None, dest)["written"] == []
    assert not dest.exists()
    mismatch = split_mesh(vertices, ["lid", "body"], dest)
    assert mismatch["written"] == []
    assert list(dest.glob("*.ply")) == []
    written = split_mesh(vertices, ["lid", "lid", "body"], dest)
    assert (dest / "lid.ply").is_file()
    assert (dest / "body.ply").is_file()
    assert len(written["written"]) == 2


def test_missing_coacd_is_not_labeled_coacd(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tools.block_e._try_coacd", lambda: None)
    tetra = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    small = collision_hulls(tetra)
    assert small["coacd"] is False
    assert small["convex_hull"] == "built"
    assert small["method"] == "trimesh"
    cloud = np.zeros((5001, 3), dtype=np.float64)
    large = collision_hulls(cloud)
    assert large["coacd"] is False
    assert large["convex_hull"] == "not_built"
    assert large["method"] != "coacd"


def test_missing_partcrafter_weights_raise(tmp_path: Path):
    with pytest.raises(GateError, match="PartCrafter"):
        run_generative_parts([], tmp_path / "missing.pt")
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"")
    with pytest.raises(GateError, match="PartCrafter"):
        run_generative_parts([], empty)
    present = tmp_path / "weights.pt"
    present.write_bytes(b"not-a-checkpoint")
    result = run_generative_parts([], present)
    assert result["generated"] is False
    assert result["parts"] == []


def test_focus_ring_joints_and_one_body_can(tmp_path: Path):
    named_parts = [
        {"id": "body", "joint": None},
        {"id": "focus_ring", "joint": "revolute", "travel": 90.0, "centroid": [0.1, 0.2, 0.3]},
        {"id": "shutter_trigger", "joint": "prismatic", "travel": 4.0, "centroid": [0.0, 0.0, 0.4]},
    ]
    named = write_can_asset(named_parts, "mesh/mesh.ply", tmp_path / "named.urdf")
    assert 'upper="1.570796"' in named["urdf"]
    assert 'upper="0.004000"' in named["urdf"]
    assert "focus_ring_joint" in named["urdf"]
    assert "shutter_trigger_joint" in named["urdf"]
    assert 'xyz="0.1 0.2 0.3"' in named["urdf"]
    usda = write_physics_joints_usda(named_parts, tmp_path / "named.usda", "mesh/mesh.ply")
    usda_text = usda.read_text(encoding="utf-8")
    assert "PhysicsRevoluteJoint" in usda_text
    assert "PhysicsPrismaticJoint" in usda_text
    assert "PhysicsDriveAPI:angular" in usda_text
    assert "physics:friction" not in usda_text
    single = write_can_asset([{"id": "body", "joint": None}], "mesh/mesh.ply", tmp_path / "asset.urdf")
    assert "<joint" not in single["urdf"]
    assert single["n_joints"] == 0
    can_usda = write_physics_joints_usda(
        [{"id": "body", "joint": None}],
        tmp_path / "asset.usda",
        "mesh/mesh.ply",
    )
    can_text = can_usda.read_text(encoding="utf-8")
    assert "PhysicsRevoluteJoint" not in can_text
    assert "PhysicsPrismaticJoint" not in can_text
    assert "PhysicsDriveAPI" not in can_text
    assert "sceneagent:coacd = false" in can_text
    assert "physics:friction" not in can_text
    assert "physics:mass" not in can_text


def test_sweep_without_telemetry_is_not_run():
    record = record_joint_sweep(None)
    assert record["joint_sweep"] == "not_run"
    assert record["interpenetration"] is None
    assert record["m5_met"] is None
    assert record["m5_state"] == "unmeasured"
    partial = record_joint_sweep({"interpenetration": False, "m5_met": True})
    assert partial["joint_sweep"] == "not_run"
    assert partial["interpenetration"] is None
    assert partial["m5_met"] is not True
    measured = record_joint_sweep(
        {
            "n_joints_swept": 1,
            "interpenetration": False,
            "max_penetration_m": 0.0,
            "m5_met": True,
        }
    )
    assert measured["joint_sweep"] == "ran"
    assert measured["interpenetration"] is False
    assert measured["m5_met"] is True


def test_block_e_scorer_stays_partial_for_one_body(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    write_can_asset([{"id": "body", "joint": None}], "mesh/mesh.ply", folder / "asset.urdf")
    scored = _block_e("can", folder, tmp_path)
    assert scored["state"] == "partial"
    assert scored["state"] != "pass"


def test_deform_routes_and_missing_press(tmp_path: Path):
    assert route_deformable("cloth") == "cloth"
    assert route_deformable("rope") == "rope"
    assert route_deformable("soft") == "soft"
    assert route_deformable(None) == "rigid"
    assert route_deformable("can") == "rigid"
    sheet = cloth_sheet()
    assert sheet["route"] == "cloth"
    assert sheet["uv_atlas"] == "unstarted"
    rope = rope_capsules(n=3)
    assert rope["route"] == "rope"
    assert rope["n_capsules"] == 3
    empty = write_deformable(tmp_path / "empty")
    assert empty["route"] == "rigid"
    assert empty["skipped"] is True
    assert empty["destruction"] == "not_run"
    assert empty["physgaussian"] is False
    assert empty["stiffness"] is None
    assert empty["cloth"] is False
    stiff = tmp_path / "stiff"
    stiff.mkdir()
    (stiff / "physics.json").write_text(
        json.dumps({"stiffness": 12.5, "deformable_type": "rigid"}),
        encoding="utf-8",
    )
    copied = write_deformable(stiff)
    assert copied["route"] == "rigid"
    assert copied["stiffness"] == 12.5
    assert not (stiff / "cloth.ply").exists()
    with pytest.raises(GateError, match="missing press picture"):
        require_press_picture(tmp_path / "press_after.png")


def test_soft_volume_raises_when_ftetwild_is_absent(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tools.block_e.ftetwild_binary", lambda: None)
    with pytest.raises(GateError, match="fTetWild"):
        soft_volume(np.zeros((4, 3)))


def test_publish_can_is_one_rigid_body_without_a_mesh(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    original = {
        "orbits": [{"orbit": "orbit1", "iou": 0.8356, "achieved": False}],
        "object": "can",
        "state": "measured",
        "achieved": False,
        "scale": "model_knowledge",
    }
    score_path = folder / "mesh_score.json"
    score_path.write_text(json.dumps(original), encoding="utf-8")
    before = score_path.read_text(encoding="utf-8")
    publish_can(tmp_path)
    assert score_path.read_text(encoding="utf-8") == before
    assert not (folder / "mesh" / "mesh.ply").exists()
    assert not (folder / "parts").exists()
    urdf = (folder / "asset.urdf").read_text(encoding="utf-8")
    assert "<joint" not in urdf
    usda = (folder / "asset.usda").read_text(encoding="utf-8")
    assert "PhysicsRevoluteJoint" not in usda
    assert "PhysicsPrismaticJoint" not in usda
    deformable = json.loads((folder / "deformable.json").read_text(encoding="utf-8"))
    assert deformable["route"] == "rigid"
    assert deformable["skipped"] is True
    assert deformable["destruction"] == "not_run"
    assert deformable["physgaussian"] is False
    sweep = json.loads((folder / "ingest" / "joint_sweep.json").read_text(encoding="utf-8"))
    assert sweep["joint_sweep"] == "not_run"
    assert sweep["interpenetration"] is None
    assert sweep["m5_met"] is None
    hulls = json.loads((folder / "ingest" / "hulls.json").read_text(encoding="utf-8"))
    assert hulls["coacd"] is False
    assert hulls["convex_hull"] == "not_built"
    report = json.loads((folder / "ingest" / "block_e.json").read_text(encoding="utf-8"))
    assert report["achieved"] is False
    assert report["state"] != "pass"
    assert report["scorer"]["state"] == "partial"
    assert report["m5"]["state"] == "unmeasured"
    assert report["m5"]["m5_met"] is None
    assert report["m5"]["interpenetration"] is None
    assert report["m5b"]["state"] == "unmeasured"
    assert report["m5b"]["m5b_met"] is None
    by_id = {step["id"]: step for step in report["parts"]}
    assert by_id["E1"]["state"] == "unstarted"
    assert by_id["E1"]["achieved"] is False
    assert by_id["E2"]["achieved"] is False
    assert by_id["E2"]["coacd"] is False
    assert by_id["E2"]["one_body"] is True
    assert by_id["E3"]["n_joints"] == 0
    assert by_id["E3"]["joint_sweep"] == "not_run"
    assert by_id["E3"]["achieved"] is False
    assert by_id["E3"]["m5_met"] is None
    assert by_id["E4"]["route"] == "rigid"
    assert by_id["E4"]["destruction"] == "not_run"
    assert by_id["E4"]["achieved"] is False
    script = Path(__file__).resolve().parents[1] / "vast-cluster" / "scripts" / "isaac_joint_sweep.py"
    text = script.read_text(encoding="utf-8")
    assert "record_joint_sweep" in text
    assert "m5_met" not in text
    assert "interpenetration" not in text


def test_sweep_script_refuses_bottle_and_vaseline():
    text = (
        Path(__file__).resolve().parents[1] / "vast-cluster" / "scripts" / "isaac_joint_sweep.py"
    ).read_text(encoding="utf-8")
    assert 'args.obj in {"bottle", "vaseline"}' in text
    assert "return 2" in text


def test_contact_plane_sits_on_the_lowest_z():
    cloud = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 3.0], [0.2, 0.4, 2.0]], dtype=np.float64)
    plane = contact_points(cloud, n=2)
    assert plane.shape == (4, 3)
    assert np.allclose(plane[:, 2], 1.0)


def test_planar_uv_is_separate_from_vertex_colour():
    mesh = {"vertices": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])}
    coloured = write_vertex_colors(mesh, np.ones((3, 3)))
    assert coloured["uv_atlas"] == "unstarted"
    atlas = write_planar_uv(mesh)
    assert atlas["uv_atlas"] == "planar"
    assert atlas["n_uv"] == 3
    assert mesh["uvs"].shape == (3, 2)


def test_block_e_score_is_not_can_only(tmp_path: Path):
    record = refresh_block_e("bottle", tmp_path)
    assert record["object"] == "bottle"
    assert record["achieved"] is False
    saved = json.loads((tmp_path / "data" / "bottle" / "ingest" / "block_e.json").read_text(encoding="utf-8"))
    assert saved["achieved"] is False

