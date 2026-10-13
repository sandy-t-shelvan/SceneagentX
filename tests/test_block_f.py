"""Block F: alignment, orientation, packaging, and Isaac inspect.

Fixtures only. A skipped check is never a pass, and a can silhouette
score is not F1.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from tools.block_f import (
    align_similarity,
    apply_inspect_retry,
    compose_scene_usda,
    inspect_scene,
    orient_background,
    publish_can,
    refresh_block_f,
    render_view,
    score_residuals,
    write_scene_usdz,
)
from tools.block_gates import mask_iou


def _intrinsics() -> dict[str, float]:
    return {"fx": 40.0, "fy": 40.0, "cx": 24.0, "cy": 24.0, "width": 48, "height": 48}


def _square_mesh() -> dict[str, np.ndarray]:
    vertices = np.array(
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.5, 0.5, 2.0],
            [-0.5, 0.5, 2.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    colors = np.full((4, 3), 0.8, dtype=np.float64)
    return {"vertices": vertices, "triangles": triangles, "colors": colors}


def _matching_view() -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    mesh = _square_mesh()
    view = render_view(mesh, np.eye(4), _intrinsics())
    return mesh, view["rgb"], view["mask"], np.eye(4), _intrinsics()


def _plane_ply(path: Path, z: float) -> Path:
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 5), np.linspace(-0.2, 0.2, 5))
    points = np.column_stack((xs.ravel(), ys.ravel(), np.full(xs.size, z)))
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(points)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for point in points:
        lines.append(f"{point[0]:.8g} {point[1]:.8g} {point[2]:.8g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_identical_masks_exceed_iou():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    image = np.zeros((32, 32, 3), dtype=np.float64)
    image[mask] = 0.7
    score = score_residuals(image, mask, image, mask)
    assert score["iou"] == mask_iou(mask, mask)
    assert score["iou"] > 0.95
    assert score["weights"] == {"photometric": 1.0, "iou": 2.0, "edge": 0.5}
    assert score["loss"] == score["photometric"] + 2.0 * (1.0 - score["iou"]) + 0.5 * score["edge"]


def test_matching_alignment_stays_unachieved_without_metres(tmp_path: Path):
    mesh, capture, mask, world_T, intrinsics = _matching_view()
    dest = tmp_path / "align.json"
    record = align_similarity(
        mesh, capture, mask, world_T, intrinsics, metres_per_unit=None, dest=dest, max_evals=1
    )
    assert record["iou"] > 0.95
    assert record["converged"] is True
    assert record["achieved"] is False
    assert record["renderer"] == "open3d"
    assert record["nvdiffrast"] is False
    assert record["metres_per_unit"] is None
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert saved["achieved"] is False
    assert saved["converged"] is True


def test_shifted_fixture_does_not_converge():
    mesh, capture, mask, world_T, intrinsics = _matching_view()
    shifted_capture = np.roll(capture, 20, axis=1)
    shifted_mask = np.roll(mask, 20, axis=1)
    record = align_similarity(
        mesh,
        shifted_capture,
        shifted_mask,
        world_T,
        intrinsics,
        metres_per_unit=1.0,
        dest=None,
        max_evals=1,
    )
    assert record["iou"] <= 0.95
    assert record["converged"] is False
    assert record["achieved"] is False


def test_missing_mesh_writes_no_align_json(tmp_path: Path):
    dest = tmp_path / "align.json"
    _, capture, mask, world_T, intrinsics = _matching_view()
    record = align_similarity(None, capture, mask, world_T, intrinsics, dest=dest)
    assert record["state"] == "unstarted"
    assert record["achieved"] is False
    assert not dest.exists()
    missing_pose = tmp_path / "no_pose.json"
    mesh, capture, mask, _, intrinsics = _matching_view()
    without_pose = align_similarity(mesh, capture, mask, None, intrinsics, dest=missing_pose)
    assert without_pose["state"] == "unstarted"
    assert not missing_pose.exists()


def test_no_background_is_skipped(tmp_path: Path):
    dest = tmp_path / "orientation.json"
    record = orient_background(tmp_path / "missing.ply", None, dest=dest)
    assert record["skipped"] is True
    assert record["state"] == "skipped"
    assert record["achieved"] is False
    assert record["table_height_error_mm"] is None
    saved = json.loads(dest.read_text(encoding="utf-8"))
    assert "plane" not in saved
    assert saved["achieved"] is False


def test_table_height_error_is_a_number_and_not_a_skip(tmp_path: Path):
    high = orient_background(_plane_ply(tmp_path / "ten.ply", 0.01), None, calibrated_table_height_m=0.0)
    assert high["skipped"] is False
    assert high["state"] == "measured"
    assert high["table_height_error_mm"] == 10.0
    assert high["achieved"] is False
    assert high["plane"]["fitted"] is True
    low = orient_background(_plane_ply(tmp_path / "two.ply", 0.002), None, calibrated_table_height_m=0.0)
    assert low["skipped"] is False
    assert isinstance(low["table_height_error_mm"], float)
    assert low["table_height_error_mm"] == 2.0
    assert low["achieved"] is True
    assert low["sweep"]["best_offset_m"] == 0.0


def test_usda_references_asset_and_omits_collider(tmp_path: Path):
    asset = tmp_path / "asset.usd"
    asset.write_text("#usda 1.0\n", encoding="utf-8")
    physics = {"static_friction": 0.4, "restitution": 0.05}
    dest = tmp_path / "scene.usda"
    record = compose_scene_usda(
        dest,
        asset_usd=asset,
        physics=physics,
        cameras=[{"name": "orbit1_000", "pose": "orbit1_000", "transform": np.eye(4).tolist()}],
        plane=None,
        franka_usd=tmp_path / "missing_franka.usd",
    )
    text = dest.read_text(encoding="utf-8")
    assert "@asset.usd@" in text
    assert "CollisionProxy" not in text
    assert "not_placed" in text
    assert "physics:staticFriction = 0.4" in text
    assert "physics:restitution = 0.05" in text
    assert record["collision_proxy"] is False
    assert record["robot"] == "not_placed"
    assert record["mjcf"] is False
    assert record["fbx"] is False
    assert not (tmp_path / "scene.mjcf").exists()
    assert not (tmp_path / "scene.fbx").exists()
    usdz = tmp_path / "scene.usdz"
    assert write_scene_usdz(usdz, usda_path=dest, asset_usd=None, physics=physics) is None
    assert not usdz.exists()
    written = write_scene_usdz(usdz, usda_path=dest, asset_usd=asset, physics=physics)
    assert written == usdz
    with zipfile.ZipFile(usdz) as archive:
        assert archive.namelist() == ["scene.usda"]
        assert "@asset.usd@" in archive.read("scene.usda").decode("utf-8")


def test_timeout_writes_no_png_and_empty_telemetry_is_not_run(tmp_path: Path):
    report = inspect_scene(folder=tmp_path, timeout=True)
    assert report["state"] == "not_run"
    assert report["reason"] == "sim.reset did not return"
    assert report["png"] is None
    assert report["settle_ok"] is None
    assert report["physx_warnings"] is None
    assert report["moved_cm"] is None
    assert report["rendered"] is False
    assert not (tmp_path / "inspect" / "frame_000.png").exists()
    saved = json.loads((tmp_path / "isaac_check.json").read_text(encoding="utf-8"))
    assert saved["error"] == "timeout"
    assert saved["png"] is None
    bare = inspect_scene(telemetry=None)
    assert bare["state"] == "not_run"
    assert bare["reason"] == "Isaac was not run. No telemetry."
    assert bare["settle_ok"] is None
    assert bare["physx_warnings"] is None
    assert bare["moved_cm"] is None
    assert bare["joint_limit_violations"] is None


def test_fail_changes_only_one_inspect_parameter(tmp_path: Path):
    current = {"scale": 1.0, "yaw_deg": 0.0, "translation": [0.0, 0.0, 0.0]}
    isaac = tmp_path / "isaac.png"
    capture = tmp_path / "capture.png"
    isaac.write_bytes(b"\x89PNG\r\nisaac")
    capture.write_bytes(b"\x89PNG\r\ncapture")
    report = inspect_scene(
        telemetry={"physx_warnings": 2, "moved_cm": 5.0, "joint_limit_violations": 0},
        params=current,
        correction={"parameter": "yaw_deg", "value": 9.0, "problem": "rotation"},
        isaac_frame=isaac,
        capture_frame=capture,
        pose="orbit1-000",
    )
    changed = [name for name in ("scale", "yaw_deg", "translation") if report["params"][name] != current[name]]
    assert report["verdict"] == "fail"
    assert changed == ["yaw_deg"]
    assert report["route"] == "SCENEAGENT-ORIENT-BACKGROUND"
    assert report["vlm_called"] is False
    assert report["review_pose"] == "orbit1-000"
    stopped = apply_inspect_retry(report["params"], "scale", 1.2, "scale", 3)
    assert stopped["stopped"] is True
    assert stopped["params"]["scale"] == report["params"]["scale"]
    assert stopped["params"]["yaw_deg"] == 9.0


def test_silhouette_score_is_not_m0_or_m6(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    (folder / "mesh_score.json").write_text(
        json.dumps({"iou": 0.99, "achieved": True, "f1_at_1cm": 0.8, "chamfer_cm": 0.2}),
        encoding="utf-8",
    )
    record = refresh_block_f("can", tmp_path)
    assert record["achieved"] is False
    assert record["m0"]["state"] == "unmeasured"
    assert record["m0"]["achieved"] is False
    assert record["m0"]["f1_at_1cm"] is None
    assert record["m0"]["chamfer_cm"] is None
    assert record["m6"]["state"] == "unmeasured"
    assert record["m6"]["achieved"] is False
    assert record["m6"]["iou"] is None
    assert record["m7"]["physx_warnings"] is None
    assert record["m7"]["moved_cm"] is None
    assert record["m7"]["settle_ok"] is None
    assert not (folder / "align.json").exists()


def test_publish_empty_can_leaves_steps_unstarted_or_skipped(tmp_path: Path):
    record = publish_can(tmp_path)
    folder = tmp_path / "data" / "can"
    assert not (folder / "align.json").exists()
    assert not (folder / "scene.usdz").exists()
    assert not (folder / "scene.usda").exists()
    assert not (folder / "inspection_report.json").exists()
    assert not (folder / "inspect" / "frame_000.png").exists()
    orient = json.loads((folder / "orientation.json").read_text(encoding="utf-8"))
    assert orient["skipped"] is True
    assert orient["achieved"] is False
    assert "plane" not in orient
    assert record["achieved"] is False
    assert record["metres_per_unit"] is None
    assert record["m6"]["state"] == "unmeasured"
    assert record["m7"]["state"] == "unmeasured"
    assert record["m0"]["state"] == "unmeasured"
    parts = {part["id"]: part for part in record["parts"]}
    assert parts["F1"]["state"] == "unstarted"
    assert parts["F1"]["achieved"] is False
    assert parts["F2"]["state"] == "skipped"
    assert parts["F2"]["achieved"] is False
    assert parts["F3"]["state"] == "unstarted"
    assert parts["F3"]["robot"] == "not_placed"
    assert parts["F4"]["state"] == "unstarted"
    assert parts["F4"]["settle_ok"] is None
    assert parts["F4"]["physx_warnings"] is None
    assert parts["F4"]["moved_cm"] is None
