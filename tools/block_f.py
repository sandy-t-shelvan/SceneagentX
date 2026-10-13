"""Block F writers: align, orient, package, inspect.

A skipped check is never a pass. This module does not extract a mesh, does
not invent metres, a table height, or a Franka placement, and does not call
nvdiffrast. The 7-DoF loop uses an Open3D raycast when that package imports,
and the same OpenCV residual definitions otherwise. The record says
renderer open3d. M6, M7, and M0 stay unmeasured until their numbers exist.
"""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tools.block_gates import (
    GateError,
    align_record,
    apply_one_inspect_parameter,
    inspect_pair,
    mask_iou,
    orientation_record,
    record_isaac_timeout,
    route_problem,
)
from tools.common.paths import object_dir, resolve_root

PHOTOMETRIC_W = 1.0
IOU_W = 2.0
EDGE_W = 0.5
LOSS_WEIGHTS = {"photometric": PHOTOMETRIC_W, "iou": IOU_W, "edge": EDGE_W}
IOU_CONVERGE = 0.95
PLATEAU_TOL = 1e-4
TABLE_HEIGHT_MM_MAX = 5.0
SETTLE_S = 2.0
MOVE_LIMIT_CM = 2.0
MAX_INSPECT_TRIES = 3
N_CAMERAS = 4
RENDERER = "open3d"
ISAAC_RESET = "sim.reset did not return"
_YAWS_DEG = (-10.0, 0.0, 10.0)
_OFFSETS_M = (-0.01, 0.0, 0.01)
_SEARCH = (
    np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([-0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, -0.05, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, -0.05, 0.0, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, -0.02, 0.0, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.02, 0.0]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.02]),
)
_MESH_NAMES = ("mesh/mesh.ply", "mesh.ply", "2dgs/mesh.ply", "mesh/fused.ply")
_WORLD_NAMES = (
    "world_T_object.json",
    "objects/can/world_T_object.json",
    "ingest/world_T_object.json",
)
_BACKGROUND_NAMES = ("background.ply", "background/splat.ply")
_ASSET_NAMES = ("asset.usd", "asset.usda", "asset.usdc")


def score_residuals(
    rendered_rgb: np.ndarray,
    rendered_mask: np.ndarray,
    capture_rgb: np.ndarray,
    capture_mask: np.ndarray,
) -> dict[str, Any]:
    """Photometric, silhouette IoU, and edge residuals. Loss uses the gate weights."""
    rendered = _rgb(rendered_rgb)
    capture = _rgb(capture_rgb)
    if rendered.shape[:2] != capture.shape[:2]:
        capture = cv2.resize(capture, (rendered.shape[1], rendered.shape[0]), interpolation=cv2.INTER_AREA)
    left = _mask(rendered_mask, rendered.shape[:2])
    right = _mask(capture_mask, rendered.shape[:2])
    iou = float(mask_iou(left, right))
    union = np.logical_or(left, right)
    if int(union.sum()) == 0:
        photometric = 1.0
    else:
        photometric = float(np.abs(rendered[union] - capture[union]).mean())
    edge = float(np.abs(_sobel(rendered) - _sobel(capture)).mean())
    loss = similarity_loss(photometric, iou, edge)
    return {
        "photometric": photometric,
        "iou": iou,
        "edge": edge,
        "loss": loss,
        "weights": dict(LOSS_WEIGHTS),
    }


def similarity_loss(photometric: float, iou: float, edge: float) -> float:
    """1.0 * photometric + 2.0 * (1 - IoU) + 0.5 * edge."""
    return PHOTOMETRIC_W * float(photometric) + IOU_W * (1.0 - float(iou)) + EDGE_W * float(edge)


def alignment_converged(iou: float, photometric_history: list[float]) -> bool:
    """True only when IoU is above 0.95 and the photometric residual has plateaued."""
    if not math.isfinite(float(iou)) or float(iou) <= IOU_CONVERGE:
        return False
    return photometric_plateaued(photometric_history)


def photometric_plateaued(history: list[float]) -> bool:
    """A single sample has not plateaued."""
    if len(history) < 2:
        return False
    previous, latest = float(history[-2]), float(history[-1])
    if not math.isfinite(previous) or not math.isfinite(latest):
        return False
    return abs(latest - previous) <= PLATEAU_TOL


def render_view(mesh: Any, c2w: np.ndarray, intrinsics: dict[str, Any]) -> dict[str, Any]:
    """Rasterize a mesh. Open3D when it imports; OpenCV otherwise. Never nvdiffrast."""
    geometry = _as_mesh(mesh)
    if geometry is None or not _intrinsics_ok(intrinsics):
        raise GateError("mesh or intrinsics are not renderable")
    pose = np.asarray(c2w, dtype=np.float64)
    if pose.shape != (4, 4):
        raise GateError("camera pose must be a 4 by 4 matrix")
    backend = "opencv"
    try:
        import open3d as o3d
    except ImportError:
        o3d = None
    if o3d is not None:
        image, hit = _render_open3d(o3d, geometry, pose, intrinsics)
        backend = "open3d"
    else:
        image, hit = _render_opencv(geometry, pose, intrinsics)
    return {
        "rgb": image,
        "mask": hit,
        "renderer": RENDERER,
        "nvdiffrast": False,
        "rasterizer": backend,
    }


def align_similarity(
    mesh: Any,
    capture: Any,
    mask: Any,
    world_T: Any,
    intrinsics: dict[str, Any] | None,
    *,
    metres_per_unit: float | None = None,
    dest: Path | None = None,
    max_evals: int = 8,
    c2w: np.ndarray | None = None,
) -> dict[str, Any]:
    """Optimise scale, rotation, and translation. Write align.json only when the inputs exist."""
    metres = _positive(metres_per_unit)
    geometry = _as_mesh(mesh)
    image = _load_capture(capture)
    silhouette = _load_mask(mask)
    params = _pose_params(world_T)
    cameras = intrinsics if _intrinsics_ok(intrinsics) else None
    if geometry is None or image is None or silhouette is None or params is None or cameras is None:
        record = align_record(False)
        record.update(
            {
                "achieved": False,
                "converged": False,
                "renderer": RENDERER,
                "nvdiffrast": False,
                "rasterizer": None,
                "metres_per_unit": metres,
                "weights": dict(LOSS_WEIGHTS),
                "dof": 7,
                "reason": "residuals are recorded only when a mesh, a capture, a mask, and world_T exist",
            }
        )
        return record
    pose = np.eye(4, dtype=np.float64) if c2w is None else np.asarray(c2w, dtype=np.float64)
    if isinstance(intrinsics, dict) and intrinsics.get("c2w") is not None and c2w is None:
        pose = np.asarray(intrinsics["c2w"], dtype=np.float64)
    best, history, n_evals, backend = _search_similarity(
        geometry, image, silhouette, params, cameras, pose, max_evals=max_evals
    )
    converged = alignment_converged(best["iou"], history)
    achieved = bool(converged and metres is not None)
    record = align_record(True)
    record.update(
        {
            "achieved": achieved,
            "converged": converged,
            "photometric_plateaued": photometric_plateaued(history),
            "silhouette": "measured",
            "edge": "measured",
            "photometric_residual": best["photometric"],
            "iou": best["iou"],
            "edge_residual": best["edge"],
            "loss": best["loss"],
            "weights": dict(LOSS_WEIGHTS),
            "renderer": RENDERER,
            "nvdiffrast": False,
            "rasterizer": backend,
            "metres_per_unit": metres,
            "dof": 7,
            "scale": best["scale"],
            "rotation_rotvec": best["rotation_rotvec"],
            "translation": best["translation"],
            "init_source": "world_T",
            "n_evals": n_evals,
            "photometric_history": history,
        }
    )
    if metres is None:
        record["achieved"] = False
        record["achieved_reason"] = "metres_per_unit is missing"
    elif not converged:
        record["achieved"] = False
        record["achieved_reason"] = "IoU is not above 0.95 or the photometric residual has not plateaued"
    else:
        record["achieved_reason"] = ""
    if backend != "open3d":
        record["renderer_note"] = (
            "nvdiffrast is not used. Open3D is not installed in this interpreter, "
            "so the rasterizer is OpenCV."
        )
    if dest is not None:
        _dump(Path(dest), record)
    return record


def orient_background(
    background_ply: Any,
    captures: list[dict[str, Any]] | None,
    *,
    calibrated_table_height_m: float | None = None,
    dest: Path | None = None,
) -> dict[str, Any]:
    """Sweep yaw and offset, then fit a plane. No splat is a skip, and a skip is not achieved."""
    kind, points = _background_points(background_ply)
    if kind == "missing":
        record = orientation_record(False)
        record["reason"] = "no background splat"
        record["table_height_error_mm"] = None
        record["renderer"] = RENDERER
        record["nvdiffrast"] = False
        if dest is not None:
            _dump(Path(dest), record)
        return record
    record = orientation_record(True)
    record["renderer"] = RENDERER
    record["nvdiffrast"] = False
    record["table_height_error_mm"] = None
    if kind != "points" or points is None:
        record["achieved"] = False
        record["reason"] = "background ply could not be read"
        if dest is not None:
            _dump(Path(dest), record)
        return record
    sweep = _sweep_background(points, captures or [])
    posed = _transform_cloud(points, float(sweep["best_yaw_deg"]), float(sweep["best_offset_m"]))
    plane = fit_dominant_plane(posed)
    calibrated = _finite(calibrated_table_height_m)
    error_mm: float | None = None
    if plane is not None and calibrated is not None:
        error_mm = round(abs(float(plane["height_m"]) - calibrated) * 1000.0, 6)
    record["sweep"] = sweep
    record["table_height_error_mm"] = error_mm
    record["table_height_limit_mm"] = TABLE_HEIGHT_MM_MAX
    if plane is not None:
        record["plane"] = plane
    record["achieved"] = bool(error_mm is not None and error_mm < TABLE_HEIGHT_MM_MAX)
    if error_mm is None:
        record["reason"] = "table height error is unmeasured; no calibrated table height was stored"
    elif record["achieved"]:
        record["reason"] = ""
    else:
        record["reason"] = "table height error is not under 5 mm"
    if dest is not None:
        _dump(Path(dest), record)
    return record


def fit_dominant_plane(points: np.ndarray) -> dict[str, Any] | None:
    """Smallest-eigenvector plane. Returns None when the cloud cannot define one."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] < 3 or cloud.shape[1] < 3:
        return None
    xyz = cloud[:, :3]
    if not np.isfinite(xyz).all():
        return None
    center = xyz.mean(axis=0)
    _, _, vh = np.linalg.svd(xyz - center, full_matrices=False)
    normal = vh[-1].astype(np.float64)
    norm = float(np.linalg.norm(normal))
    if norm == 0.0:
        return None
    normal = normal / norm
    if normal[2] < 0.0:
        normal = -normal
    helper = np.array([1.0, 0.0, 0.0]) if abs(float(normal[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
    axis_u = np.cross(normal, helper)
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    relative = xyz - center
    span_u = relative @ axis_u
    span_v = relative @ axis_v
    if abs(float(normal[2])) < 1e-8:
        height = float(center[2])
    else:
        height = float(np.dot(normal, center) / normal[2])
    return {
        "fitted": True,
        "normal": [float(value) for value in normal],
        "point": [float(value) for value in center],
        "height_m": height,
        "extent_m": [float(span_u.max() - span_u.min()), float(span_v.max() - span_v.min())],
        "axis_u": [float(value) for value in axis_u],
        "axis_v": [float(value) for value in axis_v],
    }


def compose_scene_usda(
    dest: Path,
    *,
    asset_usd: Path | None,
    physics: dict[str, Any] | None = None,
    cameras: list[dict[str, Any]] | None = None,
    background: Path | None = None,
    plane: dict[str, Any] | None = None,
    franka_usd: Path | None = None,
) -> dict[str, Any]:
    """Write scene.usda. The table collider is omitted unless a plane was fitted."""
    asset = _existing_file(asset_usd)
    if asset is None:
        return {
            "wrote": False,
            "robot": "not_placed",
            "collision_proxy": False,
            "reason": "no asset",
            "mjcf": False,
            "fbx": False,
        }
    robot, franka = _robot_placement(franka_usd)
    fitted = plane if _plane_fitted(plane) else None
    corners = _table_corners(fitted) if fitted is not None else None
    collision_proxy = corners is not None
    background_file = _existing_file(background)
    text = _usda_text(
        asset=asset,
        physics=physics if isinstance(physics, dict) else None,
        cameras=cameras or [],
        background=background_file,
        corners=corners,
        collision_proxy=collision_proxy,
        robot=robot,
        franka=franka,
    )
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return {
        "wrote": True,
        "path": str(dest),
        "robot": robot,
        "collision_proxy": collision_proxy,
        "asset": asset.name,
        "background": None if background_file is None else background_file.name,
        "mjcf": False,
        "fbx": False,
        "nvdiffrast": False,
    }


def write_scene_usdz(
    dest: Path,
    *,
    usda_path: Path | None = None,
    usda: str | None = None,
    asset_usd: Path | None = None,
    physics: dict[str, Any] | None = None,
) -> Path | None:
    """Zip scene.usda. Missing asset or physics record writes nothing."""
    if _existing_file(asset_usd) is None or not _physics_record(physics):
        return None
    payload = usda
    if payload is None and usda_path is not None and Path(usda_path).is_file():
        payload = Path(usda_path).read_text(encoding="utf-8")
    if not payload:
        return None
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("scene.usda")
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(dest, "w") as archive:
        archive.writestr(info, payload)
    return dest


def inspect_scene(
    scene: Path | None = None,
    cameras: list[dict[str, Any]] | None = None,
    telemetry: dict[str, Any] | None = None,
    *,
    folder: Path | None = None,
    timeout: bool = False,
    params: dict[str, Any] | None = None,
    correction: dict[str, Any] | None = None,
    tries: int = 0,
    isaac_frame: Path | None = None,
    capture_frame: Path | None = None,
    pose: str | None = None,
    write_report: bool = False,
) -> dict[str, Any]:
    """Describe the four-camera, 2-second settle. No telemetry is not_run, and a timeout writes no PNG."""
    del scene  # The USDZ path is recorded by the caller. This function does not invent one.
    current = _inspect_params(params)
    report: dict[str, Any] = {
        "procedure": {
            "n_cameras_required": N_CAMERAS,
            "settle_s": SETTLE_S,
            "move_limit_cm": MOVE_LIMIT_CM,
            "counters": ["physx_warnings", "moved_cm", "joint_limit_violations"],
        },
        "cameras": _describe_cameras(cameras),
        "n_cameras_supplied": len(_describe_cameras(cameras)),
        "robot": "not_placed",
        "vlm_called": False,
        "png": None,
        "rendered": False,
        "fake_image": False,
        "tries": int(tries),
        "params": current,
    }
    if timeout:
        if folder is not None:
            gate = record_isaac_timeout(Path(folder))
            report["error"] = gate["error"]
            report["isaac"] = False
        report.update(_not_run_counters())
        report["state"] = "not_run"
        report["reason"] = ISAAC_RESET
        _maybe_write_report(folder, report, write_report or folder is not None)
        _assert_no_png(folder)
        return report
    numbers = _telemetry_numbers(telemetry)
    if numbers is None:
        report.update(_not_run_counters())
        report["state"] = "not_run"
        report["reason"] = "Isaac was not run. No telemetry."
        _maybe_write_report(folder, report, write_report)
        _assert_no_png(folder)
        return report
    warnings, moved, joints = numbers
    settle_ok = warnings == 0 and moved <= MOVE_LIMIT_CM and joints == 0
    report.update(
        {
            "state": "measured",
            "reason": None,
            "settle_ok": settle_ok,
            "physx_warnings": warnings,
            "moved_cm": moved,
            "joint_limit_violations": joints,
            "verdict": "pass" if settle_ok else "fail",
        }
    )
    if report["verdict"] == "fail":
        report.update(_apply_correction(current, correction, int(tries)))
    if report["verdict"] == "fail" and isaac_frame is not None and capture_frame is not None and pose is not None:
        pair = inspect_pair(Path(isaac_frame), Path(capture_frame), pose, pose)
        report["review_images"] = pair["images"]
        report["review_pose"] = pair["pose"]
        report["vlm_called"] = False
        report["review_reason"] = "inspect pair is ready; no VLM was called"
    _maybe_write_report(folder, report, write_report)
    return report


def apply_inspect_retry(
    current: dict[str, Any],
    parameter: str,
    value: Any,
    problem: str,
    tries_completed: int,
) -> dict[str, Any]:
    """One of scale, yaw_deg, or translation. Stop after three tries."""
    params = _inspect_params(current)
    if int(tries_completed) >= MAX_INSPECT_TRIES:
        return {"stopped": True, "tries": int(tries_completed), "params": params, "route": None, "parameter": None}
    updated = apply_one_inspect_parameter(params, parameter, value)
    return {
        "stopped": False,
        "tries": int(tries_completed) + 1,
        "params": _inspect_params(updated),
        "route": route_problem(problem),
        "parameter": parameter,
    }


def publish_can(root: Path | None = None) -> dict[str, Any]:
    """Run the can writers, then score whatever files exist. Bottle and vaseline are not touched."""
    folder = object_dir("can", root)
    folder.mkdir(parents=True, exist_ok=True)
    _publish_align(folder)
    _publish_orient(folder)
    _publish_package(folder)
    return refresh_block_f("can", root)


def refresh_block_f(obj: str = "can", root: Path | None = None) -> dict[str, Any]:
    """Rewrite ingest/block_f.json from files. A missing artifact stays unachieved."""
    if obj != "can":
        raise GateError("block F score refresh is can only")
    folder = object_dir(obj, root)
    from tools.sceneagent_blocks import _block_f

    scored = _block_f(obj, folder, resolve_root(root) if root is not None else None)
    align_path = folder / "align.json"
    orient_path = folder / "orientation.json"
    report_path = folder / "inspection_report.json"
    align = _read(align_path)
    orient = _read(orient_path)
    report = _read(report_path)
    metres = _metres_from_folder(folder)
    inputs = _can_inputs(folder)
    parts = [
        _f1_status(align_path, align, inputs, metres),
        _f2_status(orient_path, orient, inputs),
        _f3_status(folder, inputs),
        _f4_status(report_path, report),
    ]
    record = {
        "block": "F",
        "object": obj,
        "state": scored.get("state") or "not_started",
        "achieved": False,
        "scorer": {
            "name": "_block_f",
            "state": scored.get("state"),
            "result": scored.get("result"),
            "artifact": scored.get("artifact"),
        },
        "m6": _m6(align, orient, metres),
        "m7": _m7(report, folder),
        "m0": _m0(folder, metres),
        "metres_per_unit": metres,
        "renderer": RENDERER,
        "nvdiffrast": False,
        "note": (
            "A skipped check is never a pass. M6, M7, and M0 stay unmeasured until their numbers exist. "
            "The can silhouette IoU in mesh_score.json is not F1. achieved stays false while metres_per_unit is missing."
        ),
        "parts": parts,
    }
    record["achieved"] = False
    _dump(folder / "ingest" / "block_f.json", record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Block F for the can")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    record = publish_can(args.root)
    summary = {
        "state": record.get("state"),
        "achieved": record.get("achieved"),
        "m6": record["m6"]["state"],
        "m7": record["m7"]["state"],
        "m0": record["m0"]["state"],
        "parts": {part["id"]: part["state"] for part in record["parts"]},
    }
    print(json.dumps(summary), flush=True)
    return 0


def _publish_align(folder: Path) -> dict[str, Any]:
    inputs = _can_inputs(folder)
    mesh = inputs["mesh"]
    world = inputs["world_T"]
    capture = inputs["capture"]
    mask = inputs["mask"]
    intrinsics = inputs["intrinsics"]
    if mesh is None or world is None or capture is None or mask is None or intrinsics is None:
        return align_similarity(
            mesh,
            capture,
            mask,
            world,
            intrinsics,
            metres_per_unit=inputs["metres"],
            dest=None,
        )
    return align_similarity(
        mesh,
        capture,
        mask,
        world,
        intrinsics,
        metres_per_unit=inputs["metres"],
        dest=folder / "align.json",
    )


def _publish_orient(folder: Path) -> dict[str, Any]:
    background = _can_inputs(folder)["background"]
    calibrated = _calibrated_table_height(folder)
    return orient_background(
        background,
        None,
        calibrated_table_height_m=calibrated,
        dest=folder / "orientation.json",
    )


def _publish_package(folder: Path) -> Path | None:
    inputs = _can_inputs(folder)
    asset = inputs["asset"]
    physics = inputs["physics"]
    if asset is None or physics is None:
        return None
    orient = _read(folder / "orientation.json")
    plane = orient.get("plane") if orient.get("skipped") is not True else None
    usda = folder / "scene.usda"
    compose_scene_usda(
        usda,
        asset_usd=asset,
        physics=physics,
        cameras=_stored_cameras(folder),
        background=inputs["background"],
        plane=plane if isinstance(plane, dict) else None,
        franka_usd=None,
    )
    return write_scene_usdz(folder / "scene.usdz", usda_path=usda, asset_usd=asset, physics=physics)


def _search_similarity(
    geometry: dict[str, Any],
    image: np.ndarray,
    silhouette: np.ndarray,
    params: np.ndarray,
    intrinsics: dict[str, Any],
    c2w: np.ndarray,
    *,
    max_evals: int,
) -> tuple[dict[str, Any], list[float], int, str]:
    limit = max(int(max_evals), 1)

    def evaluate(candidate: np.ndarray) -> tuple[dict[str, Any], str]:
        moved = dict(geometry)
        moved["vertices"] = _transform_vertices(geometry["vertices"], candidate)
        view = render_view(moved, c2w, intrinsics)
        scored = score_residuals(view["rgb"], view["mask"], image, silhouette)
        scored["scale"] = float(candidate[0])
        scored["rotation_rotvec"] = [float(value) for value in candidate[1:4]]
        scored["translation"] = [float(value) for value in candidate[4:7]]
        return scored, str(view["rasterizer"])

    best, backend = evaluate(params)
    history = [float(best["photometric"])]
    n_evals = 1
    best_params = np.array(params, dtype=np.float64, copy=True)
    if limit > 1:
        for delta in _SEARCH:
            if n_evals >= limit:
                break
            trial = best_params + delta
            if float(trial[0]) <= 0.05:
                continue
            scored, backend = evaluate(trial)
            n_evals += 1
            history.append(float(scored["photometric"]))
            if float(scored["loss"]) + 1e-9 < float(best["loss"]):
                best = scored
                best_params = trial
    confirmed, backend = evaluate(best_params)
    history.append(float(confirmed["photometric"]))
    n_evals += 1
    return confirmed, history, n_evals, backend


def _render_opencv(
    geometry: dict[str, Any],
    c2w: np.ndarray,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(geometry["vertices"], dtype=np.float64)
    triangles = np.asarray(geometry["triangles"], dtype=np.int64)
    colors = geometry.get("colors")
    if colors is None:
        colours = np.full((len(vertices), 3), 0.7, dtype=np.float64)
    else:
        colours = np.asarray(colors, dtype=np.float64)
        if colours.max() > 1.0:
            colours = colours / 255.0
    world_to_camera = np.linalg.inv(c2w)
    camera = (vertices @ world_to_camera[:3, :3].T) + world_to_camera[:3, 3]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    image = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.full((height, width), np.inf, dtype=np.float64)
    hit = np.zeros((height, width), dtype=bool)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    for tri in triangles:
        index = np.asarray(tri, dtype=np.int64)
        if np.any(index < 0) or np.any(index >= len(camera)):
            continue
        pts = camera[index]
        if np.any(pts[:, 2] <= 1e-4):
            continue
        uv = np.column_stack((fx * pts[:, 0] / pts[:, 2] + cx, fy * pts[:, 1] / pts[:, 2] + cy))
        minx = max(int(np.floor(uv[:, 0].min())), 0)
        maxx = min(int(np.ceil(uv[:, 0].max())), width - 1)
        miny = max(int(np.floor(uv[:, 1].min())), 0)
        maxy = min(int(np.ceil(uv[:, 1].max())), height - 1)
        if minx > maxx or miny > maxy:
            continue
        v0, v1, v2 = uv
        denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
        if abs(float(denom)) < 1e-8:
            continue
        ys, xs = np.mgrid[miny : maxy + 1, minx : maxx + 1]
        px = xs.astype(np.float64) + 0.5
        py = ys.astype(np.float64) + 0.5
        w0 = ((v1[1] - v2[1]) * (px - v2[0]) + (v2[0] - v1[0]) * (py - v2[1])) / denom
        w1 = ((v2[1] - v0[1]) * (px - v2[0]) + (v0[0] - v2[0]) * (py - v2[1])) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not np.any(inside):
            continue
        zeta = w0 * pts[0, 2] + w1 * pts[1, 2] + w2 * pts[2, 2]
        closer = inside & (zeta < depth[miny : maxy + 1, minx : maxx + 1])
        if not np.any(closer):
            continue
        tri_colours = colours[index]
        rgb = (
            w0[..., None] * tri_colours[0]
            + w1[..., None] * tri_colours[1]
            + w2[..., None] * tri_colours[2]
        )
        view = image[miny : maxy + 1, minx : maxx + 1]
        view_depth = depth[miny : maxy + 1, minx : maxx + 1]
        view_hit = hit[miny : maxy + 1, minx : maxx + 1]
        view[closer] = np.clip(rgb[closer], 0.0, 1.0)
        view_depth[closer] = zeta[closer]
        view_hit[closer] = True
    return image, hit


def _render_open3d(o3d: Any, geometry: dict[str, Any], c2w: np.ndarray, intrinsics: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mesh = o3d.geometry.TriangleMesh()
    vertices = np.asarray(geometry["vertices"], dtype=np.float64)
    triangles = np.asarray(geometry["triangles"], dtype=np.int32)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    colors = geometry.get("colors")
    if colors is None:
        colours = np.full((len(vertices), 3), 0.7, dtype=np.float64)
    else:
        colours = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colours)
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(legacy)
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    dirs = np.stack(((xs - cx) / fx, (ys - cy) / fy, np.ones_like(xs)), axis=-1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    world_dirs = dirs.reshape(-1, 3) @ c2w[:3, :3].T
    origins = np.repeat(c2w[:3, 3][None, :], world_dirs.shape[0], axis=0)
    rays = np.concatenate((origins, world_dirs), axis=1).astype(np.float32)
    answer = scene.cast_rays(o3d.core.Tensor(rays))
    hit = np.isfinite(answer["t_hit"].numpy()).reshape(height, width)
    rendered = np.zeros((height, width, 3), dtype=np.float64)
    if np.any(hit):
        prim = answer["primitive_ids"].numpy().reshape(height, width)
        uv = answer["primitive_uvs"].numpy().reshape(height, width, 2)
        faces = triangles[prim[hit]]
        bary_u = uv[hit, 0]
        bary_v = uv[hit, 1]
        bary_w = 1.0 - bary_u - bary_v
        rendered[hit] = (
            colours[faces[:, 0]] * bary_w[:, None]
            + colours[faces[:, 1]] * bary_u[:, None]
            + colours[faces[:, 2]] * bary_v[:, None]
        )
    return rendered, hit


def _sweep_background(points: np.ndarray, captures: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [(yaw, offset) for yaw in _YAWS_DEG for offset in _OFFSETS_M]
    scoreable = [item for item in captures if isinstance(item, dict) and _intrinsics_ok(item.get("intrinsics"))]
    if not scoreable:
        return {
            "scored": False,
            "n_candidates": len(candidates),
            "best_yaw_deg": 0.0,
            "best_offset_m": 0.0,
            "best_score": None,
        }
    best_score = -1.0
    best_yaw = 0.0
    best_offset = 0.0
    for yaw, offset in candidates:
        moved = _transform_cloud(points, yaw, offset)
        scores = []
        for capture in scoreable:
            projected = _project_cloud(moved, capture)
            target = capture.get("mask")
            if projected is None or target is None:
                continue
            scores.append(float(mask_iou(projected, _mask(target, projected.shape))))
        if not scores:
            continue
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_yaw = yaw
            best_offset = offset
    if best_score < 0.0:
        return {
            "scored": False,
            "n_candidates": len(candidates),
            "best_yaw_deg": 0.0,
            "best_offset_m": 0.0,
            "best_score": None,
        }
    return {
        "scored": True,
        "n_candidates": len(candidates),
        "best_yaw_deg": best_yaw,
        "best_offset_m": best_offset,
        "best_score": best_score,
    }


def _project_cloud(points: np.ndarray, capture: dict[str, Any]) -> np.ndarray | None:
    intrinsics = capture.get("intrinsics")
    if not _intrinsics_ok(intrinsics):
        return None
    c2w = np.eye(4, dtype=np.float64)
    if capture.get("c2w") is not None:
        c2w = np.asarray(capture["c2w"], dtype=np.float64)
    world_to_camera = np.linalg.inv(c2w)
    camera = (np.asarray(points, dtype=np.float64)[:, :3] @ world_to_camera[:3, :3].T) + world_to_camera[:3, 3]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    zed = camera[:, 2]
    valid = zed > 1e-4
    u = np.zeros(len(camera), dtype=np.float64)
    v = np.zeros(len(camera), dtype=np.float64)
    u[valid] = float(intrinsics["fx"]) * camera[valid, 0] / zed[valid] + float(intrinsics["cx"])
    v[valid] = float(intrinsics["fy"]) * camera[valid, 1] / zed[valid] + float(intrinsics["cy"])
    valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    mask = np.zeros((height, width), dtype=bool)
    mask[v[valid].astype(np.int64), u[valid].astype(np.int64)] = True
    return mask


def _transform_cloud(points: np.ndarray, yaw_deg: float, offset_m: float) -> np.ndarray:
    angle = math.radians(float(yaw_deg))
    cosine, sine = math.cos(angle), math.sin(angle)
    rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    moved = np.asarray(points, dtype=np.float64)[:, :3] @ rotation.T
    moved[:, 2] += float(offset_m)
    return moved


def _transform_vertices(vertices: np.ndarray, params: np.ndarray) -> np.ndarray:
    scale = float(params[0])
    rotation = Rotation.from_rotvec(np.asarray(params[1:4], dtype=np.float64)).as_matrix()
    translation = np.asarray(params[4:7], dtype=np.float64)
    return (scale * (np.asarray(vertices, dtype=np.float64) @ rotation.T)) + translation


def _usda_text(
    *,
    asset: Path,
    physics: dict[str, Any] | None,
    cameras: list[dict[str, Any]],
    background: Path | None,
    corners: list[np.ndarray] | None,
    collision_proxy: bool,
    robot: str,
    franka: Path | None,
) -> str:
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "World"',
        '    upAxis = "Z"',
        ")",
        'def Xform "World"',
        "{",
        f'    custom string sceneagent:robot = "{robot}"',
        f"    custom bool sceneagent:collision_proxy = {'true' if collision_proxy else 'false'}",
    ]
    if background is not None:
        lines.append(f'    custom string sceneagent:background = "{_usd_string(background.name)}"')
    lines.extend(
        [
            '    def Xform "Asset" (',
            f"        prepend references = @{_usd_string(asset.name)}@",
            "    )",
            "    {",
            "    }",
        ]
    )
    material = _material_lines(physics)
    if material:
        lines.extend(material)
    if robot == "placed" and franka is not None:
        lines.extend(
            [
                '    def Xform "Franka" (',
                f"        prepend references = @{_usd_string(franka.name)}@",
                "    )",
                "    {",
                "    }",
            ]
        )
    for index, camera in enumerate(_describe_cameras(cameras)[:N_CAMERAS]):
        transform = camera.get("transform")
        lines.append(f'    def Camera "{_ident(str(camera["name"]))}"')
        lines.append("    {")
        if isinstance(transform, list) and len(transform) == 4:
            lines.append(f"        matrix4d xformOp:transform = {_matrix_text(transform)}")
            lines.append('        uniform token[] xformOpOrder = ["xformOp:transform"]')
        if camera.get("pose"):
            lines.append(f'        custom string sceneagent:pose = "{_usd_string(str(camera["pose"]))}"')
        lines.append("    }")
        del index
    if collision_proxy and corners is not None:
        points = ", ".join("(" + ", ".join(f"{float(value):.8g}" for value in corner) + ")" for corner in corners)
        lines.extend(
            [
                '    def Xform "CollisionProxy" (',
                '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
                "    )",
                "    {",
                '        def Mesh "Table"',
                "        {",
                f"            point3f[] points = [{points}]",
                "            int[] faceVertexCounts = [4]",
                "            int[] faceVertexIndices = [0, 1, 2, 3]",
                '            uniform token orientation = "rightHanded"',
                "        }",
                "    }",
            ]
        )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _material_lines(physics: dict[str, Any] | None) -> list[str]:
    if not isinstance(physics, dict):
        return []
    static = _finite(physics.get("static_friction"))
    if static is None:
        static = _finite(physics.get("friction"))
    dynamic = _finite(physics.get("dynamic_friction"))
    restitution = _finite(physics.get("restitution"))
    if static is None and dynamic is None and restitution is None:
        return []
    lines = [
        '    def Material "PhysicsMaterial" (',
        '        prepend apiSchemas = ["PhysicsMaterialAPI"]',
        "    )",
        "    {",
    ]
    if static is not None:
        lines.append(f"        float physics:staticFriction = {static}")
    if dynamic is not None:
        lines.append(f"        float physics:dynamicFriction = {dynamic}")
    if restitution is not None:
        lines.append(f"        float physics:restitution = {restitution}")
    lines.append("    }")
    return lines


def _table_corners(plane: dict[str, Any] | None) -> list[np.ndarray] | None:
    if plane is None:
        return None
    extent = plane.get("extent_m")
    if not isinstance(extent, list) or len(extent) != 2:
        return None
    width = _finite(extent[0])
    depth = _finite(extent[1])
    if width is None or depth is None or width <= 1e-8 or depth <= 1e-8:
        return None
    center = np.asarray(plane["point"], dtype=np.float64)
    axis_u = np.asarray(plane["axis_u"], dtype=np.float64)
    axis_v = np.asarray(plane["axis_v"], dtype=np.float64)
    half_u = width / 2.0
    half_v = depth / 2.0
    return [
        center - axis_u * half_u - axis_v * half_v,
        center + axis_u * half_u - axis_v * half_v,
        center + axis_u * half_u + axis_v * half_v,
        center - axis_u * half_u + axis_v * half_v,
    ]


def _robot_placement(franka_usd: Path | None) -> tuple[str, Path | None]:
    path = _existing_file(franka_usd)
    if path is None or path.suffix.lower() not in {".usd", ".usda", ".usdc", ".usdz"}:
        return "not_placed", None
    return "placed", path


def _plane_fitted(plane: dict[str, Any] | None) -> bool:
    if not isinstance(plane, dict) or plane.get("fitted") is not True:
        return False
    normal = plane.get("normal")
    point = plane.get("point")
    return isinstance(normal, list) and len(normal) == 3 and isinstance(point, list) and len(point) == 3


def _describe_cameras(cameras: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    described = []
    for index, camera in enumerate(cameras or []):
        if not isinstance(camera, dict):
            continue
        transform = camera.get("transform", camera.get("transform_matrix"))
        described.append(
            {
                "name": str(camera.get("name") or f"cam_{index:03d}"),
                "pose": camera.get("pose"),
                "transform": transform if isinstance(transform, list) else None,
            }
        )
    return described


def _apply_correction(current: dict[str, Any], correction: dict[str, Any] | None, tries: int) -> dict[str, Any]:
    if not isinstance(correction, dict):
        return {"params": current, "stopped": False, "route": None}
    step = apply_inspect_retry(
        current,
        str(correction.get("parameter") or ""),
        correction.get("value"),
        str(correction.get("problem") or ""),
        tries,
    )
    return {"params": step["params"], "stopped": step["stopped"], "route": step["route"], "tries": step["tries"], "parameter": step["parameter"]}


def _inspect_params(current: dict[str, Any] | None) -> dict[str, Any]:
    source = current if isinstance(current, dict) else {}
    translation = source.get("translation", [0.0, 0.0, 0.0])
    if not isinstance(translation, list) or len(translation) != 3:
        translation = [0.0, 0.0, 0.0]
    scale = _finite(source.get("scale"))
    yaw = _finite(source.get("yaw_deg"))
    return {
        "scale": 1.0 if scale is None else scale,
        "yaw_deg": 0.0 if yaw is None else yaw,
        "translation": [float(value) for value in translation],
    }


def _telemetry_numbers(telemetry: dict[str, Any] | None) -> tuple[int, float, int] | None:
    if not isinstance(telemetry, dict):
        return None
    warnings = telemetry.get("physx_warnings")
    moved = telemetry.get("moved_cm")
    joints = telemetry.get("joint_limit_violations")
    if isinstance(warnings, bool) or isinstance(joints, bool) or isinstance(moved, bool):
        return None
    if not isinstance(warnings, int) or not isinstance(joints, int):
        return None
    moved_value = _finite(moved)
    if moved_value is None or warnings < 0 or joints < 0:
        return None
    return warnings, moved_value, joints


def _not_run_counters() -> dict[str, Any]:
    return {
        "settle_ok": None,
        "physx_warnings": None,
        "moved_cm": None,
        "joint_limit_violations": None,
    }


def _maybe_write_report(folder: Path | None, report: dict[str, Any], write: bool) -> None:
    if not write or folder is None:
        return
    _dump(Path(folder) / "inspection_report.json", report)


def _assert_no_png(folder: Path | None) -> None:
    if folder is None:
        return
    png = Path(folder) / "inspect" / "frame_000.png"
    if png.is_file():
        raise GateError("timeout must not leave a frame file")


def _f1_status(path: Path, align: dict[str, Any], inputs: dict[str, Any], metres: float | None) -> dict[str, Any]:
    if not path.is_file():
        folder = path.parent
        missing = [name for name in ("mesh", "world_T") if inputs.get(name) is None]
        bits = ["align.json was not written."]
        if missing:
            bits.append(_missing_sentence(missing))
        if _iou_frames(folder):
            bits.append("Images under iou/ are splat-versus-mask scores, not an F1 capture or mask.")
        if inputs.get("intrinsics") is None and (folder / "colmap" / "orbit1" / "model_select.json").is_file():
            bits.append("model_select.json has camera intrinsics and no world_T_object pose.")
        elif inputs.get("capture") is None or inputs.get("mask") is None or inputs.get("intrinsics") is None:
            bits.append(_missing_sentence([name for name in ("capture", "mask", "intrinsics") if inputs.get(name) is None]))
        return {
            "id": "F1",
            "does": "Optimise a 7-DoF similarity against photometric, silhouette, and edge residuals.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "converged": False,
            "renderer": RENDERER,
            "nvdiffrast": False,
            "metres_per_unit": metres,
            "missing": missing,
            "reason": " ".join(bit for bit in bits if bit),
        }
    converged = align.get("converged") is True
    achieved = align.get("achieved") is True and metres is not None and converged
    return {
        "id": "F1",
        "does": "Optimise a 7-DoF similarity against photometric, silhouette, and edge residuals.",
        "artifact": "align.json",
        "state": align.get("state") or "measured",
        "achieved": achieved,
        "converged": converged,
        "iou": _finite(align.get("iou")),
        "renderer": align.get("renderer") or RENDERER,
        "nvdiffrast": False,
        "metres_per_unit": metres,
        "reason": align.get("achieved_reason") or "",
    }


def _f2_status(path: Path, orient: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("background") is None:
        return {
            "id": "F2",
            "does": "Sweep background yaw and offset, fit the dominant plane, and record table height error.",
            "artifact": "orientation.json" if path.is_file() else "",
            "state": "skipped",
            "achieved": False,
            "skipped": True,
            "table_height_error_mm": None,
            "reason": "no background splat",
        }
    if not path.is_file():
        return {
            "id": "F2",
            "does": "Sweep background yaw and offset, fit the dominant plane, and record table height error.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "skipped": False,
            "table_height_error_mm": None,
            "reason": "background splat exists and orientation.json was not written",
        }
    error = orient.get("table_height_error_mm")
    error_n = _finite(error) if not isinstance(error, bool) else None
    skipped = orient.get("skipped") is True or orient.get("state") == "skipped"
    achieved = (not skipped) and error_n is not None and error_n < TABLE_HEIGHT_MM_MAX and orient.get("achieved") is True
    return {
        "id": "F2",
        "does": "Sweep background yaw and offset, fit the dominant plane, and record table height error.",
        "artifact": "orientation.json",
        "state": "skipped" if skipped else (orient.get("state") or "measured"),
        "achieved": achieved,
        "skipped": skipped,
        "table_height_error_mm": None if skipped else error_n,
        "reason": orient.get("reason") or "",
    }


def _f3_status(folder: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    usdz = folder / "scene.usdz"
    if not usdz.is_file():
        missing = []
        if inputs.get("asset") is None:
            missing.append("asset")
        if inputs.get("physics") is None:
            missing.append("physics")
        asset = inputs.get("asset")
        if asset is not None and inputs.get("physics") is None:
            reason = f"scene.usdz was not written. {Path(asset).name} is on disk. Missing physics record."
        else:
            reason = "scene.usdz was not written. " + (_missing_sentence(missing) if missing else "The package was not written.")
        return {
            "id": "F3",
            "does": "Compose scene.usda from an asset, a physics material, cameras, and an optional table proxy, then zip USDZ.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "robot": "not_placed",
            "collision_proxy": False,
            "reason": reason,
        }
    text = ""
    usda = folder / "scene.usda"
    if usda.is_file():
        text = usda.read_text(encoding="utf-8")
    return {
        "id": "F3",
        "does": "Compose scene.usda from an asset, a physics material, cameras, and an optional table proxy, then zip USDZ.",
        "artifact": "scene.usdz",
        "state": "measured",
        "achieved": False,
        "robot": "placed" if 'sceneagent:robot = "placed"' in text else "not_placed",
        "collision_proxy": "sceneagent:collision_proxy = true" in text,
        "reason": "USDZ exists. Isaac has not reported PhysX warnings or settle motion, so the package is not achieved.",
    }


def _f4_status(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        return {
            "id": "F4",
            "does": "Inspect the USDZ in Isaac for PhysX warnings, settle motion, and joint limits.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "settle_ok": None,
            "physx_warnings": None,
            "moved_cm": None,
            "reason": "inspection_report.json was not written. Isaac was not run. No telemetry.",
        }
    state = report.get("state") or "not_run"
    numbers = _telemetry_numbers(report)
    if state == "not_run" or numbers is None:
        return {
            "id": "F4",
            "does": "Inspect the USDZ in Isaac for PhysX warnings, settle motion, and joint limits.",
            "artifact": "inspection_report.json",
            "state": "not_run",
            "achieved": False,
            "settle_ok": None,
            "physx_warnings": None,
            "moved_cm": None,
            "reason": report.get("reason") or ISAAC_RESET,
        }
    warnings, moved, joints = numbers
    achieved = warnings == 0 and moved <= MOVE_LIMIT_CM and joints == 0 and report.get("settle_ok") is True
    return {
        "id": "F4",
        "does": "Inspect the USDZ in Isaac for PhysX warnings, settle motion, and joint limits.",
        "artifact": "inspection_report.json",
        "state": "measured",
        "achieved": achieved,
        "settle_ok": report.get("settle_ok") is True,
        "physx_warnings": warnings,
        "moved_cm": moved,
        "reason": "",
    }


def _m6(align: dict[str, Any], orient: dict[str, Any], metres: float | None) -> dict[str, Any]:
    iou = _finite(align.get("iou")) if align.get("state") == "measured" else None
    error = orient.get("table_height_error_mm")
    table = None if orient.get("skipped") is True or isinstance(error, bool) else _finite(error)
    if iou is None or table is None or metres is None:
        return {
            "state": "unmeasured",
            "achieved": False,
            "iou": iou,
            "table_height_error_mm": table,
            "metres_per_unit": metres,
            "reason": (
                "M6 needs a Block F silhouette IoU above 0.95 and a table height error under 5 mm, "
                "in metres. mesh_score.json is a splat-versus-mask score and is not this IoU. "
                "A skipped orientation does not supply the table height."
            ),
        }
    achieved = iou > IOU_CONVERGE and table < TABLE_HEIGHT_MM_MAX
    return {
        "state": "measured",
        "achieved": achieved,
        "iou": iou,
        "table_height_error_mm": table,
        "metres_per_unit": metres,
        "reason": "" if achieved else "IoU or table height is outside the gate",
    }


def _m7(report: dict[str, Any], folder: Path) -> dict[str, Any]:
    numbers = _telemetry_numbers(report) if report.get("state") == "measured" else None
    frame = folder / "inspect" / "frame_000.png"
    frame_ok = frame.is_file() and frame.stat().st_size > 0
    if numbers is None or not frame_ok:
        return {
            "state": "unmeasured",
            "achieved": False,
            "settle_ok": None,
            "physx_warnings": None,
            "moved_cm": None,
            "joint_limit_violations": None,
            "reason": (
                "M7 needs an Isaac frame plus PhysX warnings, settle motion, and joint-limit counts. "
                "Those numbers are not set without telemetry."
            ),
        }
    warnings, moved, joints = numbers
    achieved = warnings == 0 and moved <= MOVE_LIMIT_CM and joints == 0 and report.get("settle_ok") is True
    return {
        "state": "measured",
        "achieved": achieved,
        "settle_ok": report.get("settle_ok") is True,
        "physx_warnings": warnings,
        "moved_cm": moved,
        "joint_limit_violations": joints,
        "reason": "" if achieved else "PhysX warnings, motion, or joint limits are outside the gate",
    }


def _m0(folder: Path, metres: float | None) -> dict[str, Any]:
    """Scene-level F1 and Chamfer. mesh_score.json is not this measurement."""
    path = folder / "ingest" / "scene_eval.json"
    record = _read(path)
    f1 = _finite(record.get("f1_at_1cm")) if path.is_file() else None
    chamfer = _finite(record.get("chamfer_cm")) if path.is_file() else None
    if f1 is None or chamfer is None or metres is None:
        return {
            "state": "unmeasured",
            "achieved": False,
            "f1_at_1cm": None,
            "chamfer_cm": None,
            "metres_per_unit": metres,
            "reason": (
                "M0 needs a scene-level F1 at 1 cm and Chamfer in metres. "
                "data/can/mesh_score.json is a splat-versus-mask silhouette and is not F1."
            ),
        }
    from tools.block_gates import SCENE_CHAMFER_CM_MAX, SCENE_F1_1CM_MIN

    achieved = f1 > SCENE_F1_1CM_MIN and chamfer < SCENE_CHAMFER_CM_MAX
    return {
        "state": "measured",
        "achieved": achieved,
        "f1_at_1cm": f1,
        "chamfer_cm": chamfer,
        "metres_per_unit": metres,
        "reason": "" if achieved else "F1 or Chamfer is outside the gate",
    }


def _can_inputs(folder: Path) -> dict[str, Any]:
    capture = _first_capture(folder)
    mask = _mask_beside(folder, capture) if capture is not None else _first_mask(folder)
    return {
        "mesh": _first_file(folder, _MESH_NAMES),
        "world_T": _first_file(folder, _WORLD_NAMES),
        "background": _first_file(folder, _BACKGROUND_NAMES),
        "asset": _first_file(folder, _ASSET_NAMES),
        "physics": _physics_from_folder(folder),
        "capture": capture,
        "mask": mask,
        "intrinsics": _intrinsics_from_folder(folder),
        "metres": _metres_from_folder(folder),
    }


def _missing_sentence(missing: list[str]) -> str:
    if not missing:
        return "The required inputs are present."
    names = {
        "mesh": "mesh",
        "world_T": "world_T_object",
        "capture": "capture",
        "mask": "mask",
        "intrinsics": "camera intrinsics",
        "asset": "asset.usd",
        "physics": "physics record",
    }
    words = [names.get(item, item) for item in missing]
    if len(words) == 1:
        return f"Missing {words[0]}."
    if len(words) == 2:
        return f"Missing {words[0]} and {words[1]}."
    return "Missing " + ", ".join(words[:-1]) + f", and {words[-1]}."


def _physics_from_folder(folder: Path) -> dict[str, Any] | None:
    record = _read(folder / "physics.json")
    if _physics_record(record):
        return record
    return None


def _physics_record(physics: dict[str, Any] | None) -> bool:
    if not isinstance(physics, dict) or not physics:
        return False
    for key in ("friction", "static_friction", "dynamic_friction", "restitution", "mass_kg", "material"):
        value = physics.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if _finite(value) is not None:
            return True
    return False


def _metres_from_folder(folder: Path) -> float | None:
    record = _read(folder / "ingest" / "scale.json")
    return _positive(record.get("metres_per_unit"))


def _calibrated_table_height(folder: Path) -> float | None:
    record = _read(folder / "ingest" / "table_height.json")
    return _finite(record.get("table_height_m"))


def _intrinsics_from_folder(folder: Path) -> dict[str, Any] | None:
    for path in (
        folder / "intrinsics.json",
        folder / "colmap" / "transforms.json",
        folder / "colmap" / "orbit1" / "transforms.json",
    ):
        record = _read(path)
        if _intrinsics_ok(record):
            return record
        if _intrinsics_ok(record.get("intrinsics") if isinstance(record.get("intrinsics"), dict) else None):
            return record["intrinsics"]
    return None


def _stored_cameras(folder: Path) -> list[dict[str, Any]]:
    for path in (
        folder / "colmap" / "orbit1" / "transforms.json",
        folder / "colmap" / "transforms.json",
    ):
        record = _read(path)
        frames = record.get("frames")
        if not isinstance(frames, list):
            continue
        cameras = []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("transform_matrix"), list):
                continue
            name = Path(str(frame.get("file_path") or f"cam_{len(cameras):03d}")).stem
            cameras.append({"name": name, "pose": name, "transform": frame["transform_matrix"]})
        if cameras:
            return cameras
    return []


def _iou_frames(folder: Path) -> bool:
    iou = folder / "iou"
    if not iou.is_dir():
        return False
    return any(path.is_file() and path.stat().st_size > 0 for path in iou.rglob("*.png"))


def _first_capture(folder: Path) -> Path | None:
    for directory in (folder / "frames", folder / "colmap"):
        if directory.is_dir():
            for path in sorted(directory.rglob("*.png")):
                if path.is_file() and path.stat().st_size > 0 and "mask" not in path.name:
                    return path
    return None


def _mask_beside(folder: Path, capture: Path | None) -> Path | None:
    if capture is None:
        return _first_mask(folder)
    candidates = [
        capture.with_name(capture.stem + "_mask.png"),
        folder / "masks" / capture.name,
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    return _first_mask(folder)


def _first_mask(folder: Path) -> Path | None:
    masks = folder / "masks"
    if not masks.is_dir():
        return None
    for path in sorted(masks.rglob("*.png")):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _first_file(folder: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = folder / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _existing_file(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
    return None


def _background_points(source: Any) -> tuple[str, np.ndarray | None]:
    if source is None:
        return "missing", None
    if isinstance(source, np.ndarray):
        points = _points_array(source)
        return ("points", points) if points is not None else ("missing", None)
    path = Path(source)
    if not path.is_file() or path.stat().st_size == 0:
        return "missing", None
    points = _read_ascii_ply(path)
    if points is None:
        points = _read_trimesh_points(path)
    if points is None:
        return "unreadable", None
    return "points", points


def _points_array(source: np.ndarray) -> np.ndarray | None:
    cloud = np.asarray(source, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3 or cloud.shape[0] < 3 or not np.isfinite(cloud[:, :3]).all():
        return None
    return cloud[:, :3]


def _read_ascii_ply(path: Path) -> np.ndarray | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    header, marker, body = text.partition("end_header")
    if not marker or "format ascii" not in header:
        return None
    rows = []
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except ValueError:
            return None
    if len(rows) < 3:
        return None
    return np.asarray(rows, dtype=np.float64)


def _read_trimesh_points(path: Path) -> np.ndarray | None:
    try:
        import trimesh

        loaded = trimesh.load(path, process=False)
    except Exception:
        return None
    vertices = getattr(loaded, "vertices", None)
    if vertices is None or len(vertices) < 3:
        return None
    return np.asarray(vertices, dtype=np.float64)[:, :3]


def _as_mesh(mesh: Any) -> dict[str, Any] | None:
    if mesh is None:
        return None
    if isinstance(mesh, dict):
        vertices = np.asarray(mesh.get("vertices"), dtype=np.float64) if mesh.get("vertices") is not None else None
        triangles = np.asarray(mesh.get("triangles"), dtype=np.int64) if mesh.get("triangles") is not None else None
        if vertices is None or triangles is None:
            return None
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] < 3:
            return None
        if triangles.ndim != 2 or triangles.shape[0] < 1 or triangles.shape[1] < 3:
            return None
        if not np.isfinite(vertices[:, :3]).all():
            return None
        return {"vertices": vertices[:, :3], "triangles": triangles[:, :3], "colors": mesh.get("colors")}
    if isinstance(mesh, (str, Path)):
        path = Path(mesh)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        return _mesh_from_path(path)
    return None


def _mesh_from_path(path: Path) -> dict[str, Any] | None:
    try:
        import trimesh

        loaded = trimesh.load(path, process=False, force="mesh")
    except Exception:
        return None
    vertices = getattr(loaded, "vertices", None)
    faces = getattr(loaded, "faces", None)
    if vertices is None or faces is None or len(vertices) < 3 or len(faces) < 1:
        return None
    colors = None
    visual = getattr(loaded, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None) if visual is not None else None
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        rgb = np.asarray(vertex_colors[:, :3], dtype=np.float64)
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        colors = rgb
    return {
        "vertices": np.asarray(vertices, dtype=np.float64)[:, :3],
        "triangles": np.asarray(faces, dtype=np.int64)[:, :3],
        "colors": colors,
    }


def _pose_params(world_T: Any) -> np.ndarray | None:
    if world_T is None:
        return None
    if isinstance(world_T, (str, Path)):
        path = Path(world_T)
        if not path.is_file() or path.stat().st_size == 0:
            return None
        return _pose_params(_read(path))
    if isinstance(world_T, dict):
        if not world_T:
            return None
        matrix = world_T.get("matrix", world_T.get("transform", world_T.get("world_T")))
        if matrix is not None:
            return _pose_params(matrix)
        if "translation" not in world_T and "rotation_wxyz" not in world_T and "scale" not in world_T:
            return None
        scale = _positive(world_T.get("scale", 1.0)) or 1.0
        translation = world_T.get("translation", [0.0, 0.0, 0.0])
        quaternion = world_T.get("rotation_wxyz", [1.0, 0.0, 0.0, 0.0])
        try:
            rotation = Rotation.from_quat(
                [float(quaternion[1]), float(quaternion[2]), float(quaternion[3]), float(quaternion[0])]
            ).as_rotvec()
            trans = np.asarray(translation, dtype=np.float64)
        except (TypeError, ValueError, IndexError):
            return None
        if trans.shape != (3,) or not np.isfinite(trans).all():
            return None
        return np.array([scale, *rotation, *trans], dtype=np.float64)
    array = np.asarray(world_T, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        return None
    linear = array[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(np.mean(scales))
    if not math.isfinite(scale) or scale <= 0.0:
        return None
    rotation = linear / scale
    u_mat, _, vt_mat = np.linalg.svd(rotation)
    rotation = u_mat @ vt_mat
    if np.linalg.det(rotation) < 0.0:
        u_mat[:, -1] *= -1.0
        rotation = u_mat @ vt_mat
    rotvec = Rotation.from_matrix(rotation).as_rotvec()
    return np.array([scale, *rotvec, *array[:3, 3]], dtype=np.float64)


def _load_capture(capture: Any) -> np.ndarray | None:
    if capture is None:
        return None
    if isinstance(capture, np.ndarray):
        return _rgb(capture)
    path = Path(capture)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0


def _load_mask(mask: Any) -> np.ndarray | None:
    if mask is None:
        return None
    if isinstance(mask, np.ndarray):
        return _mask(mask, mask.shape[:2])
    path = Path(mask)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return _mask(image, image.shape[:2])


def _rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image, dtype=np.float64)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    array = array[..., :3]
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def _mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype == np.bool_:
        out = array
    elif np.issubdtype(array.dtype, np.floating):
        out = array > 0.5
    else:
        out = array > 127
    if out.shape[:2] != tuple(shape[:2]):
        resized = cv2.resize(out.astype(np.uint8), (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_NEAREST)
        out = resized > 0
    return np.asarray(out, dtype=bool)


def _sobel(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(np.clip(rgb * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx * gx + gy * gy)
    peak = float(magnitude.max())
    if peak > 0.0:
        magnitude = magnitude / peak
    return magnitude


def camera_intrinsics(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """Map a COLMAP or Nerfstudio camera record onto fx, fy, cx, cy, width, height."""
    if not isinstance(record, dict):
        return None
    if _intrinsics_ok(record):
        return {key: record[key] for key in ("fx", "fy", "cx", "cy", "width", "height")}
    mapped = {
        "fx": record.get("fx", record.get("fl_x")),
        "fy": record.get("fy", record.get("fl_y", record.get("fl_x"))),
        "cx": record.get("cx"),
        "cy": record.get("cy"),
        "width": record.get("width", record.get("w")),
        "height": record.get("height", record.get("h")),
    }
    if _intrinsics_ok(mapped):
        return mapped
    return None


def _intrinsics_ok(intrinsics: Any) -> bool:
    if not isinstance(intrinsics, dict):
        return False
    for key in ("fx", "fy", "cx", "cy", "width", "height"):
        if _finite(intrinsics.get(key)) is None:
            return False
    return int(intrinsics["width"]) > 1 and int(intrinsics["height"]) > 1


def _positive(value: Any) -> float | None:
    number = _finite(value)
    if number is None or number <= 0.0:
        return None
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _matrix_text(matrix: list[Any]) -> str:
    rows = []
    for row in np.asarray(matrix, dtype=np.float64).reshape(4, 4):
        rows.append("(" + ", ".join(f"{float(value):.8g}" for value in row) + ")")
    return "( " + ", ".join(rows) + " )"


def _usd_string(value: str) -> str:
    return value.replace("\\", "/").replace('"', "")


def _ident(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        return f"cam_{cleaned}"
    return cleaned


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _dump(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(record), indent=2) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
