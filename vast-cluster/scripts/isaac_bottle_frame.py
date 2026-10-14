#!/usr/bin/env python
"""One Isaac render of the bottle mesh from the frame_00113 COLMAP pose.

The picture is written only after a non-empty RGB buffer comes back.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# #region agent log
def _agent_log(hypothesis_id: str, message: str, data: dict) -> None:
    line = json.dumps({
        "sessionId": "e7847c",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": "isaac_bottle_frame.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    })
    for path in (Path("debug-e7847c.log"), Path("/root/SceneagentX/debug-e7847c.log")):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            continue
# #endregion

# #region agent log
def _dbg246(hypothesis_id: str, message: str, data: dict) -> None:
    line = json.dumps({
        "sessionId": "246e5f",
        "runId": "render",
        "hypothesisId": hypothesis_id,
        "location": "isaac_bottle_frame.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    })
    for path in (Path("/root/SceneagentX/debug-246e5f.log"), Path("debug-246e5f.log")):
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            continue
# #endregion

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usda", required=True)
parser.add_argument("--png", required=True)
parser.add_argument("--pose", required=True)
parser.add_argument("--steps", type=int, default=12)
parser.add_argument("--width", type=int, default=0)
parser.add_argument("--height", type=int, default=0)
parser.add_argument("--raster", action="store_true", help="Storm/pxr hydra instead of RTX")
parser.add_argument("--update-only", action="store_true", help="Use simulation_app.update instead of orchestrator.step")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
from PIL import Image
from pxr import Gf, UsdGeom, UsdLux

import omni.replicator.core as rep
import omni.usd


def _stage_loading(context) -> tuple[bool, object]:
    try:
        from isaacsim.core.utils.stage import is_stage_loading
        return bool(is_stage_loading()), "isaacsim.core.utils.stage"
    except Exception:
        pass
    try:
        from omni.isaac.core.utils.stage import is_stage_loading
        return bool(is_stage_loading()), "omni.isaac.core.utils.stage"
    except Exception:
        pass
    try:
        status = context.get_stage_loading_status()
        if isinstance(status, tuple) and len(status) >= 3:
            loaded, total = int(status[1]), int(status[2])
            return (total > 0 and loaded < total), status
        return bool(status), status
    except Exception as exc:
        return False, f"status_unavailable:{type(exc).__name__}"


def wait_until_stage_loaded(context, max_seconds: float = 45.0) -> None:
    print("STAGE_WAIT_START", flush=True)
    for _ in range(5):
        simulation_app.update()
    deadline = time.time() + max_seconds
    last = None
    while time.time() < deadline:
        loading, detail = _stage_loading(context)
        mark = (loading, str(detail))
        if mark != last:
            print("STAGE_LOAD", loading, detail, flush=True)
            last = mark
        if not loading:
            break
        simulation_app.update()
    for _ in range(12):
        simulation_app.update()
    loading, detail = _stage_loading(context)
    print("STAGE_WAIT_DONE", loading, detail, flush=True)


def _color_cube(stage) -> None:
    """Give the untextured cube a preview color so a white blow-out is not the only result."""
    from pxr import Sdf, UsdShade

    prim = stage.GetPrimAtPath("/bottle")
    if not prim.IsValid():
        return
    material = UsdShade.Material.Define(stage, "/World/CubeMat")
    shader = UsdShade.Shader.Define(stage, "/World/CubeMat/Preview")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.15, 0.35, 0.85))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.45)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(prim).Bind(material)


def _camera_matrix(pose: dict) -> np.ndarray:
    """Look at the cube from the stored eye. A raw OpenGL matrix put the camera on the surface."""
    eye = pose.get("eye")
    target = pose.get("target")
    if not isinstance(eye, list) or not isinstance(target, list):
        # OpenGL c2w is column-vector (eye in the last column). This xform
        # stores camera axes as rows and the eye on the last row.
        return np.asarray(pose["c2w_opengl"], dtype=np.float64).T
    eye_v = np.asarray(eye, dtype=np.float64) * 2.5
    target_v = np.asarray(target, dtype=np.float64)
    forward = target_v - eye_v
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)
    # USD multiplies row vectors, so the camera axes are rows and the eye is the last row.
    matrix = np.eye(4)
    matrix[0, 0:3] = right
    matrix[1, 0:3] = true_up
    matrix[2, 0:3] = -forward
    matrix[3, 0:3] = eye_v
    return matrix


def main() -> None:
    pose = json.loads(Path(args.pose).read_text(encoding="utf-8"))
    context = omni.usd.get_context()
    context.open_stage(args.usda)
    stage = context.get_stage()
    if stage is None:
        raise RuntimeError(f"kit did not open {args.usda}")
    print("STAGE_OPEN", args.usda, flush=True)
    bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPrimAtPath("/bottle"))
    box = bounds.ComputeAlignedRange()
    low, high = box.GetMin(), box.GetMax()
    # #region agent log
    _agent_log("D", "stage bounds before render", {
        "usda": args.usda,
        "min": [low[0], low[1], low[2]],
        "max": [high[0], high[1], high[2]],
    })
    # #endregion
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(400.0)
    # The blue preview shader is only for the untextured cube. scene.usda keeps its own look.
    if Path(args.usda).name == "cube.usda":
        _color_cube(stage)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    width = int(pose["width"]) if args.width <= 0 else int(args.width)
    height = int(pose["height"]) if args.height <= 0 else int(args.height)
    aperture = 20.955
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * height / width)
    camera.CreateFocalLengthAttr(float(pose["fx"]) / width * aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    matrix = _camera_matrix(pose)
    transform = Gf.Matrix4d(
        float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3]),
        float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3]),
        float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3]),
        float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3]),
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
    # #region agent log
    eye_row = [float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2])]
    print("CAMERA_EYE", eye_row, flush=True)
    _agent_log("B", "camera origin versus stage", {
        "origin": eye_row,
        "width": width,
        "height": height,
    })
    # #endregion
    wait_until_stage_loaded(context)
    if args.raster:
        try:
            import carb
            carb.settings.get_settings().set("/renderer/active", "pxr")
        except Exception as exc:
            print("RASTER_SET_FAIL", type(exc).__name__, exc, flush=True)
        print("RENDER_MODE raster", width, height, "steps", args.steps, flush=True)
    else:
        rep.settings.set_render_rtx_realtime(antialiasing="DLSS")
        print("RTX_REALTIME DLSS", width, height, "steps", args.steps, flush=True)
    # #region agent log
    _renderer = None
    try:
        import carb
        _renderer = carb.settings.get_settings().get("/renderer/active")
    except Exception as exc:
        _renderer = f"unavailable:{type(exc).__name__}"
    _dbg246("H-RTX", "renderer before first step", {
        "renderer": _renderer,
        "raster": bool(args.raster),
        "width": width,
        "height": height,
        "steps": int(args.steps),
    })
    # #endregion
    render_product = rep.create.render_product("/World/Camera", (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    print("annotator attached", flush=True)
    wait_until_stage_loaded(context, max_seconds=20.0)
    image = None
    for step in range(args.steps):
        if args.update_only:
            print("before app update", step, flush=True)
            # #region agent log
            _dbg246("H-UPDATE", "entering simulation_app.update", {"step": step})
            # #endregion
            simulation_app.update()
            print("after app update", step, flush=True)
            # #region agent log
            _dbg246("H-UPDATE", "returned from simulation_app.update", {"step": step})
            # #endregion
        else:
            # #region agent log
            _agent_log("A", "before orchestrator step", {"step": step})
            # #endregion
            print("before orchestrator step", step, flush=True)
            # #region agent log
            _dbg246("H-STEP", "entering orchestrator.step", {"step": step})
            # #endregion
            rep.orchestrator.step()
            # #region agent log
            _agent_log("A", "after orchestrator step", {"step": step})
            # #endregion
            print("after orchestrator step", step, flush=True)
            # #region agent log
            _dbg246("H-STEP", "returned from orchestrator.step", {"step": step})
            # #endregion
        raw = annotator.get_data()
        data = np.asarray(raw) if raw is not None else np.empty((0,))
        # #region agent log
        _agent_log("A", "rgb buffer", {"step": step, "size": int(data.size), "ndim": int(data.ndim)})
        # #endregion
        if data.ndim < 3 or data.size == 0:
            print(f"RGB_EMPTY step {step}", flush=True)
            continue
        rgb = data[:, :, :3]
        if rgb.size == 0:
            print(f"RGB_EMPTY step {step}", flush=True)
            continue
        spread = float(rgb.std())
        print(f"RGB step {step} shape {rgb.shape} std {spread:.3f}", flush=True)
        if spread <= 0:
            print(f"RGB_FLAT step {step}", flush=True)
            continue
        if args.steps > 1 and spread < 2:
            continue
        image = rgb
        break
    if image is None:
        raise RuntimeError("Isaac returned no varying RGB frame")
    destination = Path(args.png)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(destination) + ".partial")
    rgb8 = np.asarray(image)
    if rgb8.dtype != np.uint8:
        rgb8 = np.clip(rgb8, 0, 255).astype(np.uint8)
    Image.fromarray(rgb8).save(partial, format="PNG")
    partial.replace(destination)
    print("ISAAC_FRAME_OK", destination, destination.stat().st_size, flush=True)


try:
    main()
except Exception as exc:
    print(f"ISAAC_FRAME_FAIL {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    simulation_app.close()
