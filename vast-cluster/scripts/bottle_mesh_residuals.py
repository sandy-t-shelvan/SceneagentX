#!/usr/bin/env python
"""Compare the packaged bottle mesh with one capture frame at its COLMAP pose."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

ROOT = Path("/root/SceneagentX")
MESH = ROOT / "data/bottle/2dgs/mesh_package.ply"
TRANSFORMS = ROOT / "data/bottle/colmap/orbit1/transforms.json"
DATAPARSER = ROOT / "outputs/orbit1/splatfacto/2026-10-09_234601/dataparser_transforms.json"
WORLD_T = ROOT / "data/bottle/objects/bottle/world_T_object.json"
IMAGE = ROOT / "data/bottle/colmap/orbit1/images/frame_00113.png"
MASK = ROOT / "data/bottle/masks/frame_00113_mask.png"
OUT = ROOT / "data/bottle/2dgs/align_residuals.json"
FRAME_NAME = "frame_00113"


def training_c2w(matrix: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(transform, dtype=np.float64)
    posed = affine @ np.asarray(matrix, dtype=np.float64)
    posed[:3, 3] *= float(scale)
    return posed


def pose_for_frame() -> tuple[np.ndarray, float, float, float, float, int, int]:
    transforms = json.loads(TRANSFORMS.read_text(encoding="utf-8"))
    dataparser = json.loads(DATAPARSER.read_text(encoding="utf-8"))
    center = np.asarray(json.loads(WORLD_T.read_text(encoding="utf-8"))["translation"], dtype=np.float64)
    chosen = None
    for frame in transforms["frames"]:
        if Path(frame["file_path"]).name == f"{FRAME_NAME}.png":
            chosen = frame
            break
    if chosen is None:
        raise SystemExit(f"{FRAME_NAME} is not in transforms.json")
    c2w_gl = training_c2w(chosen["transform_matrix"], dataparser["transform"], float(dataparser["scale"]))
    c2w_gl = c2w_gl.copy()
    c2w_gl[:3, 3] -= center
    pose_path = ROOT / "data/bottle/2dgs/frame_00113_pose.json"
    pose_path.write_text(
        json.dumps(
            {
                "pose": FRAME_NAME,
                "width": int(transforms["w"]),
                "height": int(transforms["h"]),
                "fx": float(transforms["fl_x"]),
                "fy": float(transforms["fl_y"]),
                "cx": float(transforms["cx"]),
                "cy": float(transforms["cy"]),
                "c2w_opengl": c2w_gl.tolist(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    c2w = c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])
    return (
        c2w,
        float(transforms["fl_x"]),
        float(transforms["fl_y"]),
        float(transforms["cx"]),
        float(transforms["cy"]),
        int(transforms["w"]),
        int(transforms["h"]),
    )


def render_colors(mesh: o3d.geometry.TriangleMesh, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
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
    prim = answer["primitive_ids"].numpy().reshape(height, width)
    uv = answer["primitive_uvs"].numpy().reshape(height, width, 2)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    if colors.size == 0:
        colors = np.full((len(mesh.vertices), 3), 0.7, dtype=np.float64)
    rendered = np.zeros((height, width, 3), dtype=np.float64)
    if np.any(hit):
        faces = triangles[prim[hit]]
        bary_u = uv[hit, 0]
        bary_v = uv[hit, 1]
        bary_w = 1.0 - bary_u - bary_v
        rendered[hit] = (
            colors[faces[:, 0]] * bary_w[:, None]
            + colors[faces[:, 1]] * bary_u[:, None]
            + colors[faces[:, 2]] * bary_v[:, None]
        )
    return rendered, hit


def sobel(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    peak = float(mag.max())
    if peak > 0:
        mag = mag / peak
    return mag


def main() -> None:
    mesh = o3d.io.read_triangle_mesh(str(MESH))
    if len(mesh.triangles) == 0:
        raise SystemExit("package mesh has no triangles")
    c2w, fx, fy, cx, cy, width, height = pose_for_frame()
    image = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(MASK), cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise SystemExit("capture frame or mask is missing")
    if image.shape[1] != width or image.shape[0] != height:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    if mask.shape[1] != width or mask.shape[0] != height:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    rendered, hit = render_colors(mesh, c2w, fx, fy, cx, cy, width, height)
    capture = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    capture_fg = mask > 127
    mesh_fg = hit
    union = np.logical_or(capture_fg, mesh_fg)
    overlap = np.logical_and(capture_fg, mesh_fg)
    union_count = int(union.sum())
    iou = float(overlap.sum() / union_count) if union_count else 0.0
    if union_count:
        photometric = float(np.abs(rendered[union] - capture[union]).mean())
    else:
        photometric = 1.0
    edge = float(np.abs(sobel(cv2.cvtColor((rendered * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)) - sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))).mean())
    record = {
        "state": "measured",
        "achieved": False,
        "pose": FRAME_NAME,
        "capture": f"data/bottle/frames/orbit1/{FRAME_NAME}.png",
        "mesh": "data/bottle/2dgs/mesh_package.ply",
        "photometric_residual": photometric,
        "iou": iou,
        "edge_residual": edge,
        "weights": {"photometric": 1.0, "iou": 2.0, "edge": 0.5},
        "capture_foreground_pixels": int(capture_fg.sum()),
        "mesh_foreground_pixels": int(mesh_fg.sum()),
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    print("RESIDUALS_DONE")


if __name__ == "__main__":
    main()
