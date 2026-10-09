"""Choose the SceneAgent ingest path from the file the user has.

A video or a folder of images is the COLMAP path. A lone Polycam mesh
(.obj, .ply, .gltf, .glb) has no photographs and no camera poses, so COLMAP
and splatfacto are not run. The mesh surface is sampled into a splat and a
ring of synthetic views is rendered for the later labeling steps.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import json
from typing import Any

import cv2
import numpy as np

from tools.common.paths import renders_dir
from tools.common.status import load_status, save_status, set_path
from tools.mesh_to_3dgs import (
    _colors_for_faces,
    _colors_for_vertices,
    _is_mesh,
    convert_mesh,
    load_geometry,
    record_mesh_splat,
)
from tools.step02_splatfacto import splat_path

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
MESH_SUFFIXES = {".obj", ".ply", ".gltf", ".glb"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CAPTURE_DIRS = ("images", "frames", "raw")
COLMAP_BLOCKS = {"B2", "B3"}


class InputError(ValueError):
    """The path is missing, mixed, or not a video or a mesh."""


def classify_input(path: Path) -> dict[str, Any]:
    """Return input.kind of ``video`` or ``polycam_mesh``.

    A mesh that sits next to cameras.json, a capture-image folder, or a video
    is refused. The message names those files so the two paths stay separate.
    """
    path = Path(path)
    if not path.exists():
        raise InputError(f"input not found: {path}")
    if path.is_file():
        return _classify_file(path)
    return _classify_directory(path)


def colmap_refused(doc: dict[str, Any]) -> bool:
    return (doc.get("input") or {}).get("kind") == "polycam_mesh"


def refusal_message() -> str:
    return (
        "COLMAP is refused for a mesh-only Polycam export (.obj, .ply, or .gltf). "
        "No capture images or camera poses were provided.\n"
    )


def ingest_mesh(
    obj: str,
    mesh_path: Path,
    root: Path | None = None,
    length_m: float | None = None,
    n_gaussians: int = 500_000,
    n_views: int = 8,
    orbit: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Sample the mesh into a splat and write a synthetic camera orbit."""
    record = classify_input(mesh_path)
    if record["kind"] != "polycam_mesh":
        raise InputError(f"{mesh_path} is {record['kind']}, expected a Polycam mesh")
    mesh = Path(record["mesh"])
    destination = splat_path(obj, orbit, root)
    result = convert_mesh(mesh, destination, n_gaussians, seed, length_m)
    view_dir = renders_dir(obj, "synthetic", root)
    views = render_synthetic_orbit(mesh, view_dir, n_views=n_views, seed=seed)
    result.update(
        {
            "synthetic": True,
            "synthetic_views": str(view_dir),
            "n_synthetic_views": len(views),
            "psnr": None,
            "ssim": None,
            "photometric": False,
        }
    )
    record_mesh_splat(obj, orbit, result, root)
    record["synthetic"] = True
    record["n_synthetic_views"] = len(views)
    _record_input(obj, record, root)
    return result


def render_synthetic_orbit(
    mesh_path: Path,
    dest: Path,
    n_views: int = 8,
    size: tuple[int, int] = (320, 240),
    seed: int = 0,
) -> list[Path]:
    """Render a ring of views of the mesh. These are not capture photographs."""
    del seed
    if n_views < 1:
        raise ValueError("a synthetic orbit needs at least one view")
    geometry = load_geometry(mesh_path)
    dest.mkdir(parents=True, exist_ok=True)
    width, height = size
    frames: list[dict[str, Any]] = []
    written: list[Path] = []
    for index in range(n_views):
        azimuth = 360.0 * index / n_views
        image = _render_view(geometry, azimuth, width, height)
        path = dest / f"view_{index:03d}.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError(f"Could not write {path}")
        written.append(path)
        frames.append({"file": path.name, "azimuth_deg": azimuth, "elevation_deg": 20.0})
    payload = {"synthetic": True, "capture_images": False, "cameras": False, "frames": frames}
    (dest / "cameras.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return written


def _classify_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return _video_record(path, cameras=False, capture_images=True)
    if suffix in IMAGE_SUFFIXES:
        return _video_record(path, cameras=False, capture_images=True)
    if suffix in MESH_SUFFIXES:
        extras = _capture_extras(path.parent, ignore=path)
        if extras:
            raise InputError(_mixed_message(path, extras))
        return _mesh_record(path)
    raise InputError(f"unsupported input {path.name}. Expected a video or an .obj, .ply, .gltf, or .glb mesh.")


def _classify_directory(path: Path) -> dict[str, Any]:
    meshes = _files_with(path, MESH_SUFFIXES)
    videos = _files_with(path, VIDEO_SUFFIXES)
    images = _files_with(path, IMAGE_SUFFIXES)
    extras = _capture_extras(path)
    if meshes and extras:
        raise InputError(_mixed_message(meshes[0], extras))
    if len(meshes) > 1 and not videos and not images:
        names = ", ".join(item.name for item in meshes)
        raise InputError(f"{path} has more than one mesh ({names}). Pass one .obj, .ply, or .gltf.")
    if len(meshes) == 1 and not videos and not images and not extras:
        return _mesh_record(meshes[0])
    if videos or images:
        source = videos[0] if videos else path
        cameras = (path / "cameras.json").is_file() or (path / "sparse" / "0").is_dir()
        return _video_record(source, cameras=cameras, capture_images=True)
    raise InputError(f"{path} has no video and no .obj, .ply, .gltf, or .glb mesh.")


def _capture_extras(directory: Path, ignore: Path | None = None) -> list[Path]:
    if not directory.is_dir():
        return []
    extras: list[Path] = []
    cameras = directory / "cameras.json"
    if cameras.is_file():
        extras.append(cameras)
    sparse = directory / "sparse" / "0"
    if sparse.is_dir():
        extras.append(sparse)
    for name in CAPTURE_DIRS:
        folder = directory / name
        if folder.is_dir() and (_files_with(folder, IMAGE_SUFFIXES) or _files_with(folder, VIDEO_SUFFIXES)):
            extras.append(folder)
    for path in directory.iterdir():
        if ignore is not None and path.resolve() == ignore.resolve():
            continue
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            extras.append(path)
    return extras


def _files_with(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffixes)


def _video_record(path: Path, cameras: bool, capture_images: bool) -> dict[str, Any]:
    return {
        "kind": "video",
        "path": str(path),
        "mesh": None,
        "cameras": cameras,
        "capture_images": capture_images,
        "synthetic": False,
    }


def _mesh_record(path: Path) -> dict[str, Any]:
    return {
        "kind": "polycam_mesh",
        "path": str(path),
        "mesh": str(path),
        "cameras": False,
        "capture_images": False,
        "synthetic": True,
    }


def _mixed_message(mesh: Path, extras: list[Path]) -> str:
    names = ", ".join(str(path) for path in extras)
    return (
        f"{mesh} is a mesh, and these capture files are beside it: {names}. "
        "A mesh-only Polycam export cannot be mixed with images or camera poses. "
        "Pass the mesh alone, or pass the image set as a video ingest."
    )


def _record_input(obj: str, record: dict[str, Any], root: Path | None) -> None:
    doc = load_status(obj, root, create=True)
    set_path(doc, "input", record)
    save_status(obj, doc, root)


def _render_view(geometry: Any, azimuth_deg: float, width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 3), 244, dtype=np.uint8)
    if _is_mesh(geometry):
        points = np.asarray(geometry.vertices, dtype=np.float64)
        faces = np.asarray(geometry.faces)
        face_colors = _colors_for_faces(geometry, np.arange(len(faces)), len(faces))
    else:
        points = np.asarray(geometry.vertices, dtype=np.float64)
        faces = None
        face_colors = _colors_for_vertices(geometry, np.arange(len(points)), len(points))
    if len(points) == 0:
        return image
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = float(np.linalg.norm(points - center, axis=1).max())
    radius = max(radius, 1e-4)
    eye_offset = _orbit_offset(azimuth_deg, 20.0, distance=2.6 * radius)
    pixels, depth = _project(points, center + eye_offset, center, width, height)
    if faces is not None:
        order = np.argsort(depth[faces].mean(axis=1))[::-1]
        for face_index in order:
            face = faces[face_index]
            if float(depth[face].min()) <= radius * 0.05:
                continue
            polygon = np.round(pixels[face]).astype(np.int32)
            color = _bgr(face_colors[face_index])
            cv2.fillConvexPoly(image, polygon, color)
    else:
        for index in np.argsort(depth)[::-1]:
            if depth[index] <= radius * 0.05:
                continue
            center_px = tuple(np.round(pixels[index]).astype(int))
            cv2.circle(image, center_px, 2, _bgr(face_colors[index]), thickness=-1)
    return image


def _orbit_offset(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    return distance * np.array(
        [
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        ]
    )


def _project(
    points: np.ndarray,
    eye: np.ndarray,
    target: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    forward = target - eye
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.95:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(right, forward)
    rotation = np.stack([right, -up, forward], axis=0)
    camera = (rotation @ (points - eye).T).T
    depth = camera[:, 2]
    focal = (width / 2.0) / np.tan(np.deg2rad(40.0) / 2.0)
    safe = np.clip(depth, 1e-4, None)
    pixels = np.column_stack(
        (
            focal * camera[:, 0] / safe + width / 2.0,
            focal * camera[:, 1] / safe + height / 2.0,
        )
    )
    return pixels, depth


def _bgr(rgb: np.ndarray) -> tuple[int, int, int]:
    red, green, blue = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(int)
    return int(blue), int(green), int(red)
