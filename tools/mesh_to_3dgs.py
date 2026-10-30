"""Turn a Polycam mesh or point cloud into a 3D Gaussian splat.

Polycam's free export is geometry (PLY, OBJ, GLB), not camera poses and not
a trained splat. This writes the same splat.ply layout that splatfacto uses:
one Gaussian per surface sample, colored from the mesh, flattened onto the
surface normal. A viewer can open the file. PSNR and SSIM stay unset because
those scores need photographs of the same object.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
from typing import Any

import numpy as np
import trimesh

from tools.common.status import load_status, save_status, set_path
from tools.step02_splatfacto import count_gaussians

SH_C0 = 0.28209479177387814
GAUSSIAN_DTYPE = np.dtype("<f4")
# iPad LiDAR is approximately metres. Millimetre exports and empty scales are not.
PLAUSIBLE_MIN_M = 1e-3
PLAUSIBLE_MAX_M = 20.0


def rgb_to_sh(rgb: np.ndarray) -> np.ndarray:
    """Map RGB in 0..1 to the degree-0 spherical-harmonic coefficients."""
    return (np.clip(rgb, 0.0, 1.0) - 0.5) / SH_C0


def logit(probability: float) -> float:
    p = min(max(probability, 1e-4), 1.0 - 1e-4)
    return float(np.log(p / (1.0 - p)))


def normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) that rotates +Z onto each unit normal."""
    n = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(length, 1e-8, None)
    dot = n[:, 2]
    cross = np.column_stack((-n[:, 1], n[:, 0], np.zeros(len(n))))
    quat = np.column_stack((1.0 + dot, cross))
    opposite = dot < -0.999999
    quat[opposite] = np.array([0.0, 1.0, 0.0, 0.0])
    quat /= np.clip(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8, None)
    return quat


def load_geometry(path: Path) -> trimesh.Trimesh | trimesh.PointCloud:
    loaded = trimesh.load(str(path), process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if _is_mesh(geom)]
        clouds = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.PointCloud)]
        if meshes:
            loaded = trimesh.util.concatenate(meshes)
        elif clouds:
            vertices = np.vstack([np.asarray(cloud.vertices) for cloud in clouds])
            loaded = trimesh.PointCloud(vertices)
        else:
            raise ValueError(f"{path} has no mesh or point cloud")
    if _is_mesh(loaded) or isinstance(loaded, trimesh.PointCloud):
        return loaded
    raise ValueError(f"{path} is {type(loaded).__name__}, expected a mesh or point cloud")


def _is_mesh(geom: Any) -> bool:
    return isinstance(geom, trimesh.Trimesh) and len(getattr(geom, "faces", [])) > 0


def sample_surface(mesh: trimesh.Trimesh, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if float(mesh.area) <= 0:
        raise ValueError("mesh has no surface area")
    points, face_index = trimesh.sample.sample_surface(mesh, count, seed=seed)
    normals = np.asarray(mesh.face_normals[face_index], dtype=np.float64)
    colors = _colors_for_faces(mesh, face_index, len(points))
    return np.asarray(points, dtype=np.float64), normals, colors


def sample_cloud(cloud: trimesh.PointCloud, count: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(cloud.vertices, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("point cloud is empty")
    rng = np.random.default_rng(seed)
    if len(points) >= count:
        chosen = rng.choice(len(points), size=count, replace=False)
    else:
        chosen = rng.choice(len(points), size=count, replace=True)
    points = points[chosen]
    colors = _colors_for_vertices(cloud, chosen, len(points))
    normals = _cloud_normals(cloud, chosen, points)
    return points, normals, colors


def _colors_for_faces(mesh: trimesh.Trimesh, face_index: np.ndarray, count: int) -> np.ndarray:
    colors = getattr(getattr(mesh, "visual", None), "vertex_colors", None)
    if colors is None or len(colors) != len(mesh.vertices):
        return np.full((count, 3), 0.75, dtype=np.float64)
    face_colors = np.asarray(colors)[mesh.faces[face_index]][:, :, :3]
    return face_colors.mean(axis=1) / 255.0


def _colors_for_vertices(cloud: trimesh.PointCloud, index: np.ndarray, count: int) -> np.ndarray:
    colors = getattr(cloud, "colors", None)
    if colors is None or len(colors) != len(cloud.vertices):
        return np.full((count, 3), 0.75, dtype=np.float64)
    return np.asarray(colors)[index][:, :3] / 255.0


def _cloud_normals(cloud: trimesh.PointCloud, index: np.ndarray, points: np.ndarray) -> np.ndarray:
    stored = getattr(cloud, "metadata", {}) or {}
    normals = stored.get("normals")
    if normals is not None and len(normals) == len(cloud.vertices):
        return np.asarray(normals, dtype=np.float64)[index]
    return estimate_normals(points)


def estimate_normals(points: np.ndarray, neighbors: int = 16) -> np.ndarray:
    """PCA normals from nearby points in a voxel grid."""
    if len(points) < 3:
        return np.tile(np.array([0.0, 0.0, 1.0]), (len(points), 1))
    extent = np.ptp(points, axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-9)))
    cell = max((volume * neighbors / len(points)) ** (1.0 / 3.0), 1e-8)
    keys = np.floor((points - points.min(axis=0)) / cell).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for i, key in enumerate(keys):
        buckets.setdefault((int(key[0]), int(key[1]), int(key[2])), []).append(i)
    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    normals = np.zeros_like(points)
    up = np.array([0.0, 0.0, 1.0])
    for i, key in enumerate(keys):
        gathered: list[int] = []
        base = (int(key[0]), int(key[1]), int(key[2]))
        for offset in offsets:
            gathered.extend(buckets.get((base[0] + offset[0], base[1] + offset[1], base[2] + offset[2]), ()))
            if len(gathered) >= 48:
                break
        if len(gathered) < 3:
            normals[i] = up
            continue
        sample = points[gathered[:48]]
        centered = sample - sample.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normals[i] = vh[-1]
    normals[normals[:, 2] < 0] *= -1
    return normals


def apply_length(points: np.ndarray, spacing: float, length_m: float) -> tuple[np.ndarray, float, float]:
    longest = float(np.ptp(points, axis=0).max())
    if longest <= 0 or length_m <= 0:
        raise ValueError("length scale needs a positive measured length and a non-empty cloud")
    factor = length_m / longest
    center = points.mean(axis=0)
    return (points - center) * factor + center, spacing * factor, factor


def build_gaussians(
    points: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    spacing: float,
    opacity: float = 0.9,
    thickness: float = 0.15,
) -> dict[str, np.ndarray]:
    tangent = np.full(len(points), max(spacing, 1e-8), dtype=np.float64)
    normal_scale = tangent * thickness
    scales = np.column_stack((tangent, tangent, normal_scale))
    return {
        "xyz": points.astype(np.float64),
        "normals": normals.astype(np.float64),
        "f_dc": rgb_to_sh(colors),
        "opacity": np.full(len(points), logit(opacity), dtype=np.float64),
        "scale": np.log(scales),
        "rot": normals_to_quaternions(normals),
    }


def write_gaussian_ply(path: Path, gaussians: dict[str, np.ndarray]) -> None:
    xyz = gaussians["xyz"]
    count = len(xyz)
    columns = [
        xyz,
        gaussians["normals"],
        gaussians["f_dc"],
        gaussians["opacity"][:, None],
        gaussians["scale"],
        gaussians["rot"],
    ]
    packed = np.concatenate(columns, axis=1).astype(GAUSSIAN_DTYPE, copy=False)
    names = [
        "x", "y", "z",
        "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property float {name}" for name in names)
    header.append("end_header\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        packed.tofile(handle)


def mesh_spacing(mesh: trimesh.Trimesh, count: int) -> float:
    return float(np.sqrt(mesh.area / count))


def cloud_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.01
    extent = np.ptp(points, axis=0)
    volume = float(np.prod(np.maximum(extent, 1e-9)))
    return float((volume / len(points)) ** (1.0 / 3.0))


def convert_mesh(
    mesh_path: Path,
    destination: Path,
    count: int = 500_000,
    seed: int = 0,
    length_m: float | None = None,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("gaussian count must be at least 1")
    geometry = load_geometry(mesh_path)
    if _is_mesh(geometry):
        points, normals, colors = sample_surface(geometry, count, seed)
        spacing = mesh_spacing(geometry, count)
        kind = "mesh"
    else:
        points, normals, colors = sample_cloud(geometry, count, seed)
        spacing = cloud_spacing(points)
        kind = "point_cloud"
    side = float(np.ptp(points, axis=0).max())
    if length_m is None and not (PLAUSIBLE_MIN_M <= side <= PLAUSIBLE_MAX_M):
        raise ValueError(
            f"longest side is {side:.4g}, outside {PLAUSIBLE_MIN_M:g}–{PLAUSIBLE_MAX_M:g} m. "
            "Pass --length-m with a ruler measurement."
        )
    scale_factor = 1.0
    if length_m is not None:
        points, spacing, scale_factor = apply_length(points, spacing, length_m)
    gaussians = build_gaussians(points, normals, colors, spacing)
    write_gaussian_ply(destination, gaussians)
    written = count_gaussians(destination)
    return {
        "source": "mesh",
        "kind": kind,
        "mesh": str(mesh_path),
        "splat": str(destination),
        "n_gaussians": written,
        "length_m": length_m,
        "scale_factor": scale_factor,
        "psnr": None,
        "ssim": None,
        "photometric": False,
    }


def record_mesh_splat(obj: str, orbit: int, result: dict[str, Any], root: Path | None = None) -> None:
    doc = load_status(obj, root, create=True)
    set_path(doc, f"step02.orbit{orbit}", result)
    save_status(obj, doc, root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Initialize a 3D Gaussian splat from a Polycam mesh or point cloud"
    )
    parser.add_argument("--mesh", type=Path, required=True, help="OBJ, PLY, GLTF, or GLB from Polycam")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--orbit", type=int, default=1)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Defaults to data/<obj>/colmap/orbitN/splat.ply")
    parser.add_argument("--n-gaussians", type=int, default=500_000)
    parser.add_argument("--length-m", type=float, default=None, help="Scale the longest side to this length in metres")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    from tools.ingest import InputError, ingest_mesh

    try:
        result = ingest_mesh(
            args.obj,
            args.mesh,
            args.root,
            length_m=args.length_m,
            n_gaussians=args.n_gaussians,
            orbit=args.orbit,
            seed=args.seed,
        )
        if args.out is not None and Path(result["splat"]) != args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_bytes(Path(result["splat"]).read_bytes())
    except (OSError, ValueError, InputError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    sys.stdout.write(
        f"wrote {result['n_gaussians']} gaussians to {result['splat']}\n"
        f"synthetic views {result['n_synthetic_views']}  photometric scores not measured (no capture photos)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
