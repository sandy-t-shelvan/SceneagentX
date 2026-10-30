"""Block E writers for the can: mesh, one body, joints, deform route.

A skipped check is never a pass. This module does not train a splat, does
not download PartCrafter, CoACD, or fTetWild, and does not invent parts,
joints, mass, or a cloth flag. Scale stays unmet while metres_per_unit is
null, so a mesh file does not make Chamfer achieved.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tools.block_gates import (
    GateError,
    articulate_parts,
    chamfer_cm,
    destruction_record,
    f1_at_1cm,
    pivot_from_geometry,
    press_required,
    write_mesh_score,
)
from tools.colmap_lift import read_splat_columns
from tools.common.paths import object_dir, resolve_root
from tools.orbit_blocks import convex_hull_summary
from tools.partfield_parts import choose_k

OPACITY_FLOOR = 0.05
SIGMA = 2.0
_CHAMFER_CAP = 2048
_SH_C0 = 0.28209479177387814
_DEFORM_ROUTES = frozenset({"cloth", "rope", "soft"})
_STIFFNESS_KEYS = ("stiffness", "youngs_modulus", "bending_stiffness")
ISAAC_HANG = "Isaac sim.reset is known to hang; the joint sweep was not run"
_BODY = {"id": "body", "joint": None, "generated": False, "partfield": False}


def filter_object_gaussians(
    xyz: np.ndarray,
    opacity: np.ndarray,
    inside: np.ndarray,
) -> dict[str, Any]:
    """Drop opacity under 0.05 and points outside the 2-sigma box of the in-mask cloud."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise GateError("gaussian positions must be an N by 3 array")
    points = points[:, :3]
    alpha = np.asarray(opacity, dtype=np.float64).reshape(-1)
    mask = np.asarray(inside, dtype=bool).reshape(-1)
    if len(alpha) != len(points) or len(mask) != len(points):
        raise GateError("gaussian filter lengths do not match")
    keep = np.zeros(len(points), dtype=bool)
    if not mask.any():
        return _filter_report(points, keep, mask, alpha)
    center = points[mask].mean(axis=0)
    spread = points[mask].std(axis=0)
    outside = np.any(np.abs(points - center) > (SIGMA * spread), axis=1)
    low = alpha < OPACITY_FLOOR
    finite = np.isfinite(points).all(axis=1) & np.isfinite(alpha)
    keep = mask & ~outside & ~low & finite
    return _filter_report(points, keep, mask, alpha, outside=outside, low=low)


def contact_points(points: np.ndarray, n: int = 4) -> np.ndarray:
    """Grid on the lowest Z of the cloud. That plane is the unobserved contact face."""
    cloud = np.asarray(points, dtype=np.float64)[:, :3]
    z_value = float(cloud[:, 2].min())
    x0, x1 = float(cloud[:, 0].min()), float(cloud[:, 0].max())
    y0, y1 = float(cloud[:, 1].min()), float(cloud[:, 1].max())
    if x1 <= x0:
        x1 = x0 + 1e-4
    if y1 <= y0:
        y1 = y0 + 1e-4
    xs = np.linspace(x0, x1, int(n))
    ys = np.linspace(y0, y1, int(n))
    return np.array([[x, y, z_value] for x in xs for y in ys], dtype=np.float64)


def poisson_mesh(
    points: np.ndarray,
    normals: np.ndarray | None,
    depth: int = 8,
    planar_prior: bool = False,
) -> Any:
    """Open3D Poisson. Missing points or a failed reconstruct returns no mesh."""
    cloud = np.asarray(points, dtype=np.float64) if points is not None else np.zeros((0, 3))
    if cloud.ndim != 2 or cloud.shape[0] < 4 or cloud.shape[1] < 3:
        return None
    cloud = cloud[:, :3]
    if planar_prior and normals is None and np.isfinite(cloud).all():
        cloud = np.vstack([cloud, contact_points(cloud)])
    if not np.isfinite(cloud).all():
        return None
    try:
        import open3d as o3d
    except ImportError:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(cloud))
    if normals is not None:
        norms = np.asarray(normals, dtype=np.float64)
        if norms.shape[0] != len(cloud) or norms.shape[1] < 3 or not np.isfinite(norms[:, :3]).all():
            return None
        pcd.normals = o3d.utility.Vector3dVector(np.ascontiguousarray(norms[:, :3]))
    else:
        try:
            pcd.estimate_normals()
        except Exception:
            return None
    try:
        mesh, _densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=int(depth))
    except Exception:
        return None
    if mesh is None or len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None
    return mesh


def write_poisson_mesh(points: np.ndarray, normals: np.ndarray | None, dest: Path, depth: int = 8) -> Path | None:
    """Write a Poisson mesh only when reconstruct returns triangles."""
    mesh = poisson_mesh(points, normals, depth=depth)
    return _write_triangle_mesh(Path(dest), mesh)


def score_mesh(
    mesh_path: Path | None,
    reference_points: np.ndarray | None,
    metres_per_unit: float | None,
) -> dict[str, Any]:
    """Chamfer and F1. Achieved requires a finite measured scale, not only a mesh file."""
    path = Path(mesh_path) if mesh_path is not None else None
    exists = path is not None and path.is_file() and path.stat().st_size > 0
    scale = _positive(metres_per_unit)
    chamfer: float | None = None
    f1: float | None = None
    if exists and reference_points is not None:
        vertices = _load_vertices(path)
        reference = np.asarray(reference_points, dtype=np.float64)
        if (
            vertices is not None
            and len(vertices) > 0
            and reference.ndim == 2
            and reference.shape[0] > 0
            and reference.shape[1] >= 3
            and np.isfinite(vertices).all()
            and np.isfinite(reference[:, :3]).all()
        ):
            left = _sample_points(vertices)
            right = _sample_points(reference[:, :3])
            if scale is not None:
                left = left * scale
                right = right * scale
            chamfer = float(chamfer_cm(left, right))
            f1 = float(f1_at_1cm(left, right))
    scratch = Path(tempfile.mkdtemp(prefix="block-e-score-"))
    try:
        gated = write_mesh_score(scratch, path if exists else None, chamfer_cm=chamfer, f1=f1)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    gated = dict(gated)
    gated["metres_per_unit"] = scale
    gated["uv_atlas"] = "unstarted"
    if scale is None:
        gated["achieved"] = False
        if gated.get("state") == "pass":
            gated["state"] = "measured"
        gated["achieved_reason"] = "metres_per_unit is null"
    elif gated.get("achieved") is not True:
        gated["achieved"] = False
        gated["achieved_reason"] = "chamfer or F1 is outside the gate, or the mesh is missing"
    return gated


def write_planar_uv(mesh: Any) -> dict[str, Any]:
    """Write XY UVs in the mesh bbox. Vertex colour alone does not start an atlas."""
    result: dict[str, Any] = {"uv_atlas": "unstarted", "n_uv": 0}
    vertices = _vertices_of(mesh)
    if vertices is None or len(vertices) < 3 or not np.isfinite(vertices).all():
        return result
    low = vertices.min(axis=0)
    span = np.maximum(vertices.max(axis=0) - low, 1e-8)
    uvs = np.column_stack(((vertices[:, 0] - low[0]) / span[0], (vertices[:, 1] - low[1]) / span[1]))
    if isinstance(mesh, dict):
        mesh["uvs"] = uvs
    else:
        try:
            mesh.triangle_uvs = uvs
        except Exception:
            return result
    result["uv_atlas"] = "planar"
    result["n_uv"] = int(len(uvs))
    return result


def write_vertex_colors(mesh: Any, colours: np.ndarray | None) -> dict[str, Any]:
    """Store per-vertex colour when lengths match. A UV atlas is not invented."""
    result: dict[str, Any] = {"uv_atlas": "unstarted", "texture": None, "colors": None, "n_colored": 0}
    vertices = _vertices_of(mesh)
    if colours is None or vertices is None:
        return result
    rgb = np.asarray(colours, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[0] != len(vertices) or rgb.shape[1] < 3:
        return result
    if not np.isfinite(rgb[:, :3]).all():
        return result
    stored = np.clip(rgb[:, :3], 0.0, 1.0)
    result["colors"] = stored
    result["n_colored"] = int(len(stored))
    _store_colors(mesh, stored)
    return result


def resolve_parts(part_masks: Any, scores: dict[int, float] | None) -> list[dict[str, Any]]:
    """No part masks and no winning k is one body. Joints are not invented."""
    winning = choose_k(scores)
    names = _mask_names(part_masks)
    if not names or winning is None or winning <= 1:
        return [_body()]
    chosen = names[: int(winning)]
    if len(chosen) <= 1:
        return [_body()]
    return [
        {"id": name, "joint": None, "generated": False, "partfield": True}
        for name in chosen
    ]


def split_mesh(mesh: Any, labels: Any, dest: Path) -> dict[str, Any]:
    """Write parts/<id>.ply only when labels exist and match the vertex count."""
    dest = Path(dest)
    if labels is None:
        return {"written": [], "reason": "no labels"}
    vertices = _vertices_of(mesh)
    if vertices is None:
        return {"written": [], "reason": "mesh has no vertices"}
    names = np.asarray(labels)
    if names.shape[0] != len(vertices):
        return {"written": [], "reason": "label count does not match vertices"}
    if names.shape[0] == 0:
        return {"written": [], "reason": "no labels"}
    written: list[str] = []
    for label in _label_order(names):
        slug = str(label).replace(" ", "_")
        if not slug or slug == "None":
            continue
        subset = vertices[names == label]
        if len(subset) == 0:
            continue
        path = dest / f"{slug}.ply"
        write_xyz_ply(path, subset)
        written.append(str(path))
    if not written:
        return {"written": [], "reason": "no labels"}
    return {"written": written, "reason": None}


def collision_hulls(mesh: Any, max_hulls: int = 32) -> dict[str, Any]:
    """CoACD when it imports. A trimesh hull is never labeled CoACD."""
    vertices, faces = _mesh_arrays(mesh)
    cap = int(max_hulls)
    if vertices is None or len(vertices) == 0:
        return {
            "coacd": False,
            "convex_hull": "not_built",
            "n_hulls": 0,
            "max_hulls": cap,
            "method": None,
            "reason": "mesh missing",
        }
    module = _try_coacd()
    if module is not None and faces is not None and len(faces) > 0:
        try:
            parts = _coacd_parts(module, vertices, faces, cap)
        except Exception as exc:
            return {
                "coacd": False,
                "convex_hull": "not_built",
                "n_hulls": 0,
                "max_hulls": cap,
                "method": None,
                "n_points": int(len(vertices)),
                "reason": f"coacd failed: {type(exc).__name__}",
            }
        return {
            "coacd": True,
            "convex_hull": "built",
            "n_hulls": len(parts),
            "max_hulls": cap,
            "method": "coacd",
            "n_points": int(len(vertices)),
        }
    summary = dict(convex_hull_summary(vertices))
    summary["coacd"] = False
    summary["vhacd"] = False
    summary["max_hulls"] = cap
    if summary.get("convex_hull") == "built":
        summary["method"] = "trimesh"
        summary["n_hulls"] = 1
    else:
        summary["method"] = None
        summary["n_hulls"] = 0
    return summary


def run_generative_parts(renders: Any, weights: Path | None) -> dict[str, Any]:
    """Refuse a missing PartCrafter or HoloPart weight file. Nothing is downloaded."""
    del renders
    path = Path(weights) if weights else None
    if path is None or not path.is_file() or path.stat().st_size == 0:
        raise GateError("PartCrafter weights are missing")
    return {
        "generated": False,
        "parts": [],
        "reason": "weights file exists; PartCrafter was not run",
    }


def write_can_asset(
    parts: list[dict[str, Any]],
    mesh_name: str,
    dest: Path | None = None,
    obj: str = "can",
) -> dict[str, Any]:
    """URDF joints only for names the parts list already marks revolute or prismatic."""
    articulated = articulate_parts(obj, parts)
    result = dict(articulated)
    result["mesh"] = mesh_name
    result["n_joints"] = len(result["joints"])
    if dest is not None:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(result["urdf"], encoding="utf-8")
        result["path"] = str(dest)
    return result


def write_physics_joints_usda(
    parts: list[dict[str, Any]],
    dest: Path,
    mesh_name: str = "mesh/mesh.ply",
    *,
    mesh_on_disk: bool = False,
    hull: dict[str, Any] | None = None,
    obj: str = "can",
) -> Path:
    """UsdPhysics joints and a drive only for named moving parts. One body has zero joints."""
    articulated = articulate_parts(obj, parts)
    hull_record = hull or {"coacd": False, "convex_hull": "not_built"}
    coacd_flag = hull_record.get("coacd") is True and hull_record.get("method") == "coacd"
    hull_state = hull_record.get("convex_hull") if hull_record.get("convex_hull") in {"built", "not_built"} else "not_built"
    mesh_state = "written" if mesh_on_disk else "unstarted"
    lines = [
        "#usda 1.0",
        "(",
        '    defaultPrim = "can"',
        '    upAxis = "Z"',
        ")",
        'def Xform "can"',
        "{",
        f'    custom string sceneagent:visualMesh = "{mesh_state}"',
        f'    def Mesh "visual" (',
        f"        prepend references = @{mesh_name}@",
        "    )",
        "    {",
        "    }",
        '    def Mesh "collision" (',
        '        prepend apiSchemas = ["PhysicsCollisionAPI"]',
        "    )",
        "    {",
        f'        custom string sceneagent:convexHull = "{hull_state}"',
        f"        custom bool sceneagent:coacd = {'true' if coacd_flag else 'false'}",
        "    }",
    ]
    for joint in articulated["joints"]:
        kind = joint["joint"]
        if kind == "revolute":
            klass = "PhysicsRevoluteJoint"
            drive = "angular"
        elif kind == "prismatic":
            klass = "PhysicsPrismaticJoint"
            drive = "linear"
        else:
            continue
        origin = joint_origin(
            {"joint": kind, "centroid": joint.get("origin"), "origin": joint.get("origin")}
        )
        lines.extend(
            [
                f'    def Xform "{joint["id"]}"',
                "    {",
                "    }",
                f'    def {klass} "{joint["id"]}_joint" (',
                f'        prepend apiSchemas = ["PhysicsDriveAPI:{drive}"]',
                "    )",
                "    {",
                "        rel physics:body0 = </can>",
                f'        rel physics:body1 = </can/{joint["id"]}>',
                f"        point3f physics:localPos0 = ({origin[0]:.6g}, {origin[1]:.6g}, {origin[2]:.6g})",
                f"        float physics:lowerLimit = 0",
                f"        float physics:upperLimit = {float(joint['upper']):.6f}",
                f"        float drive:{drive}:physics:targetPosition = 0",
                "    }",
            ]
        )
    lines.append("}")
    lines.append("")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def joint_origin(part: dict[str, Any]) -> list[float]:
    """Pivot comes from geometry only after the joint type is already named."""
    kind = part.get("joint")
    if not kind:
        return [0.0, 0.0, 0.0]
    centroid = part.get("centroid")
    if centroid is None:
        centroid = part.get("origin")
    if centroid is None:
        return [0.0, 0.0, 0.0]
    found = pivot_from_geometry(str(kind), [float(item) for item in centroid])
    if found is None:
        return [0.0, 0.0, 0.0]
    return found


def record_joint_sweep(
    telemetry: dict[str, Any] | None = None,
    hang_reason: str = ISAAC_HANG,
) -> dict[str, Any]:
    """No telemetry is not_run. interpenetration false and m5_met true need numbers."""
    if not _sweep_telemetry(telemetry):
        return {
            "joint_sweep": "not_run",
            "interpenetration": None,
            "m5_met": None,
            "m5_state": "unmeasured",
            "n_joints_swept": None,
            "max_penetration_m": None,
            "reason": hang_reason,
        }
    assert telemetry is not None
    m5_flag = telemetry.get("m5_met")
    m5_met: bool | None
    if m5_flag is True:
        m5_met = True
    elif m5_flag is False:
        m5_met = False
    else:
        m5_met = None
    return {
        "joint_sweep": "ran",
        "interpenetration": bool(telemetry["interpenetration"]),
        "m5_met": m5_met,
        "m5_state": "measured" if m5_met is not None else "unmeasured",
        "n_joints_swept": int(telemetry["n_joints_swept"]),
        "max_penetration_m": float(telemetry["max_penetration_m"]),
        "reason": None,
    }


def route_deformable(deformable_type: str | None) -> str:
    """Cloth, rope, or soft. Anything else, including a missing type, stays rigid."""
    if deformable_type in _DEFORM_ROUTES:
        return str(deformable_type)
    return "rigid"


def cloth_sheet(nu: int = 3, nv: int = 3, size: float = 1.0) -> dict[str, Any]:
    """Uniform triangle sheet for a cloth fixture. The can writer does not call this."""
    nu = int(nu)
    nv = int(nv)
    if nu < 2 or nv < 2:
        raise GateError("cloth sheet needs at least a 2 by 2 grid")
    xs = np.linspace(0.0, float(size), nu)
    ys = np.linspace(0.0, float(size), nv)
    vertices = np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)
    faces: list[list[int]] = []
    for row in range(nv - 1):
        for col in range(nu - 1):
            corner = row * nu + col
            faces.append([corner, corner + 1, corner + nu])
            faces.append([corner + 1, corner + nu + 1, corner + nu])
    return {
        "route": "cloth",
        "vertices": vertices,
        "faces": np.asarray(faces, dtype=np.int64),
        "uv_atlas": "unstarted",
    }


def rope_capsules(n: int = 4, length: float = 1.0, radius: float = 0.01) -> dict[str, Any]:
    """Capsule chain for a rope fixture. The can writer does not call this."""
    count = int(n)
    if count < 1:
        raise GateError("rope needs at least one capsule")
    span = float(length)
    centers = np.stack((np.linspace(0.0, span, count), np.zeros(count), np.zeros(count)), axis=1)
    return {
        "route": "rope",
        "n_capsules": count,
        "centers": centers,
        "radius": float(radius),
        "joints": "spherical",
    }


def ftetwild_binary() -> str | None:
    """Path to fTetWild when it is already on PATH. This does not download it."""
    for name in ("ftetwild", "FloatTetwild", "fTetWild"):
        found = shutil.which(name)
        if found:
            return found
    return None


def soft_volume(mesh: Any) -> dict[str, Any]:
    """Tetrahedralise only when fTetWild is already installed. A missing binary raises."""
    binary = ftetwild_binary()
    if not binary:
        raise GateError("fTetWild is absent")
    vertices, faces = _mesh_arrays(mesh)
    if vertices is None or faces is None or len(faces) == 0:
        return {
            "route": "soft",
            "ftetwild": True,
            "binary": binary,
            "ran": False,
            "reason": "mesh has no triangles, so fTetWild was not run",
        }
    scratch = Path(tempfile.mkdtemp(prefix="block-e-tet-"))
    source = scratch / "in.ply"
    dest = scratch / "out.ply"
    try:
        write_xyz_ply(source, vertices)
        completed = subprocess.run(
            [binary, "-i", str(source), "-o", str(dest)],
            check=False,
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        shutil.rmtree(scratch, ignore_errors=True)
        return {
            "route": "soft",
            "ftetwild": True,
            "binary": binary,
            "ran": False,
            "reason": f"fTetWild did not finish: {type(exc).__name__}",
        }
    ran = completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0
    shutil.rmtree(scratch, ignore_errors=True)
    return {
        "route": "soft",
        "ftetwild": True,
        "binary": binary,
        "ran": ran,
        "reason": None if ran else f"fTetWild exited {completed.returncode}",
    }


def require_press_picture(path: Path) -> Path:
    """A missing press picture fails the deform check."""
    return press_required(path)


def write_deformable(folder: Path) -> dict[str, Any]:
    """Record the route. Cloth, rope, and soft builders are not run from this writer."""
    folder = Path(folder)
    physics = _read(folder / "physics.json")
    kind = physics.get("deformable_type") if isinstance(physics.get("deformable_type"), str) else None
    route = route_deformable(kind)
    destroyed = destruction_record()
    record = {
        "route": route,
        "skipped": True,
        "destruction": destroyed["destruction"],
        "physgaussian": destroyed["physgaussian"],
        "ran": destroyed["ran"],
        "stiffness": _stiffness(physics),
        "ftetwild": False,
        "cloth": False,
        "rope": False,
        "soft": False,
    }
    _dump(folder / "deformable.json", record)
    return record


def extract_object_mesh(splat: Path, mask: Path, dest: Path) -> dict[str, Any]:
    """Poisson a filtered splat. Missing inputs do not create mesh.ply."""
    dest = Path(dest)
    splat = Path(splat)
    mask = Path(mask)
    unstarted = {
        "state": "unstarted",
        "achieved": False,
        "mesh": None,
        "uv_atlas": "unstarted",
        "trained": False,
        "splatfacto": False,
    }
    if not splat.is_file() or not mask.is_file():
        if splat.is_file():
            reason = "gaussian_in_mask.npy is missing; splat.ply is on disk, so mesh.ply was not written"
        elif mask.is_file():
            reason = "splat.ply is missing; gaussian_in_mask.npy is on disk, so mesh.ply was not written"
        else:
            reason = "splat.ply and gaussian_in_mask.npy are missing, so mesh.ply was not written"
        return {**unstarted, "reason": reason}
    try:
        xyz, opacity, _scales = read_splat_columns(splat)
        inside = np.asarray(np.load(mask), dtype=bool).reshape(-1)
    except Exception as exc:
        return {**unstarted, "reason": f"splat or mask could not be read: {type(exc).__name__}"}
    if len(inside) != len(xyz):
        return {**unstarted, "reason": "mask length does not match splat"}
    filtered = filter_object_gaussians(xyz, opacity, inside)
    kept = filtered["xyz"]
    if len(kept) < 4:
        return {**unstarted, "reason": "filter kept too few gaussians", "filter": _jsonable_filter(filtered)}
    mesh = poisson_mesh(kept, None, planar_prior=True)
    if mesh is None:
        return {**unstarted, "reason": "poisson did not produce a mesh", "filter": _jsonable_filter(filtered)}
    colours = _kept_colours(splat, filtered["keep"])
    if colours is not None:
        baked = _bake_nearest(np.asarray(mesh.vertices), kept, colours)
        write_vertex_colors(mesh, baked)
    atlas = write_planar_uv(mesh)
    written = _write_triangle_mesh(dest, mesh)
    if written is None:
        if dest.is_file():
            dest.unlink()
        return {**unstarted, "reason": "mesh file was not written", "filter": _jsonable_filter(filtered)}
    return {
        "state": "written",
        "achieved": False,
        "mesh": str(written),
        "uv_atlas": atlas["uv_atlas"],
        "planar_prior": True,
        "trained": False,
        "splatfacto": False,
        "n_kept": int(filtered["n_kept"]),
        "reason": "mesh written from the existing splat; scale is still unmet until metres_per_unit is measured",
        "filter": _jsonable_filter(filtered),
        "reference_xyz": np.ascontiguousarray(kept),
    }


def can_mesh_sources(root: Path | None = None) -> tuple[Path | None, Path | None]:
    """Return splat.ply and gaussian_in_mask.npy only when both are on disk."""
    folder = object_dir("can", root)
    for orbit in (1, 2, 3):
        splat = folder / "colmap" / f"orbit{orbit}" / "splat.ply"
        mask = folder / "masks" / f"orbit{orbit}" / "gaussian_in_mask.npy"
        if splat.is_file() and mask.is_file():
            return splat, mask
    return None, None


def metres_per_unit_for(folder: Path) -> float | None:
    """A model_knowledge scale is not a measured metres_per_unit."""
    folder = Path(folder)
    for relative in ("ingest/scale.json", "scale.json"):
        record = _read(folder / relative)
        if not record:
            continue
        method = record.get("method")
        if method in (None, "", "not_measured", "model_knowledge"):
            return None
        return _positive(record.get("metres_per_unit"))
    score = _read(folder / "mesh_score.json")
    if score.get("scale") in (None, "", "model_knowledge", "not_measured"):
        return None
    return _positive(score.get("metres_per_unit"))


def attach_chamfer_block(path: Path, block: dict[str, Any]) -> dict[str, Any]:
    """Add a Chamfer block without dropping silhouette-IoU orbits."""
    existing = _read(path)
    existing["chamfer"] = block
    _dump(path, existing)
    return existing


def publish_can(root: Path | None = None) -> dict[str, Any]:
    """Write the one-body rigid can. A missing splat leaves the mesh unstarted."""
    folder = object_dir("can", root)
    folder.mkdir(parents=True, exist_ok=True)
    splat, mask = can_mesh_sources(root)
    mesh_path = folder / "mesh" / "mesh.ply"
    if splat is None or mask is None:
        mesh_status: dict[str, Any] = {
            "state": "unstarted",
            "achieved": False,
            "mesh": None,
            "uv_atlas": "unstarted",
            "trained": False,
            "splatfacto": False,
            "reason": "splat.ply or gaussian_in_mask.npy is not on disk; mesh.ply was not written",
        }
    else:
        mesh_status = extract_object_mesh(splat, mask, mesh_path)
        reference = mesh_status.pop("reference_xyz", None)
        if mesh_status.get("mesh") and mesh_path.is_file():
            block = score_mesh(mesh_path, reference, metres_per_unit_for(folder))
            block["achieved"] = False if block.get("metres_per_unit") is None else bool(block.get("achieved"))
            attach_chamfer_block(folder / "mesh_score.json", block)
            mesh_status["achieved"] = bool(block.get("achieved"))
    parts = resolve_parts(None, None)
    if mesh_path.is_file() and mesh_path.stat().st_size > 0:
        hull = collision_hulls(mesh_path)
    else:
        hull = {
            "coacd": False,
            "convex_hull": "not_built",
            "n_hulls": 0,
            "method": None,
            "reason": "mesh missing",
        }
    _dump(folder / "ingest" / "hulls.json", hull)
    asset = write_can_asset(parts, "mesh/mesh.ply", folder / "asset.urdf")
    write_physics_joints_usda(
        parts,
        folder / "asset.usda",
        "mesh/mesh.ply",
        mesh_on_disk=mesh_path.is_file() and mesh_path.stat().st_size > 0,
        hull=hull,
    )
    sweep_path = folder / "ingest" / "joint_sweep.json"
    existing_sweep = _read(sweep_path)
    if existing_sweep.get("joint_sweep") == "ran" and _sweep_telemetry(existing_sweep):
        sweep = existing_sweep
    else:
        sweep = record_joint_sweep(None)
        _dump(sweep_path, sweep)
    deformable = write_deformable(folder)
    try:
        run_generative_parts([], folder / "weights" / "partcrafter.pt")
        generated = {"generated": False, "reason": "weights file exists; PartCrafter was not run"}
    except GateError as exc:
        generated = {"generated": False, "reason": str(exc)}
    record = refresh_block_e("can", root)
    record["mesh_status"] = mesh_status
    record["n_joints"] = asset["n_joints"]
    record["deformable"] = {"route": deformable["route"], "skipped": deformable["skipped"]}
    record["generative"] = generated
    return record


def refresh_block_e(obj: str = "can", root: Path | None = None) -> dict[str, Any]:
    """Rewrite ingest/block_e.json from files. A missing artifact stays unachieved."""
    name = str(obj).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise GateError("object name must be a single folder name")
    folder = object_dir(obj, root)
    from tools.sceneagent_blocks import _block_e

    scored = _block_e(obj, folder, resolve_root(root) if root is not None else None)
    steps = [
        _e1_status(folder),
        _e2_status(folder),
        _e3_status(folder),
        _e4_status(folder),
    ]
    record = {
        "block": "E",
        "object": obj,
        "state": scored.get("state") or "not_started",
        "achieved": False,
        "scorer": {
            "name": "_block_e",
            "state": scored.get("state"),
            "result": scored.get("result"),
            "artifact": scored.get("artifact"),
        },
        "m5": {
            "state": "unmeasured",
            "achieved": False,
            "m5_met": None,
            "interpenetration": None,
            "radio": False,
            "reason": (
                "M5 needs a radio-class object, named moving parts, and an Isaac joint sweep "
                "with telemetry. Can has no radio and no sweep telemetry. A skipped check is never a pass."
            ),
        },
        "m5b": {
            "state": "unmeasured",
            "achieved": False,
            "m5b_met": None,
            "towel": False,
            "hose": False,
            "reason": (
                "M5b needs a towel and a hose with PhysX telemetry. Can is not cloth, rope, or soft. "
                "No sheet, capsule chain, or tet mesh was built."
            ),
        },
        "note": (
            "A skipped check is never a pass. M5 and M5b stay unmeasured. "
            "metres_per_unit null keeps the mesh score unachieved. "
            "The can is one body with zero joints and a rigid route."
        ),
        "parts": steps,
    }
    record["achieved"] = False
    _dump(folder / "ingest" / "block_e.json", record)
    return record


def write_xyz_ply(path: Path, points: np.ndarray) -> Path:
    """ASCII xyz point cloud. Used for part splits, not for an invented can mesh."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] < 3:
        raise GateError("ply points must be N by 3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(cloud)}",
        "property float x",
        "property float y",
        "property float z",
        "end_header",
    ]
    for point in cloud[:, :3]:
        lines.append(f"{point[0]:.17g} {point[1]:.17g} {point[2]:.17g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _body() -> dict[str, Any]:
    return dict(_BODY)


def _filter_report(
    points: np.ndarray,
    keep: np.ndarray,
    mask: np.ndarray,
    alpha: np.ndarray,
    outside: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> dict[str, Any]:
    outside = np.zeros(len(points), dtype=bool) if outside is None else outside
    low = alpha < OPACITY_FLOOR if low is None else low
    return {
        "keep": keep,
        "xyz": np.ascontiguousarray(points[keep]),
        "opacity_floor": OPACITY_FLOOR,
        "sigma": SIGMA,
        "n_in_mask": int(mask.sum()),
        "n_kept": int(keep.sum()),
        "dropped_outside_2sigma": int((mask & outside).sum()),
        "dropped_opacity": int((mask & ~outside & low).sum()),
    }


def _jsonable_filter(filtered: dict[str, Any]) -> dict[str, Any]:
    return {
        "opacity_floor": filtered["opacity_floor"],
        "sigma": filtered["sigma"],
        "n_in_mask": filtered["n_in_mask"],
        "n_kept": filtered["n_kept"],
        "dropped_outside_2sigma": filtered["dropped_outside_2sigma"],
        "dropped_opacity": filtered["dropped_opacity"],
    }


def _positive(value: Any) -> float | None:
    number = _finite(value)
    if number is None or number <= 0:
        return None
    return number


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _sample_points(points: np.ndarray, limit: int = _CHAMFER_CAP) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64)
    if len(cloud) <= limit:
        return cloud
    rng = np.random.default_rng(0)
    choice = rng.choice(len(cloud), size=limit, replace=False)
    return cloud[choice]


def _load_vertices(path: Path) -> np.ndarray | None:
    try:
        import trimesh

        loaded = trimesh.load(path, process=False)
        vertices = getattr(loaded, "vertices", None)
        if vertices is not None and len(vertices) > 0:
            return np.asarray(vertices, dtype=np.float64)
    except Exception:
        pass
    return _read_ascii_xyz(path)


def _read_ascii_xyz(path: Path) -> np.ndarray | None:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    header, marker, body = text.partition("end_header")
    if not marker or "format ascii" not in header:
        return None
    count = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    rows = []
    for line in body.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
        if len(rows) == count:
            break
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float64)


def _vertices_of(mesh: Any) -> np.ndarray | None:
    vertices, _faces = _mesh_arrays(mesh)
    return vertices


def _mesh_arrays(mesh: Any) -> tuple[np.ndarray | None, np.ndarray | None]:
    if mesh is None:
        return None, None
    if isinstance(mesh, (str, Path)):
        path = Path(mesh)
        if not path.is_file():
            return None, None
        return _load_vertices(path), _load_faces(path)
    if isinstance(mesh, np.ndarray):
        cloud = np.asarray(mesh, dtype=np.float64)
        if cloud.ndim == 2 and cloud.shape[1] >= 3:
            return cloud[:, :3], None
        return None, None
    if isinstance(mesh, dict):
        vertices = mesh.get("vertices")
        faces = mesh.get("faces")
        if vertices is None:
            return None, None
        cloud = np.asarray(vertices, dtype=np.float64)
        face_array = None if faces is None else np.asarray(faces)
        return cloud[:, :3], face_array
    vertices = getattr(mesh, "vertices", None)
    triangles = getattr(mesh, "triangles", None)
    if triangles is None:
        triangles = getattr(mesh, "faces", None)
    if vertices is None:
        return None, None
    cloud = np.asarray(vertices, dtype=np.float64)
    face_array = None if triangles is None else np.asarray(triangles)
    return cloud[:, :3], face_array


def _load_faces(path: Path) -> np.ndarray | None:
    try:
        import trimesh

        loaded = trimesh.load(path, process=False)
        faces = getattr(loaded, "faces", None)
        if faces is not None and len(faces) > 0:
            return np.asarray(faces)
    except Exception:
        return None
    return None


def _store_colors(mesh: Any, rgb: np.ndarray) -> None:
    if isinstance(mesh, dict):
        mesh["vertex_colors"] = rgb
        return
    try:
        import open3d as o3d
    except ImportError:
        return
    if isinstance(mesh, o3d.geometry.TriangleMesh):
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.ascontiguousarray(rgb))


def _write_triangle_mesh(path: Path, mesh: Any) -> Path | None:
    if mesh is None:
        return None
    try:
        import open3d as o3d
    except ImportError:
        return None
    if not isinstance(mesh, o3d.geometry.TriangleMesh):
        return None
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    wrote = o3d.io.write_triangle_mesh(str(path), mesh)
    if not wrote or not path.is_file() or path.stat().st_size == 0:
        if path.is_file():
            path.unlink()
        return None
    return path


def _mask_names(part_masks: Any) -> list[str]:
    if isinstance(part_masks, dict) and part_masks:
        return [str(key) for key in part_masks.keys()]
    return []


def _label_order(names: np.ndarray) -> list[Any]:
    seen: list[Any] = []
    for label in names.tolist():
        if label not in seen:
            seen.append(label)
    return seen


def _try_coacd() -> Any:
    try:
        import coacd
    except ImportError:
        return None
    return coacd


def _coacd_parts(module: Any, vertices: np.ndarray, faces: np.ndarray, max_hulls: int) -> list[Any]:
    mesh = module.Mesh(np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32))
    try:
        parts = module.run_coacd(mesh, max_convex_hull=int(max_hulls))
    except TypeError:
        parts = module.run_coacd(mesh)
    if not isinstance(parts, list):
        raise GateError("coacd returned no hull list")
    return parts[: int(max_hulls)]


def _sweep_telemetry(telemetry: dict[str, Any] | None) -> bool:
    if not isinstance(telemetry, dict) or telemetry.get("joint_sweep") == "not_run":
        return False
    swept = telemetry.get("n_joints_swept")
    if isinstance(swept, bool) or not isinstance(swept, int) or swept < 0:
        return False
    if not isinstance(telemetry.get("interpenetration"), bool):
        return False
    penetration = _finite(telemetry.get("max_penetration_m"))
    return penetration is not None


def _stiffness(physics: dict[str, Any]) -> float | None:
    if not physics:
        return None
    for key in _STIFFNESS_KEYS:
        if key not in physics:
            continue
        number = _finite(physics.get(key))
        if number is not None:
            return number
    return None


def _kept_colours(splat: Path, keep: np.ndarray) -> np.ndarray | None:
    rgb = _read_splat_rgb(splat)
    if rgb is None or len(rgb) != len(keep):
        return None
    return rgb[np.asarray(keep, dtype=bool)]


def _read_splat_rgb(path: Path) -> np.ndarray | None:
    raw = path.read_bytes()
    header, marker, payload = raw.partition(b"end_header\n")
    if not marker:
        return None
    count = 0
    names: list[str] = []
    for line in header.decode("ascii", errors="replace").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            if len(parts) < 3 or parts[1] != "float":
                return None
            names.append(parts[2])
    needed = ("f_dc_0", "f_dc_1", "f_dc_2")
    if not all(name in names for name in needed):
        return None
    table = np.frombuffer(payload, dtype="<f4", count=count * len(names)).reshape(count, len(names))
    cols = [names.index(name) for name in needed]
    dc = np.asarray(table[:, cols], dtype=np.float64)
    return np.clip(0.5 + _SH_C0 * dc, 0.0, 1.0)


def _bake_nearest(vertices: np.ndarray, source_xyz: np.ndarray, source_rgb: np.ndarray) -> np.ndarray | None:
    if len(vertices) == 0 or len(source_xyz) == 0 or len(source_rgb) != len(source_xyz):
        return None
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    _distance, index = cKDTree(np.asarray(source_xyz, dtype=np.float64)).query(np.asarray(vertices, dtype=np.float64), k=1)
    return np.asarray(source_rgb, dtype=np.float64)[np.asarray(index, dtype=int)]


def _e1_status(folder: Path) -> dict[str, Any]:
    mesh = folder / "mesh" / "mesh.ply"
    exists = mesh.is_file() and mesh.stat().st_size > 0
    score = _read(folder / "mesh_score.json")
    chamfer = score.get("chamfer") if isinstance(score.get("chamfer"), dict) else None
    scale = None if chamfer is None else _positive(chamfer.get("metres_per_unit"))
    if scale is None:
        scale = metres_per_unit_for(folder)
    if not exists:
        return {
            "id": "E1",
            "does": "Filter the object splat and Poisson a mesh. Score Chamfer and F1 only in measured metres.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "mesh": False,
            "uv_atlas": "unstarted",
            "metres_per_unit": None,
            "chamfer_cm": None,
            "f1_at_1cm": None,
            "trained": False,
            "reason": _missing_mesh_reason(folder),
        }
    numbers_ok = (
        chamfer is not None
        and chamfer.get("chamfer_cm") is not None
        and chamfer.get("f1_at_1cm") is not None
        and float(chamfer["chamfer_cm"]) < 3.0
        and float(chamfer["f1_at_1cm"]) > 0.20
    )
    achieved = bool(numbers_ok and scale is not None and chamfer.get("achieved") is True)
    return {
        "id": "E1",
        "does": "Filter the object splat and Poisson a mesh. Score Chamfer and F1 only in measured metres.",
        "artifact": "data/can/mesh/mesh.ply",
        "state": "partial",
        "achieved": achieved,
        "mesh": True,
        "uv_atlas": "unstarted",
        "metres_per_unit": scale,
        "chamfer_cm": None if chamfer is None else chamfer.get("chamfer_cm"),
        "f1_at_1cm": None if chamfer is None else chamfer.get("f1_at_1cm"),
        "trained": False,
        "reason": (
            "mesh.ply exists and metres_per_unit is a measured scale."
            if achieved
            else "mesh.ply exists; metres_per_unit is null, so Chamfer and F1 stay unachieved."
        ),
    }


def _missing_mesh_reason(folder: Path) -> str:
    """Name the input that is absent. A mask on disk is not reported as missing."""
    splats = []
    masks = []
    for orbit in (1, 2, 3):
        if (folder / "colmap" / f"orbit{orbit}" / "splat.ply").is_file():
            splats.append(orbit)
        if (folder / "masks" / f"orbit{orbit}" / "gaussian_in_mask.npy").is_file():
            masks.append(orbit)
    if splats and masks:
        return "splat.ply and gaussian_in_mask.npy are on disk, but poisson did not write mesh.ply."
    if masks and not splats:
        return "gaussian_in_mask.npy is on disk. splat.ply is missing, so mesh.ply was not written."
    if splats and not masks:
        return "splat.ply is on disk. gaussian_in_mask.npy is missing, so mesh.ply was not written."
    return "splat.ply and gaussian_in_mask.npy are missing, so mesh.ply was not written."


def _sources_on_disk(folder: Path) -> bool:
    for orbit in (1, 2, 3):
        splat = folder / "colmap" / f"orbit{orbit}" / "splat.ply"
        mask = folder / "masks" / f"orbit{orbit}" / "gaussian_in_mask.npy"
        if splat.is_file() and mask.is_file():
            return True
    return False


def _e2_status(folder: Path) -> dict[str, Any]:
    part_dir = folder / "parts"
    part_files = sorted(path.name for path in part_dir.glob("*.ply")) if part_dir.is_dir() else []
    hull = _read(folder / "ingest" / "hulls.json")
    coacd_flag = hull.get("coacd") is True and hull.get("method") == "coacd"
    hull_state = hull.get("convex_hull") if hull.get("convex_hull") in {"built", "not_built"} else "not_built"
    urdf = folder / "asset.urdf"
    one_body = urdf.is_file() and "<joint " not in urdf.read_text(encoding="utf-8")
    if part_files:
        state = "partial"
        reason = "Part ply files exist. Joints were not added from the labels."
    elif one_body:
        state = "partial"
        reason = (
            "No part masks, so the object stays one body. parts/*.ply was not written. "
            f"convex_hull {hull_state}; coacd {str(coacd_flag).lower()}."
        )
    else:
        state = "unstarted"
        reason = "No part masks and no one-body URDF yet."
    return {
        "id": "E2",
        "does": "One body when part masks are missing. CoACD hulls only when CoACD imports.",
        "artifact": "data/can/ingest/hulls.json" if hull else "",
        "state": state,
        "achieved": False,
        "one_body": one_body,
        "partfield": False,
        "generated": False,
        "coacd": coacd_flag,
        "convex_hull": hull_state,
        "parts_ply": part_files,
        "reason": reason,
    }


def _e3_status(folder: Path) -> dict[str, Any]:
    urdf = folder / "asset.urdf"
    usda = folder / "asset.usda"
    text = urdf.read_text(encoding="utf-8") if urdf.is_file() else ""
    usda_text = usda.read_text(encoding="utf-8") if usda.is_file() else ""
    n_joints = text.count("<joint ")
    sweep = _read(folder / "ingest" / "joint_sweep.json")
    ran = sweep.get("joint_sweep") == "ran" and _sweep_telemetry(sweep)
    if ran:
        inter = sweep.get("interpenetration") if isinstance(sweep.get("interpenetration"), bool) else None
        joint_sweep = "ran"
    else:
        inter = None
        joint_sweep = "not_run"
    if urdf.is_file() or usda.is_file():
        state = "partial"
    else:
        state = "unstarted"
    return {
        "id": "E3",
        "does": "URDF and UsdPhysics joints only for named moving parts. Isaac sweep needs telemetry.",
        "artifact": "data/can/asset.urdf" if urdf.is_file() else "",
        "state": state,
        "achieved": False,
        "n_joints": n_joints,
        "usda": usda.is_file(),
        "physics_revolute": "PhysicsRevoluteJoint" in usda_text,
        "physics_prismatic": "PhysicsPrismaticJoint" in usda_text,
        "joint_sweep": joint_sweep,
        "interpenetration": inter,
        "m5_met": None,
        "reason": (
            f"{n_joints} URDF joints. Isaac sweep is {joint_sweep}. "
            "M5 stays unmeasured without radio telemetry."
        ),
    }


def _e4_status(folder: Path) -> dict[str, Any]:
    record = _read(folder / "deformable.json")
    if not record:
        return {
            "id": "E4",
            "does": "Route cloth, rope, or soft. A missing type stays rigid and builds nothing.",
            "artifact": "",
            "state": "unstarted",
            "achieved": False,
            "route": None,
            "skipped": None,
            "destruction": "not_run",
            "physgaussian": False,
            "stiffness": None,
            "reason": "deformable.json was not written.",
        }
    destruction = record.get("destruction") if record.get("destruction") == "not_run" else "not_run"
    return {
        "id": "E4",
        "does": "Route cloth, rope, or soft. A missing type stays rigid and builds nothing.",
        "artifact": "data/can/deformable.json",
        "state": "partial",
        "achieved": False,
        "route": record.get("route"),
        "skipped": record.get("skipped") is True,
        "destruction": destruction,
        "physgaussian": record.get("physgaussian") is True,
        "stiffness": record.get("stiffness"),
        "ftetwild": record.get("ftetwild") is True,
        "reason": "Route recorded. Destruction was not run. No cloth, rope, or tet mesh was built.",
    }


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _dump(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Block E for the can")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    record = publish_can(args.root)
    print(
        json.dumps(
            {
                "state": record.get("state"),
                "achieved": record.get("achieved"),
                "m5": record["m5"]["state"],
                "m5b": record["m5b"]["state"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
