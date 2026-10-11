#!/usr/bin/env python
"""Mesh the existing bottle splat with 2DGS TSDF fusion.

The 408,236 cropped Gaussians are converted to surfels in closed form.
No optimizer step runs, and COLMAP is not rerun.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

REPO = Path("/root/SceneagentX/third_party/2d-gaussian-splatting")
sys.path.insert(0, str(REPO))

from gaussian_renderer import render  # noqa: E402
from scene.cameras import Camera  # noqa: E402
from scene.gaussian_model import GaussianModel  # noqa: E402
from utils.graphics_utils import focal2fov  # noqa: E402
from utils.mesh_utils import GaussianExtractor, post_process_mesh  # noqa: E402

ROOT = Path("/root/SceneagentX")
OBJECT_PLY = ROOT / "data/bottle/objects/bottle/object.ply"
WORLD_T = ROOT / "data/bottle/objects/bottle/world_T_object.json"
TRANSFORMS = ROOT / "data/bottle/colmap/orbit1/transforms.json"
DATAPARSER = ROOT / "outputs/orbit1/splatfacto/2026-10-09_234601/dataparser_transforms.json"
IMAGES = ROOT / "data/bottle/colmap/orbit1/images"
OUT = ROOT / "data/bottle/2dgs"
MESH_DIR = ROOT / "data/bottle/mesh"
CHAMFER_MAX_CM = 3.0
F1_MIN = 0.20


def read_table(path: Path) -> tuple[list[str], np.ndarray]:
    raw = path.read_bytes()
    header, marker, payload = raw.partition(b"end_header\n")
    if not marker:
        raise SystemExit(f"{path} has no ply header")
    names: list[str] = []
    count = 0
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            names.append(line.split()[-1])
    table = np.frombuffer(payload, dtype="<f4", count=count * len(names)).reshape(count, len(names))
    return names, np.array(table, dtype=np.float32, copy=True)


def rotation_matrices(quaternions: np.ndarray) -> np.ndarray:
    """3DGS wxyz quaternions to rotation matrices. Columns are the axes."""
    q = quaternions.astype(np.float64)
    q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
    w, x, y, z = q.T
    rot = np.empty((len(q), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def surfel_components(table: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """Turn 3-scale ellipsoids into 2D surfels. Positions and colours stay put."""
    index = {name: i for i, name in enumerate(names)}
    scales = np.exp(table[:, [index["scale_0"], index["scale_1"], index["scale_2"]]].astype(np.float64))
    quats = table[:, [index["rot_0"], index["rot_1"], index["rot_2"], index["rot_3"]]].astype(np.float64)
    rot = rotation_matrices(quats)
    order = np.argsort(scales, axis=1)
    rows = np.arange(len(scales))
    tangent_a = rot[rows, :, order[:, 2]]
    tangent_b = rot[rows, :, order[:, 1]]
    normal = rot[rows, :, order[:, 0]]
    flipped = np.einsum("ij,ij->i", np.cross(tangent_a, tangent_b), normal) < 0
    tangent_b = tangent_b.copy()
    tangent_b[flipped] *= -1
    posed = np.stack((tangent_a, tangent_b, normal), axis=2)
    xyzw = Rotation.from_matrix(posed).as_quat()
    wxyz = np.concatenate((xyzw[:, 3:4], xyzw[:, 0:3]), axis=1).astype(np.float32)
    largest = scales[rows, order[:, 2]]
    middle = scales[rows, order[:, 1]]
    kept = np.log(np.clip(np.stack((largest, middle), axis=1), 1e-12, None)).astype(np.float32)
    return {
        "xyz": table[:, :3],
        "normal": normal.astype(np.float32),
        "f_dc": table[:, [index["f_dc_0"], index["f_dc_1"], index["f_dc_2"]]],
        "f_rest": table[:, [index[f"f_rest_{i}"] for i in range(45)]],
        "opacity": table[:, index["opacity"]],
        "scale": kept,
        "rot": wxyz,
    }


def write_surfel_ply(path: Path, parts: dict[str, np.ndarray]) -> None:
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(45)]
    names += ["opacity", "scale_0", "scale_1", "rot_0", "rot_1", "rot_2", "rot_3"]
    packed = np.concatenate(
        (
            parts["xyz"],
            parts["normal"],
            parts["f_dc"],
            parts["f_rest"],
            parts["opacity"][:, None],
            parts["scale"],
            parts["rot"],
        ),
        axis=1,
    ).astype(np.float32)
    dtype = [(name, "f4") for name in names]
    elements = np.empty(len(packed), dtype=dtype)
    for column, name in enumerate(names):
        elements[name] = packed[:, column]
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))


def training_c2w(matrix: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(transform, dtype=np.float64)
    posed = affine @ np.asarray(matrix, dtype=np.float64)
    posed[:3, 3] *= float(scale)
    return posed


def to_opencv(c2w_gl: np.ndarray) -> np.ndarray:
    return c2w_gl @ np.diag([1.0, -1.0, -1.0, 1.0])


def project_fraction(xyz: np.ndarray, c2w: np.ndarray, fx: float, fy: float, cx: float, cy: float, width: int, height: int) -> float:
    w2c = np.linalg.inv(c2w)
    local = xyz @ w2c[:3, :3].T + w2c[:3, 3]
    depth = local[:, 2]
    valid = depth > 1e-4
    pixels_x = np.zeros(len(xyz))
    pixels_y = np.zeros(len(xyz))
    pixels_x[valid] = fx * local[valid, 0] / depth[valid] + cx
    pixels_y[valid] = fy * local[valid, 1] / depth[valid] + cy
    inside = valid & (pixels_x >= 0) & (pixels_y >= 0) & (pixels_x < width) & (pixels_y < height)
    return float(inside.mean())


def build_cameras(xyz: np.ndarray, image_scale: float) -> tuple[list[Camera], dict]:
    transforms = json.loads(TRANSFORMS.read_text(encoding="utf-8"))
    dataparser = json.loads(DATAPARSER.read_text(encoding="utf-8"))
    center = np.asarray(json.loads(WORLD_T.read_text(encoding="utf-8"))["translation"], dtype=np.float64)
    width = int(transforms["w"])
    height = int(transforms["h"])
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    scaled_w = max(int(round(width * image_scale)), 1)
    scaled_h = max(int(round(height * image_scale)), 1)
    fx_s = fx * scaled_w / width
    fy_s = fy * scaled_h / height
    fov_x = float(focal2fov(fx_s, scaled_w))
    fov_y = float(focal2fov(fy_s, scaled_h))
    sample = xyz[:: max(len(xyz) // 4000, 1)]
    poses = []
    seen = []
    for frame in transforms["frames"]:
        c2w = training_c2w(frame["transform_matrix"], dataparser["transform"], float(dataparser["scale"]))
        c2w = c2w.copy()
        c2w[:3, 3] -= center
        c2w = to_opencv(c2w)
        poses.append(c2w)
        seen.append(project_fraction(sample, c2w, fx, fy, float(transforms["cx"]), float(transforms["cy"]), width, height))
    seen_arr = np.asarray(seen)
    report = {
        "n_cameras": len(poses),
        "median_visible_fraction": float(np.median(seen_arr)),
        "max_visible_fraction": float(seen_arr.max()),
        "image_scale": image_scale,
        "convention": "dataparser training frame, crop center removed, opengl to opencv",
        "trained": False,
        "colmap_rerun": False,
    }
    print(
        f"visible median {report['median_visible_fraction']:.4f} max {report['max_visible_fraction']:.4f}",
        flush=True,
    )
    if report["median_visible_fraction"] < 0.01:
        raise SystemExit("cameras do not see the cropped splat; mesh was not written")
    cameras = []
    for index, (frame, c2w) in enumerate(zip(transforms["frames"], poses)):
        image_path = IMAGES / Path(frame["file_path"]).name
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        if image_scale != 1.0:
            image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        array = torch.from_numpy(np.asarray(image)).float().permute(2, 0, 1) / 255.0
        w2c = np.linalg.inv(c2w)
        cameras.append(
            Camera(
                colmap_id=int(frame.get("colmap_im_id", index)),
                R=w2c[:3, :3].T,
                T=w2c[:3, 3],
                FoVx=fov_x,
                FoVy=fov_y,
                image=array,
                gt_alpha_mask=None,
                image_name=image_path.stem,
                uid=index,
            )
        )
        if (index + 1) % 40 == 0:
            print(f"cameras {index + 1}", flush=True)
    report["n_loaded"] = len(cameras)
    return cameras, report


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(path), mesh)


def score_mesh(mesh: o3d.geometry.TriangleMesh, xyz: np.ndarray) -> dict[str, float]:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    gauss = xyz.astype(np.float32)
    to_surface = scene.compute_distance(o3d.core.Tensor(gauss)).numpy()
    sampled = mesh.sample_points_uniformly(number_of_points=min(40000, max(len(mesh.vertices), 1)))
    surface = np.asarray(sampled.points, dtype=np.float64)
    tree = cKDTree(xyz.astype(np.float64))
    to_gauss, _ = tree.query(surface, workers=-1)
    chamfer_m = 0.5 * (float(to_surface.mean()) + float(to_gauss.mean()))
    precision = float((to_gauss <= 0.01).mean())
    recall = float((to_surface <= 0.01).mean())
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "chamfer_cm": chamfer_m * 100.0,
        "f1_at_1cm": f1,
        "precision_1cm": precision,
        "recall_1cm": recall,
    }


def main() -> None:
    names, table = read_table(OBJECT_PLY)
    if len(table) != 408236:
        raise SystemExit(f"expected 408236 cropped gaussians, found {len(table)}")
    print(f"gaussians {len(table)}", flush=True)
    parts = surfel_components(table, names)
    surfel_path = OUT / "point_cloud" / "iteration_1" / "point_cloud.ply"
    write_surfel_ply(surfel_path, parts)
    print(f"surfel ply {surfel_path}", flush=True)
    cameras, camera_report = build_cameras(parts["xyz"], image_scale=0.5)
    gaussians = GaussianModel(3)
    gaussians.load_ply(str(surfel_path))
    if int(gaussians.get_xyz.shape[0]) != 408236:
        raise SystemExit("2DGS load changed the gaussian count")
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    class Pipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        depth_ratio = 1.0
        debug = False

    extractor = GaussianExtractor(gaussians, render, Pipe(), bg_color=[0, 0, 0])
    extractor.reconstruction(cameras)
    depth_trunc = float(extractor.radius * 2.0)
    voxel_size = depth_trunc / 1024.0
    sdf_trunc = 5.0 * voxel_size
    mesh = extractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    raw_path = OUT / "fuse.ply"
    save_mesh(mesh, raw_path)
    cleaned = post_process_mesh(mesh, cluster_to_keep=50)
    chosen = cleaned if len(cleaned.triangles) > 0 else mesh
    mesh_path = OUT / "mesh.ply"
    save_mesh(chosen, mesh_path)
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    save_mesh(chosen, MESH_DIR / "mesh.ply")
    print(f"vertices {len(chosen.vertices)} faces {len(chosen.triangles)}", flush=True)
    if len(chosen.triangles) == 0 or len(chosen.vertices) == 0:
        report = {
            "mesh": None,
            "reason": "2DGS TSDF wrote no triangles",
            "cameras": camera_report,
            "trained": False,
        }
        (OUT / "extract_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise SystemExit("mesh has no triangles")
    metrics = score_mesh(chosen, np.asarray(parts["xyz"], dtype=np.float64))
    achieved = metrics["chamfer_cm"] < CHAMFER_MAX_CM and metrics["f1_at_1cm"] > F1_MIN
    report = {
        "mesh": str(mesh_path),
        "vertices": int(len(chosen.vertices)),
        "faces": int(len(chosen.triangles)),
        "chamfer_cm": metrics["chamfer_cm"],
        "f1_at_1cm": metrics["f1_at_1cm"],
        "precision_1cm": metrics["precision_1cm"],
        "recall_1cm": metrics["recall_1cm"],
        "achieved": achieved,
        "depth_ratio": 1.0,
        "voxel_size": voxel_size,
        "depth_trunc": depth_trunc,
        "cameras": camera_report,
        "n_gaussians": 408236,
        "trained": False,
        "colmap_rerun": False,
        "unit_note": "coordinates treated as metres; metres_per_unit is not_measured",
    }
    (OUT / "extract_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("chamfer_cm", "f1_at_1cm", "achieved", "vertices", "faces")}, indent=2), flush=True)
    print("EXTRACT_DONE", flush=True)


if __name__ == "__main__":
    main()
