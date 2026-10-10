"""Project COLMAP cameras onto a splatfacto cloud.

The splat lives in the nerfstudio training frame. Camera matrices in
transforms.json are mapped with dataparser_transforms.json before projection.
A Gaussian is inside the object when it is on the front surface of a mask pixel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def training_c2w(matrix: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    """Map a saved camera-to-world matrix into the splatfacto training frame."""
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :] = np.asarray(transform, dtype=np.float64)
    posed = affine @ np.asarray(matrix, dtype=np.float64)
    posed[:3, 3] *= float(scale)
    return posed


def distort_opencv(
    x: np.ndarray,
    y: np.ndarray,
    k1: float,
    k2: float,
    p1: float,
    p2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward OpenCV distortion of normalized image coordinates."""
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
    x_distorted = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    y_distorted = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_distorted, y_distorted


def project_points(
    points: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: tuple[float, float, float, float],
    convention: str = "opengl",
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixels. Depth is positive in front of the camera."""
    rotation = np.asarray(c2w, dtype=np.float64)[:3, :3]
    origin = np.asarray(c2w, dtype=np.float64)[:3, 3]
    local = (np.asarray(points, dtype=np.float64) - origin) @ rotation
    if convention == "opengl":
        depth = -local[:, 2]
        x = local[:, 0]
        y = -local[:, 1]
    elif convention == "opencv":
        depth = local[:, 2]
        x = local[:, 0]
        y = local[:, 1]
    else:
        raise ValueError(f"unknown camera convention {convention}")
    safe = np.abs(depth) >= 1e-8
    x = np.divide(x, depth, out=np.zeros_like(depth), where=safe)
    y = np.divide(y, depth, out=np.zeros_like(depth), where=safe)
    x, y = distort_opencv(x, y, *distortion)
    pixels = np.column_stack((fx * x + cx, fy * y + cy))
    return pixels, depth


def select_front(
    valid: np.ndarray,
    ui: np.ndarray,
    vi: np.ndarray,
    depth: np.ndarray,
    scales: np.ndarray,
    height: int,
    width: int,
    relative: float = 0.05,
) -> np.ndarray:
    """Keep samples that sit on the nearest surface at their pixel."""
    chosen = np.zeros(valid.shape[0], dtype=bool)
    if not np.any(valid):
        return chosen
    flat = vi[valid].astype(np.int64) * width + ui[valid].astype(np.int64)
    nearest_buffer = np.full(height * width, np.inf, dtype=np.float64)
    np.minimum.at(nearest_buffer, flat, depth[valid])
    nearest = nearest_buffer[flat]
    band = np.maximum(relative * nearest, 3.0 * scales[valid])
    keep = depth[valid] <= nearest + band
    chosen[np.flatnonzero(valid)[keep]] = True
    return chosen


def read_splat_columns(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz, opacity after sigmoid, and mean gaussian scale."""
    raw = path.read_bytes()
    header, _, payload = raw.partition(b"end_header\n")
    count = 0
    names: list[str] = []
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] != "float":
                raise ValueError(f"{path} is not a float splat table")
            names.append(parts[2])
    table = np.frombuffer(payload, dtype="<f4", count=count * len(names)).reshape(count, len(names))
    xyz = np.array(table[:, 0:3], dtype=np.float64, copy=True)
    opacity = 1.0 / (1.0 + np.exp(-table[:, names.index("opacity")].astype(np.float64)))
    scale_ids = [names.index(name) for name in ("scale_0", "scale_1", "scale_2")]
    scales = np.exp(table[:, scale_ids].astype(np.float64).mean(axis=1))
    return xyz, opacity, scales


def load_scene(root: Path) -> dict:
    """Camera poses, full-resolution intrinsics, and the orbit-1 splat."""
    base = root / "data" / "bottle" / "colmap" / "orbit1"
    transforms = json.loads((base / "transforms.json").read_text(encoding="utf-8"))
    dataparser = json.loads(
        (root / "outputs" / "orbit1" / "splatfacto" / "2026-10-09_234601" / "dataparser_transforms.json").read_text(
            encoding="utf-8"
        )
    )
    xyz, opacity, scales = read_splat_columns(base / "splat.ply")
    return {
        "transforms": transforms,
        "dataparser": dataparser,
        "xyz": xyz,
        "opacity": opacity,
        "scales": scales,
        "mask_dir": root / "data" / "bottle" / "masks",
    }


def scene_span(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    """Center and a floater-resistant radius of the Gaussian cloud."""
    center = np.median(xyz, axis=0)
    distance = np.linalg.norm(xyz - center, axis=1)
    radius = max(float(np.percentile(distance, 95)), 1e-8)
    return center, radius


def camera_distance_ok(origins: np.ndarray, xyz: np.ndarray) -> bool:
    center, radius = scene_span(xyz)
    distance = float(np.median(np.linalg.norm(origins - center, axis=1)))
    print(f"camera_distance {distance:.4f} cloud_radius {radius:.4f}", flush=True)
    return 0.2 * radius <= distance <= 30.0 * radius


def pose_for(frame: dict, dataparser: dict, space: str) -> np.ndarray:
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if space == "saved":
        return matrix
    if space == "trained":
        return training_c2w(matrix, np.asarray(dataparser["transform"], dtype=np.float64), float(dataparser["scale"]))
    raise ValueError(space)


def mask_path_for(frame: dict, mask_dir: Path) -> Path:
    return mask_dir / f"{Path(frame['file_path']).stem}_mask.png"


def view_hits(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    c2w: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: tuple[float, float, float, float],
    convention: str,
    opacity_floor: float = 0.05,
    measure_coverage: bool = False,
) -> tuple[np.ndarray, float]:
    """Return the inside-mask flags for one view and, when asked, mask-pixel coverage."""
    height, width = mask.shape[:2]
    pixels, depth = project_points(xyz, c2w, fx, fy, cx, cy, distortion, convention)
    pixels = np.nan_to_num(pixels, nan=-1.0, posinf=-1.0, neginf=-1.0)
    ui = np.rint(pixels[:, 0]).astype(np.int64)
    vi = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (
        (depth > 1e-4)
        & (opacity >= opacity_floor)
        & (ui >= 0)
        & (vi >= 0)
        & (ui < width)
        & (vi < height)
    )
    front = select_front(valid, ui, vi, depth, scales, height, width)
    inside = np.zeros(len(xyz), dtype=bool)
    if np.any(front):
        inside[front] = mask[vi[front], ui[front]] > 127
    if not measure_coverage:
        return inside, 0.0
    mask_pixels = int((mask > 127).sum())
    if mask_pixels == 0 or not np.any(inside):
        return inside, 0.0
    covered = np.unique(np.column_stack((ui[inside], vi[inside])), axis=0).shape[0]
    return inside, covered / mask_pixels


def choose_projection(scene: dict, sample: int = 40000) -> tuple[str, str, float]:
    """Pick the pose space and pixel convention that lands on the bottle masks."""
    transforms = scene["transforms"]
    frames = transforms["frames"]
    low = np.percentile(scene["xyz"], 5, axis=0)
    high = np.percentile(scene["xyz"], 95, axis=0)
    print(f"cloud_p05 {low.tolist()} cloud_p95 {high.tolist()}", flush=True)
    rng = np.random.default_rng(0)
    take = rng.choice(len(scene["xyz"]), size=min(sample, len(scene["xyz"])), replace=False)
    xyz = scene["xyz"][take]
    opacity = scene["opacity"][take]
    scales = scene["scales"][take]
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    distortion = (
        float(transforms.get("k1", 0.0)),
        float(transforms.get("k2", 0.0)),
        float(transforms.get("p1", 0.0)),
        float(transforms.get("p2", 0.0)),
    )
    probe_ids = [int(round(i * (len(frames) - 1) / 5)) for i in range(6)]
    import cv2

    best: tuple[str, str, float] | None = None
    for space in ("trained", "saved"):
        origins = np.stack([pose_for(frame, scene["dataparser"], space)[:3, 3] for frame in frames])
        near = camera_distance_ok(origins, xyz)
        print(f"space {space} cameras_near_cloud {near}", flush=True)
        for convention in ("opengl", "opencv"):
            coverages = []
            fractions = []
            for index in probe_ids:
                frame = frames[index]
                mask = cv2.imread(str(mask_path_for(frame, scene["mask_dir"])), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                inside, coverage = view_hits(
                    xyz,
                    opacity,
                    scales,
                    pose_for(frame, scene["dataparser"], space),
                    mask,
                    fx,
                    fy,
                    cx,
                    cy,
                    distortion,
                    convention,
                    measure_coverage=True,
                )
                coverages.append(coverage)
                fractions.append(float(inside.mean()))
            if not coverages:
                continue
            coverage = float(np.mean(coverages))
            fraction = float(np.mean(fractions))
            print(
                f"probe space {space} convention {convention} "
                f"mask_coverage {coverage:.4f} gaussian_fraction {fraction:.4f}",
                flush=True,
            )
            if fraction >= 0.85:
                continue
            if best is None or coverage > best[2]:
                best = (space, convention, coverage)
    if best is None or best[2] < 0.01:
        raise RuntimeError("projection did not land on the bottle masks")
    return best


def _view_hits_torch(
    xyz: "object",
    opacity: "object",
    scales: "object",
    c2w: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distortion: tuple[float, float, float, float],
    convention: str,
) -> "object":
    """GPU version of view_hits. Tensors stay on the caller's device."""
    import torch

    rotation = torch.as_tensor(c2w[:3, :3], device=xyz.device, dtype=xyz.dtype)
    origin = torch.as_tensor(c2w[:3, 3], device=xyz.device, dtype=xyz.dtype)
    local = (xyz - origin) @ rotation
    if convention == "opengl":
        depth = -local[:, 2]
        x = local[:, 0]
        y = -local[:, 1]
    else:
        depth = local[:, 2]
        x = local[:, 0]
        y = local[:, 1]
    safe = depth.abs() >= 1e-8
    x = torch.where(safe, x / depth, torch.zeros_like(x))
    y = torch.where(safe, y / depth, torch.zeros_like(y))
    k1, k2, p1, p2 = distortion
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2 * radius2
    x = x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x)
    y = y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y
    ui = torch.round(fx * x + cx).long()
    vi = torch.round(fy * y + cy).long()
    height, width = mask.shape[:2]
    valid = (depth > 1e-4) & (opacity >= 0.05) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
    inside = torch.zeros(xyz.shape[0], dtype=torch.bool, device=xyz.device)
    if not bool(valid.any()):
        return inside
    flat = vi[valid] * width + ui[valid]
    nearest_buffer = torch.full((height * width,), torch.inf, device=xyz.device, dtype=xyz.dtype)
    nearest_buffer.scatter_reduce_(0, flat, depth[valid], reduce="amin", include_self=True)
    nearest = nearest_buffer[flat]
    band = torch.maximum(0.05 * nearest, 3.0 * scales[valid])
    keep = depth[valid] <= nearest + band
    front = torch.where(valid)[0][keep]
    if front.numel() == 0:
        return inside
    mask_tensor = torch.as_tensor(mask, device=xyz.device)
    hit = mask_tensor[vi[front], ui[front]] > 127
    inside[front[hit]] = True
    return inside


def lift_orbit(root: Path) -> dict:
    """Write masks/gaussian_in_mask.npy and set projected on the bottle record."""
    import cv2
    import torch

    scene = load_scene(root)
    space, convention, coverage = choose_projection(scene)
    print(f"using space {space} convention {convention} probe_coverage {coverage:.4f}", flush=True)
    transforms = scene["transforms"]
    frames = transforms["frames"]
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    distortion = (
        float(transforms.get("k1", 0.0)),
        float(transforms.get("k2", 0.0)),
        float(transforms.get("p1", 0.0)),
        float(transforms.get("p2", 0.0)),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"project_device {device}", flush=True)
    xyz = torch.as_tensor(scene["xyz"], device=device)
    opacity = torch.as_tensor(scene["opacity"], device=device)
    scales = torch.as_tensor(scene["scales"], device=device)
    check_frame = frames[len(frames) // 2]
    check_mask = cv2.imread(str(mask_path_for(check_frame, scene["mask_dir"])), cv2.IMREAD_GRAYSCALE)
    check_pose = pose_for(check_frame, scene["dataparser"], space)
    sample = np.random.default_rng(1).choice(len(scene["xyz"]), size=2000, replace=False)
    numpy_hit, _ignored = view_hits(
        scene["xyz"][sample],
        scene["opacity"][sample],
        scene["scales"][sample],
        check_pose,
        check_mask,
        fx,
        fy,
        cx,
        cy,
        distortion,
        convention,
    )
    torch_hit = _view_hits_torch(
        xyz[sample],
        opacity[sample],
        scales[sample],
        check_pose,
        check_mask,
        fx,
        fy,
        cx,
        cy,
        distortion,
        convention,
    ).detach().cpu().numpy()
    if not np.array_equal(numpy_hit, torch_hit):
        raise RuntimeError("GPU projection does not match the tested projector")
    inside = torch.zeros(len(scene["xyz"]), dtype=torch.bool, device=device)
    used = 0
    for index, frame in enumerate(frames, start=1):
        mask_file = mask_path_for(frame, scene["mask_dir"])
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (int(transforms["h"]), int(transforms["w"])):
            raise RuntimeError(f"{mask_file.name} is {mask.shape}, expected the COLMAP image size")
        hit = _view_hits_torch(
            xyz,
            opacity,
            scales,
            pose_for(frame, scene["dataparser"], space),
            mask,
            fx,
            fy,
            cx,
            cy,
            distortion,
            convention,
        )
        inside |= hit
        used += 1
        if index % 20 == 0 or index == len(frames):
            print(f"projected {index}/{len(frames)} inside {int(inside.sum())}", flush=True)
    inside = inside.detach().cpu().numpy()
    if used < 200:
        raise RuntimeError(f"only {used} masks were projected")
    n_inside = int(inside.sum())
    if n_inside < 1000 or n_inside > 0.85 * len(inside):
        raise RuntimeError(f"inside count {n_inside} is not a bottle crop of {len(inside)} gaussians")
    mask_dir = scene["mask_dir"]
    crop_path = mask_dir / "gaussian_in_mask.npy"
    np.save(crop_path, inside)
    record_path = mask_dir / "object.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["projected"] = True
    record["n_views_projected"] = used
    record["n_gaussians"] = int(len(inside))
    record["n_inside"] = n_inside
    record["n_outside"] = int(len(inside) - n_inside)
    record["crop"] = "masks/gaussian_in_mask.npy"
    record["infill"] = "skipped"
    record["infill_reason"] = "no separate background fill; the room was not infilled"
    record["method"] = "project COLMAP poses onto orbit-1 gaussians"
    record["pose_space"] = space
    record["camera_convention"] = convention
    record["probe_mask_coverage"] = round(coverage, 6)
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"LIFT_OK inside {n_inside} outside {record['n_outside']} views {used}", flush=True)
    return record


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Lift bottle masks onto orbit-1 Gaussians")
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    args = parser.parse_args()
    try:
        lift_orbit(args.root)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
