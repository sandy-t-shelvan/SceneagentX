"""Plot COLMAP camera centers for one orbit.

images.txt is the COLMAP text model. The camera center is -R^T t.
A center farther than five times the median radius is counted as far,
which is the plan's 'no camera far from the rest' check.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse

import cv2
import numpy as np

FAR_FACTOR = 5.0


def _quat_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    rotation = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )
    return rotation


def camera_center(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    rotation = _quat_to_matrix(qw, qx, qy, qz)
    translation = np.array([tx, ty, tz], dtype=np.float64)
    return -rotation.T @ translation


def read_images_txt(path: Path) -> np.ndarray:
    centers: list[np.ndarray] = []
    pose_line = True
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if pose_line and len(parts) >= 8:
            qw, qx, qy, qz, tx, ty, tz = (float(part) for part in parts[1:8])
            centers.append(camera_center(qw, qx, qy, qz, tx, ty, tz))
            pose_line = False
        else:
            pose_line = True
    if not centers:
        raise ValueError(f"No camera poses in {path}")
    return np.vstack(centers)


def far_camera_count(centers: np.ndarray, factor: float = FAR_FACTOR) -> int:
    median = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median, axis=1)
    typical = float(np.median(distances))
    limit = max(typical * factor, 1e-6)
    return int(np.count_nonzero(distances > limit))


def plot_cameras(centers: np.ndarray, destination: Path) -> int:
    """Draw an XY plot. Returns how many centers sit far from the rest."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas = np.full((640, 640, 3), 255, dtype=np.uint8)
    span = centers[:, :2]
    low = span.min(axis=0)
    high = span.max(axis=0)
    extent = np.maximum(high - low, 1e-6)
    pixels = ((span - low) / extent * 560 + 40).astype(np.int32)
    for x_coord, y_coord in pixels:
        cv2.circle(canvas, (int(x_coord), int(y_coord)), 4, (20, 90, 180), -1)
    cv2.imwrite(str(destination), canvas)
    return far_camera_count(centers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot COLMAP camera centers")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        centers = read_images_txt(args.images)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    far = plot_cameras(centers, args.out)
    sys.stdout.write(f"cameras {len(centers)} far {far}\n")
    return 0 if far == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
