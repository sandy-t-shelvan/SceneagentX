"""Synthetic orbits for the B1 dry run and unit tests.

Frames are sharp, evenly lit, and carry one tag36h11 marker so the
frame checker can run without a camera capture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def april_tag_image(tag_id: int = 0, side: int = 280) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, tag_id, side)
    # OpenCV 4.6 and earlier.
    return cv2.aruco.drawMarker(dictionary, tag_id, side)


def make_frame(
    index: int,
    size: tuple[int, int] = (640, 480),
    tag_id: int = 0,
    blur: int = 0,
    seed: int = 0,
) -> np.ndarray:
    """Gray frame, high-frequency border, and a centered AprilTag."""
    width, height = size
    rng = np.random.default_rng(seed)
    image = np.full((height, width, 3), 176, dtype=np.uint8)
    border = 36
    noise = rng.integers(0, 255, size=(border, width, 3), dtype=np.uint8)
    image[:border, :, :] = noise
    image[-border:, :, :] = noise
    side_noise = rng.integers(0, 255, size=(height, border, 3), dtype=np.uint8)
    image[:, :border, :] = side_noise
    image[:, -border:, :] = side_noise

    tag = april_tag_image(tag_id, side=240)
    if tag.ndim == 2:
        tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)
    pad = 24
    y0 = (height - tag.shape[0]) // 2
    x0 = (width - tag.shape[1]) // 2
    image[y0 - pad : y0 + tag.shape[0] + pad, x0 - pad : x0 + tag.shape[1] + pad] = 255
    image[y0 : y0 + tag.shape[0], x0 : x0 + tag.shape[1]] = tag
    if blur > 0:
        kernel = blur * 2 + 1
        image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return image


def write_orbit(directory: Path, n: int = 8, blur: int = 0, tag_id: int = 0) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index in range(n):
        frame = make_frame(index, blur=blur, tag_id=tag_id, seed=0)
        path = directory / f"frame_{index + 1:06d}.png"
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Could not write {path}")
        paths.append(path)
    return paths


def write_video(path: Path, n: int = 8, fps: float = 4.0, blur: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = make_frame(0, blur=blur)
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open a video writer for {path}")
    try:
        for index in range(n):
            writer.write(make_frame(index, blur=blur))
    finally:
        writer.release()
    return path
