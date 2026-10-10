#!/usr/bin/env python3
"""Lift bottle masks with the same gsplat view matrices the trained splat uses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.utils.eval_utils import eval_setup


ROOT = Path("/root/SceneagentX")
CONFIG = ROOT / "outputs/orbit1/splatfacto/2026-10-09_234601/config.yml"
MASK_DIR = ROOT / "data/bottle/masks"


def _bundles(pipeline):
    parser = pipeline.datamanager.dataparser
    found = []
    for split in ("train", "test"):
        outputs = parser.get_dataparser_outputs(split)
        if outputs.image_filenames:
            found.append(outputs)
    return found


def _visible(info) -> torch.Tensor:
    radii = info["radii"]
    if radii.ndim == 3:
        return radii[0].amax(dim=-1) > 0
    return radii[0] > 0


def _depths(means: torch.Tensor, viewmat: torch.Tensor) -> torch.Tensor:
    rotation = viewmat[0, :3, :3]
    translation = viewmat[0, :3, 3]
    return (means @ rotation.T)[:, 2] + translation[2]


def main() -> int:
    _config, pipeline, _ckpt, _step = eval_setup(CONFIG, test_mode="inference")
    model = pipeline.model
    means = model.means
    quats = model.quats / model.quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities).squeeze(-1)
    colors = torch.zeros((means.shape[0], 3), device=means.device, dtype=means.dtype)
    inside = torch.zeros(means.shape[0], dtype=torch.bool, device=means.device)
    anchor: torch.Tensor | None = None
    used = 0
    seen: set[str] = set()
    for outputs in _bundles(pipeline):
        cameras = outputs.cameras.to(means.device)
        for index, image_path in enumerate(outputs.image_filenames):
            stem = Path(image_path).stem
            if stem in seen:
                continue
            mask_path = MASK_DIR / f"{stem}_mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            seen.add(stem)
            camera = cameras[index : index + 1]
            viewmat = get_viewmat(camera.camera_to_worlds).to(means.device)
            intrinsics = camera.get_intrinsics_matrices().to(means.device)
            width, height = int(camera.width.item()), int(camera.height.item())
            if stem == "frame_00113":
                origin = camera.camera_to_worlds[0, :3, 3].detach().cpu().tolist()
                print(
                    f"anchor_camera fx {float(camera.fx[0]):.3f} fy {float(camera.fy[0]):.3f} origin {origin}",
                    flush=True,
                )
            if mask.shape != (height, width):
                raise RuntimeError(f"{mask_path.name} is {mask.shape}, camera is {(height, width)}")
            with torch.no_grad():
                _render, _alpha, info = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=viewmat,
                    Ks=intrinsics,
                    width=width,
                    height=height,
                    packed=False,
                    near_plane=0.01,
                    far_plane=1e10,
                    render_mode="RGB",
                    sh_degree=None,
                    rasterize_mode="classic",
                )
            if "depths" not in info:
                raise RuntimeError("gsplat did not return depths")
            means2d = info["means2d"][0]
            depth = info["depths"][0]
            visible = _visible(info) & (opacities >= 0.05) & torch.isfinite(depth) & torch.isfinite(means2d).all(dim=-1)
            ui = torch.round(means2d[:, 0]).long()
            vi = torch.round(means2d[:, 1]).long()
            valid = visible & (depth > 0.01) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
            if bool(valid.any()):
                flat = vi[valid] * width + ui[valid]
                nearest = torch.full((height * width,), torch.inf, device=means.device, dtype=depth.dtype)
                nearest.scatter_reduce_(0, flat, depth[valid], reduce="amin", include_self=True)
                keep = depth[valid] <= nearest[flat] + 1e-4
                chosen = torch.where(valid)[0][keep]
                mask_tensor = torch.as_tensor(mask, device=means.device)
                hit = mask_tensor[vi[chosen], ui[chosen]] > 127
                front = chosen[hit]
                inside[front] = True
                if stem == "frame_00113":
                    anchor = front.detach().clone()
                    print(f"anchor_front {int(anchor.numel())}", flush=True)
                if stem == "frame_00113":
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)
                    canvas[mask > 127] = (0, 140, 0)
                    front = chosen[hit]
                    pts = torch.stack((ui[front], vi[front]), dim=1).detach().cpu().numpy()
                    if len(pts):
                        canvas[pts[:, 1], pts[:, 0]] = (0, 255, 255)
                    cv2.imwrite("/tmp/lift_front.png", cv2.resize(canvas, (270, 480), interpolation=cv2.INTER_AREA))
            used += 1
            if used % 20 == 0:
                print(f"projected {used} inside {int(inside.sum())}", flush=True)
            del info
    print(f"projected {used} inside {int(inside.sum())}", flush=True)
    if used < 200:
        raise RuntimeError(f"only {used} masks were projected")
    if anchor is None or anchor.numel() < 100:
        raise RuntimeError("frame_00113 did not produce a bottle surface")
    center = means[anchor].median(dim=0).values
    anchor_dist = torch.linalg.norm(means[anchor] - center, dim=-1)
    median_dist = anchor_dist.median()
    mad = (anchor_dist - median_dist).abs().median().clamp(min=1e-3)
    radius = median_dist + 8.0 * mad
    print(
        f"anchor_median {float(median_dist):.4f} mad {float(mad):.4f} radius {float(radius):.4f}",
        flush=True,
    )
    near_object = torch.linalg.norm(means - center, dim=-1) <= radius
    inside &= near_object
    print(f"ball_radius {float(radius):.4f} inside_after_ball {int(inside.sum())}", flush=True)
    flags = inside.detach().cpu().numpy()
    xyz = means.detach().cpu().numpy()
    n_inside = int(flags.sum())
    if n_inside < 1000:
        raise RuntimeError(f"inside count {n_inside} is too small")

    def span(points: np.ndarray) -> float:
        low = np.percentile(points, 5, axis=0)
        high = np.percentile(points, 95, axis=0)
        return float((high - low).max())

    object_span = span(xyz[flags])
    scene_span = span(xyz)
    print(f"span_inside {object_span:.4f} span_scene {scene_span:.4f}", flush=True)
    if object_span > 0.35 * scene_span:
        raise RuntimeError("the mask crop covers the room, not the bottle")
    crop = MASK_DIR / "gaussian_in_mask.npy"
    np.save(crop, flags)
    record_path = MASK_DIR / "object.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.update(
        {
            "projected": True,
            "n_views_projected": used,
            "n_gaussians": int(len(flags)),
            "n_inside": n_inside,
            "n_outside": int(len(flags) - n_inside),
            "crop": "masks/gaussian_in_mask.npy",
            "infill": "skipped",
            "infill_reason": "no separate background fill; the room was not infilled",
            "method": "gsplat means projected with the trained splatfacto view matrices",
            "pose_space": "splatfacto_dataparser",
            "camera_convention": "gsplat",
            "span_inside": round(object_span, 6),
            "span_scene": round(scene_span, 6),
        }
    )
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"LIFT_OK inside {n_inside} outside {record['n_outside']} views {used}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2)
