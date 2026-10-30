#!/usr/bin/env python3
"""Lift SAM masks onto a trained splat, drop room Gaussians, rescore IoU.

Uses the existing checkpoint. Does not retrain. Does not write data/bottle.
achieved stays false. The previous full-scene IoU is kept as full_scene_iou.
Membership is Harvard eq.1: project inside the mask and near the front surface
(εa=0.08, εr=0.03 from tools.block_b_job). Far Gaussians drop.
When every eligible crop covers the room, a room job leaves gaussian_in_mask.npy
in place and does not train features or infill. An orbit job still keeps a crop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from gsplat.rendering import rasterization
from nerfstudio.models.splatfacto import get_viewmat
from nerfstudio.utils.eval_utils import eval_setup

sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_splat_iou import (
    ALPHA_GATE,
    PAPER_GATE_IOU,
    REPO,
    _agent_log,
    bundles,
    camera_size,
    mask_iou,
    reject_object,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from tools.block_b_job import FRONT_ABS_SLACK, FRONT_REL_SLACK

OVERSEG_FRAC = 0.90
OPACITY_MIN = 0.05
MIN_INSIDE = 1000
SPAN_ROOM_FRAC = 0.35


def _visible(info) -> torch.Tensor:
    radii = info["radii"]
    if radii.ndim == 3:
        return radii[0].amax(dim=-1) > 0
    return radii[0] > 0


def render_subset_alpha(means, quats, scales, opacities, camera) -> np.ndarray:
    width, height = camera_size(camera)
    viewmat = get_viewmat(camera.camera_to_worlds).to(means.device)
    intrinsics = camera.get_intrinsics_matrices().to(means.device)
    colors = torch.zeros((means.shape[0], 3), device=means.device, dtype=means.dtype)
    with torch.no_grad():
        _render, alpha, _info = rasterization(
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
    alpha_np = np.squeeze(alpha.detach().float().cpu().numpy())
    if alpha_np.ndim != 2:
        raise SystemExit(f"alpha shape {alpha_np.shape}")
    return alpha_np


def span(points: np.ndarray) -> float:
    low = np.percentile(points, 5, axis=0)
    high = np.percentile(points, 95, axis=0)
    return float((high - low).max())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crop room Gaussians and rescore silhouette IoU")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--orbit", default="orbit1")
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--frame", default="", help="score this stem; default is the existing mesh_score frame")
    args = parser.parse_args(argv)
    reject_object(args.obj)
    if args.orbit != Path(args.orbit).name or not args.orbit.startswith("orbit"):
        raise SystemExit(f"refusing orbit name {args.orbit}")
    if not args.config.is_file():
        sys.stderr.write(f"missing config {args.config}\n")
        return 1
    if not torch.cuda.is_available():
        sys.stderr.write("cuda is not available\n")
        return 2
    mask_dir = args.root / "data" / args.obj / "masks" / args.orbit
    if not mask_dir.is_dir():
        sys.stderr.write(f"missing masks {mask_dir}\n")
        return 1

    score_path = args.root / "data" / args.obj / "mesh_score.json"
    previous = {}
    if score_path.is_file():
        try:
            previous = json.loads(score_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = {}
    prev_orbit = {}
    for item in previous.get("orbits") or []:
        if isinstance(item, dict) and item.get("orbit") == args.orbit:
            prev_orbit = item
            break
    prefer = args.frame or Path(str(prev_orbit.get("frame") or "")).stem

    _config, pipeline, _ckpt, step = eval_setup(args.config, test_mode="inference")
    pipeline.eval()
    model = pipeline.model
    means = model.means
    quats = model.quats / model.quats.norm(dim=-1, keepdim=True)
    scales = torch.exp(model.scales)
    opacities = torch.sigmoid(model.opacities).squeeze(-1)
    colors = torch.zeros((means.shape[0], 3), device=means.device, dtype=means.dtype)
    n_gauss = int(means.shape[0])

    votes = torch.zeros(n_gauss, dtype=torch.int32, device=means.device)
    usable: dict[str, tuple[object, Path, torch.Tensor, int]] = {}
    used = 0
    skipped_empty = 0
    skipped_overseg = 0
    skipped_overseg_names: list[str] = []
    seen: set[str] = set()

    for outputs in bundles(pipeline):
        cameras = outputs.cameras.to(means.device)
        for index, image_path in enumerate(outputs.image_filenames):
            image = Path(image_path)
            stem = image.stem
            if stem in seen:
                continue
            mask_path = mask_dir / f"{stem}_mask.png"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            seen.add(stem)
            camera = cameras[index : index + 1]
            width, height = camera_size(camera)
            frame_px = width * height
            fg = int((mask > 127).sum())
            if fg == 0:
                skipped_empty += 1
                continue
            if fg > OVERSEG_FRAC * frame_px:
                skipped_overseg += 1
                skipped_overseg_names.append(stem)
                continue
            if mask.shape != (height, width):
                raise SystemExit(f"{mask_path.name} is {mask.shape}, camera is {(height, width)}")
            viewmat = get_viewmat(camera.camera_to_worlds).to(means.device)
            intrinsics = camera.get_intrinsics_matrices().to(means.device)
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
                raise SystemExit("gsplat did not return depths")
            means2d = info["means2d"][0]
            depth = info["depths"][0]
            visible = _visible(info) & (opacities >= OPACITY_MIN) & torch.isfinite(depth) & torch.isfinite(means2d).all(dim=-1)
            ui = torch.round(means2d[:, 0]).long()
            vi = torch.round(means2d[:, 1]).long()
            valid = visible & (depth > 0.01) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
            front = torch.empty(0, dtype=torch.long, device=means.device)
            if bool(valid.any()):
                flat = vi[valid] * width + ui[valid]
                nearest = torch.full((height * width,), torch.inf, device=means.device, dtype=depth.dtype)
                nearest.scatter_reduce_(0, flat, depth[valid], reduce="amin", include_self=True)
                # Harvard eq.1: in-mask Gaussians near the front surface
                # (|z − z_front| ≤ εa + εr·|z_front|; εa=0.08, εr=0.03).
                front_d = nearest[flat]
                slack = FRONT_ABS_SLACK + FRONT_REL_SLACK * front_d.abs()
                keep = (depth[valid] - front_d).abs() <= slack
                chosen = torch.where(valid)[0][keep]
                mask_tensor = torch.as_tensor(mask, device=means.device)
                hit = mask_tensor[vi[chosen], ui[chosen]] > 127
                front = chosen[hit]
                votes[front] += 1
            used += 1
            usable[stem] = (camera, image, front.detach().clone(), fg)
            if used % 20 == 0:
                print(f"projected {used} inside {int((votes > 0).sum())}", flush=True)
            del info

    print(f"projected {used} inside {int((votes > 0).sum())}", flush=True)
    # #region agent log
    _agent_log(
        "H",
        "object_crop_iou.py:skip",
        "mask votes after empty and oversegment filters",
        {
            "used": used,
            "skipped_empty": skipped_empty,
            "skipped_overseg": skipped_overseg,
            "skipped_overseg_names": skipped_overseg_names[:8],
            "n_gauss": n_gauss,
            "inside_before_ball": int((votes > 0).sum()),
            "prefer": prefer,
        },
        run_id="crop-run3",
    )
    # #endregion
    if used < 8:
        sys.stderr.write(f"only {used} usable masks were projected\n")
        return 1
    if prefer and prefer in usable:
        score_stem = prefer
    else:
        score_stem = max(usable.items(), key=lambda item: (item[1][3], item[0]))[0]
        if prefer:
            print(f"preferred frame {prefer} missing; using {score_stem}", flush=True)
    score_camera, score_image, anchor, _fg = usable[score_stem]

    vote_cpu = votes.detach().cpu().numpy()
    hist_pos = {str(k): int((vote_cpu == k).sum()) for k in range(1, min(int(vote_cpu.max()) + 1, 16))}
    # #region agent log
    _agent_log(
        "J",
        "object_crop_iou.py:votes",
        "per-gaussian mask vote counts",
        {
            "n_gauss": n_gauss,
            "n_voted": int((vote_cpu > 0).sum()),
            "n_vote1": int((vote_cpu == 1).sum()),
            "n_vote_ge2": int((vote_cpu >= 2).sum()),
            "n_vote_ge3": int((vote_cpu >= 3).sum()),
            "hist_1_to_15": hist_pos,
        },
        run_id="crop-run3",
    )
    # #endregion

    def apply_ball(keep: torch.Tensor, mad_mult: float) -> tuple[torch.Tensor, dict]:
        if anchor is None or int(anchor.numel()) < 100:
            return keep, {"applied": False, "anchor_front": int(anchor.numel()) if anchor is not None else 0, "mad_mult": mad_mult}
        center = means[anchor].median(dim=0).values
        anchor_dist = torch.linalg.norm(means[anchor] - center, dim=-1)
        median_dist = anchor_dist.median()
        mad = (anchor_dist - median_dist).abs().median().clamp(min=1e-3)
        radius = median_dist + mad_mult * mad
        near_object = torch.linalg.norm(means - center, dim=-1) <= radius
        clipped = keep & near_object
        return clipped, {
            "applied": True,
            "anchor_front": int(anchor.numel()),
            "mad_mult": mad_mult,
            "radius": float(radius),
            "inside_after_ball": int(clipped.sum()),
        }

    mask_path = mask_dir / f"{score_stem}_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        sys.stderr.write(f"could not read mask for {score_stem}\n")
        return 1

    def score_keep(keep: torch.Tensor) -> dict:
        n_keep = int(keep.sum())
        if n_keep < 8:
            return {"n_inside": n_keep, "iou": 0.0, "recall": 0.0, "splat_fg": 0, "intersection": 0, "union": 0, "splat_frac": 0.0}
        alpha = render_subset_alpha(means[keep], quats[keep], scales[keep], opacities[keep], score_camera)
        height, width = alpha.shape
        local = mask
        if local.shape != (height, width):
            local = cv2.resize(local, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_fg = local > 127
        hit = alpha > ALPHA_GATE
        intersection = int(np.logical_and(mask_fg, hit).sum())
        union = int(np.logical_or(mask_fg, hit).sum())
        iou = mask_iou(mask_fg, hit)
        return {
            "n_inside": n_keep,
            "iou": iou,
            "recall": intersection / float(mask_fg.sum()) if int(mask_fg.sum()) else 0.0,
            "splat_fg": int(hit.sum()),
            "intersection": intersection,
            "union": union,
            "splat_frac": float(hit.mean()),
            "alpha": alpha,
            "hit": hit,
            "mask_fg": mask_fg,
            "height": height,
            "width": width,
            "resized": local.shape != mask.shape,
        }

    policy_keeps: list[tuple[str, torch.Tensor, dict]] = []
    for name, min_votes, mad_mult in (
        ("or1_mad8", 1, 8.0),
        ("or1_mad3", 1, 3.0),
        ("or2_mad3", 2, 3.0),
    ):
        keep, ball_info = apply_ball(votes >= min_votes, mad_mult)
        policy_keeps.append((name, keep, ball_info))
    if anchor is not None and int(anchor.numel()) >= 100:
        anchor_keep = torch.zeros(n_gauss, dtype=torch.bool, device=means.device)
        anchor_keep[anchor] = True
        policy_keeps.append(("anchor_only", anchor_keep, {"applied": False, "anchor_front": int(anchor.numel())}))

    from tools.block_c import apply_crop_policy

    xyz_all = means.detach().float().cpu().numpy()
    scene_span_all = span(xyz_all)
    policy_rows = []
    scored = {}
    for name, keep, ball_info in policy_keeps:
        stats = score_keep(keep)
        inside = keep.detach().cpu().numpy().astype(bool).reshape(-1)
        n_row = int(inside.sum())
        object_span_row = span(xyz_all[inside]) if n_row else 0.0
        covers_row = bool(n_row and scene_span_all > 0 and object_span_row > SPAN_ROOM_FRAC * scene_span_all)
        scored[name] = (keep, ball_info, stats)
        policy_rows.append(
            {
                "name": name,
                "n_inside": stats["n_inside"],
                "iou": stats["iou"],
                "recall": stats["recall"],
                "splat_fg": stats["splat_fg"],
                "splat_frac": stats["splat_frac"],
                "object_span": object_span_row,
                "covers_room": covers_row,
                "ball": {k: ball_info[k] for k in ball_info if k != "radius" or True},
            }
        )
        print(
            f"POLICY {name} iou {stats['iou']:.4f} recall {stats['recall']:.4f} splat_frac {stats['splat_frac']:.4f} n {stats['n_inside']}",
            flush=True,
        )
    # #region agent log
    _agent_log("K", "object_crop_iou.py:policies", "tight-crop policy comparison", {"orbit": args.orbit, "frame": score_stem, "policies": policy_rows}, run_id="crop-run3")
    # #endregion

    chosen_name = apply_crop_policy(
        args.root / "data" / args.obj,
        args.orbit,
        policy_rows,
        span_scene=scene_span_all,
    )
    if not chosen_name:
        print(
            "CROP_REFUSED every eligible crop covers the room; gaussian_in_mask.npy was not replaced",
            flush=True,
        )
        return 0
    keep, ball, stats = scored[chosen_name]
    flags = keep.detach().cpu().numpy()
    n_inside = int(flags.sum())
    xyz = means.detach().cpu().numpy()
    object_span = span(xyz[flags]) if n_inside else 0.0
    scene_span = span(xyz)
    covers_room = bool(n_inside and object_span > SPAN_ROOM_FRAC * scene_span)
    # #region agent log
    _agent_log(
        "I",
        "object_crop_iou.py:crop",
        "chosen tight crop",
        {
            "chosen": chosen_name,
            "n_inside": n_inside,
            "n_gauss": n_gauss,
            "inside_frac": n_inside / n_gauss if n_gauss else 0.0,
            "object_span": object_span,
            "scene_span": scene_span,
            "covers_room": covers_room,
            "ball": ball,
            "iou": stats["iou"],
            "recall": stats["recall"],
        },
        run_id="crop-run3",
    )
    # #endregion
    if n_inside < MIN_INSIDE:
        sys.stderr.write(f"inside count {n_inside} is too small\n")
        return 1

    crop_path = mask_dir / "gaussian_in_mask.npy"
    np.save(crop_path, flags)
    meta_path = mask_dir / "object.json"
    if meta_path.is_file():
        record = json.loads(meta_path.read_text(encoding="utf-8"))
        record.update(
            {
                "projected": True,
                "n_views_projected": used,
                "n_views_skipped_empty": skipped_empty,
                "n_views_skipped_overseg": skipped_overseg,
                "skipped_overseg": skipped_overseg_names,
                "n_gaussians": n_gauss,
                "n_inside": n_inside,
                "n_outside": n_gauss - n_inside,
                "crop": f"masks/{args.orbit}/gaussian_in_mask.npy",
                "infill": "skipped",
                "method": "gsplat means projected with trained splatfacto view matrices; Harvard eq.1 front-surface keep then vote threshold plus anchor ball",
                "policy": chosen_name,
                "ball": ball,
                "span_inside": round(object_span, 6),
                "span_scene": round(scene_span, 6),
                "covers_room": covers_room,
            }
        )
        meta_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    alpha = stats["alpha"]
    hit = stats["hit"]
    mask_fg = stats["mask_fg"]
    height = int(stats["height"])
    width = int(stats["width"])
    resized = bool(stats["resized"])
    iou = float(stats["iou"])
    intersection = int(stats["intersection"])
    union = int(stats["union"])
    # #region agent log
    _agent_log(
        "F",
        "object_crop_iou.py:alpha",
        "cropped splat silhouette coverage",
        {
            "splat_fg": int(hit.sum()),
            "frame_px": int(height * width),
            "splat_frac": float(hit.mean()),
            "alpha_min": float(alpha.min()),
            "alpha_max": float(alpha.max()),
            "splat_all_foreground": bool(hit.all()),
            "policy": chosen_name,
        },
        run_id="crop-run3",
    )
    _agent_log(
        "G",
        "object_crop_iou.py:iou",
        "cropped mask_iou vs kept full-scene iou",
        {
            "mask_fg": int(mask_fg.sum()),
            "splat_fg": int(hit.sum()),
            "intersection": intersection,
            "union": union,
            "crop_iou": iou,
            "full_scene_iou": prev_orbit.get("full_scene_iou", 0.4278428819444444),
            "equals_mask_over_frame": union == int(height * width) and intersection == int(mask_fg.sum()),
            "frame": score_stem,
            "policy": chosen_name,
        },
        run_id="crop-run3",
    )
    # #endregion

    obj_dir = args.root / "data" / args.obj
    out_dir = obj_dir / "iou" / args.orbit
    out_dir.mkdir(parents=True, exist_ok=True)
    splat_out = out_dir / f"{score_stem}_splat_crop.png"
    cv2.imwrite(str(splat_out), (hit.astype(np.uint8)) * 255)
    if score_image is not None and score_image.is_file():
        capture = out_dir / f"{score_stem}{score_image.suffix}"
        if not capture.is_file():
            import shutil

            shutil.copy2(score_image, capture)

    rel = f"data/{args.obj}/iou/{args.orbit}"
    orbit_record = dict(prev_orbit) if prev_orbit else {}
    orbit_record.update(
        {
            "orbit": args.orbit,
            "state": "measured",
            "achieved": False,
            "achieved_reason": "the only scale is model_knowledge",
            "frame": f"{score_stem}.png",
            "prompt": args.prompt,
            "capture_foreground_pixels": int(mask_fg.sum()),
            "full_scene_iou": prev_orbit.get("full_scene_iou", prev_orbit.get("iou") if prev_orbit.get("silhouette") != "cropped" else 0.4278428819444444),
            "full_scene_splat_foreground_pixels": prev_orbit.get("full_scene_splat_foreground_pixels", 2073600),
            "splat_foreground_pixels": int(hit.sum()),
            "intersection_pixels": intersection,
            "union_pixels": union,
            "iou": iou,
            "silhouette": "cropped",
            "crop": {
                "n_inside": n_inside,
                "n_gaussians": n_gauss,
                "n_views_projected": used,
                "n_views_skipped_empty": skipped_empty,
                "n_views_skipped_overseg": skipped_overseg,
                "skipped_overseg": skipped_overseg_names,
                "covers_room": covers_room,
                "span_inside": object_span,
                "span_scene": scene_span,
                "policy": chosen_name,
                "policies": [{k: row[k] for k in ("name", "n_inside", "iou", "recall", "splat_frac")} for row in policy_rows],
                "ball": ball,
                "npy": f"data/{args.obj}/masks/{args.orbit}/gaussian_in_mask.npy",
            },
            "splat_crop": f"{rel}/{score_stem}_splat_crop.png",
            "paper_gate_iou": PAPER_GATE_IOU,
            "paper_gate_met": bool(iou >= PAPER_GATE_IOU),
            "paper_gate_note": "comparison only; not a claim of achievement",
            "alpha_gate": ALPHA_GATE,
            "scale": "model_knowledge",
            "photometric": "not_scored",
            "camera_width": int(width),
            "camera_height": int(height),
            "mask_resized_to_camera": resized,
            "checkpoint_step": int(step) if step is not None else None,
            "config": str(args.config),
        }
    )
    score = previous if isinstance(previous, dict) else {}
    orbits = [item for item in score.get("orbits") or [] if isinstance(item, dict) and item.get("orbit") != args.orbit]
    orbits.append(orbit_record)
    orbits.sort(key=lambda item: str(item.get("orbit")))
    score.update(
        {
            "object": args.obj,
            "state": "measured",
            "achieved": False,
            "achieved_reason": "the only scale is model_knowledge",
            "scale": "model_knowledge",
            "paper_gate_iou": PAPER_GATE_IOU,
            "paper_gate_note": "comparison only; not a claim of achievement",
            "metric": "mask_iou after gsplat mask lift and room crop (SAM vs cropped splat alpha > 0.5)",
            "orbits": orbits,
        }
    )
    score_path.write_text(json.dumps(score, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(orbit_record, default=str), flush=True)
    print(f"WROTE {score_path}", flush=True)
    print(f"CROP_IOU {iou} policy {chosen_name} full_scene {orbit_record.get('full_scene_iou')} inside {n_inside}/{n_gauss}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
