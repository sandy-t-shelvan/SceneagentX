#!/usr/bin/env python3
"""Block B B3: LangSplat feature field on a frozen Can splat.

Adds a 3-d feature per Gaussian per B2 level (whole/part/subpart), rasterises
those features with gsplat, and matches the B2 latent maps. Does not retrain
splatfacto. Does not rewrite mask-lift numbers.

Every orbit trains on its frozen splat. Orbit 1 writes the original semantics
paths. Orbit 2 and orbit 3 write beside it, under data/can/semantics/<orbit>/.
The mask lift npy is read and not rewritten here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from object_langsplat_b2 import (  # noqa: E402
    LEVEL_PART,
    LEVEL_SUBPART,
    LEVEL_WHOLE,
    MIN_PIXELS,
    PART_GRID,
    SUBPART_GRID,
    _crop,
    _grid_tiles,
    _load_rgb_mask,
    colmap_images,
    reject_object,
)

LATENT_DIM = 3
N_LEVELS = 3
FEAT_DIM = LATENT_DIM * N_LEVELS
ORBIT = "orbit1"
MAX_SIDE = 256
STEPS = 1000
LR = 0.025
HOLD_OUT = 8


def _find_lift_span(record):
    if isinstance(record, dict):
        if "covers_room" in record and "span_inside" in record:
            return record
        for value in record.values():
            found = _find_lift_span(value)
            if found is not None:
                return found
    elif isinstance(record, list):
        for value in record:
            found = _find_lift_span(value)
            if found is not None:
                return found
    return None


def _task_job(folder: Path) -> str | None:
    spec = _load(folder / "request.json")
    job = spec.get("job")
    if job in {"orbit", "room"}:
        return str(job)
    return None


def _feature_field_refusal(lift, job=None) -> str | None:
    """Half-scene refusal applies to a room job. An orbit job skips it."""
    if job == "orbit":
        return None
    if not isinstance(lift, dict):
        return None
    span_in = lift.get("span_inside")
    span_sc = lift.get("span_scene")
    ratio = None
    if _is_num(span_in) and _is_num(span_sc) and float(span_sc) > 0:
        ratio = float(span_in) / float(span_sc)
    if lift.get("covers_room") is not True and (ratio is None or ratio < 0.5):
        return None
    reason = (
        "mask lift covers the room "
        f"(span_inside {lift.get('span_inside')} span_scene {lift.get('span_scene')}); "
        "feature field was not trained"
    )
    # #region agent log
    try:
        line = json.dumps({
            "sessionId": "9f1777",
            "runId": "b3-room-guard",
            "hypothesisId": "covers-room",
            "location": "object_langsplat_b3.py:_feature_field_refusal",
            "message": "feature field refused because the lift covers the room",
            "data": {
                "span_inside": lift.get("span_inside"),
                "span_scene": lift.get("span_scene"),
                "n_inside": lift.get("n_inside"),
            },
            "timestamp": int(__import__("time").time() * 1000),
        })
        with (Path(__file__).resolve().parents[2] / "debug-9f1777.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except OSError:
        pass
    # #endregion
    return reason


def _dump(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _config_for(root: Path, obj: str, orbit: str) -> Path | None:
    block_a = _load(root / "data" / obj / "ingest" / "block_a.json")
    for row in block_a.get("orbits") or []:
        if isinstance(row, dict) and row.get("orbit") == orbit:
            raw = row.get("config")
            if raw:
                path = Path(str(raw))
                if path.is_file():
                    return path
    base = root / "outputs" / f"{obj}_{orbit}" / "splatfacto"
    if not base.is_dir():
        return None
    found = list(base.glob("*/config.yml"))
    if not found:
        return None
    return max(found, key=lambda path: path.stat().st_mtime)


def _is_num(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _preserve_lift(folder: Path) -> tuple[dict, dict, dict]:
    """Copy the current orbit1 projection. Do not keep a stale lift from an older run."""
    existing = _load(folder / "ingest" / "block_b_b3.json")
    orbits = existing.get("mask_lift_orbits") if isinstance(existing.get("mask_lift_orbits"), dict) else {}
    reviews = existing.get("reviews") if isinstance(existing.get("reviews"), dict) else {}
    obj = _load(folder / "masks" / ORBIT / "object.json")
    span_in = obj.get("span_inside")
    span_sc = obj.get("span_scene")
    judged = None
    if _is_num(span_in) and _is_num(span_sc) and float(span_sc) > 0:
        judged = "rejected" if float(span_in) / float(span_sc) >= 0.5 else "kept"
    lift = {
        "state": "measured" if obj.get("projected") is True else "unstarted",
        "orbit": ORBIT,
        "projected": obj.get("projected") is True,
        "n_inside": obj.get("n_inside"),
        "n_outside": obj.get("n_outside"),
        "n_gaussians": obj.get("n_gaussians"),
        "n_views_projected": obj.get("n_views_projected"),
        "n_views_skipped_empty": obj.get("n_views_skipped_empty"),
        "n_views_skipped_overseg": obj.get("n_views_skipped_overseg"),
        "covers_room": obj.get("covers_room"),
        "span_inside": span_in,
        "span_scene": span_sc,
        "judge_projection": judged,
        "infill": obj.get("infill") or "skipped",
        "source": f"data/{folder.name}/masks/{ORBIT}/object.json",
        "mask_npy_rewritten_by_feature_field": False,
    }
    if not reviews:
        reviews = {"asked": False, "reason": "VLM was not called"}
    return lift, orbits, reviews


def _field_layout(folder: Path, orbit: str) -> dict[str, Path | str | bool]:
    """Orbit 1 keeps the original semantics paths. Later orbits sit beside that record."""
    if orbit == ORBIT:
        semantics = folder / "semantics"
        return {
            "semantics": semantics,
            "features": semantics / "gaussian_features.npy",
            "index": semantics / "gaussian_lang_idx.npy",
            "sidecar": semantics / "feature_field.json",
            "features_rel": f"data/{folder.name}/semantics/gaussian_features.npy",
            "index_rel": f"data/{folder.name}/semantics/gaussian_lang_idx.npy",
            "write_shared_score": True,
        }
    semantics = folder / "semantics" / orbit
    return {
        "semantics": semantics,
        "features": semantics / "gaussian_features.npy",
        "index": semantics / "gaussian_lang_idx.npy",
        "sidecar": semantics / "feature_field.json",
        "features_rel": f"data/{folder.name}/semantics/{orbit}/gaussian_features.npy",
        "index_rel": f"data/{folder.name}/semantics/{orbit}/gaussian_lang_idx.npy",
        "write_shared_score": False,
    }


def _write_record(folder: Path, state: str, reason: str, extra: dict | None = None, orbit: str = ORBIT) -> None:
    layout = _field_layout(folder, orbit)
    numbers = extra.get("feature_field_numbers") if extra and isinstance(extra.get("feature_field_numbers"), dict) else {}
    sidecar = {
        "feature_field": state,
        "achieved": False,
        "orbit": orbit,
        "splat_retrained": False,
        "reason": reason,
        "file": layout["features_rel"] if state == "measured" else None,
    }
    sidecar.update(numbers)
    _dump(Path(layout["sidecar"]), sidecar)
    if not layout["write_shared_score"]:
        return
    lift, orbits, reviews = _preserve_lift(folder)
    record = {
        "block": "B",
        "sub_step": "B3",
        "object": folder.name,
        "does": "Train a LangSplat feature field on the frozen splat by rasterising low-d features and matching B2 maps. Mask lift stays the existing measurement.",
        "state": "partial" if state != "measured" else "measured",
        "achieved": False,
        "reason": reason,
        "feature_field": state,
        "splat_retrained": False,
        "lift_retrained": False,
        "mask_lift": lift,
        "reviews": reviews or {"asked": False, "reason": "VLM was not called"},
        "milestone": "none. Feature-field lift is recorded; mask lift is measured, not achieved.",
        "mask_lift_orbits": orbits,
    }
    if extra:
        record.update(extra)
    _dump(folder / "ingest" / "block_b_b3.json", record)


def _write_fail(folder: Path, reason: str, extra: dict | None = None, orbit: str = ORBIT) -> int:
    _write_record(folder, "fail", reason, extra, orbit=orbit)
    print(f"B3_FAIL {reason}", flush=True)
    return 1


def _consume_view(latents: np.ndarray, levels: np.ndarray, cursor: int, rgb: np.ndarray, fg: np.ndarray):
    whole = _crop(rgb, fg)
    if whole is None:
        return None, cursor
    if cursor >= len(latents) or int(levels[cursor]) != LEVEL_WHOLE:
        raise RuntimeError(f"B2 latent alignment lost at whole cursor={cursor}")
    whole_z = latents[cursor]
    cursor += 1
    parts: list[tuple[np.ndarray, np.ndarray]] = []
    for tile in _grid_tiles(fg, PART_GRID):
        if _crop(rgb, tile) is None:
            continue
        if cursor >= len(latents) or int(levels[cursor]) != LEVEL_PART:
            raise RuntimeError(f"B2 latent alignment lost at part cursor={cursor}")
        parts.append((tile.astype(bool), latents[cursor]))
        cursor += 1
    subs: list[tuple[np.ndarray, np.ndarray]] = []
    for tile in _grid_tiles(fg, SUBPART_GRID):
        if _crop(rgb, tile) is None:
            continue
        if cursor >= len(latents) or int(levels[cursor]) != LEVEL_SUBPART:
            raise RuntimeError(f"B2 latent alignment lost at subpart cursor={cursor}")
        subs.append((tile.astype(bool), latents[cursor]))
        cursor += 1
    return (whole_z, parts, subs, fg.astype(bool)), cursor


def _resize_bool(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray((mask.astype(np.uint8) * 255))
    return np.asarray(image.resize((width, height), Image.NEAREST)) > 127


def _resize_feat(feat: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.zeros((height, width, feat.shape[-1]), dtype=np.float32)
    for channel in range(feat.shape[-1]):
        plane = Image.fromarray(feat[:, :, channel].astype(np.float32), mode="F")
        out[:, :, channel] = np.asarray(plane.resize((width, height), Image.BILINEAR))
    return out


def _view_maps(
    whole_z: np.ndarray,
    parts,
    subs,
    fg_src: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    base_h, base_w = fg_src.shape
    whole_map = np.zeros((base_h, base_w, 3), dtype=np.float32)
    part_map = np.zeros((base_h, base_w, 3), dtype=np.float32)
    sub_map = np.zeros((base_h, base_w, 3), dtype=np.float32)
    fg = fg_src.astype(bool)
    whole_map[fg] = whole_z
    for tile, value in parts:
        if tile.shape != fg.shape:
            tile = _resize_bool(tile, base_w, base_h)
        part_map[tile] = value
    for tile, value in subs:
        if tile.shape != fg.shape:
            tile = _resize_bool(tile, base_w, base_h)
        sub_map[tile] = value
    stacked = np.concatenate([whole_map, part_map, sub_map], axis=-1)
    if (base_w, base_h) != (width, height):
        stacked = _resize_feat(stacked, width, height)
        fg = _resize_bool(fg, width, height)
    return stacked.astype(np.float32), fg


def _map_hw(width: int, height: int, max_side: int) -> tuple[int, int, float]:
    scale = min(1.0, float(max_side) / float(max(width, height)))
    mw = max(1, int(round(width * scale)))
    mh = max(1, int(round(height * scale)))
    return mw, mh, scale


def _camera_size(camera) -> tuple[int, int]:
    width = camera.width
    height = camera.height
    if torch.is_tensor(width):
        width = int(width.reshape(-1)[0])
        height = int(height.reshape(-1)[0])
    else:
        width = int(width)
        height = int(height)
    return width, height


def _scale_k(intrinsics: torch.Tensor, scale: float) -> torch.Tensor:
    scaled = intrinsics.clone()
    scaled[..., 0, 0] *= scale
    scaled[..., 1, 1] *= scale
    scaled[..., 0, 2] *= scale
    scaled[..., 1, 2] *= scale
    return scaled


def _l1_masked(pred: torch.Tensor, target: torch.Tensor, fg: torch.Tensor) -> torch.Tensor:
    if bool(fg.any()):
        loss_fg = (pred[fg] - target[fg]).abs().mean()
    else:
        loss_fg = pred.new_zeros(())
    bg = ~fg
    loss_bg = pred[bg].abs().mean() if bool(bg.any()) else pred.new_zeros(())
    return loss_fg + 0.05 * loss_bg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangSplat feature field on the frozen can splat")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    parser.add_argument("--orbit", default=ORBIT)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--max-side", type=int, default=MAX_SIDE)
    parser.add_argument("--lr", type=float, default=LR)
    args = parser.parse_args(argv)
    reject_object(args.obj)
    folder = args.root / "data" / args.obj
    if args.orbit not in ("orbit1", "orbit2", "orbit3"):
        print(f"B3_REFUSE unsupported orbit {args.orbit}", flush=True)
        return 2
    orbit = args.orbit
    job = _task_job(folder)
    mask_record_path = folder / "masks" / orbit / "object.json"
    if mask_record_path.is_file():
        try:
            lift = _find_lift_span(json.loads(mask_record_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return _write_fail(folder, f"could not read {mask_record_path}", orbit=orbit)
        refusal = _feature_field_refusal(lift, job=job)
        if refusal:
            refused = folder / "semantics" / orbit / "feature_field_refused.json"
            if orbit == ORBIT:
                refused = folder / "semantics" / "feature_field_refused.json"
            _dump(refused, {
                "feature_field": "refused",
                "achieved": False,
                "orbit": orbit,
                "reason": refusal,
                "prior_measurement_kept": True,
            })
            print(f"B3_REFUSE {refusal}", flush=True)
            return 2

    if not torch.cuda.is_available():
        return _write_fail(folder, "cuda is not available", orbit=orbit)

    npz = folder / "semantics" / "features.codebook.npz"
    if not npz.is_file():
        return _write_fail(folder, f"data/{args.obj}/semantics/features.codebook.npz missing; B2 maps are required", orbit=orbit)
    try:
        with np.load(npz) as book:
            names = set(book.files)
            if not {"codebook", "latents", "levels", "orbits"} <= names:
                return _write_fail(folder, f"features.codebook.npz is not a LangSplat codebook; keys={sorted(names)}", orbit=orbit)
            latents = book["latents"].astype(np.float32)
            levels = book["levels"]
            orbits = book["orbits"].astype(str)
            codebook = book["codebook"].astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        return _write_fail(folder, f"could not read B2 codebook: {exc}", orbit=orbit)
    if latents.ndim != 2 or latents.shape[1] != LATENT_DIM:
        return _write_fail(folder, f"B2 latents shape {latents.shape} is not (N, {LATENT_DIM})", orbit=orbit)

    pick = orbits == orbit
    latents = latents[pick]
    levels = levels[pick]
    if latents.shape[0] < MIN_PIXELS:
        return _write_fail(folder, f"too few {orbit} B2 latents: {latents.shape[0]}", orbit=orbit)
    if float(np.max(np.abs(latents))) == 0.0:
        refused = folder / "semantics" / orbit / "feature_field_refused.json"
        if orbit == ORBIT:
            refused = folder / "semantics" / "feature_field_refused.json"
        _dump(refused, {
            "feature_field": "refused",
            "achieved": False,
            "orbit": orbit,
            "reason": "B2 latents are all zero; feature field was not trained",
            "prior_measurement_kept": True,
        })
        print("B3_REFUSE B2 latents are all zero; feature field was not trained", flush=True)
        return 2

    config = _config_for(args.root, args.obj, orbit)
    if config is None or not config.is_file():
        return _write_fail(folder, f"frozen splat config missing for {orbit}", orbit=orbit)

    try:
        from gsplat.rendering import rasterization
        from nerfstudio.models.splatfacto import get_viewmat
        from nerfstudio.utils.eval_utils import eval_setup
    except Exception as exc:  # noqa: BLE001
        return _write_fail(folder, f"nerfstudio/gsplat import failed: {exc}", orbit=orbit)

    print(f"B3_LOAD_SPLAT {config}", flush=True)
    try:
        _cfg, pipeline, _ckpt, step = eval_setup(config, test_mode="inference")
        pipeline.eval()
        model = pipeline.model
        means = model.means.detach()
        quats = (model.quats / model.quats.norm(dim=-1, keepdim=True)).detach()
        scales = torch.exp(model.scales).detach()
        opacities = torch.sigmoid(model.opacities).squeeze(-1).detach()
        device = means.device
        n_gauss = int(means.shape[0])
    except Exception as exc:  # noqa: BLE001
        return _write_fail(folder, f"eval_setup failed: {exc}", orbit=orbit)
    print(f"B3_SPLAT n={n_gauss} ckpt_step={step} device={device}", flush=True)

    cameras_by_stem: dict[str, object] = {}
    from object_splat_iou import bundles

    for outputs in bundles(pipeline):
        cameras = outputs.cameras.to(device)
        for index, image_path in enumerate(outputs.image_filenames):
            stem = Path(image_path).stem
            cameras_by_stem.setdefault(stem, cameras[index : index + 1])
    print(f"B3_CAMERAS {len(cameras_by_stem)}", flush=True)

    views = colmap_images(folder / "colmap" / orbit)
    mask_dir = folder / "masks" / orbit
    cursor = 0
    packed: list[tuple[str, object, np.ndarray, np.ndarray]] = []
    consumed_views = 0
    try:
        for view in views:
            loaded = _load_rgb_mask(view, mask_dir / f"{view.stem}_mask.png")
            if loaded is None:
                continue
            rgb, fg = loaded
            consumed, cursor = _consume_view(latents, levels, cursor, rgb, fg)
            if consumed is None:
                continue
            consumed_views += 1
            camera = cameras_by_stem.get(view.stem)
            if camera is None:
                continue
            width, height = _camera_size(camera)
            mw, mh, _scale = _map_hw(width, height, args.max_side)
            whole_z, parts, subs, fg_src = consumed
            target, fg_map = _view_maps(whole_z, parts, subs, fg_src, mw, mh)
            packed.append((view.stem, camera, target, fg_map))
    except Exception as exc:  # noqa: BLE001
        return _write_fail(
            folder,
            f"B2 map rebuild failed: {exc}",
            {"n_consumed_views": consumed_views, "cursor": cursor},
            orbit=orbit,
        )
    if cursor != len(latents):
        return _write_fail(
            folder,
            f"B2 latent alignment leftover: cursor={cursor} n={len(latents)}",
            {"n_consumed_views": consumed_views},
            orbit=orbit,
        )
    if len(packed) < 8:
        return _write_fail(folder, f"only {len(packed)} {orbit} views have both B2 maps and splat cameras", orbit=orbit)
    print(f"B3_MAPS views={len(packed)} consumed={consumed_views} latents={len(latents)}", flush=True)

    init = torch.zeros((n_gauss, FEAT_DIM), device=device, dtype=means.dtype)
    mean_whole = torch.as_tensor(latents[levels == LEVEL_WHOLE].mean(axis=0), device=device, dtype=means.dtype)
    mean_part = torch.as_tensor(latents[levels == LEVEL_PART].mean(axis=0), device=device, dtype=means.dtype)
    mean_sub = torch.as_tensor(latents[levels == LEVEL_SUBPART].mean(axis=0), device=device, dtype=means.dtype)
    mask_path = folder / "masks" / orbit / "gaussian_in_mask.npy"
    n_inside_init = 0
    if mask_path.is_file():
        inside = np.load(mask_path)
        if inside.dtype != np.bool_ and inside.dtype != bool:
            inside = inside.astype(bool)
        if inside.shape[0] == n_gauss:
            idx = torch.as_tensor(inside, device=device)
            init[idx, 0:3] = mean_whole
            init[idx, 3:6] = mean_part
            init[idx, 6:9] = mean_sub
            n_inside_init = int(inside.sum())
            if n_inside_init:
                xyz = means.detach().float().cpu().numpy()
                low = np.percentile(xyz, 5, axis=0)
                high = np.percentile(xyz, 95, axis=0)
                scene_span = float((high - low).max())
                obj_xyz = xyz[np.asarray(inside).reshape(-1)]
                low = np.percentile(obj_xyz, 5, axis=0)
                high = np.percentile(obj_xyz, 95, axis=0)
                object_span = float((high - low).max())
                if job != "orbit" and scene_span > 0 and object_span >= 0.5 * scene_span:
                    refusal = _feature_field_refusal({
                        "covers_room": True,
                        "span_inside": object_span,
                        "span_scene": scene_span,
                        "n_inside": n_inside_init,
                    })
                    refused = folder / "semantics" / orbit / "feature_field_refused.json"
                    if orbit == ORBIT:
                        refused = folder / "semantics" / "feature_field_refused.json"
                    _dump(refused, {
                        "feature_field": "refused",
                        "achieved": False,
                        "orbit": orbit,
                        "reason": refusal,
                        "prior_measurement_kept": True,
                    })
                    print(f"B3_REFUSE {refusal}", flush=True)
                    return 2
    features = torch.nn.Parameter(init)
    opt = torch.optim.Adam([features], lr=args.lr)

    hold = HOLD_OUT if len(packed) >= 16 else 0
    train_views = packed[:-hold] if hold else packed
    eval_views = packed[-hold:] if hold else []
    print(f"B3_SPLIT train={len(train_views)} eval={len(eval_views)} steps={args.steps}", flush=True)

    def _raster(camera, feat: torch.Tensor):
        width, height = _camera_size(camera)
        mw, mh, scale = _map_hw(width, height, args.max_side)
        viewmat = get_viewmat(camera.camera_to_worlds).to(device)
        intrinsics = _scale_k(camera.get_intrinsics_matrices().to(device), scale)
        render, _alpha, _info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=feat,
            viewmats=viewmat,
            Ks=intrinsics,
            width=mw,
            height=mh,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode="RGB",
            sh_degree=None,
            rasterize_mode="classic",
        )
        pred = render[0]
        if pred.shape[-1] != FEAT_DIM:
            raise RuntimeError(f"gsplat feature raster returned {tuple(pred.shape)}, expected *x{FEAT_DIM}")
        return pred

    try:
        stem0, cam0, target0, fg0 = train_views[0]
        pred0 = _raster(cam0, features)
        print(f"B3_RASTER_OK {stem0} {tuple(pred0.shape)}", flush=True)
        del pred0
    except Exception as exc:  # noqa: BLE001
        return _write_fail(folder, f"gsplat feature rasterisation failed: {exc}", orbit=orbit)

    last = 0.0
    rng = np.random.default_rng(0)
    for step_i in range(args.steps):
        pick_i = int(rng.integers(0, len(train_views)))
        _stem, camera, target_np, fg_np = train_views[pick_i]
        target = torch.as_tensor(target_np, device=device, dtype=features.dtype)
        fg = torch.as_tensor(fg_np, device=device)
        pred = _raster(camera, features)
        loss = _l1_masked(pred, target, fg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach().item())
        if step_i == 0 or (step_i + 1) % 100 == 0 or step_i + 1 == args.steps:
            print(f"B3_TRAIN {step_i + 1}/{args.steps} loss {last:.6f}", flush=True)
        if (step_i + 1) % 200 == 0:
            torch.cuda.empty_cache()

    eval_loss = None
    if eval_views:
        losses = []
        with torch.no_grad():
            for _stem, camera, target_np, fg_np in eval_views:
                target = torch.as_tensor(target_np, device=device, dtype=features.dtype)
                fg = torch.as_tensor(fg_np, device=device)
                pred = _raster(camera, features)
                losses.append(float(_l1_masked(pred, target, fg).item()))
        eval_loss = float(np.mean(losses)) if losses else None
        print(f"B3_EVAL l1={eval_loss}", flush=True)

    feat_np = features.detach().float().cpu().numpy().reshape(n_gauss, N_LEVELS, LATENT_DIM)
    layout = _field_layout(folder, orbit)
    semantics = Path(layout["semantics"])
    semantics.mkdir(parents=True, exist_ok=True)
    feat_path = Path(layout["features"])
    idx_path = Path(layout["index"])
    shared_features = folder / "semantics" / "gaussian_features.npy"
    shared_index = folder / "semantics" / "gaussian_lang_idx.npy"
    shared_sidecar = folder / "semantics" / "feature_field.json"
    shared = {shared_features.resolve(), shared_index.resolve(), shared_sidecar.resolve()}
    if orbit != ORBIT and feat_path.resolve() in shared:
        return _write_fail(folder, f"{orbit} refused to write over the orbit1 feature field", orbit=orbit)
    np.save(feat_path, feat_np.astype(np.float32))
    whole_feat = feat_np[:, 0, :]
    dist = ((whole_feat[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=2)
    assign = dist.argmin(axis=1).astype(np.int32)
    np.save(idx_path, assign)

    numbers = {
        "orbit": orbit,
        "n_gaussians": n_gauss,
        "n_views": len(packed),
        "n_train": len(train_views),
        "n_eval": len(eval_views),
        "n_b2_latents": int(latents.shape[0]),
        "n_inside_init": n_inside_init,
        "latent_dim": LATENT_DIM,
        "n_levels": N_LEVELS,
        "feature_dim": FEAT_DIM,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "max_side": int(args.max_side),
        "train_l1": last,
        "eval_l1": eval_loss,
        "config": str(config),
        "splat_step": int(step) if step is not None else None,
        "file": str(layout["features_rel"]),
        "index_file": str(layout["index_rel"]),
        "b2_maps": f"data/{folder.name}/semantics/features.codebook.npz latents aligned to B1 masks",
        "physics_vector_untouched": True,
        "mask_npy_rewritten": False,
    }
    reason = (
        f"LangSplat feature field trained on frozen {orbit} splat "
        f"({n_gauss} Gaussians, {len(train_views)} train views, train L1 {last:.4f}"
        + (f", eval L1 {eval_loss:.4f}" if eval_loss is not None else ", eval unstarted")
        + "); mask lift kept; achieved stays false"
    )
    _write_record(
        folder,
        "measured",
        reason,
        {"feature_field_numbers": numbers},
        orbit=orbit,
    )
    print("B3_FEATURE_FIELD_OK", json.dumps(numbers), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
