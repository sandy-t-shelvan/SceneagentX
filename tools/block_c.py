"""Block C crop and infill for any object under data/<obj>/.

C1 reads the Block B mask, object.json, and instances.json. A room-covering
mask is not written as a crop: the first room-sized result and a retry that
is still room-sized both stay refused. An orbit job skips that half-scene
test and keeps infill skipped, and that skip is not a pass. A room capture
with an object-sized mask writes the object and background plies. Infill is
scheduled only when the spec says the room is delivered with the object
removed; a missing spec keeps the previous requirement. Infill stays not_run
until a real LaMa image and a real fine-tune ply exist. M3 stays false until
the seam review has no seams, PSNR outside the hole drops by less than 1.5 dB,
and the floater count is under 50.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.block_b_job import FRAME_COVER
from tools.common.paths import object_dir, resolve_root
from tools.design_workspace import _read_splat_table, crop_splat
from tools.orbit_blocks import count_floaters

OPACITY_MIN = 0.05
FLOATER_MAX = 50
PSNR_DROP_MAX = 1.5
WIDEN_RADIUS = 8
ROOM_SPAN = 0.5
SEAM_QUESTION = "Do the three novel views show seams, floating fragments, or a missing table surface?"
ALLOWED_SEAM_CORRECTION = ("wider_hole_mask",)
_ORBIT = re.compile(r"orbit[1-9][0-9]*\Z")


def require_object_name(obj: str) -> str:
    """Refuse a path. A single folder name is the only accepted object."""
    name = str(obj)
    if name in {"", ".", ".."} or name != Path(name).name or "/" in name or "\\" in name:
        raise ValueError(f"{obj!r} is not a single folder name")
    return name


def require_orbit(orbit: str | int) -> str:
    """Accept orbit1 or the integer 1. Refuse a path."""
    if isinstance(orbit, bool):
        raise ValueError(f"{orbit!r} is not an orbit name")
    if isinstance(orbit, int):
        if orbit < 1:
            raise ValueError(f"{orbit!r} is not an orbit name")
        return f"orbit{orbit}"
    text = str(orbit).strip()
    if not _ORBIT.fullmatch(text):
        raise ValueError(f"{orbit!r} is not an orbit name")
    return text


def mask_file(folder: Path, orbit: str) -> Path:
    """The gaussian mask for one orbit.

    A per-orbit file wins. Orbit 1 may use masks/gaussian_in_mask.npy when that
    is the file on disk. Later orbits do not reuse that flat file.
    """
    nested = folder / "masks" / orbit / "gaussian_in_mask.npy"
    if nested.is_file():
        return nested
    if orbit == "orbit1":
        flat = folder / "masks" / "gaussian_in_mask.npy"
        if flat.is_file():
            return flat
    return nested


def mask_covers_room(lift: dict[str, Any] | None) -> bool:
    """True when the lift covers the room, the span is at least half, or a view covers the frame."""
    if not isinstance(lift, dict):
        return False
    if lift.get("covers_room") is True or _view_fractions_cover(lift):
        return True
    span_in = lift.get("span_inside")
    span_sc = lift.get("span_scene")
    if isinstance(span_in, bool) or isinstance(span_sc, bool):
        return False
    if not isinstance(span_in, (int, float)) or not isinstance(span_sc, (int, float)):
        return False
    if not np.isfinite(span_in) or not np.isfinite(span_sc) or float(span_sc) <= 0:
        return False
    return float(span_in) / float(span_sc) >= ROOM_SPAN


def _eligible_crops(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if float(row.get("recall") or 0) >= 0.98 and int(row.get("n_inside") or 0) >= 1000
    ]


def _eligible_iou_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["iou"]), -float(row.get("splat_frac") or 0))


def _open_iou_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["iou"]), float(row.get("recall") or 0))


def choose_crop_policy(rows: list[dict[str, Any]]) -> str | None:
    """Pick the crop whose silhouette matches, unless every eligible set covers the room.

    Eligibility matches the lift: recall at least 0.98 and at least 1000
    inside points. Among eligible rows, a set that does not cover the room
    wins over a higher-IoU set that does. If every eligible set covers the
    room, return no crop. Rows below that gate keep the previous fallback.
    """
    eligible = _eligible_crops(rows)
    if eligible:
        compact = [row for row in eligible if row.get("covers_room") is not True]
        if not compact:
            return None
        return str(max(compact, key=_eligible_iou_key)["name"])
    if not rows:
        return None
    compact = [row for row in rows if row.get("covers_room") is not True]
    chosen_pool = compact or list(rows)
    return str(max(chosen_pool, key=_open_iou_key)["name"])


def _highest_iou_name(rows: list[dict[str, Any]]) -> str | None:
    """The name the lift used to keep when every eligible crop covers the room."""
    eligible = _eligible_crops(rows)
    pool = eligible or list(rows)
    if not pool:
        return None
    key = _eligible_iou_key if eligible else _open_iou_key
    return str(max(pool, key=key)["name"])


def _finite_span(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), 6)


def _record_room_crop_refusal(
    folder: Path,
    orbit: str,
    rows: list[dict[str, Any]],
    span_scene: float | None,
) -> None:
    """Mark the orbit lift as covering the room without replacing the mask file."""
    mask_dir = Path(folder) / "masks" / orbit
    mask_dir.mkdir(parents=True, exist_ok=True)
    meta_path = mask_dir / "object.json"
    record = _load_json(meta_path)
    record["covers_room"] = True
    record["achieved"] = False
    record["infill"] = "not_run"
    record["policy"] = None
    record["projected"] = True
    eligible = _eligible_crops(rows)
    span_inside = None
    if eligible:
        best = max(eligible, key=_eligible_iou_key)
        span_inside = _finite_span(best.get("object_span"))
    scene = _finite_span(span_scene)
    record["span_inside"] = span_inside if span_inside is not None else record.get("span_inside")
    record["span_scene"] = scene if scene is not None else record.get("span_scene")
    meta_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def apply_crop_policy(
    folder: Path,
    orbit: str,
    rows: list[dict[str, Any]],
    *,
    span_scene: float | None = None,
) -> str | None:
    """Return the crop to write, or None when the mask file must stay put.

    choose_crop_policy returns no crop when every eligible set covers the room.
    A room capture then writes covers_room on masks/<orbit>/object.json, leaves
    achieved false, and does not replace gaussian_in_mask.npy. Features and
    infill stay unstarted. An orbit job, read from request.json, still receives
    the highest-IoU name so that capture can keep a crop.
    """
    chosen = choose_crop_policy(rows)
    if not chosen and _capture_job(Path(folder)) == "orbit":
        return _highest_iou_name(rows)
    if chosen:
        return chosen
    _record_room_crop_refusal(Path(folder), orbit, rows, span_scene)
    return None


def crop_decision(folder: Path, lift: dict[str, Any] | None, n_outside: int | None) -> str:
    """Return refused, object_only, or room.

    A room job does not crop the first room-sized mask. After the one retry,
    a smaller lift uses the room crop, and a lift that is still room-sized
    stays refused. An orbit job skips that half-scene test and does not
    schedule infill. A room-covering mask is still not a crop when every
    Gaussian is inside it, which means the object is the whole splat.
    """
    record = lift if isinstance(lift, dict) else {}
    outside = None if isinstance(n_outside, bool) or n_outside is None else int(n_outside)
    if _capture_job(Path(folder)) == "orbit":
        return "object_only"
    if mask_covers_room(record) and outside != 0:
        return "refused"
    if _object_only(Path(folder), record, outside):
        return "object_only"
    return "room"


def _object_only(folder: Path, lift: dict[str, Any], n_outside: int | None) -> bool:
    """The object is the scene, so there is no separate room to fill."""
    if n_outside == 0:
        return True
    job = _capture_job(folder)
    if job == "room":
        return False
    if job == "orbit":
        return not mask_covers_room(lift)
    reason = str(lift.get("infill_reason") or "")
    return "no separate background" in reason


def _capture_job(folder: Path) -> str | None:
    spec = _load_json(Path(folder) / "request.json")
    job = spec.get("job")
    if job in {"orbit", "room"}:
        return str(job)
    return None


def _view_fractions_cover(lift: dict[str, Any]) -> bool:
    fractions = lift.get("view_fractions")
    if not isinstance(fractions, list):
        return False
    for item in fractions:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        if not np.isfinite(item):
            continue
        if FRAME_COVER <= float(item) <= 1.0:
            return True
    return False


def delivers_room_without_object(spec: dict[str, Any] | None) -> bool:
    """True when the spec says the room is delivered with the object removed."""
    if not isinstance(spec, dict):
        return False
    deliverable = " ".join(str(spec.get("deliverable") or "").lower().replace("_", " ").replace("-", " ").split())
    if deliverable in {"object removed", "room without object", "background only"}:
        return True
    prompt = " ".join(str(spec.get("prompt") or "").lower().split())
    return "with the object removed" in prompt or "object removed" in prompt


def _infill_required(folder: Path) -> bool:
    """Infill runs for a room crop only when the spec asks for the object to be removed.

    A missing spec keeps the current room-crop requirement. An orbit job does not schedule it.
    """
    if _capture_job(folder) == "orbit":
        return False
    spec = _load_json(Path(folder) / "request.json")
    if not spec or spec.get("job") not in {"orbit", "room"}:
        return True
    return delivers_room_without_object(spec)


def read_block_b(folder: Path, orbit: str) -> dict[str, Any]:
    """Mask, object.json, instances, and the feature-field state already on disk.

    A refused field stays refused. This does not train one.
    """
    folder = Path(folder)
    lift_path = _sidecar(folder, orbit, "object")
    instance_path = _sidecar(folder, orbit, "instances")
    field = _feature_field(folder, orbit)
    instances = _load_json(instance_path)
    summary = _instance_summary(instances)
    return {
        "lift": _load_json(lift_path),
        "instances": summary,
        "feature_field": field["feature_field"],
        "feature_field_reason": field.get("reason"),
        "sources": {
            "mask": _existing_rel(folder, mask_file(folder, orbit)),
            "object_json": _existing_rel(folder, lift_path),
            "instances": _existing_rel(folder, instance_path),
        },
    }


def _sidecar(folder: Path, orbit: str, kind: str) -> Path:
    if kind == "object":
        nested = folder / "masks" / orbit / "object.json"
        flat = folder / "masks" / "object.json"
    else:
        nested = folder / "semantics" / orbit / "instances.json"
        flat = folder / "semantics" / "instances.json"
    if nested.is_file():
        return nested
    if orbit == "orbit1" and flat.is_file():
        return flat
    return nested


def _feature_field(folder: Path, orbit: str) -> dict[str, Any]:
    """Prefer an on-disk refusal over an older measured sidecar."""
    refused = [folder / "semantics" / orbit / "feature_field_refused.json"]
    sidecars = [folder / "semantics" / orbit / "feature_field.json"]
    if orbit == "orbit1":
        refused.append(folder / "semantics" / "feature_field_refused.json")
        sidecars.append(folder / "semantics" / "feature_field.json")
    for path in refused:
        if path.is_file():
            loaded = _load_json(path)
            reason = loaded.get("reason") or "feature field was refused"
            return {"feature_field": "refused", "reason": reason, "achieved": False}
    for path in sidecars:
        if not path.is_file():
            continue
        loaded = _load_json(path)
        state = loaded.get("feature_field")
        if state == "refused":
            return {"feature_field": "refused", "reason": loaded.get("reason"), "achieved": False}
        if state == "measured":
            return {"feature_field": "measured", "reason": loaded.get("reason"), "achieved": False}
    return {"feature_field": "unstarted", "reason": None, "achieved": False}


def _instance_summary(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("instances") if isinstance(record.get("instances"), list) else []
    chosen = None
    for row in rows:
        if isinstance(row, dict) and row.get("foreground") is True:
            chosen = row
            break
    if chosen is None:
        for row in rows:
            if isinstance(row, dict):
                chosen = row
                break
    review = record.get("review") if isinstance(record.get("review"), dict) else {}
    ready = record.get("block_c_ready") if "block_c_ready" in record else None
    return {
        "name": None if chosen is None else chosen.get("name"),
        "foreground": None if chosen is None else chosen.get("foreground") is True,
        "n_gaussians": None if chosen is None else chosen.get("n_gaussians"),
        "block_c_ready": ready,
        "review_problem": review.get("problem"),
        "review_verdict": review.get("verdict"),
    }


def _apply_block_b(row: dict[str, Any], block_b: dict[str, Any], counts: tuple[int, int] | None) -> None:
    lift = block_b["lift"] if isinstance(block_b.get("lift"), dict) else {}
    if counts is None:
        row["n_inside"] = lift.get("n_inside")
        row["n_outside"] = lift.get("n_outside")
    else:
        row["n_inside"] = int(counts[0])
        row["n_outside"] = int(counts[1])
    if "covers_room" in lift:
        row["covers_room"] = lift.get("covers_room") is True
    elif lift:
        row["covers_room"] = mask_covers_room(lift)
    else:
        row["covers_room"] = None
    row["span_inside"] = lift.get("span_inside")
    row["span_scene"] = lift.get("span_scene")
    summary = block_b.get("instances") if isinstance(block_b.get("instances"), dict) else {}
    row["instance_name"] = summary.get("name")
    row["instance_n_gaussians"] = summary.get("n_gaussians")
    row["review_problem"] = summary.get("review_problem")
    row["block_c_ready"] = summary.get("block_c_ready")
    row["feature_field"] = block_b.get("feature_field") or "unstarted"
    row["feature_field_reason"] = block_b.get("feature_field_reason")
    row["block_b"] = block_b.get("sources")
    row["achieved"] = False


def _refuse_note(lift: dict[str, Any], field: str, folder: Path | None = None) -> str:
    note = (
        "Mask covers the room "
        f"(covers_room {lift.get('covers_room') is True}, "
        f"span_inside {lift.get('span_inside')}, span_scene {lift.get('span_scene')}). "
        "The crop was not written. Infill was not run. It is not a pass."
    )
    if field == "refused":
        note += " The feature field stays refused and was not trained."
    if folder is not None and _load_json(Path(folder) / "mask_retry.json").get("detection") == "rerun":
        note += " The higher box threshold was already tried. The retry was not repeated."
    return note


def _drop_crop_outputs(paths: dict[str, Path]) -> None:
    _unlink(
        (
            paths["object_ply"],
            paths["background_ply"],
            paths["world_t"],
            paths["infilled"],
            paths["object_ply"].with_name("object.partial.ply"),
            paths["background_ply"].with_name("background.partial.ply"),
        )
    )


def orbit_paths(obj: str, orbit: str | int, root: Path | None = None) -> dict[str, Path]:
    """Paths for one object orbit. The object name is one folder."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    folder = object_dir(name, root)
    background = folder / "background" / orbit_id
    return {
        "folder": folder,
        "splat": folder / "colmap" / orbit_id / "splat.ply",
        "mask": mask_file(folder, orbit_id),
        "object_ply": folder / "objects" / name / orbit_id / "object.ply",
        "background_ply": background / "background.ply",
        "world_t": folder / "objects" / name / orbit_id / "world_T_object.json",
        "background_dir": background,
        "holes": background / "holes",
        "inpaint_dir": background / "inpaint",
        "renders": background / "renders",
        "review": background / "review",
        "infilled": background / "background_infilled.ply",
        "stage": background / "stage.json",
        "record": folder / "ingest" / "block_c.json",
    }


def ready_orbits(obj: str, root: Path | None = None) -> list[str]:
    """Orbits that already have both splat.ply and gaussian_in_mask.npy."""
    name = require_object_name(obj)
    folder = object_dir(name, root)
    colmap = folder / "colmap"
    if not colmap.is_dir():
        return []
    found = []
    for child in sorted(colmap.iterdir()):
        if not child.is_dir():
            continue
        try:
            orbit_id = require_orbit(child.name)
        except ValueError:
            continue
        if (child / "splat.ply").is_file() and mask_file(folder, orbit_id).is_file():
            found.append(orbit_id)
    return found


def objects_with_splats(root: Path | None = None) -> list[tuple[str, list[str]]]:
    """Every data/<obj> orbit that already has a splat and a mask."""
    data = resolve_root(root) / "data"
    if not data.is_dir():
        return []
    rows = []
    for child in sorted(data.iterdir()):
        if not child.is_dir():
            continue
        try:
            name = require_object_name(child.name)
        except ValueError:
            continue
        orbits = ready_orbits(name, root)
        if orbits:
            rows.append((name, orbits))
    return rows


def hole_mask(full_alpha: np.ndarray, background_alpha: np.ndarray) -> np.ndarray | None:
    """A hole is where background alpha is below the full-splat alpha.

    A shape mismatch or a non-finite value returns None. No mask is invented.
    """
    full = np.asarray(full_alpha, dtype=np.float64)
    back = np.asarray(background_alpha, dtype=np.float64)
    if full.shape != back.shape or full.ndim != 2 or full.size == 0:
        return None
    if not np.isfinite(full).all() or not np.isfinite(back).all():
        return None
    return back < full


def widen_mask(mask: np.ndarray, radius: int = WIDEN_RADIUS) -> np.ndarray:
    """Dilate an existing hole mask. An empty mask stays empty."""
    src = np.asarray(mask, dtype=bool)
    if src.ndim != 2:
        raise ValueError("hole mask must be 2-d")
    if radius < 1 or not bool(src.any()):
        return src.copy()
    from scipy.ndimage import binary_dilation

    structure = np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool)
    return np.asarray(binary_dilation(src, structure=structure), dtype=bool)


def psnr_outside(reference: np.ndarray, candidate: np.ndarray, outside: np.ndarray) -> float | None:
    """PSNR on pixels outside the hole. A missing region is not 0 dB."""
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    mask = np.asarray(outside, dtype=bool)
    if ref.shape != cand.shape or ref.ndim != 3 or mask.shape != ref.shape[:2]:
        return None
    if ref.size == 0 or not bool(mask.any()):
        return None
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        return None
    if float(np.nanmax(ref)) > 1.5 or float(np.nanmax(cand)) > 1.5:
        ref = ref / 255.0
        cand = cand / 255.0
    diff = ref[mask] - cand[mask]
    mse = float(np.mean(np.square(diff)))
    if not np.isfinite(mse):
        return None
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(1.0 / mse))


def psnr_drop_db(before: float | None, after: float | None) -> float | None:
    """How many dB the outside-hole PSNR fell. Missing inputs stay None."""
    if before is None or after is None:
        return None
    if isinstance(before, bool) or isinstance(after, bool):
        return None
    try:
        left = float(before)
        right = float(after)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(left) or not np.isfinite(right):
        return None
    return left - right


def infill_status(installed: bool, n_written: int) -> str:
    """LaMa has to write at least one image. Otherwise infill stays not_run."""
    if not installed or int(n_written) < 1:
        return "not_run"
    return "ran"


def fine_tune_status(infill: str, wrote_ply: bool) -> str:
    """No inpaint means the fine-tune stays not_run, even if a ply was requested."""
    if infill != "ran" or not wrote_ply:
        return "not_run"
    return "ran"


def m3_met(row: dict[str, Any]) -> bool:
    """True only when seams, PSNR drop, and floaters all pass.

    A VLM verdict alone is not M3. A skipped or failed infill is not a pass.
    """
    if row.get("infill") != "ran" or row.get("fine_tune") != "ran":
        return False
    if row.get("seam_second_fail") is True:
        return False
    floaters = row.get("floater_count")
    if type(floaters) is not int or floaters >= FLOATER_MAX:
        return False
    drop = row.get("psnr_drop_db")
    if isinstance(drop, bool) or not isinstance(drop, (int, float)):
        return False
    if not float(drop) < PSNR_DROP_MAX:
        return False
    vlm = row.get("vlm") if isinstance(row.get("vlm"), dict) else {}
    if vlm.get("asked") is not True or vlm.get("verdict") != "pass" or vlm.get("seams") is not False:
        return False
    return True


def review_seams(
    images: list[Path],
    review_dir: Path,
    record_path: Path | None = None,
) -> dict[str, Any]:
    """Copy the review PNGs, then ask the seam question. A missing PNG is not asked."""
    from tools.block_gates import prepare_review_image
    from tools.common.vlm import MissingReviewImage, VlmError, review

    paths = [Path(image) for image in images]
    if len(paths) < 6:
        return _not_asked("three before views and three after views are required")
    copied: list[Path] = []
    try:
        for image in paths:
            copied.append(prepare_review_image(image, review_dir / image.name))
    except MissingReviewImage as exc:
        return _not_asked(str(exc))
    try:
        verdict = review("C", copied, SEAM_QUESTION, list(ALLOWED_SEAM_CORRECTION), record_path)
    except (VlmError, OSError, ValueError) as exc:
        return _not_asked(_public(str(exc)))
    seams = not (verdict.get("verdict") == "pass" and verdict.get("problem") == "none")
    return {
        "asked": True,
        "verdict": verdict.get("verdict"),
        "problem": verdict.get("problem"),
        "seams": seams,
        "correction": verdict.get("correction"),
    }


def apply_seam_policy(
    row: dict[str, Any],
    ask: Callable[[bool], dict[str, Any]],
    rerun: Callable[[bool], dict[str, Any]],
) -> dict[str, Any]:
    """One seam fail retries C3 with a wider mask, then C4. A second fail is recorded."""
    updated = dict(row)
    first = ask(False)
    updated["vlm"] = first
    updated["seam_attempts"] = 1 if first.get("asked") is True else 0
    updated["seam_second_fail"] = False
    if first.get("asked") is True and first.get("seams") is True:
        retry = rerun(True)
        updated["retry"] = {
            "wider_hole_mask": True,
            "infill": retry.get("infill"),
            "fine_tune": retry.get("fine_tune"),
        }
        if retry.get("infill") == "ran":
            updated["infill"] = "ran"
        if retry.get("fine_tune") == "ran":
            updated["fine_tune"] = "ran"
            if retry.get("psnr_drop_db") is not None:
                updated["psnr_drop_db"] = retry.get("psnr_drop_db")
                updated["psnr_before"] = retry.get("psnr_before", updated.get("psnr_before"))
                updated["psnr_after"] = retry.get("psnr_after", updated.get("psnr_after"))
        second = ask(True)
        updated["vlm_retry"] = second
        if second.get("asked") is True:
            updated["seam_attempts"] = 2
            updated["vlm"] = second
            if second.get("seams") is True:
                updated["seam_second_fail"] = True
    updated["m3_met"] = m3_met(updated)
    updated["learned_prior"] = False
    return updated


def run_c1(obj: str, orbit: str | int, root: Path | None = None) -> dict[str, Any]:
    """Crop one orbit from the Block B mask. A room-covering mask writes nothing."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    paths = orbit_paths(name, orbit_id, root)
    splat = paths["splat"]
    mask_path = paths["mask"]
    block_b = read_block_b(paths["folder"], orbit_id)
    if not splat.is_file() or not mask_path.is_file():
        row = _base_row(name, orbit_id, "skipped", _missing_reason(splat, mask_path))
        row["infill"] = "skipped"
        _apply_block_b(row, block_b, None)
        _store_orbit(name, root, orbit_id, row)
        _note_infill_gate(name, root, row)
        return row
    try:
        flags = np.asarray(np.load(mask_path), dtype=bool).reshape(-1)
    except (OSError, ValueError) as exc:
        row = _base_row(name, orbit_id, "failed", _public(f"mask could not be read: {type(exc).__name__}: {exc}"))
        row["note"] = "gaussian_in_mask.npy could not be read. The crop was not written."
        _apply_block_b(row, block_b, None)
        _store_orbit(name, root, orbit_id, row)
        _note_infill_gate(name, root, row)
        return row
    n_inside = int(flags.sum())
    n_outside = int(flags.size - n_inside)
    lift = block_b["lift"] if isinstance(block_b.get("lift"), dict) else {}
    judged = dict(lift)
    from tools.task_spec import orbit_view_fractions

    extra = orbit_view_fractions(paths["folder"], orbit_id)
    if extra:
        stored = list(judged.get("view_fractions") or []) if isinstance(judged.get("view_fractions"), list) else []
        judged["view_fractions"] = stored + extra
    decision = crop_decision(paths["folder"], judged, n_outside)
    if decision == "refused":
        _drop_crop_outputs(paths)
        note = _refuse_note(judged, str(block_b.get("feature_field") or ""), paths["folder"])
        row = _base_row(name, orbit_id, "measured", note)
        row["crop"] = "refused"
        row["infill"] = "not_run"
        row["infill_required"] = True
        _apply_block_b(row, block_b, (n_inside, n_outside))
        _store_orbit(name, root, orbit_id, row)
        _note_infill_gate(name, root, row)
        return row
    temps = (
        paths["object_ply"].with_name("object.partial.ply"),
        paths["background_ply"].with_name("background.partial.ply"),
    )
    created: list[Path] = []
    try:
        stats = crop_splat(splat, flags, temps[0], background=temps[1])
        for source, dest in (
            (temps[0], paths["object_ply"]),
            (temps[1], paths["background_ply"]),
        ):
            dest.parent.mkdir(parents=True, exist_ok=True)
            source.replace(dest)
            created.append(dest)
        center = stats.get("center")
        if isinstance(center, list) and len(center) == 3:
            _write_world(paths["world_t"], center)
            created.append(paths["world_t"])
        elif paths["world_t"].is_file():
            paths["world_t"].unlink()
    except (KeyboardInterrupt, SystemExit):
        _unlink(temps)
        _unlink(created)
        raise
    except Exception as exc:
        _unlink(temps)
        _unlink(created)
        row = _base_row(name, orbit_id, "failed", _public(f"crop did not finish: {type(exc).__name__}: {exc}"))
        row["note"] = "crop_splat did not finish. object.ply, background.ply, and world_T_object were not invented."
        row["crop"] = decision
        _apply_block_b(row, block_b, (n_inside, n_outside))
        _store_orbit(name, root, orbit_id, row)
        _note_infill_gate(name, root, row)
        return row
    finally:
        _unlink(temps)
    floater_count = _floater_count(paths["object_ply"])
    if decision == "object_only":
        note = "Object ply was cropped. The object is the scene, so infill stays skipped and is not a pass."
        row = _base_row(name, orbit_id, "measured", note)
        row["infill"] = "skipped"
        row["infill_required"] = False
    else:
        summary = block_b.get("instances") if isinstance(block_b.get("instances"), dict) else {}
        problem = summary.get("review_problem")
        review = f" Instance review is {problem}, so this crop is not a pass." if problem else ""
        required = _infill_required(paths["folder"])
        if required:
            note = (
                "Object and background plies were cropped. "
                "Infill is required for this room capture and has not run."
                + review
            )
            row = _base_row(name, orbit_id, "partial", note)
            row["infill"] = "not_run"
        else:
            note = (
                "Object and background plies were cropped. "
                "The spec does not deliver the room with the object removed, so infill is not scheduled. "
                "It is not a pass."
                + review
            )
            row = _base_row(name, orbit_id, "partial", note)
            row["infill"] = "skipped"
        row["infill_required"] = required
    row["crop"] = decision
    row.update(
        {
            "n_object": int(stats["n_in"]),
            "n_background": int(stats["n_background"]),
            "n_source": int(stats["n_source"]),
            "n_removed_floaters": int(stats["n_removed_floaters"]),
            "floater_count": floater_count,
            "object_ply": _rel(paths["folder"], paths["object_ply"]),
            "background_ply": _rel(paths["folder"], paths["background_ply"]),
            "world_T_object": _rel(paths["folder"], paths["world_t"]) if paths["world_t"].is_file() else None,
            "translation": stats.get("center"),
        }
    )
    _apply_block_b(row, block_b, (n_inside, n_outside))
    _store_orbit(name, root, orbit_id, row)
    _note_infill_gate(name, root, row)
    return row


def _note_infill_gate(name: str, root: Path | None, row: dict[str, Any]) -> None:
    """A skipped or failed infill stays unmet. The switch only lets the walk continue."""
    from tools.gate_bypass import continue_past

    if row.get("m3_met") is True and row.get("infill") == "ran":
        return
    infill = row.get("infill")
    check = "skipped" if infill == "skipped" else "unmet"
    continue_past(
        name,
        root,
        "block_c_infill",
        f"crop {row.get('crop')} infill {infill}; m3 stays unmet",
        check=check,
    )


def record_later_stages(
    obj: str,
    orbit: str | int,
    root: Path | None,
    *,
    holes: dict[str, Any] | None = None,
    infill: str | None = None,
    fine_tune: str | None = None,
    psnr_before: float | None = None,
    psnr_after: float | None = None,
    n_inpainted: int | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Update one orbit in block_c.json. Does not invent plies or images."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    stored = load_block_c(name, root)
    orbits = stored.get("orbits") if isinstance(stored.get("orbits"), dict) else {}
    row = dict(orbits.get(orbit_id) or _base_row(name, orbit_id, "partial", ""))
    if row.get("crop") == "refused":
        return row
    if row.get("crop") == "object_only":
        infill = "skipped"
        fine_tune = "not_run"
    if holes is not None:
        row["holes"] = holes
    if infill is not None:
        row["infill"] = infill
    if fine_tune is not None:
        row["fine_tune"] = fine_tune
    if n_inpainted is not None:
        row["n_inpainted"] = int(n_inpainted)
    if psnr_before is not None or psnr_after is not None:
        row["psnr_before"] = psnr_before
        row["psnr_after"] = psnr_after
        row["psnr_drop_db"] = psnr_drop_db(psnr_before, psnr_after)
    if note:
        row["note"] = _public(note)
    if row.get("infill") != "ran":
        row["fine_tune"] = "not_run"
        row["psnr_before"] = None
        row["psnr_after"] = None
        row["psnr_drop_db"] = None
    row["m3_met"] = False if row.get("vlm") is None else m3_met(row)
    row["learned_prior"] = False
    row["splat_retrained"] = False
    row["colmap_rerun"] = False
    _store_orbit(name, root, orbit_id, row)
    return row


def save_orbit_row(obj: str, orbit: str | int, root: Path | None, row: dict[str, Any]) -> dict[str, Any]:
    """Write one orbit into data/<obj>/ingest/block_c.json."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    clean = dict(row)
    clean["orbit"] = orbit_id
    clean["object"] = name
    clean["learned_prior"] = False
    clean["splat_retrained"] = False
    clean["colmap_rerun"] = False
    clean["m3_met"] = m3_met(clean)
    _store_orbit(name, root, orbit_id, clean)
    return clean


def load_block_c(obj: str, root: Path | None = None) -> dict[str, Any]:
    path = orbit_paths(obj, 1, root)["record"]
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def orbit_record(obj: str, orbit: str | int, root: Path | None = None) -> dict[str, Any] | None:
    """The stored orbit row, or None when block_c.json has no such orbit."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    loaded = load_block_c(name, root)
    orbits = loaded.get("orbits") if isinstance(loaded.get("orbits"), dict) else {}
    row = orbits.get(orbit_id)
    return dict(row) if isinstance(row, dict) else None


def _store_orbit(obj: str, root: Path | None, orbit: str, row: dict[str, Any]) -> Path:
    path = object_dir(obj, root) / "ingest" / "block_c.json"
    existing = load_block_c(obj, root)
    orbits = existing.get("orbits") if isinstance(existing.get("orbits"), dict) else {}
    clean = _jsonable(row)
    clean["learned_prior"] = False
    clean["m3_met"] = m3_met(clean)
    clean["achieved"] = clean["m3_met"] is True
    orbits[orbit] = clean
    file_m3 = _file_m3(orbits)
    record = {
        "block": "C",
        "object": obj,
        "learned_prior": False,
        "splat_retrained": False,
        "colmap_rerun": False,
        "m3_met": file_m3,
        "achieved": file_m3,
        "orbits": orbits,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _file_m3(orbits: dict[str, Any]) -> bool:
    rows = [item for item in orbits.values() if isinstance(item, dict)]
    active = [item for item in rows if item.get("state") != "skipped"]
    if not active:
        return False
    return all(item.get("m3_met") is True for item in active)


def _base_row(obj: str, orbit: str, state: str, note: str) -> dict[str, Any]:
    return {
        "orbit": orbit,
        "object": obj,
        "state": state,
        "n_object": None,
        "n_background": None,
        "n_source": None,
        "n_removed_floaters": None,
        "floater_count": None,
        "object_ply": None,
        "background_ply": None,
        "world_T_object": None,
        "translation": None,
        "holes": {"status": "not_run", "n_masks": 0},
        "infill": "not_run",
        "fine_tune": "not_run",
        "n_inpainted": 0,
        "psnr_before": None,
        "psnr_after": None,
        "psnr_drop_db": None,
        "vlm": {"asked": False, "verdict": None, "seams": None},
        "seam_attempts": 0,
        "seam_second_fail": False,
        "crop": None,
        "n_inside": None,
        "n_outside": None,
        "covers_room": None,
        "span_inside": None,
        "span_scene": None,
        "infill_required": False,
        "achieved": False,
        "feature_field": "unstarted",
        "instance_name": None,
        "block_c_ready": None,
        "m3_met": False,
        "learned_prior": False,
        "splat_retrained": False,
        "colmap_rerun": False,
        "note": note,
    }


def _missing_reason(splat: Path, mask: Path) -> str:
    has_splat = splat.is_file()
    has_mask = mask.is_file()
    if not has_splat and not has_mask:
        return "splat.ply and gaussian_in_mask.npy are missing, so the crop was not written."
    if not has_splat:
        return "gaussian_in_mask.npy is on disk. splat.ply is missing, so the crop was not written."
    return "splat.ply is on disk. gaussian_in_mask.npy is missing, so the crop was not written."


def _write_world(path: Path, translation: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "translation": [float(value) for value in translation],
        "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "scale": 1.0,
    }
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _floater_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        _header, table = _read_splat_table(path)
    except (OSError, ValueError):
        return None
    if len(table) == 0:
        return 0
    return int(count_floaters(table[:, 0:3]))


def _not_asked(reason: str) -> dict[str, Any]:
    return {
        "asked": False,
        "verdict": None,
        "problem": None,
        "seams": None,
        "reason": _public(reason),
    }


def _load_json(path: Path) -> dict[str, Any]:
    file = Path(path)
    if not file.is_file():
        return {}
    try:
        loaded = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _existing_rel(folder: Path, path: Path) -> str | None:
    if not Path(path).is_file():
        return None
    return _rel(folder, Path(path))


def _public(text: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    cleaned = text
    if key and key in cleaned:
        cleaned = cleaned.replace(key, "[redacted]")
    return cleaned


def _rel(folder: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(folder.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _unlink(paths: list[Path] | tuple[Path, ...]) -> None:
    for path in paths:
        if path.is_file():
            path.unlink()


def _real_image(path: Path) -> bool:
    """A real review or inpaint image is a non-trivial PNG or JPEG."""
    try:
        size = path.stat().st_size
        header = path.read_bytes()[:8]
    except OSError:
        return False
    if size < 32:
        return False
    return header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")


def real_images(directory: Path) -> list[Path]:
    """Inpaint or review images that are actually pictures."""
    folder = Path(directory)
    if not folder.is_dir():
        return []
    return [path for path in sorted(folder.iterdir()) if path.is_file() and _real_image(path)]


def write_hole_mask(path: Path, full_alpha: np.ndarray, background_alpha: np.ndarray) -> np.ndarray | None:
    """Write one hole mask. A bad pair writes nothing and returns None."""
    mask = hole_mask(full_alpha, background_alpha)
    if mask is None:
        return None
    destination = Path(path)
    if destination.suffix != ".npy":
        destination = destination.with_suffix(".npy")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(mask, dtype=bool))
    return mask


def run_c2(
    obj: str,
    orbit: str | int,
    root: Path | None = None,
    views: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Record hole masks for one orbit. No alpha arrays stays not_run.

    This does not call a renderer. m3 stays false until infill and the fine-tune pass too.
    """
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    existing = orbit_record(name, orbit_id, root)
    if existing and existing.get("crop") in {"refused", "object_only"}:
        return existing
    if existing and existing.get("infill_required") is False:
        return existing
    if not views:
        return record_later_stages(
            name,
            orbit_id,
            root,
            holes={"status": "not_run", "n_masks": 0, "reason": "no alpha arrays"},
            note="No full-splat alpha was supplied, so no hole mask was written.",
        )
    result = write_hole_masks(orbit_paths(name, orbit_id, root)["holes"], list(views))
    note = (
        "Hole masks are where background alpha is below the full splat."
        if result.get("status") == "ran"
        else "No hole mask was written."
    )
    return record_later_stages(name, orbit_id, root, holes=result, note=note)


def _load_lama() -> Callable[..., Any] | None:
    """The LaMa class, or None when the package cannot be imported."""
    try:
        from simple_lama_inpainting import SimpleLama
    except ImportError:
        return None
    return SimpleLama


def _c3_not_run(obj: str, orbit: str, root: Path | None, note: str) -> dict[str, Any]:
    """Infill and the fine-tune stay not_run. No image and no infilled ply are written."""
    return record_later_stages(
        obj,
        orbit,
        root,
        infill="not_run",
        fine_tune="not_run",
        n_inpainted=0,
        note=note,
    )


def run_c3(
    obj: str,
    orbit: str | int,
    root: Path | None = None,
    background: np.ndarray | None = None,
    masks: list[np.ndarray] | None = None,
    loader: Callable[[], Callable[..., Any] | None] | None = None,
) -> dict[str, Any]:
    """Inpaint hole masks with LaMa, conditioned on the surrounding background.

    If LaMa cannot be imported, infill stays not_run and no image is written.
    This does not start the fine-tune.
    """
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    existing = orbit_record(name, orbit_id, root)
    if existing and existing.get("crop") in {"refused", "object_only"}:
        return existing
    if existing and existing.get("infill_required") is False:
        return existing
    load = _load_lama if loader is None else loader
    factory = load()
    if factory is None:
        return _c3_not_run(
            name,
            orbit_id,
            root,
            "LaMa is not importable, so infill stays not_run. The fine-tune was not started.",
        )
    if background is None or not masks:
        return _c3_not_run(
            name,
            orbit_id,
            root,
            "No background image or hole mask was supplied, so infill stays not_run. The fine-tune was not started.",
        )
    written = _lama_inpaint(factory, background, masks, orbit_paths(name, orbit_id, root)["inpaint_dir"])
    if written < 1:
        return _c3_not_run(
            name,
            orbit_id,
            root,
            "LaMa did not write an image, so infill stays not_run. The fine-tune was not started.",
        )
    return record_later_stages(
        name,
        orbit_id,
        root,
        infill="ran",
        fine_tune="not_run",
        n_inpainted=written,
        note="LaMa wrote inpaint images. The fine-tune was not started.",
    )


def _lama_inpaint(
    factory: Callable[..., Any],
    background: np.ndarray,
    masks: list[np.ndarray],
    directory: Path,
) -> int:
    """Save LaMa output only. A failed call deletes the partial image and returns 0."""
    from PIL import Image

    image = np.asarray(background)
    if image.ndim != 3 or image.shape[2] != 3:
        return 0
    try:
        model = factory()
    except Exception:
        return 0
    directory = Path(directory)
    written = 0
    for index, mask in enumerate(masks):
        hole = np.asarray(mask, dtype=bool)
        if hole.shape != image.shape[:2]:
            continue
        dest = directory / f"view_{index:04d}.png"
        try:
            rgb = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            hole_image = Image.fromarray(np.where(hole, 255, 0).astype(np.uint8), mode="L")
            result = model(rgb, hole_image)
            if not isinstance(result, Image.Image):
                result = Image.fromarray(np.asarray(result))
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.save(dest)
            if _real_image(dest):
                written += 1
            elif dest.is_file():
                dest.unlink()
        except Exception:
            if dest.is_file():
                dest.unlink()
    return written


def write_hole_masks(directory: Path, views: list[tuple[str, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    """Write one hole mask per rendered view. No render writes nothing."""
    if not views:
        return {"status": "not_run", "n_masks": 0, "reason": "no render"}
    destination = Path(directory)
    partial = destination.with_name(destination.name + "_partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    written: list[Path] = []
    try:
        for stem, full_alpha, background_alpha in views:
            safe = Path(str(stem)).name
            if safe in {"", ".", ".."} or safe != str(stem):
                raise ValueError(f"unsafe view name {stem!r}")
            mask = hole_mask(full_alpha, background_alpha)
            if mask is None:
                raise ValueError(f"{safe} alpha shapes do not match")
            path = partial / f"{safe}_hole.npy"
            np.save(path, np.asarray(mask, dtype=bool))
            written.append(path)
    except (KeyboardInterrupt, SystemExit):
        shutil.rmtree(partial, ignore_errors=True)
        raise
    except (OSError, ValueError) as exc:
        shutil.rmtree(partial, ignore_errors=True)
        return {"status": "not_run", "n_masks": 0, "reason": _public(str(exc))}
    if destination.exists():
        shutil.rmtree(destination)
    partial.rename(destination)
    return {"status": "ran", "n_masks": len(written)}


def score_disk_stages(obj: str, orbit: str | int, root: Path | None = None) -> dict[str, Any]:
    """Infill and fine-tune stay not_run when no real inpaint image is on disk.

    This does not write a ply, a hole mask, or an inpaint image.
    """
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    stored = orbit_record(name, orbit_id, root)
    if stored and (stored.get("state") == "skipped" or stored.get("crop") in {"refused", "object_only"}):
        return stored
    if stored and stored.get("infill_required") is False:
        return stored
    paths = orbit_paths(name, orbit_id, root)
    n_images = len(real_images(paths["inpaint_dir"]))
    infill = "ran" if n_images >= 1 else "not_run"
    wrote_ply = paths["infilled"].is_file() and paths["infilled"].stat().st_size > 64
    fine = fine_tune_status(infill, wrote_ply)
    note = None
    if n_images < 1 and stored and stored.get("crop") == "room":
        note = "Infill is required for this room capture and has not run."
    elif n_images < 1:
        note = "No real inpaint image is on disk, so infill and fine-tune stay not_run."
    return record_later_stages(
        name,
        orbit_id,
        root,
        infill=infill,
        fine_tune=fine,
        n_inpainted=n_images,
        note=note,
    )


def merge_stage_file(obj: str, orbit: str | int, root: Path | None = None) -> dict[str, Any]:
    """Copy a host stage.json into block_c.json. A claimed run without files stays not_run."""
    name = require_object_name(obj)
    orbit_id = require_orbit(orbit)
    paths = orbit_paths(name, orbit_id, root)
    if not paths["stage"].is_file():
        return record_later_stages(
            name,
            orbit_id,
            root,
            holes={"status": "not_run", "n_masks": 0, "reason": "no render"},
            infill="not_run",
            fine_tune="not_run",
            note="No render stage file, so hole masks, infill, and fine-tune stay not_run.",
        )
    try:
        loaded = json.loads(paths["stage"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return record_later_stages(
            name,
            orbit_id,
            root,
            holes={"status": "not_run", "n_masks": 0},
            infill="not_run",
            fine_tune="not_run",
            note=_public(f"stage file could not be read: {type(exc).__name__}"),
        )
    if not isinstance(loaded, dict):
        loaded = {}
    stored = orbit_record(name, orbit_id, root)
    if stored and stored.get("crop") in {"refused", "object_only"}:
        return stored
    if loaded.get("crop") in {"refused", "object_only"}:
        base = dict(stored or _base_row(name, orbit_id, "measured", ""))
        base["crop"] = loaded["crop"]
        base["state"] = "measured"
        base["infill"] = "skipped" if loaded["crop"] == "object_only" else "not_run"
        base["infill_required"] = loaded["crop"] != "object_only"
        base["fine_tune"] = "not_run"
        base["achieved"] = False
        if isinstance(loaded.get("note"), str):
            base["note"] = loaded["note"]
        return save_orbit_row(name, orbit_id, root, base)
    holes = loaded.get("holes") if isinstance(loaded.get("holes"), dict) else {"status": "not_run", "n_masks": 0}
    n_files = len(list(paths["holes"].glob("*_hole.npy"))) if paths["holes"].is_dir() else 0
    if holes.get("status") == "ran" and n_files < 1:
        holes = {"status": "not_run", "n_masks": 0, "reason": "no render"}
    elif holes.get("status") == "ran":
        holes = dict(holes)
        holes["n_masks"] = n_files
    n_images = len(real_images(paths["inpaint_dir"]))
    infill = "ran" if loaded.get("infill") == "ran" and n_images >= 1 else "not_run"
    wrote_ply = paths["infilled"].is_file() and paths["infilled"].stat().st_size > 64
    fine = "ran" if loaded.get("fine_tune") == "ran" and fine_tune_status(infill, wrote_ply) == "ran" else "not_run"
    before = loaded.get("psnr_before") if fine == "ran" else None
    after = loaded.get("psnr_after") if fine == "ran" else None
    note = loaded.get("note") if isinstance(loaded.get("note"), str) else None
    return record_later_stages(
        name,
        orbit_id,
        root,
        holes=holes,
        infill=infill,
        fine_tune=fine,
        n_inpainted=n_images if infill == "ran" else 0,
        psnr_before=before if isinstance(before, (int, float)) and not isinstance(before, bool) else None,
        psnr_after=after if isinstance(after, (int, float)) and not isinstance(after, bool) else None,
        note=note,
    )


def refresh_orbit_chain(obj: str, root: Path | None = None) -> list[str]:
    """Point each ### C section at ingest/block_c.json. A missing markdown stays missing."""
    from tools.orbit_chain import refresh_c_line

    name = require_object_name(obj)
    folder = object_dir(name, root)
    markdown = folder / "ingest" / "orbit_chain.md"
    if not markdown.is_file():
        return []
    text = markdown.read_text(encoding="utf-8")
    old = "Block C is not run and is not a pass."
    new = "Block C is read from ingest/block_c.json. A missing file stays skipped and is not a pass."
    if old in text:
        markdown.write_text(text.replace(old, new, 1), encoding="utf-8")
    loaded = load_block_c(name, root)
    orbits = loaded.get("orbits") if isinstance(loaded.get("orbits"), dict) else {}
    updated = []
    for orbit in orbits:
        if refresh_c_line(folder, markdown, str(orbit)):
            updated.append(str(orbit))
    return updated


def sync_chain_records(obj: str, root: Path | None = None) -> list[str]:
    """Copy each orbit into chain/<orbit>/ingest/block_c.json. A refused crop drops that pose."""
    from tools.orbit_chain import _c_from_row, _write_record

    name = require_object_name(obj)
    folder = object_dir(name, root)
    loaded = load_block_c(name, root)
    orbits = loaded.get("orbits") if isinstance(loaded.get("orbits"), dict) else {}
    written: list[str] = []
    for orbit, row in orbits.items():
        if not isinstance(row, dict):
            continue
        chain = folder / "chain" / str(orbit)
        if not chain.is_dir():
            continue
        if row.get("crop") == "refused":
            pose = chain / "world_T_object.json"
            if pose.is_file():
                pose.unlink()
        recorded = _c_from_row(str(orbit), row)
        recorded["object"] = name
        _write_record(chain / "ingest" / "block_c.json", recorded)
        written.append(str(orbit))
    return written


def main(argv: list[str] | None = None) -> int:
    """Crop every ready orbit from its Block B mask. A room-covering mask is not a crop."""
    parser = argparse.ArgumentParser(description="Crop Block C for one object")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--orbit", action="append", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--refresh-chain", action="store_true")
    args = parser.parse_args(argv)
    name = require_object_name(args.obj)
    orbits = [require_orbit(item) for item in args.orbit] if args.orbit else ready_orbits(name, args.root)
    if not orbits:
        print(f"{name}: no orbit has both splat.ply and gaussian_in_mask.npy")
        return 0
    for orbit in orbits:
        row = run_c1(name, orbit, args.root)
        brief = {
            "orbit": row.get("orbit"),
            "state": row.get("state"),
            "crop": row.get("crop"),
            "n_inside": row.get("n_inside"),
            "n_outside": row.get("n_outside"),
            "n_object": row.get("n_object"),
            "n_background": row.get("n_background"),
            "floater_count": row.get("floater_count"),
            "infill": row.get("infill"),
            "fine_tune": row.get("fine_tune"),
            "feature_field": row.get("feature_field"),
            "achieved": row.get("achieved"),
            "m3_met": row.get("m3_met"),
            "note": row.get("note"),
        }
        print(json.dumps(brief), flush=True)
    folder = object_dir(name, args.root)
    if (folder / "chain").is_dir():
        synced = sync_chain_records(name, args.root)
        print(json.dumps({"chain": synced}), flush=True)
    if args.refresh_chain or (folder / "ingest" / "orbit_chain.md").is_file():
        pointed = refresh_orbit_chain(name, args.root)
        print(json.dumps({"orbit_chain": pointed}), flush=True)
    return 0


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
