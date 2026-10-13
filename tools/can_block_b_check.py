"""Compare a can orbit Block B score file to the artifacts it claims to copy.

A skipped or missing check is a fail. It is never a pass and never flips
achieved. This module does not train, does not call SAM or DINOv2, and
does not invent mIoU from silhouette IoU.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.block_gates import judge_projection

ORBITS = ("orbit2", "orbit3")
ALL_ORBITS = ("orbit1", "orbit2", "orbit3")
REVIEW_STEMS = {"orbit1": "frame_00167", "orbit2": "frame_00001", "orbit3": "frame_00125"}
FRAME_PIXELS = {"orbit1": 1080 * 1920, "orbit2": 2160 * 3840, "orbit3": 2160 * 3840}
M1_PSNR = 27.0
M1_LPIPS = 0.20
REGISTERED_MIN = 90.0
REPROJ_MAX_PX = 1.0
STEP_LABELS = {
    "A": "measured",
    "B1": "measured",
    "B2": "LangSplat unstarted",
    "B3": "mask lift measured",
    "B4": "mIoU unmeasured",
}


def check_orbit(folder: Path, orbit: str) -> dict[str, Any]:
    """Walk Block A → B1 → B2 → B3 → B4 for one can orbit and write a debug record."""
    folder = Path(folder)
    if orbit not in ALL_ORBITS:
        raise ValueError(f"can Block B checker only scores {ALL_ORBITS}, not {orbit}")
    score_path = folder / "ingest" / f"block_b_{orbit}.json"
    codebook_name = "codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json"
    sources = {
        "score": score_path,
        "ns_eval": folder / "ingest" / f"ns_eval_{orbit}.json",
        "block_a": folder / "ingest" / "block_a.json",
        "object": folder / "masks" / orbit / "object.json",
        "codebook": folder / codebook_name,
        "instances": folder / "semantics" / orbit / "instances.json",
        "mesh_score": folder / "mesh_score.json",
        "unregistered": folder / "debug" / f"unregistered_{orbit}.json",
    }
    loaded = {name: _load(path) for name, path in sources.items()}
    present = {name: path.is_file() for name, path in sources.items()}
    steps = {
        "A": _check_a(folder, orbit, loaded, present),
        "B1": _check_b1(orbit, loaded, present),
        "B2": _check_b2(folder, orbit, loaded, present),
        "B3": _check_b3(folder, orbit, loaded, present),
        "B4": _check_b4(orbit, loaded, present),
    }
    score = loaded["score"]
    m1_met = False
    m2_met = False
    if steps["A"].get("metres_per_unit") is not None and score.get("m1_met") is True:
        m1_met = True
    if (
        steps["B4"]["numbers"].get("miou") is not None
        and int(steps["B4"]["numbers"].get("hand_labels") or 0) >= 20
        and score.get("m2_met") is True
    ):
        m2_met = True
    record = {
        "object": "can",
        "orbit": orbit,
        "ok": all(step["ok"] for step in steps.values()),
        "achieved": False,
        "m1_met": m1_met,
        "m2_met": m2_met,
        "steps": steps,
        "sources": {name: _rel(path) for name, path in sources.items()},
        "present": present,
    }
    dest = folder / "debug" / f"block_b_{orbit}_check.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def probe_b123(folder: Path, orbit: str) -> dict[str, Any]:
    """Quality signals for B1–B3. Does not flip achieved. Missing files are skips."""
    folder = Path(folder)
    if orbit not in ALL_ORBITS:
        raise ValueError(f"unknown orbit {orbit}")
    obj = _load(folder / "masks" / orbit / "object.json")
    codebook_path = folder / ("codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json")
    codebook = _load(codebook_path)
    n_views = obj.get("n_views")
    n_det = obj.get("n_views_with_detection")
    largest = obj.get("largest_foreground_pixels")
    frame_px = FRAME_PIXELS[orbit]
    mask_frac = (largest / frame_px) if isinstance(largest, int) and frame_px else None
    span_in = _number(obj.get("span_inside"))
    span_sc = _number(obj.get("span_scene"))
    judged = judge_projection(span_in, span_sc) if span_in is not None and span_sc is not None else None
    stem = REVIEW_STEMS[orbit]
    reviews = {
        name: (folder / "reviews" / f"{stem}{suffix}").is_file()
        for name, suffix in (
            ("capture", ".png"),
            ("mask", "_mask.png"),
            ("splat", "_splat.png"),
            ("splat_crop", "_splat_crop.png"),
        )
    }
    npy = folder / "masks" / orbit / "gaussian_in_mask.npy"
    record = {
        "orbit": orbit,
        "B1": {
            "object_json": bool(obj),
            "n_views": n_views,
            "n_views_with_detection": n_det,
            "empty_views": isinstance(n_views, int) and isinstance(n_det, int) and n_det < n_views,
            "largest_foreground_pixels": largest,
            "frame_pixels": frame_px,
            "largest_mask_frac": mask_frac,
            "oversegment_largest": mask_frac is not None and mask_frac >= 0.8,
        },
        "B2": {
            "codebook_file": codebook_path.name,
            "codebook_exists": codebook_path.is_file(),
            "dim": len(codebook["vector"]) if isinstance(codebook.get("vector"), list) else None,
            "learned_prior": codebook.get("learned_prior") is True,
            "langsplat": codebook.get("langsplat_codebook") is True,
        },
        "B3": {
            "projected": obj.get("projected") is True,
            "n_inside": obj.get("n_inside"),
            "n_views_projected": obj.get("n_views_projected"),
            "n_views_skipped_empty": obj.get("n_views_skipped_empty"),
            "covers_room": obj.get("covers_room"),
            "judge_projection": judged,
            "npy_exists": npy.is_file(),
            "reviews": reviews,
            "asked": False,
        },
    }
    dest = folder / "debug" / f"block_b_{orbit}_b123_probe.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def _check_a(folder: Path, orbit: str, loaded: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    mismatch: list[str] = []
    skipped: list[str] = []
    score = loaded["score"]
    ns_eval = loaded["ns_eval"]
    block_a = loaded["block_a"]
    if not present["score"]:
        skipped.append("ingest/block_b_{} score missing".format(orbit))
    if not present["ns_eval"]:
        skipped.append(f"ingest/ns_eval_{orbit}.json missing")
    results = ns_eval.get("results") if isinstance(ns_eval.get("results"), dict) else {}
    psnr = _number(results.get("psnr"))
    lpips = _number(results.get("lpips"))
    row = _orbit_row(block_a, orbit)
    if not present["block_a"] or not row:
        skipped.append("ingest/block_a.json orbit row missing")
    registered = _number(row.get("registered_pct"))
    reproj = _number(row.get("mean_reproj_px"))
    photometric_pass = (
        psnr is not None and lpips is not None and psnr > M1_PSNR and lpips < M1_LPIPS
    )
    register_pass = (
        registered is not None
        and reproj is not None
        and registered >= REGISTERED_MIN
        and reproj <= REPROJ_MAX_PX
    )
    metres = _metres_per_unit(folder, score)
    claimed_m1 = score.get("m1_met") is True or score.get("m1_achieved") is True
    if metres is None and claimed_m1:
        mismatch.append("m1_met/m1_achieved true while metres_per_unit is missing")
    score_m1 = score.get("m1") if isinstance(score.get("m1"), dict) else {}
    score_orbit = score_m1.get("orbit") if isinstance(score_m1.get("orbit"), dict) else {}
    if psnr is not None and _number(score_orbit.get("psnr")) is not None:
        if abs(float(score_orbit["psnr"]) - psnr) > 1e-6:
            mismatch.append("score PSNR does not match ns_eval")
    if lpips is not None and _number(score_orbit.get("lpips")) is not None:
        if abs(float(score_orbit["lpips"]) - lpips) > 1e-6:
            mismatch.append("score LPIPS does not match ns_eval")
    achieved_claimed = score.get("m1_achieved") is True or score.get("m1_met") is True
    ok = not skipped and not mismatch
    return _step(
        "A",
        ok,
        {
            "psnr": psnr,
            "lpips": lpips,
            "photometric_pass": photometric_pass,
            "registered_pct": registered,
            "mean_reproj_px": reproj,
            "register_pass": register_pass,
            "metres_per_unit": metres,
        },
        mismatch,
        achieved_claimed,
        skipped,
        metres=metres,
    )


def _check_b1(orbit: str, loaded: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    mismatch: list[str] = []
    skipped: list[str] = []
    if not present["object"]:
        skipped.append(f"masks/{orbit}/object.json missing")
    part = _part(loaded["score"], "B1")
    empirical = _empirical(part)
    obj = loaded["object"]
    n_views = obj.get("n_views")
    n_det = obj.get("n_views_with_detection")
    for key in ("n_views", "n_views_with_detection", "largest_foreground_pixels"):
        if key in obj and empirical.get(key) != obj.get(key):
            mismatch.append(f"B1 {key} does not match object.json")
    if obj.get("detections") is not None and empirical.get("detections") != obj.get("detections"):
        mismatch.append("B1 detections do not match object.json")
    empty = (
        isinstance(n_views, int)
        and isinstance(n_det, int)
        and n_det < n_views
    )
    if empty and part.get("achieved") is True:
        mismatch.append("B1 empty views are not a pass")
    if empty and part.get("state") == "pass":
        mismatch.append("B1 empty views are measured, not a pass")
    achieved_claimed = part.get("achieved") is True
    ok = not skipped and not mismatch
    return _step(
        "B1",
        ok,
        {
            "n_views": n_views,
            "n_views_with_detection": n_det,
            "detection_rate": (n_det / n_views) if isinstance(n_views, int) and n_views else None,
            "largest_foreground_pixels": obj.get("largest_foreground_pixels"),
            "detections": obj.get("detections"),
            "empty_views": empty,
        },
        mismatch,
        achieved_claimed,
        skipped,
    )


def _check_b2(folder: Path, orbit: str, loaded: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    mismatch: list[str] = []
    skipped: list[str] = []
    if not present["codebook"]:
        skipped.append(f"codebook_{orbit}.json missing")
    part = _part(loaded["score"], "B2")
    empirical = _empirical(part)
    codebook = loaded["codebook"]
    vector = codebook.get("vector")
    dim = len(vector) if isinstance(vector, list) else None
    object_vector = empirical.get("object_vector") if isinstance(empirical.get("object_vector"), dict) else {}
    sidecar = _load(folder / "semantics" / "langsplat_codebook.json")
    has_book = _npz_is_langsplat(folder / "semantics" / "features.codebook.npz")
    if codebook.get("learned_prior") is True or empirical.get("learned_prior") is True or object_vector.get("learned_prior") is True:
        mismatch.append("learned_prior cannot be true")
    if codebook.get("n_copied") is not None and object_vector.get("n_copied") not in (None, codebook.get("n_copied")):
        mismatch.append("B2 n_copied does not match codebook")
    if dim is not None and object_vector.get("dim") not in (None, dim):
        mismatch.append("B2 dim does not match codebook vector")
    if sidecar.get("langsplat_codebook") is True and not has_book:
        mismatch.append("langsplat_codebook true but features.codebook.npz is not a LangSplat codebook")
    achieved_claimed = part.get("achieved") is True or object_vector.get("achieved") is True
    ok = not skipped and not mismatch
    return _step(
        "B2",
        ok,
        {
            "dinov2": codebook.get("dinov2") is True,
            "openclip": codebook.get("openclip") is True,
            "learned_prior": False if codebook.get("learned_prior") is not True else True,
            "langsplat": has_book,
            "langsplat_state": "measured" if has_book else "unstarted",
            "dim": dim,
            "n_copied": codebook.get("n_copied"),
            "n_gaussians": codebook.get("n_gaussians"),
        },
        mismatch,
        achieved_claimed,
        skipped,
    )


def _check_b3(folder: Path, orbit: str, loaded: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    mismatch: list[str] = []
    skipped: list[str] = []
    if not present["object"]:
        skipped.append(f"masks/{orbit}/object.json missing")
    part = _part(loaded["score"], "B3")
    empirical = _empirical(part)
    lift = empirical.get("mask_lift") if isinstance(empirical.get("mask_lift"), dict) else {}
    obj = loaded["object"]
    span_inside = _number(obj.get("span_inside"))
    span_scene = _number(obj.get("span_scene"))
    judged = None
    if span_inside is not None and span_scene is not None:
        judged = judge_projection(span_inside, span_scene)
    for key in ("n_inside", "n_outside", "n_gaussians", "n_views_projected", "n_views_skipped_empty"):
        if key in obj and lift.get(key) not in (None, obj.get(key)) and lift.get(key) != obj.get(key):
            mismatch.append(f"B3 {key} does not match object.json")
    if judged is not None and lift.get("judge_projection") not in (None, judged):
        mismatch.append(f"B3 judge_projection should be {judged}")
    if "covers_room" in obj and lift.get("covers_room") not in (None, obj.get("covers_room")):
        mismatch.append("B3 covers_room does not match object.json")
    if lift.get("infill") == "skipped" and part.get("achieved") is True:
        mismatch.append("B3 skipped infill is not a pass")
    reviews = empirical.get("reviews") if isinstance(empirical.get("reviews"), dict) else {}
    stem = REVIEW_STEMS[orbit]
    review_names = [f"{stem}.png", f"{stem}_mask.png", f"{stem}_splat.png", f"{stem}_splat_crop.png"]
    missing_reviews = [name for name in review_names if not (folder / "reviews" / name).is_file()]
    asked = reviews.get("asked")
    if missing_reviews:
        asked = False
        if reviews.get("asked") is True:
            mismatch.append("missing review PNG cannot set asked true")
    if part.get("achieved") is True:
        mismatch.append("B3 mask lift is measured, not achieved")
    field = empirical.get("feature_field")
    feat_path = folder / "semantics" / "gaussian_features.npy"
    if field == "measured" and not feat_path.is_file():
        mismatch.append("feature_field measured but gaussian_features.npy missing")
    if field in ("skipped", "unstarted") and part.get("achieved") is True:
        mismatch.append("B3 skipped feature field is not a pass")
    achieved_claimed = part.get("achieved") is True
    ok = not skipped and not mismatch
    return _step(
        "B3",
        ok,
        {
            "n_inside": obj.get("n_inside"),
            "n_outside": obj.get("n_outside"),
            "n_gaussians": obj.get("n_gaussians"),
            "covers_room": obj.get("covers_room"),
            "span_inside": span_inside,
            "span_scene": span_scene,
            "span_ratio": (span_inside / span_scene) if span_inside is not None and span_scene else None,
            "judge_projection": judged,
            "infill": lift.get("infill") or obj.get("infill"),
            "asked": asked is True,
            "missing_reviews": missing_reviews,
            "feature_field": field,
        },
        mismatch,
        achieved_claimed,
        skipped,
    )


def _check_b4(orbit: str, loaded: dict[str, Any], present: dict[str, bool]) -> dict[str, Any]:
    mismatch: list[str] = []
    skipped: list[str] = []
    if not present["instances"]:
        skipped.append(f"semantics/{orbit}/instances.json missing")
    part = _part(loaded["score"], "B4")
    empirical = _empirical(part)
    instances = loaded["instances"]
    miou = instances.get("miou")
    hand_labels = instances.get("hand_labels", 0)
    silhouette = _silhouette_iou(loaded["mesh_score"], orbit)
    score_miou = empirical.get("miou", loaded["score"].get("miou"))
    if silhouette is not None and _number(score_miou) is not None and abs(float(score_miou) - silhouette) <= 1e-9:
        mismatch.append("silhouette IoU is not mIoU")
    if score_miou is not None and miou is None and _number(score_miou) is not None:
        mismatch.append("score invented miou; instances.miou is null")
    claimed_m2 = loaded["score"].get("m2_met") is True or loaded["score"].get("m2_achieved") is True or part.get("achieved") is True
    if claimed_m2 and (miou is None or int(hand_labels or 0) == 0):
        mismatch.append("m2_met/m2_achieved true while miou is null or hand_labels is 0")
    if instances.get("learned_prior") is True or empirical.get("learned_prior") is True:
        mismatch.append("learned_prior cannot be true")
    for key in ("instance_count", "n_object", "n_unassigned"):
        if key in instances and empirical.get(key) not in (None, instances.get(key)) and empirical.get(key) != instances.get(key):
            mismatch.append(f"B4 {key} does not match instances.json")
    achieved_claimed = claimed_m2
    ok = not skipped and not mismatch
    return _step(
        "B4",
        ok,
        {
            "instance_count": instances.get("instance_count"),
            "n_object": instances.get("n_object"),
            "n_unassigned": instances.get("n_unassigned"),
            "miou": miou,
            "hand_labels": hand_labels,
            "silhouette_iou": silhouette,
            "silhouette_is_not_miou": True,
        },
        mismatch,
        achieved_claimed,
        skipped,
    )


def _step(
    name: str,
    ok: bool,
    numbers: dict[str, Any],
    mismatch: list[str],
    achieved_claimed: bool,
    skipped: list[str],
    **extra: Any,
) -> dict[str, Any]:
    if skipped:
        ok = False
    record = {
        "ok": ok,
        "label": "skipped" if skipped else STEP_LABELS[name],
        "numbers": numbers,
        "mismatch": mismatch,
        "achieved_claimed": achieved_claimed,
        "skipped": skipped,
    }
    record.update(extra)
    return record


def _part(score: dict[str, Any], part_id: str) -> dict[str, Any]:
    for part in score.get("parts") or []:
        if isinstance(part, dict) and part.get("id") == part_id:
            return part
    return {}


def _empirical(part: dict[str, Any]) -> dict[str, Any]:
    empirical = part.get("empirical")
    return empirical if isinstance(empirical, dict) else {}


def _orbit_row(block_a: dict[str, Any], orbit: str) -> dict[str, Any]:
    for row in block_a.get("orbits") or []:
        if isinstance(row, dict) and row.get("orbit") == orbit:
            return row
    return {}


def _metres_per_unit(folder: Path, score: dict[str, Any]) -> float | None:
    score_m1 = score.get("m1") if isinstance(score.get("m1"), dict) else {}
    for source in (
        score_m1,
        score,
        _load(folder / "ingest" / "scale.json"),
        _load(folder / "ingest" / "block_a.json"),
        _load(folder / "ingest" / "ingest_report.json"),
    ):
        metres = _number(source.get("metres_per_unit"))
        method = source.get("scale_method") or source.get("method") or source.get("scale")
        if metres is not None and method not in (None, "", "not_measured", "unmeasured"):
            return metres
    return None


def _silhouette_iou(mesh_score: dict[str, Any], orbit: str) -> float | None:
    if not mesh_score:
        return None
    for row in mesh_score.get("orbits") or []:
        if isinstance(row, dict) and row.get("orbit") == orbit:
            return _number(row.get("iou"))
    return _number(mesh_score.get("iou"))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _npz_is_langsplat(path: Path) -> bool:
    """True only when the npz has a codebook plus three-level CLIP, not the physics vector."""
    if not path.is_file():
        return False
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return "codebook.npy" in names and "clip_features.npy" in names and "levels.npy" in names


def _rel(path: Path) -> str:
    try:
        return path.as_posix()
    except ValueError:
        return str(path)
