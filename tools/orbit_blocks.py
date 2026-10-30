"""Measurements on the existing bottle orbits and the Polycam scan.

This module does not train a splat, does not start COLMAP, and does not
set learned_prior true. A ply is never sent through COLMAP.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.common.paths import resolve_root
from tools.review_contract import write_report
from tools.step01_colmap import evaluate_colmap, parse_analyzer
from tools.step02_splatfacto import count_gaussians, parse_eval, splat_path

BOTTLE_ORBITS = (
    "data/bottle/raw/orbit1.MOV",
    "data/bottle/raw/orbit2.MOV",
    "data/bottle/raw/orbit3.MOV",
)
ORBIT1_CHECKPOINT = "outputs/orbit1/splatfacto/2026-10-09_234601"
_HULL_LIMIT = 5000


def orbit_capture(obj: str, root: Path | None = None) -> dict[str, Any]:
    """Name the existing bottle videos. The scan ply has no orbit."""
    if obj != "bottle":
        return {
            "kind": "polycam_ply",
            "orbit_video": None,
            "existing_capture": True,
            "colmap": "not_called",
            "note": "The camera object is a Polycam ply and has no orbit video. COLMAP was not started.",
        }
    base = resolve_root(root)
    orbits = [
        {"path": relative, "exists": (base / relative).is_file(), "existing_capture": True}
        for relative in BOTTLE_ORBITS
    ]
    return {
        "used_for_eval": BOTTLE_ORBITS[0],
        "checkpoint": ORBIT1_CHECKPOINT,
        "ns_train": "not_run",
        "colmap": "not_rerun",
        "orbits": orbits,
        "existing_capture": True,
    }


def colour_calibration_record(
    exposure_reviewed: bool,
    frames_reviewed: int = 0,
    note: str = "",
    **extra: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "exposure_reviewed": bool(exposure_reviewed),
        "frames_reviewed": int(frames_reviewed),
        "appearance_embeddings_retrained": False,
        "capture": BOTTLE_ORBITS[0],
        "existing_capture": True,
        "note": note
        or (
            "Exposure was reviewed on the existing orbit-1 frames. The splat was not retrained."
            if exposure_reviewed
            else "Exposure was not reviewed. The splat was not retrained."
        ),
    }
    record.update(extra)
    record["appearance_embeddings_retrained"] = False
    return record


def build_ingest_report(obj: str, folder: Path, root: Path | None = None) -> dict[str, Any]:
    """Block A report. PSNR and LPIPS come only from a saved ns-eval file."""
    ply = splat_path(obj, 1, root)
    analyzer = folder / "colmap" / "orbit1" / "analyzer.txt"
    metrics = parse_analyzer(analyzer.read_text(encoding="utf-8")) if analyzer.is_file() else {}
    models_path = folder / "colmap" / "orbit1" / "models.json"
    models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.is_file() else []
    gates = evaluate_colmap(metrics, models if isinstance(models, list) else [], 0) if metrics else {}
    measured = _load_eval(folder)
    psnr = _finite(measured.get("psnr"))
    lpips = _finite(measured.get("lpips"))
    ssim = _finite(measured.get("ssim"))
    scale = _read(folder / "ingest" / "scale.json")
    scale_measured = scale.get("method") not in (None, "", "not_measured") and _finite(scale.get("metres_per_unit")) is not None
    m1 = bool(psnr is not None and psnr > 27.0 and lpips is not None and lpips < 0.20 and scale_measured)
    if psnr is None or lpips is None:
        note = "Existing splat kept. Held-out PSNR and LPIPS were not measured. COLMAP was not rerun."
    elif not scale_measured:
        note = (
            f"Held-out PSNR {psnr} and LPIPS {lpips} are from the existing orbit-1 checkpoint. "
            "Scale is not_measured, so M1 is not met. The splat was not retrained."
        )
    else:
        note = "Held-out metrics and metric scale were measured on the existing capture."
    return {
        "block": "A",
        "capture": orbit_capture(obj, root),
        "splat": str(ply) if ply.is_file() else None,
        "n_gaussians": count_gaussians(ply) if ply.is_file() else None,
        "registered_pct": metrics.get("registered_pct"),
        "mean_reproj_px": metrics.get("mean_reproj_px"),
        "colmap_gates": gates,
        "checkpoint": ORBIT1_CHECKPOINT if obj == "bottle" else None,
        "ns_train": "not_run",
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "m1_met": m1,
        "colour_calibration": _colour(folder),
        "note": note,
    }


def resolve_scene_mean(codebook: dict[str, Any]) -> dict[str, Any]:
    """Outside Gaussians get a scene mean only when that vector was measured separately."""
    vector = codebook.get("vector")
    scene = codebook.get("scene_mean")
    separate = codebook.get("scene_mean_measured") is True and isinstance(scene, list) and len(scene) > 0
    if separate and isinstance(vector, list) and list(scene) == list(vector):
        return {
            "measured": False,
            "reason": "scene_mean matches the one object vector; it was not copied onto the background",
        }
    if separate:
        return {"measured": True, "vector": scene, "reason": "separate scene-mean measurement"}
    if isinstance(vector, list):
        return {
            "measured": False,
            "reason": "one mixed object vector is not a separate scene-mean measurement",
        }
    return {"measured": False, "reason": "no codebook vector"}


def assign_feature_indices(inside: np.ndarray, scene_mean_measured: bool) -> np.ndarray:
    """Label 0 is the object. Label 1 is a separately measured scene mean."""
    mask = np.asarray(inside, dtype=bool).reshape(-1)
    index = np.full(mask.shape, -1, dtype=np.int8)
    index[mask] = 0
    if scene_mean_measured:
        index[~mask] = 1
    return index


def count_floaters(points: np.ndarray, sigma: float = 2.0) -> int:
    """Count points outside the mean ± sigma box of the cropped cloud."""
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 3:
        return 0
    xyz = cloud[:, 0:3]
    center = xyz.mean(axis=0)
    spread = xyz.std(axis=0)
    outside = np.any(np.abs(xyz - center) > (sigma * spread), axis=1)
    return int(outside.sum())


def extend_physics_record(record: dict[str, Any]) -> dict[str, Any]:
    """Fill D2 fields only from numbers already stored. learned_prior cannot become true."""
    per = record.get("per_gaussian")
    if record.get("learned_prior") is True or (isinstance(per, dict) and per.get("learned_prior") is True):
        raise RuntimeError("learned_prior must stay false")
    material = record.get("material")
    mass = record.get("mass_kg")
    if isinstance(per, dict):
        copied = dict(per)
        copied["learned_prior"] = False
        record["per_gaussian"] = copied
    record["learned_prior"] = False
    if "static_friction" not in record and _finite(record.get("friction")) is not None:
        record["static_friction"] = record["friction"]
        derived = dict(record.get("derived") or {})
        derived["static_friction"] = "copied from the existing friction value; not a new measurement"
        record["derived"] = derived
    rigidity = _finite(record.get("rigidity"))
    if "deformable_type" not in record and rigidity is not None and rigidity >= 0.9:
        record["deformable_type"] = "rigid"
        derived = dict(record.get("derived") or {})
        derived["deformable_type"] = "derived from rigidity >= 0.9; not a new model call"
        record["derived"] = derived
    if record.get("drop_test") in (None, ""):
        record["drop_test"] = "not_run"
        record["drop_test_reason"] = "Isaac sim.reset is known to hang; the drop test was not run"
    if material is not None:
        record["material"] = material
    if mass is not None:
        record["mass_kg"] = mass
    return record


def convex_hull_summary(points: np.ndarray, full_cloud_limit: int = _HULL_LIMIT) -> dict[str, Any]:
    """Hull a tiny fixture. A bottle-sized cloud is not hulled."""
    cloud = np.asarray(points, dtype=np.float64)
    count = int(len(cloud))
    if count > full_cloud_limit:
        return {"convex_hull": "not_built", "n_points": count, "reason": "the full cloud was not hulled"}
    import trimesh

    hull = trimesh.convex.convex_hull(cloud)
    return {
        "convex_hull": "built",
        "n_points": count,
        "n_vertices": int(len(hull.vertices)),
        "n_faces": int(len(hull.faces)),
        "full_cloud": False,
    }


def fill_missing_d2(obj: str, root: Path | None, frame: Path | None) -> str:
    """One structured call for missing D2 numbers. Material and mass stay as stored.

    Returns ``model not asked`` when the key or an existing frame is absent.
    The key is never written to disk.
    """
    import os
    import urllib.error
    import urllib.request

    path = resolve_root(root) / "data" / obj / "physics.json"
    record = _read(path)
    if not record:
        return "model not asked"
    material = record.get("material")
    mass = record.get("mass_kg")
    extend_physics_record(record)
    missing = [name for name in ("dynamic_friction", "restitution") if name not in record]
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    record["d2_capture"] = BOTTLE_ORBITS[0] if obj == "bottle" else None
    if frame is not None:
        record["d2_frame"] = str(frame)
    if not missing or not key or frame is None or not frame.is_file():
        record["d2_model"] = "model not asked"
        if material is not None:
            record["material"] = material
        if mass is not None:
            record["mass_kg"] = mass
        _dump(path, record)
        return "model not asked"
    try:
        parsed = _ask_d2(frame.read_bytes(), key, missing)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        record["d2_model"] = "response not usable"
        if material is not None:
            record["material"] = material
        if mass is not None:
            record["mass_kg"] = mass
        _dump(path, record)
        return "response not usable"
    for name, value in parsed.items():
        if name in missing:
            record[name] = value
    record["d2_model"] = "gpt-6-astra"
    record["learned_prior"] = False
    per = dict(record.get("per_gaussian") or {})
    per["learned_prior"] = False
    record["per_gaussian"] = per
    if material is not None:
        record["material"] = material
    if mass is not None:
        record["mass_kg"] = mass
    _dump(path, record)
    return "filled"


def _ask_d2(image: bytes, api_key: str, missing: list[str]) -> dict[str, Any]:
    import urllib.request

    encoded = __import__("base64").b64encode(image).decode("ascii")
    payload = {
        "model": "gpt-6-astra",
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Reply with JSON only for these missing fields of the object in the picture: "
                            + ", ".join(missing)
                            + ". Each value is a number from 0 to 1. Do not include material or mass_kg."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
                ],
            }
        ],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise TypeError("d2 response must be an object")
    kept: dict[str, Any] = {}
    for name in missing:
        if name not in parsed:
            continue
        value = float(parsed[name])
        if value == value and 0.0 <= value <= 1.0:
            kept[name] = value
    return kept


def write_isaac_check(folder: Path, error: str, png: Path | None = None) -> dict[str, Any]:
    """Record one Isaac attempt. A missing file is not a rendered frame."""
    has_frame = png is not None and png.is_file() and png.stat().st_size > 0
    record = {
        "rendered": bool(has_frame),
        "png": str(png) if has_frame else None,
        "error": None if has_frame else error,
        "alignment": "skipped",
        "reason": error,
    }
    _dump(folder / "isaac_check.json", record)
    return record


def apply_measured_records(obj: str, folder: Path, root: Path | None = None) -> None:
    """Refresh records from artifacts that already exist. Does not train."""
    _dump(folder / "ingest" / "ingest_report.json", build_ingest_report(obj, folder, root))
    _keep_unmeasured_scale(folder)
    _refresh_features(folder, root)
    _write_crop_report(obj, folder, root)
    _annotate_physics(folder)
    _write_hull_note(obj, folder)
    _annotate_sisters(obj, folder)
    _record_gpu_minutes(obj, folder, root)
    instances = _read(folder / "semantics" / "instances.json")
    renders = instances.get("review_views") if isinstance(instances.get("review_views"), list) else []
    write_report(
        folder / "report.json",
        "partial",
        {
            "object": obj,
            "learned_prior": False,
            "capture": BOTTLE_ORBITS[0] if obj == "bottle" else "polycam ply; no orbit video",
            "existing_capture": True,
        },
        renders,
        [],
    )


def score_a(obj: str, folder: Path, root: Path | None, ply: Path, count: int | None, colmap_ok: bool) -> dict[str, Any]:
    ingest = _read(folder / "ingest" / "ingest_report.json")
    psnr = _finite(ingest.get("psnr"))
    lpips = _finite(ingest.get("lpips"))
    orbit = "existing capture data/bottle/raw/orbit1.MOV" if obj == "bottle" else "no orbit video"
    if ply.is_file() and psnr is not None and lpips is not None:
        state = "pass" if ingest.get("m1_met") is True else "partial"
        unmet = "scale not_measured, so M1 is not met; splat was not retrained"
        met = "M1 met"
        result = f"{orbit}; splat kept gaussians {count}; PSNR {psnr}; LPIPS {lpips}; {met if state == 'pass' else unmet}"
    elif ply.is_file() and colmap_ok:
        state = "partial"
        result = (
            f"{orbit}; splat kept gaussians {count}; COLMAP not restarted; "
            "PSNR and LPIPS were not measured, so M1 is not met"
        )
    elif ply.is_file():
        state = "partial"
        result = f"{orbit}; splat kept gaussians {count}; metric scale and held-out PSNR were not measured, so M1 is not met"
    else:
        state = "not_started"
        result = "no scene splat"
    return {"state": state, "result": result, "artifact": str(ply)}


def score_c(folder: Path) -> dict[str, Any] | None:
    masks = _read(folder / "masks" / "object.json")
    reports = list((folder / "objects").glob("*/crop_report.json")) if (folder / "objects").is_dir() else []
    if not reports and masks.get("projected") is not True:
        return None
    floater = None
    holes = "not_run"
    if reports:
        crop = _read(reports[0])
        floater = crop.get("floater_count")
        hole_record = crop.get("holes") if isinstance(crop.get("holes"), dict) else {}
        holes = str(hole_record.get("status") or "not_run")
    cropped = any((folder / "objects").glob("*/object.ply")) if (folder / "objects").is_dir() else False
    if masks.get("infill") == "ran" and (folder / "background.ply").is_file():
        return {"state": "pass", "result": "background infill ran", "artifact": "masks/object.json"}
    if masks.get("projected") is True and cropped:
        return {
            "state": "partial",
            "result": (
                f"foreground Gaussians were cropped and recentered; floater_count {floater}; "
                f"holes {holes}; predictive background infill was not run"
            ),
            "artifact": "masks/object.json",
        }
    if masks.get("projected") is True:
        return {
            "state": "partial",
            "result": "object crop exists; predictive background infill was not run",
            "artifact": "masks/object.json",
        }
    return None


def score_g(folder: Path) -> dict[str, Any]:
    sisters = _read(folder / "sisters" / "objects.json")
    objects = sisters.get("objects") if isinstance(sisters.get("objects"), list) else []
    count = len(objects)
    if count >= 9 and sisters.get("clip_gate") is True and _finite(sisters.get("clip_similarity")) is not None:
        state = "pass"
        result = f"{count} sisters passed the similarity gate"
    elif count > 0:
        state = "partial"
        result = f"{count} sisters exist; fewer than 9 or an unmeasured similarity stays partial"
    else:
        state = "not_started"
        result = str(sisters.get("note") or "no sisters")
    return {"state": state, "result": result, "artifact": "sisters/objects.json"}


def _load_eval(folder: Path) -> dict[str, Any]:
    path = folder / "ingest" / "ns_eval.json"
    if not path.is_file():
        return {}
    parsed = parse_eval(path.read_text(encoding="utf-8"))
    return {"psnr": parsed.get("psnr"), "ssim": parsed.get("ssim"), "lpips": parsed.get("lpips")}


def _colour(folder: Path) -> dict[str, Any]:
    saved = _read(folder / "ingest" / "colour_calibration.json")
    if saved:
        saved["appearance_embeddings_retrained"] = False
        return saved
    return colour_calibration_record(False, 0)


def _keep_unmeasured_scale(folder: Path) -> None:
    """Prefer tag / metric depth / polycam / ruler; never invent VLM metres."""
    from tools.metric_depth import is_measured_record, recover_metric_scale

    path = folder / "ingest" / "scale.json"
    existing = _read(path)
    if is_measured_record(existing):
        return
    obj = folder.name
    root = folder.parent.parent if folder.parent.name == "data" else folder.parent
    # recover_metric_scale resolves tag → depth probe → not_measured ScoreOnly.
    try:
        record = recover_metric_scale(obj, root, orbit=1)
    except Exception:
        record = {}
    if is_measured_record(record):
        return
    if record:
        # recover already wrote scale.json; keep capture notes for bottle reports.
        saved = _read(path)
        if saved.get("method") == "not_measured":
            saved.setdefault("capture", BOTTLE_ORBITS[0])
            saved.setdefault("existing_capture", True)
            _dump(path, saved)
        return
    _dump(
        path,
        {
            "metres_per_unit": None,
            "method": "not_measured",
            "reason": "No AprilTag board, metric depth, or known length was recorded for this object.",
            "capture": BOTTLE_ORBITS[0],
            "existing_capture": True,
            "achieved": False,
        },
    )


def _refresh_features(folder: Path, root: Path | None) -> None:
    codebook = _read(folder / "codebook.json")
    mask_path = folder / "masks" / "gaussian_in_mask.npy"
    scene = resolve_scene_mean(codebook)
    if mask_path.is_file() and isinstance(codebook.get("vector"), list):
        index = assign_feature_indices(np.load(mask_path), scene["measured"] is True)
        semantics = folder / "semantics"
        semantics.mkdir(parents=True, exist_ok=True)
        np.save(semantics / "gaussian_feature_idx.npy", index)
        scene_mean = (
            np.asarray(scene["vector"], dtype=np.float32) if scene["measured"] else np.array([], dtype=np.float32)
        )
        np.savez(
            semantics / "features.codebook.npz",
            object_vector=np.asarray(codebook["vector"], dtype=np.float32),
            scene_mean=scene_mean,
        )
    _write_instances(folder, root, scene)


def _write_instances(folder: Path, root: Path | None, scene: dict[str, Any]) -> None:
    masks = _read(folder / "masks" / "object.json")
    codebook = _read(folder / "codebook.json")
    name = str(masks.get("label") or codebook.get("label") or folder.name)
    sources, kind = _review_sources(folder)
    review_views = _publish_reviews(folder, sources) if len(sources) == 4 else []
    record = {
        "block": "B",
        "capture": orbit_capture(folder.name, root),
        "instances": [
            {
                "name": name,
                "foreground": True,
                "n_gaussians": masks.get("n_inside"),
                "detector": masks.get("segmenter"),
                "projected": masks.get("projected") is True,
            }
        ]
        if masks
        else [],
        "background": {
            "n_gaussians": masks.get("n_outside"),
            "feature": "scene_mean" if scene["measured"] else "scene_mean_not_a_separate_measurement",
            "scene_mean_measured": scene["measured"],
            "reason": scene["reason"],
        },
        "codebook": "one object vector" if isinstance(codebook.get("vector"), list) else "missing",
        "langsplat_codebook": False,
        "miou": None,
        "review_kind": kind,
        "review_views": review_views,
    }
    _dump(folder / "semantics" / "instances.json", record)


def _review_sources(folder: Path) -> tuple[list[Path], str]:
    synthetic = [folder / "renders" / "synthetic" / f"view_{index:03d}.png" for index in range(4)]
    if all(path.is_file() for path in synthetic):
        return synthetic, "existing_renders"
    frames = sorted((folder / "frames" / "orbit1").glob("*.png"))
    if len(frames) >= 4:
        return frames[:4], "existing_orbit1_frames"
    return [], "none"


def _publish_reviews(folder: Path, sources: list[Path]) -> list[str]:
    review = folder / "review"
    review.mkdir(parents=True, exist_ok=True)
    written = []
    manifest = []
    for index, source in enumerate(sources):
        destination = review / f"view_{index}.png"
        if destination.resolve() != source.resolve():
            destination.write_bytes(source.read_bytes())
        written.append(str(destination))
        manifest.append({"view": destination.name, "source": str(source), "existing_capture": True})
    _dump(review / "sources.json", {"kind": "copied_existing_files", "views": manifest})
    return written


def _write_crop_report(obj: str, folder: Path, root: Path | None) -> None:
    crops = list((folder / "objects").glob("*/object.ply")) if (folder / "objects").is_dir() else []
    if not crops:
        return
    ply = crops[0]
    xyz = None
    floater_count = None
    try:
        xyz = _ply_xyz(ply)
        floater_count = count_floaters(xyz) if xyz is not None else None
    except (OSError, ValueError):
        floater_count = None
    holes = folder / "background" / "holes"
    hole_files = sorted(holes.glob("*.png")) if holes.is_dir() else []
    hole_record: dict[str, Any]
    if hole_files:
        hole_record = {"status": "present", "n_masks": len(hole_files)}
    else:
        hole_record = {
            "status": "not_run",
            "reason": "training-view alpha was not rendered; hole masks were not written",
        }
    masks = _read(folder / "masks" / "object.json")
    infill = "ran" if masks.get("infill") == "ran" else "skipped"
    m3_met = isinstance(floater_count, int) and floater_count < 50 and infill == "ran"
    report = {
        "block": "C",
        "capture": orbit_capture(obj, root),
        "object": ply.parent.name,
        "n_cropped": None if xyz is None else int(len(xyz)),
        "floater_count": floater_count,
        "floater_rule": "outside 2-sigma of the cropped cloud",
        "holes": hole_record,
        "infill": infill,
        "fine_tune": "not_run",
        "lama": "not_installed",
        "diffusers": "not_installed",
        "m3_met": m3_met,
        "note": "Infill stayed skipped. LaMa and diffusers were not installed, and the 500-iteration fine-tune was not run.",
    }
    _dump(ply.parent / "crop_report.json", report)


def _annotate_physics(folder: Path) -> None:
    path = folder / "physics.json"
    if not path.is_file():
        return
    record = _read(path)
    if not record:
        return
    extend_physics_record(record)
    _dump(path, record)


def _write_hull_note(obj: str, folder: Path) -> None:
    candidates = list((folder / "objects").glob("*/object.ply")) if (folder / "objects").is_dir() else []
    scene = folder / "colmap" / "orbit1" / "splat.ply"
    if scene.is_file():
        candidates.append(scene)
    if not candidates:
        return
    count = count_gaussians(candidates[0]) or 0
    record: dict[str, Any] = {
        "convex_hull": "not_built",
        "n_points": count,
        "reason": "the full cloud was not hulled" if count > _HULL_LIMIT else "no tiny sample was requested for this object",
        "chamfer_cm": None,
        "f1": None,
        "mesh_score_file": False,
        "joints_actuated": False,
    }
    for name in ("mesh_score.json", "chamfer.json"):
        loaded = _read(folder / name)
        if loaded:
            record["chamfer_cm"] = loaded.get("chamfer_cm")
            record["f1"] = loaded.get("f1")
            record["mesh_score_file"] = True
            record["mesh_score_source"] = name
            break
    record["capture"] = orbit_capture(obj, None if obj not in {"bottle", "scan"} else _root_from_folder(folder))
    _dump(folder / "convex_hull.json", record)


def _annotate_sisters(obj: str, folder: Path) -> None:
    path = folder / "sisters" / "objects.json"
    if path.is_file():
        record = _read(path)
        objects = record.get("objects") if isinstance(record.get("objects"), list) else []
        record["n"] = len(objects)
        if record.get("clip_gate") is not True:
            record["clip_gate"] = "not_measured"
            record["clip_similarity"] = None
        if obj == "bottle":
            record.setdefault("note", "bottle sisters were not started" if not objects else "existing sisters kept")
            record["capture"] = {"path": BOTTLE_ORBITS[0], "existing_capture": True}
        _dump(path, record)
        return
    record = {
        "objects": [],
        "n": 0,
        "clip_gate": "not_measured",
        "clip_similarity": None,
        "generator": None,
        "note": "bottle sisters were not started" if obj == "bottle" else "sisters were not started",
        "existing_capture": True,
    }
    if obj == "bottle":
        record["capture"] = {"path": BOTTLE_ORBITS[0], "existing_capture": True, "also": list(BOTTLE_ORBITS)}
    _dump(path, record)


def _record_gpu_minutes(obj: str, folder: Path, root: Path | None) -> None:
    minutes = _finite(_read(folder / "ingest" / "ns_eval_minutes.json").get("gpu_minutes"))
    if minutes is None or minutes < 0:
        return
    from tools.common.status import load_status, save_status

    doc = load_status(obj, root, create=True)
    doc["gpu_minutes"] = minutes
    save_status(obj, doc, root)


def _ply_xyz(path: Path) -> np.ndarray | None:
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
            if len(parts) >= 3 and parts[1] == "float":
                names.append(parts[2])
    if count <= 0 or not {"x", "y", "z"} <= set(names):
        return None
    width = len(names)
    table = np.frombuffer(payload, dtype="<f4", count=count * width).reshape(count, width)
    columns = [names.index(axis) for axis in ("x", "y", "z")]
    return np.asarray(table[:, columns], dtype=np.float64)


def _root_from_folder(folder: Path) -> Path | None:
    if folder.parent.name == "data":
        return folder.parent.parent
    return None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _dump(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
