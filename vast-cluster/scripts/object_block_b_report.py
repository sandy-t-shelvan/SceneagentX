#!/usr/bin/env python3
"""Write data/<object>/ingest/block_b.json from finished Block B artifacts.

Does not train. Does not invent metres_per_unit, mIoU, or a colour review.
LangSplat is measured only when features.codebook.npz contains codebook.npy,
clip_features.npy, and levels.npy. hand_labels stays 0 and miou stays null.
One named instance is not mIoU. M1 and M2 stay unmet.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ORBITS = ("orbit1", "orbit2", "orbit3")
REPO = Path("/root/SceneagentX")


def reject_object(obj: str) -> None:
    if obj != Path(obj).name or obj in {"", ".", "..", "bottle"}:
        raise SystemExit(f"refusing object name {obj}")


def _load(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _dump(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def langsplat_npz_measured(path: Path) -> bool:
    """True only for the three LangSplat arrays. A physics vector is not this file."""
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return "codebook.npy" in names and "clip_features.npy" in names and "levels.npy" in names


def _feature_state(folder: Path, orbit: str) -> tuple[str, dict]:
    if orbit == "orbit1":
        sidecar = folder / "semantics" / "feature_field.json"
        refused = folder / "semantics" / "feature_field_refused.json"
        features = folder / "semantics" / "gaussian_features.npy"
    else:
        sidecar = folder / "semantics" / orbit / "feature_field.json"
        refused = folder / "semantics" / orbit / "feature_field_refused.json"
        features = folder / "semantics" / orbit / "gaussian_features.npy"
    side = _load(sidecar)
    refuse = _load(refused)
    if side.get("feature_field") == "measured" and features.is_file() and side.get("splat_retrained") is not True:
        return "trained", side
    if refuse.get("feature_field") == "refused" or refused.is_file():
        return "refused", refuse or side
    if features.is_file():
        return "trained", side
    return "unstarted", side or refuse


def _instances(folder: Path, orbit: str) -> dict:
    path = folder / "semantics" / orbit / "instances.json"
    record = _load(path)
    if orbit == "orbit1" and not record:
        record = _load(folder / "semantics" / "instances.json")
    return record


def _review(folder: Path, orbit: str) -> dict:
    path = folder / "reviews" / f"b4_{orbit}_instance.json"
    review = _load(path)
    if review:
        return review
    instances = _instances(folder, orbit)
    nested = instances.get("review") if isinstance(instances.get("review"), dict) else {}
    return nested


def _orbit_score(folder: Path, orbit: str, block_a: dict, langsplat: bool) -> dict:
    obj = _load(folder / "masks" / orbit / "object.json")
    first = _load(folder / "ingest" / "b1_first" / orbit / "object.json")
    if not first:
        first = obj
    n_views = first.get("n_views")
    n_det = first.get("n_views_with_detection")
    rate = None
    if isinstance(n_views, int) and n_views > 0 and isinstance(n_det, int):
        rate = n_det / n_views
    span_in = _finite(obj.get("span_inside"))
    span_sc = _finite(obj.get("span_scene"))
    ratio = span_in / span_sc if span_in is not None and span_sc not in (None, 0) else None
    field, field_rec = _feature_state(folder, orbit)
    instances = _instances(folder, orbit)
    review = _review(folder, orbit)
    a_row = {}
    for row in block_a.get("orbits") or []:
        if isinstance(row, dict) and row.get("orbit") == orbit:
            a_row = row
            break
    names = [row.get("name") for row in instances.get("instances") or [] if isinstance(row, dict)]
    return {
        "block": "B",
        "object": folder.name,
        "orbit": orbit,
        "state": "measured",
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "m2_reason": "no 20 hand-labelled views; miou null; one named instance is not mIoU",
        "hand_labels": 0,
        "miou": None,
        "colour_review": "unrun",
        "metres_per_unit": None,
        "splatfacto_trained_this_run": False,
        "block_a": {
            "psnr": a_row.get("psnr"),
            "lpips": a_row.get("lpips"),
            "ssim": a_row.get("ssim"),
            "registered_pct": a_row.get("registered_pct"),
            "gate_passed": a_row.get("gate_passed"),
            "train_finished": a_row.get("train_finished"),
            "splatfacto_trained_this_run": False,
        },
        "parts": [
            {
                "id": "B1",
                "state": "kept" if first else "missing",
                "achieved": False,
                "prompt": first.get("prompt"),
                "requested_prompt": "vaseline." if folder.name == "vaseline" else None,
                "b1_rerun": False,
                "first_measurement": f"data/{folder.name}/ingest/b1_first/{orbit}/object.json",
                "n_views": n_views,
                "n_views_with_detection": n_det,
                "detection_rate": rate,
                "detections": first.get("detections"),
                "largest_frame": first.get("largest_frame"),
                "largest_foreground_pixels": first.get("largest_foreground_pixels"),
                "hand_labels": 0,
                "miou": None,
            },
            {
                "id": "B2",
                "state": "measured" if langsplat else "unstarted",
                "achieved": False,
                "langsplat_measured": langsplat,
                "file": f"data/{folder.name}/semantics/features.codebook.npz",
                "required_arrays": ["codebook.npy", "clip_features.npy", "levels.npy"],
                "physics_vector_renamed": False,
                "physics_vector_file": f"data/{folder.name}/codebook.json",
                "physics_vector_present": (folder / "codebook.json").is_file(),
            },
            {
                "id": "B3",
                "state": "measured" if obj.get("projected") is True else "unstarted",
                "achieved": False,
                "projected": obj.get("projected") is True,
                "covers_room": obj.get("covers_room"),
                "span_inside": span_in,
                "span_scene": span_sc,
                "span_ratio": ratio,
                "n_inside": obj.get("n_inside"),
                "n_gaussians": obj.get("n_gaussians"),
                "feature_field": field,
                "feature_field_reason": field_rec.get("reason"),
                "splat_retrained": False,
            },
            {
                "id": "B4",
                "state": "measured" if instances else "unstarted",
                "achieved": False,
                "m2_met": False,
                "hand_labels": 0,
                "miou": None,
                "instance_count": instances.get("instance_count"),
                "instance_names": names,
                "named_instance_is_not_miou": True,
                "verdict": review.get("verdict"),
                "review_asked": review.get("asked") is True,
                "review_reason": review.get("reason"),
                "problem": review.get("problem"),
            },
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write Block B scores from artifacts")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args(argv)
    reject_object(args.obj)
    folder = args.root / "data" / args.obj
    block_a = _load(folder / "ingest" / "block_a.json")
    npz = folder / "semantics" / "features.codebook.npz"
    langsplat = langsplat_npz_measured(npz)
    orbits = [_orbit_score(folder, orbit, block_a, langsplat) for orbit in ORBITS]
    report = {
        "block": "B",
        "object": args.obj,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "hand_labels": 0,
        "miou": None,
        "metres_per_unit": None,
        "colour_review": "unrun",
        "langsplat_measured": langsplat,
        "note": (
            "A skipped check is not a pass. M1 stays unmet while metres_per_unit is null. "
            "M2 stays unmet while hand_labels is 0 and miou is null. "
            "One instance named after the object is not mIoU. "
            "The physics vector was not renamed."
        ),
        "orbits": orbits,
    }
    dest = folder / "ingest" / "block_b.json"
    _dump(dest, report)
    for row in orbits:
        _dump(folder / "ingest" / f"block_b_{row['orbit']}.json", row)
    print(f"WROTE {dest}", flush=True)
    summary = []
    for row in orbits:
        parts = {part["id"]: part for part in row["parts"]}
        summary.append(
            {
                "orbit": row["orbit"],
                "detection_rate": parts["B1"]["detection_rate"],
                "langsplat_measured": langsplat,
                "covers_room": parts["B3"]["covers_room"],
                "span_ratio": parts["B3"]["span_ratio"],
                "feature_field": parts["B3"]["feature_field"],
                "b4_verdict": parts["B4"]["verdict"],
                "miou": None,
                "hand_labels": 0,
                "m1_met": False,
                "m2_met": False,
            }
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
