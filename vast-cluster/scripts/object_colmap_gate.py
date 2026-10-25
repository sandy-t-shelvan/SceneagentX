#!/usr/bin/env python3
"""Gate one orbit before splatfacto.

Registered percent is images in the chosen sparse model divided by frames
extracted from the video. model_analyzer's own image count is only the
images inside that model, so a 2-image fragment would otherwise read as
100 percent. Mean reprojection comes from ``colmap model_analyzer`` on
the chosen model. Thresholds are REGISTERED_MIN and REPROJ_MAX_PX in
tools/step01_colmap.py.

Exit 0 when both gates pass, 4 when they were measured and failed (the
caller must not train), 1 when the model or the analyzer could not be read.
Writes data/<object>/debug/colmap_gate_<orbit>.json. Refuses the bottle.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step01_colmap import REGISTERED_MIN, REPROJ_MAX_PX, parse_analyzer

REPO = Path("/root/SceneagentX")
NONPHYSICAL_REPROJ_PX = 1_000_000.0


def mean_reproj_from_errors(errors: list[float]) -> tuple[float | None, int]:
    """Mean of point errors. Values above one million pixels are not a residual."""
    kept: list[float] = []
    excluded = 0
    for err in errors:
        if not math.isfinite(err) or err > NONPHYSICAL_REPROJ_PX:
            excluded += 1
            continue
        kept.append(err)
    if not kept:
        return None, excluded
    return sum(kept) / len(kept), excluded


def _point_errors(model: Path) -> list[float]:
    from nerfstudio.data.utils.colmap_parsing_utils import read_points3D_binary

    points = read_points3D_binary(model / "points3D.bin")
    return [float(point.error) for point in points.values()]


def reject_object(obj: str) -> None:
    if obj != Path(obj).name or obj in {"", ".", "..", "bottle"}:
        raise SystemExit(f"refusing object name {obj}")


def registered_percent(registered_images: int, extracted_frames: int) -> float | None:
    if extracted_frames <= 0 or registered_images < 0:
        return None
    return 100.0 * registered_images / extracted_frames


def gate_passed(registered_pct: float | None, mean_reproj_px: float | None) -> bool:
    registered_ok = registered_pct is not None and registered_pct >= REGISTERED_MIN
    reproj_ok = mean_reproj_px is not None and mean_reproj_px < REPROJ_MAX_PX
    return registered_ok and reproj_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record COLMAP registration gates for one orbit")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--orbit", required=True)
    parser.add_argument("--root", type=Path, default=REPO)
    args = parser.parse_args(argv)
    reject_object(args.obj)
    if args.orbit != Path(args.orbit).name or not args.orbit.startswith("orbit"):
        raise SystemExit(f"refusing orbit name {args.orbit}")
    out_dir = args.root / "data" / args.obj / "colmap" / args.orbit
    debug = args.root / "data" / args.obj / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    select_path = out_dir / "model_select.json"
    if not select_path.is_file():
        sys.stderr.write(f"missing {select_path}\n")
        return 1
    select = json.loads(select_path.read_text(encoding="utf-8"))
    chosen = Path(str(select.get("chosen_model") or ""))
    if not chosen.is_dir():
        sys.stderr.write(f"chosen model is not a directory: {chosen}\n")
        return 1
    extracted = int(select.get("extracted_frames") or 0)
    registered = int(select.get("chosen_registered_images") or 0)
    proc = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(chosen)],
        check=False,
        capture_output=True,
        text=True,
    )
    analyzer_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    analyzer_path = debug / f"colmap_analyzer_{args.orbit}.txt"
    analyzer_path.write_text(analyzer_text, encoding="utf-8")
    metrics = parse_analyzer(analyzer_text)
    registered_pct = registered_percent(registered, extracted)
    analyzer_mean = metrics.get("mean_reproj_px")
    mean_reproj = analyzer_mean
    points_mean = None
    points_excluded = 0
    reproj_source = "colmap model_analyzer"
    analyzer_unusable = analyzer_mean is None or not math.isfinite(analyzer_mean) or analyzer_mean >= 100.0
    if analyzer_unusable:
        try:
            points_mean, points_excluded = mean_reproj_from_errors(_point_errors(chosen))
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write(f"points3D reprojection unread: {exc}\n")
            points_mean = None
        if points_mean is not None and points_excluded > 0:
            mean_reproj = points_mean
            reproj_source = (
                "mean of points3D.error excluding non-physical values; "
                "model_analyzer mean was non-physical"
            )
    passed = gate_passed(registered_pct, mean_reproj)
    if registered_pct is None:
        reason = "registered percent could not be computed"
    elif mean_reproj is None:
        reason = "mean reprojection could not be read from colmap model_analyzer"
    elif registered_pct < REGISTERED_MIN and (mean_reproj >= REPROJ_MAX_PX):
        reason = f"registered_pct {registered_pct:.4f} is under {REGISTERED_MIN} and mean reprojection {mean_reproj} px is at least {REPROJ_MAX_PX}"
    elif registered_pct < REGISTERED_MIN:
        reason = f"registered_pct {registered_pct:.4f} is under {REGISTERED_MIN}"
    elif mean_reproj >= REPROJ_MAX_PX:
        reason = f"mean reprojection {mean_reproj} px is at least {REPROJ_MAX_PX}"
    else:
        reason = ""
    record = {
        "object": args.obj,
        "orbit": args.orbit,
        "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chosen_model": str(chosen),
        "registered_images": registered,
        "extracted_frames": extracted,
        "registered_pct": registered_pct,
        "analyzer_registered_pct": metrics.get("registered_pct"),
        "mean_reproj_px": mean_reproj,
        "analyzer_mean_reproj_px": analyzer_mean,
        "points_mean_reproj_px": points_mean,
        "points_excluded_nonphysical": points_excluded,
        "mean_reproj_source": reproj_source,
        "registered_min": REGISTERED_MIN,
        "reproj_max_px": REPROJ_MAX_PX,
        "model_analyzer_exit": proc.returncode,
        "analyzer_log": str(analyzer_path),
        "passed": passed,
        "train": "allowed" if passed else "skipped",
        "reason": reason,
    }
    gate_path = debug / f"colmap_gate_{args.orbit}.json"
    gate_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        f"GATE {args.orbit} registered_pct={registered_pct} mean_reproj_px={mean_reproj} "
        f"passed={str(passed).lower()} train={record['train']}",
        flush=True,
    )
    print(f"WROTE {gate_path}", flush=True)
    if proc.returncode != 0 and mean_reproj is None:
        sys.stderr.write(f"colmap model_analyzer exit {proc.returncode}\n")
        return 1
    if not passed:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
