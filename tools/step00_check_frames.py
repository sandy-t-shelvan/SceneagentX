"""B1 capture check.

Extract frames at 4 fps, then record Laplacian variance, mean luminance,
and tag36h11 detections. Real captures must pass the plan gates. The
--demo flag runs the same code on a short synthetic orbit.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import statistics
from typing import Any

import cv2
import numpy as np

from tools.common.paths import frames_dir, load_capture_config, raw_dir, renders_dir, reports_dir
from tools.common.status import begin_step, finish_step, set_metric

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def demo_thresholds(base: dict[str, Any] | None = None) -> dict[str, Any]:
    thresholds = dict(base or load_capture_config())
    # A synthetic orbit is only long enough to prove the checker runs.
    thresholds["frame_count_min"] = 1
    thresholds["frame_count_max"] = 10000
    thresholds["orbits"] = [1]
    return thresholds


class TagDetector:
    def __init__(self) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self._dictionary = dictionary
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
        else:
            self._detector = None
            self._parameters = cv2.aruco.DetectorParameters_create()

    def found(self, gray: np.ndarray) -> bool:
        if self._detector is not None:
            _corners, ids, _rejected = self._detector.detectMarkers(gray)
        else:
            _corners, ids, _rejected = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._parameters
            )
        return ids is not None and len(ids) > 0


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def find_raw_orbit(obj: str, orbit: int, root: Path | None = None) -> Path | None:
    raw = raw_dir(obj, root)
    folder = raw / f"orbit{orbit}"
    if folder.is_dir() and (list_images(folder) or _videos_in(folder)):
        return folder
    if not raw.is_dir():
        return None
    for path in sorted(raw.iterdir()):
        if path.is_file() and path.stem.lower() == f"orbit{orbit}" and path.suffix.lower() in VIDEO_SUFFIXES:
            return path
    return None


def _videos_in(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )


def extract_video(video: Path, dest: Path, extract_fps: float) -> list[Path]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video {video}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = extract_fps
    stride = max(int(round(source_fps / extract_fps)), 1)
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    index = 0
    kept = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                kept += 1
                path = dest / f"frame_{kept:06d}.jpg"
                if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    raise RuntimeError(f"Could not write {path}")
                written.append(path)
            index += 1
    finally:
        capture.release()
    return written


def copy_images(images: list[Path], dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for kept, source in enumerate(images, start=1):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        path = dest / f"frame_{kept:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            raise RuntimeError(f"Could not write {path}")
        written.append(path)
    return written


def clear_frames(dest: Path) -> None:
    if not dest.exists():
        return
    for path in dest.iterdir():
        if path.is_file():
            path.unlink()


def prepare_frames(obj: str, orbit: int, extract_fps: float, root: Path | None = None) -> list[Path]:
    source = find_raw_orbit(obj, orbit, root)
    dest = frames_dir(obj, orbit, root)
    if source is None:
        return []
    clear_frames(dest)
    if source.is_file():
        return extract_video(source, dest, extract_fps)
    images = list_images(source)
    if images:
        return copy_images(images, dest)
    videos = _videos_in(source)
    if len(videos) != 1:
        raise RuntimeError(
            f"{source} needs one video or a folder of images, found {len(videos)} videos"
        )
    return extract_video(videos[0], dest, extract_fps)


def measure_frame(path: Path, detector: TagDetector) -> dict[str, Any]:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return {"file": path.name, "laplacian": 0.0, "luminance": 0.0, "tag": False, "readable": False}
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    luminance = float(gray.mean())
    return {
        "file": path.name,
        "laplacian": laplacian,
        "luminance": luminance,
        "tag": bool(detector.found(gray)),
        "readable": True,
    }


def summarize(rows: list[dict[str, Any]], low_cutoff: float) -> dict[str, float]:
    if not rows:
        return {
            "n_frames": 0,
            "laplacian_median": 0.0,
            "frac_below": 1.0,
            "luminance_mean": 0.0,
            "luminance_std": 0.0,
            "tag_fraction": 0.0,
        }
    laplacians = [float(row["laplacian"]) for row in rows]
    luminances = [float(row["luminance"]) for row in rows]
    tags = [1.0 if row["tag"] else 0.0 for row in rows]
    luminance_mean = float(statistics.fmean(luminances))
    luminance_std = float(statistics.pstdev(luminances)) if len(luminances) > 1 else 0.0
    below = sum(1 for value in laplacians if value < low_cutoff)
    return {
        "n_frames": len(rows),
        "laplacian_median": float(statistics.median(laplacians)),
        "frac_below": below / len(rows),
        "luminance_mean": luminance_mean,
        "luminance_std": luminance_std,
        "tag_fraction": float(statistics.fmean(tags)),
    }


def evaluate_gates(stats: dict[str, float], thresholds: dict[str, Any]) -> dict[str, bool]:
    mean = stats["luminance_mean"]
    ratio = thresholds["luminance_std_ratio_max"]
    exposure = mean > 0 and stats["luminance_std"] < ratio * mean
    return {
        "frame_count": thresholds["frame_count_min"] <= stats["n_frames"] <= thresholds["frame_count_max"],
        "sharpness": (
            stats["laplacian_median"] > thresholds["laplacian_median_min"]
            and stats["frac_below"] < thresholds["low_sharp_fraction_max"]
        ),
        "tag": stats["tag_fraction"] >= thresholds["tag_fraction_min"],
        "exposure": exposure,
    }


def write_contact_sheet(frames: list[Path], dest: Path, every: int) -> Path | None:
    chosen = [frame for index, frame in enumerate(frames) if index % every == 0]
    if not chosen:
        return None
    tiles = []
    for frame_path in chosen:
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        tile = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
        cv2.putText(
            tile,
            frame_path.stem.replace("frame_", ""),
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            frame_path.stem.replace("frame_", ""),
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    if not tiles:
        return None
    columns = min(8, len(tiles))
    rows_n = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows_n * 120, columns * 160, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, columns)
        sheet[r * 120 : (r + 1) * 120, c * 160 : (c + 1) * 160] = tile
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dest), sheet):
        raise RuntimeError(f"Could not write contact sheet {dest}")
    return dest


def count_stills(obj: str, root: Path | None = None) -> int:
    stills = raw_dir(obj, root) / "stills"
    if not stills.is_dir():
        return 0
    return sum(1 for path in stills.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES)


def orbit_report(
    obj: str,
    orbit: int,
    thresholds: dict[str, Any],
    root: Path | None = None,
    detector: TagDetector | None = None,
) -> dict[str, Any]:
    frames = prepare_frames(obj, orbit, float(thresholds["extract_fps"]), root)
    checker = detector or TagDetector()
    rows = [measure_frame(path, checker) for path in frames]
    stats = summarize(rows, float(thresholds["low_sharp_cutoff"]))
    gates = evaluate_gates(stats, thresholds)
    sheet = renders_dir(obj, "step00", root) / f"orbit{orbit}_sheet.png"
    sheet_path = write_contact_sheet(frames, sheet, int(thresholds["sheet_every"]))
    report_path = reports_dir(obj, root) / f"step00_orbit{orbit}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "orbit": orbit,
        "source": str(find_raw_orbit(obj, orbit, root) or ""),
        "n_frames": int(stats["n_frames"]),
        "laplacian_median": stats["laplacian_median"],
        "frac_below": stats["frac_below"],
        "luminance_mean": stats["luminance_mean"],
        "luminance_std": stats["luminance_std"],
        "tag_fraction": stats["tag_fraction"],
        "tag_family": thresholds["tag_family"],
        "gates": gates,
        "passed": all(gates.values()) and stats["n_frames"] > 0,
        "contact_sheet": str(sheet_path) if sheet_path else "",
        "missing_raw": find_raw_orbit(obj, orbit, root) is None,
    }
    if report["missing_raw"]:
        report["passed"] = False
        report["gates"] = {name: False for name in gates}
    payload = dict(report)
    payload["frames"] = rows
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return report


def format_orbit(report: dict[str, Any]) -> str:
    gates = report["gates"]
    failed = [name for name, ok in gates.items() if not ok]
    gate_text = "pass" if report["passed"] else "fail " + ",".join(failed)
    ratio = 0.0
    if report["luminance_mean"]:
        ratio = report["luminance_std"] / report["luminance_mean"]
    return (
        f"orbit {report['orbit']}: frames {report['n_frames']}  "
        f"lap_med {report['laplacian_median']:.1f}  "
        f"low {report['frac_below']:.3f}  "
        f"tag {report['tag_fraction']:.3f}  "
        f"lum_std/mean {ratio:.4f}  {gate_text}"
    )


def check_object(
    obj: str,
    root: Path | None = None,
    orbits: list[int] | None = None,
    thresholds: dict[str, Any] | None = None,
    profile: str = "plan",
) -> int:
    config = dict(thresholds or load_capture_config())
    required = orbits or list(config["orbits"])
    begin_step(obj, "B1", root)
    detector = TagDetector()
    reports = [orbit_report(obj, orbit, config, root, detector) for orbit in required]
    passed = all(item["passed"] for item in reports)
    stills = count_stills(obj, root)
    summary = {
        "profile": profile,
        "passed": passed,
        "n_stills": stills,
        "stills_expected": config.get("stills_expected", 5),
        "orbits": {f"orbit{item['orbit']}": _public_orbit(item) for item in reports},
    }
    set_metric(obj, "step00", summary, root)
    finish_step(
        obj,
        "B1",
        "pass" if passed else "fail",
        root,
        actual={"passed": passed, "n_stills": stills, "profile": profile},
    )
    for item in reports:
        sys.stdout.write(format_orbit(item) + "\n")
    if stills < int(config.get("stills_expected", 5)):
        sys.stdout.write(
            f"stills: {stills} (plan asks for {config.get('stills_expected', 5)}; not part of the B1 numeric gates)\n"
        )
    sys.stdout.write(f"B1: {'pass' if passed else 'fail'}  profile {profile}\n")
    sys.stdout.flush()
    return 0 if passed else 1


def _public_orbit(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "frames"}


def _with_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    updated = dict(config)
    if args.min_frames is not None:
        updated["frame_count_min"] = args.min_frames
    if args.max_frames is not None:
        updated["frame_count_max"] = args.max_frames
    if args.min_tag_fraction is not None:
        updated["tag_fraction_min"] = args.min_tag_fraction
    if args.laplacian_median_min is not None:
        updated["laplacian_median_min"] = args.laplacian_median_min
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check captured orbits for the SceneAgent marker pipeline")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--orbit", type=int, action="append", dest="orbits")
    parser.add_argument("--demo", action="store_true", help="Run on a synthetic orbit under data/_demo")
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--min-tag-fraction", type=float, default=None)
    parser.add_argument("--laplacian-median-min", type=float, default=None)
    args = parser.parse_args(argv)
    try:
        if args.demo:
            from tools.synthetic import write_orbit

            raw = raw_dir("_demo", args.root) / "orbit1"
            write_orbit(raw, n=8)
            return check_object(
                "_demo",
                args.root,
                orbits=[1],
                thresholds=demo_thresholds(),
                profile="demo",
            )
        thresholds = _with_overrides(load_capture_config(), args)
        return check_object(args.obj, args.root, orbits=args.orbits, thresholds=thresholds, profile="plan")
    except (RuntimeError, OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    raise SystemExit(main())
