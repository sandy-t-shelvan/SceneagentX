"""B2 COLMAP wrapper for one orbit.

Calls ns-process-data when it is on PATH, then colmap model_analyzer.
Tests pass a saved analyzer log and skip both binaries. The numeric gates
are the plan's: at least 90% registered, mean reprojection under 1 px,
and a single sparse model named 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import re
import shutil
from typing import Any

from tools.check_env import CmdResult, Runner, default_runner
from tools.common.paths import frames_dir, object_dir, raw_dir, renders_dir
from tools.common.status import StatusError, begin_step, finish_step, load_status, save_status, set_path
from tools.viz_cameras import plot_cameras, read_images_txt

REGISTERED_MIN = 80.0
REPROJ_MAX_PX = 1.0


def colmap_dir(obj: str, orbit: int, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "colmap" / f"orbit{orbit}"


def sparse_dir(obj: str, orbit: int, root: Path | None = None) -> Path:
    return colmap_dir(obj, orbit, root) / "sparse"


def parse_analyzer(text: str) -> dict[str, float | None]:
    """Pull registration and reprojection out of a COLMAP or nerfstudio log."""
    registered = _first_float(
        text,
        r"registered[^\d]{0,40}(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:images\s+)?registered",
    )
    num_images = _first_float(text, r"num_images:\s*(\d+)", r"Images:\s*(\d+)")
    num_reg = _first_float(
        text,
        r"num_reg_images:\s*(\d+)",
        r"Registered images:\s*(\d+)",
    )
    if registered is None and num_images and num_reg is not None and num_images > 0:
        registered = 100.0 * num_reg / num_images
    reproj = _first_float(
        text,
        r"mean_reprojection_error:\s*(\d+(?:\.\d+)?)",
        r"Mean reprojection error:\s*(\d+(?:\.\d+)?)",
        r"reprojection[^\d]{0,20}(\d+(?:\.\d+)?)\s*px",
    )
    return {"registered_pct": registered, "mean_reproj_px": reproj}


def _first_float(text: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def model_names(sparse: Path) -> list[str]:
    if not sparse.is_dir():
        return []
    names = [path.name for path in sparse.iterdir() if path.is_dir()]
    return sorted(names)


def evaluate_colmap(
    metrics: dict[str, float | None],
    models: list[str],
    far_cameras: int = 0,
) -> dict[str, bool]:
    registered = metrics.get("registered_pct")
    reproj = metrics.get("mean_reproj_px")
    return {
        "registered": registered is not None and registered >= REGISTERED_MIN,
        "reprojection": reproj is not None and reproj < REPROJ_MAX_PX,
        "one_model": models == ["0"],
        "cameras": far_cameras == 0,
    }


def _video_or_frames(obj: str, orbit: int, root: Path | None) -> tuple[str, Path] | None:
    raw = raw_dir(obj, root)
    for suffix in (".mp4", ".mov", ".avi"):
        video = raw / f"orbit{orbit}{suffix}"
        if video.is_file():
            return "video", video
    frames = frames_dir(obj, orbit, root)
    if frames.is_dir() and any(frames.iterdir()):
        return "images", frames
    image_dir = raw / f"orbit{orbit}"
    if image_dir.is_dir() and any(image_dir.iterdir()):
        return "images", image_dir
    return None


def _run_process(
    kind: str,
    source: Path,
    output: Path,
    runner: Runner,
) -> CmdResult:
    if runner is default_runner and shutil.which("ns-process-data") is None:
        return CmdResult(127, "", "ns-process-data is not on PATH. Run this in the nerf env on the GPU box.")
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ns-process-data",
        kind,
        "--data",
        str(source),
        "--output-dir",
        str(output),
    ]
    # Video frames match their neighbors. The vocab-tree matcher is the wrong
    # retry: nerfstudio's tree is legacy flann, and COLMAP 4.1 expects faiss.
    if kind == "images":
        cmd.extend(["--matching-method", "sequential"])
    return runner(cmd, 60 * 60)


def _run_analyzer(model: Path, runner: Runner) -> CmdResult:
    if runner is default_runner and shutil.which("colmap") is None:
        return CmdResult(127, "", "colmap is not on PATH. Run this in the nerf env on the GPU box.")
    return runner(["colmap", "model_analyzer", "--path", str(model)], 10 * 60)


def run_orbit(
    obj: str,
    orbit: int,
    root: Path | None = None,
    runner: Runner | None = None,
    analyzer_text: str | None = None,
    mark_block: bool = True,
) -> int:
    """Process one orbit and write step01.orbitN. Returns 0 when the gates pass."""
    active = runner or default_runner
    owns_block = mark_block and orbit == 1
    if owns_block:
        begin_step(obj, "B2", root)
    source = _video_or_frames(obj, orbit, root)
    output = colmap_dir(obj, orbit, root)
    log_parts: list[str] = []
    if analyzer_text is None:
        if source is None:
            return _finish(obj, orbit, root, owns_block, False, {"error": "missing raw footage"}, {})
        kind, path = source
        processed = _run_process(kind, path, output, active)
        log_parts.append(processed.stdout)
        log_parts.append(processed.stderr)
        if processed.returncode != 0:
            text = "\n".join(log_parts)
            return _finish(
                obj,
                orbit,
                root,
                owns_block,
                False,
                {"error": text.strip() or "ns-process-data failed"},
                parse_analyzer(text),
            )
        model = sparse_dir(obj, orbit, root) / "0"
        analyzed = _run_analyzer(model, active)
        log_parts.append(analyzed.stdout)
        log_parts.append(analyzed.stderr)
        if analyzed.returncode != 0:
            text = "\n".join(log_parts)
            return _finish(
                obj,
                orbit,
                root,
                owns_block,
                False,
                {"error": text.strip() or "colmap model_analyzer failed"},
                parse_analyzer(text),
            )
        analyzer_text = "\n".join(log_parts)
    metrics = parse_analyzer(analyzer_text)
    models = model_names(sparse_dir(obj, orbit, root))
    images = sparse_dir(obj, orbit, root) / "0" / "images.txt"
    far = 0
    if images.is_file():
        centers = read_images_txt(images)
        plot_path = renders_dir(obj, "step01", root) / f"orbit{orbit}_cameras.png"
        far = plot_cameras(centers, plot_path)
    gates = evaluate_colmap(metrics, models, far)
    passed = all(gates.values())
    record = {
        "registered_pct": metrics.get("registered_pct"),
        "mean_reproj_px": metrics.get("mean_reproj_px"),
        "n_models": len(models),
        "models": models,
        "far_cameras": far,
        "gates": gates,
        "passed": passed,
    }
    return _finish(obj, orbit, root, owns_block, passed, record, metrics)


def _finish(
    obj: str,
    orbit: int,
    root: Path | None,
    owns_block: bool,
    passed: bool,
    record: dict[str, Any],
    metrics: dict[str, float | None],
) -> int:
    doc = load_status(obj, root)
    payload = dict(record)
    if metrics.get("registered_pct") is not None:
        payload["registered_pct"] = metrics["registered_pct"]
    if metrics.get("mean_reproj_px") is not None:
        payload["mean_reproj_px"] = metrics["mean_reproj_px"]
    set_path(doc, f"step01.orbit{orbit}", payload)
    save_status(obj, doc, root)
    if owns_block:
        finish_step(obj, "B2", "pass" if passed else "fail", root, actual=record)
    if not passed:
        error = record.get("error")
        if error:
            sys.stderr.write(f"{error}\n")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COLMAP one orbit and record B2 gates")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--orbit", type=int, required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return run_orbit(args.obj, args.orbit, args.root, mark_block=args.orbit == 1)
    except (StatusError, OSError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
