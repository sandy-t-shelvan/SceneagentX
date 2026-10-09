"""B3 splatfacto wrapper for one orbit.

Holds out every 10th frame, trains splatfacto, and reads ns-eval.
Tests parse a saved eval JSON and a PLY header, so they do not train.
Gates: PSNR >= 28, SSIM >= 0.90, Gaussian count from 0.3 million to 3 million.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import re
import shutil
import time
from typing import Any

from tools.check_env import CmdResult, Runner, default_runner
from tools.common.paths import frames_dir, renders_dir
from tools.common.status import StatusError, begin_step, finish_step, load_status, save_status, set_path
from tools.step01_colmap import colmap_dir

PSNR_MIN = 28.0
SSIM_MIN = 0.90
GAUSSIANS_MIN = 300_000
GAUSSIANS_MAX = 3_000_000


def holdout_names(names: list[str]) -> list[str]:
    """Every 10th name in sorted order. Frame 10, 20, 30, ... are held out."""
    ordered = sorted(names)
    return [name for index, name in enumerate(ordered, start=1) if index % 10 == 0]


def write_holdout(frame_names: list[str], destination: Path) -> list[str]:
    held = holdout_names(frame_names)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(held) + ("\n" if held else ""), encoding="utf-8")
    return held


def parse_eval(text: str) -> dict[str, float | None]:
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = json.loads(stripped.splitlines()[0])
        results = payload.get("results", payload)
        return {
            "psnr": _number(results.get("psnr")),
            "ssim": _number(results.get("ssim")),
        }
    return {
        "psnr": _search_float(text, r"psnr[^0-9]{0,12}(\d+(?:\.\d+)?)"),
        "ssim": _search_float(text, r"ssim[^0-9]{0,12}(\d+(?:\.\d+)?)"),
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _search_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def count_gaussians(ply_path: Path) -> int | None:
    header = ply_path.read_bytes().split(b"end_header", 1)[0].decode("ascii", errors="replace")
    for line in header.splitlines():
        if line.startswith("element vertex"):
            return int(line.split()[-1])
    return None


def parse_minutes(text: str) -> float | None:
    return _search_float(text, r"minutes[^0-9]{0,8}(\d+(?:\.\d+)?)")


def evaluate_splat(psnr: float | None, ssim: float | None, n_gaussians: int | None) -> dict[str, bool]:
    return {
        "psnr": psnr is not None and psnr >= PSNR_MIN,
        "ssim": ssim is not None and ssim >= SSIM_MIN,
        "gaussians": n_gaussians is not None and GAUSSIANS_MIN <= n_gaussians <= GAUSSIANS_MAX,
    }


def splat_path(obj: str, orbit: int, root: Path | None = None) -> Path:
    return colmap_dir(obj, orbit, root) / "splat.ply"


def run_orbit(
    obj: str,
    orbit: int,
    root: Path | None = None,
    runner: Runner | None = None,
    eval_text: str | None = None,
    minutes: float | None = None,
    mark_block: bool = True,
) -> int:
    active = runner or default_runner
    owns_block = mark_block and orbit == 1
    if owns_block:
        begin_step(obj, "B3", root)
    frames = frames_dir(obj, orbit, root)
    names = sorted(path.name for path in frames.iterdir() if path.is_file()) if frames.is_dir() else []
    holdout_file = colmap_dir(obj, orbit, root) / "holdout.txt"
    if names:
        write_holdout(names, holdout_file)
    started = time.perf_counter()
    log = ""
    if eval_text is None:
        processed = _train(obj, orbit, root, active)
        log = processed.stdout + "\n" + processed.stderr
        if processed.returncode != 0:
            return _finish(obj, orbit, root, owns_block, False, {"error": log.strip() or "splatfacto failed"}, None, None, None, None)
        eval_text = log
        if minutes is None:
            minutes = parse_minutes(log)
            if minutes is None:
                minutes = (time.perf_counter() - started) / 60.0
    parsed = parse_eval(eval_text)
    ply = splat_path(obj, orbit, root)
    n_gaussians = count_gaussians(ply) if ply.is_file() else None
    if minutes is None:
        minutes = parse_minutes(eval_text)
    gates = evaluate_splat(parsed["psnr"], parsed["ssim"], n_gaussians)
    passed = all(gates.values())
    record = {
        "psnr": parsed["psnr"],
        "ssim": parsed["ssim"],
        "n_gaussians": n_gaussians,
        "minutes": minutes,
        "gates": gates,
        "passed": passed,
    }
    return _finish(
        obj,
        orbit,
        root,
        owns_block,
        passed,
        record,
        parsed["psnr"],
        parsed["ssim"],
        n_gaussians,
        minutes,
    )


def _train(obj: str, orbit: int, root: Path | None, runner: Runner) -> CmdResult:
    if shutil.which("ns-train") is None and runner is default_runner:
        return CmdResult(127, "", "ns-train is not on PATH. Run this in the nerf env on the GPU box.")
    data = colmap_dir(obj, orbit, root)
    return runner(
        ["ns-train", "splatfacto", "--data", str(data), "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True"],
        3 * 60 * 60,
    )


def _finish(
    obj: str,
    orbit: int,
    root: Path | None,
    owns_block: bool,
    passed: bool,
    record: dict[str, Any],
    psnr: float | None,
    ssim: float | None,
    n_gaussians: int | None,
    minutes: float | None,
) -> int:
    doc = load_status(obj, root)
    payload = dict(record)
    payload["psnr"] = psnr
    payload["ssim"] = ssim
    payload["n_gaussians"] = n_gaussians
    payload["minutes"] = minutes
    set_path(doc, f"step02.orbit{orbit}", payload)
    save_status(obj, doc, root)
    if owns_block:
        finish_step(obj, "B3", "pass" if passed else "fail", root, actual=payload)
    error = record.get("error")
    if error and not passed:
        sys.stderr.write(f"{error}\n")
    novel = renders_dir(obj, "step02", root)
    novel.mkdir(parents=True, exist_ok=True)
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train splatfacto for one orbit and record B3 gates")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--orbit", type=int, required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return run_orbit(args.obj, args.orbit, args.root, mark_block=args.orbit == 1)
    except (StatusError, OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
