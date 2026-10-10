"""Walk one object without waiting between skills.

A finished COLMAP report or splat.ply is kept. A failed skill restarts once.
The second fault stops that skill. A ply never starts COLMAP. A missing
vision key does not invent physics or joints.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from typing import Any

from tools.agent_loop import SkillFault, _record_fault, choose_path, finish_route, run_skill
from tools.common.paths import object_dir
from tools.common.status import finish_step, load_status, save_status, set_path
from tools.step01_colmap import colmap_dir, evaluate_colmap, model_names, parse_analyzer, sparse_dir
from tools.step02_splatfacto import count_gaussians, splat_path

SEQUENCE = ("B2", "B3", "B4", "B5", "B6", "B7")


def iterate(obj: str, root: Path | None = None) -> int:
    """Keep finished work, skip what this machine cannot honestly run, then stop."""
    route = choose_path(obj, root)
    sys.stdout.write(f"path {route}\n")
    if route == "stop":
        sys.stdout.write("stopped before COLMAP\n")
        return 2
    if route == "ply":
        sys.stdout.write("COLMAP not called\n")
        _write_log(obj, root, route, [{"block": "B2", "action": "not called", "reason": "ply"}])
        finish_route(obj, root, False)
        return 0
    rows: list[dict[str, str]] = []
    for block_id in SEQUENCE:
        action, reason = _video_decision(obj, block_id, root)
        sys.stdout.write(f"{block_id} {action}  {reason}\n")
        rows.append({"block": block_id, "action": action, "reason": reason})
        if action == "kept":
            _mark_kept(obj, block_id, reason, root)
            continue
        if action == "skip":
            _record_fault(obj, block_id, 1, reason, False, root)
            continue
        code, _visual = run_skill(
            obj,
            block_id,
            lambda _parameter, block_id=block_id: _run_block(obj, block_id, root),
            root,
        )
        if code != 0:
            _write_log(obj, root, route, rows)
            sys.stdout.write(f"stopped at {block_id}\n")
            return code
    _write_log(obj, root, route, rows)
    finish_route(obj, root, False)
    sys.stdout.write("iterate done\n")
    return 0


def _video_decision(obj: str, block_id: str, root: Path | None) -> tuple[str, str]:
    if block_id == "B2":
        report = _colmap_report(obj, root)
        if report is None:
            return "run", "no passing COLMAP report"
        return (
            "kept",
            f"registered {report['registered_pct']} reproj {report['mean_reproj_px']} px  COLMAP not restarted",
        )
    if block_id == "B3":
        ply = splat_path(obj, 1, root)
        if not ply.is_file():
            return "run", "splat.ply missing"
        count = count_gaussians(ply)
        return "kept", f"splat.ply gaussians {count}  training not restarted  psnr not measured"
    if block_id == "B4":
        return "skip", "no AprilTag measurement  alignment not started"
    if block_id == "B5":
        record = _read_json(object_dir(obj, root) / "masks" / "object.json")
        if record.get("grounding_dino") is True and record.get("sam") is True:
            detected = record.get("n_views_with_detection")
            return "kept", f"GroundingDINO and SAM  detections {detected}"
        return "run", "masks/object.json is missing a detector result"
    if block_id == "B6":
        record = _read_json(object_dir(obj, root) / "masks" / "object.json")
        crop = object_dir(obj, root) / "masks" / "gaussian_in_mask.npy"
        if record.get("projected") is True and crop.is_file():
            return "kept", f"projected inside {record.get('n_inside')}"
        return "skip", "camera poses are not on this machine  lift not run"
    if block_id == "B7":
        record = _read_json(object_dir(obj, root) / "codebook.json")
        if record.get("dinov2") is True and record.get("openclip") is True:
            return "kept", "DINOv2 and OpenCLIP kept  vector not copied onto Gaussians"
        return "skip", "DINOv2 and OpenCLIP run on the GPU  geometric stand-in not written"
    return "skip", "no skill"


def _colmap_report(obj: str, root: Path | None) -> dict[str, Any] | None:
    doc = load_status(obj, root, create=False) if _status_exists(obj, root) else {}
    stored = ((doc.get("step01") or {}).get("orbit1") or {}) if isinstance(doc, dict) else {}
    if stored.get("passed") is True and stored.get("registered_pct") is not None:
        return stored
    text_path = colmap_dir(obj, 1, root) / "analyzer.txt"
    if not text_path.is_file():
        return None
    metrics = parse_analyzer(text_path.read_text(encoding="utf-8"))
    models = _stored_models(obj, root)
    gates = evaluate_colmap(metrics, models, 0)
    if not all(gates.values()):
        return None
    return {
        "registered_pct": metrics.get("registered_pct"),
        "mean_reproj_px": metrics.get("mean_reproj_px"),
        "models": models,
        "gates": gates,
        "passed": True,
    }


def _stored_models(obj: str, root: Path | None) -> list[str]:
    path = colmap_dir(obj, 1, root) / "models.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list) and all(isinstance(item, str) for item in loaded):
            return loaded
    return model_names(sparse_dir(obj, 1, root))


def _run_block(obj: str, block_id: str, root: Path | None) -> None:
    if block_id == "B2":
        from tools.step01_colmap import run_orbit

        code = run_orbit(obj, 1, root)
        if code != 0:
            raise SkillFault("COLMAP gates failed. The retry uses sequential matching on frames.")
        return
    if block_id == "B3":
        if splat_path(obj, 1, root).is_file():
            return
        raise SkillFault("splat.ply is missing. Training was not started from this loop.")
    if block_id == "B5":
        raise SkillFault("GroundingDINO and SAM run on the GPU. They were not started from this loop.")
    raise SkillFault(f"{block_id} was not run")


def _mark_kept(obj: str, block_id: str, reason: str, root: Path | None) -> None:
    if block_id == "B2":
        report = _colmap_report(obj, root) or {}
        doc = load_status(obj, root)
        set_path(doc, "step01.orbit1", report)
        if not (doc.get("input") or {}).get("kind"):
            set_path(doc, "input", {"kind": "video", "cameras": True, "capture_images": True})
        save_status(obj, doc, root)
    if block_id == "B3":
        ply = splat_path(obj, 1, root)
        doc = load_status(obj, root)
        set_path(
            doc,
            "step02.orbit1",
            {
                "splat": str(ply),
                "n_gaussians": count_gaussians(ply),
                "psnr": None,
                "ssim": None,
                "photometric": "not measured",
                "source": "splatfacto",
                "kept": True,
                "trained_again": False,
            },
        )
        save_status(obj, doc, root)
    finish_step(obj, block_id, "pass", root, actual={"kept": True, "reason": reason})


def _write_log(obj: str, root: Path | None, route: str, rows: list[dict[str, str]]) -> None:
    path = object_dir(obj, root) / "iterate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"path": route, "colmap": "not called" if route == "ply" else "video", "steps": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _status_exists(obj: str, root: Path | None) -> bool:
    from tools.common.paths import status_path

    return status_path(obj, root).is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keep finished skills and continue to the next one")
    parser.add_argument("--obj", default="bottle")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return iterate(args.obj, args.root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
