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
import os
from typing import Any

from tools.agent_loop import SkillFault, _record_fault, ask_question, choose_path, finish_route, run_skill
from tools.common.paths import object_dir
from tools.common.status import append_run, finish_step, load_status, run_timestamp, save_status, set_path
from tools.common.vlm import MissingReviewImage, VlmError, review
from tools.harvard_agents import allowed_params
from tools.step01_colmap import colmap_dir, evaluate_colmap, model_names, parse_analyzer, sparse_dir
from tools.step02_splatfacto import count_gaussians, splat_path

SEQUENCE = ("B2", "B3", "B4", "B5", "B6", "B7")


def iterate(obj: str, root: Path | None = None) -> int:
    """Keep finished work. One runs.jsonl line records this invocation."""
    code = _iterate_walk(obj, root)
    _log_walk(obj, root, code)
    return code


def _iterate_walk(obj: str, root: Path | None = None) -> int:
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
            from tools.gate_bypass import continue_past

            if continue_past(obj, root, f"iterate.{block_id}", reason or f"{block_id} stopped", check="unmet"):
                continue
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
        stored = _read_json(object_dir(obj, root) / "ingest" / "ingest_report.json").get("psnr")
        if isinstance(stored, (int, float)) and not isinstance(stored, bool):
            return "kept", f"splat.ply gaussians {count}  training not restarted  psnr {stored}"
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
            return "kept", "DINOv2 and OpenCLIP object vector for physics kept"
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
        previous = ((doc.get("step02") or {}).get("orbit1") or {})
        ingest = _read_json(object_dir(obj, root) / "ingest" / "ingest_report.json")
        psnr = previous.get("psnr") if isinstance(previous.get("psnr"), (int, float)) and not isinstance(previous.get("psnr"), bool) else ingest.get("psnr")
        if isinstance(psnr, bool) or not isinstance(psnr, (int, float)):
            psnr = None
        ssim = previous.get("ssim") if isinstance(previous.get("ssim"), (int, float)) and not isinstance(previous.get("ssim"), bool) else ingest.get("ssim")
        if isinstance(ssim, bool) or not isinstance(ssim, (int, float)):
            ssim = None
        set_path(
            doc,
            "step02.orbit1",
            {
                "splat": str(ply),
                "n_gaussians": count_gaussians(ply),
                "psnr": psnr,
                "ssim": ssim,
                "lpips": ingest.get("lpips") if isinstance(ingest.get("lpips"), (int, float)) and not isinstance(ingest.get("lpips"), bool) else None,
                "photometric": "not measured" if psnr is None else "measured",
                "source": "splatfacto",
                "kept": True,
                "trained_again": False,
                "capture": "data/bottle/raw/orbit1.MOV" if obj == "bottle" else None,
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


SKILL_STEPS: dict[str, dict[str, str]] = {
    "ingest": {"skill": "SCENEAGENT-INGEST-DATA", "block": "B2", "question": "Held-out colour should match the capture."},
    "see": {
        "skill": "SCENEAGENT-RUN-STEP",
        "block": "B5",
        "question": "The red overlay should cover the bottle and nothing else.",
    },
    "codebook": {
        "skill": "SCENEAGENT-RUN-STEP",
        "block": "B7",
        "question": "The crop should be the object, not the room.",
    },
    "parts": {"skill": "SCENEAGENT-RUN-STEP", "block": "B9", "question": "Part colours should match the named parts."},
    "physics": {"skill": "SCENEAGENT-RUN-STEP", "block": "B10", "question": "The material guess should match the object."},
    "joints": {"skill": "SCENEAGENT-RUN-STEP", "block": "B11", "question": "The named joint should match the picture."},
    "mesh": {"skill": "SCENEAGENT-RUN-STEP", "block": "B8", "question": "The extracted surface should not be melted or full of holes."},
    "align": {
        "skill": "SCENEAGENT-ALIGN-SCENE",
        "block": "B4",
        "question": "The rendered asset should sit on the capture frame.",
    },
    "inspect": {
        "skill": "SCENEAGENT-INSPECT-ISAAC",
        "block": "B12",
        "question": "The Isaac frame should match the capture frame from the same COLMAP pose.",
    },
    "press": {"skill": "SCENEAGENT-RUN-STEP", "block": "B15", "question": "The press should deform only the soft region."},
    "sisters": {
        "skill": "SCENEAGENT-OBJECT-SISTERS",
        "block": "B13",
        "question": "The sister should stay in class with the original.",
    },
}
_HELD_SKILLS = {"policy", "editor", "vla", "photogrammetry", "lidar", "a2"}


def iterate_skill(obj: str, skill: str, root: Path | None = None, vlm_mode: str = "full") -> int:
    """Run one skill: measure, then vision, then one allowed correction, then stop."""
    if vlm_mode not in {"off", "propose", "full"}:
        raise ValueError("vlm-mode must be off, propose, or full")
    if skill in _HELD_SKILLS:
        ask_question(
            obj,
            "B16",
            f"{skill} stays not_started until Block 9 Isaac inspect has a real frame.",
            root,
        )
        from tools.gate_bypass import continue_past

        continue_past(obj, root, "held_skill", f"{skill} stays not_started", check="skipped")
        _log_try(obj, skill, 1, root, False, {"name": "held", "pass": False}, {"asked": False}, None, False)
        return 2
    spec = SKILL_STEPS.get(skill)
    if spec is None:
        raise KeyError(f"unknown skill {skill}")
    names = allowed_params(spec["skill"])
    for try_n in (1, 2):
        measure = _measure(obj, skill, root)
        outputs = list(measure.get("outputs") or [])
        if not measure["pass"]:
            _log_try(obj, spec["skill"], try_n, root, False, measure, {"asked": False}, None, False, outputs)
            from tools.gate_bypass import continue_past

            continue_past(
                obj,
                root,
                "measurement_gate",
                str(measure.get("reason") or f"{spec['block']} measurement gate unmet"),
                check="unmet",
            )
            ask_question(obj, spec["block"], f"{spec['block']} failed the measurement gate. What should change?", root)
            return 2
        if vlm_mode == "off":
            _log_try(
                obj,
                spec["skill"],
                try_n,
                root,
                True,
                measure,
                {"asked": False, "reason": "vlm-mode off"},
                None,
                True,
                outputs,
            )
            return 0
        if vlm_mode == "propose":
            _log_try(
                obj,
                spec["skill"],
                try_n,
                root,
                True,
                measure,
                {"asked": False, "reason": "propose mode does not record a verdict"},
                None,
                True,
                outputs,
            )
            return 0
        record = object_dir(obj, root) / "reviews" / f"{skill}_{try_n}.json"
        try:
            images = _review_images(obj, skill, root)
            verdict = review(skill, images, spec["question"], names, record_path=record)
        except MissingReviewImage as exc:
            _log_try(obj, spec["skill"], try_n, root, False, measure, {"asked": False, "error": str(exc)}, None, False, outputs)
            from tools.gate_bypass import continue_past

            continue_past(obj, root, "vision_gate", str(exc), check="unmet")
            ask_question(obj, spec["block"], f"{spec['block']} had no review picture. What should change?", root)
            return 2
        except (VlmError, OSError, ValueError, json.JSONDecodeError) as exc:
            _log_try(obj, spec["skill"], try_n, root, False, measure, {"asked": False, "error": str(exc)}, None, False, outputs)
            from tools.gate_bypass import continue_past

            continue_past(obj, root, "vision_gate", str(exc), check="unmet")
            ask_question(obj, spec["block"], f"{spec['block']} vision gate failed. What should change?", root)
            return 2
        vision = {
            "asked": True,
            "verdict": verdict["verdict"],
            "confidence": verdict["confidence"],
            "problem": verdict["problem"],
            "review_file": f"reviews/{skill}_{try_n}.json",
        }
        if verdict["verdict"] == "pass":
            _log_try(obj, spec["skill"], try_n, root, True, measure, vision, None, True, outputs)
            return 0
        parameter = verdict["correction"]["parameter"]
        value = verdict["correction"]["value"]
        vision_gate = "vlm_confidence" if verdict.get("confidence_unmet") is True else "vision_gate"
        if parameter not in names:
            _log_try(obj, spec["skill"], try_n, root, False, measure, vision, None, False, outputs)
            from tools.gate_bypass import continue_past

            continue_past(obj, root, vision_gate, f"{spec['block']} vision gate unmet", check="unmet")
            ask_question(obj, spec["block"], f"{spec['block']} proposed a disallowed parameter. What should change?", root)
            return 2
        if try_n == 2:
            _log_try(
                obj,
                spec["skill"],
                try_n,
                root,
                False,
                measure,
                vision,
                {"parameter": parameter, "value": value},
                False,
                outputs,
            )
            from tools.gate_bypass import continue_past

            continue_past(obj, root, "second_fault", f"{spec['block']} used its two tries", check="unmet")
            ask_question(obj, spec["block"], f"{spec['block']} used its two tries. What should change on the next try?", root)
            return 2
        _apply_one(obj, root, str(parameter), value)
        _log_try(
            obj,
            spec["skill"],
            try_n,
            root,
            False,
            measure,
            vision,
            {"parameter": parameter, "value": value},
            False,
            outputs,
        )
    from tools.gate_bypass import continue_past

    continue_past(obj, root, "second_fault", f"{spec['block']} stopped", check="unmet")
    ask_question(obj, spec["block"], f"{spec['block']} stopped. What should change on the next try?", root)
    return 2


def _measure(obj: str, skill: str, root: Path | None) -> dict[str, Any]:
    """Numbers only. This does not call the vision model or invent an artifact."""
    from tools.block_gates import measure_skill

    return measure_skill(obj, skill, root)


def _review_images(obj: str, skill: str, root: Path | None) -> list[Path]:
    folder = object_dir(obj, root)
    source = _source_png(obj, skill, root)
    if source is None:
        return []
    dest = folder / "reviews" / f"{skill}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != source.resolve():
        dest.write_bytes(source.read_bytes())
    return [dest]


def _source_png(obj: str, skill: str, root: Path | None) -> Path | None:
    folder = object_dir(obj, root)
    candidates = [
        folder / "masks" / "view_000_mask.png",
        folder / "reviews" / f"{skill}.png",
        folder / "inspect" / "frame_000.png",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _apply_one(obj: str, root: Path | None, parameter: str, value: Any) -> None:
    path = object_dir(obj, root) / "skill_params.json"
    current: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            current = loaded
    current[parameter] = value
    path.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")


def _log_walk(obj: str, root: Path | None, code: int) -> None:
    _log_try(
        obj,
        "iterate",
        1,
        root,
        code == 0,
        {"name": "kept_blocks", "pass": code == 0},
        {"asked": False},
        None,
        code == 0,
        ["iterate.json"],
    )


def _measure_gate(measure: dict[str, Any], measure_pass: bool) -> dict[str, Any]:
    """Keep the gate log to scalars. A skipped infill stays infill_pass false."""
    gate: dict[str, Any] = {"name": measure.get("name") or "measure", "pass": measure_pass}
    for key, value in measure.items():
        if key in {"name", "pass", "outputs"}:
            continue
        if value is None or isinstance(value, (str, int, float, bool)):
            gate[key] = value
    return gate


def _log_try(
    obj: str,
    skill_id: str,
    try_n: int,
    root: Path | None,
    measure_pass: bool,
    measure: dict[str, Any],
    vision: dict[str, Any],
    correction: dict[str, Any] | None,
    kept: bool,
    outputs: list[str] | None = None,
) -> None:
    append_run(
        obj,
        {
            "ts": run_timestamp(),
            "object": obj,
            "skill": skill_id,
            "try": try_n,
            "host": os.environ.get("SCENEAGENT_HOST", "").strip() or "laptop",
            "outputs": list(outputs or []),
            "measure_gate": _measure_gate(measure, measure_pass),
            "vision_gate": vision,
            "correction": correction,
            "kept": kept,
        },
        root,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Keep finished skills or run one skill and its gates")
    parser.add_argument("obj_pos", nargs="?")
    parser.add_argument("skill_pos", nargs="?")
    parser.add_argument("--obj", default=None)
    parser.add_argument("--skill", default=None)
    parser.add_argument("--vlm-mode", choices=("off", "propose", "full"), default="full")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    obj = args.obj or args.obj_pos or "bottle"
    skill = args.skill or args.skill_pos
    try:
        if skill:
            return iterate_skill(obj, skill, args.root, vlm_mode=args.vlm_mode)
        return iterate(obj, args.root)
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
