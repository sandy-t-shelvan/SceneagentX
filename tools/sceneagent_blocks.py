"""Score one object against the SceneAgent system design.

The design lives in docs/sceneagent_system_design.md. This module does not
retrain a splat, does not start COLMAP, and does not set learned_prior true.
A vision review is asked only when a picture and an API key are both present.
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

from tools.common.paths import object_dir
from tools.step01_colmap import evaluate_colmap, parse_analyzer
from tools.step02_splatfacto import count_gaussians, splat_path

DESIGN = "docs/sceneagent_system_design.md"
_BLOCKS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K")


def run_design(obj: str, root: Path | None = None) -> dict[str, Any]:
    """Write one report per design block and refresh the change log."""
    from tools.design_workspace import build_workspace

    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    build_workspace(obj, root)
    steps = [_score(block_id, obj, folder, root) for block_id in _BLOCKS]
    state = {
        "design": DESIGN,
        "object": obj,
        "model": "gpt-6-astra",
        "learned_prior": False,
        "steps": steps,
    }
    from tools.common.status import load_status

    load_status(obj, root, create=True)
    (folder / "pipeline_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    design_dir = folder / "design"
    design_dir.mkdir(parents=True, exist_ok=True)
    for step in steps:
        (design_dir / f"{step['id']}.json").write_text(json.dumps(step, indent=2) + "\n", encoding="utf-8")
    from tools.changelog import write_changelog

    write_changelog(obj, root)
    return state


def _score(block_id: str, obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    scorers = {
        "A": _block_a,
        "B": _block_b,
        "C": _block_c,
        "D": _block_d,
        "E": _block_e,
        "F": _block_f,
        "G": _block_g,
        "H": _block_h,
        "I": _block_i,
        "J": _block_j,
        "K": _block_k,
    }
    step = scorers[block_id](obj, folder, root)
    step["id"] = block_id
    step["learned_prior"] = False
    return step


def _block_a(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    ply = splat_path(obj, 1, root)
    count = count_gaussians(ply) if ply.is_file() else None
    analyzer = folder / "colmap" / "orbit1" / "analyzer.txt"
    report = parse_analyzer(analyzer.read_text(encoding="utf-8")) if analyzer.is_file() else {}
    models = ["0"] if (folder / "colmap" / "orbit1" / "models.json").is_file() else []
    gates = evaluate_colmap(report, models, 0) if report else {}
    colmap_ok = bool(gates) and all(gates.values())
    if ply.is_file() and colmap_ok:
        state = "partial"
        result = (
            f"splat kept gaussians {count}; COLMAP not restarted; "
            "PSNR and LPIPS were not measured, so M1 is not met"
        )
    elif ply.is_file():
        state = "partial"
        result = f"splat kept gaussians {count}; metric scale and held-out PSNR were not measured"
    else:
        state = "not_started"
        result = "no scene splat"
    return {"state": state, "result": result, "artifact": str(ply)}


def _block_b(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    masks = _read(folder / "masks" / "object.json")
    codebook = _read(folder / "codebook.json")
    detected = masks.get("grounding_dino") is True and masks.get("sam") is True
    projected = masks.get("projected") is True
    encoded = codebook.get("dinov2") is True and codebook.get("openclip") is True
    if detected and projected and encoded:
        state = "partial"
        result = (
            f"detections {masks.get('n_views_with_detection')}; "
            f"inside {masks.get('n_inside')}; one object vector; "
            "LangSplat codebook and mIoU were not measured"
        )
    elif detected or encoded:
        state = "partial"
        result = "masks or the object vector exist; the codebook lift is incomplete"
    else:
        state = "not_started"
        result = "GroundingDINO, SAM, and the feature vector are missing"
    return {"state": state, "result": result, "artifact": "masks/object.json"}


def _block_c(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    masks = _read(folder / "masks" / "object.json")
    if masks.get("infill") == "ran" and (folder / "background.ply").is_file():
        state = "pass"
        result = "background infill ran"
    elif masks.get("projected") is True and any((folder / "objects").glob("*/object.ply")):
        state = "partial"
        result = "foreground Gaussians were cropped and recentered; predictive background infill was not run"
    elif masks.get("projected") is True:
        state = "partial"
        result = "object crop exists; predictive background infill was not run"
    else:
        state = "not_started"
        result = "no crop to infill"
    return {"state": state, "result": result, "artifact": "masks/object.json"}


def _block_d(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    physics = _read(folder / "physics.json")
    per = physics.get("per_gaussian") if isinstance(physics.get("per_gaussian"), dict) else {}
    if physics.get("learned_prior") is True or per.get("learned_prior") is True:
        raise RuntimeError("learned_prior must stay false")
    if physics.get("material") and physics.get("mass_kg") is not None:
        state = "partial"
        result = (
            f"material {physics.get('material')}; mass_kg {physics.get('mass_kg')}; "
            "learned_prior false; mass was not checked on a scale"
        )
    else:
        state = "not_started"
        result = "no physics record"
    return {"state": state, "result": result, "artifact": "physics.json"}


def _block_e(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    urdf = folder / "asset.urdf"
    text = urdf.read_text(encoding="utf-8") if urdf.is_file() else ""
    joints = text.count("<joint ")
    if joints > 0:
        state = "partial"
        result = f"{joints} URDF joints; Isaac did not sweep them; no deformable press"
    elif urdf.is_file():
        state = "partial"
        result = "URDF is one body; joint review did not keep a moving part"
    else:
        state = "not_started"
        result = "no URDF"
    return {"state": state, "result": result, "artifact": "asset.urdf"}


def _block_f(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    isaac = _read(folder / "isaac_check.json")
    frame = _first_frame(folder)
    rendered = isaac.get("rendered") is True and frame is not None
    if rendered:
        state = "partial"
        result = f"Isaac frame {frame.name}; photometric alignment was not run"
    elif isaac:
        state = "partial"
        result = f"Isaac check recorded; no RGB frame; reason {isaac.get('reason') or isaac.get('error') or 'reset did not finish'}"
    else:
        state = "not_started"
        result = "Isaac inspection has not run"
    return {"state": state, "result": result, "artifact": "isaac_check.json"}


def _block_g(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    sisters = _read(folder / "sisters" / "objects.json")
    count = len(sisters.get("objects") or []) if isinstance(sisters.get("objects"), list) else 0
    if count >= 9 and sisters.get("clip_gate") is True:
        state = "pass"
        result = f"{count} sisters passed the similarity gate"
    elif count > 0:
        state = "partial"
        result = f"{count} sisters exist; the site asks for 9 and a CLIP gate, which was not measured"
    else:
        state = "not_started"
        result = "no sisters"
    return {"state": state, "result": result, "artifact": "sisters/objects.json"}


def _block_h(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    asked = bool(os.environ.get("OPENAI_API_KEY", "").strip()) and _first_frame(folder) is not None
    if asked:
        result = "gpt-6-astra can review the saved frame; the skill swarm is file-state, not a second framework"
    else:
        result = "orchestrator state written; no review picture, so the model was not asked"
    return {"state": "partial", "result": result, "artifact": "pipeline_state.json"}


def _block_i(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    return {"state": "not_started", "result": "Three.js editor was not built", "artifact": ""}


def _block_j(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    return {"state": "not_started", "result": "demonstration factory was not run", "artifact": ""}


def _block_k(obj: str, folder: Path, root: Path | None) -> dict[str, Any]:
    return {"state": "not_started", "result": "VLA finetune and real-robot trials were not run", "artifact": ""}


def _first_frame(folder: Path) -> Path | None:
    inspect = folder / "inspect"
    if not inspect.is_dir():
        return None
    frames = sorted(inspect.glob("*.png"))
    return frames[0] if frames else None


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score design blocks and update the change log")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    state = run_design(args.obj, args.root)
    for step in state["steps"]:
        sys.stdout.write(f"{step['id']} {step['state']}  {step['result']}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
