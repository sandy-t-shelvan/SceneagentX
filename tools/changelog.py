"""Write a Notion-style change log for one SceneAgent object.

Reads status.json, paper_check.json when it is already there, and the
artifact files. This module never calls a vision API and never reads a key.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import subprocess
import zipfile
from datetime import datetime
from typing import Any

from tools.common.paths import object_dir, repo_root
from tools.common.status import find_step, load_status
from tools.harvard_agents import SKILLS, write_agents
from tools.paper_check import PAPER_URL, build_report

# Earliest finished milestone first. The page lists the newest result first.
_FINISHED = (
    "input",
    "3dgs",
    "semantics",
    "decomposition",
    "physics",
    "articulation",
    "alignment",
    "orientation",
    "package",
    "inspector",
    "object_sisters",
    "scene_sisters",
    "policy",
)
_STEP = {
    "semantics": "B5",
    "decomposition": "B9",
    "physics": "B10",
    "articulation": "B11",
    "alignment": "B4",
    "orientation": "B4",
    "package": "B12",
    "inspector": "B12",
    "object_sisters": "B13",
    "scene_sisters": "B14",
    "policy": "B16",
}


def write_changelog(obj: str, root: Path | None = None) -> Path:
    folder = object_dir(obj, root)
    write_agents(obj, root)
    status = load_status(obj, root, create=False)
    rows, scored = _rows(obj, root, folder)
    entries = [_entry(row, folder, status, scored) for row in rows]
    entries.sort(key=lambda item: item["rank"], reverse=True)
    text = _render(obj, entries, folder)
    if "sk-" in text or "Bearer " in text:
        raise RuntimeError("changelog refused to write a secret")
    path = folder / "changelog.md"
    path.write_text(text, encoding="utf-8")
    return path


def _rows(obj: str, root: Path | None, folder: Path) -> tuple[list[dict[str, Any]], bool]:
    path = folder / "paper_check.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        blocks = loaded.get("blocks") if isinstance(loaded, dict) else None
        if isinstance(blocks, list) and blocks:
            return blocks, True
    return build_report(obj, root)["blocks"], False


def _entry(row: dict[str, Any], folder: Path, status: dict[str, Any], scored: bool) -> dict[str, Any]:
    block_id = str(row.get("id") or "block")
    status_word, harvard = _demote_skips(block_id, *_labels(row.get("match")), folder, status)
    progressed = _progressed(block_id, status)
    return {
        "id": block_id,
        "status": status_word,
        "harvard": harvard,
        "artifact": str(row.get("artifact") or block_id),
        "evidence": _evidence(block_id, folder, status),
        "note": str(row.get("note") or row.get("paper_requirement") or ""),
        "rank": _FINISHED.index(block_id) if block_id in _FINISHED else -1,
        "in_log": scored or progressed or row.get("match") in (True, "partial"),
    }


def _labels(match: Any) -> tuple[str, str]:
    if match is True or match == "yes":
        return "Achieved", "yes"
    if match == "partial":
        return "Partial", "partial"
    return "Not yet", "no"


def achieved_word(
    match: Any,
    *,
    skipped: bool = False,
    asked: bool | None = None,
    kind: str = "",
) -> str:
    """Skipped work and asked: false are never written as Achieved."""
    if asked is False:
        return "Not yet"
    if skipped and kind == "alignment":
        return "Partial"
    if skipped:
        return "Not yet"
    if match is True or match == "yes":
        return "Achieved"
    if match == "partial":
        return "Partial"
    return "Not yet"


def _demote_skips(
    block_id: str,
    status_word: str,
    harvard: str,
    folder: Path,
    status: dict[str, Any],
) -> tuple[str, str]:
    """A skipped alignment or orientation is not Achieved. asked: false is not Achieved."""
    if _review_asked_false(folder, block_id):
        return "Not yet", "no"
    if status_word != "Achieved":
        return status_word, harvard
    if block_id == "alignment" and _alignment_skipped(folder, status):
        return "Partial", "partial"
    if block_id == "orientation" and _orientation_skipped(folder, status):
        return "Not yet", "no"
    if block_id == "semantics" and _infill_skipped(folder):
        return "Partial", "partial"
    return status_word, harvard


_REVIEW_KEYS = {
    "inspector": ("inspect", "isaac"),
    "semantics": ("see", "mask"),
    "alignment": ("align",),
    "orientation": ("orient",),
    "physics": ("physics",),
    "decomposition": ("parts",),
    "articulation": ("joint", "motion"),
    "package": ("package",),
    "object_sisters": ("sister",),
    "policy": ("policy",),
}


def _review_asked_false(folder: Path, block_id: str) -> bool:
    keys = _REVIEW_KEYS.get(block_id, (block_id,))
    reviews = folder / "reviews"
    if reviews.is_dir():
        for path in reviews.glob("*.json"):
            if _file_asked_false(path, keys):
                return True
    runs = folder / "runs.jsonl"
    if runs.is_file():
        for line in runs.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            vision = row.get("vision_gate") if isinstance(row.get("vision_gate"), dict) else {}
            skill = str(row.get("skill") or "")
            if vision.get("asked") is False and any(key in skill.lower() for key in keys):
                return True
    return False


def _file_asked_false(path: Path, keys: tuple[str, ...]) -> bool:
    blob = path.name.lower()
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(loaded, dict):
        return False
    step = str(loaded.get("step") or "")
    if not any(key in blob or key in step.lower() for key in keys):
        return False
    if loaded.get("asked") is False:
        return True
    vision = loaded.get("vision_gate")
    return isinstance(vision, dict) and vision.get("asked") is False


def _alignment_skipped(folder: Path, status: dict[str, Any]) -> bool:
    record = _read_json(folder / "align.json")
    if record is None:
        record = ((status.get("align") or {}).get("report") or {})
    return record.get("photometric") == "skipped"


def _orientation_skipped(folder: Path, status: dict[str, Any]) -> bool:
    record = _read_json(folder / "orientation.json")
    if record is None:
        record = status.get("orientation") or {}
    return record.get("skipped") is True


def _infill_skipped(folder: Path) -> bool:
    record = _read_json(folder / "masks" / "object.json") or {}
    return record.get("infill") == "skipped"


def _progressed(block_id: str, status: dict[str, Any]) -> bool:
    if block_id == "input":
        return bool(status.get("input"))
    if block_id == "3dgs":
        return bool(((status.get("step02") or {}).get("orbit1") or {}).get("splat"))
    step_id = _STEP.get(block_id)
    if step_id is None:
        return False
    return find_step(status, step_id).get("state") != "not_started"


def _evidence(block_id: str, folder: Path, status: dict[str, Any]) -> str:
    readers = {
        "input": _ev_input,
        "3dgs": _ev_3dgs,
        "semantics": _ev_semantics,
        "physics": _ev_physics,
        "decomposition": _ev_parts,
        "articulation": _ev_urdf,
        "alignment": _ev_align,
        "orientation": _ev_orientation,
        "package": _ev_package,
        "inspector": _ev_inspect,
        "object_sisters": _ev_object_sisters,
        "scene_sisters": _ev_scene_sisters,
        "policy": _ev_policy,
    }
    reader = readers.get(block_id)
    if reader is None:
        return "no artifact metric for this block"
    return reader(folder, status)


def _ev_input(folder: Path, status: dict[str, Any]) -> str:
    record = status.get("input") or {}
    if not record:
        return "input record missing"
    return (
        f"kind={record.get('kind')}; cameras={_flag(record.get('cameras'))}; "
        f"capture_images={_flag(record.get('capture_images'))}"
    )


def _ev_3dgs(folder: Path, status: dict[str, Any]) -> str:
    orbit = ((status.get("step02") or {}).get("orbit1") or {})
    splat = Path(str(orbit.get("splat") or ""))
    count = _ply_vertex_count(splat)
    photo = _flag(orbit.get("photometric"))
    source = orbit.get("source")
    if count is None:
        return f"splat missing; photometric={photo}; source={source}"
    return f"n_gaussians={count}; photometric={photo}; source={source}"


def _ev_semantics(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "masks" / "object.json")
    if record is None:
        return "masks/object.json missing"
    return (
        f"segmenter={record.get('segmenter')}; "
        f"grounding_dino={_flag(record.get('grounding_dino'))}; "
        f"sam={_flag(record.get('sam'))}; "
        f"n_gaussians={record.get('n_gaussians')}"
    )


def _ev_physics(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "physics.json")
    if record is None:
        return "physics.json missing"
    codebook = _read_json(folder / "codebook.json")
    per = record.get("per_gaussian") or {}
    if codebook is None:
        dinov2 = "missing"
        openclip = "missing"
    else:
        dinov2 = _flag(codebook.get("dinov2"))
        openclip = _flag(codebook.get("openclip"))
    return (
        f"source={record.get('source')}; model={record.get('model')}; "
        f"material={record.get('material')}; friction={record.get('friction')}; "
        f"rigidity={record.get('rigidity')}; mass_kg={record.get('mass_kg')}; "
        f"density_kg_m3={record.get('density_kg_m3')}; "
        f"copied_onto={per.get('n')}; learned_prior={_flag(per.get('learned_prior'))}; "
        f"dinov2={dinov2}; openclip={openclip}"
    )


def _ev_parts(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "parts.json")
    if record is None:
        return "parts.json missing"
    parts = record.get("parts") or []
    count = len(parts) if isinstance(parts, list) else 0
    vertices = None
    if isinstance(parts, list) and parts and isinstance(parts[0], dict):
        vertices = parts[0].get("vertex_count")
    return f"n_parts={count}; partfield={_flag(record.get('partfield'))}; vertex_count={vertices}"


def _ev_urdf(folder: Path, status: dict[str, Any]) -> str:
    path = folder / "asset.urdf"
    if not path.is_file():
        return "asset.urdf missing"
    joints = path.read_text(encoding="utf-8").lower().count("<joint")
    motion = _read_json(folder / "motion.json") or {}
    source = motion.get("source") or "urdf"
    return f"n_joints={joints}; source={source}"


def _ev_align(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "align.json")
    if record is None:
        record = ((status.get("align") or {}).get("report") or {})
    if not record:
        return "align.json missing"
    return f"photometric={record.get('photometric')}; scale_m={record.get('scale_m')}"


def _ev_orientation(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "orientation.json")
    if record is None:
        record = status.get("orientation") or {}
    if not record:
        return "orientation.json missing"
    return f"skipped={_flag(record.get('skipped'))}"


def _ev_package(folder: Path, status: dict[str, Any]) -> str:
    usdz = folder / "scene.usdz"
    chamfer = (find_step(status, "B8").get("actual") or {}).get("chamfer_cm")
    if not usdz.is_file():
        return f"scene.usdz missing; chamfer_cm={chamfer}"
    return (
        f"usdz_bytes={usdz.stat().st_size}; has_mesh={_flag(_usdz_has_mesh(usdz))}; "
        f"chamfer_cm={chamfer}"
    )


def _ev_inspect(folder: Path, status: dict[str, Any]) -> str:
    record = status.get("inspect") or {}
    if not record:
        return "inspect record missing"
    return f"isaac={_flag(record.get('isaac'))}"


def _ev_object_sisters(folder: Path, status: dict[str, Any]) -> str:
    record = _read_json(folder / "sisters" / "objects.json")
    if record is None:
        return "sisters/objects.json missing"
    objects = record.get("objects") or []
    count = len(objects) if isinstance(objects, list) else 0
    return f"generator={record.get('generator')}; trellis={_flag(record.get('trellis'))}; n={count}"


def _ev_scene_sisters(folder: Path, status: dict[str, Any]) -> str:
    scenes = _read_json(folder / "sisters" / "scenes.json")
    objects = _read_json(folder / "sisters" / "objects.json") or {}
    if scenes is None:
        return "sisters/scenes.json missing"
    layouts = scenes.get("scenes") or []
    count = len(layouts) if isinstance(layouts, list) else 0
    return (
        f"n_scenes={count}; generator={objects.get('generator')}; "
        f"trellis={_flag(objects.get('trellis'))}"
    )


def _ev_policy(folder: Path, status: dict[str, Any]) -> str:
    isaac = _flag((status.get("inspect") or {}).get("isaac"))
    states = [f"{block_id}={find_step(status, block_id).get('state')}" for block_id in ("B15", "B16", "B17")]
    return f"isaac={isaac}; " + "; ".join(states)


def _ply_vertex_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        for _ in range(80):
            line = handle.readline()
            if not line or line.startswith(b"end_header"):
                return None
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])
    return None


def _usdz_has_mesh(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.endswith((".usda", ".usdc", ".usd")) and b"def Mesh" in archive.read(name):
                    return True
    except zipfile.BadZipFile:
        return False
    return False


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must be a JSON object")
    return loaded


def _flag(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _agent_heading(folder: Path) -> str:
    record = _read_json(folder / "agents.json") or {}
    if record.get("visual_review") is True:
        return "## Skill router"
    if record.get("ran") is True:
        return "## Skill router"
    return "## Agents not run"


def _fault_lines(folder: Path) -> list[str]:
    record = _read_json(folder / "agents.json") or {}
    faults = record.get("faults") if isinstance(record.get("faults"), list) else []
    lines = ["## Faults", ""]
    if not faults:
        lines.append("- No fault restarted a skill.")
        return lines
    for item in faults:
        if not isinstance(item, dict):
            continue
        lines.append(
            f"- {item.get('block')} attempt {item.get('attempt')} "
            f"fault={item.get('fault')} restarted={item.get('restarted')}"
        )
    return lines


def _git_line() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "Git commit: unavailable"
    state = "dirty working tree" if dirty else "clean"
    return f"Git commit: {commit} ({state})"


def _cuda_line(folder: Path) -> str:
    record = _read_json(folder / "cuda_pin.json")
    if record is None:
        return "CUDA pin: not recorded"
    return (
        f"CUDA pin: nvcc {record.get('nvcc')} ; "
        f"torch {record.get('torch')} ; "
        f"torch.version.cuda {record.get('torch_cuda')} ; "
        f"match {record.get('match')}"
    )


def _render(obj: str, entries: list[dict[str, Any]], folder: Path) -> str:
    updated = datetime.now().astimezone().isoformat(timespec="seconds")
    counts = {word: sum(1 for entry in entries if entry["status"] == word) for word in ("Achieved", "Partial", "Not yet")}
    lines = [
        f"# SceneAgent change log — {obj}",
        "",
        f"Last updated: {updated}",
        "",
        f"Paper: Hollis, Fan, Yang 2026. {PAPER_URL}",
        "",
        f"Achieved: {counts['Achieved']}  ·  Partial: {counts['Partial']}  ·  Not yet: {counts['Not yet']}",
        "",
        _git_line(),
        _cuda_line(folder),
        "",
        _agent_heading(folder),
        "",
        "The file check is not SCENEAGENT-RUN-STEP.",
        "",
    ]
    for skill in SKILLS:
        lines.append(f"- {skill['id']}: {skill['ours']}")
    lines.extend(["", *_fault_lines(folder), "", "## Change log", ""])
    for entry in entries:
        if not entry["in_log"]:
            continue
        lines.append(
            f"- **{entry['id']}** — {entry['status']} — Harvard match {entry['harvard']} — {entry['evidence']}"
        )
    if lines[-1] == "":
        lines.append("- No block has a result yet.")
    lines.append("")
    for entry in entries:
        lines.extend(
            [
                f"## {entry['id']}",
                "",
                "| Property | Value |",
                "| --- | --- |",
                f"| Status | {_cell(entry['status'])} |",
                f"| Harvard match | {_cell(entry['harvard'])} |",
                f"| Artifact | `{_cell(entry['artifact'])}` |",
                f"| Evidence | {_cell(entry['evidence'])} |",
                f"| Note | {_cell(entry['note'])} |",
                "",
            ]
        )
    lines.extend(_design_lines(folder))
    return "\n".join(lines) + "\n"


def _design_lines(folder: Path) -> list[str]:
    """Append the system-design block results when a run wrote them."""
    record = _read_json(folder / "pipeline_state.json")
    if not record:
        return []
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        return []
    lines = ["## Design blocks", "", "Source: docs/sceneagent_system_design.md", ""]
    for step in steps:
        if not isinstance(step, dict):
            continue
        lines.append(
            f"- **{step.get('id')}** — {step.get('state')} — {step.get('result')}"
        )
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write data/<object>/changelog.md")
    parser.add_argument("--obj", default="scan")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    path = write_changelog(args.obj, args.root)
    sys.stdout.write(f"{path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
