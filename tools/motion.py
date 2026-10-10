"""Name movable parts from pictures and write URDF joints.

gpt-6-astra supplies the object name and how each part moves. Projection
assigns the cloud points. A second call checks the result. One fault
restarts only the named step. A ply never starts COLMAP.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any, Callable

import numpy as np

from tools.common.paths import frames_dir, object_dir, renders_dir
from tools.common.status import load_status, save_status, set_path
from tools.harvard_agents import write_agents
from tools.ingest import colmap_refused
from tools.video_input import find_video, record_absent

_MODEL = "gpt-6-astra"
_URL = "https://api.openai.com/v1/chat/completions"
_AXES = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
_PRESS_CAP_M = 0.02
_TURN_CAP_DEG = 120.0
Ask = Callable[..., dict[str, Any]]


def _gate(text: str) -> None:
    sys.stdout.write(text + "\n")
    sys.stdout.flush()


def run_motion(obj: str, root: Path | None = None, ask: Ask | None = None) -> int:
    """Write joints from a vision reply. Returns 0, or 2 when the second fault stops."""
    kind, images = _images(obj, root)
    _gate(f"path {kind}")
    _gate("COLMAP not called" if kind == "ply" else "COLMAP not started in this step")
    _gate(f"pictures {len(images)}")
    if kind == "stop":
        _gate("model not asked")
        return 0
    if kind == "video_no_frames":
        _gate("model not asked")
        _write_motion(obj, root, {"source": "vlm", "skipped": True, "reason": "video has no extracted frames"})
        return 0
    if ask is None and not os.environ.get("OPENAI_API_KEY", "").strip():
        _gate("model not asked")
        return 0
    caller = ask or ask_model
    proposal = _validated(caller(images, role="propose", prior=None))
    _gate("model asked" if proposal.get("called") else "model not asked")
    _gate_parts(proposal.get("parts") or [])
    if not proposal.get("called"):
        _write_motion(obj, root, {"source": "vlm", "skipped": True, "reason": "model was not asked"})
        _close(obj, root, False)
        return 0
    faults = 0
    saw_model = proposal.get("called") is True
    kept: dict[str, Any] | None = None
    kept_labels: dict[str, int] | None = None
    while True:
        labels = assign_parts(obj, proposal.get("parts") or [], images, root)
        _gate_points(proposal.get("parts") or [], labels)
        _write_outputs(obj, root, proposal, labels)
        _gate_files(obj, root)
        review = caller(images[:1], role="review", prior=proposal)
        if review.get("called") is True:
            saw_model = True
        decision = str(review.get("decision") or "pass")
        if decision != "fault":
            _gate("review pass")
            _close(obj, root, saw_model)
            return 0
        faults += 1
        block = str(review.get("block") or "B11")
        fault_text = str(review.get("fault") or "fault")
        if block == "B5":
            _remember_fault(obj, block, faults, fault_text, True, root)
            _restart_prompt(obj, root, review)
            _close(obj, root, saw_model)
            return 0
        if block == "B4":
            _remember_fault(obj, block, faults, fault_text, True, root)
            _restart_up_axis(obj, root, review)
            _close(obj, root, saw_model)
            return 0
        if block == "B6":
            _remember_fault(obj, block, faults, fault_text, True, root)
            _restart_lift(obj, root)
            _close(obj, root, saw_model)
            return 0
        if faults >= 2:
            _gate("review second fault stop")
            if kept is not None and kept_labels is not None:
                _write_outputs(obj, root, kept, kept_labels)
            _remember_fault(obj, block, faults, fault_text, False, root)
            from tools.agent_loop import ask_question

            ask_question(obj, block, f"{block} used its two tries. What should change on the next try?", root)
            _close(obj, root, saw_model)
            return 2
        nxt = _validated(caller(images, role="propose", prior=review))
        if nxt.get("called") is True:
            saw_model = True
        if _motion_signature(nxt) == _motion_signature(proposal):
            _remember_fault(obj, block, faults, fault_text, False, root)
            from tools.agent_loop import ask_question

            ask_question(
                obj,
                block,
                f"{block} would repeat the same motion. What should change on the next try?",
                root,
            )
            _gate("review same motion stop")
            _close(obj, root, saw_model)
            return 2
        _remember_fault(obj, block, faults, fault_text, True, root)
        _gate(f"review fault {block}")
        kept = proposal
        kept_labels = labels
        proposal = nxt
        _gate_parts(proposal.get("parts") or [])


def _motion_signature(proposal: dict[str, Any]) -> str:
    parts = proposal.get("parts") or []
    slim = [
        (part.get("id"), part.get("joint"), part.get("axis"), part.get("travel"))
        for part in parts
        if isinstance(part, dict)
    ]
    return json.dumps(slim)


def _gate_parts(parts: list[dict[str, Any]]) -> None:
    if not parts:
        _gate("no movable parts")
        return
    for part in parts:
        _gate(
            f"part {part.get('id')} joint {part.get('joint')} "
            f"axis {part.get('axis')} travel {part.get('travel')}"
        )


def _gate_points(parts: list[dict[str, Any]], labels: dict[str, int]) -> None:
    for part in parts:
        count = int(labels.get(str(part.get("id")), 0))
        if count == 0:
            _gate(f"dropped {part.get('id')}")
        else:
            _gate(f"points {part.get('id')} {count}")
    if "body" in labels:
        _gate(f"points body {labels['body']}")


def _gate_files(obj: str, root: Path | None) -> None:
    folder = object_dir(obj, root)
    parts = json.loads((folder / "parts.json").read_text(encoding="utf-8"))
    ids = [row.get("id") for row in parts.get("parts") or [] if isinstance(row, dict)]
    _gate("parts " + " ".join(str(item) for item in ids))
    urdf = (folder / "asset.urdf").read_text(encoding="utf-8")
    names = []
    for line in urdf.splitlines():
        if "<joint " in line and 'name="' in line:
            names.append(line.split('name="', 1)[1].split('"', 1)[0])
    _gate("urdf_joints " + (" ".join(names) if names else "none"))
    physics = json.loads((folder / "physics.json").read_text(encoding="utf-8"))
    per_part = physics.get("per_part") or []
    total = sum(float(item.get("mass_kg") or 0) for item in per_part if isinstance(item, dict))
    learned = (physics.get("per_gaussian") or {}).get("learned_prior")
    _gate(f"mass_sum {total:.6f} learned_prior {learned}")
    sisters = folder / "sisters" / "objects.json"
    if sisters.is_file():
        record = json.loads(sisters.read_text(encoding="utf-8"))
        _gate(f"trellis {record.get('trellis')}")
    masks = list((folder / "masks").glob("view_*_mask.png"))
    _gate(f"mask_pngs {len(masks)}")


def ask_model(images: list[Path], role: str, prior: dict[str, Any] | None) -> dict[str, Any]:
    """One vision call. No key or no picture means the model was not asked."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key or not images:
        return {"called": False, "object": "", "parts": [], "decision": "pass"}
    try:
        parsed = _post(key, images, role, prior)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return {"called": False, "object": "", "parts": [], "decision": "pass"}
    parsed["called"] = True
    return parsed


def assign_parts(
    obj: str,
    parts: list[dict[str, Any]],
    images: list[Path],
    root: Path | None,
) -> dict[str, int]:
    """Count cloud points near each named image point. Empty names stay one body."""
    if not parts:
        return {}
    from tools.ingest import _project
    from tools.mesh_stages import _mesh_file, _splat_file, _view_camera, read_gaussian_table

    xyz = read_gaussian_table(_splat_file(obj, root))[:, :3]
    geometry = __import__("tools.mesh_to_3dgs", fromlist=["load_geometry"]).load_geometry(_mesh_file(obj, root))
    frames = _camera_frames(images[0].parent if images else renders_dir(obj, "synthetic", root))
    by_name = {item["file"]: item for item in frames}
    counts = {part["id"]: 0 for part in parts}
    claimed = np.zeros(len(xyz), dtype=bool)
    for part in parts:
        view = next((path for path in images if path.name == part["view"]), None)
        if view is None:
            continue
        frame = by_name.get(view.name) or {"azimuth_deg": 0.0, "elevation_deg": 20.0}
        import cv2

        image = cv2.imread(str(view), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        eye, center, radius = _view_camera(geometry, float(frame["azimuth_deg"]), float(frame["elevation_deg"]))
        pixels, depth = _project(xyz, eye, center, width, height)
        target = np.array([part["point"][0] * width, part["point"][1] * height], dtype=np.float64)
        near = np.linalg.norm(pixels - target, axis=1) <= 18.0
        hit = near & (depth > radius * 0.05) & ~claimed
        counts[part["id"]] = int(hit.sum())
        claimed |= hit
    counts["body"] = int((~claimed).sum())
    return counts


def _images(obj: str, root: Path | None) -> tuple[str, list[Path]]:
    doc = load_status(obj, root, create=True)
    if colmap_refused(doc) or (doc.get("input") or {}).get("kind") == "polycam_mesh":
        views = sorted(renders_dir(obj, "synthetic", root).glob("view_*.png"))
        return "ply", views[:8]
    if find_video(obj, root) is None:
        record_absent(obj, root)
        return "stop", []
    frames = sorted(frames_dir(obj, 1, root).glob("*.png"))
    if not frames:
        return "video_no_frames", []
    return "video", frames[:8]


def _validated(raw: dict[str, Any]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw.get("parts") or []:
        if not isinstance(item, dict):
            continue
        joint = str(item.get("joint") or "")
        axis = str(item.get("axis") or "")
        if joint not in {"prismatic", "revolute"} or axis not in _AXES:
            continue
        name = _slug(str(item.get("id") or "part"))
        if name in seen or name == "body":
            continue
        seen.add(name)
        point = item.get("point") or [0.5, 0.5]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            point = [0.5, 0.5]
        x = min(1.0, max(0.0, float(point[0])))
        y = min(1.0, max(0.0, float(point[1])))
        travel = float(item.get("travel") or 0.0)
        if joint == "prismatic":
            travel = min(_PRESS_CAP_M, max(0.0, travel if travel <= 1.0 else travel / 100.0))
        else:
            travel = min(_TURN_CAP_DEG, max(0.0, travel))
        part = {
            "id": name,
            "joint": joint,
            "axis": axis,
            "travel": travel,
            "view": str(item.get("view") or ""),
            "point": [x, y],
        }
        for key in ("material", "friction", "rigidity"):
            if key in item and item[key] not in (None, ""):
                part[key] = item[key]
        if "friction" in part:
            part["friction"] = max(0.0, float(part["friction"]))
        if "rigidity" in part:
            part["rigidity"] = min(1.0, max(0.0, float(part["rigidity"])))
        if "material" in part:
            part["material"] = str(part["material"]).strip()[:80]
        parts.append(part)
    return {
        "called": raw.get("called") is True,
        "object": str(raw.get("object") or "").strip()[:80],
        "parts": parts,
        "decision": str(raw.get("decision") or "pass"),
        "block": str(raw.get("block") or ""),
        "fault": str(raw.get("fault") or ""),
    }


def _write_outputs(obj: str, root: Path | None, proposal: dict[str, Any], counts: dict[str, int]) -> None:
    physics = _physics(obj, root)
    mass = float(physics.get("mass_kg") or 1.0)
    named = [part for part in proposal["parts"] if counts.get(part["id"], 0) > 0]
    total_points = sum(counts.values()) or 1
    used = 0.0
    part_rows = []
    for part in named:
        share = mass * counts[part["id"]] / total_points
        share = round(share, 6)
        used += share
        row = {
            "id": part["id"],
            "parent": "body",
            "joint": part["joint"],
            "axis": part["axis"],
            "travel": part["travel"],
            "vertex_count": counts[part["id"]],
            "mass_kg": share,
            "material": part.get("material", physics.get("material")),
            "friction": part.get("friction", physics.get("friction")),
            "rigidity": part.get("rigidity", physics.get("rigidity")),
        }
        if isinstance(part.get("origin"), (list, tuple)) and len(part["origin"]) >= 3:
            row["origin"] = [round(float(part["origin"][0]), 6), round(float(part["origin"][1]), 6), round(float(part["origin"][2]), 6)]
        part_rows.append(row)
    body_mass = round(max(0.0, mass - used), 6)
    parts_doc = {
        "segmenter": "vlm" if named else "single_mesh",
        "partfield": False,
        "source": "vlm" if named else "single_mesh",
        "object": proposal.get("object") or "",
        "parts": [
            {
                "id": "body",
                "parent": None,
                "joint": None,
                "vertex_count": counts.get("body", total_points),
                "mass_kg": body_mass,
                "material": physics.get("material"),
                "friction": physics.get("friction"),
                "rigidity": physics.get("rigidity"),
            },
            *part_rows,
        ],
    }
    folder = object_dir(obj, root)
    (folder / "parts.json").write_text(json.dumps(parts_doc, indent=2) + "\n", encoding="utf-8")
    (folder / "asset.urdf").write_text(_urdf_text(obj, root, parts_doc), encoding="utf-8")
    physics["per_part"] = [
        {
            "id": row["id"],
            "mass_kg": row["mass_kg"],
            "material": row["material"],
            "friction": row["friction"],
            "rigidity": row["rigidity"],
        }
        for row in parts_doc["parts"]
    ]
    per = dict(physics.get("per_gaussian") or {})
    per["learned_prior"] = False
    physics["per_gaussian"] = per
    (folder / "physics.json").write_text(json.dumps(physics, indent=2) + "\n", encoding="utf-8")
    masks = folder / "masks" / "object.json"
    if proposal.get("object") and masks.is_file():
        record = json.loads(masks.read_text(encoding="utf-8"))
        record["vlm_object"] = proposal["object"]
        masks.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _write_motion(
        obj,
        root,
        {
            "source": "vlm",
            "model": _MODEL if proposal.get("called") else None,
            "object": proposal.get("object") or "",
            "parts": proposal["parts"],
            "applied": [row["id"] for row in part_rows],
        },
    )


def _urdf_text(obj: str, root: Path | None, parts_doc: dict[str, Any]) -> str:
    from tools.common.status import StatusError
    from tools.mesh_stages import _mesh_file

    try:
        mesh_name = _mesh_file(obj, root).name
    except (OSError, KeyError, TypeError, StatusError):
        mesh_name = "mesh.ply"
    # mesh path is resolved by caller via status; use the helper with root later
    lines = [
        '<?xml version="1.0"?>',
        f'<robot name="{_slug(obj)}">',
    ]
    for row in parts_doc["parts"]:
        lines.append(f'  <link name="{row["id"]}">')
        lines.append("    <inertial>")
        lines.append(f'      <mass value="{row["mass_kg"]}"/>')
        lines.append('      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>')
        lines.append("    </inertial>")
        if row["id"] == "body":
            lines.append("    <visual>")
            lines.append(f'      <geometry><mesh filename="{mesh_name}"/></geometry>')
            lines.append("    </visual>")
            lines.append("    <collision>")
            lines.append(f'      <geometry><mesh filename="{mesh_name}"/></geometry>')
            lines.append("    </collision>")
        lines.append("  </link>")
    for row in parts_doc["parts"]:
        if not row.get("joint"):
            continue
        axis = _AXES[row["axis"]]
        origin = row.get("origin") or [0.0, 0.0, 0.0]
        if row["joint"] == "prismatic":
            upper = f"{float(row['travel']):.6f}"
            kind = "prismatic"
        else:
            upper = f"{float(row['travel']) * 3.14159265 / 180.0:.6f}"
            kind = "revolute"
        lines.extend(
            [
                f'  <joint name="{row["id"]}_joint" type="{kind}">',
                '    <parent link="body"/>',
                f'    <child link="{row["id"]}"/>',
                f'    <origin xyz="{float(origin[0]):.6f} {float(origin[1]):.6f} {float(origin[2]):.6f}" rpy="0 0 0"/>',
                f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>',
                f'    <limit lower="0" upper="{upper}" effort="1" velocity="0.1"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def _physics(obj: str, root: Path | None) -> dict[str, Any]:
    path = object_dir(obj, root) / "physics.json"
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            return loaded
    return {
        "material": "unknown",
        "friction": 0.5,
        "rigidity": 0.8,
        "mass_kg": 1.0,
        "density_kg_m3": 1000.0,
        "per_gaussian": {"learned_prior": False},
    }


def _camera_frames(view_dir: Path) -> list[dict[str, Any]]:
    path = view_dir / "cameras.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if isinstance(frames, list):
            return frames
    return []


def _restart_prompt(obj: str, root: Path | None, review: dict[str, Any]) -> None:
    prompt = str(review.get("prompt") or review.get("object") or "").strip()[:80]
    if not prompt:
        return
    path = object_dir(obj, root) / "masks" / "object.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    record["prompt"] = prompt
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _restart_up_axis(obj: str, root: Path | None, review: dict[str, Any]) -> None:
    doc = load_status(obj, root, create=True)
    if (doc.get("input") or {}).get("kind") != "polycam_mesh":
        return
    raw = review.get("up_axis")
    if not isinstance(raw, (list, tuple)) or len(raw) < 3:
        return
    axis = np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-8:
        return
    axis = axis / norm
    from tools.mesh_stages import _rotation_to_z

    path = object_dir(obj, root) / "align.json"
    record = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    record["up_axis"] = [round(float(value), 6) for value in axis]
    record["rotation_to_z"] = [[round(float(value), 6) for value in row] for row in _rotation_to_z(axis)]
    record["photometric"] = record.get("photometric") or "skipped"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    report = dict((doc.get("align") or {}).get("report") or {})
    report.update({"up_axis": record["up_axis"], "photometric": record["photometric"]})
    set_path(doc, "align.report", report)
    save_status(obj, doc, root)


def _restart_lift(obj: str, root: Path | None) -> None:
    from tools.mesh_stages import project_masks

    project_masks(obj, root)
    path = object_dir(obj, root) / "masks" / "object.json"
    if not path.is_file():
        return
    record = json.loads(path.read_text(encoding="utf-8"))
    record["infill"] = "skipped"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def _write_motion(obj: str, root: Path | None, payload: dict[str, Any]) -> None:
    path = object_dir(obj, root) / "motion.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _remember_fault(obj: str, block: str, attempt: int, fault: str, restarted: bool, root: Path | None) -> None:
    folder = object_dir(obj, root)
    path = folder / "agents.json"
    faults: list[dict[str, Any]] = []
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and isinstance(loaded.get("faults"), list):
            faults = [item for item in loaded["faults"] if isinstance(item, dict)]
    faults.append(
        {"block": block, "attempt": attempt, "fault": fault.replace("\n", " ")[:240], "restarted": restarted}
    )
    write_agents(obj, root, faults=faults)


def _close(obj: str, root: Path | None, saw_model: bool) -> None:
    write_agents(obj, root, ran=True, visual_review=True if saw_model else None)
    from tools.changelog import write_changelog

    write_changelog(obj, root)


def _post(api_key: str, images: list[Path], role: str, prior: dict[str, Any] | None) -> dict[str, Any]:
    if role == "propose":
        instructions = (
            "Reply with JSON only. "
            '{"object":"short name","parts":[{"id":"shutter","joint":"prismatic","axis":"+z",'
            '"travel":0.005,"view":"view_000.png","point":[0.5,0.4]}]} '
            "joint is prismatic or revolute. axis is +x -x +y -y +z -z. "
            "travel is metres for a press or degrees for a turn. "
            "Name only pieces that move in real life."
        )
    else:
        instructions = (
            "Review these parts. Reply JSON "
            '{"decision":"pass"} or {"decision":"fault","block":"B11","fault":"short reason"}.'
        )
    content: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
    if prior and role == "review":
        content.append({"type": "text", "text": json.dumps({"parts": prior.get("parts"), "object": prior.get("object")})})
    for image in images[:4]:
        encoded = base64.standard_b64encode(image.read_bytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    payload = {
        "model": _MODEL,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        _URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    text = body["choices"][0]["message"]["content"]
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise TypeError("vision JSON must be an object")
    return parsed


def _slug(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name.strip().lower())
    if not cleaned or cleaned[0].isdigit():
        return f"part_{cleaned}"
    return cleaned


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Name movable parts from pictures")
    parser.add_argument("--obj", default="scan")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    return run_motion(args.obj, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
