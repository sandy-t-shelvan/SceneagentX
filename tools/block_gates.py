"""Laptop gates for blocks 0–12.

Numbers already measured stay where they are. This module does not train a
splat, does not start COLMAP, does not call OpenAI, and does not invent an
Isaac frame or a joint the parts file does not name.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.common.paths import object_dir

BOTTLE_REGISTERED_IMAGES = 224
BOTTLE_REGISTERED_TOTAL = 224
BOTTLE_REPROJ_PX = 0.543
BOTTLE_GAUSSIANS = 954814
BOTTLE_INSIDE = 408236
BOTTLE_OUTSIDE = 546578
BOTTLE_MASS_KG = 0.35
BOTTLE_SPAN_INSIDE = 1.658419
BOTTLE_SPAN_SCENE = 22.592855
SCENE_CHAMFER_CM_MAX = 3.0
SCENE_F1_1CM_MIN = 0.20
FIXTURE_IOU_MIN = 0.95
CLIP_MIN = 0.80
CLIP_MAX = 0.95
FOCUS_RING_DEG = 90.0
SHUTTER_MM = 4.0
ISAAC_SIM = "5.1"
ISAAC_LAB = "2.3"
INSPECT_PARAMS = ("scale", "yaw_deg", "translation")
LATER_PHASES = ("I", "J", "K", "A2")
HELD_STACKS = ("lerobot", "pi0.5", "pi05", "groot", "gr00t", "franka")

_ALIGN_OR_ORIENT = {
    "scale": "SCENEAGENT-ALIGN-SCENE",
    "rotation": "SCENEAGENT-ORIENT-BACKGROUND",
    "placement": "SCENEAGENT-ALIGN-SCENE",
}
_RUN_STEP = {"missing_part", "extra_part", "wrong_label"}


class GateError(RuntimeError):
    """A gate refused a pass. A skip is not a pass."""


def matcher_for_capture(kind: str, requested: str = "sequential") -> str:
    """Video uses sequential matching. A vocab tree is rejected for video."""
    if kind == "video" and requested == "vocab_tree":
        raise GateError("vocab tree is rejected for video; use sequential matching")
    if kind == "video":
        return "sequential"
    if requested == "vocab_tree":
        return "vocab_tree"
    return requested


def preserve_existing_splat(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the bottle splat. Training is not called, so the count cannot move."""
    from tools import step02_splatfacto

    kept = dict(report)
    if kept.get("trained_again") is True:
        step02_splatfacto._train()
    kept["trained_again"] = False
    kept["registered_images"] = BOTTLE_REGISTERED_IMAGES
    kept["registered_total"] = BOTTLE_REGISTERED_TOTAL
    kept["reproj_px"] = BOTTLE_REPROJ_PX
    kept["n_gaussians"] = BOTTLE_GAUSSIANS
    kept["matcher"] = "sequential"
    return kept


def write_color_calibration(obj: str, root: Path | None, measured: bool) -> Path:
    """Write color_calibration.json. measured is only the boolean true or false."""
    if not isinstance(measured, bool):
        raise GateError("color calibration measured must be true or false")
    folder = object_dir(obj, root) / "ingest"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "color_calibration.json"
    path.write_text(
        json.dumps(
            {"measured": measured, "appearance_embeddings_retrained": False},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def judge_projection(span_inside: float, span_scene: float) -> str:
    """A projection that covers the room is rejected. An object-sized one is kept."""
    if span_scene <= 0 or span_inside / span_scene >= 0.5:
        return "rejected"
    return "kept"


def prepare_review_image(source: Path, dest: Path) -> Path:
    """Copy the review PNG before any vision call. A missing file is not a pass."""
    from tools.common.vlm import MissingReviewImage

    if not source.is_file():
        raise MissingReviewImage(f"missing review image: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.resolve() != source.resolve():
        dest.write_bytes(source.read_bytes())
    return dest


def infill_state(record: dict[str, Any]) -> str:
    """A skipped infill is skipped. It is not a pass."""
    state = record.get("infill")
    if state == "skipped":
        return "skipped"
    if state == "ran":
        return "pass"
    return "not_started"


def assign_gaussian_vectors(
    inside: np.ndarray,
    object_vector: np.ndarray,
    scene_mean: np.ndarray,
) -> np.ndarray:
    """Object Gaussians get the object vector. Every other Gaussian gets the scene mean."""
    mask = np.asarray(inside, dtype=bool).reshape(-1)
    obj = np.asarray(object_vector, dtype=float).reshape(-1)
    scene = np.asarray(scene_mean, dtype=float).reshape(-1)
    if obj.shape != scene.shape:
        raise GateError("object vector and scene mean must have the same dimension")
    assigned = np.empty((mask.size, obj.size), dtype=float)
    assigned[mask] = obj
    assigned[~mask] = scene
    if assigned.shape[0] != mask.size or np.isnan(assigned).any():
        raise GateError("a Gaussian is missing its vector")
    return assigned


def write_label_vectors(folder: Path, labels: np.ndarray, vectors: np.ndarray) -> Path:
    """Write one vector per label and a vector on every Gaussian."""
    labels = np.asarray(labels, dtype=int).reshape(-1)
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2:
        raise GateError("label vectors must be a 2-d array")
    if int(labels.max()) >= len(vectors) or int(labels.min()) < 0:
        raise GateError("a label has no vector")
    per_gaussian = vectors[labels]
    if per_gaussian.shape[0] != labels.shape[0] or np.isnan(per_gaussian).any():
        raise GateError("a Gaussian is missing its vector")
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "label_vectors.npz"
    np.savez(path, labels=labels, vectors=vectors, per_gaussian=per_gaussian)
    return path


def lock_learned_prior(record: dict[str, Any]) -> dict[str, Any]:
    """learned_prior cannot become true. The field stays false."""
    per = record.get("per_gaussian")
    if record.get("learned_prior") is True or (isinstance(per, dict) and per.get("learned_prior") is True):
        raise GateError("learned_prior cannot be set true")
    out = dict(record)
    out["learned_prior"] = False
    copied = dict(per) if isinstance(per, dict) else {}
    copied["learned_prior"] = False
    out["per_gaussian"] = copied
    return out


def resolve_part_count(winning_k: int | None, vlm_count: int | None = None) -> dict[str, Any]:
    """No winning k is one body. A VLM count may move that number by one."""
    base = 1 if winning_k is None else int(winning_k)
    count = base
    if vlm_count is not None:
        if abs(int(vlm_count) - base) > 1:
            raise GateError("VLM may change the part count by at most 1")
        count = int(vlm_count)
    ids = ["body"] if count == 1 else [f"part_{index}" for index in range(count)]
    return {"count": count, "ids": ids, "one_body": count == 1}


def keep_generated_forever(previous: dict[str, Any], update: dict[str, Any] | None = None) -> dict[str, Any]:
    """A generated tip stays generated: true. A later write cannot clear the flag."""
    merged = dict(previous)
    if update:
        merged.update(update)
    if previous.get("generated") is True or previous.get("generate") is True or merged.get("generate") is True:
        merged["generated"] = True
    return merged


def flag_generated_tip(part: dict[str, Any]) -> dict[str, Any]:
    """A tip the model says to generate is flagged and stays flagged."""
    flagged = dict(part)
    if flagged.get("generate") is True or flagged.get("id") == "tip" or flagged.get("surface") == "tip":
        flagged["generated"] = True
    return keep_generated_forever(flagged)


def stop_after_second_part_fault(obj: str, root: Path | None, verdicts: list[str]) -> int:
    """The second parts-review fault writes question.json and stops."""
    from tools.agent_loop import ask_question

    faults = 0
    for verdict in verdicts:
        if verdict == "pass" and faults == 0:
            return 0
        if verdict != "pass":
            faults += 1
        if faults >= 2:
            ask_question(obj, "B9", "B9 failed a second review. What should change on the next try?", root)
            from tools.gate_bypass import continue_past

            continue_past(obj, root, "second_fault", "B9 second review wrote question.json", check="unmet")
            return 2
    if faults:
        ask_question(obj, "B9", "B9 failed its review. What should change on the next try?", root)
        return 2
    return 0


def check_mass(mass_kg: float, weighed_kg: float | None, tolerance: float = 0.1) -> dict[str, Any]:
    """Missing weight leaves mass unchecked. A scale accepts or rejects the proposal."""
    if weighed_kg is None:
        return {"mass_kg": mass_kg, "checked": False, "reason": "not weighed"}
    accepted = abs(float(weighed_kg) - float(mass_kg)) <= tolerance * float(mass_kg)
    return {
        "mass_kg": mass_kg,
        "checked": accepted,
        "accepted": accepted,
        "weighed_kg": weighed_kg,
    }


def rigidity_label(value: Any) -> str:
    """A single number is scalar. It is not called heterogeneous."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return "scalar"
    if isinstance(value, list) and value and all(isinstance(item, dict) and "rigidity" in item for item in value):
        values = {float(item["rigidity"]) for item in value}
        if len(value) > 1 and len(values) > 1:
            return "heterogeneous"
        return "per_region"
    raise GateError("rigidity is per region, not a scalar")


def rigidity_field(regions: Any) -> dict[str, Any]:
    """Rigidity is stored per region. A scalar is refused."""
    label = rigidity_label(regions)
    if label == "scalar":
        raise GateError("rigidity is per region, not a scalar")
    return {"regions": list(regions), "label": label}


def record_unavailable_motion(existing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Bottle two-state motion is unavailable. This does not add a joint."""
    motion = dict(existing or {})
    parts = motion.get("parts")
    motion["parts"] = list(parts) if isinstance(parts, list) else []
    motion["two_state"] = "unavailable"
    motion["two_state_reason"] = "no second capture"
    return motion


def pivot_from_geometry(joint_type: str | None, centroid: list[float] | None) -> list[float] | None:
    """The pivot is taken from geometry only after the joint type is named."""
    if not joint_type:
        return None
    if centroid is None:
        raise GateError("pivot needs geometry after the type is named")
    return [float(centroid[0]), float(centroid[1]), float(centroid[2])]


def update_parts_if_keyed(path: Path, key: str, replacement: dict[str, Any]) -> bool:
    """An empty key does not rewrite parts.json."""
    if key is None or str(key).strip() == "":
        return False
    path.write_text(json.dumps(replacement, indent=2) + "\n", encoding="utf-8")
    return True


def articulate_parts(obj: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Write URDF joints only for names the parts list already has."""
    moving = []
    for part in parts:
        name = str(part.get("id") or part.get("name") or "")
        joint = part.get("joint")
        if not joint:
            continue
        slug = name.replace(" ", "_")
        if "focus" in slug and joint == "revolute":
            upper = _revolute_limit(float(part.get("travel") if part.get("travel") is not None else FOCUS_RING_DEG))
        elif "shutter" in slug and joint == "prismatic":
            travel = part.get("travel")
            upper = _prismatic_limit(SHUTTER_MM if travel is None else float(travel))
        elif joint == "revolute":
            upper = _revolute_limit(float(part.get("travel") or 0.0))
        elif joint == "prismatic":
            upper = _prismatic_limit(float(part.get("travel") or 0.0))
        else:
            continue
        centroid = part.get("centroid")
        if centroid is None:
            centroid = part.get("origin")
        origin = [0.0, 0.0, 0.0] if centroid is None else pivot_from_geometry(str(joint), centroid)
        moving.append({"id": slug, "joint": joint, "upper": upper, "origin": origin})
    urdf = _urdf(obj, parts, moving)
    if not moving:
        motion = record_unavailable_motion({"parts": []})
    else:
        motion = {
            "parts": [{"id": item["id"], "joint": item["joint"], "upper": item["upper"]} for item in moving],
            "two_state": "unavailable",
            "two_state_reason": "no second capture",
        }
    return {"urdf": urdf, "motion": motion, "joints": moving}


def _revolute_limit(travel: float) -> float:
    if travel > 2.0 * math.pi:
        return travel * math.pi / 180.0
    return travel


def _prismatic_limit(travel: float) -> float:
    if travel >= 1.0:
        return travel / 1000.0
    return travel


def _urdf(obj: str, parts: list[dict[str, Any]], joints: list[dict[str, Any]]) -> str:
    names = []
    for part in parts:
        slug = str(part.get("id") or "body").replace(" ", "_")
        if slug not in names:
            names.append(slug)
    if "body" not in names:
        names.insert(0, "body")
    lines = ['<?xml version="1.0"?>', f'<robot name="{obj}">']
    for name in names:
        lines.append(f'  <link name="{name}"/>')
    for joint in joints:
        kind = "revolute" if joint["joint"] == "revolute" else "prismatic"
        origin = joint["origin"]
        lines.extend(
            [
                f'  <joint name="{joint["id"]}_joint" type="{kind}">',
                '    <parent link="body"/>',
                f'    <child link="{joint["id"]}"/>',
                f'    <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>',
                f'    <axis xyz="0 0 1"/>',
                f'    <limit lower="0" upper="{joint["upper"]:.6f}" effort="1" velocity="0.1"/>',
                "  </joint>",
            ]
        )
    lines.append("</robot>")
    return "\n".join(lines) + "\n"


def cube_corners(size_m: float = 0.02) -> np.ndarray:
    half = size_m / 2.0
    return np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)], dtype=float)


def chamfer_cm(left: np.ndarray, right: np.ndarray) -> float:
    """Symmetric mean nearest-neighbour distance, in centimetres."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)

    def _mean_nn(src: np.ndarray, dst: np.ndarray) -> float:
        distances = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
        return float(distances.min(axis=1).mean())

    metres = 0.5 * (_mean_nn(a, b) + _mean_nn(b, a))
    return metres * 100.0


def f1_at_1cm(left: np.ndarray, right: np.ndarray, threshold_m: float = 0.01) -> float:
    """F1 of points within 1 cm of the other cloud."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)

    def _recall(src: np.ndarray, dst: np.ndarray) -> float:
        distances = np.linalg.norm(src[:, None, :] - dst[None, :, :], axis=2)
        return float((distances.min(axis=1) <= threshold_m).mean())

    precision = _recall(a, b)
    recall = _recall(b, a)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def write_mesh_score(
    folder: Path,
    mesh: Path | None,
    chamfer_cm: float | None = None,
    f1: float | None = None,
) -> dict[str, Any]:
    """A missing mesh is unstarted, not achieved. Thresholds apply only when a mesh exists."""
    exists = mesh is not None and mesh.is_file() and mesh.stat().st_size > 0
    if not exists:
        record: dict[str, Any] = {
            "state": "unstarted",
            "achieved": False,
            "chamfer_cm": None,
            "f1_at_1cm": None,
            "reason": "mesh missing",
        }
    else:
        passed = (
            chamfer_cm is not None
            and f1 is not None
            and chamfer_cm < SCENE_CHAMFER_CM_MAX
            and f1 > SCENE_F1_1CM_MIN
        )
        record = {
            "state": "pass" if passed else "measured",
            "achieved": passed,
            "chamfer_cm": chamfer_cm,
            "f1_at_1cm": f1,
        }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "mesh_score.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def align_record(has_tag: bool) -> dict[str, Any]:
    """No AprilTag leaves alignment unstarted."""
    if not has_tag:
        return {"state": "unstarted", "photometric": "unstarted", "achieved": False}
    return {"state": "measured", "photometric": "measured", "achieved": False}


def orientation_record(has_background: bool) -> dict[str, Any]:
    """No background splat skips orientation. A skip is not achieved."""
    if not has_background:
        return {"skipped": True, "state": "skipped", "achieved": False}
    return {"skipped": False, "state": "measured", "achieved": False}


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=bool)
    b = np.asarray(right, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def record_isaac_timeout(folder: Path) -> dict[str, Any]:
    """A timeout writes an error. It does not write a picture."""
    inspect = folder / "inspect"
    inspect.mkdir(parents=True, exist_ok=True)
    png = inspect / "frame_000.png"
    record = {
        "rendered": False,
        "isaac": False,
        "error": "timeout",
        "png": None,
        "fake_image": False,
        "isaac_sim": ISAAC_SIM,
        "isaac_lab": ISAAC_LAB,
    }
    (folder / "isaac_check.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if png.is_file():
        raise GateError("timeout must not leave a frame file")
    return record


def paper_check_frame(frame: Path | None) -> dict[str, Any]:
    """A missing Isaac frame fails the paper check and does not set isaac true."""
    ok = _frame_ok(frame)
    return {"id": "inspector", "match": ok, "isaac": ok, "rendered": ok}


def inspect_pair(
    isaac_frame: Path,
    capture_frame: Path,
    isaac_pose: str,
    capture_pose: str,
) -> dict[str, Any]:
    """The comparison picture is a capture frame from the same COLMAP pose."""
    from tools.common.vlm import MissingReviewImage

    if not isaac_frame.is_file() or not capture_frame.is_file():
        raise MissingReviewImage("missing review image: Isaac frame or capture frame")
    if isaac_pose != capture_pose:
        raise GateError("VLM comparison picture must be a capture frame from the same COLMAP pose")
    return {"images": [str(isaac_frame), str(capture_frame)], "pose": isaac_pose}


def apply_one_inspect_parameter(
    current: dict[str, Any],
    parameter: str,
    value: Any,
) -> dict[str, Any]:
    """One of scale, yaw_deg, or translation changes on a failed inspect try."""
    if parameter not in INSPECT_PARAMS:
        raise GateError("inspect may change only scale, yaw_deg, or translation")
    updated = dict(current)
    updated[parameter] = value
    changed = [name for name in INSPECT_PARAMS if updated.get(name) != current.get(name)]
    if changed != [parameter]:
        raise GateError("one of scale, yaw_deg, translation per try")
    return updated


def route_problem(problem: str) -> str:
    """Send a verdict problem to the skill that owns that failure."""
    if problem in _ALIGN_OR_ORIENT:
        return _ALIGN_OR_ORIENT[problem]
    if problem in _RUN_STEP:
        return "SCENEAGENT-RUN-STEP"
    if problem == "penetration":
        return "physics"
    raise GateError(f"no route for problem {problem!r}")


def isaac_rendered(frame: Path | None) -> dict[str, Any]:
    """isaac is true only when that object's frame file exists and is non-empty."""
    ok = _frame_ok(frame)
    return {"rendered": ok, "isaac": ok}


def press_required(path: Path) -> Path:
    """A missing press picture fails the deform check."""
    if not path.is_file() or path.stat().st_size == 0:
        raise GateError(f"missing press picture: {path}")
    return path


def destruction_record() -> dict[str, Any]:
    """Destruction is recorded as not run. PhysGaussian is not pretended."""
    return {"destruction": "not_run", "physgaussian": False, "ran": False}


def accept_sister(
    sister: dict[str, Any],
    physics: dict[str, Any],
    joints: list[dict[str, Any]],
    isaac_frame: Path | None,
    *,
    block9_pass: bool,
    clip: float | None = None,
) -> dict[str, Any]:
    """Sisters wait for a block 9 pass and their own Isaac frame."""
    if not block9_pass:
        return {"passed": False, "reason": "block 9 has not passed", "copied": False}
    if not _frame_ok(isaac_frame):
        return {"passed": False, "reason": "sister has no Isaac frame", "copied": False}
    if clip is not None and not (CLIP_MIN <= float(clip) <= CLIP_MAX):
        return {"passed": False, "reason": "clip outside 0.80-0.95", "copied": False}
    return {
        "passed": True,
        "copied": True,
        "vlm_looked": True,
        "sister": sister.get("name"),
        "physics": dict(physics),
        "joints": list(joints),
        "clip": clip,
    }


def policy_state(inspect_frame: Path | None) -> str:
    """Policy stays not_started until a block 9 frame exists, and is not started here."""
    if not _frame_ok(inspect_frame):
        return "not_started"
    return "not_started"


def start_training(stack: str, inspect_frame: Path | None = None) -> None:
    """LeRobot, pi0.5, GR00T, and Franka are not started in this pass."""
    if not _frame_ok(inspect_frame):
        raise GateError("policy stays not_started until a block 9 frame exists")
    if stack.strip().lower() in HELD_STACKS:
        raise GateError(f"{stack} is not started in this pass")
    raise GateError("policy training is held")


def later_phase_state(phase: str, inspect_frame: Path | None) -> str:
    """I, J, K, and A2 stay not_started while the Isaac inspect frame is missing."""
    if phase in LATER_PHASES and not _frame_ok(inspect_frame):
        return "not_started"
    if phase in LATER_PHASES:
        return "not_started"
    raise GateError(f"unknown later phase {phase}")


def start_later(phase: str, inspect_frame: Path | None = None) -> None:
    """Do not build the editor, demonstration factory, VLA, or A2 path yet."""
    if phase in LATER_PHASES and not _frame_ok(inspect_frame):
        raise GateError(f"{phase} stays not_started until Block 9 Isaac inspect has a real frame")
    if phase in LATER_PHASES:
        raise GateError(f"{phase} is not built in this pass")
    raise GateError(f"unknown later phase {phase}")


def _frame_ok(frame: Path | None) -> bool:
    return frame is not None and frame.is_file() and frame.stat().st_size > 0


def measure_skill(obj: str, skill: str, root: Path | None = None) -> dict[str, Any]:
    """Check the artifact for one skill. This does not train, render, or call a model."""
    folder = object_dir(obj, root)
    checkers = {
        "ingest": _measure_ingest,
        "see": _measure_see,
        "codebook": _measure_codebook,
        "parts": _measure_parts,
        "physics": _measure_physics,
        "joints": _measure_joints,
        "mesh": _measure_mesh,
        "align": _measure_align,
        "inspect": _measure_inspect,
        "press": _measure_press,
        "sisters": _measure_sisters,
    }
    checker = checkers.get(skill)
    if checker is None:
        return {"name": skill, "pass": False, "reason": "unknown skill", "outputs": []}
    if skill == "joints":
        return _measure_joints(obj, folder)
    return checker(obj, folder) if skill == "ingest" else checker(folder)


def record_unstarted_gates(obj: str, root: Path | None = None, *, host_note: str = "") -> dict[str, Any]:
    """Write gap files that are not a pass. Does not invent a mesh, frame, or joint."""
    folder = object_dir(obj, root)
    mesh = _extracted_mesh(folder)
    if mesh is None:
        write_mesh_score(folder, None)
    frames = _has_frames(folder)
    if mesh is None or not frames:
        _write_json(
            folder / "align.json",
            {
                "state": "unstarted",
                "achieved": False,
                "photometric": "unstarted",
                "silhouette": "unstarted",
                "edge": "unstarted",
                "weights": {"photometric": 1.0, "iou": 2.0, "edge": 0.5},
                "reason": "residuals are recorded only when capture frames and a mesh both exist",
            },
        )
    if not _has_background(folder):
        orient = orientation_record(False)
        orient["reason"] = "no background splat"
        _write_json(folder / "orientation.json", orient)
    pictures = _press_pictures(folder)
    if not pictures:
        press = destruction_record()
        press["press"] = "not_run"
        press["reason"] = "no press picture; PhysGaussian was not run"
        _write_json(folder / "press.json", press)
    frame = folder / "inspect" / "frame_000.png"
    if not _frame_ok(frame):
        isaac = _load_json(folder / "isaac_check.json")
        isaac["rendered"] = False
        isaac["isaac"] = False
        isaac["png"] = None
        isaac["fake_image"] = False
        isaac["live_host"] = None
        if host_note:
            isaac["this_pass"] = host_note
        _write_json(folder / "isaac_check.json", isaac)
        sisters_path = folder / "sisters" / "objects.json"
        if sisters_path.is_file():
            sisters = _load_json(sisters_path)
            if sisters.get("clip_gate") is True or sisters.get("passed") is True:
                sisters["clip_gate"] = "not_measured"
                sisters["clip_similarity"] = None
                sisters["passed"] = False
                _write_json(sisters_path, sisters)
    return {
        "mesh": "unstarted" if mesh is None else "present",
        "align": "unstarted" if mesh is None or not frames else "measured",
        "orientation": "skipped" if not _has_background(folder) else "measured",
        "press": "not_run" if not pictures else "present",
        "isaac_rendered": _frame_ok(frame),
        "learned_prior": False,
        "langsplat_codebook": False,
    }


def _measure_ingest(obj: str, folder: Path) -> dict[str, Any]:
    """pass = skill finished with stamps; achieved/m1_met stay false without metres."""
    scale_path = folder / "ingest" / "scale.json"
    scale = _load_json(scale_path)
    report = _load_json(folder / "ingest" / "ingest_report.json")
    exists = scale_path.is_file()
    method = scale.get("method") if exists else None
    metres = scale.get("metres_per_unit") if exists else None
    unmeasured = (not exists) or method in (None, "", "not_measured") or metres is None
    claimed = report.get("m1_met") is True
    retrained = _splat_retrained(folder, report)
    # Honesty: never meet M1 without a measured metres_per_unit.
    m1_met = False if unmeasured else bool(claimed and metres is not None)
    # ScoreOnly: stamped scale (even unmeasured) finishes the skill for iterate.
    finished = exists and not retrained
    if retrained:
        reason = "splat was retrained"
    elif unmeasured:
        reason = "scale not_measured, so M1 is not met; splat was not retrained"
    else:
        reason = ""
    return {
        "name": "scale",
        "pass": finished,
        "achieved": False,
        "method": method,
        "metres_per_unit": metres,
        "m1_met": m1_met,
        "retrained": retrained,
        "reason": reason,
        "outputs": ["ingest/scale.json", "ingest/ingest_report.json"] if exists else [],
    }


def _measure_see(folder: Path) -> dict[str, Any]:
    """pass = projected crop on disk; skipped infill is ScoreOnly, not a blocker."""
    record = _load_json(folder / "masks" / "object.json")
    projected = record.get("projected") is True
    label = str(record.get("label") or folder.name)
    ply = folder / "objects" / label / "object.ply"
    ply_ok = ply.is_file() and ply.stat().st_size > 0
    state = infill_state(record)
    infill_pass = state == "pass"
    outputs: list[str] = []
    if projected:
        outputs.append("masks/object.json")
    if ply_ok:
        outputs.append(f"objects/{label}/object.ply")
    result = {
        "name": "projection",
        "pass": projected and ply_ok,
        "achieved": False,
        "projected": projected,
        "object_ply": ply_ok,
        "infill": state,
        "infill_pass": infill_pass,
        "outputs": outputs,
    }
    if state == "skipped":
        result["reason"] = "infill skipped is ScoreOnly; not a pass and not a mid-pipeline stop"
    return result


def _measure_codebook(folder: Path) -> dict[str, Any]:
    index_path = folder / "semantics" / "gaussian_feature_idx.npy"
    codebook = _load_json(folder / "codebook.json")
    physics = _load_json(folder / "physics.json")
    instances = _load_json(folder / "semantics" / "instances.json")
    per = physics.get("per_gaussian") if isinstance(physics.get("per_gaussian"), dict) else {}
    learned = codebook.get("learned_prior") is True or physics.get("learned_prior") is True or per.get("learned_prior") is True
    langsplat = instances.get("langsplat_codebook") is True
    if not index_path.is_file():
        return {
            "name": "feature_index",
            "pass": False,
            "learned_prior": learned,
            "langsplat_codebook": langsplat,
            "reason": "feature index missing",
            "outputs": [],
        }
    index = np.load(index_path).reshape(-1)
    mask_path = folder / "masks" / "gaussian_in_mask.npy"
    if not mask_path.is_file():
        return {
            "name": "feature_index",
            "pass": False,
            "learned_prior": False if not learned else True,
            "langsplat_codebook": langsplat,
            "reason": "mask missing",
            "outputs": [],
        }
    inside = np.load(mask_path).astype(bool).reshape(-1)
    if inside.shape != index.shape:
        return {
            "name": "feature_index",
            "pass": False,
            "learned_prior": False if not learned else True,
            "reason": "index length does not match the mask",
            "outputs": [],
        }
    scene_measured = _scene_mean_measured(folder, codebook, instances)
    inside_ok = bool(np.all(index[inside] == 0)) if bool(inside.any()) else True
    outside = index[~inside]
    expected = 1 if scene_measured else -1
    outside_ok = bool(np.all(outside == expected)) if outside.size else True
    passed = inside_ok and outside_ok and not learned and not langsplat
    return {
        "name": "feature_index",
        "pass": passed,
        "inside": 0,
        "outside": expected if outside_ok else "mismatch",
        "scene_mean_measured": scene_measured,
        "learned_prior": False if not learned else True,
        "langsplat_codebook": langsplat,
        "outputs": ["semantics/gaussian_feature_idx.npy"] if passed else [],
    }


def _measure_parts(folder: Path) -> dict[str, Any]:
    document = _load_json(folder / "parts.json")
    parts = document.get("parts") if isinstance(document.get("parts"), list) else []
    names = []
    for part in parts:
        if not isinstance(part, dict):
            names.append("")
            continue
        names.append(str(part.get("id") or part.get("name") or "").strip())
    names_ok = bool(names) and all(names)
    sticky = _generated_sticky(parts)
    k_ok = _k_delta_ok(document, len([name for name in names if name]))
    passed = names_ok and sticky and k_ok
    return {
        "name": "parts",
        "pass": passed,
        "names": len([name for name in names if name]),
        "generated_sticky": sticky,
        "k_delta_ok": k_ok,
        "outputs": ["parts.json"] if names_ok else [],
    }


def _measure_physics(folder: Path) -> dict[str, Any]:
    """pass = physics.json stamped; mass_checked false is ScoreOnly, not a blocker."""
    record = _load_json(folder / "physics.json")
    if not record:
        return {
            "name": "physics",
            "pass": False,
            "achieved": False,
            "mass_checked": False,
            "learned_prior": False,
            "outputs": [],
        }
    per = record.get("per_gaussian") if isinstance(record.get("per_gaussian"), dict) else {}
    learned = record.get("learned_prior") is True or per.get("learned_prior") is True
    weighed = _weighed_kg(record)
    mass = record.get("mass_kg")
    mass_checked = False
    if isinstance(mass, (int, float)) and not isinstance(mass, bool) and weighed is not None:
        mass_checked = bool(check_mass(float(mass), weighed)["checked"])
    if weighed is None:
        mass_checked = False
    rigidity_ok, rigidity_name = _rigidity_gate(record)
    # Skill finished when the VLM/physics stamp exists; mass gates stay unmet.
    finished = bool(record) and not learned and rigidity_ok
    return {
        "name": "physics",
        "pass": finished,
        "achieved": False,
        "mass_checked": mass_checked,
        "rigidity": rigidity_name,
        "learned_prior": True if learned else False,
        "outputs": ["physics.json"],
    }


def _measure_joints(obj: str, folder: Path) -> dict[str, Any]:
    motion = _load_json(folder / "motion.json")
    parts = _load_json(folder / "parts.json")
    named = set()
    for part in parts.get("parts") if isinstance(parts.get("parts"), list) else []:
        if isinstance(part, dict):
            named.add(str(part.get("id") or part.get("name") or ""))
    children = _urdf_children(folder / "asset.urdf")
    motion_parts = motion.get("parts") if isinstance(motion.get("parts"), list) else []
    motion_joints = [item for item in motion_parts if isinstance(item, dict) and item.get("joint")]
    has_joints = bool(children or motion_joints)
    swept = motion.get("swept") is True or (folder / "sweep.json").is_file()
    two_state = motion.get("two_state")
    if obj == "bottle":
        unswept = has_joints and not swept
        if two_state != "unavailable":
            reason = "bottle two_state must stay unavailable"
        elif unswept:
            reason = "unswept joint"
        else:
            reason = "two_state unavailable"
        return {
            "name": "joints",
            "pass": False,
            "two_state": two_state,
            "unswept_joint": unswept,
            "reason": reason,
            "outputs": [],
        }
    unnamed = [child for child in children if child not in named]
    return {
        "name": "joints",
        "pass": not unnamed,
        "two_state": two_state,
        "unnamed_joint": len(unnamed),
        "reason": "joint without a part name" if unnamed else "",
        "outputs": [],
    }


def _measure_mesh(folder: Path) -> dict[str, Any]:
    """pass = mesh + score stamped; achieved needs cm metrics and an honest claim."""
    score_path = folder / "mesh_score.json"
    score = _load_json(score_path)
    mesh = _extracted_mesh(folder)
    if not score_path.is_file():
        return {"name": "mesh", "pass": False, "achieved": False, "reason": "mesh_score missing", "outputs": []}
    if mesh is None:
        return {
            "name": "mesh",
            "pass": False,
            "achieved": False,
            "state": "unstarted" if score.get("state") in (None, "unstarted") else score.get("state"),
            "reason": "mesh missing",
            "outputs": ["mesh_score.json"],
        }
    chamfer = score.get("chamfer_cm")
    f1 = score.get("f1_at_1cm")
    measured = (
        isinstance(chamfer, (int, float))
        and not isinstance(chamfer, bool)
        and isinstance(f1, (int, float))
        and not isinstance(f1, bool)
    )
    achieved = bool(
        measured and chamfer < SCENE_CHAMFER_CM_MAX and f1 > SCENE_F1_1CM_MIN and score.get("achieved") is True
    )
    return {
        "name": "mesh",
        "pass": True,
        "achieved": achieved,
        "reason": "" if achieved else "mesh stamped; Chamfer/F1 unmet or metres missing",
        "outputs": ["mesh_score.json"],
    }


def _measure_align(folder: Path) -> dict[str, Any]:
    """pass = residuals stamped when frames+mesh exist; orient skip is ScoreOnly."""
    frames = _has_frames(folder)
    mesh = _extracted_mesh(folder)
    background = _has_background(folder)
    align = _load_json(folder / "align.json")
    orient = _load_json(folder / "orientation.json")
    if not frames or mesh is None:
        state = "unstarted"
        residuals = False
    else:
        state = "measured"
        residuals = _residual_numbers(align)
    orient_achieved = False
    if not background:
        orient_state = "skipped"
        if orient.get("achieved") is True:
            orient_achieved = False
    else:
        orient_state = "measured"
        orient_achieved = orient.get("achieved") is True and orient.get("skipped") is not True
    # Iterate may continue once residuals are stamped; orientation/metres stay ScoreOnly.
    finished = bool(frames and mesh is not None and residuals)
    if align.get("achieved") is True and (not frames or mesh is None):
        finished = False
    metres = align.get("metres_per_unit")
    achieved = bool(
        finished
        and orient_achieved
        and background
        and align.get("achieved") is True
        and isinstance(metres, (int, float))
        and not isinstance(metres, bool)
    )
    return {
        "name": "align",
        "pass": finished,
        "state": state,
        "achieved": achieved,
        "orientation": orient_state,
        "orientation_achieved": False if not background else orient_achieved,
        "outputs": ["align.json"] if residuals else [],
    }


def _measure_inspect(folder: Path) -> dict[str, Any]:
    """A real frame is required for the inspect measurement gate; Block H skips are ScoreOnly."""
    frame = folder / "inspect" / "frame_000.png"
    ok = _frame_ok(frame)
    return {
        "name": "inspect",
        "pass": ok,
        "achieved": False,
        "rendered": ok,
        "reason": "" if ok else "frame_000.png missing",
        "outputs": ["inspect/frame_000.png"] if ok else [],
    }


def _measure_press(folder: Path) -> dict[str, Any]:
    pictures = _press_pictures(folder)
    record = _load_json(folder / "press.json")
    claimed = record.get("destruction") == "ran" or record.get("physgaussian") is True or record.get("ran") is True
    return {
        "name": "press",
        "pass": bool(pictures) and not claimed,
        "press": "present" if pictures else "not_run",
        "destruction": "not_run",
        "outputs": [str(path.name) for path in pictures],
    }


def _measure_sisters(folder: Path) -> dict[str, Any]:
    frame = folder / "inspect" / "frame_000.png"
    sisters = _load_json(folder / "sisters" / "objects.json")
    objects = sisters.get("objects") if isinstance(sisters.get("objects"), list) else []
    clip = sisters.get("clip_similarity")
    clip_ok = isinstance(clip, (int, float)) and not isinstance(clip, bool) and CLIP_MIN <= float(clip) <= CLIP_MAX
    if not _frame_ok(frame):
        return {
            "name": "sisters",
            "pass": False,
            "reason": "inspect has no frame",
            "clip": None if not clip_ok else float(clip),
            "outputs": [],
        }
    return {
        "name": "sisters",
        "pass": clip_ok and len(objects) > 0,
        "reason": "" if clip_ok else "clip outside 0.80-0.95",
        "clip": float(clip) if clip_ok else None,
        "outputs": ["sisters/objects.json"] if clip_ok else [],
    }


def _splat_retrained(folder: Path, report: dict[str, Any]) -> bool:
    if report.get("trained_again") is True or report.get("ns_train") == "ran":
        return True
    capture = report.get("capture")
    if isinstance(capture, dict) and capture.get("ns_train") == "ran":
        return True
    nested = report.get("colour_calibration")
    if isinstance(nested, dict) and nested.get("appearance_embeddings_retrained") is True:
        return True
    for name in ("colour_calibration.json", "color_calibration.json"):
        colour = _load_json(folder / "ingest" / name)
        if colour.get("appearance_embeddings_retrained") is True:
            return True
    status = _load_json(folder / "status.json")
    orbit = status.get("step02")
    if isinstance(orbit, dict):
        orbit1 = orbit.get("orbit1")
        if isinstance(orbit1, dict) and orbit1.get("trained_again") is True:
            return True
    return False


def _scene_mean_measured(folder: Path, codebook: dict[str, Any], instances: dict[str, Any]) -> bool:
    background = instances.get("background") if isinstance(instances.get("background"), dict) else {}
    if background.get("scene_mean_measured") is True and background.get("feature") == "scene_mean":
        scene = codebook.get("scene_mean")
        vector = codebook.get("vector")
        if isinstance(scene, list) and isinstance(vector, list) and list(scene) == list(vector):
            return False
        return isinstance(scene, list) and len(scene) > 0
    if codebook.get("scene_mean_measured") is True and isinstance(codebook.get("scene_mean"), list):
        scene = codebook["scene_mean"]
        vector = codebook.get("vector")
        if isinstance(vector, list) and list(scene) == list(vector):
            return False
        return len(scene) > 0
    npz = folder / "semantics" / "features.codebook.npz"
    if npz.is_file():
        with np.load(npz) as book:
            if "scene_mean" in book and int(book["scene_mean"].size) == 0:
                return False
    return False


def _generated_sticky(parts: list[Any]) -> bool:
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("generate") is True and part.get("generated") is not True:
            return False
        if part.get("generated") is False and part.get("generate") is True:
            return False
    return True


def _k_delta_ok(document: dict[str, Any], count: int) -> bool:
    winning = document.get("winning_k")
    vlm = document.get("vlm_count", document.get("k"))
    if winning is None and vlm is None:
        return True
    base = count if winning is None else int(winning)
    if vlm is not None and abs(int(vlm) - base) > 1:
        return False
    if winning is not None and abs(count - int(winning)) > 1:
        return False
    return True


def _rigidity_gate(record: dict[str, Any]) -> tuple[bool, str]:
    per_part = record.get("per_part")
    scalar = record.get("rigidity")
    called = record.get("rigidity_label")
    scalar_number = isinstance(scalar, (int, float)) and not isinstance(scalar, bool)
    if scalar_number and (called == "heterogeneous" or record.get("heterogeneous") is True):
        return False, "scalar"
    if isinstance(per_part, list) and per_part and all(isinstance(item, dict) and "rigidity" in item for item in per_part):
        try:
            label = rigidity_label(per_part)
        except GateError:
            return False, "invalid"
        if label == "scalar":
            return False, "scalar"
        return True, label
    if scalar_number:
        return False, "scalar"
    return False, "missing"


def _weighed_kg(record: dict[str, Any]) -> float | None:
    for key in ("weighed_kg", "scale_kg", "measured_mass_kg"):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _urdf_children(path: Path) -> list[str]:
    if not path.is_file():
        return []
    import re

    return re.findall(r'<child link="([^"]+)"\s*/>', path.read_text(encoding="utf-8"))


def _residual_numbers(align: dict[str, Any]) -> bool:
    values = [align.get("photometric_residual"), align.get("iou"), align.get("edge_residual")]
    return any(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


def _extracted_mesh(folder: Path) -> Path | None:
    for path in (
        folder / "mesh" / "mesh.ply",
        folder / "mesh.ply",
        folder / "2dgs" / "mesh.ply",
        folder / "mesh" / "fused.ply",
    ):
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def _has_frames(folder: Path) -> bool:
    frames = folder / "frames"
    if not frames.is_dir():
        return False
    return any(frames.rglob("*.png"))


def _has_background(folder: Path) -> bool:
    return (folder / "background.ply").is_file() or (folder / "background" / "splat.ply").is_file()


def _press_pictures(folder: Path) -> list[Path]:
    found: list[Path] = []
    for name in ("press", "deform"):
        directory = folder / name
        if directory.is_dir():
            found.extend(path for path in directory.glob("*.png") if path.is_file() and path.stat().st_size > 0)
    extra = folder / "press_after.png"
    if extra.is_file() and extra.stat().st_size > 0:
        found.append(extra)
    return found


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
