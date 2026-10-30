"""Store the user's pipeline prompt and check the one target mask.

SCENEAGENT-RUN-PIPELINE reads the prompt on the laptop. The sentence is
stored unchanged. One vision parse fills the task spec. A second look at
the target overlay may change text_prompt or box_threshold once. A room
job whose lift already covers the room records that same one box_threshold
retry from the numbers on disk, with no vision call. The splat is not
retrained, and the nine sisters are not written.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from tools.block_b_job import FRAME_COVER, feature_field_may_start
from tools.common.paths import object_dir
from tools.common.vlm import MissingReviewImage, VlmError, parse_pipeline_prompt, review
from tools.harvard_agents import allowed_params

MASK_PARAMS = ("text_prompt", "box_threshold")
DEFAULT_BOX_THRESHOLD = 0.25
BOX_THRESHOLD_STEP = 0.05
ROOM_SPAN = 0.5
Segment = Callable[[str, dict[str, Any]], dict[str, Any]]
ReviewFn = Callable[[Path, int], dict[str, Any]]


def mask_correction_params() -> tuple[str, ...]:
    """The mask retry may name only these SCENEAGENT-RUN-STEP parameters."""
    allowed = set(allowed_params("SCENEAGENT-RUN-STEP"))
    chosen = tuple(name for name in MASK_PARAMS if name in allowed)
    if set(chosen) != set(MASK_PARAMS):
        raise RuntimeError("SCENEAGENT-RUN-STEP allow-list lost text_prompt or box_threshold")
    return chosen


def request_path(obj: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "request.json"


def sample_four_frames(folder: Path, source: Path | None = None) -> list[Path]:
    """Up to four frames beside the capture. A missing video yields none."""
    candidates: list[Path] = []
    if source is not None and source.is_dir():
        candidates = _images_in(source)
    elif source is not None and source.is_file() and source.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        candidates = [source]
    if not candidates:
        frames = folder / "frames"
        if frames.is_dir():
            candidates = _images_in(frames)
    if len(candidates) >= 4:
        return [candidates[int(index * (len(candidates) - 1) / 3)] for index in range(4)]
    if source is not None and source.is_file() and source.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}:
        extracted = _extract_four_frames(source, folder / "request_frames")
        if extracted:
            return extracted
    return candidates[:4]


def accept_pipeline_prompt(
    obj: str,
    prompt: str,
    root: Path | None = None,
    source: Path | None = None,
    frames: list[Path] | None = None,
) -> int:
    """Write the user's text, then one laptop parse. A missing key stops the run."""
    folder = object_dir(obj, root)
    path = request_path(obj, root)
    _write_json(path, {"prompt": prompt})
    pictures = list(frames) if frames is not None else sample_four_frames(folder, source)
    try:
        parsed = parse_pipeline_prompt(prompt, pictures)
    except VlmError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    _write_json(
        path,
        {
            "prompt": prompt,
            "job": parsed["job"],
            "target": parsed["target"],
            "seen": parsed["seen"],
            "deliverable": parsed["deliverable"],
        },
    )
    return 0


def load_request(folder: Path) -> dict[str, Any]:
    path = Path(folder) / "request.json"
    if not path.is_file():
        raise FileNotFoundError("task spec must be written before segmentation")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or "prompt" not in loaded:
        raise FileNotFoundError("task spec must be written before segmentation")
    return loaded


def next_mask_action(verdict: dict[str, Any], attempt: int) -> dict[str, Any]:
    """One retry. A second miss stops. Merge and split are not actions."""
    base = {"retrain": False, "sisters": False}
    if verdict.get("verdict") == "pass":
        return {**base, "action": "keep"}
    if attempt >= 2:
        return {**base, "action": "stop", "reason": "second miss"}
    correction = verdict.get("correction") if isinstance(verdict.get("correction"), dict) else {}
    parameter = correction.get("parameter")
    if parameter not in mask_correction_params():
        return {**base, "action": "stop", "reason": "no allowed correction"}
    return {
        **base,
        "action": "retry",
        "parameter": parameter,
        "value": correction.get("value"),
    }


def review_target_overlay(overlay: Path, target: str, record_path: Path | None = None) -> dict[str, Any]:
    """One look at the target overlay. The allow-list is text_prompt and box_threshold."""
    question = (
        f"This overlay is the mask for {target}. "
        "Does the mask cover that object and leave the rest of the scene out? "
        "If it fails, correct only text_prompt or box_threshold."
    )
    return review(
        "SCENEAGENT-RUN-STEP",
        [overlay],
        question,
        mask_correction_params(),
        record_path=record_path,
    )


def run_target_mask_check(
    folder: Path,
    segment: Segment,
    review_fn: ReviewFn | None = None,
) -> dict[str, Any]:
    """Read the spec, then segment the target. One retry, then stop with the mask."""
    folder = Path(folder)
    spec = load_request(folder)
    target = spec.get("target")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("task spec has no target")
    target = target.strip()
    look = review_fn or (
        lambda path, attempt: review_target_overlay(
            path,
            target,
            folder / "reviews" / f"target_mask_{attempt}.json",
        )
    )
    measurement = segment(target, {})
    attempts = [measurement]
    action = _look(look, measurement, 1)
    if action["action"] == "retry":
        measurement = segment(target, {action["parameter"]: action["value"]})
        attempts.append(measurement)
        action = _look(look, measurement, 2)
        if action["action"] == "retry":
            action = {"action": "stop", "retrain": False, "sisters": False, "reason": "second miss"}
    fractions = [float(item) for item in (measurement.get("view_fractions") or [])]
    dropped = [index for index, fraction in enumerate(fractions) if fraction >= FRAME_COVER]
    span = measurement.get("span_ratio")
    may_start = feature_field_may_start(
        spec.get("job") if isinstance(spec.get("job"), str) else None,
        view_fractions=fractions,
        covers_room=measurement.get("covers_room") is True,
        span_ratio=float(span) if isinstance(span, (int, float)) and not isinstance(span, bool) else None,
    )
    record = {
        "prompt": spec.get("prompt"),
        "target": target,
        "attempts": len(attempts),
        "action": action["action"],
        "reason": action.get("reason"),
        "retrain": False,
        "sisters": False,
        "asked": action.get("asked", True),
        "feature_field": "not_started" if not may_start else "not_this_step",
        "dropped_views": dropped,
        "mask": {key: value for key, value in measurement.items() if key != "overlay"},
    }
    _write_json(folder / "reviews" / "target_mask_check.json", record)
    return record


def raised_box_threshold() -> float:
    """One step above the 0.25 GroundingDINO default. The splat stays put."""
    return float(f"{DEFAULT_BOX_THRESHOLD + BOX_THRESHOLD_STEP:.2f}")


def lift_covers_room(lift: dict[str, Any] | None, view_fractions: list[float] | None = None) -> bool:
    """True when the lift covers the room, the span is at least half, or a view covers the frame."""
    if isinstance(lift, dict) and lift.get("covers_room") is True:
        return True
    ratio = _span_ratio(lift)
    if ratio is not None and ratio >= ROOM_SPAN:
        return True
    return _fractions_cover(view_fractions) or _fractions_cover(_stored_fractions(lift))


def record_room_mask_retry(folder: Path) -> dict[str, Any]:
    """Write one box_threshold retry when a room job's lift already covers the room.

    A missing job keeps the current refusal. An orbit job skips the half-scene
    test. The orbit is chosen by covers_room, span, or a frame-covering view.
    A second call does not write another retry. No vision call is required.
    """
    folder = Path(folder)
    if not (folder / "request.json").is_file():
        return {"action": "skip", "reason": "missing job", "feature_field": "not_started", "wrote": False}
    spec = load_request(folder)
    job = spec.get("job") if spec.get("job") in {"orbit", "room"} else None
    if job == "orbit":
        return {"action": "skip", "reason": "orbit job", "feature_field": "not_this_step", "wrote": False}
    if job != "room":
        return {"action": "skip", "reason": "missing job", "feature_field": "not_started", "wrote": False}
    failing = failing_room_orbits(folder)
    if not failing:
        return {"action": "skip", "reason": "lift does not cover the room", "wrote": False}
    target = spec.get("target")
    if not isinstance(target, str) or not target.strip():
        return {"action": "stop", "reason": "task spec has no target", "feature_field": "not_started", "wrote": False}
    existing = _load_mask_retry(folder)
    if existing is not None:
        if existing.get("detection") == "rerun":
            return _refuse_second_room_mask(folder, spec, failing)
        return {
            **existing,
            "action": "retry",
            "feature_field": "not_started",
            "wrote": False,
            "retrain": False,
            "sisters": False,
        }
    payload = {
        "target": target.strip(),
        "params": {"box_threshold": raised_box_threshold()},
        "retrain": False,
        "sisters": False,
        "splat_retrained": False,
        "orbits": [row["orbit"] for row in failing],
        "measured": failing,
        "reason": "lift covers the room",
        "detection": "pending",
        "attempts": 1,
        "feature_field": "not_started",
    }
    _write_json(folder / "mask_retry.json", payload)
    _write_mask_check(folder, spec, failing, action="retry", attempts=1, reason="lift covers the room")
    payload["action"] = "retry"
    payload["wrote"] = True
    return payload


def failing_room_orbits(folder: Path) -> list[dict[str, Any]]:
    """Orbits whose lift covers the room. The directory name is not a special case."""
    rows = []
    for orbit, lift in _orbit_lifts(folder):
        fractions = orbit_view_fractions(folder, orbit)
        if not lift_covers_room(lift, fractions):
            continue
        rows.append(
            {
                "orbit": orbit,
                "covers_room": lift.get("covers_room") is True,
                "span_inside": lift.get("span_inside"),
                "span_scene": lift.get("span_scene"),
                "span_ratio": _span_ratio(lift),
                "view_fractions": fractions or _stored_fractions(lift),
            }
        )
    return rows


def orbit_view_fractions(folder: Path, orbit: str) -> list[float]:
    """Foreground fraction of each view mask. About 90 percent of the frame counts."""
    return _png_fractions(Path(folder) / "masks" / orbit)


def gate_detected_target(obj: str, root: Path | None = None) -> int:
    """One look when an overlay already exists. No overlay is not a pass.

    A room-sized lift records one box_threshold retry before that look, so a
    span already on disk does not wait for a vision call. A text_prompt or
    box_threshold correction is recorded for the next detection. The same
    overlay is not reviewed again, so the one retry is not spent twice.
    """
    folder = object_dir(obj, root)
    if not (folder / "request.json").is_file():
        return 0
    try:
        recorded = record_room_mask_retry(folder)
    except (OSError, json.JSONDecodeError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    if recorded.get("action") == "retry":
        return 0
    if recorded.get("action") == "stop":
        _write_stop_line(recorded)
        return 2
    overlay = find_target_overlay(folder)
    if overlay is None:
        return 0
    try:
        spec = load_request(folder)
    except (OSError, json.JSONDecodeError, FileNotFoundError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    target = spec.get("target")
    if not isinstance(target, str) or not target.strip():
        sys.stderr.write("task spec has no target\n")
        return 2
    target = target.strip()
    try:
        verdict = review_target_overlay(overlay, target, folder / "reviews" / "target_mask_1.json")
    except (VlmError, MissingReviewImage, OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    action = next_mask_action(verdict, 1)
    fractions = _mask_fractions(folder)
    if action["action"] == "retry":
        _write_json(
            folder / "mask_retry.json",
            {
                "target": target,
                "params": {action["parameter"]: action["value"]},
                "retrain": False,
                "sisters": False,
            },
        )
        return 0
    if action["action"] == "stop":
        _write_json(
            folder / "reviews" / "target_mask_check.json",
            {
                "prompt": spec.get("prompt"),
                "target": target,
                "attempts": 1,
                "action": "stop",
                "reason": action.get("reason"),
                "retrain": False,
                "sisters": False,
                "feature_field": "not_started",
                "view_fractions": fractions,
            },
        )
        sys.stderr.write("target mask check stopped with the measured mask\n")
        return 2
    return 0


def find_target_overlay(folder: Path) -> Path | None:
    reviews = folder / "reviews"
    if reviews.is_dir():
        masks = sorted(path for path in reviews.glob("*_mask.png") if path.is_file())
        if masks:
            return masks[0]
    masks_root = folder / "masks"
    if masks_root.is_dir():
        found = sorted(path for path in masks_root.glob("*/*_mask.png") if path.is_file())
        if found:
            return found[0]
    return None


def _look(review_fn: ReviewFn, measurement: dict[str, Any], attempt: int) -> dict[str, Any]:
    overlay = measurement.get("overlay")
    if not overlay:
        return {"action": "stop", "retrain": False, "sisters": False, "reason": "no target overlay", "asked": False}
    try:
        verdict = review_fn(Path(overlay), attempt)
    except MissingReviewImage as exc:
        return {"action": "stop", "retrain": False, "sisters": False, "reason": str(exc), "asked": False}
    if not isinstance(verdict, dict):
        return {"action": "stop", "retrain": False, "sisters": False, "reason": "second miss" if attempt >= 2 else "no allowed correction"}
    action = next_mask_action(verdict, attempt)
    action["asked"] = True
    return action


def _mask_fractions(folder: Path) -> list[float]:
    masks = folder / "masks"
    if not masks.is_dir():
        return []
    fractions: list[float] = []
    for child in sorted(path for path in masks.iterdir() if path.is_dir()):
        fractions.extend(_png_fractions(child))
    return fractions


def _png_fractions(directory: Path) -> list[float]:
    if not directory.is_dir():
        return []
    found = sorted(path for path in directory.glob("*_mask.png") if path.is_file())
    if not found:
        return []
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        return []
    fractions: list[float] = []
    for path in found:
        array = np.asarray(Image.open(path).convert("L"))
        total = int(array.size)
        if total == 0:
            continue
        fractions.append(float((array > 127).sum()) / float(total))
    return fractions


def _orbit_lifts(folder: Path) -> list[tuple[str, dict[str, Any]]]:
    masks = Path(folder) / "masks"
    found: list[tuple[str, dict[str, Any]]] = []
    if not masks.is_dir():
        return found
    for child in sorted(path for path in masks.iterdir() if path.is_dir()):
        if child.name != Path(child.name).name or child.name in {"", ".", ".."}:
            continue
        loaded = _read_object(child / "object.json")
        if loaded is not None:
            found.append((child.name, loaded))
    if not any(name == "orbit1" for name, _lift in found):
        flat = _read_object(masks / "object.json")
        if flat is not None:
            found.insert(0, ("orbit1", flat))
    return found


def _read_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _load_mask_retry(folder: Path) -> dict[str, Any] | None:
    loaded = _read_object(Path(folder) / "mask_retry.json")
    return loaded


def _span_ratio(lift: dict[str, Any] | None) -> float | None:
    if not isinstance(lift, dict):
        return None
    span_in = lift.get("span_inside")
    span_sc = lift.get("span_scene")
    if isinstance(span_in, bool) or isinstance(span_sc, bool):
        return None
    if not isinstance(span_in, (int, float)) or not isinstance(span_sc, (int, float)):
        return None
    if not math.isfinite(span_in) or not math.isfinite(span_sc) or float(span_sc) <= 0:
        return None
    return float(span_in) / float(span_sc)


def _stored_fractions(lift: dict[str, Any] | None) -> list[float]:
    if not isinstance(lift, dict) or not isinstance(lift.get("view_fractions"), list):
        return []
    fractions = []
    for item in lift["view_fractions"]:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        if math.isfinite(item):
            fractions.append(float(item))
    return fractions


def _fractions_cover(fractions: list[float] | None) -> bool:
    if not fractions:
        return False
    return any(math.isfinite(item) and FRAME_COVER <= float(item) <= 1.0 for item in fractions)


def _refuse_second_room_mask(folder: Path, spec: dict[str, Any], failing: list[dict[str, Any]]) -> dict[str, Any]:
    """The one retry already ran. A room-sized lift stays refused with the measured numbers."""
    record = _write_mask_check(folder, spec, failing, action="stop", attempts=2, reason="second miss")
    record["wrote"] = False
    record["retrain"] = False
    record["sisters"] = False
    return record


def _write_mask_check(
    folder: Path,
    spec: dict[str, Any],
    failing: list[dict[str, Any]],
    *,
    action: str,
    attempts: int,
    reason: str,
) -> dict[str, Any]:
    fractions: list[float] = []
    for row in failing:
        fractions.extend(float(item) for item in row.get("view_fractions") or [])
    may_start = feature_field_may_start(
        "room",
        view_fractions=fractions or None,
        covers_room=any(row.get("covers_room") is True for row in failing),
        span_ratio=_max_span(failing),
    )
    if any(_fractions_cover(row.get("view_fractions")) for row in failing):
        may_start = False
    record = {
        "prompt": spec.get("prompt"),
        "target": spec.get("target"),
        "attempts": attempts,
        "action": action,
        "reason": reason,
        "retrain": False,
        "sisters": False,
        "asked": False,
        "feature_field": "not_started" if not may_start else "not_this_step",
        "measured": failing,
    }
    _write_json(folder / "reviews" / "target_mask_check.json", record)
    return record


def _max_span(failing: list[dict[str, Any]]) -> float | None:
    ratios = [row.get("span_ratio") for row in failing if isinstance(row.get("span_ratio"), (int, float))]
    if not ratios:
        return None
    return float(max(ratios))


def _write_stop_line(recorded: dict[str, Any]) -> None:
    if recorded.get("reason") != "second miss":
        sys.stderr.write(f"{recorded.get('reason') or 'target mask check stopped'}\n")
        return
    bits = []
    for row in recorded.get("measured") or []:
        if not isinstance(row, dict):
            continue
        bits.append(
            f"{row.get('orbit')} span_inside {row.get('span_inside')} span_scene {row.get('span_scene')}"
        )
    detail = "; ".join(bits)
    line = "target mask stayed room-sized"
    if detail:
        line += f" ({detail})"
    sys.stderr.write(line + "; the retry was not repeated\n")


def _images_in(folder: Path) -> list[Path]:
    suffixes = {".png", ".jpg", ".jpeg"}
    return sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def _extract_four_frames(video: Path, dest: Path) -> list[Path]:
    try:
        import cv2
    except ImportError:
        return []
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return []
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count <= 0:
        capture.release()
        return []
    dest.mkdir(parents=True, exist_ok=True)
    indexes = [0] if count == 1 else [int(index * (count - 1) / 3) for index in range(4)]
    saved: list[Path] = []
    for nth, index in enumerate(dict.fromkeys(indexes)):
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        path = dest / f"frame_{nth:02d}.png"
        if cv2.imwrite(str(path), frame):
            saved.append(path)
    capture.release()
    return saved


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
