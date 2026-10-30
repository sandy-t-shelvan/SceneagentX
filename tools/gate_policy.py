"""Three-tier agentic gates: HardStop, RepairLoop, and ScoreOnly.

HardStop ends the walk when the next tool would invent data or cannot run.
RepairLoop retries under the skill limits, then writes question.json.
ScoreOnly records the unmet check and continues with achieved false.

This module never sets achieved, m1_met, m2_met, or milestone to true.
A skip is not a pass.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

HARD_STOP = "hard_stop"
REPAIR = "repair"
SCORE_ONLY = "score_only"

_CLASSES = frozenset({HARD_STOP, REPAIR, SCORE_ONLY})
_CHECKS = frozenset({"skipped", "unmet", "present_unachieved", "not_run"})

# Explicit reason → class. Anything else falls through to heuristics.
_REASON_CLASS: dict[str, str] = {
    "no_input": HARD_STOP,
    "no_vast": HARD_STOP,
    "unknown_skill": HARD_STOP,
    "second_room_refuse": HARD_STOP,
    "feature_field_refused": HARD_STOP,
    "covers_room": REPAIR,
    "extra_part": REPAIR,
    "mask_quality": REPAIR,
    "align_residual": REPAIR,
    "parts_count": REPAIR,
    "articulation_critic": REPAIR,
    "package_visual_fail": REPAIR,
    "skill_skip": SCORE_ONLY,
    "package_present": SCORE_ONLY,
    "package_missing": SCORE_ONLY,
    "no_metres": SCORE_ONLY,
    "metres_per_unit_missing": SCORE_ONLY,
    "hand_labels": SCORE_ONLY,
    "miou": SCORE_ONLY,
    "mass_unchecked": SCORE_ONLY,
    "physics_prior_missing": SCORE_ONLY,
    "infill_skipped": SCORE_ONLY,
    "infill_not_run": SCORE_ONLY,
    "orient_skipped": SCORE_ONLY,
    "inspect_not_run": SCORE_ONLY,
    "sisters_skipped": SCORE_ONLY,
    "chamfer_without_gt": SCORE_ONLY,
    "story_unsupported": SCORE_ONLY,
}

_SCORE_ONLY_SKILLS = frozenset(
    {
        "SCENEAGENT-ORIENT-BACKGROUND",
        "SCENEAGENT-INSPECT-ISAAC",
        "SCENEAGENT-OBJECT-SISTERS",
        "SCENEAGENT-SCENE-SISTERS",
        "SCENEAGENT-PACKAGE-DATA-PROCESSED",
    }
)


def classify(
    reason: str,
    *,
    skill_id: str | None = None,
    detail: str = "",
    attempts: int | None = None,
) -> str:
    """Return hard_stop, repair, or score_only. Never invents a milestone."""
    key = str(reason or "").strip().lower().replace(" ", "_")
    if key in _REASON_CLASS:
        gate = _REASON_CLASS[key]
        if gate == REPAIR and attempts is not None and attempts >= 2 and key in {
            "covers_room",
            "mask_quality",
        }:
            # Room story: second covers_room after one retry is HardStop.
            if key == "covers_room" or "second" in str(detail).lower():
                return HARD_STOP
        return gate
    blob = f"{reason} {detail} {skill_id or ''}".lower()
    if "second miss" in blob or "second room" in blob:
        return HARD_STOP
    if skill_id in _SCORE_ONLY_SKILLS:
        return SCORE_ONLY
    if any(
        token in blob
        for token in (
            "metres_per_unit",
            "no metres",
            "hand label",
            "miou",
            "mass_checked",
            "physics_prior",
            "infill skipped",
            "infill: skipped",
            "not_run",
            "story_unsupported",
            "unmeasured",
            "present_unachieved",
            "package missing",
            "package file present",
        )
    ):
        return SCORE_ONLY
    if "skipped" in blob and skill_id in _SCORE_ONLY_SKILLS:
        return SCORE_ONLY
    if any(token in blob for token in ("no video", "no input", "no running vast", "unknown skill")):
        return HARD_STOP
    if any(token in blob for token in ("covers_room", "extra_part", "residual", "mask bleed")):
        return REPAIR
    # Unknown unmet exits stay HardStop so the walk does not invent a continue.
    return HARD_STOP


def skill_exit(gate_class: str) -> int:
    """Exit code for a classified gate. ScoreOnly is 0 so the walk continues."""
    kind = gate_class if gate_class in _CLASSES else HARD_STOP
    if kind == SCORE_ONLY:
        return 0
    return 2


def walk_continues(gate_class: str) -> bool:
    """True when the Block H walk should keep going without SCENEAGENT_BYPASS_GATES."""
    return gate_class == SCORE_ONLY


def record_ledger(
    obj: str | None,
    root: Path | None,
    *,
    gate: str,
    gate_class: str,
    detail: str = "",
    check: str = "unmet",
    reason: str = "",
) -> dict[str, Any]:
    """Append one ledger row. Milestones in the file stay false."""
    kind = gate_class if gate_class in _CLASSES else SCORE_ONLY
    status = check if check in _CHECKS else "unmet"
    entry = {
        "gate": str(gate),
        "class": kind,
        "check": status,
        "reason": str(reason or ""),
        "detail": " ".join(str(detail).split())[:500],
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "milestone": False,
        "passed": False,
    }
    path = _path(obj, root) if obj else None
    payload = _load(path) if path is not None else {}
    entries = payload.get("entries")
    if not isinstance(entries, list):
        entries = []
    entries.append(entry)
    payload = {
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "milestone": False,
        "passed": False,
        "entries": entries,
    }
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"gate ledger record failed: {exc}\n")
    return entry


def classify_skill_result(skill_id: str, code: int, detail: str = "") -> str:
    """Map a non-zero skill exit to a gate class for the Block H walk.

    Exit 0 always continues; callers should not stop on success.
    Orient/inspect/sisters/package are ScoreOnly. Align without metres is
    ScoreOnly. Other non-zero exits stay HardStop unless the detail matches
    an explicit ScoreOnly or Repair reason.
    """
    if code == 0:
        return SCORE_ONLY
    if skill_id in _SCORE_ONLY_SKILLS:
        return SCORE_ONLY
    if skill_id == "SCENEAGENT-ALIGN-SCENE":
        return classify("no_metres", skill_id=skill_id, detail=detail)
    if "second miss" in detail.lower() or "room-sized" in detail.lower():
        return HARD_STOP
    if not str(detail or "").strip():
        return HARD_STOP
    return classify(detail, skill_id=skill_id, detail=detail)


def _path(obj: str | None, root: Path | None) -> Path | None:
    if not obj:
        return None
    from tools.common.paths import object_dir

    return object_dir(obj, root) / "gate_ledger.json"


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
