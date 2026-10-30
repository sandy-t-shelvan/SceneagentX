"""Temporary opt-in so a failed gate does not end the Block H walk.

SCENEAGENT_BYPASS_GATES is off unless it is set to 1, true, yes, or on.
A bypassed check is recorded as skipped or unmet. It is not a pass, and this
module never writes achieved, m1_met, m2_met, or a milestone as true.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

ENV_NAME = "SCENEAGENT_BYPASS_GATES"
_ON = {"1", "true", "yes", "on"}
_CHECKS = {"skipped", "unmet"}


def enabled() -> bool:
    """True only when the operator turned the switch on."""
    return os.environ.get(ENV_NAME, "").strip().lower() in _ON


def record(
    obj: str | None,
    root: Path | None,
    gate: str,
    detail: str,
    *,
    check: str = "unmet",
) -> dict[str, Any]:
    """Append one bypassed gate. Milestones in the file stay false."""
    kind = check if check in _CHECKS else "unmet"
    entry = {
        "gate": str(gate),
        "bypassed": True,
        "check": kind,
        "detail": " ".join(str(detail).split())[:500],
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "milestone": False,
        "passed": False,
    }
    path = _path(obj, root) if obj else None
    payload = _load(path) if path is not None else {}
    gates = payload.get("gates")
    if not isinstance(gates, list):
        gates = []
    gates.append(entry)
    payload = {
        "switch": ENV_NAME,
        "achieved": False,
        "m1_met": False,
        "m2_met": False,
        "milestone": False,
        "passed": False,
        "gates": gates,
    }
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            sys.stderr.write(f"gate bypass record failed: {exc}\n")
    sys.stderr.write(f"gate bypass {entry['gate']}: {entry['check']}; {entry['detail']}\n")
    return entry


def continue_past(
    obj: str | None,
    root: Path | None,
    gate: str,
    detail: str,
    *,
    check: str = "unmet",
) -> bool:
    """When the switch is on, record the unmet check and tell the caller to continue.

    When the switch is off, this writes nothing and returns False so the caller
    stops as before. A True return is not a pass.
    """
    if not enabled():
        return False
    record(obj, root, gate, detail, check=check)
    return True


def _path(obj: str | None, root: Path | None) -> Path | None:
    if not obj:
        return None
    from tools.common.paths import object_dir

    return object_dir(obj, root) / "gate_bypass.json"


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}
