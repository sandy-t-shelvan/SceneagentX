"""Shared review report written by every SceneAgent tool.

The shape is fixed: status, metrics, renders, and suggested fixes.
This module does not train, does not call COLMAP, and does not call an API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_report(
    path: Path,
    status: str,
    metrics: dict[str, Any] | None = None,
    renders: list[str] | None = None,
    suggested_fixes: list[str] | None = None,
) -> dict[str, Any]:
    """Write ``{status, metrics, renders, suggested_fixes}``."""
    report = {
        "status": status,
        "metrics": dict(metrics or {}),
        "renders": [str(item) for item in (renders or [])],
        "suggested_fixes": [str(item) for item in (suggested_fixes or [])],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
