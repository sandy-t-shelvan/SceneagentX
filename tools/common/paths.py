"""Repository paths.

Metrics and frames live under data/<object>/ so a second object (B17)
does not overwrite the marker. The plan's short path renders/step00/
is data/<object>/renders/step00/.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def resolve_root(root: Path | None) -> Path:
    return repo_root() if root is None else Path(root)


def schema_path(name: str = "status.json") -> Path:
    return repo_root() / "schemas" / name


def capture_config_path() -> Path:
    return repo_root() / "configs" / "capture.json"


def load_capture_config() -> dict:
    return json.loads(capture_config_path().read_text(encoding="utf-8"))


def object_dir(obj: str, root: Path | None = None) -> Path:
    return resolve_root(root) / "data" / obj


def status_path(obj: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "status.json"


def raw_dir(obj: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "raw"


def frames_dir(obj: str, orbit: int, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "frames" / f"orbit{orbit}"


def renders_dir(obj: str, step: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "renders" / step


def reports_dir(obj: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "reports"
