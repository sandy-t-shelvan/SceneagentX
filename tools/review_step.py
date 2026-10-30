"""Fail a block when a required artifact file is missing.

This is a file check. It does not call a model, and it is not
SCENEAGENT-RUN-STEP.
"""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.paths import object_dir
from tools.harvard_agents import write_agents

REQUIRED: dict[str, tuple[str, ...]] = {
    "B4": ("align.json", "orientation.json"),
    "B5": ("masks/object.json",),
    "B6": ("masks/object.json", "masks/gaussian_in_mask.npy"),
    "B7": ("codebook.json",),
    "B8": ("status.json",),
    "B9": ("parts.json",),
    "B10": ("physics.json",),
    "B11": ("asset.urdf",),
    "B12": ("scene.usdz",),
    "B13": ("sisters/objects.json",),
    "B14": ("sisters/scenes.json",),
}


class ReviewError(RuntimeError):
    """A required file is missing. This is not a visual agent failure."""


def review_block(obj: str, block_id: str, root: Path | None = None) -> Path:
    folder = object_dir(obj, root)
    names = REQUIRED.get(block_id)
    if names is None:
        raise ReviewError(f"{block_id} has no file check")
    missing = [name for name in names if not (folder / name).is_file()]
    if block_id == "B6":
        record = _object_record(folder)
        infill = record.get("infill")
        if infill not in {"skipped", "ran"}:
            missing.append("masks/object.json infill")
        if record.get("projected") is not True:
            missing.append("masks/object.json projected")
    if block_id == "B10":
        physics = _read(folder / "physics.json")
        per = physics.get("per_gaussian") or {}
        if per.get("learned_prior") is not False:
            missing.append("physics.json learned_prior false")
    if missing:
        joined = ", ".join(missing)
        raise ReviewError(
            f"{block_id} is missing {joined}. "
            "This file check is not SCENEAGENT-RUN-STEP."
        )
    write_agents(obj, root)
    path = folder / "review.json"
    path.write_text(
        json.dumps(
            {
                "block": block_id,
                "kind": "file_check",
                "sceneagent_run_step": False,
                "passed": True,
                "note": "Fails when a file is missing. Does not look at a render.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _object_record(folder: Path) -> dict:
    return _read(folder / "masks" / "object.json")


def _read(path: Path) -> dict:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    return loaded
