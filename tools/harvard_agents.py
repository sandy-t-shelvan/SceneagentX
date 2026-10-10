"""The nine Harvard SceneAgent skills. This repo does not run them."""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.paths import object_dir

SKILLS: tuple[dict[str, str], ...] = (
    {
        "id": "SCENEAGENT-RUN-PIPELINE",
        "does": "Chooses the path from the input status",
        "ours": "choose_path returns ply, video, or stop. A ply never starts COLMAP.",
    },
    {
        "id": "SCENEAGENT-INGEST-DATA",
        "does": "COLMAP plus splatfacto-big for RGB, or accept a mesh or splat ply",
        "ours": "Video frames use sequential COLMAP. A passing report is kept and not rerun.",
    },
    {
        "id": "SCENEAGENT-RUN-STEP",
        "does": "Masks, semantics, physics, crop, and infill, then a visual fix",
        "ours": "One skill runs, the file is checked, and the first fault restarts only that skill.",
    },
    {
        "id": "SCENEAGENT-ALIGN-SCENE",
        "does": "Photometric, silhouette, and edge residuals",
        "ours": "Bounding-box scale. The three residuals stay skipped.",
    },
    {
        "id": "SCENEAGENT-ORIENT-BACKGROUND",
        "does": "Isaac Lab sweeps when a background splat exists",
        "ours": "Skipped. A skip is not Achieved.",
    },
    {
        "id": "SCENEAGENT-PACKAGE-DATA-PROCESSED",
        "does": "USDZ for Isaac, MuJoCo, and Unreal",
        "ours": "A USDA zip. It is not loaded in a simulator.",
    },
    {
        "id": "SCENEAGENT-INSPECT-ISAAC",
        "does": "Renders the USDZ in Isaac and fixes what is wrong",
        "ours": "Not run.",
    },
    {
        "id": "SCENEAGENT-OBJECT-SISTERS",
        "does": "TRELLIS geometric and visual variations",
        "ours": "TRELLIS can write the meshes. This skill is not an agent.",
    },
    {
        "id": "SCENEAGENT-SCENE-SISTERS",
        "does": "New layouts of those sisters",
        "ours": "Layouts of whatever object files exist. Not an agent.",
    },
)


def write_agents(
    obj: str,
    root: Path | None = None,
    ran: bool | None = None,
    visual_review: bool | None = None,
    faults: list[dict] | None = None,
) -> Path:
    """Write the skill record. Omitted flags keep the values already on disk."""
    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "agents.json"
    existing: dict = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            existing = loaded
    if ran is None:
        ran = bool(existing.get("ran", False))
    if visual_review is None:
        visual_review = bool(existing.get("visual_review", False))
    if faults is None:
        stored = existing.get("faults")
        faults = stored if isinstance(stored, list) else []
    if visual_review:
        ran = True
    if visual_review:
        note = "gpt-6-astra returned a decision. The file check is not SCENEAGENT-RUN-STEP."
    elif ran:
        note = "The skill router ran. No vision decision was returned. The file check is not SCENEAGENT-RUN-STEP."
    else:
        note = "These skills are not run. The file check is not SCENEAGENT-RUN-STEP."
    payload = {
        "ran": ran,
        "visual_review": visual_review,
        "faults": faults,
        "note": note,
        "skills": list(SKILLS),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
