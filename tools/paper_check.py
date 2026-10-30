"""Compare one Polycam object with the Harvard SceneAgent pipeline.

Laptop artifacts are scored as they are. GPU programs that did not run stay
unmatched. This file never reads or writes an API key.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
from typing import Any

from tools.common.paths import object_dir
from tools.common.status import find_step, load_status, save_status, set_path

PAPER_URL = "https://computationalrobotics.seas.harvard.edu/SceneAgent/"
_POLICY_BLOCKS = (
    ("B15", "Soft-body drop needs PhysGaussian on the Vast VM. Not run."),
    ("B16", "Policy loop needs Isaac Lab, a robot, and checkpoints. Not run."),
    ("B17", "A second hinged object was not run."),
)
_STEP_FOR = {
    "semantics": "B5",
    "physics": "B10",
    "decomposition": "B9",
    "articulation": "B11",
    "alignment": "B4",
    "package": "B12",
    "object_sisters": "B13",
    "scene_sisters": "B14",
    "policy": "B16",
}


def build_report(obj: str, root: Path | None = None) -> dict[str, Any]:
    folder = object_dir(obj, root)
    doc = load_status(obj, root, create=False)
    rows = [
        _input_row(doc),
        _gaussian_row(doc),
        _semantics_row(folder),
        _physics_row(folder),
        _parts_row(folder),
        _urdf_row(folder),
        _align_row(doc),
        _orientation_row(doc),
        _package_row(folder, doc),
        _inspect_row(doc),
        _object_sisters_row(folder),
        _scene_sisters_row(folder),
        _policy_row(doc),
    ]
    return {
        "object": obj,
        "paper": "Hollis, Fan, Yang 2026",
        "paper_url": PAPER_URL,
        "blocks": rows,
    }


def write_paper_check(obj: str, root: Path | None = None) -> Path:
    report = build_report(obj, root)
    blob = json.dumps(report, indent=2) + "\n"
    if "sk-" in blob or "Bearer " in blob:
        raise RuntimeError("paper check refused to write a secret")
    folder = object_dir(obj, root)
    path = folder / "paper_check.json"
    path.write_text(blob, encoding="utf-8")
    doc = load_status(obj, root, create=False)
    for row in report["blocks"]:
        block_id = _STEP_FOR.get(row["id"])
        if block_id is None:
            continue
        step = find_step(doc, block_id)
        actual = dict(step.get("actual") or {})
        actual["paper_match"] = row["match"]
        step["actual"] = actual
    for block_id, reason in _POLICY_BLOCKS:
        step = find_step(doc, block_id)
        if step.get("state") == "pass":
            continue
        step["state"] = "fail"
        actual = dict(step.get("actual") or {})
        actual["match"] = False
        actual["reason"] = reason
        step["actual"] = actual
    set_path(
        doc,
        "paper_check",
        {
            "report": str(path),
            "blocks": [{"id": row["id"], "match": row["match"]} for row in report["blocks"]],
        },
    )
    save_status(obj, doc, root)
    _print_table(report["blocks"])
    return path


def _input_row(doc: dict[str, Any]) -> dict[str, Any]:
    record = doc.get("input") or {}
    match = (
        record.get("kind") == "polycam_mesh"
        and record.get("cameras") is False
        and record.get("capture_images") is False
    )
    return _row(
        "input",
        "Mesh-only LiDAR or photogrammetry, with no photographs and no COLMAP.",
        "status.json input",
        match,
        "input.kind is polycam_mesh and cameras and capture_images are false."
        if match
        else "The input is not a mesh-only Polycam scan.",
    )


def _gaussian_row(doc: dict[str, Any]) -> dict[str, Any]:
    orbit = ((doc.get("step02") or {}).get("orbit1") or {})
    splat = Path(str(orbit.get("splat") or ""))
    count = int(orbit.get("n_gaussians") or 0)
    match = splat.is_file() and count > 0 and orbit.get("photometric") is False and orbit.get("source") == "mesh"
    note = (
        f"Surface splat has {count} Gaussians and photometric is false. splatfacto was not run."
        if match
        else "A mesh-surface splat with photometric false is missing."
    )
    return _row(
        "3dgs",
        "3D Gaussians from the mesh or point cloud. Photometric scores stay unset without photos.",
        str(splat) if splat else "splat.ply",
        match,
        note,
    )


def _semantics_row(folder: Path) -> dict[str, Any]:
    record = _read_json(folder / "masks" / "object.json") or {}
    ran = record.get("grounding_dino") is True and record.get("sam") is True
    projected = record.get("projected") is True
    infill = record.get("infill")
    segmenter = record.get("segmenter") or "missing"
    if ran and projected and infill == "ran":
        match: bool | str = True
        note = "GroundingDINO and SAM masks were projected onto the Gaussians, and background infill ran."
    elif ran and projected and infill == "skipped":
        match = "partial"
        inside = record.get("n_inside")
        note = (
            f"GroundingDINO and SAM masks were projected onto {inside} Gaussians. "
            "Background infill is skipped for this scan, so it is not Achieved."
        )
    elif projected and infill == "skipped":
        match = "partial"
        note = (
            f"Masks from segmenter {segmenter} were projected onto the Gaussians. "
            "GroundingDINO and SAM did not both run. Background infill is skipped, so it is not Achieved."
        )
    else:
        match = False
        note = (
            f"Masks use segmenter {segmenter}. "
            "GroundingDINO and SAM did not run, so no detector labels were invented."
            if not ran
            else "Detector labels exist, but the masks were not projected onto the Gaussians."
        )
    return _row(
        "semantics",
        "GroundingDINO and SAM masks, lifted onto each Gaussian. Background infill is separate.",
        "masks/object.json",
        match,
        note,
    )


def _physics_row(folder: Path) -> dict[str, Any]:
    record = _read_json(folder / "physics.json") or {}
    codebook = _read_json(folder / "codebook.json") or {}
    per = record.get("per_gaussian") or {}
    source = record.get("source")
    learned = per.get("learned_prior") is True
    copied = per.get("copied_from") == "object"
    object_vlm = source == "vlm" and record.get("model") == "gpt-6-astra" and copied
    full = (
        object_vlm
        and learned
        and codebook.get("dinov2") is True
        and codebook.get("openclip") is True
    )
    if full:
        match: bool | str = True
        note = "Per-Gaussian physics came from DINOv2, OpenCLIP, and a learned prior."
    elif object_vlm and not learned:
        match = "partial"
        count = per.get("n")
        ran = [
            name
            for name, flag in (("DINOv2", codebook.get("dinov2")), ("OpenCLIP", codebook.get("openclip")))
            if flag is True
        ]
        missed = [
            name
            for name, flag in (("DINOv2", codebook.get("dinov2")), ("OpenCLIP", codebook.get("openclip")))
            if flag is not True
        ]
        if ran and not missed:
            encoders = f"{' and '.join(ran)} ran. "
        elif ran and missed:
            encoders = f"{' and '.join(ran)} ran. {' and '.join(missed)} did not. "
        else:
            encoders = "DINOv2 and OpenCLIP were not run. "
        note = (
            f"gpt-6-astra wrote object-level material, mass, friction, and rigidity, "
            f"copied onto {count} Gaussians. {encoders}"
            "The learned per-Gaussian prior is not released, so learned_prior stays false. "
            "OpenUSD stores one rigidity, not a different value on each Gaussian. "
            "The vision model does not review later steps."
        )
    else:
        match = False
        note = (
            f"source is {source or 'missing'}. gpt-6-astra did not supply the numbers. "
            "DINOv2, OpenCLIP, and a learned per-Gaussian prior were not run."
        )
    return _row(
        "physics",
        "Predictive physics: material, mass, friction, and rigidity from semantics and a vision model, copied onto the object's Gaussians.",
        "physics.json",
        match,
        note,
    )


def _parts_row(folder: Path) -> dict[str, Any]:
    record = _read_json(folder / "parts.json") or {}
    parts = record.get("parts") or []
    count = len(parts) if isinstance(parts, list) else 0
    partfield = record.get("partfield") is True
    if partfield and count >= 1:
        match: bool | str = True
        note = f"PartField returned {count} parts."
    elif count == 1 and not partfield:
        match = "partial"
        note = "partfield=false and n_parts=1. One rigid body is a stand-in; movable pieces were not invented."
    elif count > 1 and record.get("source") == "vlm":
        match = "partial"
        note = f"gpt-6-astra named {count - 1} movable parts. PartField did not choose the split."
    else:
        match = False
        note = "Parts are missing."
    return _row(
        "decomposition",
        "Part decomposition. One rigid body is enough only when the object has no movable pieces.",
        "parts.json",
        match,
        note,
    )


def _urdf_row(folder: Path) -> dict[str, Any]:
    path = folder / "asset.urdf"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    joints = text.lower().count("<joint")
    if path.is_file() and joints == 0:
        match = "partial"
        note = "URDF has zero joints. Partial until a real part split exists."
    elif path.is_file() and joints > 0:
        motion = _read_json(folder / "motion.json") or {}
        match = "partial"
        if motion.get("source") == "vlm":
            note = (
                f"URDF has {joints} joints from gpt-6-astra. "
                "Partial because this is not the unreleased Harvard articulator."
            )
        else:
            note = f"URDF has {joints} joints, without a PartField split to check them against."
    else:
        match = False
        note = "URDF is missing."
    return _row(
        "articulation",
        "Articulated URDF. Joints follow the part split.",
        "asset.urdf",
        match,
        note,
    )


def _align_row(doc: dict[str, Any]) -> dict[str, Any]:
    report = ((doc.get("align") or {}).get("report") or {})
    scale = report.get("scale_m")
    has_scale = isinstance(scale, (int, float)) and scale > 0
    photometric = report.get("photometric")
    residuals_ran = photometric not in (None, "skipped") and report.get("silhouette") not in (None, "skipped")
    if residuals_ran and has_scale:
        match: bool | str = True
        note = f"Photometric alignment ran. Scale is {scale} m."
    elif photometric == "skipped" and has_scale:
        match = "partial"
        note = (
            f"Scale is {scale} m from the cloud. Photometric, silhouette, and edge residuals "
            "are skipped, so alignment is not Achieved."
        )
    else:
        match = False
        note = "Mesh-only alignment was not recorded."
    return _row(
        "alignment",
        "Alignment. A mesh with no photos uses scale and an upright axis, not photometric alignment.",
        "align.json",
        match,
        note,
    )


def _orientation_row(doc: dict[str, Any]) -> dict[str, Any]:
    record = doc.get("orientation") or {}
    if record.get("isaac") is True:
        match: bool | str = True
        note = "Isaac Lab oriented the background."
    elif record.get("skipped") is True:
        match = False
        note = (
            "Orientation is skipped because this scan has no separate background. "
            "A skip is not Achieved."
        )
    else:
        match = False
        note = "Orientation was not recorded."
    return _row(
        "orientation",
        "Background orientation. A single object with no room background skips this.",
        "orientation.json",
        match,
        note,
    )


def _package_row(folder: Path, doc: dict[str, Any]) -> dict[str, Any]:
    usdz = folder / "scene.usdz"
    chamfer = (find_step(doc, "B8").get("actual") or {}).get("chamfer_cm")
    physics = _read_json(folder / "physics.json") or {}
    parts = _read_json(folder / "parts.json") or {}
    per = physics.get("per_gaussian") or {}
    full = (
        usdz.is_file()
        and isinstance(chamfer, (int, float))
        and physics.get("source") == "vlm"
        and parts.get("partfield") is True
        and per.get("learned_prior") is True
    )
    if full:
        match: bool | str = True
        note = f"USDZ exists. Chamfer is {chamfer} cm."
    elif usdz.is_file() and isinstance(chamfer, (int, float)):
        match = "partial"
        sisters = _read_json(folder / "sisters" / "objects.json") or {}
        trellis_note = (
            "TRELLIS sisters exist. "
            if sisters.get("trellis") is True
            else "TRELLIS has not run. "
        )
        note = (
            f"USDZ exists. Chamfer is {chamfer} cm. Partial because parts are a single body "
            "and learned per-Gaussian physics was not run. "
            f"{trellis_note}"
            "An Isaac render has not happened, so the file is not called packaged."
        )
    else:
        match = False
        note = "USDZ or Chamfer is missing."
    return _row(
        "package",
        "USDZ package and Chamfer distance in centimetres.",
        "scene.usdz",
        match,
        note,
    )


def _inspect_row(doc: dict[str, Any]) -> dict[str, Any]:
    record = doc.get("inspect") or {}
    match = record.get("isaac") is True
    note = (
        "Isaac Lab rendered the inspection view."
        if match
        else "Isaac Lab did not run. inspect.isaac is false."
    )
    return _row(
        "inspector",
        "Isaac Lab inspection render.",
        "status.json inspect",
        match,
        note,
    )


def _object_sisters_row(folder: Path) -> dict[str, Any]:
    record = _read_json(folder / "sisters" / "objects.json") or {}
    match = record.get("trellis") is True
    note = (
        "TRELLIS wrote the object sisters."
        if match
        else "Object sisters are procedural. trellis=false, so they are not the paper's TRELLIS sisters."
    )
    return _row(
        "object_sisters",
        "Generative object sisters from TRELLIS.",
        "sisters/objects.json",
        match,
        note,
    )


def _scene_sisters_row(folder: Path) -> dict[str, Any]:
    scenes = _read_json(folder / "sisters" / "scenes.json") or {}
    objects = _read_json(folder / "sisters" / "objects.json") or {}
    layouts = scenes.get("scenes") or []
    count = len(layouts) if isinstance(layouts, list) else 0
    trellis = objects.get("trellis") is True
    if count > 0 and trellis:
        match: bool | str = True
        note = f"{count} scene layouts use TRELLIS objects."
    elif count > 0:
        match = "partial"
        note = f"{count} scene layouts exist. The objects in them are not TRELLIS sisters."
    else:
        match = False
        note = "Scene layouts are missing."
    return _row(
        "scene_sisters",
        "Scene-level rearrangements of the object sisters.",
        "sisters/scenes.json",
        match,
        note,
    )


def _policy_row(doc: dict[str, Any]) -> dict[str, Any]:
    states = [find_step(doc, block_id).get("state") for block_id, _reason in _POLICY_BLOCKS]
    isaac = (doc.get("inspect") or {}).get("isaac") is True
    match = isaac and all(state == "pass" for state in states)
    note = (
        "Policy blocks passed inside Isaac Lab."
        if match
        else "Not run. Needs a robot, Isaac Lab, and checkpoints on the Vast VM."
    )
    return _row(
        "policy",
        "Downstream policy on a robot. Not part of the mesh package.",
        "status.json B15 B16 B17",
        match,
        note,
    )


def _row(block_id: str, requirement: str, artifact: str, match: bool | str, note: str) -> dict[str, Any]:
    return {
        "id": block_id,
        "paper_requirement": requirement,
        "artifact": artifact,
        "match": match,
        "note": note,
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must be a JSON object")
    return loaded


def _print_table(rows: list[dict[str, Any]]) -> None:
    sys.stdout.write(f"{'id':<16} {'match':<8} note\n")
    for row in rows:
        match = row["match"]
        label = "partial" if match == "partial" else ("true" if match is True else "false")
        sys.stdout.write(f"{row['id']:<16} {label:<8} {row['note']}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score laptop artifacts against the SceneAgent paper")
    parser.add_argument("--obj", default="scan")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    write_paper_check(args.obj, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
