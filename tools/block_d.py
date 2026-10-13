"""Block D writers for can: scene-mean copy, gated prior, VLM start, fuse/bake.

This module does not train PhysXNet, does not invent mass, and does not set
learned_prior true. A missing ply, checkpoint, or API key leaves the step
unstarted. A skipped check is never a pass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tools.block_gates import GateError, assign_gaussian_vectors, check_mass, lock_learned_prior
from tools.common.paths import object_dir, resolve_root

_D2_FIELDS = (
    "material",
    "mass_kg",
    "density",
    "static_friction",
    "dynamic_friction",
    "restitution",
    "rigidity",
    "deformable_type",
)
_DEFORMABLE = frozenset({"rigid", "cloth", "rope", "soft"})
_FRICTION_KEYS = ("static_friction", "dynamic_friction", "restitution")
ISAAC_HANG = "Isaac sim.reset is known to hang; the drop test was not run"
_N_MATERIALS = 8
_GEOM_DIM = 8


def apply_scene_mean(
    inside: np.ndarray,
    object_vector: np.ndarray,
    scene_mean: np.ndarray,
) -> dict[str, Any]:
    """Copy the object vector inside the mask. Outside gets a scene mean only when it differs."""
    from tools.orbit_blocks import assign_feature_indices, resolve_scene_mean

    codebook = {
        "vector": _as_list(object_vector),
        "scene_mean": _as_list(scene_mean),
        "scene_mean_measured": True,
    }
    resolved = resolve_scene_mean(codebook)
    assigned = assign_gaussian_vectors(inside, object_vector, scene_mean)
    index = assign_feature_indices(inside, resolved["measured"] is True)
    return {
        "assigned": assigned,
        "index": index,
        "scene_mean_measured": resolved["measured"] is True,
        "reason": resolved["reason"],
        "learned_prior": False,
    }


def write_scene_mean_labels(
    folder: Path,
    inside: np.ndarray,
    object_vector: np.ndarray,
    scene_mean: np.ndarray,
) -> dict[str, Any]:
    """Rewrite gaussian_feature_idx.npy. Label 1 is written only when the scene mean is separate."""
    result = apply_scene_mean(inside, object_vector, scene_mean)
    semantics = folder / "semantics"
    semantics.mkdir(parents=True, exist_ok=True)
    np.save(semantics / "gaussian_feature_idx.npy", result["index"])
    return result


def build_prior_input(
    vectors: np.ndarray,
    xyz: np.ndarray,
    size: np.ndarray,
    opacity: np.ndarray,
    colour: np.ndarray,
) -> np.ndarray:
    """Object-or-scene vector plus xyz (3), mean scale (1), opacity (1), colour (3)."""
    features = np.asarray(vectors, dtype=np.float64)
    if features.ndim != 2:
        raise GateError("prior vectors must be a 2-d array")
    n_rows, dim = features.shape
    pos = np.asarray(xyz, dtype=np.float64).reshape(n_rows, 3)
    scale = np.asarray(size, dtype=np.float64).reshape(n_rows, 1)
    alpha = np.asarray(opacity, dtype=np.float64).reshape(n_rows, 1)
    rgb = np.asarray(colour, dtype=np.float64).reshape(n_rows, 3)
    stacked = np.concatenate((features, pos, scale, alpha, rgb), axis=1)
    if stacked.shape != (n_rows, dim + _GEOM_DIM):
        raise GateError("prior input width must be vector dim plus 8")
    return stacked


def write_prior_input(
    obj: str,
    root: Path | None = None,
    *,
    ply: Path | None = None,
    vectors: np.ndarray | None = None,
) -> dict[str, Any]:
    """Write prior_input.npy only when a ply and assigned vectors both exist."""
    folder = object_dir(obj, root)
    from tools.colmap_lift import read_splat_columns
    from tools.step02_splatfacto import splat_path

    splat = ply if ply is not None else splat_path(obj, 1, root)
    if splat is None or not Path(splat).is_file():
        return {"state": "unstarted", "reason": "splat.ply is missing; geometry was not invented", "file": None}
    if vectors is None:
        return {"state": "unstarted", "reason": "assigned vectors are missing", "file": None}
    xyz, opacity, scales = read_splat_columns(Path(splat))
    colour = read_splat_colour(Path(splat), n_rows=xyz.shape[0])
    stacked = build_prior_input(vectors, xyz, scales, opacity, colour)
    out = folder / "semantics" / "prior_input.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stacked)
    return {"state": "written", "file": str(out), "shape": list(stacked.shape), "learned_prior": False}


def read_splat_colour(path: Path, n_rows: int) -> np.ndarray:
    """Return f_dc_0..2 when the ply has those columns. Missing colour is zeros, not a guess."""
    raw = path.read_bytes()
    header, _, payload = raw.partition(b"end_header\n")
    count = 0
    names: list[str] = []
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] != "float":
                raise ValueError(f"{path} is not a float splat table")
            names.append(parts[2])
    table = np.frombuffer(payload, dtype="<f4", count=count * len(names)).reshape(count, len(names))
    needed = ("f_dc_0", "f_dc_1", "f_dc_2")
    if all(name in names for name in needed):
        cols = [names.index(name) for name in needed]
        return np.asarray(table[:, cols], dtype=np.float64)
    return np.zeros((n_rows, 3), dtype=np.float64)


class PhysicsPrior:
    """Linear stack: concat → material logits, mass, friction, rigidity in [0, 1]."""

    def __init__(
        self,
        in_dim: int,
        n_materials: int = _N_MATERIALS,
        hidden: int = 32,
        weights: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.in_dim = int(in_dim)
        self.n_materials = int(n_materials)
        self.hidden = int(hidden)
        if weights is None:
            rng = np.random.default_rng(0)
            weights = {
                "w1": rng.normal(0.0, 0.05, (self.in_dim, self.hidden)),
                "b1": np.zeros(self.hidden, dtype=np.float64),
                "w2": rng.normal(0.0, 0.05, (self.hidden, self.hidden)),
                "b2": np.zeros(self.hidden, dtype=np.float64),
                "w_out": rng.normal(0.0, 0.05, (self.hidden, self.n_materials + 3)),
                "b_out": np.zeros(self.n_materials + 3, dtype=np.float64),
            }
        self.weights = {name: np.asarray(value, dtype=np.float64) for name, value in weights.items()}

    def forward(self, x: np.ndarray) -> dict[str, np.ndarray]:
        hidden = np.tanh(x @ self.weights["w1"] + self.weights["b1"])
        hidden = np.tanh(hidden @ self.weights["w2"] + self.weights["b2"])
        raw = hidden @ self.weights["w_out"] + self.weights["b_out"]
        logits = raw[:, : self.n_materials]
        bound = 1.0 / (1.0 + np.exp(-raw[:, self.n_materials :]))
        return {
            "material_logits": logits,
            "mass": bound[:, 0],
            "friction": bound[:, 1],
            "rigidity": bound[:, 2],
        }

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            in_dim=self.in_dim,
            n_materials=self.n_materials,
            hidden=self.hidden,
            **self.weights,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "PhysicsPrior":
        stored = np.load(path)
        return cls(
            int(stored["in_dim"]),
            n_materials=int(stored["n_materials"]),
            hidden=int(stored["hidden"]),
            weights={name: stored[name] for name in ("w1", "b1", "w2", "b2", "w_out", "b_out")},
        )


def run_learned_prior(x: np.ndarray, checkpoint: Path | None) -> dict[str, Any]:
    """Run a saved prior. A missing checkpoint is a hard refuse, not an untrained forward."""
    if checkpoint is None or not Path(checkpoint).is_file():
        raise GateError("learned prior checkpoint is missing")
    features = np.asarray(x, dtype=np.float64)
    if features.ndim != 2:
        raise GateError("prior input must be a 2-d array")
    model = PhysicsPrior.load(Path(checkpoint))
    if features.shape[1] != model.in_dim:
        raise GateError("prior input width does not match the checkpoint")
    outputs = model.forward(features)
    locked = lock_learned_prior({"learned_prior": False, "source": "checkpoint"})
    return {
        "p_i": outputs,
        "n": int(features.shape[0]),
        "learned_prior": False,
        "per_gaussian": locked["per_gaussian"],
    }


def reduce_prior(
    p_i: dict[str, np.ndarray],
    inside: np.ndarray,
    part_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Object mean over in-mask rows, inherited part records, and the continuous field."""
    mask = np.asarray(inside, dtype=bool).reshape(-1)
    if not mask.any():
        raise GateError("reduce_prior needs in-mask Gaussians")
    field = {name: np.asarray(value) for name, value in p_i.items()}
    object_record = _mean_record(field, mask)
    names = part_ids if part_ids else ["body"]
    parts = [{"id": name, **object_record, "inherit": True} for name in names]
    return {
        "object": object_record,
        "parts": parts,
        "field": field,
        "learned_prior": False,
    }


def start_d2(
    obj: str,
    frames: list[Path],
    name: str,
    bbox: dict[str, Any],
    *,
    root: Path | None = None,
    ask: Callable[[list[Path], str, dict[str, Any], str], dict[str, Any]] | None = None,
    api_key: str | None = None,
) -> str:
    """Create physics.json from four renders, a name, and a COLMAP bbox.

    No key or an unusable response does not write a class-default mass.
    """
    if obj == "bottle":
        raise GateError("start_d2 refuses bottle; bottle physics.json stays untouched")
    if obj == "vaseline":
        raise GateError("start_d2 refuses vaseline")
    path = object_dir(obj, root) / "physics.json"
    key = (api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
    existing = [Path(frame) for frame in frames if Path(frame).is_file()]
    if len(existing) != 4:
        return "model not asked"
    if ask is None and not key:
        return "model not asked"
    try:
        parsed = ask(existing, name, bbox, key) if ask is not None else _ask_start_d2(existing, name, bbox, key)
        record = write_d2_record(parsed, name, bbox, existing)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError, GateError):
        return "response not usable"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return "filled"


def write_d2_record(
    parsed: dict[str, Any],
    name: str,
    bbox: dict[str, Any],
    frames: list[Path],
) -> dict[str, Any]:
    """Validate a VLM payload and attach unchecked mass plus locked learned_prior."""
    record = parse_d2_response(parsed)
    mass = record.get("mass_kg")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool):
        raise GateError("d2 response is missing mass_kg")
    weighed = check_mass(float(mass), None)
    record["name"] = name
    record["bbox"] = dict(bbox)
    record["metres_per_unit"] = bbox.get("metres_per_unit")
    record["mass_check"] = weighed
    record["mass_checked"] = False
    record["source"] = "vlm"
    record["d2_model"] = "gpt-6-astra"
    record["d2_frames"] = [str(frame) for frame in frames]
    record["drop_test"] = "not_run"
    record["slide_test"] = "not_run"
    record["drop_test_reason"] = ISAAC_HANG
    return lock_learned_prior(record)


def parse_d2_response(parsed: dict[str, Any]) -> dict[str, Any]:
    """Keep the D2 schema. Missing mass stays missing; nothing is invented."""
    if not isinstance(parsed, dict):
        raise TypeError("d2 response must be an object")
    record: dict[str, Any] = {}
    material = parsed.get("material")
    if isinstance(material, str) and material.strip():
        record["material"] = material.strip()
    for name in ("mass_kg", "density", "static_friction", "dynamic_friction", "restitution", "rigidity"):
        value = _finite(parsed.get(name))
        if value is not None:
            record[name] = value
    kind = parsed.get("deformable_type")
    if kind in _DEFORMABLE:
        record["deformable_type"] = kind
    ranges = parsed.get("ranges")
    if isinstance(ranges, dict):
        record["ranges"] = ranges
    provenance = parsed.get("provenance")
    record["provenance"] = provenance if isinstance(provenance, dict) else {"source": "vlm"}
    if "rigidity" in record and not 0.0 <= float(record["rigidity"]) <= 1.0:
        raise GateError("rigidity must be in [0, 1]")
    if not record.get("material") or record.get("mass_kg") is None:
        raise GateError("d2 response is missing material or mass_kg")
    return record


def fuse_physics(
    prior: dict[str, Any] | None,
    vlm: dict[str, Any] | None,
    class_defaults: dict[str, Any] | None,
) -> dict[str, Any]:
    """Weighted fuse. Prior weight is 0 when missing. VLM owns mass. Class never fills mass."""
    prior_rec = dict(prior or {})
    vlm_rec = dict(vlm or {})
    defaults = dict(class_defaults or {})
    prior_ok = bool(prior_rec)
    fused: dict[str, Any] = {"provenance": {}, "learned_prior": False}
    for name in _D2_FIELDS:
        if name == "mass_kg":
            if _has_value(vlm_rec, name):
                fused[name] = vlm_rec[name]
                fused["provenance"][name] = "vlm"
            continue
        if name in _FRICTION_KEYS:
            if prior_ok and _has_value(prior_rec, name):
                fused[name] = prior_rec[name]
                fused["provenance"][name] = "prior"
            elif _has_value(vlm_rec, name):
                fused[name] = vlm_rec[name]
                fused["provenance"][name] = "vlm"
            elif _has_value(defaults, name):
                fused[name] = defaults[name]
                fused["provenance"][name] = "class_default"
            continue
        if prior_ok and _has_value(prior_rec, name):
            fused[name] = prior_rec[name]
            fused["provenance"][name] = "prior"
        elif _has_value(vlm_rec, name):
            fused[name] = vlm_rec[name]
            fused["provenance"][name] = "vlm"
    if "mass_kg" in defaults and "mass_kg" not in fused:
        fused["provenance"]["mass_kg"] = "class_default_refused"
    return lock_learned_prior(fused)


def write_physics_material_usda(path: Path, physics: dict[str, Any], prim: str = "Object") -> Path:
    """Bake UsdPhysics MaterialAPI and MassAPI. Mass is omitted when it is missing."""
    ident = _ident(prim)
    static = _finite(physics.get("static_friction"))
    dynamic = _finite(physics.get("dynamic_friction"))
    restitution = _finite(physics.get("restitution"))
    mass = _finite(physics.get("mass_kg"))
    lines = [
        "#usda 1.0",
        f'def Material "PhysicsMaterial" (',
        '    prepend apiSchemas = ["PhysicsMaterialAPI"]',
        ")",
        "{",
    ]
    if static is not None:
        lines.append(f"    float physics:staticFriction = {static}")
    if dynamic is not None:
        lines.append(f"    float physics:dynamicFriction = {dynamic}")
    if restitution is not None:
        lines.append(f"    float physics:restitution = {restitution}")
    lines.append("}")
    lines.append("")
    mass_schema = '    prepend apiSchemas = ["PhysicsMassAPI"]' if mass is not None else ""
    if mass is not None:
        lines.extend(
            [
                f'def Xform "{ident}" (',
                mass_schema,
                ")",
                "{",
                f"    float physics:mass = {mass}",
                "}",
                "",
            ]
        )
    else:
        lines.extend([f'def Xform "{ident}"', "{", "}", ""])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_gaussian_physics(
    path: Path,
    n_gaussians: int,
    prior_field: np.ndarray | None,
    vlm: dict[str, Any] | None,
) -> Path | None:
    """Write gaussian_physics.npy. A VLM-only map is a constant copy. No VLM means skip."""
    if prior_field is not None:
        field = np.asarray(prior_field, dtype=np.float64)
        copied_from = "prior"
    elif vlm:
        values = [_finite(vlm.get(name)) for name in ("mass_kg", "static_friction", "rigidity")]
        if all(item is None for item in values):
            return None
        row = np.array([0.0 if item is None else item for item in values], dtype=np.float64)
        field = np.repeat(row.reshape(1, -1), int(n_gaussians), axis=0)
        copied_from = "vlm"
    else:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, field)
    sidecar = path.with_suffix(".json")
    sidecar.write_text(
        json.dumps({"copied_from": copied_from, "n": int(field.shape[0]), "learned_prior": False}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return path


def record_isaac_drop_slide(
    telemetry: dict[str, Any] | None = None,
    hang_reason: str = ISAAC_HANG,
) -> dict[str, Any]:
    """Record drop/slide numbers, or not_run. settle_within_2s is never true without telemetry."""
    settle = _finite((telemetry or {}).get("settle_s")) if telemetry else None
    drop_ok = telemetry.get("drop_ok") is True if telemetry else False
    slide_ok = telemetry.get("slide_ok") is True if telemetry else False
    if telemetry is None or settle is None:
        return {
            "drop_test": "not_run",
            "slide_test": "not_run",
            "settle_within_2s": None,
            "reason": hang_reason,
            "learned_prior": False,
        }
    return {
        "drop_test": "ran" if drop_ok else "ran",
        "slide_test": "ran" if slide_ok else "ran",
        "settle_s": settle,
        "settle_within_2s": bool(settle <= 2.0),
        "drop_ok": drop_ok,
        "slide_ok": slide_ok,
        "learned_prior": False,
    }


def can_d2_frames(root: Path | None = None) -> list[Path]:
    """Four existing can renders: three orbit IoU captures and one review frame."""
    folder = object_dir("can", root)
    candidates = [
        folder / "iou" / "orbit1" / "frame_00167.png",
        folder / "iou" / "orbit2" / "frame_00001.png",
        folder / "iou" / "orbit3" / "frame_00125.png",
        folder / "reviews" / "frame_00001.png",
    ]
    return [path for path in candidates if path.is_file()]


def can_colmap_bbox(root: Path | None = None) -> dict[str, Any]:
    """COLMAP span already stored on the can crop. Metres were not measured."""
    masks = _read(object_dir("can", root) / "masks" / "orbit1" / "object.json")
    return {
        "span_inside": _finite(masks.get("span_inside")),
        "span_scene": _finite(masks.get("span_scene")),
        "units": "colmap",
        "metres_per_unit": None,
    }


def refresh_block_d(obj: str, root: Path | None = None) -> dict[str, Any]:
    """Rewrite ingest/block_d.json from artifacts. achieved stays false without the file."""
    folder = object_dir(obj, root)
    codebook = _read(folder / "codebook.json")
    physics = _read(folder / "physics.json")
    instances = _read(folder / "semantics" / "instances.json")
    idx_path = folder / "semantics" / "gaussian_feature_idx.npy"
    prior_path = folder / "semantics" / "prior_input.npy"
    usda_path = folder / "physics_material.usda"
    gauss_path = folder / "gaussian_physics.npy"
    from tools.orbit_blocks import resolve_scene_mean

    scene = resolve_scene_mean(codebook) if codebook else {"measured": False, "reason": "no codebook vector"}
    n_unassigned = None
    if idx_path.is_file():
        index = np.load(idx_path)
        n_unassigned = int((index == -1).sum())
    from tools.sceneagent_blocks import _block_d

    scored = _block_d(obj, folder, resolve_root(root) if root is not None else None)
    d12_measured = scene.get("measured") is True
    d13_written = prior_path.is_file()
    checkpoint = folder / "semantics" / "physics_prior.npz"
    d14_ran = checkpoint.is_file()
    d15_path = folder / "semantics" / "prior_reduced.json"
    d2_written = bool(physics.get("material") and physics.get("mass_kg") is not None)
    d3_usda = usda_path.is_file()
    d3_npy = gauss_path.is_file()
    drop = physics.get("drop_test") or "not_run"
    record = {
        "block": "D",
        "object": obj,
        "state": scored.get("state") or "not_started",
        "achieved": False,
        "learned_prior": False,
        "physics_json": bool(physics),
        "scorer": {
            "name": "_block_d",
            "state": scored.get("state"),
            "result": scored.get("result"),
            "artifact": scored.get("artifact"),
        },
        "m4": {
            "state": "unmeasured",
            "achieved": False,
            "m4_met": False,
            "m4_achieved": False,
            "dataset": False,
            "n_weighed_objects": 0,
            "median_mass_error": None,
            "material_class_accuracy": None,
            "settle_within_2s": None,
            "reason": (
                "M4 needs 20 weighed objects, median mass error below 30 percent, "
                "material class accuracy above 85 percent, and Isaac settle telemetry. "
                "None of those exist for can. A skipped check is never a pass."
            ),
        },
        "m4b": {
            "state": "unmeasured",
            "achieved": False,
            "m4b_met": False,
            "m4b_achieved": False,
            "reason": "M4b marker-cap breakaway force was not measured.",
        },
        "note": _block_d_note(d2_written, d12_measured, d13_written, d14_ran),
        "parts": [
            _part_d11(folder, codebook),
            {
                "id": "D1.2",
                "does": "Copy the object vector onto every Gaussian of object k; Gaussians outside any object get the scene-mean vector.",
                "artifact": "data/can/codebook.json, data/can/semantics/gaussian_feature_idx.npy, data/can/semantics/instances.json",
                "state": "measured" if d12_measured else "partial",
                "achieved": False,
                "learned_prior": False,
                "empirical": {
                    "copied_onto_object_gaussians": codebook.get("copied_onto_gaussians") is True,
                    "scene_mean_measured": d12_measured,
                    "outside": codebook.get("outside")
                    or (
                        "scene mean assigned"
                        if d12_measured
                        else "not assigned; no scene-mean vector was computed"
                    ),
                    "assignment": "label 1 outside" if d12_measured else "-1 unassigned for non-object Gaussians",
                    "n_copied_orbit1": codebook.get("n_copied"),
                    "n_unassigned_orbit1": n_unassigned,
                    "background_reason": (instances.get("background") or {}).get("reason") or scene.get("reason"),
                },
                "reason": (
                    "Object Gaussians received the one object vector. A separate scene-mean vector was measured."
                    if d12_measured
                    else "Object Gaussians received the one object vector. Scene-mean for outside Gaussians was not measured."
                ),
            },
            {
                "id": "D1.3",
                "does": "Per-Gaussian input = object vector plus that Gaussian's xyz, size, opacity, and colour.",
                "artifact": "data/can/semantics/prior_input.npy" if d13_written else "",
                "state": "measured" if d13_written else "unstarted",
                "achieved": False,
                "learned_prior": False,
                "reason": (
                    "prior_input.npy was written from splat.ply columns."
                    if d13_written
                    else "No per-Gaussian concatenated input was built. Can splat.ply is not on this machine."
                ),
            },
            {
                "id": "D1.4",
                "does": "Learned physics prior maps that input to p_i over material, mass, friction, and 0–1 rigidity.",
                "artifact": str(checkpoint) if d14_ran else "",
                "state": "measured" if d14_ran else "unstarted",
                "achieved": False,
                "learned_prior": False,
                "empirical": {
                    "physxnet": False,
                    "mlp": True,
                    "checkpoint": d14_ran,
                    "ran_on_can": False,
                },
                "reason": (
                    "PhysicsPrior Linear stack is implemented. run_learned_prior raises without a checkpoint. "
                    "No untrained forward was run on can Gaussians. learned_prior stays false."
                ),
            },
            {
                "id": "D1.5",
                "does": "Reduce p_i to object / part / continuous field records.",
                "artifact": str(d15_path) if d15_path.is_file() else "",
                "state": "measured" if d15_path.is_file() else "unstarted",
                "achieved": False,
                "learned_prior": False,
                "reason": (
                    "Reduced prior records were written."
                    if d15_path.is_file()
                    else "No prior exists to reduce. Can has no heterogeneous field and no production p_i."
                ),
            },
            {
                "id": "D2",
                "does": "VLM estimate from 4 renders, object name, and metric bbox → physics JSON schema.",
                "artifact": "data/can/physics.json" if d2_written else "",
                "state": "measured" if d2_written else "unstarted",
                "achieved": False,
                "learned_prior": False,
                "empirical": {
                    "physics_json": bool(physics),
                    "material": physics.get("material"),
                    "mass_kg": physics.get("mass_kg"),
                    "mass_invented": False,
                    "mass_checked_against_scale": bool((physics.get("mass_check") or {}).get("checked")),
                    "vlm_asked": physics.get("d2_model") == "gpt-6-astra",
                    "fill_missing_d2_ran": False,
                },
                "reason": (
                    "physics.json was written from a VLM response. Mass was not checked on a scale. learned_prior stays false."
                    if d2_written
                    else (
                        "Can has no physics.json. start_d2 does not write a class-default mass when the key is missing. "
                        "Bottle's VLM JSON was not copied here."
                    )
                ),
            },
            {
                "id": "D3",
                "does": "Weighted fuse of prior + VLM + class defaults; bake UsdPhysics MaterialAPI/MassAPI, gaussian_physics.npy; Isaac drop/slide.",
                "artifact": _d3_artifact(d3_usda, d3_npy),
                "state": "partial" if d3_usda or d3_npy else "unstarted",
                "achieved": False,
                "learned_prior": False,
                "empirical": {
                    "weighted_fuse": bool(physics.get("provenance")),
                    "physics_material_usda": d3_usda,
                    "gaussian_physics_npy": d3_npy,
                    "drop_test": drop,
                    "slide_test": physics.get("slide_test") or "not_run",
                    "isaac_drop_implemented": True,
                    "settle_within_2s": None,
                },
                "reason": (
                    "USD bake exists; Isaac drop/slide has no telemetry so settle_within_2s stays unmeasured."
                    if d3_usda or d3_npy
                    else "Prior and VLM record are missing, so there is nothing to fuse. No USD bake. Isaac drop/slide was not run."
                ),
            },
        ],
    }
    out = folder / "ingest" / "block_d.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return record


def _part_d11(folder: Path, codebook: dict[str, Any]) -> dict[str, Any]:
    orbits = []
    for orbit, name in (("orbit1", "codebook.json"), ("orbit2", "codebook_orbit2.json"), ("orbit3", "codebook_orbit3.json")):
        item = codebook if orbit == "orbit1" else _read(folder / name)
        if not item:
            continue
        vector = item.get("vector")
        orbits.append(
            {
                "orbit": orbit,
                "file": f"data/can/{name}",
                "view": item.get("view"),
                "cropped": item.get("cropped") is True,
                "n_copied": item.get("n_copied"),
                "n_gaussians": item.get("n_gaussians"),
                "dim": len(vector) if isinstance(vector, list) else None,
                "learned_prior": False,
            }
        )
    has_vector = isinstance(codebook.get("vector"), list) and len(codebook["vector"]) > 0
    return {
        "id": "D1.1",
        "does": "Masked crop of object k (Block B masks) → DINOv2 + OpenCLIP → one feature vector per object.",
        "artifact": "data/can/codebook.json, data/can/codebook_orbit2.json, data/can/codebook_orbit3.json, data/can/semantics/features.physics_vector.npz",
        "state": "measured" if has_vector else "unstarted",
        "achieved": False,
        "learned_prior": False,
        "empirical": {
            "encoder": codebook.get("encoder") or "dinov2+openclip",
            "dinov2": codebook.get("dinov2") is True,
            "openclip": codebook.get("openclip") is True,
            "dim": len(codebook["vector"]) if isinstance(codebook.get("vector"), list) else None,
            "copy": codebook.get("copy") or "one_object_vector",
            "langsplat_codebook": False,
            "orbits": orbits,
        },
        "reason": (
            "32-d object vectors exist on orbits 1–3. That is D1.1 only. achieved stays false because D1 is not a finished prior."
            if has_vector
            else "No object vector."
        ),
    }


def _block_d_note(d2_written: bool, d12_measured: bool, d13_written: bool, d14_ran: bool) -> str:
    bits = [
        "A skipped check is never a pass.",
        "learned_prior stays false.",
        "M4 stays unmeasured.",
    ]
    if not d2_written:
        bits.append("Can has no physics.json, so Block D overall is not_started.")
    if d12_measured:
        bits.append("D1.2 scene mean was measured separately.")
    else:
        bits.append("D1.2 scene mean was not a separate measurement.")
    if not d13_written:
        bits.append("D1.3 prior_input.npy was not written (no ply).")
    if not d14_ran:
        bits.append("D1.4 has no checkpoint; no untrained forward was run on can Gaussians.")
    bits.append("Mass was not invented from a soda-can guess.")
    return " ".join(bits)


def _d3_artifact(usda: bool, npy: bool) -> str:
    names = []
    if usda:
        names.append("physics_material.usda")
    if npy:
        names.append("gaussian_physics.npy")
    return ", ".join(names)


def _ask_start_d2(frames: list[Path], name: str, bbox: dict[str, Any], api_key: str) -> dict[str, Any]:
    import base64
    import urllib.request

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Reply with JSON only for this object. Name: "
                + name
                + ". COLMAP bbox: "
                + json.dumps(bbox)
                + ". metres_per_unit is null; do not invent metres or a class-default mass. "
                "Schema keys: material, mass_kg, density, static_friction, dynamic_friction, "
                "restitution, rigidity (0 to 1), deformable_type (rigid|cloth|rope|soft), "
                "ranges (object of [low, high] per numeric field), provenance."
            ),
        }
    ]
    for frame in frames:
        encoded = base64.b64encode(frame.read_bytes()).decode("ascii")
        mime = "image/jpeg" if frame.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    payload = {
        "model": "gpt-6-astra",
        "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": content}],
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        body = json.loads(response.read().decode("utf-8"))
    parsed = json.loads(body["choices"][0]["message"]["content"])
    if not isinstance(parsed, dict):
        raise TypeError("d2 response must be an object")
    return parsed


def _mean_record(field: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in field.items():
        arr = np.asarray(value)
        rows = arr[mask] if arr.shape[0] == mask.size else arr
        out[name] = np.mean(rows, axis=0).tolist() if rows.size else None
    return out


def _has_value(record: dict[str, Any], name: str) -> bool:
    if name not in record:
        return False
    value = record[name]
    if value is None or value == "":
        return False
    if name == "material":
        return isinstance(value, str) and bool(value.strip())
    if name == "deformable_type":
        return value in _DEFORMABLE
    return _finite(value) is not None


def _as_list(vector: np.ndarray | list[float]) -> list[float]:
    return [float(value) for value in np.asarray(vector, dtype=float).reshape(-1)]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ident(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        return f"object_{cleaned}"
    return cleaned


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
