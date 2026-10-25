"""Write the system-design workspace from artifacts that already exist.

The contract is in docs/sceneagent_system_design.md. This does not train a
splat, does not call COLMAP, and does not invent the learned physics prior.
Outside Gaussians stay unmarked until a real scene-mean vector is computed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.common.paths import object_dir
from tools.step01_colmap import evaluate_colmap, parse_analyzer
from tools.step02_splatfacto import count_gaussians, splat_path


def build_workspace(obj: str, root: Path | None = None) -> dict[str, str]:
    """Write ingest, semantics, scale, and the cropped object splat."""
    folder = object_dir(obj, root)
    written: dict[str, str] = {}
    written["ingest_report"] = str(_write_ingest(obj, folder, root))
    written["scale"] = str(_write_scale(folder))
    written["instances"] = str(_write_instances(folder))
    field = _write_feature_field(folder)
    if field is not None:
        written["feature_index"] = str(field)
    cropped = _crop_object(obj, folder, root)
    if cropped is not None:
        written["object_ply"] = str(cropped)
    from tools.orbit_blocks import apply_measured_records

    apply_measured_records(obj, folder, root)
    return written


def _write_ingest(obj: str, folder: Path, root: Path | None) -> Path:
    ply = splat_path(obj, 1, root)
    analyzer = folder / "colmap" / "orbit1" / "analyzer.txt"
    metrics = parse_analyzer(analyzer.read_text(encoding="utf-8")) if analyzer.is_file() else {}
    models_path = folder / "colmap" / "orbit1" / "models.json"
    models = json.loads(models_path.read_text(encoding="utf-8")) if models_path.is_file() else []
    gates = evaluate_colmap(metrics, models if isinstance(models, list) else [], 0) if metrics else {}
    report = {
        "block": "A",
        "splat": str(ply) if ply.is_file() else None,
        "n_gaussians": count_gaussians(ply) if ply.is_file() else None,
        "registered_pct": metrics.get("registered_pct"),
        "mean_reproj_px": metrics.get("mean_reproj_px"),
        "colmap_gates": gates,
        "psnr": None,
        "lpips": None,
        "m1_met": False,
        "note": "Existing splat kept. Held-out PSNR and LPIPS were not measured.",
    }
    return _dump(folder / "ingest" / "ingest_report.json", report)


def _write_scale(folder: Path) -> Path:
    record = {
        "metres_per_unit": None,
        "method": "not_measured",
        "reason": "No AprilTag board or known length was recorded for this object.",
    }
    return _dump(folder / "ingest" / "scale.json", record)


def _write_instances(folder: Path) -> Path:
    masks = _read(folder / "masks" / "object.json")
    codebook = _read(folder / "codebook.json")
    name = str(masks.get("label") or codebook.get("label") or folder.name)
    inside = masks.get("n_inside")
    outside = masks.get("n_outside")
    record = {
        "block": "B",
        "instances": [
            {
                "name": name,
                "foreground": True,
                "n_gaussians": inside,
                "detector": masks.get("segmenter"),
                "projected": masks.get("projected") is True,
            }
        ]
        if masks
        else [],
        "background": {
            "n_gaussians": outside,
            "feature": "scene_mean_not_computed",
        },
        "codebook": "one object vector" if codebook.get("dinov2") is True else "missing",
        "langsplat_codebook": False,
        "miou": None,
    }
    return _dump(folder / "semantics" / "instances.json", record)


def _write_feature_field(folder: Path) -> Path | None:
    """Index 0 is the object vector. Outside points stay -1 until a scene mean exists."""
    mask_path = folder / "masks" / "gaussian_in_mask.npy"
    codebook = _read(folder / "codebook.json")
    vector = codebook.get("vector")
    if not mask_path.is_file() or not isinstance(vector, list):
        return None
    inside = np.load(mask_path)
    if inside.dtype != np.bool_ and inside.dtype != bool:
        inside = inside.astype(bool)
    index = np.full(inside.shape, -1, dtype=np.int8)
    index[inside] = 0
    semantics = folder / "semantics"
    semantics.mkdir(parents=True, exist_ok=True)
    index_path = semantics / "gaussian_feature_idx.npy"
    np.save(index_path, index)
    np.savez(
        semantics / "features.codebook.npz",
        object_vector=np.asarray(vector, dtype=np.float32),
        scene_mean=np.array([], dtype=np.float32),
    )
    return index_path


def _outside_box(xyz: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """True where a point is outside the mean ± sigma box. Same rule as count_floaters."""
    cloud = np.asarray(xyz, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 3:
        return np.zeros((0 if cloud.ndim != 2 else cloud.shape[0],), dtype=bool)
    points = cloud[:, 0:3]
    center = points.mean(axis=0)
    spread = points.std(axis=0)
    return np.any(np.abs(points - center) > (sigma * spread), axis=1)


def crop_splat(
    splat: Path,
    mask: np.ndarray,
    destination: Path,
    background: Path | None = None,
) -> dict[str, Any]:
    """Write masked Gaussians, recentered, and the scene Gaussians that remain.

    Opacity below 0.05 is dropped. Object Gaussians outside the 2-sigma box are
    dropped and counted. The translation removed from the object ply is the bbox
    center of the Gaussians that were written. Infill is not run. background.ply
    keeps the Gaussians outside the mask, in the original world frame.
    """
    header, table = _read_splat_table(splat)
    flags = np.asarray(mask, dtype=bool).reshape(-1)
    if len(flags) != len(table):
        raise ValueError(f"mask length {len(flags)} does not match {len(table)} gaussians")
    names = _property_names(header)
    opacity = 1.0 / (1.0 + np.exp(-table[:, names.index("opacity")].astype(np.float64)))
    keep = flags & (opacity >= 0.05)
    chosen = np.array(table[keep], copy=True)
    n_removed = 0
    if len(chosen):
        outside = _outside_box(chosen[:, 0:3])
        n_removed = int(outside.sum())
        if n_removed:
            chosen = np.array(chosen[~outside], copy=True)
    if len(chosen) == 0:
        center = None
        floater_count = 0
        _write_splat(destination, header, np.zeros((0, table.shape[1]), dtype=np.float32))
    else:
        chosen = np.array(chosen, copy=True)
        floater_count = int(_outside_box(chosen[:, 0:3]).sum())
        xyz = chosen[:, 0:3]
        center_arr = (xyz.min(axis=0) + xyz.max(axis=0)) / 2.0
        chosen[:, 0:3] = xyz - np.asarray(center_arr, dtype=np.float32)
        _write_splat(destination, header, chosen)
        center = [float(value) for value in center_arr]
    background_rows = np.array(table[~flags], copy=True)
    if background is not None:
        _write_splat(background, header, background_rows)
    return {
        "n_in": int(len(chosen)),
        "n_background": int(len(background_rows)),
        "n_source": int(len(table)),
        "n_removed_floaters": n_removed,
        "floater_count": floater_count,
        "center": center,
        "infill": "not_run",
    }


def _crop_object(obj: str, folder: Path, root: Path | None) -> Path | None:
    mask_path = folder / "masks" / "gaussian_in_mask.npy"
    ply = splat_path(obj, 1, root)
    if not mask_path.is_file() or not ply.is_file():
        return None
    masks = _read(folder / "masks" / "object.json")
    name = str(masks.get("label") or obj)
    destination = folder / "objects" / name / "object.ply"
    partial = destination.with_name("object.partial.ply")
    try:
        stats = crop_splat(ply, np.load(mask_path), partial)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        if partial.is_file():
            partial.unlink()
        if destination.is_file() and destination.stat().st_size > 0:
            _dump(
                destination.parent / "crop_error.json",
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "kept_existing_object_ply": True,
                    "note": "crop_splat did not finish. The existing object.ply was kept and no Gaussians were invented.",
                },
            )
            return destination
        _dump(
            destination.parent / "crop_report.json",
            {
                "block": "C",
                "object": name,
                "error": f"{type(exc).__name__}: {exc}",
                "object_ply": None,
                "floater_count": None,
                "infill": "skipped",
                "fine_tune": "not_run",
                "m3_met": False,
                "note": "crop_splat did not finish. object.ply was not invented.",
            },
        )
        return None
    partial.replace(destination)
    error_note = destination.parent / "crop_error.json"
    if error_note.is_file():
        error_note.unlink()
    center = stats.get("center")
    if isinstance(center, list) and len(center) == 3:
        _dump(
            folder / "objects" / name / "world_T_object.json",
            {
                "translation": center,
                "rotation_wxyz": [1.0, 0.0, 0.0, 0.0],
                "scale": 1.0,
                "infill": "not_run",
            },
        )
    return destination


def _read_splat_table(path: Path) -> tuple[bytes, np.ndarray]:
    raw = path.read_bytes()
    header, marker, payload = raw.partition(b"end_header\n")
    if not marker:
        raise ValueError(f"{path} has no PLY header")
    count = 0
    names: list[str] = []
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
        elif line.startswith("property"):
            parts = line.split()
            if parts[1] != "float":
                raise ValueError(f"{path} is not a float splat")
            names.append(parts[2])
    width = len(names)
    table = np.frombuffer(payload, dtype="<f4", count=count * width).reshape(count, width)
    return header + marker, table


def _property_names(header: bytes) -> list[str]:
    names = []
    for line in header.decode("ascii").splitlines():
        if line.startswith("property"):
            names.append(line.split()[-1])
    return names


def _write_splat(path: Path, header: bytes, table: np.ndarray) -> None:
    text = header.decode("ascii")
    lines = []
    for line in text.splitlines():
        if line.startswith("element vertex"):
            lines.append(f"element vertex {len(table)}")
        else:
            lines.append(line)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = table.astype("<f4", copy=False).tobytes()
    path.write_bytes(("\n".join(lines) + "\n").encode("ascii") + blob)


def _dump(path: Path, record: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}
