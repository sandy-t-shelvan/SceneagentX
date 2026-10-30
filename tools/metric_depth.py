"""Metric scale without an AprilTag.

Writes data/<obj>/ingest/scale.json with metres_per_unit only when a real
measurement method ran:

  depth_anything | foundation_stereo | polycam | ruler | tag36h11

A VLM or model_knowledge height guess is never written as metres_per_unit.
Missing Depth Anything / FoundationStereo weights leave metres_per_unit null
with achieved false (ScoreOnly). skip ≠ pass.

Vast GPU (when ready):
  export SCENEAGENT_DEPTH_ANYTHING_WEIGHTS=/workspace/models/depth_anything_3
  # optional stereo pair:
  export SCENEAGENT_FOUNDATION_STEREO_WEIGHTS=/workspace/models/foundation_stereo
  pip install the matching DA3 / FoundationStereo package into the sa env.
  Then: python tools/metric_depth.py --obj <name> --orbit 1
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import math
import os
from typing import Any, Callable

import numpy as np

from tools.common.paths import frames_dir, object_dir
from tools.gate_policy import SCORE_ONLY, classify, record_ledger, skill_exit

MEASURED_METHODS = frozenset(
    {
        "depth_anything",
        "foundation_stereo",
        "polycam",
        "ruler",
        "tag36h11",
    }
)
REFUSED_AS_CHECKED = frozenset(
    {
        "model_knowledge",
        "vlm",
        "vlm_height",
        "object_knowledge",
        "knowledge",
    }
)
UNMEASURED = "not_measured"

DepthPredictor = Callable[[np.ndarray, dict[str, Any]], np.ndarray]

_DA3_MODEL_CACHE: dict[str, Any] = {}


class ScaleRefuseError(ValueError):
    """Raised when a caller asks to stamp a VLM/knowledge guess as measured."""


class OrbitDepthInputError(ValueError):
    """Honest missing-input failure for orbit depth recover (not a model crash)."""

    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def scale_path(obj: str, root: Path | None = None) -> Path:
    return object_dir(obj, root) / "ingest" / "scale.json"


def normalize_method(method: str | None) -> str:
    return str(method or "").strip().lower().replace("-", "_").replace(" ", "_")


def is_measured_record(record: dict[str, Any] | None) -> bool:
    if not isinstance(record, dict):
        return False
    method = normalize_method(record.get("method"))
    if method not in MEASURED_METHODS:
        return False
    metres = record.get("metres_per_unit")
    try:
        value = float(metres)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def refuse_knowledge_as_metres(method: str) -> None:
    """Never allow model_knowledge / VLM as a checked metres_per_unit."""
    key = normalize_method(method)
    if key in REFUSED_AS_CHECKED or key.startswith("vlm") or "knowledge" in key:
        raise ScaleRefuseError(
            f"method {method!r} is model_knowledge/VLM and cannot set metres_per_unit"
        )


def metres_per_unit_from_depth_pairs(
    colmap_depths: np.ndarray,
    metric_depths: np.ndarray,
    *,
    min_pairs: int = 32,
) -> float:
    """Median metric_z / colmap_z for corresponding samples.

    colmap_depths are reconstruction units (positive Z in camera frame).
    metric_depths are metres from a metric depth model at the same pixels.
    """
    colmap = np.asarray(colmap_depths, dtype=np.float64).reshape(-1)
    metric = np.asarray(metric_depths, dtype=np.float64).reshape(-1)
    if colmap.shape != metric.shape:
        raise ValueError("colmap and metric depth arrays must match")
    valid = (
        np.isfinite(colmap)
        & np.isfinite(metric)
        & (colmap > 1e-6)
        & (metric > 1e-6)
    )
    if int(valid.sum()) < min_pairs:
        raise ValueError(f"need at least {min_pairs} valid depth pairs, got {int(valid.sum())}")
    ratios = metric[valid] / colmap[valid]
    scale = float(np.median(ratios))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("depth ratio did not yield a positive finite metres_per_unit")
    return scale


def metres_per_unit_from_ruler(length_m: float, length_units: float) -> float:
    if length_m <= 0 or length_units <= 0:
        raise ValueError("ruler length_m and length_units must be positive")
    return float(length_m) / float(length_units)


def unmeasured_record(
    *,
    reason: str,
    attempted: str | None = None,
    status: str = "not_run",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "metres_per_unit": None,
        "method": UNMEASURED,
        "achieved": False,
        "status": status,
        "reason": reason,
    }
    if attempted:
        record["attempted"] = normalize_method(attempted)
    return record


def measured_record(
    metres_per_unit: float,
    method: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refuse_knowledge_as_metres(method)
    key = normalize_method(method)
    if key not in MEASURED_METHODS:
        raise ScaleRefuseError(f"method {method!r} is not an allowed measured scale method")
    value = float(metres_per_unit)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("metres_per_unit must be a positive finite number")
    # Same shape as polycam / ruler / tag36h11: finite metres + achieved true.
    # m1_met is never set here; block_gates / gate_policy keep milestones honest.
    record: dict[str, Any] = {
        "metres_per_unit": value,
        "method": key,
        "achieved": True,
        "status": "measured",
        "reason": f"scale from {key}",
    }
    if extra:
        record.update(extra)
    return record


def write_scale_json(obj: str, record: dict[str, Any], root: Path | None = None) -> Path:
    """Persist scale.json. Refuses knowledge methods with a finite metres value."""
    method = normalize_method(record.get("method"))
    metres = record.get("metres_per_unit")
    if metres is not None:
        refuse_knowledge_as_metres(method)
        if method not in MEASURED_METHODS:
            raise ScaleRefuseError(f"refusing to write metres_per_unit with method {method!r}")
    path = scale_path(obj, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(record)
    if payload.get("metres_per_unit") is None:
        payload["achieved"] = False
        if normalize_method(payload.get("method")) in REFUSED_AS_CHECKED or "knowledge" in method:
            # Never leave a knowledge method stamp as if it were measured.
            payload["method"] = UNMEASURED
            payload["status"] = "refused_knowledge"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_scale_json(obj: str, root: Path | None = None) -> dict[str, Any]:
    path = scale_path(obj, root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def depth_backend_status(prefer: str = "depth_anything") -> dict[str, Any]:
    """Probe whether a metric depth model can run. Does not invent numbers."""
    prefer_key = normalize_method(prefer) or "depth_anything"
    if prefer_key == "foundation_stereo":
        weights = os.environ.get("SCENEAGENT_FOUNDATION_STEREO_WEIGHTS", "").strip()
        env_key = "SCENEAGENT_FOUNDATION_STEREO_WEIGHTS"
        import_names = ("foundation_stereo", "FoundationStereo")
    else:
        prefer_key = "depth_anything"
        weights = os.environ.get("SCENEAGENT_DEPTH_ANYTHING_WEIGHTS", "").strip()
        env_key = "SCENEAGENT_DEPTH_ANYTHING_WEIGHTS"
        import_names = ("depth_anything_3", "depth_anything_v2", "depth_anything")

    if not weights or not Path(weights).exists():
        return {
            "available": False,
            "method": prefer_key,
            "status": "missing_weights",
            "reason": (
                f"{prefer_key} weights not found. Set {env_key} to the checkpoint "
                "directory on the Vast host (sa env), then re-run metric_depth."
            ),
        }

    module_ok = False
    for name in import_names:
        try:
            __import__(name)
            module_ok = True
            break
        except ImportError:
            continue
    if not module_ok:
        return {
            "available": False,
            "method": prefer_key,
            "status": "missing_package",
            "reason": (
                f"{prefer_key} Python package is not importable in this env. "
                f"Install it on Vast into the sa env; weights path is {weights}."
            ),
        }
    return {
        "available": True,
        "method": prefer_key,
        "status": "ready",
        "reason": f"{prefer_key} weights and package are present",
        "weights": weights,
    }


def predict_metric_depth(
    image_bgr: np.ndarray,
    intrinsics: dict[str, Any] | None = None,
    *,
    prefer: str = "depth_anything",
    predictor: DepthPredictor | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return (depth_m HxW, meta). Raises when the model is missing or fails."""
    prefer_key = normalize_method(prefer) or "depth_anything"
    if predictor is not None:
        depth = np.asarray(predictor(image_bgr, dict(intrinsics or {})), dtype=np.float64)
        if depth.ndim != 2 or depth.size == 0:
            raise ValueError("depth predictor must return a non-empty HxW array")
        return depth, {"method": prefer_key, "status": "measured"}

    status = depth_backend_status(prefer_key)
    if not status.get("available"):
        raise RuntimeError(status.get("reason") or "metric depth backend unavailable")

    if prefer_key == "foundation_stereo":
        raise RuntimeError(
            "foundation_stereo weights/package are present but no in-process stereo "
            "runner is wired yet; pass predictor= or use depth_anything"
        )
    if prefer_key != "depth_anything":
        raise RuntimeError(f"no in-process runner for method {prefer_key}")

    depth, meta = _predict_depth_anything(image_bgr, intrinsics, weights=str(status["weights"]))
    meta.setdefault("method", "depth_anything")
    meta.setdefault("status", "measured")
    return depth, meta


def estimate_from_depth_map(
    depth_m: np.ndarray,
    colmap_depths: np.ndarray,
    *,
    method: str = "depth_anything",
    min_pairs: int = 32,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    refuse_knowledge_as_metres(method)
    scale = metres_per_unit_from_depth_pairs(colmap_depths, depth_m, min_pairs=min_pairs)
    payload = {"n_pairs": int(np.asarray(colmap_depths).size), "min_pairs": min_pairs}
    if extra:
        payload.update(extra)
    return measured_record(scale, method, extra=payload)


def recover_metric_scale(
    obj: str,
    root: Path | None = None,
    *,
    orbit: int = 1,
    method: str | None = None,
    length_m: float | None = None,
    length_units: float | None = None,
    metres_per_unit: float | None = None,
    colmap_depths: Any = None,
    metric_depths: np.ndarray | None = None,
    predictor: DepthPredictor | None = None,
    image_bgr: np.ndarray | None = None,
    force: bool = False,
    record_gate: bool = True,
) -> dict[str, Any]:
    """Resolve scale.json for one object. Never invents metres from a VLM.

    Priority when method is omitted:
      existing measured → tag36h11 file → depth backend → not_measured ScoreOnly
    Explicit ruler / polycam / depth paths run when requested.
    """
    existing = read_scale_json(obj, root)
    if not force and is_measured_record(existing):
        return dict(existing)

    requested = normalize_method(method) if method else ""
    if requested:
        refuse_knowledge_as_metres(requested)

    try:
        if requested in REFUSED_AS_CHECKED:
            refuse_knowledge_as_metres(requested)

        if requested == "ruler" or (length_m is not None and length_units is not None and not requested):
            if length_m is None or length_units is None:
                raise ValueError("ruler scale needs --length-m and --length-units")
            record = measured_record(
                metres_per_unit_from_ruler(length_m, length_units),
                "ruler",
                extra={"length_m": float(length_m), "length_units": float(length_units)},
            )
            write_scale_json(obj, record, root)
            return record

        if requested == "polycam":
            if metres_per_unit is None:
                raise ValueError("polycam scale needs --metres-per-unit (typically 1.0 for LiDAR metres)")
            record = measured_record(float(metres_per_unit), "polycam")
            write_scale_json(obj, record, root)
            return record

        if requested == "tag36h11":
            if metres_per_unit is None:
                raise ValueError("tag36h11 scale needs metres_per_unit from tag corners")
            record = measured_record(float(metres_per_unit), "tag36h11")
            write_scale_json(obj, record, root)
            return record

        tag = _read_json(object_dir(obj, root) / "ingest" / "tag_measurement.json")
        if not requested and is_measured_record(tag):
            record = measured_record(
                float(tag["metres_per_unit"]),
                str(tag.get("method") or "tag36h11"),
                extra={"source": "ingest/tag_measurement.json"},
            )
            write_scale_json(obj, record, root)
            return record

        depth_method = requested if requested in {"depth_anything", "foundation_stereo"} else "depth_anything"
        if metric_depths is not None and colmap_depths is not None:
            record = estimate_from_depth_map(
                np.asarray(metric_depths, dtype=np.float64),
                np.asarray(colmap_depths, dtype=np.float64),
                method=depth_method,
            )
            write_scale_json(obj, record, root)
            return record

        if predictor is not None:
            record = _recover_with_predictor(
                obj,
                root,
                orbit=orbit,
                depth_method=depth_method,
                predictor=predictor,
                image_bgr=image_bgr,
                colmap_depths=colmap_depths,
            )
            write_scale_json(obj, record, root)
            return record

        # Live backend: probe when missing; run DA3 when ready + COLMAP/frames present.
        if requested in {"depth_anything", "foundation_stereo"} or method is None:
            status = depth_backend_status(depth_method if requested else "depth_anything")
            if not status.get("available"):
                record = unmeasured_record(
                    reason=str(status.get("reason") or "metric depth unavailable"),
                    attempted=str(status.get("method") or depth_method),
                    status=str(status.get("status") or "not_run"),
                )
                write_scale_json(obj, record, root)
                if record_gate:
                    _score_only_unmeasured(obj, root, record)
                return record

            try:
                record = _recover_live_depth(
                    obj,
                    root,
                    orbit=orbit,
                    depth_method=str(status.get("method") or depth_method),
                    image_bgr=image_bgr,
                    colmap_depths=colmap_depths,
                )
            except OrbitDepthInputError as exc:
                record = unmeasured_record(
                    reason=exc.reason,
                    attempted=str(status.get("method") or depth_method),
                    status=exc.status,
                )
                write_scale_json(obj, record, root)
                if record_gate:
                    _score_only_unmeasured(obj, root, record)
                return record
            write_scale_json(obj, record, root)
            return record

        raise ValueError(f"unsupported scale method: {method}")
    except ScaleRefuseError:
        raise
    except OrbitDepthInputError as exc:
        record = unmeasured_record(
            reason=exc.reason,
            attempted=requested or "depth_anything",
            status=exc.status,
        )
        write_scale_json(obj, record, root)
        if record_gate:
            _score_only_unmeasured(obj, root, record)
        return record
    except (ValueError, RuntimeError, OSError) as exc:
        record = unmeasured_record(reason=str(exc), attempted=requested or "depth_anything", status="failed")
        write_scale_json(obj, record, root)
        if record_gate:
            _score_only_unmeasured(obj, root, record)
        return record


def sample_depth_at_pixels(depth_m: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-neighbour sample of an HxW depth map at pixel coordinates (u, v)."""
    depth = np.asarray(depth_m, dtype=np.float64)
    pixels = np.asarray(uv, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_m must be HxW")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("uv must be Nx2")
    height, width = depth.shape
    u = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, width - 1)
    v = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, height - 1)
    return depth[v, u]


def transforms_path(obj: str, orbit: int, root: Path | None = None) -> Path | None:
    """Return the first existing Nerfstudio/COLMAP transforms.json for an orbit."""
    base = object_dir(obj, root) / "colmap"
    candidates = [
        base / f"orbit{orbit}" / "transforms.json",
        base / "transforms.json",
        object_dir(obj, root) / "ingest" / "cameras.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def load_orbit_intrinsics(obj: str, orbit: int, root: Path | None = None) -> dict[str, Any]:
    path = transforms_path(obj, orbit, root)
    if path is None:
        raise OrbitDepthInputError("missing_transforms", "missing_transforms: no transforms.json for orbit")
    data = _read_json(path)
    if not data:
        raise OrbitDepthInputError("missing_transforms", f"missing_transforms: unreadable {path}")
    intrinsics = _intrinsics_from_transforms(data)
    if intrinsics is None:
        raise OrbitDepthInputError(
            "missing_intrinsics",
            "missing_intrinsics: transforms.json lacks fx/fy/cx/cy/width/height",
        )
    return {"path": path, "transforms": data, "intrinsics": intrinsics}


def colmap_depth_samples_for_orbit(
    obj: str,
    orbit: int,
    root: Path | None = None,
    *,
    max_samples: int = 4096,
    min_pairs: int = 32,
) -> dict[str, Any]:
    """Project reconstruction points into the first transforms frame → {z, uv, ...}."""
    loaded = load_orbit_intrinsics(obj, orbit, root)
    transforms = loaded["transforms"]
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise OrbitDepthInputError("missing_transforms", "missing_transforms: transforms.json has no frames")
    frame = frames[0]
    if not isinstance(frame, dict) or "transform_matrix" not in frame:
        raise OrbitDepthInputError("missing_transforms", "missing_transforms: first frame lacks transform_matrix")

    xyz, source = _load_orbit_points(obj, orbit, root, loaded["path"].parent)
    if xyz is None or xyz.size == 0:
        raise OrbitDepthInputError(
            "missing_colmap_depths",
            "missing_colmap_depths: no sparse points3D or splat.ply to project",
        )

    from tools.colmap_lift import project_points

    intrinsics = loaded["intrinsics"]
    c2w = np.asarray(frame["transform_matrix"], dtype=np.float64)
    k1 = float(transforms.get("k1", 0.0) or 0.0)
    k2 = float(transforms.get("k2", 0.0) or 0.0)
    p1 = float(transforms.get("p1", 0.0) or 0.0)
    p2 = float(transforms.get("p2", 0.0) or 0.0)
    pixels, depth = project_points(
        xyz,
        c2w,
        float(intrinsics["fx"]),
        float(intrinsics["fy"]),
        float(intrinsics["cx"]),
        float(intrinsics["cy"]),
        (k1, k2, p1, p2),
        convention="opengl",
    )
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    ui = np.rint(pixels[:, 0]).astype(np.int64)
    vi = np.rint(pixels[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(depth)
        & (depth > 1e-4)
        & (ui >= 0)
        & (vi >= 0)
        & (ui < width)
        & (vi < height)
    )
    if int(valid.sum()) < min_pairs:
        raise OrbitDepthInputError(
            "missing_colmap_depths",
            f"missing_colmap_depths: only {int(valid.sum())} projected points in view (need {min_pairs})",
        )
    idx = np.flatnonzero(valid)
    if idx.size > max_samples:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(idx, size=max_samples, replace=False))
    uv = np.column_stack((ui[idx].astype(np.float64), vi[idx].astype(np.float64)))
    return {
        "z": depth[idx].astype(np.float64),
        "uv": uv,
        "source": source,
        "frame_file": str(frame.get("file_path") or ""),
        "intrinsics": intrinsics,
        "transforms_path": str(loaded["path"]),
    }


def _recover_with_predictor(
    obj: str,
    root: Path | None,
    *,
    orbit: int,
    depth_method: str,
    predictor: DepthPredictor,
    image_bgr: np.ndarray | None,
    colmap_depths: Any,
) -> dict[str, Any]:
    if image_bgr is None:
        image_bgr = _load_orbit_frame(obj, orbit, root)
    if image_bgr is None:
        raise OrbitDepthInputError("missing_frames", "missing_frames: no RGB under frames/orbitN")
    samples = _normalize_colmap_samples(obj, orbit, root, colmap_depths)
    intrinsics = samples.get("intrinsics")
    depth_m, meta = predict_metric_depth(
        image_bgr,
        intrinsics if isinstance(intrinsics, dict) else None,
        prefer=depth_method,
        predictor=predictor,
    )
    metric_z = sample_depth_at_pixels(depth_m, np.asarray(samples["uv"], dtype=np.float64))
    return estimate_from_depth_map(
        metric_z,
        np.asarray(samples["z"], dtype=np.float64),
        method=str(meta.get("method") or depth_method),
        extra={"orbit": orbit, "n_depth_pixels": int(np.asarray(samples["z"]).size)},
    )


def _recover_live_depth(
    obj: str,
    root: Path | None,
    *,
    orbit: int,
    depth_method: str,
    image_bgr: np.ndarray | None,
    colmap_depths: Any,
) -> dict[str, Any]:
    if depth_method == "foundation_stereo":
        raise RuntimeError(
            "foundation_stereo is ready but no stereo runner is wired; metres_per_unit stays null"
        )
    # COLMAP/transforms before frames so a bare host stamps missing_transforms honestly.
    if colmap_depths is None:
        load_orbit_intrinsics(obj, orbit, root)
    samples = _normalize_colmap_samples(obj, orbit, root, colmap_depths)
    if image_bgr is None:
        image_bgr = _load_orbit_frame(obj, orbit, root)
    if image_bgr is None:
        raise OrbitDepthInputError("missing_frames", "missing_frames: no RGB under frames/orbitN")
    intrinsics = samples.get("intrinsics")
    if not isinstance(intrinsics, dict):
        # Prefer transforms intrinsics when samples came from injection without them.
        try:
            intrinsics = load_orbit_intrinsics(obj, orbit, root)["intrinsics"]
        except OrbitDepthInputError:
            intrinsics = None
    depth_m, meta = predict_metric_depth(
        image_bgr,
        intrinsics,
        prefer=depth_method,
    )
    metric_z = sample_depth_at_pixels(depth_m, np.asarray(samples["uv"], dtype=np.float64))
    extra: dict[str, Any] = {
        "orbit": orbit,
        "n_depth_pixels": int(np.asarray(samples["z"]).size),
        "backend": str(meta.get("backend") or depth_method),
    }
    if samples.get("source"):
        extra["colmap_source"] = samples["source"]
    if samples.get("transforms_path"):
        extra["transforms_path"] = samples["transforms_path"]
    return estimate_from_depth_map(
        metric_z,
        np.asarray(samples["z"], dtype=np.float64),
        method=str(meta.get("method") or depth_method),
        extra=extra,
    )


def _normalize_colmap_samples(
    obj: str,
    orbit: int,
    root: Path | None,
    colmap_depths: Any,
) -> dict[str, Any]:
    if colmap_depths is None:
        return colmap_depth_samples_for_orbit(obj, orbit, root)
    if isinstance(colmap_depths, dict):
        if colmap_depths.get("uv") is None or colmap_depths.get("z") is None:
            raise ValueError("colmap_depths dict needs 'z' and 'uv' arrays")
        return colmap_depths
    raise ValueError("live/predictor path needs colmap_depths as {'z': ..., 'uv': Nx2} or auto-load")


# DA3METRIC-LARGE FAQ: metric metres = focal_px * net_output / 300
# (focal typically mean of fx, fy). Nested models set is_metric=1 themselves.
DA3_METRIC_FOCAL_REF = 300.0


def coerce_da3_is_metric(value: Any) -> int:
    """Normalize Prediction.is_metric; DA3 may leave an empty addict.Dict instead of 0/1."""
    if value is None:
        return 0
    if isinstance(value, dict):
        # Package bug: getattr(dict_output, "is_metric", 0) returns {} for addict.Dict.
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_da3_metric_checkpoint(model_name: str | None, weights: str | None = None) -> bool:
    """True for DA3METRIC / da3metric-* weights (focal-normalized metric series)."""
    blob = f"{model_name or ''} {weights or ''}".lower().replace("_", "").replace("-", "")
    return "da3metric" in blob


def da3_focal_normalized_to_metres(
    depth: np.ndarray,
    *,
    fx: float,
    fy: float,
    image_width: int,
    image_height: int,
    focal_ref: float = DA3_METRIC_FOCAL_REF,
) -> tuple[np.ndarray, float]:
    """Convert DA3METRIC focal-normalized depth to metres at the depth-map resolution.

    Uses focal lengths scaled from the capture image size to depth.shape, matching
    DA3's input_processor resize of K before the forward pass.
    Returns (depth_m, focal_used_px).
    """
    arr = np.asarray(depth, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("depth must be a non-empty HxW array")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width/height must be positive")
    if not (math.isfinite(fx) and math.isfinite(fy) and fx > 0 and fy > 0):
        raise ValueError("fx/fy must be positive finite focal lengths in pixels")
    if not (math.isfinite(focal_ref) and focal_ref > 0):
        raise ValueError("focal_ref must be positive")
    dh, dw = arr.shape
    sx = dw / float(image_width)
    sy = dh / float(image_height)
    focal = 0.5 * (float(fx) * sx + float(fy) * sy)
    return arr * (focal / float(focal_ref)), float(focal)


def _da3_checkpoint_name(weights: str, model: Any | None = None) -> str:
    name = str(getattr(model, "model_name", "") or "").strip()
    if name:
        return name
    cfg = Path(weights) / "config.json"
    data = _read_json(cfg)
    return str(data.get("model_name") or Path(weights).name or "").strip()


def _predict_depth_anything(
    image_bgr: np.ndarray,
    intrinsics: dict[str, Any] | None,
    *,
    weights: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run DA3 and return metric depth (metres) at input resolution.

    DA3METRIC-LARGE emits focal-normalized depth; official conversion is
    metres = focal_px * depth / 300. Nested metric models may already set
    is_metric=1 / scale_factor. Relative checkpoints stay refused.
    """
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_bgr must be HxWx3")
    rgb = image[:, :, :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    # OpenCV loads BGR; DA3 expects RGB arrays.
    rgb = rgb[:, :, ::-1].copy()
    height, width = rgb.shape[:2]

    try:
        model = _da3_model(weights)
    except ImportError as exc:
        raise RuntimeError(f"depth_anything_3 import failed: {exc}") from exc
    device = next(model.parameters()).device
    model_name = _da3_checkpoint_name(weights, model)
    metric_ckpt = is_da3_metric_checkpoint(model_name, weights)

    fx = fy = cx = cy = None
    if intrinsics is not None:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        # Scale principal point if the caller resized the RGB vs transforms size.
        src_w = float(intrinsics.get("width") or width)
        src_h = float(intrinsics.get("height") or height)
        if src_w > 0 and src_h > 0 and (abs(src_w - width) > 0.5 or abs(src_h - height) > 0.5):
            sx = width / src_w
            sy = height / src_h
            fx, fy, cx, cy = fx * sx, fy * sy, cx * sx, cy * sy

    if metric_ckpt and (fx is None or fy is None):
        raise RuntimeError(
            "DA3METRIC checkpoint needs camera intrinsics (fx/fy) for focal/300 "
            "metric conversion; refusing to invent metres_per_unit"
        )

    infer_kwargs: dict[str, Any] = {}
    if fx is not None and fy is not None and cx is not None and cy is not None:
        k = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        infer_kwargs["intrinsics"] = k[None, ...]

    prediction = model.inference([rgb], **infer_kwargs)
    depth = np.asarray(prediction.depth[0], dtype=np.float64)
    if depth.ndim != 2:
        raise RuntimeError("DA3 depth must be HxW")

    is_metric = coerce_da3_is_metric(getattr(prediction, "is_metric", 0))
    scale_factor = getattr(prediction, "scale_factor", None)
    try:
        scale_ok = scale_factor is not None and math.isfinite(float(scale_factor)) and float(scale_factor) > 0
    except (TypeError, ValueError):
        scale_ok = False

    meta: dict[str, Any] = {
        "method": "depth_anything",
        "status": "measured",
        "backend": "depth_anything_3",
        "weights": weights,
        "model_name": model_name,
        "is_metric": is_metric,
        "device": str(device),
    }

    if is_metric == 1 or scale_ok:
        # Nested / already-metric path: depth is already metres (or scaled in-model).
        if depth.shape != (height, width):
            depth = _resize_depth(depth, width, height)
        if scale_ok:
            meta["scale_factor"] = float(scale_factor)
        meta["metric_path"] = "prediction_is_metric" if is_metric == 1 else "scale_factor"
        return depth, meta

    if metric_ckpt:
        # Official DA3METRIC FAQ: metres = focal * net_output / 300 at processed K.
        depth_m, focal_used = da3_focal_normalized_to_metres(
            depth,
            fx=float(fx),
            fy=float(fy),
            image_width=width,
            image_height=height,
        )
        if depth_m.shape != (height, width):
            depth_m = _resize_depth(depth_m, width, height)
        meta["metric_path"] = "focal_normalized_over_300"
        meta["focal_px"] = focal_used
        meta["focal_ref"] = DA3_METRIC_FOCAL_REF
        meta["is_metric"] = 1
        return depth_m, meta

    raise RuntimeError(
        "DA3 prediction is not metric (is_metric!=1) and checkpoint is not "
        "DA3METRIC; refusing to invent metres_per_unit"
    )


def _da3_model(weights: str) -> Any:
    cached = _DA3_MODEL_CACHE.get(weights)
    if cached is not None:
        return cached
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model.eval()
    _DA3_MODEL_CACHE[weights] = model
    return model


def _resize_depth(depth: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv is required to resize DA3 depth to the capture frame") from exc
    resized = cv2.resize(depth.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(resized, dtype=np.float64)


def _intrinsics_from_transforms(data: dict[str, Any]) -> dict[str, Any] | None:
    frames = data.get("frames")
    frame0 = frames[0] if isinstance(frames, list) and frames and isinstance(frames[0], dict) else {}
    mapped = {
        "fx": data.get("fx", data.get("fl_x", frame0.get("fl_x", frame0.get("fx")))),
        "fy": data.get("fy", data.get("fl_y", frame0.get("fl_y", frame0.get("fy", data.get("fl_x"))))),
        "cx": data.get("cx", frame0.get("cx")),
        "cy": data.get("cy", frame0.get("cy")),
        "width": data.get("width", data.get("w", frame0.get("w", frame0.get("width")))),
        "height": data.get("height", data.get("h", frame0.get("h", frame0.get("height")))),
    }
    try:
        values = {key: float(mapped[key]) for key in ("fx", "fy", "cx", "cy", "width", "height")}
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values.values()):
        return None
    if values["width"] <= 1 or values["height"] <= 1:
        return None
    if values["fx"] <= 0 or values["fy"] <= 0:
        return None
    return values


def _load_orbit_points(
    obj: str,
    orbit: int,
    root: Path | None,
    colmap_orbit_dir: Path,
) -> tuple[np.ndarray | None, str]:
    sparse_txt = colmap_orbit_dir / "sparse" / "0" / "points3D.txt"
    if sparse_txt.is_file():
        xyz = _read_points3d_txt(sparse_txt)
        if xyz is not None and xyz.size:
            return xyz, f"sparse:{sparse_txt}"
    for splat in (
        colmap_orbit_dir / "splat.ply",
        object_dir(obj, root) / "colmap" / f"orbit{orbit}" / "splat.ply",
    ):
        if splat.is_file():
            try:
                from tools.colmap_lift import read_splat_columns

                xyz, _, _ = read_splat_columns(splat)
            except (OSError, ValueError, ImportError):
                xyz = None
            if xyz is not None and xyz.size:
                return np.asarray(xyz, dtype=np.float64), f"splat:{splat}"
    return None, ""


def _read_points3d_txt(path: Path) -> np.ndarray | None:
    points: list[list[float]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
    if not points:
        return None
    return np.asarray(points, dtype=np.float64)


def _score_only_unmeasured(obj: str, root: Path | None, record: dict[str, Any]) -> None:
    gate_class = classify("metres_per_unit_missing", detail=str(record.get("reason") or ""))
    assert gate_class == SCORE_ONLY
    record_ledger(
        obj,
        root,
        gate="metres_per_unit",
        gate_class=SCORE_ONLY,
        detail=str(record.get("reason") or "scale not_measured"),
        check="not_run" if record.get("status") in (
            "not_run",
            "missing_weights",
            "missing_package",
            "missing_frames",
            "missing_transforms",
            "missing_intrinsics",
            "missing_colmap_depths",
        ) else "unmet",
        reason="metres_per_unit_missing",
    )


def _load_orbit_frame(obj: str, orbit: int, root: Path | None) -> np.ndarray | None:
    try:
        import cv2
    except ImportError:
        return None
    folder = frames_dir(obj, orbit, root)
    candidates: list[Path] = []
    if folder.is_dir():
        candidates.extend(sorted(folder.iterdir()))
    # Nerfstudio layout: colmap/orbitN/images/frame_*.png next to transforms.json.
    images_dir = object_dir(obj, root) / "colmap" / f"orbit{orbit}" / "images"
    if images_dir.is_dir():
        candidates.extend(sorted(images_dir.iterdir()))
    for path in candidates:
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            return image
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write ingest/scale.json from metric depth, Polycam, ruler, or tag — never from a VLM guess",
    )
    parser.add_argument("--obj", required=True)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--orbit", type=int, default=1)
    parser.add_argument(
        "--method",
        default=None,
        help="depth_anything | foundation_stereo | polycam | ruler | tag36h11 (model_knowledge refused)",
    )
    parser.add_argument("--length-m", type=float, default=None, help="Ruler length in metres")
    parser.add_argument("--length-units", type=float, default=None, help="Same length in reconstruction units")
    parser.add_argument("--metres-per-unit", type=float, default=None, help="Explicit measured scale (polycam/tag)")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing measured scale.json")
    parser.add_argument("--probe", action="store_true", help="Print depth backend status and exit")
    args = parser.parse_args(argv)

    if args.probe:
        status = depth_backend_status(args.method or "depth_anything")
        sys.stdout.write(json.dumps(status, indent=2) + "\n")
        return 0 if status.get("available") else skill_exit(SCORE_ONLY)

    try:
        record = recover_metric_scale(
            args.obj,
            args.root,
            orbit=args.orbit,
            method=args.method,
            length_m=args.length_m,
            length_units=args.length_units,
            metres_per_unit=args.metres_per_unit,
            force=args.force,
        )
    except ScaleRefuseError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2

    sys.stdout.write(json.dumps(record, indent=2) + "\n")
    if is_measured_record(record):
        return 0
    # Unmeasured is ScoreOnly so Block H ingest continues.
    return skill_exit(SCORE_ONLY)


if __name__ == "__main__":
    raise SystemExit(main())
