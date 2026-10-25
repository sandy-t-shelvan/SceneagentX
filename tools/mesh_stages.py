"""SceneAgent stages for a Polycam mesh that has no photographs.

COLMAP and splatfacto are not used. Masks come from the synthetic orbit
silhouettes. Physics numbers come from a local prior table. When
OPENAI_API_KEY is set, one synthetic view is sent to a vision model and
the numbers are stored with source "vlm". A missing key or a failed call
keeps the prior table. Object sisters are procedural scale and color
changes. Isaac Lab, GroundingDINO, PartField, and TRELLIS are recorded as
not run when those programs are absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64
import json
import os
import urllib.error
import urllib.request
import zipfile
from typing import Any

import cv2
import numpy as np
import trimesh

from tools.common.paths import object_dir, renders_dir
from tools.common.status import begin_step, finish_step, get_path, load_status, save_status, set_path
from tools.mesh_to_3dgs import SH_C0, _is_mesh, load_geometry

MESH_STAGE_ORDER = (
    "B5",
    "B6",
    "B7",
    "B9",
    "B10",
    "B11",
    "B4",
    "B8",
    "B12",
    "B13",
    "B14",
)
STAGE_NEEDS = {
    "B4": ("B4",),
    "B5": ("B5", "B6"),
    "B6": ("B5", "B6"),
    "B7": ("B5", "B6", "B7"),
    "B8": ("B4", "B8"),
    "B9": ("B5", "B6", "B9"),
    "B10": ("B5", "B6", "B7", "B10"),
    "B11": ("B5", "B6", "B9", "B10", "B11"),
    "B12": ("B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12"),
    "B13": ("B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13"),
    "B14": ("B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14"),
}
GPU_ONLY = {"B15", "B16", "B17"}
_VISION_MODEL = "gpt-6-astra"
_VISION_URL = "https://api.openai.com/v1/chat/completions"


def run_mesh_through(obj: str, through: str, root: Path | None = None) -> int:
    """Run the mesh stages required to finish ``through``. Returns 0 on success."""
    from tools.agent_loop import finish_route, run_skill

    needed = set(STAGE_NEEDS[through])
    saw_model = False
    for block_id in MESH_STAGE_ORDER:
        if block_id not in needed:
            continue
        code, called = run_skill(
            obj,
            block_id,
            lambda parameter, block_id=block_id: _dispatch(obj, block_id, root),
            root,
        )
        saw_model = saw_model or called
        if code != 0:
            finish_route(obj, root, saw_model)
            sys.stderr.write(f"Stopped at {block_id} after a second fault.\n")
            return code
        sys.stdout.write(f"{block_id} pass  polycam_mesh\n")
    finish_route(obj, root, saw_model)
    return 0


def _refresh_changelog(obj: str, root: Path | None) -> None:
    from tools.changelog import write_changelog

    path = write_changelog(obj, root)
    sys.stdout.write(f"changelog  {path}\n")


def _dispatch(obj: str, block_id: str, root: Path | None) -> None:
    if block_id in {"B5", "B6"}:
        if block_id == "B5":
            _masks(obj, root)
        else:
            _lift(obj, root)
    elif block_id == "B7":
        _codebook(obj, root)
    elif block_id == "B9":
        _parts(obj, root)
    elif block_id == "B10":
        _physics(obj, root)
    elif block_id == "B11":
        _urdf(obj, root)
    elif block_id == "B4":
        _align(obj, root)
    elif block_id == "B8":
        _chamfer(obj, root)
    elif block_id == "B12":
        _package(obj, root)
    elif block_id == "B13":
        _object_sisters(obj, root)
    elif block_id == "B14":
        _scene_sisters(obj, root)
    else:
        raise KeyError(block_id)


def read_gaussian_table(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    header, _, payload = raw.partition(b"end_header\n")
    count = 0
    for line in header.decode("ascii").splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    table = np.frombuffer(payload, dtype="<f4", count=count * 17).reshape(count, 17)
    return np.array(table, dtype=np.float64, copy=True)


def _splat_file(obj: str, root: Path | None) -> Path:
    doc = load_status(obj, root)
    return Path(get_path(doc, "step02.orbit1.splat"))


def _mesh_file(obj: str, root: Path | None) -> Path:
    doc = load_status(obj, root)
    return Path(get_path(doc, "step02.orbit1.mesh"))


def _masks(obj: str, root: Path | None) -> None:
    begin_step(obj, "B5", root)
    view_dir = renders_dir(obj, "synthetic", root)
    mask_dir = object_dir(obj, root) / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for view in sorted(view_dir.glob("view_*.png")):
        image = cv2.imread(str(view), cv2.IMREAD_COLOR)
        if image is None:
            continue
        background = np.array([244, 244, 244], dtype=np.int16)
        foreground = np.abs(image.astype(np.int16) - background).max(axis=2) > 8
        mask = np.where(foreground, 255, 0).astype(np.uint8)
        destination = mask_dir / f"{view.stem}_mask.png"
        cv2.imwrite(str(destination), mask)
        written.append(destination.name)
    table = read_gaussian_table(_splat_file(obj, root))
    indices = np.arange(len(table), dtype=np.int64)
    index_path = mask_dir / "gaussian_indices.txt"
    index_path.write_text("\n".join(str(int(i)) for i in indices) + "\n", encoding="utf-8")
    record = {
        "label": obj,
        "background": "none",
        "views": "synthetic",
        "segmenter": "silhouette",
        "grounding_dino": False,
        "sam": False,
        "n_gaussians": int(len(indices)),
        "masks": written,
        "indices": str(index_path),
    }
    parameter = _restart_parameter(obj, "B5", root)
    if "prompt" in parameter:
        record["prompt"] = parameter["prompt"]
    if "box_threshold" in parameter:
        record["box_threshold"] = parameter["box_threshold"]
    (mask_dir / "object.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B5", "pass", root, actual=record)


def _restart_parameter(obj: str, block_id: str, root: Path | None) -> dict[str, Any]:
    path = object_dir(obj, root) / "agent_parameter.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or loaded.get("block") != block_id:
        return {}
    return loaded


def _lift(obj: str, root: Path | None) -> None:
    begin_step(obj, "B6", root)
    actual = project_masks(obj, root)
    finish_step(obj, "B6", "pass", root, actual=actual)


def project_masks(obj: str, root: Path | None = None) -> dict[str, Any]:
    """Project each view mask onto the surface Gaussians.

    Background infill stays skipped. This scan has no separate room to fill.
    """
    from tools.ingest import _project

    mask_dir = object_dir(obj, root) / "masks"
    record_path = mask_dir / "object.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    xyz = read_gaussian_table(_splat_file(obj, root))[:, :3]
    geometry = load_geometry(_mesh_file(obj, root))
    view_dir = renders_dir(obj, "synthetic", root)
    frames = _camera_frames(view_dir)
    inside = np.zeros(len(xyz), dtype=bool)
    used = 0
    for frame in frames:
        mask_path = mask_dir / f"{Path(frame['file']).stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        height, width = mask.shape[:2]
        eye, center, radius = _view_camera(geometry, float(frame["azimuth_deg"]), float(frame["elevation_deg"]))
        pixels, depth = _project(xyz, eye, center, width, height)
        xs = np.rint(pixels[:, 0]).astype(np.int32)
        ys = np.rint(pixels[:, 1]).astype(np.int32)
        valid = (
            (depth > radius * 0.05)
            & (xs >= 0)
            & (ys >= 0)
            & (xs < width)
            & (ys < height)
        )
        hit = np.zeros(len(xyz), dtype=bool)
        if np.any(valid):
            hit[valid] = mask[ys[valid], xs[valid]] > 127
        inside |= hit
        used += 1
    crop_path = mask_dir / "gaussian_in_mask.npy"
    np.save(crop_path, inside)
    record["projected"] = True
    record["n_views_projected"] = used
    record["n_inside"] = int(inside.sum())
    record["n_outside"] = int((~inside).sum())
    record["crop"] = str(crop_path)
    record["infill"] = "skipped"
    record["infill_reason"] = "no separate background to fill on this scan"
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {
        "lifted": int(inside.sum()),
        "n_outside": int((~inside).sum()),
        "background": "none",
        "method": "project view masks onto gaussians",
        "infill": "skipped",
        "projected": True,
    }


def _camera_frames(view_dir: Path) -> list[dict[str, Any]]:
    path = view_dir / "cameras.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames")
        if isinstance(frames, list) and frames:
            return frames
    views = sorted(view_dir.glob("view_*.png"))
    count = max(len(views), 1)
    return [
        {"file": path.name, "azimuth_deg": 360.0 * index / count, "elevation_deg": 20.0}
        for index, path in enumerate(views)
    ]


def _view_camera(geometry: Any, azimuth_deg: float, elevation_deg: float) -> tuple[np.ndarray, np.ndarray, float]:
    from tools.ingest import _orbit_offset

    if _is_mesh(geometry):
        points = np.asarray(geometry.vertices, dtype=np.float64)
    else:
        points = np.asarray(geometry.vertices, dtype=np.float64)
        if len(points) > 8000:
            points = points[np.random.default_rng(0).choice(len(points), size=8000, replace=False)]
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.linalg.norm(points - center, axis=1).max()), 1e-4)
    eye = center + _orbit_offset(azimuth_deg, elevation_deg, distance=2.6 * radius)
    return eye, center, radius


def _codebook(obj: str, root: Path | None) -> None:
    begin_step(obj, "B7", root)
    table = read_gaussian_table(_splat_file(obj, root))
    rgb = np.clip(table[:, 6:9] * SH_C0 + 0.5, 0.0, 1.0).mean(axis=0)
    extent = np.ptp(table[:, :3], axis=0)
    longest = max(float(extent.max()), 1e-8)
    vector = np.concatenate((rgb, extent / longest, [1.0, 0.0])).astype(float)
    record = {
        "encoder": "geometric",
        "dinov2": False,
        "openclip": False,
        "label": obj,
        "vector": [round(float(value), 6) for value in vector],
        "n_gaussians": int(len(table)),
    }
    path = object_dir(obj, root) / "codebook.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B7", "pass", root, actual={"codebook": str(path), "encoder": "geometric"})


def _physics(obj: str, root: Path | None) -> None:
    begin_step(obj, "B10", root)
    mesh = load_geometry(_mesh_file(obj, root))
    volume = abs(float(mesh.volume)) if _is_volume(mesh) else 0.0
    estimate = _vlm_physics(obj, root)
    if estimate is None:
        prior = _prior_for(obj)
        source = "prior_table"
        material = prior["material"]
        friction = float(prior["friction"])
        rigidity = float(prior["rigidity"])
        density = float(prior["density_kg_m3"])
        mass = density * volume if volume > 0 else float(prior["mass_kg"])
        used_vlm = False
    else:
        source = "vlm"
        material = estimate["material"]
        friction = estimate["friction"]
        rigidity = estimate["rigidity"]
        density = estimate["density_kg_m3"]
        mass = estimate["mass_kg"]
        used_vlm = True
    record = {
        "label": obj,
        "source": source,
        "vlm": used_vlm,
        "material": material,
        "friction": round(friction, 6),
        "rigidity": round(rigidity, 6),
        "mass_kg": round(mass, 6),
        "density_kg_m3": round(density, 6),
        "per_gaussian": {
            "copied_from": "object",
            "n": int(read_gaussian_table(_splat_file(obj, root)).shape[0]),
            "learned_prior": False,
            "variation": "geometry",
            "dinov2": False,
            "openclip": False,
        },
        "gaps": _physics_gaps(),
    }
    if used_vlm:
        record["model"] = _VISION_MODEL
    from tools.orbit_blocks import extend_physics_record

    extend_physics_record(record)
    path = object_dir(obj, root) / "physics.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    _write_gaussian_variation(obj, root, friction, rigidity)
    actual = {
        "physics": str(path),
        "source": source,
        "per_gaussian": "copied_from_object",
        "learned_prior": False,
    }
    if used_vlm:
        actual["model"] = _VISION_MODEL
    finish_step(obj, "B10", "pass", root, actual=actual)


def _physics_gaps() -> dict[str, str]:
    return {
        "learned_prior": "not released; learned_prior stays false",
        "heterogeneous_openusd_rigidity": "not stored; one object rigidity plus a color and normal shift",
        "vlm_review_of_later_steps": "not run",
    }


def annotate_physics_gaps(obj: str, root: Path | None = None) -> Path:
    """Record the missing prior without calling the vision model again."""
    path = object_dir(obj, root) / "physics.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    per = dict(record.get("per_gaussian") or {})
    per["learned_prior"] = False
    record["per_gaussian"] = per
    record["gaps"] = _physics_gaps()
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def _write_gaussian_variation(obj: str, root: Path | None, friction: float, rigidity: float) -> None:
    """Shift the object values by color and normal. This is not a learned prior."""
    table = read_gaussian_table(_splat_file(obj, root))
    rgb = np.clip(table[:, 6:9] * SH_C0 + 0.5, 0.0, 1.0)
    brightness = rgb.mean(axis=1)
    centered = brightness - float(brightness.mean())
    scale = float(np.max(np.abs(centered)))
    color_delta = centered / scale * 0.05 if scale > 1e-8 else np.zeros(len(table))
    rigidity_i = np.clip(rigidity + color_delta, 0.0, 1.0)
    friction_i = np.clip(friction + 0.02 * table[:, 4], 0.05, 1.0)
    destination = object_dir(obj, root) / "physics_gaussians.npz"
    np.savez_compressed(
        destination,
        rigidity=rigidity_i.astype(np.float32),
        friction=friction_i.astype(np.float32),
    )
    summary = object_dir(obj, root) / "physics_gaussians.json"
    summary.write_text(
        json.dumps(
            {
                "source": "vlm_object_plus_geometry",
                "learned_prior": False,
                "dinov2": False,
                "openclip": False,
                "n": int(len(table)),
                "rigidity_min": round(float(rigidity_i.min()), 6),
                "rigidity_max": round(float(rigidity_i.max()), 6),
                "friction_min": round(float(friction_i.min()), 6),
                "friction_max": round(float(friction_i.max()), 6),
                "file": str(destination),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _vlm_physics(obj: str, root: Path | None) -> dict[str, Any] | None:
    """One synthetic view, or None so the caller keeps the prior table."""
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write("B10 vision call skipped: OPENAI_API_KEY is unset\n")
        return None
    view = renders_dir(obj, "synthetic", root) / "view_000.png"
    if not view.is_file():
        sys.stderr.write("B10 vision call skipped: synthetic view_000.png is missing\n")
        return None
    image = view.read_bytes()
    try:
        try:
            body = _post_vision(image, api_key, json_mode=True)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                raise
            sys.stderr.write("B10 vision call retried without json mode: HTTP 400\n")
            body = _post_vision(image, api_key, json_mode=False)
        return _physics_from_completion(body)
    except urllib.error.HTTPError as exc:
        sys.stderr.write(f"B10 vision call failed: HTTP {exc.code}\n")
        return None
    except urllib.error.URLError:
        sys.stderr.write("B10 vision call failed: network error\n")
        return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        sys.stderr.write("B10 vision response was not usable\n")
        return None


def _post_vision(image: bytes, api_key: str, json_mode: bool = True) -> dict[str, Any]:
    encoded = base64.b64encode(image).decode("ascii")
    payload: dict[str, Any] = {
        "model": _VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Estimate rigid-body physics for the object in this image. "
                            "Reply with one JSON object and no other text. "
                            "Keys: material (string), friction (number >= 0), "
                            "rigidity (number from 0 to 1), mass_kg (number > 0), "
                            "density_kg_m3 (number > 0)."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded}"},
                    },
                ],
            }
        ],
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        _VISION_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _physics_from_completion(body: dict[str, Any]) -> dict[str, Any]:
    parsed = _json_object(_message_text(body))
    material = str(parsed["material"]).strip()
    friction = float(parsed["friction"])
    rigidity = float(parsed["rigidity"])
    mass = float(parsed["mass_kg"])
    density = float(parsed["density_kg_m3"])
    if not material or any(char in material for char in "\r\n"):
        raise ValueError("material must be a single line")
    if len(material) > 80:
        raise ValueError("material is too long")
    if not _finite(friction) or friction < 0:
        raise ValueError("friction must be finite and >= 0")
    if not _finite(rigidity) or not 0.0 <= rigidity <= 1.0:
        raise ValueError("rigidity must be from 0 to 1")
    if not _finite(mass) or mass <= 0 or not _finite(density) or density <= 0:
        raise ValueError("mass and density must be finite and > 0")
    return {
        "material": material,
        "friction": friction,
        "rigidity": rigidity,
        "mass_kg": mass,
        "density_kg_m3": density,
    }


def _message_text(body: dict[str, Any]) -> str:
    content = body["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError("vision message content is missing")
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict):
            parts.append(str(part.get("text", "")))
    return "".join(parts)


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if "```" in cleaned:
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise TypeError("vision JSON must be an object")
    return parsed


def _finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def _parts(obj: str, root: Path | None) -> None:
    begin_step(obj, "B9", root)
    existing = object_dir(obj, root) / "parts.json"
    if existing.is_file():
        record = json.loads(existing.read_text(encoding="utf-8"))
        if record.get("source") == "vlm":
            joints = sum(1 for part in record.get("parts") or [] if isinstance(part, dict) and part.get("joint"))
            finish_step(obj, "B9", "pass", root, actual={"parts": str(existing), "n_parts": len(record.get("parts") or []), "n_joints": joints})
            return
    mesh = load_geometry(_mesh_file(obj, root))
    record = {
        "segmenter": "single_mesh",
        "partfield": False,
        "parts": [
            {
                "id": "body",
                "parent": None,
                "joint": None,
                "vertex_count": int(len(mesh.vertices)),
            }
        ],
    }
    path = object_dir(obj, root) / "parts.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B9", "pass", root, actual={"parts": str(path), "n_parts": 1, "n_joints": 0})


def _urdf(obj: str, root: Path | None) -> None:
    begin_step(obj, "B11", root)
    parts_path = object_dir(obj, root) / "parts.json"
    urdf_path = object_dir(obj, root) / "asset.urdf"
    if parts_path.is_file() and urdf_path.is_file():
        record = json.loads(parts_path.read_text(encoding="utf-8"))
        if record.get("source") == "vlm":
            joints = urdf_path.read_text(encoding="utf-8").lower().count("<joint")
            finish_step(obj, "B11", "pass", root, actual={"urdf": str(urdf_path), "n_joints": joints, "source": "vlm"})
            return
    physics = json.loads((object_dir(obj, root) / "physics.json").read_text(encoding="utf-8"))
    mesh_name = _mesh_file(obj, root).name
    urdf = (
        f'<?xml version="1.0"?>\n'
        f'<robot name="{_ident(obj)}">\n'
        f'  <link name="body">\n'
        f'    <inertial>\n'
        f'      <mass value="{physics["mass_kg"]}"/>\n'
        f'      <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>\n'
        f"    </inertial>\n"
        f"    <visual>\n"
        f"      <geometry><mesh filename=\"{mesh_name}\"/></geometry>\n"
        f"    </visual>\n"
        f"    <collision>\n"
        f"      <geometry><mesh filename=\"{mesh_name}\"/></geometry>\n"
        f"    </collision>\n"
        f"  </link>\n"
        f"</robot>\n"
    )
    path = object_dir(obj, root) / "asset.urdf"
    path.write_text(urdf, encoding="utf-8")
    finish_step(obj, "B11", "pass", root, actual={"urdf": str(path), "n_joints": 0})


def _align(obj: str, root: Path | None) -> None:
    begin_step(obj, "B4", root)
    geometry = load_geometry(_mesh_file(obj, root))
    points = np.asarray(geometry.vertices, dtype=np.float64)
    longest = float(np.ptp(points, axis=0).max())
    up = _up_axis(geometry)
    rotation = _rotation_to_z(up)
    record = {
        "photometric": "skipped",
        "photometric_reason": "no capture images",
        "silhouette": "skipped",
        "edge": "skipped",
        "placement": "identity",
        "scale_m": longest,
        "up_axis": [round(float(value), 6) for value in up],
        "rotation_to_z": [[round(float(value), 6) for value in row] for row in rotation],
    }
    path = object_dir(obj, root) / "align.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    orient = {
        "skipped": True,
        "reason": "no separate background",
        "up_from": "mesh",
    }
    orient_path = object_dir(obj, root) / "orientation.json"
    orient_path.write_text(json.dumps(orient, indent=2) + "\n", encoding="utf-8")
    doc = load_status(obj, root)
    set_path(doc, "align.report", record)
    set_path(doc, "orientation", orient)
    save_status(obj, doc, root)
    finish_step(obj, "B4", "pass", root, actual={"align": str(path), "photometric": "skipped"})


def _chamfer(obj: str, root: Path | None) -> None:
    begin_step(obj, "B8", root)
    geometry = load_geometry(_mesh_file(obj, root))
    xyz = read_gaussian_table(_splat_file(obj, root))[:, :3]
    if _is_mesh(geometry):
        distances_cm = _distances_cm(geometry, xyz)
    else:
        distances_cm = _distances_to_cloud(np.asarray(geometry.vertices, dtype=np.float64), xyz)
    record = {
        "source_mesh": str(_mesh_file(obj, root)),
        "two_dgs": False,
        "chamfer_cm": round(float(distances_cm.mean()), 6),
        "n_points": int(len(xyz)),
    }
    finish_step(obj, "B8", "pass", root, actual=record)


def _package(obj: str, root: Path | None) -> None:
    begin_step(obj, "B12", root)
    mesh = _surface(load_geometry(_mesh_file(obj, root)))
    physics = json.loads((object_dir(obj, root) / "physics.json").read_text(encoding="utf-8"))
    usdz = object_dir(obj, root) / "scene.usdz"
    _write_usdz(usdz, _ident(obj), mesh, physics, translation=(0.0, 0.0, 0.0))
    preview_dir = object_dir(obj, root) / "inspect"
    preview_dir.mkdir(parents=True, exist_ok=True)
    views = sorted(renders_dir(obj, "synthetic", root).glob("view_*.png"))
    preview = preview_dir / "preview.png"
    if views:
        preview.write_bytes(views[0].read_bytes())
    inspect = {
        "isaac": False,
        "passed": False,
        "reason": "Isaac Lab is not installed",
        "preview": str(preview) if views else "",
    }
    doc = load_status(obj, root)
    set_path(doc, "inspect", inspect)
    save_status(obj, doc, root)
    chamfer = next(step for step in doc["steps"] if step["id"] == "B8")["actual"]["chamfer_cm"]
    finish_step(
        obj,
        "B12",
        "pass",
        root,
        actual={"usdz": str(usdz), "chamfer_cm": chamfer, "isaac": False},
    )


def _object_sisters(obj: str, root: Path | None) -> None:
    begin_step(obj, "B13", root)
    mesh = _surface(load_geometry(_mesh_file(obj, root)))
    physics = json.loads((object_dir(obj, root) / "physics.json").read_text(encoding="utf-8"))
    folder = object_dir(obj, root) / "sisters" / "objects"
    folder.mkdir(parents=True, exist_ok=True)
    changes = []
    for index, (scale, color_delta) in enumerate(((0.92, -0.08), (1.0, 0.0), (1.08, 0.08)), start=1):
        sister = _vary_mesh(mesh, scale, color_delta)
        path = folder / f"sister_{index:02d}.usdz"
        _write_usdz(path, f"{_ident(obj)}_sister_{index:02d}", sister, physics, translation=(0.0, 0.0, 0.0))
        changes.append({"file": path.name, "scale": scale, "color_delta": color_delta, "generator": "procedural"})
    manifest = {"generator": "procedural", "trellis": False, "objects": changes}
    manifest_path = object_dir(obj, root) / "sisters" / "objects.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B13", "pass", root, actual={"manifest": str(manifest_path), "n": len(changes), "trellis": False})


def adopt_trellis_sisters(obj: str, glbs: list[Path], root: Path | None = None, n_views: int = 8) -> Path:
    """Replace procedural sisters with TRELLIS meshes and rebuild the layouts."""
    if len(glbs) < 1:
        raise ValueError("TRELLIS did not return a mesh")
    physics = json.loads((object_dir(obj, root) / "physics.json").read_text(encoding="utf-8"))
    folder = object_dir(obj, root) / "sisters" / "objects"
    folder.mkdir(parents=True, exist_ok=True)
    loaded: list[trimesh.Trimesh] = []
    changes = []
    for index, glb in enumerate(glbs, start=1):
        mesh = _as_mesh(trimesh.load(glb, force="mesh"))
        loaded.append(mesh)
        path = folder / f"sister_{index:02d}.usdz"
        _write_usdz(path, f"{_ident(obj)}_sister_{index:02d}", mesh, physics, translation=(0.0, 0.0, 0.0))
        changes.append(
            {
                "file": path.name,
                "generator": "trellis",
                "source": Path(glb).name,
                "conditioned_on_views": n_views,
            }
        )
    manifest = {
        "generator": "trellis",
        "trellis": True,
        "conditioned_on": "synthetic views",
        "n_views": n_views,
        "objects": changes,
    }
    manifest_path = object_dir(obj, root) / "sisters" / "objects.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    scene_dir = object_dir(obj, root) / "sisters" / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    offsets = [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (-0.1, 0.12, 0.0)]
    written = []
    for index, mesh in enumerate(loaded[:3]):
        name = f"layout_{index + 1:02d}"
        path = scene_dir / f"{name}.usdz"
        _write_usdz(path, _ident(obj), mesh, physics, translation=offsets[index])
        written.append({"file": path.name, "translation_m": list(offsets[index]), "generator": "trellis"})
    scenes_path = object_dir(obj, root) / "sisters" / "scenes.json"
    scenes_path.write_text(json.dumps({"scenes": written, "trellis": True}, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B13", "pass", root, actual={"manifest": str(manifest_path), "n": len(changes), "trellis": True})
    finish_step(obj, "B14", "pass", root, actual={"manifest": str(scenes_path), "n": len(written), "trellis": True})
    return manifest_path


def _as_mesh(loaded: Any) -> trimesh.Trimesh:
    if isinstance(loaded, trimesh.Trimesh):
        return loaded
    if isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump(concatenate=True)
        if isinstance(dumped, trimesh.Trimesh):
            return dumped
    raise TypeError("TRELLIS output is not a mesh")


def _scene_sisters(obj: str, root: Path | None) -> None:
    begin_step(obj, "B14", root)
    mesh = _surface(load_geometry(_mesh_file(obj, root)))
    physics = json.loads((object_dir(obj, root) / "physics.json").read_text(encoding="utf-8"))
    folder = object_dir(obj, root) / "sisters" / "scenes"
    folder.mkdir(parents=True, exist_ok=True)
    layouts = [
        {"name": "layout_01", "translation": (0.0, 0.0, 0.0)},
        {"name": "layout_02", "translation": (0.15, 0.0, 0.0)},
        {"name": "layout_03", "translation": (-0.1, 0.12, 0.0)},
    ]
    written = []
    for layout in layouts:
        path = folder / f"{layout['name']}.usdz"
        _write_usdz(path, _ident(obj), mesh, physics, translation=layout["translation"])
        written.append({"file": path.name, "translation_m": list(layout["translation"])})
    manifest = {"scenes": written}
    manifest_path = object_dir(obj, root) / "sisters" / "scenes.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    finish_step(obj, "B14", "pass", root, actual={"manifest": str(manifest_path), "n": len(written)})


def _distances_cm(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Mean is not required here. One distance per point, in centimetres."""
    best = np.full(len(points), np.inf)
    for triangle in np.asarray(mesh.triangles, dtype=np.float64):
        best = np.minimum(best, _point_triangle(points, triangle))
    return best * 100.0


def _point_triangle(points: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    corner_a, corner_b, corner_c = triangle
    ab = corner_b - corner_a
    ac = corner_c - corner_a
    ap = points - corner_a
    d00 = float(np.dot(ab, ab))
    d01 = float(np.dot(ab, ac))
    d11 = float(np.dot(ac, ac))
    denom = d00 * d11 - d01 * d01
    if denom <= 1e-12:
        return np.linalg.norm(ap, axis=1)
    d20 = ap @ ab
    d21 = ap @ ac
    beta = (d11 * d20 - d01 * d21) / denom
    gamma = (d00 * d21 - d01 * d20) / denom
    alpha = 1.0 - beta - gamma
    weights = np.clip(np.column_stack((alpha, beta, gamma)), 0.0, None)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
    closest = weights[:, 0:1] * corner_a + weights[:, 1:2] * corner_b + weights[:, 2:3] * corner_c
    return np.linalg.norm(points - closest, axis=1)


def _prior_for(name: str) -> dict[str, Any]:
    lowered = name.lower()
    if any(word in lowered for word in ("speaker", "audio", "headphone")):
        return {"material": "plastic", "friction": 0.45, "rigidity": 0.9, "density_kg_m3": 700.0, "mass_kg": 0.3}
    return {"material": "unknown", "friction": 0.5, "rigidity": 0.8, "density_kg_m3": 500.0, "mass_kg": 0.2}


def _is_volume(mesh: trimesh.Trimesh) -> bool:
    return bool(getattr(mesh, "is_volume", False))


def _surface(geometry: Any) -> trimesh.Trimesh:
    """Faces for USDZ. A Polycam point cloud becomes a convex hull of a sample."""
    if _is_mesh(geometry):
        return geometry
    points = np.asarray(geometry.vertices, dtype=np.float64)
    if len(points) > 8000:
        points = points[np.random.default_rng(0).choice(len(points), size=8000, replace=False)]
    return trimesh.convex.convex_hull(points)


def _distances_to_cloud(reference: np.ndarray, points: np.ndarray) -> np.ndarray:
    if len(reference) > 4000:
        reference = reference[np.random.default_rng(0).choice(len(reference), size=4000, replace=False)]
    best = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), 4096):
        batch = points[start : start + 4096]
        best[start : start + len(batch)] = np.linalg.norm(batch[:, None, :] - reference[None, :, :], axis=2).min(axis=1)
    return best * 100.0


def _up_axis(mesh: Any) -> np.ndarray:
    if not _is_mesh(mesh):
        extent = np.ptp(np.asarray(mesh.vertices, dtype=np.float64), axis=0)
        axis = int(np.argmin(extent))
        up = np.zeros(3, dtype=np.float64)
        up[axis] = 1.0
        return up
    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    up = normals[int(np.argmax(areas))]
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    if up[2] < 0:
        up = -up
    return up


def _rotation_to_z(up: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(up, target)
    dot = float(np.dot(up, target))
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    if dot > 0.999999:
        return np.eye(3)
    if dot < -0.999999:
        return np.diag([1.0, -1.0, -1.0])
    return np.eye(3) + skew + (skew @ skew) * (1.0 / (1.0 + dot))


def _vary_mesh(mesh: trimesh.Trimesh, scale: float, color_delta: float) -> trimesh.Trimesh:
    sister = mesh.copy()
    center = sister.vertices.mean(axis=0)
    sister.vertices = (sister.vertices - center) * scale + center
    colors = getattr(getattr(sister, "visual", None), "vertex_colors", None)
    if colors is not None and len(colors) == len(sister.vertices):
        rgb = np.clip(colors[:, :3].astype(np.int16) + int(color_delta * 255), 0, 255).astype(np.uint8)
        alpha = colors[:, 3:4] if colors.shape[1] == 4 else np.full((len(colors), 1), 255, dtype=np.uint8)
        sister.visual.vertex_colors = np.concatenate((rgb, alpha), axis=1)
    return sister


def _write_usdz(
    path: Path,
    name: str,
    mesh: trimesh.Trimesh,
    physics: dict[str, Any],
    translation: tuple[float, float, float],
) -> None:
    usda = _usda(name, mesh, physics, translation)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("scene.usda")
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, usda)


def _usda(
    name: str,
    mesh: trimesh.Trimesh,
    physics: dict[str, Any],
    translation: tuple[float, float, float],
) -> str:
    points = ", ".join(f"({point[0]:.6g}, {point[1]:.6g}, {point[2]:.6g})" for point in np.asarray(mesh.vertices))
    counts = ", ".join("3" for _ in mesh.faces)
    indices = ", ".join(str(int(index)) for index in np.asarray(mesh.faces).reshape(-1))
    tx, ty, tz = translation
    return (
        "#usda 1.0\n"
        f'def Xform "{name}"\n'
        "{\n"
        f'    custom string physics:material = "{physics["material"]}"\n'
        f'    custom float physics:friction = {physics["friction"]}\n'
        f'    custom float physics:rigidity = {physics["rigidity"]}\n'
        f'    custom float physics:mass = {physics["mass_kg"]}\n'
        f'    custom float physics:density = {physics["density_kg_m3"]}\n'
        "    custom bool physics:learned_prior = false\n"
        "    custom bool physics:heterogeneous_rigidity = false\n"
        '    custom string physics:vlm_review = "not_run"\n'
        f"{_usd_physics_notes(physics)}"
        f"    double3 xformOp:translate = ({tx:.6g}, {ty:.6g}, {tz:.6g})\n"
        '    uniform token[] xformOpOrder = ["xformOp:translate"]\n'
        '    def Mesh "Body"\n'
        "    {\n"
        f"        point3f[] points = [{points}]\n"
        f"        int[] faceVertexCounts = [{counts}]\n"
        f"        int[] faceVertexIndices = [{indices}]\n"
        "    }\n"
        "}\n"
    )


def _usd_physics_notes(physics: dict[str, Any]) -> str:
    """Bake stored friction numbers. learned_prior stays false in the caller."""
    lines = ['    custom string physics:dropTest = "not_run"']
    static = physics.get("static_friction")
    if isinstance(static, (int, float)) and not isinstance(static, bool):
        lines.insert(0, f"    custom float physics:staticFriction = {static}")
    dynamic = physics.get("dynamic_friction")
    if isinstance(dynamic, (int, float)) and not isinstance(dynamic, bool):
        lines.append(f"    custom float physics:dynamicFriction = {dynamic}")
    restitution = physics.get("restitution")
    if isinstance(restitution, (int, float)) and not isinstance(restitution, bool):
        lines.append(f"    custom float physics:restitution = {restitution}")
    kind = physics.get("deformable_type")
    if isinstance(kind, str) and kind in {"rigid", "cloth", "rope", "soft"}:
        lines.append(f'    custom string physics:deformableType = "{kind}"')
    return "\n".join(lines) + "\n"


def _ident(name: str) -> str:
    cleaned = "".join(char if char.isalnum() or char == "_" else "_" for char in name)
    if not cleaned or cleaned[0].isdigit():
        return f"object_{cleaned}"
    return cleaned
