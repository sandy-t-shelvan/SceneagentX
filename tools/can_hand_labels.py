"""Compare rendered instance masks to can hand labels.

A missing or placeholder label is a fail. It is never a pass and never
writes mIoU. SAM auto-masks are a separate record and do not count as
hand labels. This module does not set m2_met or achieved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.block_gates import mask_iou
from tools.common.paths import object_dir, repo_root

REQUIRED_VIEWS = 20
PLACEHOLDER_MARK = b"PLACEHOLDER"
MASK_SUFFIXES = (".png", ".npy")


def score_object(folder: Path) -> dict[str, Any]:
    """Score ``data/<object>/hand_labels`` against rendered instance masks."""
    folder = Path(folder)
    labels = folder / "hand_labels"
    record = score_hand_labels(labels, folder / "hand_labels" / "renders")
    dest = labels / "score.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def score_hand_labels(labels: Path, rendered: Path | None = None) -> dict[str, Any]:
    """Walk the 20-view set. Missing labels keep ``miou`` null and refuse a pass."""
    labels = Path(labels)
    manifest = _load(labels / "manifest.json")
    sam_auto = _load(labels / "sam_auto.json")
    views = [row for row in (manifest.get("views") or []) if isinstance(row, dict)]
    skipped: list[str] = []
    mismatch: list[str] = []
    drawn: list[dict[str, Any]] = []
    empty_slots: list[str] = []
    sam_copies: list[str] = []
    if len(views) != REQUIRED_VIEWS:
        skipped.append(f"hand_labels manifest must list {REQUIRED_VIEWS} views, found {len(views)}")
    if not views:
        skipped.append("hand_labels/manifest.json missing or empty")
    sam_bytes = _sam_auto_bytes(sam_auto, labels)
    for view in views:
        view_id = str(view.get("id") or "")
        mask_path = _mask_path(labels, view)
        if not _is_drawn_mask(mask_path):
            empty_slots.append(view_id or str(mask_path))
            continue
        raw = mask_path.read_bytes()
        if raw in sam_bytes or _looks_like_sam_copy(view, sam_auto):
            sam_copies.append(view_id)
            mismatch.append(f"{view_id} is a SAM auto-mask, not a hand label")
            continue
        drawn.append({"id": view_id, "mask": _rel(mask_path)})
    n_drawn = len(drawn)
    if empty_slots:
        skipped.append(f"{len(empty_slots)} hand labels missing")
    if sam_copies:
        skipped.append(f"{len(sam_copies)} SAM auto-masks were copied into hand-label slots")
    ious: list[float] = []
    missing_renders: list[str] = []
    if n_drawn == REQUIRED_VIEWS and not sam_copies:
        render_root = Path(rendered) if rendered is not None else labels / "renders"
        for view in views:
            hand = _load_mask(_mask_path(labels, view))
            render_path = _render_path(render_root, view)
            if hand is None or not render_path.is_file():
                missing_renders.append(str(view.get("id") or render_path))
                continue
            rendered_mask = _load_mask(render_path)
            if rendered_mask is None:
                missing_renders.append(str(view.get("id") or render_path))
                continue
            ious.append(mask_iou(hand, rendered_mask))
        if missing_renders:
            skipped.append("rendered instance masks missing")
    elif n_drawn == REQUIRED_VIEWS:
        skipped.append("SAM auto-masks cannot be scored as hand labels")
    miou = float(sum(ious) / len(ious)) if ious and not missing_renders and n_drawn == REQUIRED_VIEWS else None
    if miou is not None and _silhouette_collision(labels.parent, miou):
        mismatch.append("silhouette IoU is not mIoU")
        miou = None
    ok = not skipped and not mismatch and miou is not None
    record = {
        "object": manifest.get("object") or labels.parent.name,
        "ok": ok,
        "passed": False,
        "achieved": False,
        "m2_met": False,
        "n_views": len(views),
        "hand_labels": n_drawn,
        "drawn": n_drawn,
        "missing": empty_slots,
        "sam_copies": sam_copies,
        "miou": miou,
        "skipped": skipped,
        "mismatch": mismatch,
        "reason": _reason(n_drawn, skipped, mismatch, miou),
        "sam_auto": "data/can/hand_labels/sam_auto.json",
        "orbit1_reprompt": manifest.get("orbit1_reprompt") or "skipped",
    }
    return record


def _is_drawn_mask(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if path.name.endswith(".placeholder") or path.suffix == ".slot":
        return False
    raw = path.read_bytes()
    if raw.startswith(PLACEHOLDER_MARK) or raw.strip() == PLACEHOLDER_MARK:
        return False
    if path.suffix.lower() not in MASK_SUFFIXES:
        return False
    return _load_mask(path) is not None


def _load_mask(path: Path) -> np.ndarray | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    if path.suffix.lower() == ".npy":
        loaded = np.load(path)
        return np.asarray(loaded) > 0
    try:
        image = Image.open(path)
    except OSError:
        return None
    array = np.asarray(image.convert("L"))
    if array.size == 0:
        return None
    return array > 0


def _mask_path(labels: Path, view: dict[str, Any]) -> Path:
    listed = view.get("mask")
    if isinstance(listed, str) and listed:
        path = Path(listed)
        if path.is_absolute():
            return path
        if listed.startswith("data/"):
            return _resolve_data_path(listed, labels)
        return labels / path
    view_id = str(view.get("id") or "unknown")
    return labels / "masks" / f"{view_id}.png"


def _render_path(rendered: Path, view: dict[str, Any]) -> Path:
    listed = view.get("render")
    if isinstance(listed, str) and listed:
        path = Path(listed)
        return path if path.is_absolute() else rendered / path.name
    return rendered / f"{view.get('id')}.png"


def _sam_auto_bytes(sam_auto: dict[str, Any], labels: Path) -> set[bytes]:
    blobs: set[bytes] = set()
    for row in sam_auto.get("views") or []:
        if not isinstance(row, dict):
            continue
        for key in ("local_mask", "mask"):
            value = row.get(key)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = _resolve_data_path(value, labels) if value.startswith("data/") else labels.parent / value
            if path.is_file() and path.stat().st_size:
                blobs.add(path.read_bytes())
    return blobs


def _looks_like_sam_copy(view: dict[str, Any], sam_auto: dict[str, Any]) -> bool:
    if view.get("kind") == "sam_auto" or view.get("drawn") is False:
        return False
    for row in sam_auto.get("views") or []:
        if isinstance(row, dict) and row.get("id") == view.get("id") and row.get("copied_as_hand_label") is True:
            return True
    return False


def _resolve_data_path(value: str, labels: Path) -> Path:
    planted = labels.parents[2] / value if len(labels.parents) >= 3 else None
    if planted is not None and (planted.is_file() or planted.parent.is_dir()):
        return planted
    return repo_root() / value


def _silhouette_collision(folder: Path, miou: float) -> bool:
    mesh = _load(folder / "mesh_score.json")
    for row in mesh.get("orbits") or []:
        if isinstance(row, dict) and isinstance(row.get("iou"), (int, float)):
            if abs(float(row["iou"]) - miou) <= 1e-9:
                return True
    if isinstance(mesh.get("iou"), (int, float)) and abs(float(mesh["iou"]) - miou) <= 1e-9:
        return True
    return False


def _reason(n_drawn: int, skipped: list[str], mismatch: list[str], miou: float | None) -> str:
    if n_drawn == 0:
        return "placeholder slots are not drawn masks; SAM auto-masks are a separate record; miou stays null"
    if skipped or mismatch or miou is None:
        return "; ".join(skipped + mismatch) or "mIoU was not measured"
    return "mIoU measured against drawn hand labels; m2_met stays false until the Block B gate says so"


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _rel(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description="Score can hand labels against rendered instance masks")
    parser.add_argument("--obj", default="can")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    record = score_object(object_dir(args.obj, args.root))
    print(json.dumps({k: record[k] for k in ("ok", "hand_labels", "miou", "m2_met", "achieved", "reason")}, indent=2))
    return 0 if record["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
