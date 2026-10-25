"""Cluster can Gaussians and write Block B instance records.

Uses mask-id votes plus B3 features. A single lifted mask is not itself
an instance. Hand labels are not scored here. mIoU stays null and M2
stays unmet. This module does not set m2_met or achieved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.common.paths import object_dir, repo_root

SILHOUETTE_KEEP = 0.35
MIN_SIZE_FRAC = 0.05
MIN_SIZE_ABS = 200
MAX_K = 4
REVIEW_PARAMS = ("merge", "split", "prompt")
ORBITS = ("orbit1", "orbit2", "orbit3")


def capture_for(folder: Path) -> dict[str, Any]:
    """Capture paths for this object folder. A new video is not recorded as the can."""
    name = folder.name
    orbits = []
    for index in (1, 2, 3):
        video = folder / "raw" / f"orbit{index}.MOV"
        orbits.append(
            {
                "path": f"data/{name}/raw/orbit{index}.MOV",
                "exists": video.is_file(),
                "existing_capture": video.is_file(),
            }
        )
    return {
        "used_for_eval": f"data/{name}/raw/orbit1.MOV",
        "ns_train": "not_run",
        "colmap": "not_rerun",
        "orbits": orbits,
        "existing_capture": any(row["exists"] for row in orbits),
    }


def flatten_features(features: np.ndarray) -> np.ndarray:
    """Turn a (N, 3, 3) B3 field or an (N, D) matrix into (N, D)."""
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 3:
        return arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError(f"features shape {arr.shape} is not (N, D) or (N, 3, 3)")
    return arr


def cluster_gaussians(
    inside: np.ndarray,
    features: np.ndarray,
    lang_idx: np.ndarray,
    means: np.ndarray | None = None,
    votes: np.ndarray | None = None,
    *,
    object_name: str = "can",
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    """Cluster inside-mask Gaussians by mask-id votes and B3 features.

    The boolean lift mask is a candidate set, not one instance. k is chosen
    from a sampled silhouette. Tiny leftover clusters stay unassigned.
    """
    mask = np.asarray(inside, dtype=bool).reshape(-1)
    feat = flatten_features(features)
    codes = np.asarray(lang_idx).reshape(-1)
    n = int(mask.shape[0])
    if feat.shape[0] != n or codes.shape[0] != n:
        raise ValueError("inside, features, and lang_idx must share the Gaussian axis")
    vote = np.zeros(n, dtype=np.float32) if votes is None else np.asarray(votes, dtype=np.float32).reshape(-1)
    if vote.shape[0] != n:
        raise ValueError("votes must match the Gaussian count")
    xyz = None if means is None else np.asarray(means, dtype=np.float32).reshape(-1, 3)
    if xyz is not None and xyz.shape[0] != n:
        raise ValueError("means must match the Gaussian count")
    rng = np.random.default_rng(0) if rng is None else rng

    candidates = np.flatnonzero(mask)
    assignment = np.full(n, -1, dtype=np.int32)
    meta = {
        "method": "mask_id_votes + B3 feature k-means",
        "one_mask_is_not_one_instance": True,
        "n_gaussians": n,
        "n_inside": int(mask.sum()),
        "n_candidates": int(candidates.size),
        "chosen_k": 0,
        "silhouettes": {},
        "dropped_small": [],
    }
    if candidates.size == 0:
        return {"assignment": assignment, "instances": [], "meta": meta}

    packed = _pack_points(feat[candidates], vote[candidates], None if xyz is None else xyz[candidates])
    labels, silhouettes, chosen_k = _choose_k(packed, rng)
    meta["silhouettes"] = silhouettes
    meta["chosen_k"] = int(chosen_k)

    min_size = max(MIN_SIZE_ABS, int(MIN_SIZE_FRAC * candidates.size))
    kept: list[tuple[int, np.ndarray]] = []
    for cluster_id in range(chosen_k):
        members = candidates[labels == cluster_id]
        if members.size < min_size:
            meta["dropped_small"].append({"cluster": int(cluster_id), "n": int(members.size)})
            continue
        kept.append((int(members.size), members))
    kept.sort(key=lambda row: (-row[0], int(row[1][0]) if row[1].size else 0))

    instances: list[dict[str, Any]] = []
    for inst_i, (_size, members) in enumerate(kept):
        assignment[members] = inst_i
        instances.append(_instance_record(inst_i, members, codes, vote, xyz, object_name=object_name))
    meta["n_instances"] = len(instances)
    meta["n_object"] = int((assignment >= 0).sum())
    meta["n_unassigned"] = int(n - meta["n_object"])
    return {"assignment": assignment, "instances": instances, "meta": meta}


def apply_hand_score(record: dict[str, Any], score: dict[str, Any]) -> dict[str, Any]:
    """Copy the scorer's numbers. Missing labels keep miou null."""
    updated = dict(record)
    n_drawn = int(score.get("hand_labels") or 0)
    miou = score.get("miou")
    if n_drawn < 20 or miou is None:
        miou = None
    updated["miou"] = miou
    updated["hand_labels"] = n_drawn
    updated["m2_met"] = False
    updated["achieved"] = False
    updated["hand_score_reason"] = score.get("reason")
    updated["hand_score_skipped"] = list(score.get("skipped") or [])
    return updated


def review_instance_views(
    images: list[str | Path],
    record_path: str | Path | None = None,
    *,
    object_name: str,
) -> dict[str, Any]:
    """Ask the VLM once. The question names object_name. Missing pictures stay asked false."""
    from tools.common.vlm import MissingReviewImage, VlmError, instance_review_question, review

    if not images:
        return {"asked": False, "reason": "no instance review pictures", "verdict": None}
    question = instance_review_question(object_name)
    try:
        verdict = review("B4", images, question, REVIEW_PARAMS, record_path=record_path)
    except MissingReviewImage as exc:
        return {"asked": False, "reason": str(exc), "verdict": None}
    except (VlmError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {"asked": False, "reason": str(exc), "verdict": None}
    return {
        "asked": True,
        "reason": "vlm reviewed 4 views of the clustered instance",
        "verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "problem": verdict.get("problem"),
        "correction": verdict.get("correction"),
        "evidence": verdict.get("evidence"),
    }


def write_instance_record(
    folder: Path,
    *,
    orbit: str | None,
    clustered: dict[str, Any],
    mask_lift: dict[str, Any],
    review: dict[str, Any] | None = None,
    hand_score: dict[str, Any] | None = None,
    review_views: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one instances.json. Hand-label scores are not copied in. M2 stays unmet."""
    folder = Path(folder)
    instances = list(clustered.get("instances") or [])
    meta = clustered.get("meta") if isinstance(clustered.get("meta"), dict) else {}
    n_gauss = int(mask_lift.get("n_gaussians") or 0)
    n_inside = int(mask_lift.get("n_inside") or 0)
    n_object = int(meta.get("n_object") if meta.get("n_object") is not None else sum(int(row.get("n_gaussians") or 0) for row in instances))
    n_unassigned = int(n_gauss - n_object) if n_gauss else int(meta.get("n_unassigned") or 0)
    review = review or {"asked": False, "reason": "VLM was not asked", "verdict": None}
    asked = review.get("asked") is True
    kind = "vlm" if asked else "none"
    record: dict[str, Any] = {
        "block": "B",
        "capture": capture_for(folder),
        "instances": instances,
        "background": {
            "foreground": False,
            "n_gaussians": n_unassigned,
            "codebook_index": None,
            "bbox_aabb": None,
            "bbox_obb": None,
            "feature": "scene_mean_not_a_separate_measurement",
            "scene_mean_measured": False,
            "reason": "one mixed object vector is not a separate scene-mean measurement",
        },
        "codebook": "langsplat gaussian_lang_idx" if orbit in (None, "orbit1") else f"data/{folder.name}/codebook_{orbit}.json",
        "langsplat_codebook": orbit in (None, "orbit1"),
        "miou": None,
        "hand_labels": 0,
        "instance_count": len(instances),
        "instance_count_note": f"{len(instances)} vs one {folder.name} in the capture; instance count is not a substitute for mIoU",
        "learned_prior": False,
        "n_object": n_object,
        "n_unassigned": n_unassigned,
        "mask_lift_n_inside": n_inside,
        "mask_lift_n_outside": int(mask_lift.get("n_outside") or 0),
        "cluster": meta,
        "review_kind": kind,
        "review_views": list(review_views or []),
        "review": review,
        "m2_met": False,
        "achieved": False,
        "block_c_ready": False,
    }
    if review.get("verdict") == "fail" or review.get("problem") == "extra_part":
        record["block_c_reason"] = (
            "B4 review failed"
            + (f" with {review.get('problem')}" if review.get("problem") else "")
            + "; Block C stays unstarted until the instance is split and the review passes"
        )
        # #region agent log
        try:
            import time
            line = json.dumps({
                "sessionId": "9f1777",
                "runId": "extra-part",
                "hypothesisId": "floor-in-cluster",
                "location": "can_instances.py:write_instance_record",
                "message": "extra_part fail blocks Block C",
                "data": {
                    "orbit": orbit,
                    "problem": review.get("problem"),
                    "verdict": review.get("verdict"),
                },
                "timestamp": int(time.time() * 1000),
            })
            with (Path(__file__).resolve().parents[1] / "debug-9f1777.log").open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        # #endregion
    if orbit is not None:
        record["orbit"] = orbit
    if extra:
        record.update(extra)
    # hand_score is accepted so older callers still run. It is not copied.
    _ = hand_score
    dest = folder / "semantics" / "instances.json" if orbit is None else folder / "semantics" / orbit / "instances.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def write_orbit_membership(
    folder: Path,
    orbit: str,
    *,
    mask_lift: dict[str, Any],
    codebook_index: int | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Record an orbit that has no B3 feature field. Do not claim a cluster."""
    n_inside = int(mask_lift.get("n_inside") or 0)
    n_outside = int(mask_lift.get("n_outside") or 0)
    clustered = {
        "instances": [
            {
                "name": folder.name,
                "foreground": True,
                "n_gaussians": n_inside,
                "codebook_index": codebook_index,
                "bbox_aabb": None,
                "bbox_obb": None,
                "detector": "grounding_dino_sam2",
                "projected": True,
                "clustered": False,
            }
        ],
        "meta": {
            "method": "mask-lift membership only; B3 feature field is orbit1-only",
            "one_mask_is_not_one_instance": True,
            "clustered": False,
            "n_gaussians": int(mask_lift.get("n_gaussians") or 0),
            "n_inside": n_inside,
            "n_object": n_inside,
            "n_unassigned": n_outside,
            "reason": "no B3 features on this orbit; clustering was not claimed",
        },
    }
    return write_instance_record(
        folder,
        orbit=orbit,
        clustered=clustered,
        mask_lift=mask_lift,
        extra=extra,
    )


def _instance_record(
    inst_i: int,
    members: np.ndarray,
    codes: np.ndarray,
    vote: np.ndarray,
    xyz: np.ndarray | None,
    object_name: str = "can",
) -> dict[str, Any]:
    code = None
    if members.size:
        clipped = np.clip(codes[members].astype(np.int64), 0, None)
        code = int(np.bincount(clipped).argmax())
    bbox = bounding_boxes(xyz[members]) if xyz is not None and members.size else {"aabb": None, "obb": None}
    return {
        "name": object_name if inst_i == 0 else f"{object_name}_{inst_i + 1}",
        "foreground": True,
        "n_gaussians": int(members.size),
        "codebook_index": code,
        "bbox_aabb": bbox["aabb"],
        "bbox_obb": bbox["obb"],
        "mask_vote_mean": float(vote[members].mean()) if members.size else 0.0,
        "detector": "grounding_dino_sam2",
        "projected": True,
        "clustered": True,
    }


def bounding_boxes(points: np.ndarray) -> dict[str, Any]:
    """Axis-aligned box plus a PCA-oriented box. Empty input is null."""
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if cloud.shape[0] == 0:
        return {"aabb": None, "obb": None}
    low = cloud.min(axis=0)
    high = cloud.max(axis=0)
    aabb = [low.tolist(), high.tolist()]
    centered = cloud - cloud.mean(axis=0)
    if cloud.shape[0] < 3:
        return {"aabb": aabb, "obb": None}
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    local = centered @ axes
    return {
        "aabb": aabb,
        "obb": {
            "center": cloud.mean(axis=0).tolist(),
            "axes": axes.T.tolist(),
            "extents": (local.max(axis=0) - local.min(axis=0)).tolist(),
        },
    }


def _pack_points(feat: np.ndarray, vote: np.ndarray, xyz: np.ndarray | None) -> np.ndarray:
    parts = [_zscore(feat)]
    if vote.max() > 0:
        parts.append(_zscore(vote.reshape(-1, 1)) * 0.5)
    if xyz is not None:
        parts.append(_zscore(xyz) * 0.5)
    return np.concatenate(parts, axis=1).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-6] = 1.0
    return (arr - mean) / std


def _choose_k(points: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[str, float | None], int]:
    labels_one = np.zeros(points.shape[0], dtype=np.int32)
    silhouettes: dict[str, float | None] = {"1": None}
    best_k = 1
    best_labels = labels_one
    best_sil = -1.0
    for k in range(2, MAX_K + 1):
        if points.shape[0] < k * MIN_SIZE_ABS:
            silhouettes[str(k)] = None
            continue
        labels = _kmeans(points, k, rng)
        sil = _silhouette_sample(points, labels, rng)
        silhouettes[str(k)] = sil
        if sil is not None and sil > SILHOUETTE_KEEP and sil > best_sil:
            best_k = k
            best_labels = labels
            best_sil = sil
    return best_labels, silhouettes, best_k


def _kmeans(points: np.ndarray, k: int, rng: np.random.Generator, steps: int = 25) -> np.ndarray:
    n = points.shape[0]
    centers = np.empty((k, points.shape[1]), dtype=np.float32)
    centers[0] = points[int(rng.integers(0, n))]
    closest = np.full(n, np.inf, dtype=np.float64)
    for index in range(1, k):
        dist = ((points - centers[index - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, dist)
        weight = closest.copy()
        total = float(weight.sum())
        if total <= 0:
            centers[index] = points[int(rng.integers(0, n))]
            continue
        pick = float(rng.random()) * total
        centers[index] = points[int(np.searchsorted(np.cumsum(weight), pick).clip(0, n - 1))]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(steps):
        dist = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = dist.argmin(axis=1).astype(np.int32)
        for cluster_id in range(k):
            members = points[labels == cluster_id]
            if members.size:
                centers[cluster_id] = members.mean(axis=0)
    return labels


def _silhouette_sample(points: np.ndarray, labels: np.ndarray, rng: np.random.Generator, max_n: int = 1500) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    if points.shape[0] > max_n:
        pick = rng.choice(points.shape[0], size=max_n, replace=False)
        points = points[pick]
        labels = labels[pick]
    if len(np.unique(labels)) < 2:
        return None
    dist = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2) + 1e-12)
    scores: list[float] = []
    for index, label in enumerate(labels):
        same = labels == label
        other = ~same
        if int(same.sum()) <= 1 or not bool(other.any()):
            continue
        a = float(dist[index, same].sum() / (int(same.sum()) - 1))
        b = min(float(dist[index, labels == other_id].mean()) for other_id in np.unique(labels) if other_id != label)
        denom = max(a, b)
        if denom <= 0:
            continue
        scores.append((b - a) / denom)
    if not scores:
        return None
    return float(np.mean(scores))


def _review_images(folder: Path) -> list[Path]:
    review_dir = folder / "semantics" / "reviews"
    paired: list[Path] = []
    for index in range(4):
        render = review_dir / f"instance_0_view_{index}.png"
        source = review_dir / f"instance_0_source_{index}.png"
        if render.is_file():
            paired.append(render)
        if source.is_file():
            paired.append(source)
    if paired:
        return paired
    return sorted(path for path in review_dir.glob("instance_0_view_*.png") if path.is_file())[:4]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optionally VLM-review B4 instances. Hand labels are not scored.")
    parser.add_argument("--obj", default="can")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args(argv)
    folder = object_dir(args.obj, args.root)
    review_record: dict[str, Any] | None = None
    if args.review:
        images = _review_images(folder)
        dest = folder / "reviews" / "b4_instance.json"
        review_record = review_instance_views(images, dest, object_name=args.obj)
        for path in (
            folder / "semantics" / "instances.json",
            folder / "semantics" / "orbit1" / "instances.json",
        ):
            if not path.is_file():
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                continue
            record["review"] = review_record
            record["review_kind"] = "vlm" if review_record.get("asked") is True else "none"
            record["review_views"] = [str(path.relative_to(repo_root())).replace("\\", "/") if path.is_absolute() else str(path) for path in images]
            path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"review": review_record}, indent=2))
    print(json.dumps({"ok": True, "m2_met": False, "achieved": False, "hand_labels": "not_scored"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
