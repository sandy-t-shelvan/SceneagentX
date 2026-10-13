"""Can B4 clustering. One lifted mask is not itself an instance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.can_instances import (
    apply_hand_score,
    bounding_boxes,
    cluster_gaussians,
    review_instance_views,
    write_instance_record,
)


def test_two_feature_groups_are_not_one_instance():
    inside = np.ones(800, dtype=bool)
    features = np.zeros((800, 3, 3), dtype=np.float32)
    features[:400] = 0.0
    features[400:] = 10.0
    lang = np.zeros(800, dtype=np.int32)
    lang[400:] = 7
    votes = np.full(800, 4, dtype=np.int32)
    result = cluster_gaussians(inside, features, lang, votes=votes)
    assert result["meta"]["one_mask_is_not_one_instance"] is True
    assert result["meta"]["n_inside"] == 800
    assert len(result["instances"]) == 2
    assert {row["codebook_index"] for row in result["instances"]} == {0, 7}
    assert all(row["foreground"] is True for row in result["instances"])


def test_one_feature_group_stays_one_instance():
    inside = np.ones(600, dtype=bool)
    features = np.zeros((600, 3, 3), dtype=np.float32)
    features += np.random.default_rng(0).normal(0, 0.01, features.shape).astype(np.float32)
    lang = np.full(600, 3, dtype=np.int32)
    result = cluster_gaussians(inside, features, lang, votes=np.full(600, 5))
    assert len(result["instances"]) == 1
    assert result["instances"][0]["name"] == "can"
    assert result["instances"][0]["codebook_index"] == 3
    assert result["instances"][0]["n_gaussians"] == 600


def test_write_keeps_miou_null_and_hand_labels_zero(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    clustered = cluster_gaussians(
        np.ones(400, dtype=bool),
        np.zeros((400, 9), dtype=np.float32),
        np.zeros(400, dtype=np.int32),
        means=np.stack([np.linspace(0, 1, 400), np.zeros(400), np.zeros(400)], axis=1),
    )
    record = write_instance_record(
        folder,
        orbit=None,
        clustered=clustered,
        mask_lift={"n_gaussians": 400, "n_inside": 400, "n_outside": 0},
        hand_score={"miou": None, "hand_labels": 0, "reason": "20 hand labels missing", "skipped": ["20 hand labels missing"]},
    )
    assert record["miou"] is None
    assert record["hand_labels"] == 0
    assert record["m2_met"] is False
    assert record["achieved"] is False
    assert record["instances"][0]["bbox_aabb"] is not None
    assert (folder / "semantics" / "instances.json").is_file()


def test_silhouette_iou_is_not_copied_as_miou():
    record = apply_hand_score(
        {"miou": None, "hand_labels": 0, "achieved": False, "m2_met": False},
        {"miou": None, "hand_labels": 0, "reason": "missing", "skipped": ["missing"]},
    )
    assert record["miou"] is None
    assert record["hand_labels"] == 0
    bumped = apply_hand_score(record, {"miou": 0.4278428819444444, "hand_labels": 0, "reason": "invented"})
    assert bumped["miou"] is None
    assert bumped["hand_labels"] == 0
    assert bumped["m2_met"] is False
    assert bumped["achieved"] is False


def test_review_without_images_is_asked_false():
    result = review_instance_views([])
    assert result["asked"] is False
    assert result["verdict"] is None


def test_review_missing_file_is_asked_false(tmp_path: Path):
    result = review_instance_views([tmp_path / "missing.png"])
    assert result["asked"] is False
    assert result["verdict"] is None


def test_bounding_boxes_use_means():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    boxes = bounding_boxes(points)
    assert boxes["aabb"][0] == [0.0, 0.0, 0.0]
    assert boxes["aabb"][1] == [1.0, 2.0, 3.0]
    assert boxes["obb"]["extents"] is not None
