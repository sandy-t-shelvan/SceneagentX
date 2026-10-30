"""Can B4 clustering. One lifted mask is not itself an instance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.can_instances import (
    REVIEW_PARAMS,
    apply_hand_score,
    bounding_boxes,
    cluster_gaussians,
    review_instance_views,
    write_instance_record,
)
from tools.common.vlm import instance_review_question


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
    named = cluster_gaussians(inside, features, lang, votes=np.full(600, 5), object_name="vaseline")
    assert named["instances"][0]["name"] == "vaseline"
    assert len(named["instances"]) == 1
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
    result = review_instance_views([], object_name="bottle")
    assert result["asked"] is False
    assert result["verdict"] is None


def test_review_missing_file_is_asked_false(tmp_path: Path):
    result = review_instance_views([tmp_path / "missing.png"], object_name="lamp")
    assert result["asked"] is False
    assert result["verdict"] is None


def test_instance_review_question_names_the_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bottle = instance_review_question("bottle")
    lamp = instance_review_question("lamp")
    assert bottle == (
        "The bottle sits in a room with other surfaces and objects "
        "(table, floor, carpet, walls, nearby items). "
        "The instance under review must contain only the bottle. "
        "Tabletop, carpet, floor, and background fragments attached to it "
        "are extra_part, not part of the object. "
        "These images are four views of one clustered instance. "
        "Is the instance complete, bleeding into neighbours, or wrongly labelled?"
    )
    assert lamp == bottle.replace("bottle", "lamp")
    assert " can " not in f" {bottle} "
    assert " can " not in f" {lamp} "
    can = instance_review_question("can")
    assert "only the can" in can
    assert "only the bottle" not in can
    seen: dict[str, object] = {}

    def fake_review(step_id, images, question, allowed, record_path=None):
        seen["step_id"] = step_id
        seen["question"] = question
        seen["allowed"] = tuple(allowed)
        seen["record_path"] = record_path
        return {
            "verdict": "fail",
            "confidence": 0.9,
            "problem": "extra_part",
            "correction": {"parameter": "split", "value": None, "reason": "floor"},
            "evidence": "carpet",
        }

    image = tmp_path / "view.png"
    image.write_bytes(b"png")
    record = tmp_path / "b4.json"
    monkeypatch.setattr("tools.common.vlm.review", fake_review)
    result = review_instance_views([image], record, object_name="bottle")
    assert result["asked"] is True
    assert result["problem"] == "extra_part"
    assert seen["step_id"] == "B4"
    assert seen["question"] == bottle
    assert seen["allowed"] == REVIEW_PARAMS
    assert seen["record_path"] == record


def test_extra_part_fail_blocks_block_c(tmp_path: Path):
    record = write_instance_record(
        tmp_path,
        orbit="orbit2",
        clustered={"instances": [{"name": "can", "n_gaussians": 32366}], "meta": {"n_object": 32366}},
        mask_lift={"n_gaussians": 149863, "n_inside": 32366, "n_outside": 117497},
        review={
            "asked": True,
            "verdict": "fail",
            "problem": "extra_part",
            "correction": {"parameter": "split"},
        },
    )
    assert record["block_c_ready"] is False
    assert record["achieved"] is False
    assert "extra_part" in record["block_c_reason"]
    assert "Block C" in record["block_c_reason"]


def test_bounding_boxes_use_means():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    boxes = bounding_boxes(points)
    assert boxes["aabb"][0] == [0.0, 0.0, 0.0]
    assert boxes["aabb"][1] == [1.0, 2.0, 3.0]
    assert boxes["obb"]["extents"] is not None
