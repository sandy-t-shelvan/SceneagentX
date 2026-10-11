"""Block 2 / B+C: crop the object, copy the review PNG, and do not pass a skipped infill."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.block_gates import (
    BOTTLE_INSIDE,
    BOTTLE_OUTSIDE,
    BOTTLE_SPAN_INSIDE,
    BOTTLE_SPAN_SCENE,
    infill_state,
    judge_projection,
    prepare_review_image,
)
from tools.common.vlm import MissingReviewImage
from tools.iterate import iterate_skill


def test_projection_covering_room_rejected():
    assert judge_projection(20.0, 22.0) == "rejected"


def test_projection_on_object_kept():
    assert judge_projection(BOTTLE_SPAN_INSIDE, BOTTLE_SPAN_SCENE) == "kept"
    assert BOTTLE_INSIDE == 408236
    assert BOTTLE_OUTSIDE == 546578


def test_review_png_copied_before_vlm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    source = tmp_path / "mask.png"
    source.write_bytes(b"\x89PNG\r\n\x1a\nreview")
    dest = tmp_path / "reviews" / "see.png"
    seen: dict[str, Path] = {}

    def fake_review(step_id, images, question, allowed, record_path=None):
        del step_id, question, allowed, record_path
        seen["image"] = Path(images[0])
        assert seen["image"].is_file()
        assert seen["image"].read_bytes() == source.read_bytes()
        return {
            "verdict": "pass",
            "confidence": 0.9,
            "problem": "none",
            "correction": {"parameter": None, "value": None, "reason": ""},
            "evidence": "copied",
        }

    monkeypatch.setattr("tools.common.vlm.review", fake_review)
    copied = prepare_review_image(source, dest)
    from tools.common.vlm import review

    review("see", [copied], "The overlay should cover the object.", ["box_threshold"])
    assert seen["image"] == dest
    assert dest.is_file()


def test_missing_png_records_asked_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("OpenAI was called")

    monkeypatch.setattr("tools.common.vlm._http_post", boom)
    masks = tmp_path / "data" / "bottle" / "masks"
    masks.mkdir(parents=True)
    (masks / "object.json").write_text(json.dumps({"projected": True}), encoding="utf-8")
    cropped = tmp_path / "data" / "bottle" / "objects" / "bottle" / "object.ply"
    cropped.parent.mkdir(parents=True)
    cropped.write_bytes(b"ply\nformat ascii 1.0\nend_header\n")
    code = iterate_skill("bottle", "see", tmp_path, vlm_mode="full")
    assert code == 2
    runs = (tmp_path / "data" / "bottle" / "runs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    vision = json.loads(runs[-1])["vision_gate"]
    assert vision["asked"] is False
    assert vision.get("verdict") != "pass"
    with pytest.raises(MissingReviewImage):
        prepare_review_image(tmp_path / "missing.png", tmp_path / "out.png")


def test_infill_skipped_is_not_pass():
    assert infill_state({"infill": "skipped"}) == "skipped"
    assert infill_state({"infill": "skipped"}) != "pass"
