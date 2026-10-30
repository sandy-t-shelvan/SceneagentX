"""Block 1 / A: keep the bottle COLMAP and splat numbers. Do not retrain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.block_gates import (
    BOTTLE_GAUSSIANS,
    BOTTLE_REGISTERED_IMAGES,
    BOTTLE_REGISTERED_TOTAL,
    BOTTLE_REPROJ_PX,
    GateError,
    matcher_for_capture,
    preserve_existing_splat,
    write_color_calibration,
)


def test_sequential_matching_for_video():
    assert matcher_for_capture("video") == "sequential"
    assert matcher_for_capture("video", "sequential") == "sequential"


def test_vocab_tree_rejected_for_video():
    with pytest.raises(GateError, match="vocab tree"):
        matcher_for_capture("video", "vocab_tree")


def test_existing_splat_not_retrained(monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("splat was retrained")

    monkeypatch.setattr("tools.step02_splatfacto._train", boom)
    monkeypatch.setattr("tools.step01_colmap.run_orbit", boom)
    report = preserve_existing_splat(
        {
            "registered_images": BOTTLE_REGISTERED_IMAGES,
            "registered_total": BOTTLE_REGISTERED_TOTAL,
            "reproj_px": BOTTLE_REPROJ_PX,
            "n_gaussians": BOTTLE_GAUSSIANS,
        }
    )
    assert report["trained_again"] is False
    assert report["registered_images"] == 224
    assert report["registered_total"] == 224
    assert report["reproj_px"] == 0.543
    assert report["n_gaussians"] == 954814
    assert report["matcher"] == "sequential"


def test_color_calibration_record_written(tmp_path: Path):
    path = write_color_calibration("bottle", tmp_path, False)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "color_calibration.json"
    assert saved["measured"] is False
    assert isinstance(saved["measured"], bool)
    path = write_color_calibration("bottle", tmp_path, True)
    assert json.loads(path.read_text(encoding="utf-8"))["measured"] is True
    assert saved["appearance_embeddings_retrained"] is False
