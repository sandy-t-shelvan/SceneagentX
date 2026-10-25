"""Can M2 hand-label scorer. A skipped or missing label is never a pass."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.can_hand_labels import REQUIRED_VIEWS, score_hand_labels, score_object
from tools.common.paths import repo_root

LIVE_MANIFEST = repo_root() / "data" / "can" / "hand_labels" / "manifest.json"
LIVE_B1 = repo_root() / "data" / "can" / "ingest" / "block_b_b1.json"


def test_missing_labels_do_not_fail_block_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tools.can_instances import main

    def boom(*_args, **_kwargs):
        raise AssertionError("hand labels were scored")

    monkeypatch.setattr("tools.can_hand_labels.score_object", boom)
    record = score_hand_labels(_plant_slots(tmp_path / "labels"))
    assert record["ok"] is False
    assert record["m2_met"] is False
    assert main(["--obj", "lamp", "--root", str(tmp_path)]) == 0


def test_missing_labels_refuse_to_pass(tmp_path: Path):
    labels = _plant_slots(tmp_path)
    record = score_hand_labels(labels)
    assert record["ok"] is False
    assert record["passed"] is False
    assert record["achieved"] is False
    assert record["m2_met"] is False
    assert record["hand_labels"] == 0
    assert record["miou"] is None
    assert any("missing" in item for item in record["skipped"])


def test_placeholder_slots_are_not_drawn_labels(tmp_path: Path):
    labels = _plant_slots(tmp_path, placeholders=True)
    record = score_hand_labels(labels)
    assert record["hand_labels"] == 0
    assert record["miou"] is None
    assert record["ok"] is False


def test_sam_auto_mask_copied_into_slot_is_refused(tmp_path: Path):
    labels = _plant_slots(tmp_path)
    sam = np.zeros((8, 8), dtype=np.uint8)
    sam[2:6, 2:6] = 255
    sam_path = tmp_path / "data" / "can" / "iou" / "orbit2" / "frame_00001_mask.png"
    sam_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sam, mode="L").save(sam_path)
    _write_mask(labels / "masks" / "orbit2_frame_00001.png", sam)
    record = score_hand_labels(labels)
    assert "orbit2_frame_00001" in record["sam_copies"]
    assert record["hand_labels"] == 0
    assert record["miou"] is None
    assert record["ok"] is False
    assert any("SAM" in item for item in record["mismatch"] + record["skipped"])


def test_twenty_drawn_masks_and_renders_measure_miou(tmp_path: Path):
    labels = _plant_slots(tmp_path)
    renders = labels / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    hand = np.zeros((8, 8), dtype=np.uint8)
    hand[1:7, 1:7] = 255
    rendered = np.zeros((8, 8), dtype=np.uint8)
    rendered[1:7, 1:6] = 255
    for index in range(REQUIRED_VIEWS):
        view_id = f"orbit2_frame_{index + 1:05d}" if index < 10 else f"orbit3_frame_{index - 9:05d}"
        _write_mask(labels / "masks" / f"{view_id}.png", hand)
        _write_mask(renders / f"{view_id}.png", rendered)
    record = score_hand_labels(labels, renders)
    assert record["hand_labels"] == REQUIRED_VIEWS
    assert record["miou"] == pytest.approx(5 / 6)
    assert record["m2_met"] is False
    assert record["achieved"] is False
    assert record["passed"] is False
    assert record["ok"] is True


def test_partial_labels_do_not_invent_miou(tmp_path: Path):
    labels = _plant_slots(tmp_path)
    _write_mask(labels / "masks" / "orbit2_frame_00001.png", np.ones((4, 4), dtype=np.uint8) * 255)
    record = score_hand_labels(labels)
    assert record["hand_labels"] == 1
    assert record["miou"] is None
    assert record["ok"] is False
    assert record["m2_met"] is False


@pytest.mark.skipif(not LIVE_MANIFEST.is_file(), reason="data/can/hand_labels/manifest.json is absent")
def test_live_can_set_keeps_m2_false_and_miou_null():
    root = repo_root() / "data" / "can"
    record = score_object(root)
    assert record["n_views"] == REQUIRED_VIEWS
    assert record["hand_labels"] == 0
    assert record["miou"] is None
    assert record["m2_met"] is False
    assert record["achieved"] is False
    assert record["passed"] is False
    assert record["ok"] is False
    instances = json.loads((root / "semantics" / "instances.json").read_text(encoding="utf-8"))
    assert instances["hand_labels"] == 0
    assert instances["miou"] is None
    block_b = json.loads((root / "ingest" / "block_b.json").read_text(encoding="utf-8"))
    assert block_b.get("m2_met", False) is False
    assert block_b.get("achieved", False) is False
    assert block_b["m2_achieved"] is False


@pytest.mark.skipif(not LIVE_B1.is_file(), reason="data/can/ingest/block_b_b1.json is absent")
def test_orbit1_b1_record_stays_at_26_percent():
    b1 = json.loads(LIVE_B1.read_text(encoding="utf-8"))
    orbit1 = next(row for row in b1["orbits"] if row["orbit"] == "orbit1")
    assert orbit1["n_views"] == 367
    assert orbit1["n_views_with_detection"] == 96
    assert orbit1["detection_rate"] == pytest.approx(96 / 367)
    assert b1["hand_labels"] == 0
    assert b1["miou"] is None
    assert b1["achieved"] is False


def _plant_slots(tmp_path: Path, *, placeholders: bool = False) -> Path:
    labels = tmp_path / "data" / "can" / "hand_labels"
    (labels / "masks").mkdir(parents=True, exist_ok=True)
    (labels / "views").mkdir(parents=True, exist_ok=True)
    views = []
    for index in range(REQUIRED_VIEWS):
        if index < 10:
            view_id = f"orbit2_frame_{index + 1:05d}"
            orbit = "orbit2"
        else:
            view_id = f"orbit3_frame_{index - 9:05d}"
            orbit = "orbit3"
        views.append(
            {
                "id": view_id,
                "orbit": orbit,
                "frame": f"{view_id.split('_', 1)[1]}.png",
                "view": f"views/{view_id}.jpg",
                "mask": f"masks/{view_id}.png",
                "drawn": False,
            }
        )
        if placeholders:
            (labels / "masks" / f"{view_id}.png.placeholder").write_bytes(b"PLACEHOLDER")
    (labels / "manifest.json").write_text(
        json.dumps(
            {
                "object": "can",
                "n_views": REQUIRED_VIEWS,
                "hand_labels": 0,
                "miou": None,
                "m2_met": False,
                "achieved": False,
                "orbit1_reprompt": "skipped",
                "views": views,
            }
        ),
        encoding="utf-8",
    )
    (labels / "sam_auto.json").write_text(
        json.dumps(
            {
                "kind": "sam_auto",
                "not_hand_labels": True,
                "views": [
                    {
                        "id": "orbit2_frame_00001",
                        "local_mask": "data/can/iou/orbit2/frame_00001_mask.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return labels


def _write_mask(path: Path, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(pixels, dtype=np.uint8), mode="L").save(path)
