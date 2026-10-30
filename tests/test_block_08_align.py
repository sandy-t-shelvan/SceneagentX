"""Block 8 / F: no tag is unstarted, a skipped orientation is not Achieved."""

from __future__ import annotations

import numpy as np

from tools.block_gates import FIXTURE_IOU_MIN, align_record, mask_iou, orientation_record
from tools.changelog import achieved_word


def test_no_tag_records_unstarted():
    record = align_record(False)
    assert record["state"] == "unstarted"
    assert record["achieved"] is False


def test_no_background_skips_orientation():
    record = orientation_record(False)
    assert record["skipped"] is True
    assert record["state"] == "skipped"


def test_skipped_never_scored_achieved():
    assert achieved_word(True, skipped=True, kind="orientation") != "Achieved"
    assert achieved_word(True, skipped=True, kind="alignment") != "Achieved"
    assert achieved_word(True, asked=False) != "Achieved"
    assert orientation_record(False)["achieved"] is False


def test_residuals_on_fixture_converge():
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True
    score = mask_iou(mask, mask)
    assert score > FIXTURE_IOU_MIN
    assert score > 0.95
