"""Block 4 / E: part count, a generated tip, and a second review fault."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.block_gates import (
    GateError,
    flag_generated_tip,
    keep_generated_forever,
    resolve_part_count,
    stop_after_second_part_fault,
)


def test_no_winning_k_gives_one_body():
    resolved = resolve_part_count(None)
    assert resolved["count"] == 1
    assert resolved["one_body"] is True
    assert resolved["ids"] == ["body"]


def test_vlm_may_change_count_by_one():
    assert resolve_part_count(3, 4)["count"] == 4
    assert resolve_part_count(3, 2)["count"] == 2
    with pytest.raises(GateError, match="at most 1"):
        resolve_part_count(3, 5)


def test_generated_tip_flagged():
    part = flag_generated_tip({"id": "tip", "generate": True, "surface": "tip"})
    assert part["generated"] is True
    cleared = keep_generated_forever(part, {"generated": False, "note": "scanned"})
    assert cleared["generated"] is True


def test_second_review_fault_stops(tmp_path: Path):
    code = stop_after_second_part_fault("bottle", tmp_path, ["fail", "fail"])
    assert code == 2
    question = json.loads((tmp_path / "data" / "bottle" / "question.json").read_text(encoding="utf-8"))
    assert question["block"] == "B9"
    assert "second" in question["question"].lower()
