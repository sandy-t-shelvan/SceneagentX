"""Block 6 / E: joints only when the parts file names them. Bottle two-state stays unavailable."""

from __future__ import annotations

import json
from pathlib import Path

from tools.block_gates import articulate_parts, pivot_from_geometry, record_unavailable_motion, update_parts_if_keyed


def test_urdf_joints_match_accepted_names(tmp_path: Path):
    named = articulate_parts(
        "scan",
        [
            {"id": "body", "joint": None},
            {"id": "focus_ring", "joint": "revolute", "travel": 90.0},
            {"id": "shutter_trigger", "joint": "prismatic", "travel": 4.0},
        ],
    )
    assert 'upper="1.570796"' in named["urdf"]
    assert 'upper="0.004000"' in named["urdf"]
    assert "focus_ring_joint" in named["urdf"]
    assert "shutter_trigger_joint" in named["urdf"]
    single = articulate_parts("scan", [{"id": "body", "joint": None}])
    assert "<joint" not in single["urdf"]
    assert single["motion"]["two_state"] == "unavailable"
    assert single["motion"]["parts"] == []
    (tmp_path / "asset.urdf").write_text(named["urdf"], encoding="utf-8")


def test_pivot_from_geometry_after_type():
    assert pivot_from_geometry(None, [0.1, 0.2, 0.3]) is None
    assert pivot_from_geometry("revolute", [0.1, 0.2, 0.3]) == [0.1, 0.2, 0.3]


def test_no_second_capture_records_unavailable():
    before = {"source": "vlm", "parts": []}
    motion = record_unavailable_motion(before)
    assert motion["two_state"] == "unavailable"
    assert motion["two_state_reason"] == "no second capture"
    assert motion["parts"] == []
    assert "invented" not in json.dumps(motion)


def test_empty_key_does_not_rewrite_parts(tmp_path: Path):
    path = tmp_path / "parts.json"
    original = '{"parts": [{"id": "body"}]}\n'
    path.write_text(original, encoding="utf-8")
    wrote = update_parts_if_keyed(path, "", {"parts": [{"id": "invented"}]})
    assert wrote is False
    assert path.read_text(encoding="utf-8") == original
