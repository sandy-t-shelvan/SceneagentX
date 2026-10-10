"""Part names stay one body until a vision reply asks for a different count."""

from tools.bottle_parts import json_object, parse_part_reply
from tools.partfield_parts import choose_k


def test_missing_partfield_score_does_not_invent_a_split():
    assert choose_k(None) is None
    assert choose_k({1: 0.4, 2: 0.4}) is None


def test_part_reply_keeps_names_and_only_a_changed_count():
    parsed = parse_part_reply(
        {
            "object": "bottle",
            "parts": [{"id": "cap", "note": "lid"}, {"id": "Body"}],
            "split_wrong": True,
            "requested_part_count": 2,
        },
        current_count=1,
    )
    assert parsed["parts"] == ["cap", "body"]
    assert parsed["requested_part_count"] == 2
    same = parse_part_reply({"parts": ["cap"], "requested_part_count": 1}, current_count=1)
    assert same["requested_part_count"] is None


def test_json_object_strips_a_fence():
    assert json_object('```json\n{"parts": []}\n```') == {"parts": []}


def test_pivot_waits_until_a_joint_is_named():
    import numpy as np

    from tools.bottle_parts import place_pivots

    points = np.column_stack((np.zeros(20), np.zeros(20), np.linspace(0, 2, 20)))
    unnamed = [{"id": "cap", "joint": None}]
    place_pivots(unnamed, points)
    assert "origin" not in unnamed[0]
    named = [{"id": "cap", "joint": "revolute"}]
    place_pivots(named, points)
    assert named[0]["origin"][2] > 1.5
