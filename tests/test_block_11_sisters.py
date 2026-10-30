"""Block 11 / G: sisters wait for a block 9 pass and their own Isaac frame."""

from __future__ import annotations

from pathlib import Path

from tools.block_gates import accept_sister


def test_sisters_wait_for_block9_pass(tmp_path: Path):
    frame = tmp_path / "frame_000.png"
    frame.write_bytes(b"\x89PNG\r\nframe")
    waiting = accept_sister({"name": "sister-1"}, {"mass_kg": 0.35}, [], frame, block9_pass=False, clip=0.9)
    assert waiting["passed"] is False
    assert waiting["copied"] is False
    assert "block 9" in waiting["reason"]


def test_sister_without_isaac_frame_not_passed(tmp_path: Path):
    result = accept_sister(
        {"name": "sister-1"},
        {"mass_kg": 0.35},
        [{"id": "lid"}],
        tmp_path / "missing.png",
        block9_pass=True,
        clip=0.9,
    )
    assert result["passed"] is False
    assert result["copied"] is False


def test_physics_and_joints_copied_to_sister(tmp_path: Path):
    frame = tmp_path / "frame_000.png"
    frame.write_bytes(b"\x89PNG\r\nframe")
    physics = {"mass_kg": 0.35, "learned_prior": False}
    joints = [{"id": "focus_ring", "travel": 90.0}]
    result = accept_sister({"name": "sister-1"}, physics, joints, frame, block9_pass=True, clip=0.88)
    assert result["passed"] is True
    assert result["physics"] == physics
    assert result["joints"] == joints
    assert result["vlm_looked"] is True
    drifted = accept_sister({"name": "sister-2"}, physics, joints, frame, block9_pass=True, clip=0.2)
    assert drifted["passed"] is False
