"""Block 12 / J+K and the later I/J/K/A2 hold. Nothing here starts a policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.block_gates import GateError, later_phase_state, policy_state, start_later, start_training
from tools.iterate import iterate_skill


def test_status_stays_not_started_until_block9_frame(tmp_path: Path):
    missing = tmp_path / "inspect" / "frame_000.png"
    assert policy_state(missing) == "not_started"
    assert policy_state(None) == "not_started"
    with pytest.raises(GateError, match="not_started"):
        start_training("lerobot", missing)
    with pytest.raises(GateError, match="not_started"):
        start_training("pi0.5", None)
    with pytest.raises(GateError, match="not_started"):
        start_training("GR00T", None)
    with pytest.raises(GateError, match="not_started"):
        start_training("Franka", None)
    frame = tmp_path / "frame_000.png"
    frame.write_bytes(b"\x89PNG\r\nreal")
    assert policy_state(frame) == "not_started"
    with pytest.raises(GateError, match="not started"):
        start_training("pi0.5", frame)


def test_pipeline_does_not_start_editor_factory_vla_or_photogrammetry_without_inspect_frame(tmp_path: Path):
    missing = tmp_path / "inspect" / "frame_000.png"
    for phase in ("I", "J", "K", "A2"):
        assert later_phase_state(phase, missing) == "not_started"
        with pytest.raises(GateError, match="inspect"):
            start_later(phase, missing)
    code = iterate_skill("scan", "policy", tmp_path, vlm_mode="off")
    assert code == 2
    assert not (tmp_path / "data" / "scan" / "lerobot").exists()
    for name in ("editor", "vla", "photogrammetry"):
        assert iterate_skill("scan", name, tmp_path, vlm_mode="off") == 2
