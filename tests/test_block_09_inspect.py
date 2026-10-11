"""Block 9 / F: Isaac inspect. A missing frame is not a render, and the GPU load test skips."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.block_gates import (
    INSPECT_PARAMS,
    apply_one_inspect_parameter,
    inspect_pair,
    isaac_rendered,
    paper_check_frame,
    record_isaac_timeout,
    route_problem,
)


def test_timeout_writes_error_not_fake_image(tmp_path: Path):
    record = record_isaac_timeout(tmp_path)
    assert record["error"] == "timeout"
    assert record["rendered"] is False
    assert record["isaac"] is False
    assert record["png"] is None
    assert not (tmp_path / "inspect" / "frame_000.png").exists()
    saved = json.loads((tmp_path / "isaac_check.json").read_text(encoding="utf-8"))
    assert saved["error"] == "timeout"
    assert saved["isaac_sim"] == "5.1"
    assert saved["isaac_lab"] == "2.3"


def test_missing_frame_fails_paper_check(tmp_path: Path):
    missing = paper_check_frame(tmp_path / "inspect" / "frame_000.png")
    assert missing["match"] is False
    assert missing["isaac"] is False
    frame = tmp_path / "inspect" / "frame_000.png"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(b"")
    empty = paper_check_frame(frame)
    assert empty["match"] is False
    assert empty["isaac"] is False


def test_vlm_sees_frame_beside_photo(tmp_path: Path):
    isaac = tmp_path / "frame_000.png"
    capture = tmp_path / "frame_00113.png"
    isaac.write_bytes(b"\x89PNG\r\nisaac")
    capture.write_bytes(b"\x89PNG\r\ncapture")
    pair = inspect_pair(isaac, capture, "orbit1-000", "orbit1-000")
    assert pair["pose"] == "orbit1-000"
    assert pair["images"] == [str(isaac), str(capture)]
    with pytest.raises(Exception, match="same COLMAP pose"):
        inspect_pair(isaac, capture, "orbit1-000", "orbit1-010")


def test_one_parameter_changed_on_fail():
    current = {"scale": 1.0, "yaw_deg": 0.0, "translation": [0.0, 0.0, 0.0]}
    updated = apply_one_inspect_parameter(current, "yaw_deg", 5.0)
    changed = [name for name in INSPECT_PARAMS if updated[name] != current[name]]
    assert changed == ["yaw_deg"]
    assert route_problem("scale") == "SCENEAGENT-ALIGN-SCENE"
    assert route_problem("rotation") == "SCENEAGENT-ORIENT-BACKGROUND"
    assert route_problem("placement") == "SCENEAGENT-ALIGN-SCENE"
    assert route_problem("missing_part") == "SCENEAGENT-RUN-STEP"
    assert route_problem("extra_part") == "SCENEAGENT-RUN-STEP"
    assert route_problem("wrong_label") == "SCENEAGENT-RUN-STEP"
    assert route_problem("penetration") == "physics"
    bottle = isaac_rendered(None)
    assert bottle["rendered"] is False
    assert bottle["isaac"] is False


@pytest.mark.gpu
def test_usdz_loads_in_isaac():
    if not os.environ.get("SCENEAGENT_HOST"):
        pytest.skip("SCENEAGENT_HOST is unset")
    raise AssertionError("Isaac load requires the live host and was not run from the laptop")
