"""The gate switch is off unless asked. A bypass continues and is not a pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_loop import run_block_h
from tools.block_gates import stop_after_second_part_fault
from tools.gate_bypass import ENV_NAME


def test_default_stops_and_bypass_does_not_mark_achieved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    calls: list[str] = []

    def first() -> int:
        calls.append("first")
        return 2

    def second() -> int:
        calls.append("second")
        return 0

    skills = [
        ("SCENEAGENT-INGEST-DATA", first),
        ("SCENEAGENT-RUN-STEP", second),
    ]
    code = run_block_h("can", tmp_path, skills=skills)
    assert code == 2
    assert calls == ["first"]
    folder = tmp_path / "data" / "can"
    assert not (folder / "gate_bypass.json").is_file()
    status = json.loads((folder / "block_h.json").read_text(encoding="utf-8"))
    assert status["achieved"] is False
    assert status["m1_met"] is False
    assert status["m2_met"] is False
    assert status["milestone"] is False

    monkeypatch.setenv(ENV_NAME, "1")
    calls.clear()
    code = run_block_h("can", tmp_path, skills=skills)
    assert code == 0
    assert calls == ["first", "second"]
    recorded = json.loads((folder / "gate_bypass.json").read_text(encoding="utf-8"))
    assert recorded["achieved"] is False
    assert recorded["m1_met"] is False
    assert recorded["m2_met"] is False
    assert recorded["milestone"] is False
    assert recorded["passed"] is False
    gate = recorded["gates"][0]
    assert gate["gate"] == "SCENEAGENT-INGEST-DATA"
    assert gate["bypassed"] is True
    assert gate["check"] == "unmet"
    assert gate["achieved"] is False
    assert gate["m1_met"] is False
    assert gate["m2_met"] is False
    assert gate["passed"] is False


def test_second_fault_still_returns_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    assert stop_after_second_part_fault("can", tmp_path, ["fail", "fail"]) == 2
    assert not (tmp_path / "data" / "can" / "gate_bypass.json").is_file()

    monkeypatch.setenv(ENV_NAME, "1")
    assert stop_after_second_part_fault("can", tmp_path, ["fail", "fail"]) == 2
    recorded = json.loads((tmp_path / "data" / "can" / "gate_bypass.json").read_text(encoding="utf-8"))
    assert recorded["achieved"] is False
    assert recorded["m1_met"] is False
    assert recorded["m2_met"] is False
    match = [item for item in recorded["gates"] if item["gate"] == "second_fault"]
    assert match
    assert match[0]["check"] == "unmet"
    assert match[0]["passed"] is False
    assert match[0]["milestone"] is False
