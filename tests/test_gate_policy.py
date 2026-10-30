"""Realistic agentic gates: HardStop, RepairLoop, ScoreOnly. Skip is not a pass."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_loop import _skill_align, _skill_package, _skill_skip, run_block_h
from tools.block_gates import measure_skill
from tools.gate_bypass import ENV_NAME
from tools.gate_policy import (
    HARD_STOP,
    REPAIR,
    SCORE_ONLY,
    classify,
    classify_skill_result,
    record_ledger,
    skill_exit,
    walk_continues,
)


def test_classify_tiers():
    assert classify("second_room_refuse") == HARD_STOP
    assert classify("no_vast") == HARD_STOP
    assert classify("no_input") == HARD_STOP
    assert classify("covers_room") == REPAIR
    assert classify("align_residual") == REPAIR
    assert classify("skill_skip") == SCORE_ONLY
    assert classify("no_metres") == SCORE_ONLY
    assert classify("hand_labels") == SCORE_ONLY
    assert classify("miou") == SCORE_ONLY
    assert classify("package_present") == SCORE_ONLY
    assert skill_exit(SCORE_ONLY) == 0
    assert skill_exit(HARD_STOP) == 2
    assert skill_exit(REPAIR) == 2
    assert walk_continues(SCORE_ONLY) is True
    assert walk_continues(HARD_STOP) is False


def test_second_room_refuse_is_hard_stop():
    assert classify("second_room_refuse") == HARD_STOP
    assert classify("covers_room", detail="second miss", attempts=2) == HARD_STOP
    assert classify_skill_result("SCENEAGENT-RUN-PIPELINE", 2, "second miss; room-sized") == HARD_STOP
    assert skill_exit(HARD_STOP) == 2


def test_skill_skip_exits_zero_and_ledger_skipped(tmp_path: Path):
    code = _skill_skip("orient", "can", tmp_path)
    assert code == 0
    ledger = json.loads((tmp_path / "data" / "can" / "gate_ledger.json").read_text(encoding="utf-8"))
    assert ledger["achieved"] is False
    assert ledger["m1_met"] is False
    assert ledger["m2_met"] is False
    assert ledger["milestone"] is False
    assert ledger["passed"] is False
    entry = ledger["entries"][-1]
    assert entry["class"] == SCORE_ONLY
    assert entry["check"] == "skipped"
    assert entry["achieved"] is False
    assert entry["passed"] is False


def test_package_present_exits_zero_unachieved(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    (folder / "scene.usdz").write_bytes(b"usdz")
    code = _skill_package("can", tmp_path)
    assert code == 0
    ledger = json.loads((folder / "gate_ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][-1]
    assert entry["check"] == "present_unachieved"
    assert entry["achieved"] is False
    assert entry["class"] == SCORE_ONLY
    assert ledger["milestone"] is False


def test_package_missing_exits_zero_not_invented(tmp_path: Path):
    code = _skill_package("can", tmp_path)
    assert code == 0
    ledger = json.loads((tmp_path / "data" / "can" / "gate_ledger.json").read_text(encoding="utf-8"))
    assert ledger["entries"][-1]["reason"] == "package_missing"
    assert ledger["achieved"] is False


def test_no_metres_align_continues_score_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tools.align.run_align", lambda *_a, **_k: 1)
    code = _skill_align("can", tmp_path)
    assert code == 0
    ledger = json.loads((tmp_path / "data" / "can" / "gate_ledger.json").read_text(encoding="utf-8"))
    entry = ledger["entries"][-1]
    assert entry["reason"] == "no_metres"
    assert entry["class"] == SCORE_ONLY
    assert entry["achieved"] is False
    assert entry["m1_met"] is False


def test_unmeasured_scale_passes_iterate_gate_without_m1(tmp_path: Path):
    folder = tmp_path / "data" / "can" / "ingest"
    folder.mkdir(parents=True)
    (folder / "scale.json").write_text(
        json.dumps({"method": "not_measured", "metres_per_unit": None}),
        encoding="utf-8",
    )
    (folder / "ingest_report.json").write_text(json.dumps({"m1_met": False}), encoding="utf-8")
    measure = measure_skill("can", "ingest", tmp_path)
    assert measure["pass"] is True
    assert measure["achieved"] is False
    assert measure["m1_met"] is False
    assert measure["metres_per_unit"] is None


def test_mesh_pass_split_from_achieved(tmp_path: Path):
    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    (folder / "mesh_score.json").write_text(
        json.dumps({"chamfer_cm": None, "f1_at_1cm": None, "achieved": False}),
        encoding="utf-8",
    )
    mesh = folder / "objects" / "can" / "mesh.ply"
    mesh.parent.mkdir(parents=True)
    mesh.write_bytes(b"ply\n")
    # Prefer a path _extracted_mesh finds
    (folder / "mesh.ply").write_bytes(b"ply\n")
    measure = measure_skill("can", "mesh", tmp_path)
    assert measure["pass"] is True
    assert measure["achieved"] is False


def test_orchestrate_walks_past_score_only_without_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    calls: list[str] = []

    def skip_like() -> int:
        calls.append("skip")
        return _skill_skip("orient", "can", tmp_path)

    def after() -> int:
        calls.append("after")
        return 0

    skills = [
        ("SCENEAGENT-ORIENT-BACKGROUND", skip_like),
        ("SCENEAGENT-OBJECT-SISTERS", after),
    ]
    code = run_block_h("can", tmp_path, skills=skills)
    assert code == 0
    assert calls == ["skip", "after"]
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["achieved"] is False
    assert status["m1_met"] is False
    assert status["m2_met"] is False
    assert status["milestone"] is False
    assert status["stopped"] is None
    assert not (tmp_path / "data" / "can" / "gate_bypass.json").is_file()


def test_hard_stop_still_ends_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_NAME, raising=False)
    calls: list[str] = []

    def stop() -> int:
        calls.append("stop")
        return 2

    def later() -> int:
        calls.append("later")
        return 0

    code = run_block_h(
        "can",
        tmp_path,
        skills=[
            ("SCENEAGENT-INGEST-DATA", stop),
            ("SCENEAGENT-RUN-STEP", later),
        ],
    )
    assert code == 2
    assert calls == ["stop"]
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["stopped"] == "SCENEAGENT-INGEST-DATA"
    assert status["achieved"] is False


def test_ledger_never_sets_milestones_true(tmp_path: Path):
    entry = record_ledger(
        "can",
        tmp_path,
        gate="hand_labels",
        gate_class=SCORE_ONLY,
        detail="miou null",
        check="unmet",
        reason="miou",
    )
    assert entry["achieved"] is False
    assert entry["m1_met"] is False
    assert entry["m2_met"] is False
    assert entry["milestone"] is False
    assert entry["passed"] is False
    payload = json.loads((tmp_path / "data" / "can" / "gate_ledger.json").read_text(encoding="utf-8"))
    assert payload["achieved"] is False
    assert payload["passed"] is False


def test_bypass_still_never_achieves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_NAME, "1")
    calls: list[str] = []

    def first() -> int:
        calls.append("first")
        return 2

    def second() -> int:
        calls.append("second")
        return 0

    code = run_block_h(
        "can",
        tmp_path,
        skills=[
            ("SCENEAGENT-INGEST-DATA", first),
            ("SCENEAGENT-RUN-STEP", second),
        ],
    )
    assert code == 0
    assert calls == ["first", "second"]
    recorded = json.loads((tmp_path / "data" / "can" / "gate_bypass.json").read_text(encoding="utf-8"))
    assert recorded["achieved"] is False
    assert recorded["m1_met"] is False
    assert recorded["m2_met"] is False
    assert recorded["milestone"] is False
    assert recorded["passed"] is False
