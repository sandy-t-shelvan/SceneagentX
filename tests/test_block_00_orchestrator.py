"""Block 0 / H: one skill, hard rules before the planner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_loop import SkillFault, UnknownSkillError, accept_planner_skill, advance_once, choose_path, main
from tools.common.status import load_status, save_status, set_path


def test_ply_never_selects_colmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("COLMAP was selected")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", boom)
    monkeypatch.setattr("tools.step02_splatfacto._train", boom)
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": "polycam_mesh", "cameras": False, "capture_images": False})
    save_status("scan", doc, tmp_path)
    code = main(["scan", "--once", "--root", str(tmp_path), "--skill", "SCENEAGENT-INGEST-DATA"])
    assert code == 0
    once = json.loads((tmp_path / "data" / "scan" / "once.json").read_text(encoding="utf-8"))
    assert once["colmap"] == "not called"
    assert once["n_skills"] == 1
    assert once["skills"] == ["SCENEAGENT-INGEST-DATA"]


def test_missing_orbit1_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("COLMAP was started")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", boom)
    code = main(["bottle", "--once", "--root", str(tmp_path)])
    assert code == 2
    assert choose_path("bottle", tmp_path) == "stop"
    text = (tmp_path / "data" / "bottle" / "video_path.json").read_text(encoding="utf-8")
    assert "not_started" in text


def test_corrupted_mask_restarts_only_mask_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("a skill other than the mask skill ran")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", boom)
    monkeypatch.setattr("tools.step02_splatfacto._train", boom)
    raw = tmp_path / "data" / "cam" / "raw"
    raw.mkdir(parents=True)
    (raw / "orbit1.mp4").write_bytes(b"video")
    mask = tmp_path / "data" / "cam" / "masks"
    mask.mkdir()
    (mask / "object.json").write_text(json.dumps({"corrupt": True, "prompt": "camera."}), encoding="utf-8")
    code = main(["cam", "--once", "--root", str(tmp_path)])
    assert code == 0
    saved = json.loads((mask / "object.json").read_text(encoding="utf-8"))
    assert saved["corrupt"] is False
    assert saved["box_threshold"] == 0.22
    once = json.loads((tmp_path / "data" / "cam" / "once.json").read_text(encoding="utf-8"))
    assert once["skills"] == ["SCENEAGENT-RUN-STEP"]
    assert once["n_skills"] == 1
    agents = json.loads((tmp_path / "data" / "cam" / "agents.json").read_text(encoding="utf-8"))
    assert {item["block"] for item in agents["faults"]} == {"B5"}


def test_second_fault_writes_question_and_stops(tmp_path: Path):
    raw = tmp_path / "data" / "bottle" / "raw"
    raw.mkdir(parents=True)
    (raw / "orbit1.mp4").write_bytes(b"video")
    calls: list[str] = []

    def runner(obj: str, skill: str, root: Path | None, parameter: dict | None) -> None:
        del obj, root, parameter
        calls.append(skill)
        raise SkillFault("still wrong")

    code = advance_once("bottle", tmp_path, planner=lambda _obj, _root: "SCENEAGENT-RUN-STEP", runner=runner)
    assert code == 2
    assert calls == ["SCENEAGENT-RUN-STEP", "SCENEAGENT-RUN-STEP"]
    question = json.loads((tmp_path / "data" / "bottle" / "question.json").read_text(encoding="utf-8"))
    assert "change" in question["question"].lower()
    once = json.loads((tmp_path / "data" / "bottle" / "once.json").read_text(encoding="utf-8"))
    assert once["n_skills"] == 1


def test_planner_rejects_unknown_skill_id(tmp_path: Path):
    with pytest.raises(UnknownSkillError):
        accept_planner_skill("NOT-A-SKILL")
    raw = tmp_path / "data" / "bottle" / "raw"
    raw.mkdir(parents=True)
    (raw / "orbit1.mp4").write_bytes(b"video")
    code = main(["bottle", "--once", "--root", str(tmp_path), "--skill", "NOT-A-SKILL"])
    assert code == 2
    assert not (tmp_path / "data" / "bottle" / "once.json").exists()
