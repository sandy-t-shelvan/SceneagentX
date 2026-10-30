"""Vision motion names joints without sending a ply to COLMAP."""

from __future__ import annotations

import json
from pathlib import Path

from tools.common.status import load_status, save_status, set_path
from tools.motion import _validated, run_motion
from tools.paper_check import build_report


def _stage(tmp_path: Path, kind: str = "polycam_mesh") -> None:
    folder = tmp_path / "data" / "scan"
    (folder / "masks").mkdir(parents=True)
    (folder / "renders" / "synthetic").mkdir(parents=True)
    (folder / "renders" / "synthetic" / "view_000.png").write_bytes(b"\x89PNG\r\n")
    (folder / "masks" / "object.json").write_text(
        json.dumps({"label": "camera", "prompt": "camera.", "infill": "skipped", "projected": True}),
        encoding="utf-8",
    )
    (folder / "physics.json").write_text(
        json.dumps(
            {
                "material": "aluminum alloy and steel",
                "friction": 0.5,
                "rigidity": 0.98,
                "mass_kg": 2.8,
                "density_kg_m3": 3500.0,
                "per_gaussian": {"learned_prior": False, "n": 4},
            }
        ),
        encoding="utf-8",
    )
    (folder / "align.json").write_text(
        json.dumps({"photometric": "skipped", "scale_m": 0.25, "up_axis": [1, 0, 0]}),
        encoding="utf-8",
    )
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": kind, "cameras": False, "capture_images": kind != "polycam_mesh"})
    save_status("scan", doc, tmp_path)


def _part(name: str, joint: str, travel: float) -> dict:
    return {
        "id": name,
        "joint": joint,
        "axis": "+z" if joint == "prismatic" else "+y",
        "travel": travel,
        "view": "view_000.png",
        "point": [0.4, 0.5],
        "friction": 0.4,
        "rigidity": 0.9,
    }


def test_named_parts_become_joints(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr(
        "tools.motion.assign_parts",
        lambda *args, **kwargs: {"shutter": 10, "dial": 8, "body": 82},
    )

    def ask(images, role, prior):
        if role == "review":
            return {"called": True, "decision": "pass"}
        return {"called": True, "object": "camera", "parts": [_part("shutter", "prismatic", 0.005), _part("dial", "revolute", 90)]}

    assert run_motion("scan", tmp_path, ask) == 0
    urdf = (tmp_path / "data" / "scan" / "asset.urdf").read_text(encoding="utf-8")
    assert urdf.lower().count("<joint") == 2
    assert 'type="prismatic"' in urdf
    assert 'type="revolute"' in urdf
    masks = json.loads((tmp_path / "data" / "scan" / "masks" / "object.json").read_text(encoding="utf-8"))
    assert masks["vlm_object"] == "camera"
    physics = json.loads((tmp_path / "data" / "scan" / "physics.json").read_text(encoding="utf-8"))
    assert physics["per_gaussian"]["learned_prior"] is False
    assert abs(sum(item["mass_kg"] for item in physics["per_part"]) - 2.8) < 1e-4
    row = {item["id"]: item for item in build_report("scan", tmp_path)["blocks"]}
    assert row["articulation"]["match"] == "partial"
    assert "gpt-6-astra" in row["articulation"]["note"]


def test_travel_is_capped():
    parsed = _validated(
        {
            "called": True,
            "parts": [
                {"id": "shutter", "joint": "prismatic", "axis": "+z", "travel": 5, "point": [0.2, 0.2]},
                {"id": "dial", "joint": "revolute", "axis": "+y", "travel": 400, "point": [0.2, 0.2]},
            ],
        }
    )
    by_id = {part["id"]: part for part in parsed["parts"]}
    assert by_id["shutter"]["travel"] == 0.02
    assert by_id["dial"]["travel"] == 120


def test_no_parts_leave_the_urdf_joint_free(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"body": 10})

    def ask(images, role, prior):
        if role == "review":
            return {"called": True, "decision": "pass"}
        return {"called": True, "object": "camera", "parts": []}

    assert run_motion("scan", tmp_path, ask) == 0
    urdf = (tmp_path / "data" / "scan" / "asset.urdf").read_text(encoding="utf-8")
    assert "<joint" not in urdf.lower()


def test_ply_does_not_call_colmap(tmp_path: Path, monkeypatch):
    _stage(tmp_path)

    def boom(*args, **kwargs):
        raise AssertionError("colmap")

    monkeypatch.setattr("tools.step01_colmap.run_orbit", boom)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"body": 4})

    def ask(images, role, prior):
        return {"called": True, "decision": "pass", "object": "camera", "parts": []}

    assert run_motion("scan", tmp_path, ask) == 0


def test_missing_video_does_not_call_the_model(tmp_path: Path):
    _stage(tmp_path, kind="video")
    calls = []

    def ask(images, role, prior):
        calls.append(role)
        return {"called": True, "decision": "pass", "parts": []}

    assert run_motion("scan", tmp_path, ask) == 0
    assert calls == []
    text = (tmp_path / "data" / "scan" / "video_path.json").read_text(encoding="utf-8")
    assert "not_started" in text


def test_second_fault_stops(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"shutter": 3, "body": 7})
    calls = []

    def ask(images, role, prior):
        calls.append(role)
        if role == "review":
            return {"called": True, "decision": "fault", "block": "B11", "fault": "bad axis"}
        part = _part("shutter", "prismatic", 0.004)
        if calls.count("propose") > 1:
            part = dict(part)
            part["axis"] = "-z"
        return {"called": True, "object": "camera", "parts": [part]}

    assert run_motion("scan", tmp_path, ask) == 2
    assert calls == ["propose", "review", "propose", "review"]
    agents = json.loads((tmp_path / "data" / "scan" / "agents.json").read_text(encoding="utf-8"))
    assert agents["visual_review"] is True
    assert agents["faults"][0]["restarted"] is True
    assert agents["faults"][-1]["restarted"] is False
    assert "bad axis" in (tmp_path / "data" / "scan" / "changelog.md").read_text(encoding="utf-8")
    question = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    assert question["block"] == "B11"
    assert "change" in question["question"].lower()
    urdf = (tmp_path / "data" / "scan" / "asset.urdf").read_text(encoding="utf-8")
    assert 'xyz="0.0 0.0 1.0"' in urdf or 'xyz="0 0 1"' in urdf


def test_same_motion_is_not_retried(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"shutter": 3, "body": 7})
    calls = []

    def ask(images, role, prior):
        calls.append(role)
        if role == "review":
            return {"called": True, "decision": "fault", "block": "B11", "fault": "bad axis"}
        return {"called": True, "object": "camera", "parts": [_part("shutter", "prismatic", 0.004)]}

    assert run_motion("scan", tmp_path, ask) == 2
    assert calls == ["propose", "review", "propose"]
    question = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    assert "same motion" in question["question"]


def test_detection_fault_changes_only_the_prompt(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"body": 4})

    def ask(images, role, prior):
        if role == "review":
            return {"called": True, "decision": "fault", "block": "B5", "fault": "wrong name", "prompt": "radio."}
        return {"called": True, "object": "radio", "parts": []}

    assert run_motion("scan", tmp_path, ask) == 0
    masks = json.loads((tmp_path / "data" / "scan" / "masks" / "object.json").read_text(encoding="utf-8"))
    assert masks["prompt"] == "radio."
    assert masks["infill"] == "skipped"


def test_up_axis_fault_keeps_photometric_skipped(tmp_path: Path, monkeypatch):
    _stage(tmp_path)
    monkeypatch.setattr("tools.motion.assign_parts", lambda *args, **kwargs: {"body": 4})

    def ask(images, role, prior):
        if role == "review":
            return {"called": True, "decision": "fault", "block": "B4", "fault": "on its side", "up_axis": [0, 0, 1]}
        return {"called": True, "object": "camera", "parts": []}

    assert run_motion("scan", tmp_path, ask) == 0
    align = json.loads((tmp_path / "data" / "scan" / "align.json").read_text(encoding="utf-8"))
    assert align["photometric"] == "skipped"
    assert align["up_axis"] == [0.0, 0.0, 1.0]
