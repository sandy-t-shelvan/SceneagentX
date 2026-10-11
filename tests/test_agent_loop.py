"""The skill router restarts one block and never sends a ply to COLMAP."""

from __future__ import annotations

import json
from pathlib import Path

from tools.agent_loop import SkillFault, choose_path, run_skill
from tools.common.status import load_status, save_status, set_path
from tools.pipeline import run


def test_ply_path_never_selects_video(tmp_path: Path, monkeypatch):
    called = {"colmap": False}

    def colmap(*args, **kwargs):
        called["colmap"] = True
        return 0

    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", colmap)
    mesh = tmp_path / "scan.ply"
    mesh.write_bytes(b"ply\n")
    # classify needs a real mesh; use a tiny obj instead
    mesh = tmp_path / "scan.obj"
    _box().export(mesh)
    code = run("scan", through="B2", root=tmp_path, source=mesh, n_gaussians=8, n_views=2)
    assert code == 2
    assert called["colmap"] is False
    assert choose_path("scan", tmp_path) == "ply"


def test_missing_video_stops_before_colmap(tmp_path: Path):
    doc = load_status("scan", tmp_path)
    set_path(doc, "input", {"kind": "video", "cameras": False, "capture_images": True})
    save_status("scan", doc, tmp_path)
    assert choose_path("scan", tmp_path) == "stop"
    text = (tmp_path / "data" / "scan" / "video_path.json").read_text(encoding="utf-8")
    assert "not_started" in text


def test_second_fault_restarts_only_that_skill(tmp_path: Path):
    folder = tmp_path / "data" / "scan" / "masks"
    folder.mkdir(parents=True)
    seen = []

    def run_once(parameter):
        seen.append(parameter)
        (folder / "object.json").write_text("{}\n", encoding="utf-8")

    def checker(obj, block_id, root):
        return {
            "decision": "restart",
            "called": True,
            "fault": "bad mask",
            "parameter": {"prompt": "camera.", "box_threshold": 0.3, "retrain": True},
        }

    code, visual = run_skill("scan", "B5", run_once, tmp_path, checker)
    assert code == 2
    assert visual is True
    assert len(seen) == 2
    assert seen[1]["prompt"] == "camera."
    assert seen[1]["box_threshold"] == 0.3
    assert "retrain" not in seen[1]
    agents = json.loads((tmp_path / "data" / "scan" / "agents.json").read_text(encoding="utf-8"))
    assert agents["visual_review"] is True
    assert agents["faults"][-1]["restarted"] is False
    assert agents["faults"][0]["restarted"] is True
    changelog = (tmp_path / "data" / "scan" / "changelog.md").read_text(encoding="utf-8")
    assert "bad mask" in changelog
    assert "**alignment** — Achieved" not in changelog
    assert "crewai" not in changelog.lower()


def test_pass_does_not_invent_isaac_or_a_prior(tmp_path: Path):
    folder = tmp_path / "data" / "scan" / "masks"
    folder.mkdir(parents=True)
    (folder / "object.json").write_text(
        json.dumps({"infill": "skipped", "projected": True}),
        encoding="utf-8",
    )

    def run_once(parameter):
        return None

    def checker(obj, block_id, root):
        return {"decision": "pass", "called": True}

    code, visual = run_skill("scan", "B5", run_once, tmp_path, checker)
    assert code == 0
    assert visual is True
    doc = load_status("scan", tmp_path)
    assert (doc.get("inspect") or {}).get("isaac") is not True


def test_a_broken_measurement_is_repaired_only_when_it_improves(tmp_path: Path):
    state = {"error": 2.0}

    def run_once(parameter):
        if parameter and parameter.get("box_threshold") == 0.4:
            state["error"] = 0.0
        else:
            state["error"] = 2.0

    def checker(obj, block_id, root):
        if state["error"] > 0:
            return {
                "decision": "restart",
                "called": False,
                "fault": "scale is off",
                "parameter": {"box_threshold": 0.4},
                "score": state["error"],
            }
        return {"decision": "pass", "called": False, "score": state["error"]}

    code, visual = run_skill("scan", "B2", run_once, tmp_path, checker)
    assert code == 0
    assert visual is False
    assert state["error"] == 0.0
    assert not (tmp_path / "data" / "scan" / "question.json").exists()


def test_a_worse_repair_is_thrown_away(tmp_path: Path):
    state = {"threshold": None, "error": 0.2}

    def run_once(parameter):
        state["threshold"] = None if not parameter else parameter.get("box_threshold")
        state["error"] = 5.0 if state["threshold"] == 0.9 else 0.2

    def checker(obj, block_id, root):
        if state["threshold"] != 0.9:
            return {
                "decision": "restart",
                "called": False,
                "fault": "break it",
                "parameter": {"box_threshold": 0.9},
                "score": state["error"],
            }
        return {"decision": "pass", "called": False, "score": state["error"]}

    code, _visual = run_skill("scan", "B2", run_once, tmp_path, checker)
    assert code == 2
    assert state["error"] == 0.2
    question = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    assert "worse" in question["question"].lower()


def test_the_same_try_asks_one_question(tmp_path: Path):
    calls = []

    def run_once(parameter):
        calls.append(parameter)
        raise SkillFault("registration failed")

    code, visual = run_skill("scan", "B2", run_once, tmp_path)
    assert code == 2
    assert visual is False
    assert calls == [None]
    first = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    code, _visual = run_skill("scan", "B2", run_once, tmp_path)
    assert code == 2
    second = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    assert first == second
    assert "change" in first["question"].lower()


def test_three_changed_retries_then_one_question(tmp_path: Path):
    calls = []
    folder = tmp_path / "data" / "scan" / "masks"
    folder.mkdir(parents=True)

    def run_once(parameter):
        calls.append(None if not parameter else parameter.get("box_threshold"))
        (folder / "object.json").write_text("{}\n", encoding="utf-8")

    def checker(obj, block_id, root):
        return {
            "decision": "restart",
            "called": False,
            "fault": "change the threshold",
            "parameter": {"box_threshold": round(0.2 + 0.1 * len(calls), 2), "retrain": True},
        }

    code, visual = run_skill("scan", "B5", run_once, tmp_path, checker)
    assert code == 2
    assert visual is False
    assert calls == [None, 0.3, 0.4, 0.5]
    question = json.loads((tmp_path / "data" / "scan" / "question.json").read_text(encoding="utf-8"))
    assert question["block"] == "B5"
    report = json.loads((tmp_path / "data" / "scan" / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["learned_prior"] is False
    assert report["metrics"]["gpu_minutes"] is None


def test_corrupt_mask_restarts_only_that_skill(tmp_path: Path, monkeypatch):
    trained = {"n": 0}

    def refuse(*args, **kwargs):
        trained["n"] += 1
        raise AssertionError("retrain")

    monkeypatch.setattr("tools.step02_splatfacto._train", refuse)
    monkeypatch.setattr("tools.step01_colmap.run_orbit", refuse)
    folder = tmp_path / "data" / "scan" / "masks"
    folder.mkdir(parents=True)
    mask = folder / "object.json"
    mask.write_text(
        json.dumps({"prompt": "camera.", "projected": True, "infill": "skipped", "corrupt": True}),
        encoding="utf-8",
    )

    def run_once(parameter):
        if parameter and parameter.get("box_threshold") == 0.22:
            mask.write_text(
                json.dumps(
                    {
                        "prompt": "camera.",
                        "projected": True,
                        "infill": "skipped",
                        "corrupt": False,
                        "box_threshold": 0.22,
                    }
                ),
                encoding="utf-8",
            )

    def checker(obj, block_id, root):
        current = json.loads(mask.read_text(encoding="utf-8"))
        if current.get("corrupt") is True:
            return {
                "decision": "restart",
                "called": False,
                "fault": "bad mask",
                "parameter": {"box_threshold": 0.22, "retrain": True},
            }
        return {"decision": "pass", "called": False}

    code, _visual = run_skill("scan", "B5", run_once, tmp_path, checker)
    assert code == 0
    assert trained["n"] == 0
    saved = json.loads(mask.read_text(encoding="utf-8"))
    assert saved["corrupt"] is False
    assert saved["box_threshold"] == 0.22
    agents = json.loads((tmp_path / "data" / "scan" / "agents.json").read_text(encoding="utf-8"))
    assert agents["faults"]
    assert {item["block"] for item in agents["faults"]} == {"B5"}
    assert not (tmp_path / "data" / "scan" / "question.json").exists()


def test_measurements_do_not_call_the_model(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    picture = tmp_path / "data" / "scan" / "renders" / "synthetic"
    picture.mkdir(parents=True)
    (picture / "view_000.png").write_bytes(b"\x89PNG\r\n")
    called = {"n": 0}

    def vision(api_key, image, block_id):
        called["n"] += 1
        return {"decision": "pass"}

    monkeypatch.setattr("tools.agent_loop._vision_decision", vision)

    def run_once(parameter):
        return None

    code, visual = run_skill("scan", "B2", run_once, tmp_path)
    assert code == 0
    assert visual is False
    assert called["n"] == 0
    masks = tmp_path / "data" / "scan" / "masks"
    masks.mkdir()
    (masks / "object.json").write_text("{}\n", encoding="utf-8")
    (masks / "view_000_mask.png").write_bytes(b"\x89PNG\r\n")
    code, visual = run_skill("scan", "B5", run_once, tmp_path)
    assert code == 0
    assert visual is True
    assert called["n"] == 1


def _box():
    import trimesh

    return trimesh.creation.box(extents=(0.1, 0.04, 0.03))
