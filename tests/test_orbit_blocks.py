"""Orbit-capture records stay measured only when a number exists."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.agent_loop import run_skill
from tools.orbit_blocks import (
    assign_feature_indices,
    build_ingest_report,
    colour_calibration_record,
    convex_hull_summary,
    count_floaters,
    extend_physics_record,
    orbit_capture,
    resolve_scene_mean,
    write_isaac_check,
)
from tools.sceneagent_blocks import run_design


def test_ingest_report_records_the_existing_orbit_and_does_not_retrain(tmp_path: Path):
    folder = tmp_path / "data" / "bottle"
    colmap = folder / "colmap" / "orbit1"
    colmap.mkdir(parents=True)
    (colmap / "analyzer.txt").write_text(
        "num_images: 224\nnum_reg_images: 224\nmean_reprojection_error: 0.54\n",
        encoding="utf-8",
    )
    (colmap / "models.json").write_text('["0"]\n', encoding="utf-8")
    (folder / "ingest").mkdir(parents=True)
    (folder / "ingest" / "ns_eval.json").write_text(
        json.dumps({"results": {"psnr": 28.1, "ssim": 0.88, "lpips": 0.126}}),
        encoding="utf-8",
    )
    (folder / "ingest" / "colour_calibration.json").write_text(
        json.dumps(colour_calibration_record(True, 224, "reviewed existing orbit frames")),
        encoding="utf-8",
    )
    report = build_ingest_report("bottle", folder, tmp_path)
    assert report["capture"]["used_for_eval"] == "data/bottle/raw/orbit1.MOV"
    assert report["capture"]["ns_train"] == "not_run"
    assert report["capture"]["colmap"] == "not_rerun"
    assert report["psnr"] == pytest.approx(28.1)
    assert report["lpips"] == pytest.approx(0.126)
    assert report["m1_met"] is False
    assert report["colour_calibration"]["exposure_reviewed"] is True
    assert report["colour_calibration"]["appearance_embeddings_retrained"] is False


def test_bottle_orbit_files_exist_and_the_scan_has_no_video():
    from tools.common.paths import repo_root

    record = orbit_capture("bottle", repo_root())
    assert [item["path"] for item in record["orbits"]] == [
        "data/bottle/raw/orbit1.MOV",
        "data/bottle/raw/orbit2.MOV",
        "data/bottle/raw/orbit3.MOV",
    ]
    assert all(item["exists"] is True for item in record["orbits"])
    scan = orbit_capture("scan", repo_root())
    assert scan["orbit_video"] is None
    assert scan["colmap"] == "not_called"


def test_two_labels_assign_every_gaussian_and_one_vector_does_not():
    inside = np.array([True, False, True, False])
    mixed = resolve_scene_mean({"vector": [0.1, 0.2]})
    assert mixed["measured"] is False
    assert assign_feature_indices(inside, False).tolist() == [0, -1, 0, -1]
    separate = resolve_scene_mean({"vector": [0.1, 0.2], "scene_mean": [0.4, 0.5], "scene_mean_measured": True})
    assert separate["measured"] is True
    both = assign_feature_indices(inside, True)
    assert set(both.tolist()) == {0, 1}
    copied = resolve_scene_mean({"vector": [0.1, 0.2], "scene_mean": [0.1, 0.2], "scene_mean_measured": True})
    assert copied["measured"] is False


def test_floater_count_is_outside_two_sigma():
    cloud = np.zeros((40, 3), dtype=np.float64)
    cloud[:20, 0] = np.linspace(-0.2, 0.2, 20)
    cloud[20:, 1] = np.linspace(-0.2, 0.2, 20)
    cloud = np.vstack([cloud, np.array([[30.0, 30.0, 30.0]])])
    assert count_floaters(cloud) == 1


def test_tiny_hull_is_built_and_a_large_cloud_is_not():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    built = convex_hull_summary(points)
    assert built["convex_hull"] == "built"
    assert built["n_faces"] >= 1
    assert convex_hull_summary(np.zeros((6000, 3)))["convex_hull"] == "not_built"


def test_learned_prior_cannot_become_true():
    kept = extend_physics_record({"material": "steel", "mass_kg": 0.35, "friction": 0.5, "rigidity": 0.98})
    assert kept["learned_prior"] is False
    assert kept["material"] == "steel"
    assert kept["mass_kg"] == 0.35
    assert kept["static_friction"] == 0.5
    assert kept["deformable_type"] == "rigid"
    assert kept["drop_test"] == "not_run"
    assert "dynamic_friction" not in kept
    with pytest.raises(RuntimeError, match="learned_prior"):
        extend_physics_record({"material": "steel", "mass_kg": 1, "per_gaussian": {"learned_prior": True}})


def test_fewer_than_nine_or_unmeasured_similarity_stays_partial(tmp_path: Path):
    folder = tmp_path / "data" / "scan" / "sisters"
    folder.mkdir(parents=True)
    (folder / "objects.json").write_text(
        json.dumps({"objects": [{"file": "a"}, {"file": "b"}, {"file": "c"}], "trellis": True}),
        encoding="utf-8",
    )
    state = run_design("scan", tmp_path)
    by_id = {step["id"]: step for step in state["steps"]}
    assert by_id["G"]["state"] == "partial"
    assert "stays partial" in by_id["G"]["result"]
    assert by_id["I"]["state"] == "not_started"
    assert by_id["J"]["state"] == "not_started"
    assert by_id["K"]["state"] == "not_started"
    report = json.loads((tmp_path / "data" / "scan" / "report.json").read_text(encoding="utf-8"))
    assert set(report) == {"status", "metrics", "renders", "suggested_fixes"}
    assert report["metrics"]["learned_prior"] is False


def test_blocks_i_j_k_stay_not_started(tmp_path: Path):
    state = run_design("marker", tmp_path)
    by_id = {step["id"]: step for step in state["steps"]}
    assert by_id["I"]["state"] == "not_started"
    assert by_id["J"]["state"] == "not_started"
    assert by_id["K"]["state"] == "not_started"


def test_isaac_timeout_writes_the_error_and_no_frame(tmp_path: Path):
    folder = tmp_path / "data" / "scan"
    folder.mkdir(parents=True)
    record = write_isaac_check(folder, "sim.reset did not return within 120 seconds")
    assert record["rendered"] is False
    assert record["png"] is None
    assert record["alignment"] == "skipped"
    saved = json.loads((folder / "isaac_check.json").read_text(encoding="utf-8"))
    assert "120" in saved["error"]
    assert list(folder.glob("*.png")) == []


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
    assert "three retries" in question["question"]
    report = json.loads((tmp_path / "data" / "scan" / "report.json").read_text(encoding="utf-8"))
    assert report["metrics"]["learned_prior"] is False


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
        json.dumps({"prompt": "bottle.", "projected": True, "infill": "skipped", "corrupt": True}),
        encoding="utf-8",
    )

    def run_once(parameter):
        if parameter and parameter.get("box_threshold") == 0.22:
            mask.write_text(
                json.dumps(
                    {
                        "prompt": "bottle.",
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
    agents = json.loads((tmp_path / "data" / "scan" / "agents.json").read_text(encoding="utf-8"))
    assert {item["block"] for item in agents["faults"]} == {"B5"}
