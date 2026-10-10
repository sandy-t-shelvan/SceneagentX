import json

import cv2

from tools.common.status import load_status
from tools.pipeline import run
from tools.step00_check_frames import (
    check_object,
    demo_thresholds,
    evaluate_gates,
    main,
    summarize,
)
from tools.synthetic import write_orbit, write_video
from tools.check_env import CmdResult


PLAN = {
    "extract_fps": 4,
    "tag_family": "tag36h11",
    "sheet_every": 20,
    "orbits": [1, 2, 3],
    "frame_count_min": 150,
    "frame_count_max": 250,
    "laplacian_median_min": 100,
    "low_sharp_cutoff": 50,
    "low_sharp_fraction_max": 0.10,
    "tag_fraction_min": 0.80,
    "luminance_std_ratio_max": 0.05,
    "stills_expected": 5,
}


def _stats(**overrides):
    base = {
        "n_frames": 200,
        "laplacian_median": 150.0,
        "frac_below": 0.02,
        "luminance_mean": 120.0,
        "luminance_std": 2.0,
        "tag_fraction": 0.9,
    }
    base.update(overrides)
    return base


def test_plan_gates_match_the_inequalities():
    assert evaluate_gates(_stats(), PLAN) == {
        "frame_count": True,
        "sharpness": True,
        "tag": True,
        "exposure": True,
    }
    assert evaluate_gates(_stats(n_frames=149), PLAN)["frame_count"] is False
    assert evaluate_gates(_stats(n_frames=251), PLAN)["frame_count"] is False
    assert evaluate_gates(_stats(laplacian_median=100), PLAN)["sharpness"] is False
    assert evaluate_gates(_stats(frac_below=0.10), PLAN)["sharpness"] is False
    assert evaluate_gates(_stats(tag_fraction=0.79), PLAN)["tag"] is False
    assert evaluate_gates(_stats(luminance_std=6.0, luminance_mean=120.0), PLAN)["exposure"] is False
    assert evaluate_gates(_stats(tag_fraction=0.80), PLAN)["tag"] is True


def test_sharp_tagged_orbit_passes_demo_profile(tmp_path):
    write_orbit(tmp_path / "data" / "marker" / "raw" / "orbit1", n=8)
    code = check_object("marker", tmp_path, orbits=[1], thresholds=demo_thresholds(PLAN), profile="demo")
    assert code == 0
    doc = load_status("marker", tmp_path)
    orbit = doc["step00"]["orbits"]["orbit1"]
    assert orbit["n_frames"] == 8
    assert orbit["tag_fraction"] >= 0.80
    assert orbit["laplacian_median"] > 100
    assert orbit["passed"] is True
    sheet = tmp_path / "data" / "marker" / "renders" / "step00" / "orbit1_sheet.png"
    assert sheet.is_file()
    assert cv2.imread(str(sheet)) is not None
    step = next(item for item in doc["steps"] if item["id"] == "B1")
    assert step["state"] == "pass"


def test_strict_short_orbit_fails_frame_count(tmp_path):
    write_orbit(tmp_path / "data" / "marker" / "raw" / "orbit1", n=8)
    code = check_object("marker", tmp_path, orbits=[1], thresholds=PLAN, profile="plan")
    assert code == 1
    doc = load_status("marker", tmp_path)
    assert doc["step00"]["orbits"]["orbit1"]["gates"]["frame_count"] is False
    assert next(item for item in doc["steps"] if item["id"] == "B1")["state"] == "fail"


def test_blur_fails_sharpness_and_tag(tmp_path):
    write_orbit(tmp_path / "data" / "marker" / "raw" / "orbit1", n=4, blur=21)
    thresholds = demo_thresholds(PLAN)
    code = check_object("marker", tmp_path, orbits=[1], thresholds=thresholds, profile="demo")
    assert code == 1
    orbit = load_status("marker", tmp_path)["step00"]["orbits"]["orbit1"]
    assert orbit["gates"]["sharpness"] is False


def test_video_extracts_at_4_fps(tmp_path):
    video = tmp_path / "data" / "marker" / "raw" / "orbit1.avi"
    write_video(video, n=20, fps=20.0)
    thresholds = demo_thresholds(PLAN)
    code = check_object("marker", tmp_path, orbits=[1], thresholds=thresholds, profile="demo")
    assert code == 0
    report = json.loads(
        (tmp_path / "data" / "marker" / "reports" / "step00_orbit1.json").read_text(encoding="utf-8")
    )
    assert report["n_frames"] == 4
    assert summarize(report["frames"], 50)["n_frames"] == 4


def test_missing_orbit_fails(tmp_path):
    code = check_object("marker", tmp_path, orbits=[1, 2, 3], thresholds=PLAN)
    assert code == 1
    orbits = load_status("marker", tmp_path)["step00"]["orbits"]
    assert orbits["orbit2"]["missing_raw"] is True


def test_cli_demo(tmp_path):
    assert main(["--demo", "--root", str(tmp_path)]) == 0
    doc = load_status("_demo", tmp_path)
    assert doc["step00"]["profile"] == "demo"
    assert doc["step00"]["passed"] is True


def test_pipeline_stops_when_b0_fails(tmp_path):
    def runner(cmd, timeout):
        del timeout
        if cmd[0] == "nvidia-smi":
            return CmdResult(1, "", "no device")
        return CmdResult(1, "", "missing")

    code = run("marker", through="B1", root=tmp_path, runner=runner)
    assert code == 1
    doc = load_status("marker", tmp_path)
    assert next(step for step in doc["steps"] if step["id"] == "B0")["state"] == "fail"
    assert next(step for step in doc["steps"] if step["id"] == "B1")["state"] == "not_started"


def test_pipeline_rejects_later_block(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.check_object", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_splat_orbit", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_align", lambda *args, **kwargs: 0)
    video = tmp_path / "data" / "marker" / "raw" / "orbit1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    assert run("marker", through="B5", root=tmp_path) == 3
