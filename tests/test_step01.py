from pathlib import Path

import numpy as np

from tools.common.status import load_status
from tools.check_env import CmdResult
from tools.step01_colmap import _run_process, evaluate_colmap, parse_analyzer, run_orbit
from tools.viz_cameras import far_camera_count, plot_cameras, read_images_txt

ANALYZER = """
num_images: 200
num_reg_images: 190
mean_reprojection_error: 0.62
"""


def _pose(image_id: int, center: tuple[float, float, float]) -> str:
    tx, ty, tz = (-center[0], -center[1], -center[2])
    return f"{image_id} 1 0 0 0 {tx} {ty} {tz} 1 cam{image_id}.png\n0 0 -1\n"


def _ring(start_id: int, radius: float, z_value: float, count: int = 8) -> str:
    lines = []
    for index in range(count):
        angle = 2 * np.pi * index / count
        center = (radius * float(np.cos(angle)), radius * float(np.sin(angle)), z_value)
        lines.append(_pose(start_id + index, center))
    return "".join(lines)


def test_analyzer_percent_and_gates():
    metrics = parse_analyzer(ANALYZER)
    assert metrics["registered_pct"] == 95.0
    assert metrics["mean_reproj_px"] == 0.62
    assert evaluate_colmap(metrics, ["0"]) == {
        "registered": True,
        "reprojection": True,
        "one_model": True,
        "cameras": True,
    }
    assert evaluate_colmap(metrics, ["0", "1"])["one_model"] is False
    assert evaluate_colmap({"registered_pct": 89.9, "mean_reproj_px": 0.5}, ["0"])["registered"] is False
    assert evaluate_colmap({"registered_pct": 95, "mean_reproj_px": 1.0}, ["0"])["reprojection"] is False


def test_two_rings_plot_and_colmap_pass(tmp_path: Path):
    sparse = tmp_path / "data" / "marker" / "colmap" / "orbit1" / "sparse" / "0"
    sparse.mkdir(parents=True)
    images = sparse / "images.txt"
    images.write_text(_ring(1, 1.0, 0.0) + _ring(9, 1.0, 0.3), encoding="utf-8")
    centers = read_images_txt(images)
    assert len(centers) == 16
    assert far_camera_count(centers) == 0
    code = run_orbit("marker", 1, tmp_path, analyzer_text=ANALYZER)
    assert code == 0
    doc = load_status("marker", tmp_path)
    orbit = doc["step01"]["orbit1"]
    assert orbit["registered_pct"] == 95.0
    assert orbit["mean_reproj_px"] == 0.62
    assert orbit["n_models"] == 1
    assert orbit["passed"] is True
    assert (tmp_path / "data" / "marker" / "renders" / "step01" / "orbit1_cameras.png").is_file()
    assert next(step for step in doc["steps"] if step["id"] == "B2")["state"] == "pass"


def test_frame_matching_is_sequential(tmp_path: Path):
    calls = []

    def runner(cmd, timeout):
        calls.append(cmd)
        return CmdResult(0, "", "")

    frames = tmp_path / "frames"
    frames.mkdir()
    _run_process("images", frames, tmp_path / "out", runner)
    assert calls[0][0:2] == ["ns-process-data", "images"]
    assert "--matching-method" in calls[0]
    assert calls[0][calls[0].index("--matching-method") + 1] == "sequential"


def test_far_camera_is_flagged(tmp_path: Path):
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [50.0, 0.0, 0.0]])
    assert far_camera_count(centers) == 1
    assert plot_cameras(centers, tmp_path / "cameras.png") == 1
