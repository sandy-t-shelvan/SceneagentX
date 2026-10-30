from pathlib import Path

from tools.common.status import load_status
from tools.step02_splatfacto import count_gaussians, evaluate_splat, holdout_names, parse_eval, run_orbit

EVAL = '{"results": {"psnr": 29.4, "ssim": 0.93, "lpips": 0.08}}\n'


def _ply(path: Path, vertices: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"ply\nformat ascii 1.0\nelement vertex {vertices}\nproperty float x\nend_header\n", encoding="utf-8")


def test_holdout_is_every_tenth_name():
    names = [f"frame_{index:04d}.png" for index in range(1, 21)]
    assert holdout_names(names) == ["frame_0010.png", "frame_0020.png"]


def test_eval_and_gaussian_gates(tmp_path: Path):
    ply = tmp_path / "splat.ply"
    _ply(ply, 500_000)
    parsed = parse_eval(EVAL)
    assert parsed["psnr"] == 29.4
    assert parsed["ssim"] == 0.93
    assert count_gaussians(ply) == 500_000
    assert evaluate_splat(29.4, 0.93, 500_000) == {"psnr": True, "ssim": True, "gaussians": True}
    assert evaluate_splat(27.9, 0.93, 500_000)["psnr"] is False
    assert evaluate_splat(29.4, 0.89, 500_000)["ssim"] is False
    assert evaluate_splat(29.4, 0.93, 299_999)["gaussians"] is False
    assert evaluate_splat(29.4, 0.93, 3_000_001)["gaussians"] is False


def test_run_orbit_records_b3(tmp_path: Path):
    frames = tmp_path / "data" / "marker" / "frames" / "orbit1"
    frames.mkdir(parents=True)
    for index in range(1, 21):
        (frames / f"frame_{index:04d}.png").write_bytes(b"x")
    _ply(tmp_path / "data" / "marker" / "colmap" / "orbit1" / "splat.ply", 800_000)
    code = run_orbit("marker", 1, tmp_path, eval_text=EVAL + "minutes: 18.5\n", minutes=18.5)
    assert code == 0
    doc = load_status("marker", tmp_path)
    orbit = doc["step02"]["orbit1"]
    assert orbit["psnr"] == 29.4
    assert orbit["ssim"] == 0.93
    assert orbit["n_gaussians"] == 800_000
    assert orbit["minutes"] == 18.5
    assert orbit["passed"] is True
    holdout = (tmp_path / "data" / "marker" / "colmap" / "orbit1" / "holdout.txt").read_text(encoding="utf-8")
    assert "frame_0010.png" in holdout
    assert "frame_0001.png" not in holdout
    assert next(step for step in doc["steps"] if step["id"] == "B3")["state"] == "pass"
