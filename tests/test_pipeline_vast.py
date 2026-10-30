"""B0, COLMAP, and splatfacto use the running Vast host. The laptop keeps B1."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.check_env import CmdResult
from tools.common.status import load_status, save_status, set_path
from tools.pipeline import VastRunner, run
from tools.step02_splatfacto import run_orbit
from tools.vast_host import VastHost

HOST = VastHost(instance_id=9, status="running", host="203.0.113.10", port=2222)

NVIDIA_4090 = """
NVIDIA-SMI 550.54.15              Driver Version: 550.54.15    CUDA Version: 12.4
|   0  NVIDIA GeForce RTX 4090        Off  |   00000000:01:00.0 Off |                  Off |
|  0%   35C    P8              15W /  450W |    200MiB /  24564MiB |      0%      Default |
"""

PLY = b"ply\nformat ascii 1.0\nelement vertex 800000\nproperty float x\nend_header\n"


def _key(tmp_path: Path, monkeypatch) -> Path:
    key = tmp_path / "ssh_sceneagentX_1"
    key.write_text("key", encoding="utf-8")
    monkeypatch.setattr("tools.pipeline.ssh_key_path", lambda: key)
    return key


def _ban_local_gpu(monkeypatch) -> None:
    def which(name: str):
        if name == "nvidia-smi":
            raise AssertionError("local nvidia-smi")
        return None

    def local_run(*args, **kwargs):
        raise AssertionError(args[0] if args else kwargs)

    monkeypatch.setattr("tools.check_env.shutil.which", which)
    monkeypatch.setattr("tools.check_env.subprocess.run", local_run)


def _patch_remote(monkeypatch, handler):
    """Replace only ssh and scp. Git and other local commands keep working."""
    real = subprocess.run

    def wrapped(argv, *args, **kwargs):
        program = argv[0] if isinstance(argv, (list, tuple)) and argv else ""
        if program not in {"ssh", "scp"}:
            return real(argv, *args, **kwargs)
        return handler(list(argv))

    monkeypatch.setattr("tools.pipeline.subprocess.run", wrapped)


def _stage_copy(argv: list[str]) -> subprocess.CompletedProcess:
    dest = argv[-1]
    if not str(dest).startswith("root@"):
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        if "-r" in argv:
            if path.exists():
                if path.is_dir():
                    return subprocess.CompletedProcess(argv, 0, "", "")
                path.unlink()
            path.mkdir()
            (path / "0").mkdir()
        else:
            path.write_bytes(b"splat-from-vast\n")
    return subprocess.CompletedProcess(argv, 0, "", "")


def test_b0_uses_the_vast_host_and_does_not_call_local_nvidia_smi(tmp_path: Path, monkeypatch, capsys):
    key = _key(tmp_path, monkeypatch)
    _ban_local_gpu(monkeypatch)
    seen = {"lookup": 0, "ssh": []}

    def lookup():
        seen["lookup"] += 1
        return HOST

    def remote_run(argv):
        if argv[0] == "nvidia-smi":
            raise AssertionError("local nvidia-smi")
        assert argv[0] == "ssh"
        seen["ssh"].append(argv)
        remote = argv[-1]
        if "import torch" in remote:
            stdout = "True\n"
        elif "import gsplat" in remote:
            stdout = ""
        elif "ns-train" in remote:
            stdout = "splatfacto\n"
        else:
            stdout = NVIDIA_4090
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr("tools.pipeline.running_vast_host", lookup)
    _patch_remote(monkeypatch, remote_run)
    code = run("vaseline", through="B0", root=tmp_path)
    out, err = capsys.readouterr()
    assert code == 0, out + err
    assert seen["lookup"] == 1
    assert seen["ssh"]
    assert all(argv[0] == "ssh" for argv in seen["ssh"])
    first = seen["ssh"][0]
    assert f"root@{HOST.host}" in first
    assert str(HOST.port) in first
    assert str(key) in first
    remote = first[-1]
    assert "cd /root/SceneagentX" in remote
    assert "MAMBA_ROOT_PREFIX=/root/.local/share/mamba" in remote
    assert "/root/.local/share/mamba/envs/nerf/bin" in remote
    assert "nvidia-smi" in remote
    joined = "\n".join(argv[-1] for argv in seen["ssh"])
    for env in ("sa", "nerf", "gsam2", "2dgs", "partfield", "physgauss", "trellis"):
        assert f"'{env}'" in joined
    step = next(item for item in load_status("vaseline", tmp_path)["steps"] if item["id"] == "B0")
    assert step["state"] == "pass"


def test_no_vast_instance_fails_b0_without_a_laptop_gpu(tmp_path: Path, monkeypatch, capsys):
    _ban_local_gpu(monkeypatch)
    seen = {"lookup": 0}

    def lookup():
        seen["lookup"] += 1
        return None

    def remote_run(argv):
        raise AssertionError(argv)

    monkeypatch.setattr("tools.pipeline.running_vast_host", lookup)
    _patch_remote(monkeypatch, remote_run)
    code = run("vaseline", through="B1", root=tmp_path)
    err = capsys.readouterr().err
    assert code == 1
    assert seen["lookup"] == 1
    assert "no running Vast instance" in err
    assert "Stopped at B0" in err
    doc = load_status("vaseline", tmp_path)
    failed = next(item for item in doc["steps"] if item["id"] == "B0")
    assert failed["state"] == "fail"
    assert failed["actual"]["error"] == "no running Vast instance"
    assert next(item for item in doc["steps"] if item["id"] == "B1")["state"] == "not_started"


def test_injected_runner_does_not_look_up_vast(tmp_path: Path, monkeypatch):
    _ban_local_gpu(monkeypatch)
    seen = []

    def lookup():
        raise AssertionError("vast lookup")

    def runner(cmd, timeout):
        del timeout
        seen.append(list(cmd))
        if cmd[0] == "nvidia-smi":
            return CmdResult(1, "", "no device")
        return CmdResult(1, "", "missing")

    monkeypatch.setattr("tools.pipeline.running_vast_host", lookup)
    code = run("marker", through="B0", root=tmp_path, runner=runner)
    assert code == 1
    assert seen[0] == ["nvidia-smi"]
    assert all(cmd[0] != "ssh" for cmd in seen)
    step = next(item for item in load_status("marker", tmp_path)["steps"] if item["id"] == "B0")
    assert step["state"] == "fail"


def test_remote_runner_rewrites_data_paths_and_copies_products(tmp_path: Path, monkeypatch):
    key = _key(tmp_path, monkeypatch)
    calls = []

    def remote_run(argv):
        calls.append(argv)
        if argv[0] == "scp":
            return _stage_copy(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    _patch_remote(monkeypatch, remote_run)
    runner = VastRunner(HOST, tmp_path)
    video = tmp_path / "data" / "vaseline" / "raw" / "orbit1.mp4"
    output = tmp_path / "data" / "vaseline" / "colmap" / "orbit1"
    result = runner(
        ["ns-process-data", "video", "--data", str(video), "--output-dir", str(output)],
        30,
    )
    assert result.returncode == 0
    remote = calls[0][-1]
    assert calls[0][0] == "ssh"
    assert str(key) in calls[0]
    assert f"root@{HOST.host}" in calls[0]
    assert "/root/SceneagentX/data/vaseline/raw/orbit1.mp4" in remote
    assert "/root/SceneagentX/data/vaseline/colmap/orbit1" in remote
    assert "/root/.local/share/mamba/envs/nerf" in remote
    sparse = next(argv for argv in calls if argv[0] == "scp" and "-r" in argv)
    assert sparse[-2].endswith("/colmap/orbit1/sparse")
    assert (output / "sparse" / "0").is_dir()

    calls.clear()
    trained = runner(["ns-train", "splatfacto", "--data", str(output)], 30)
    assert trained.returncode == 0
    assert runner.copied_splat is True
    ply_copy = next(argv for argv in calls if argv[0] == "scp" and argv[-2].endswith("/splat.ply"))
    assert "root@203.0.113.10:" in ply_copy[-2]
    assert (output / "splat.ply").read_bytes() == b"splat-from-vast\n"


def test_missing_raw_video_is_uploaded_before_colmap_and_products_come_back(tmp_path: Path, monkeypatch):
    _key(tmp_path, monkeypatch)
    video = tmp_path / "data" / "vaseline" / "raw" / "orbit1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    events = []

    def remote_run(argv):
        if argv[0] == "ssh":
            shell = argv[-1]
            if shell.startswith("test "):
                events.append("missing")
                return subprocess.CompletedProcess(argv, 1, "", "")
            if shell.startswith("mkdir"):
                events.append("mkdir")
                assert "/root/SceneagentX/data/vaseline/raw" in shell
                return subprocess.CompletedProcess(argv, 0, "", "")
            raise AssertionError(shell)
        if argv[0] == "scp":
            if str(argv[-1]).startswith("root@"):
                events.append("upload")
                assert argv[-1].endswith("/data/vaseline/raw/orbit1.mp4")
                assert str(video) == argv[-2]
            else:
                events.append("download:" + Path(argv[-1]).name)
            return _stage_copy(argv)
        raise AssertionError(argv[0])

    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: HOST)
    _patch_remote(monkeypatch, remote_run)
    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: events.append("b0") or 0)
    monkeypatch.setattr("tools.pipeline.check_object", lambda *args, **kwargs: events.append("b1") or 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", lambda *args, **kwargs: events.append("colmap") or 0)
    monkeypatch.setattr("tools.pipeline.run_splat_orbit", lambda *args, **kwargs: events.append("splat") or 0)
    code = run("vaseline", through="B3", root=tmp_path)
    assert code == 0
    assert events.index("b1") < events.index("upload") < events.index("colmap")
    assert events.index("colmap") < events.index("splat") < events.index("download:splat.ply.partial")
    assert "download:sparse.partial" in events
    orbit = tmp_path / "data" / "vaseline" / "colmap" / "orbit1"
    assert (orbit / "splat.ply").read_bytes() == b"splat-from-vast\n"
    assert (orbit / "sparse" / "0").is_dir()
    assert not (orbit / "splat.ply.partial").exists()


def test_a_raw_video_already_on_the_host_is_not_uploaded(tmp_path: Path, monkeypatch):
    _key(tmp_path, monkeypatch)
    video = tmp_path / "data" / "can" / "raw" / "orbit1.mov"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    uploads = []

    def remote_run(argv):
        if argv[0] == "ssh" and argv[-1].startswith("test "):
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[0] == "scp" and str(argv[-1]).startswith("root@"):
            uploads.append(argv)
        if argv[0] == "scp":
            return _stage_copy(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: HOST)
    _patch_remote(monkeypatch, remote_run)
    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.check_object", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_splat_orbit", lambda *args, **kwargs: 0)
    assert run("can", through="B3", root=tmp_path) == 0
    assert uploads == []


def test_passing_checkpoint_is_not_retrained(tmp_path: Path, capsys):
    ply = tmp_path / "data" / "marker" / "colmap" / "orbit1" / "splat.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(PLY)
    doc = load_status("marker", tmp_path)
    set_path(doc, "step02.orbit1", {"passed": True, "psnr": 29.4, "ssim": 0.93, "n_gaussians": 800000})
    save_status("marker", doc, tmp_path)

    def runner(cmd, timeout):
        del timeout
        raise AssertionError(cmd)

    assert run_orbit("marker", 1, tmp_path, runner=runner) == 0
    assert "orbit1 passing checkpoint kept" in capsys.readouterr().out
    assert ply.read_bytes() == PLY
    kept = load_status("marker", tmp_path)
    assert kept["step02"]["orbit1"]["psnr"] == 29.4
    assert kept["step02"]["orbit1"]["n_gaussians"] == 800000
    assert next(item for item in kept["steps"] if item["id"] == "B3")["attempts"] == 0


def test_a_failed_splat_still_trains(tmp_path: Path):
    ply = tmp_path / "data" / "marker" / "colmap" / "orbit1" / "splat.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(PLY)
    doc = load_status("marker", tmp_path)
    set_path(doc, "step02.orbit1", {"passed": False, "psnr": 10.0, "ssim": 0.1, "n_gaussians": 800000})
    save_status("marker", doc, tmp_path)
    seen = []

    def runner(cmd, timeout):
        del timeout
        seen.append(list(cmd))
        return CmdResult(1, "", "train failed")

    assert run_orbit("marker", 1, tmp_path, runner=runner) == 1
    assert seen[0][0] == "ns-train"


def test_pipeline_keeps_a_passing_checkpoint(tmp_path: Path, monkeypatch, capsys):
    _key(tmp_path, monkeypatch)
    video = tmp_path / "data" / "vaseline" / "raw" / "orbit1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")
    ply = tmp_path / "data" / "vaseline" / "colmap" / "orbit1" / "splat.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(PLY)
    doc = load_status("vaseline", tmp_path)
    set_path(doc, "step02.orbit1", {"passed": True, "psnr": 31.0, "ssim": 0.95, "n_gaussians": 800000})
    save_status("vaseline", doc, tmp_path)
    commands = []

    def remote_run(argv):
        commands.append(argv)
        if argv[0] == "scp":
            raise AssertionError(argv)
        shell = argv[-1]
        if "ns-train" in shell or "ns-process-data" in shell:
            raise AssertionError(shell)
        if shell.startswith("test "):
            return subprocess.CompletedProcess(argv, 0, "", "")
        raise AssertionError(shell)

    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: HOST)
    _patch_remote(monkeypatch, remote_run)
    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.check_object", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", lambda *args, **kwargs: 0)
    code = run("vaseline", through="B3", root=tmp_path)
    out, err = capsys.readouterr()
    assert code == 0, out + err
    assert "orbit1 passing checkpoint kept" in out
    assert ply.read_bytes() == PLY
    assert load_status("vaseline", tmp_path)["step02"]["orbit1"]["psnr"] == 31.0
    assert commands
    assert all(argv[0] == "ssh" and argv[-1].startswith("test ") for argv in commands)
