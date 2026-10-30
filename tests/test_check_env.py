from tools.check_env import (
    CmdResult,
    driver_ok,
    parse_nvidia_smi,
    probe,
    run_b0,
)
from tools.common.status import load_status

NVIDIA_4090 = """
NVIDIA-SMI 550.54.15              Driver Version: 550.54.15    CUDA Version: 12.4
|   0  NVIDIA GeForce RTX 4090        Off  |   00000000:01:00.0 Off |                  Off |
|  0%   35C    P8              15W /  450W |    200MiB /  24564MiB |      0%      Default |
"""


def test_parse_4090_meets_plan_gate():
    parsed = parse_nvidia_smi(NVIDIA_4090)
    assert parsed["gpu_count"] == 1
    assert parsed["memory_mib"] == 24564
    assert parsed["driver"] == "550.54.15"
    assert parsed["driver_ok"] is True
    assert parsed["memory_ok"] is True


def test_old_driver_and_small_card_fail():
    text = "Driver Version: 535.129.03\n|  100MiB /  8192MiB |"
    parsed = parse_nvidia_smi(text)
    assert driver_ok((535, 129, 3)) is False
    assert parsed["driver_ok"] is False
    assert parsed["memory_ok"] is False


def _stack(cuda_token: str, gsplat_code: int = 0, help_text: str = "method splatfacto\n"):
    def runner(cmd, timeout):
        del timeout
        joined = " ".join(cmd)
        if cmd[0] == "nvidia-smi":
            return CmdResult(0, NVIDIA_4090, "")
        if "import torch" in joined:
            env = cmd[cmd.index("-n") + 1]
            return CmdResult(0, f"{env} {cuda_token}\n", "")
        if "import gsplat" in joined:
            return CmdResult(gsplat_code, "", "")
        if "ns-train" in joined:
            return CmdResult(0, help_text, "")
        return CmdResult(1, "", joined)

    return runner


def test_probe_passes_when_stack_is_present():
    report = probe(_stack("True"))
    assert report["passed"] is True
    assert report["envs"]["partfield"] is True
    assert report["nerfstudio"]["splatfacto_listed"] is True


def test_probe_fails_when_an_env_misses_cuda():
    report = probe(_stack("False"))
    assert report["passed"] is False
    assert report["envs"]["sa"] is False


def test_run_b0_records_failure(tmp_path):
    code = run_b0("marker", tmp_path, _stack("False"))
    assert code == 1
    doc = load_status("marker", tmp_path)
    step = next(item for item in doc["steps"] if item["id"] == "B0")
    assert step["state"] == "fail"
    assert step["attempts"] == 1
    assert step["actual"]["probe"]["gpu"]["passed"] is True
