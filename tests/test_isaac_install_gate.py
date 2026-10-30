"""Isaac install gate. llvmpipe is not a GPU, Sim comes before Lab, and the seven torch pins do not move."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.block_gates import ISAAC_LAB, ISAAC_SIM, record_isaac_timeout
from tools.isaac_install_gate import (
    ISAAC_VENV,
    MAMBA_ENV_ROOT,
    SEVEN_ENVS,
    decide_isaac_install,
    egl_probe_retry,
    env_pins_unchanged,
    isaac_venv_is_separate,
    lab_install_order,
    parse_vulkan_devices,
    pins,
    validate_install_order,
)

LLVMPIPE_ONLY = """
VkPhysicalDeviceProperties:
---------------------------
        apiVersion     = 1.3.255
        deviceName     = llvmpipe (LLVM 12.0.0, 256 bits)
"""

GPU_AND_LLVMPIPE = """
GPU0:
        apiVersion     = 1.4.329
        deviceName     = NVIDIA GeForce RTX 4090
GPU1:
        apiVersion     = 1.3.255
        deviceName     = llvmpipe (LLVM 12.0.0, 256 bits)
"""


def test_parse_vulkan_devices():
    assert parse_vulkan_devices(LLVMPIPE_ONLY) == ["llvmpipe (LLVM 12.0.0, 256 bits)"]
    assert parse_vulkan_devices(GPU_AND_LLVMPIPE) == [
        "NVIDIA GeForce RTX 4090",
        "llvmpipe (LLVM 12.0.0, 256 bits)",
    ]
    assert parse_vulkan_devices("") == []


def test_decide_isaac_install():
    llvm = decide_isaac_install(
        {
            "vulkan_devices": parse_vulkan_devices(LLVMPIPE_ONLY),
            "glibc_at_least_2_35": False,
            "jammy_loader": True,
            "egl_nvidia_present": True,
        }
    )
    assert llvm["install"] is False
    assert "llvmpipe" in llvm["reason"]
    empty = decide_isaac_install({"vulkan_devices": [], "jammy_loader": True, "egl_nvidia_present": True})
    assert empty["install"] is False
    gpu = {
        "vulkan_devices": parse_vulkan_devices(GPU_AND_LLVMPIPE),
        "glibc_at_least_2_35": False,
        "jammy_loader": True,
        "egl_nvidia_present": True,
    }
    assert decide_isaac_install(gpu) == {"install": True, "reason": "Vulkan names NVIDIA GeForce RTX 4090"}
    assert decide_isaac_install({**gpu, "egl_nvidia_present": False})["install"] is False
    assert decide_isaac_install({**gpu, "jammy_loader": False})["install"] is False
    assert decide_isaac_install({**gpu, "jammy_loader": False, "glibc_at_least_2_35": True})["install"] is True


def test_lab_install_order():
    order = lab_install_order()
    assert order == ("sim_pip", "import_isaacsim", "cmake", "lab_clone_2_3", "isaaclab_install")
    assert validate_install_order(order) == order
    with pytest.raises(ValueError, match="before import_isaacsim"):
        validate_install_order(("sim_pip", "lab_clone_2_3", "isaaclab_install", "import_isaacsim"))
    with pytest.raises(ValueError, match="before import_isaacsim"):
        validate_install_order(("sim_pip", "cmake", "isaaclab_install"))
    with pytest.raises(ValueError, match="before sim_pip"):
        validate_install_order(("import_isaacsim", "sim_pip", "cmake", "lab_clone_2_3", "isaaclab_install"))


def test_egl_probe_retry():
    assert egl_probe_retry(0) == {"action": "run"}
    assert egl_probe_retry(1) == {"action": "retry", "pin": "cmake==3.22"}
    assert egl_probe_retry(2) == {"action": "stop"}
    assert egl_probe_retry(3) == {"action": "stop"}


def test_env_pins_unchanged():
    before = {
        "2dgs": "2.1.2+cu121",
        "nerf": "2.1.2+cu118",
        "partfield": "2.4.0+cu121",
        "physgauss": "2.1.2+cu121",
        "trellis": "2.4.0+cu121",
        "sa": "2.5.1+cu124",
        "gsam2": "2.3.1+cu121",
    }
    assert env_pins_unchanged(before, dict(before)) == {"unchanged": True, "moved": []}
    moved = env_pins_unchanged(before, {**before, "2dgs": "2.7.0+cu128"})
    assert moved == {"unchanged": False, "moved": ["2dgs"]}
    missing = dict(before)
    del missing["sa"]
    assert env_pins_unchanged(before, missing) == {"unchanged": False, "moved": ["sa"]}


def test_isaac_venv_is_separate():
    assert SEVEN_ENVS == ("2dgs", "nerf", "partfield", "physgauss", "trellis", "sa", "gsam2")
    assert isaac_venv_is_separate() is True
    assert not ISAAC_VENV.startswith(MAMBA_ENV_ROOT)
    assert ISAAC_VENV not in {f"{MAMBA_ENV_ROOT}/{env}" for env in SEVEN_ENVS}


def test_pins_match_block_gates():
    assert pins() == {"isaac_sim": ISAAC_SIM, "isaac_lab": ISAAC_LAB}
    assert pins() == {"isaac_sim": "5.1", "isaac_lab": "2.3"}


def test_hang_leaves_no_png(tmp_path: Path):
    record = record_isaac_timeout(tmp_path)
    assert record["isaac"] is False
    assert record["png"] is None
    assert not (tmp_path / "inspect" / "frame_000.png").exists()
