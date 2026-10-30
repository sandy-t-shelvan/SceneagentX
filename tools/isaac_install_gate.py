"""Laptop decisions for the Isaac install. No SSH, no subprocess, no download.

Isaac Sim 5.1 and Isaac Lab 2.3 are installed only after Vulkan names a real
GPU, and only beside the seven existing environments, never inside them.
"""

from __future__ import annotations

import re

from tools.block_gates import ISAAC_LAB, ISAAC_SIM

SEVEN_ENVS = ("2dgs", "nerf", "partfield", "physgauss", "trellis", "sa", "gsam2")
MAMBA_ENV_ROOT = "/root/.local/share/mamba/envs"
ISAAC_VENV = "/root/jammy/opt/isaac"
LAB_STEPS = ("sim_pip", "import_isaacsim", "cmake", "lab_clone_2_3", "isaaclab_install")
CMAKE_RETRY_PIN = "cmake==3.22"

_DEVICE_LINE = re.compile(r"^\s*deviceName\s*=\s*(.+?)\s*$")


def pins() -> dict[str, str]:
    """The Sim/Lab pair pinned in block_gates; nothing new is hardcoded here."""
    return {"isaac_sim": ISAAC_SIM, "isaac_lab": ISAAC_LAB}


def parse_vulkan_devices(text: str) -> list[str]:
    """Every deviceName value in vulkaninfo output, in order."""
    return [m.group(1) for m in (_DEVICE_LINE.match(line) for line in text.splitlines()) if m]


def _is_llvmpipe(name: str) -> bool:
    """llvmpipe is the software renderer, not a GPU."""
    return "llvmpipe" in name.lower()


def decide_isaac_install(facts: dict) -> dict:
    """Install only with a non-llvmpipe Vulkan device, a glibc 2.35 path, and the NVIDIA EGL library."""
    devices = list(facts.get("vulkan_devices") or [])
    gpus = [d for d in devices if not _is_llvmpipe(d)]
    if not devices:
        return {"install": False, "reason": "vulkaninfo names no device"}
    if not gpus:
        return {"install": False, "reason": "Vulkan lists only llvmpipe: " + ", ".join(devices)}
    if not facts.get("glibc_at_least_2_35") and not facts.get("jammy_loader"):
        return {"install": False, "reason": "glibc below 2.35 and no Ubuntu 22.04 loader"}
    if not facts.get("egl_nvidia_present"):
        return {"install": False, "reason": "libEGL_nvidia.so.0 is absent"}
    return {"install": True, "reason": "Vulkan names " + gpus[0]}


def lab_install_order() -> tuple[str, ...]:
    """Sim pip, prove import isaacsim, cmake, clone Lab 2.3, then isaaclab install."""
    return LAB_STEPS


def validate_install_order(steps) -> tuple[str, ...]:
    """Reject any order that touches Lab before import isaacsim, or imports before Sim pip."""
    steps = tuple(steps)
    index = {name: i for i, name in enumerate(steps)}
    imp = index.get("import_isaacsim")
    for lab_step in ("lab_clone_2_3", "isaaclab_install"):
        if lab_step in index and (imp is None or index[lab_step] < imp):
            raise ValueError(f"{lab_step} before import_isaacsim")
    if imp is not None and (index.get("sim_pip") is None or imp < index["sim_pip"]):
        raise ValueError("import_isaacsim before sim_pip")
    return steps


def egl_probe_retry(attempts: int) -> dict:
    """Run once, retry exactly once with cmake 3.22, then stop."""
    if attempts <= 0:
        return {"action": "run"}
    if attempts == 1:
        return {"action": "retry", "pin": CMAKE_RETRY_PIN}
    return {"action": "stop"}


def env_pins_unchanged(before: dict[str, str], after: dict[str, str]) -> dict:
    """Name every one of the seven envs whose torch string moved or went missing."""
    moved = [env for env in SEVEN_ENVS if env not in before or env not in after or before[env] != after[env]]
    return {"unchanged": not moved, "moved": moved}


def isaac_venv_is_separate() -> bool:
    """The Isaac venv lives outside the micromamba root and matches none of the seven env paths."""
    root = MAMBA_ENV_ROOT.rstrip("/") + "/"
    env_paths = {root + env for env in SEVEN_ENVS}
    return not ISAAC_VENV.startswith(root) and ISAAC_VENV not in env_paths
