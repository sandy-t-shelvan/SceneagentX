"""B0 environment probe.

On the GPU box this checks nvidia-smi, the seven micromamba envs, gsplat,
and that nerfstudio lists splatfacto. On a machine without that stack the
same command records the failure in status.json and exits 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Callable, Sequence

from tools.common.status import StatusError, begin_step, finish_step, load_status

ENVS: tuple[str, ...] = ("sa", "nerf", "gsam2", "2dgs", "partfield", "physgauss", "trellis")

# A 4090 reports about 24564 MiB. Treat 22 GiB and up as the plan's 24 GB class.
MIN_MEMORY_MIB = 22 * 1024
MIN_DRIVER = (550, 0, 0)


@dataclass
class CmdResult:
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[[Sequence[str], float], CmdResult]


def default_runner(cmd: Sequence[str], timeout: float) -> CmdResult:
    try:
        completed = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CmdResult(returncode=127, stdout="", stderr=str(exc))
    return CmdResult(completed.returncode, completed.stdout or "", completed.stderr or "")


def parse_driver(text: str) -> tuple[int, ...]:
    match = re.search(r"Driver Version:\s+([\d.]+)", text)
    if not match:
        return ()
    parts = []
    for piece in match.group(1).split("."):
        if piece.isdigit():
            parts.append(int(piece))
    return tuple(parts)


def driver_ok(version: tuple[int, ...]) -> bool:
    if not version:
        return False
    padded = version + (0, 0, 0)
    return padded[:3] >= MIN_DRIVER


def parse_nvidia_smi(text: str) -> dict:
    totals = [int(value) for value in re.findall(r"/\s*(\d+)\s*MiB", text)]
    memory = max(totals) if totals else 0
    driver = parse_driver(text)
    names = re.findall(r"\|\s+\d+\s+(NVIDIA .+?)\s+(?:On|Off)\s+\|", text)
    gpu_count = len(names) if names else (1 if memory and driver else 0)
    return {
        "gpu_count": gpu_count,
        "gpu_names": names,
        "driver": ".".join(str(part) for part in driver),
        "memory_mib": memory,
        "driver_ok": driver_ok(driver),
        "memory_ok": memory >= MIN_MEMORY_MIB,
    }


def probe_gpu(runner: Runner) -> dict:
    if runner is default_runner and shutil.which("nvidia-smi") is None:
        return {
            "gpu_count": 0,
            "gpu_names": [],
            "driver": "",
            "memory_mib": 0,
            "driver_ok": False,
            "memory_ok": False,
            "passed": False,
            "error": "nvidia-smi not on PATH",
        }
    result = runner(["nvidia-smi"], 20)
    parsed = parse_nvidia_smi(result.stdout + "\n" + result.stderr)
    if result.returncode != 0 and parsed["gpu_count"] == 0:
        parsed["error"] = (result.stderr or result.stdout or "nvidia-smi failed").strip()
    parsed["passed"] = bool(parsed["driver_ok"] and parsed["memory_ok"] and parsed["gpu_count"] >= 1)
    return parsed


def probe_torch_envs(runner: Runner) -> dict[str, bool]:
    found: dict[str, bool] = {}
    for env in ENVS:
        code = f"import torch; print('{env}', torch.cuda.is_available())"
        result = runner(["micromamba", "run", "-n", env, "python", "-c", code], 180)
        line = (result.stdout or "").strip().splitlines()
        token = line[-1].split()[-1] if line else ""
        found[env] = result.returncode == 0 and token == "True"
    return found


def probe_nerfstudio(runner: Runner) -> dict:
    gsplat = runner(["micromamba", "run", "-n", "nerf", "python", "-c", "import gsplat"], 180)
    help_text = runner(["micromamba", "run", "-n", "nerf", "ns-train", "--help"], 180)
    combined = help_text.stdout + "\n" + help_text.stderr
    return {
        "gsplat": gsplat.returncode == 0,
        "splatfacto_listed": help_text.returncode == 0 and "splatfacto" in combined,
    }


def setup_minutes_ok(minutes: float | None) -> bool | None:
    """None means setup.sh has not recorded a time yet, so this row is not failed."""
    if minutes is None:
        return None
    return minutes < 30


def probe(runner: Runner | None = None) -> dict:
    run = runner or default_runner
    if runner is not None:
        micromamba = "injected"
    else:
        micromamba = shutil.which("micromamba")
    gpu = probe_gpu(run)
    if micromamba is None:
        envs = {name: False for name in ENVS}
        nerf = {"gsplat": False, "splatfacto_listed": False, "error": "micromamba not on PATH"}
    else:
        envs = probe_torch_envs(run)
        nerf = probe_nerfstudio(run)
    envs_ok = all(envs.values())
    passed = bool(gpu.get("passed") and envs_ok and nerf.get("gsplat") and nerf.get("splatfacto_listed"))
    return {
        "passed": passed,
        "gpu": gpu,
        "envs": envs,
        "nerfstudio": nerf,
        "micromamba": micromamba is not None,
    }


def format_report(report: dict, minutes: float | None) -> str:
    gpu = report["gpu"]
    lines = [
        f"GPU count: {gpu.get('gpu_count')}  driver: {gpu.get('driver') or 'missing'}  "
        f"memory_mib: {gpu.get('memory_mib')}  pass: {gpu.get('passed')}",
    ]
    if gpu.get("error"):
        lines.append(f"GPU error: {gpu['error']}")
    for name in ENVS:
        lines.append(f"env {name}: cuda {report['envs'].get(name)}")
    nerf = report["nerfstudio"]
    lines.append(f"gsplat: {nerf.get('gsplat')}  splatfacto: {nerf.get('splatfacto_listed')}")
    if minutes is None:
        lines.append("setup_minutes: not recorded")
    else:
        lines.append(f"setup_minutes: {minutes}  under 30: {minutes < 30}")
    lines.append(f"B0: {'pass' if report['passed'] else 'fail'}")
    return "\n".join(lines)


def run_b0(obj: str, root: Path | None = None, runner: Runner | None = None) -> int:
    begin_step(obj, "B0", root)
    report = probe(runner)
    doc = load_status(obj, root)
    minutes = doc.get("setup_minutes")
    state = "pass" if report["passed"] else "fail"
    finish_step(obj, "B0", state, root, actual={"probe": report, "setup_minutes": minutes})
    sys.stdout.write(format_report(report, minutes) + "\n")
    sys.stdout.flush()
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the B0 GPU and micromamba stack")
    parser.add_argument("--obj", default="marker")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        return run_b0(args.obj, args.root)
    except (StatusError, OSError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
