"""Read the running Vast instance from the CLI.

Do not cache an IP or port. A replaced instance keeps the old address closed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class VastError(RuntimeError):
    """The CLI returned nothing this machine can SSH to."""


@dataclass(frozen=True)
class VastHost:
    instance_id: int
    status: str
    host: str
    port: int


def vastai_executable() -> str:
    local = Path(__file__).resolve().parents[1] / ".venv" / "Scripts" / "vastai.exe"
    if local.is_file():
        return str(local)
    found = shutil.which("vastai")
    if found:
        return found
    raise VastError("vastai is not installed")


def list_instances() -> list[dict]:
    completed = subprocess.run(
        [vastai_executable(), "show", "instances", "--raw"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise VastError(completed.stderr.strip() or "vastai show instances failed")
    payload = json.loads(completed.stdout)
    if not isinstance(payload, list):
        raise VastError("vastai did not return a list of instances")
    return payload


def current_host(instances: list[dict] | None = None) -> VastHost:
    rows = list_instances() if instances is None else instances
    if not rows:
        raise VastError("no Vast instance is listed")
    running = [row for row in rows if str(row.get("actual_status")) == "running"]
    chosen = running[0] if running else rows[0]
    host = str(chosen.get("public_ipaddr") or "")
    port = int(chosen.get("direct_port_start") or 0)
    if not host or port <= 0:
        raise VastError("the listed instance has no direct SSH port")
    return VastHost(
        instance_id=int(chosen["id"]),
        status=str(chosen.get("actual_status") or ""),
        host=host,
        port=port,
    )


def start_instance(instance_id: int) -> str:
    completed = subprocess.run(
        [vastai_executable(), "start", "instance", str(instance_id)],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (completed.stdout or completed.stderr).strip()
    if completed.returncode != 0 and "queued" not in text.lower():
        raise VastError(text or "vastai start instance failed")
    return text
