"""Print a Vast.ai on-demand create command. Does not rent, and does not store an API key."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import subprocess
from typing import Any

DISK_GB = 250
IMAGE = "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel"
MIN_CORES = 24
MIN_DRIVER = (580, 0, 0)

QUERY = (
    "gpu_name=RTX_4090 num_gpus=1 gpu_ram>=24 cpu_cores>=24 cpu_ram>=64 "
    "direct_port_count>=1 disk_space>=250 reliability>0.98 cuda_vers>=12.1 "
    "driver_version >= 580.00.00"
)


def create_command(offer_id: int) -> str:
    return (
        f"vastai create instance {offer_id} --image {IMAGE} "
        f"--disk {DISK_GB} --ssh --direct --label sceneagent --cancel-unavail"
    )


def effective_cores(offer: dict[str, Any]) -> float:
    if offer.get("cpu_cores_effective") is not None:
        return float(offer["cpu_cores_effective"])
    return float(offer.get("cpu_cores") or 0)


def gpu_ram_gb(offer: dict[str, Any]) -> float:
    ram = float(offer.get("gpu_ram") or 0)
    if ram > 200:
        return ram / 1024.0
    return ram


def driver_tuple(value: Any) -> tuple[int, int, int]:
    parts = [int(piece) for piece in str(value).split(".") if piece.isdigit()]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def price(offer: dict[str, Any]) -> float:
    for key in ("dph_total", "dph", "discountedTotalPerHour"):
        if offer.get(key) is not None:
            return float(offer[key])
    return 1e9


def matches(offer: dict[str, Any]) -> bool:
    name = str(offer.get("gpu_name") or "").replace(" ", "_")
    if "4090" not in name:
        return False
    if int(offer.get("num_gpus") or 1) != 1:
        return False
    if effective_cores(offer) < MIN_CORES:
        return False
    if gpu_ram_gb(offer) < 22:
        return False
    ram = float(offer.get("cpu_ram") or 0)
    ram_gb = ram / 1024.0 if ram >= 1000 else ram
    if ram_gb < 64:
        return False
    if float(offer.get("disk_space") or 0) < DISK_GB:
        return False
    if int(offer.get("direct_port_count") or 0) < 1:
        return False
    if driver_tuple(offer.get("driver_version") or "0") < MIN_DRIVER:
        return False
    return True


def select_offer(offers: list[dict[str, Any]]) -> dict[str, Any] | None:
    fitting = [offer for offer in offers if matches(offer)]
    if not fitting:
        return None
    return sorted(fitting, key=price)[0]


def search_live() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "vastai",
            "search",
            "offers",
            "--on-demand",
            "--storage",
            str(DISK_GB),
            "--raw",
            "-o",
            "dph",
            QUERY,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "vastai search failed")
    payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        payload = payload.get("offers", [])
    if not isinstance(payload, list):
        raise RuntimeError("vastai search did not return a list of offers")
    return payload


def format_choice(offer: dict[str, Any]) -> str:
    offer_id = int(offer["id"])
    return "\n".join(
        [
            f"offer {offer_id}",
            f"gpu {offer.get('gpu_name')}  cores {effective_cores(offer):.0f}  "
            f"ram_gb {gpu_ram_gb(offer):.1f}  disk {offer.get('disk_space')}  "
            f"driver {offer.get('driver_version')}  $/hr {price(offer):.4f}",
            create_command(offer_id),
            "This command is not run here. It starts billing only after you execute it.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find one on-demand RTX 4090 and print the create command")
    parser.add_argument("--fixture", type=Path, default=None, help="Offer JSON. Skips the network.")
    args = parser.parse_args(argv)
    try:
        if args.fixture:
            offers = json.loads(args.fixture.read_text(encoding="utf-8"))
        else:
            offers = search_live()
    except (OSError, json.JSONDecodeError, RuntimeError, KeyError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    chosen = select_offer(offers)
    if chosen is None:
        sys.stderr.write("No on-demand RTX 4090 matched 24 cores, 64 GB RAM, 250 GB disk, and driver 580.\n")
        return 1
    sys.stdout.write(format_choice(chosen) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
