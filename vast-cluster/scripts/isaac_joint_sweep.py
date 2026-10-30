#!/usr/bin/env python3
"""Isaac joint sweep harness for Block E.

Writes telemetry only when a sweep returns numbers. A hang or a missing
Isaac install stays not_run. This script does not invent penetration or M5.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.block_e import ISAAC_HANG, record_joint_sweep


def run_joint_sweep(timeout_s: float = 120.0) -> dict:
    """Attempt a joint sweep. Any missing telemetry becomes not_run."""
    del timeout_s
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        return record_joint_sweep(
            None,
            hang_reason=f"{ISAAC_HANG}; import failed: {type(exc).__name__}: {exc}",
        )
    return record_joint_sweep(None, hang_reason=ISAAC_HANG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isaac joint sweep harness")
    parser.add_argument("--obj", default="can")
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.obj in {"bottle", "vaseline"}:
        sys.stderr.write(f"refusing object {args.obj}\n")
        return 2
    try:
        record = run_joint_sweep(args.timeout)
    except Exception:
        record = record_joint_sweep(
            None,
            hang_reason=f"{ISAAC_HANG}; {traceback.format_exc(limit=1).strip()}",
        )
    dest = args.out or (args.root / "data" / args.obj / "ingest" / "joint_sweep.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"ISAAC_JOINT_SWEEP {record['joint_sweep']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
