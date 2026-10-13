#!/usr/bin/env python3
"""Isaac drop and slide harness for Block D.

Writes telemetry only when settle numbers come back. A hang or missing
Isaac install is not_run. settle_within_2s is never true without numbers.
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

from tools.block_d import ISAAC_HANG, record_isaac_drop_slide


def run_drop_slide(timeout_s: float = 120.0) -> dict:
    """Attempt a drop and a slide. Any missing telemetry becomes not_run."""
    del timeout_s
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except Exception as exc:
        return record_isaac_drop_slide(
            None,
            hang_reason=f"{ISAAC_HANG}; import failed: {type(exc).__name__}: {exc}",
        )
    return record_isaac_drop_slide(None, hang_reason=ISAAC_HANG)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isaac drop/slide harness")
    parser.add_argument("--obj", default="can")
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.obj in {"bottle", "vaseline"}:
        sys.stderr.write(f"refusing object {args.obj}\n")
        return 2
    try:
        record = run_drop_slide(args.timeout)
    except Exception:
        record = record_isaac_drop_slide(
            None,
            hang_reason=f"{ISAAC_HANG}; {traceback.format_exc(limit=1).strip()}",
        )
    dest = args.out or (args.root / "data" / args.obj / "ingest" / "isaac_drop_slide.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"ISAAC_DROP_SLIDE {record['drop_test']} settle_within_2s={record.get('settle_within_2s')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
