"""Find a video for the COLMAP path. A missing file does not start COLMAP."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json

from tools.common.paths import object_dir, raw_dir

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def find_video(obj: str, root: Path | None = None) -> Path | None:
    raw = raw_dir(obj, root)
    if not raw.is_dir():
        return None
    for suffix in VIDEO_SUFFIXES:
        path = raw / f"orbit1{suffix}"
        if path.is_file():
            return path
    return None


def record_absent(obj: str, root: Path | None = None) -> Path:
    folder = object_dir(obj, root)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "video_path.json"
    path.write_text(
        json.dumps(
            {
                "provided": False,
                "colmap": "not_started",
                "reason": "No data/<obj>/raw/orbit1 video was provided.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report whether a video is ready for the Vast COLMAP path")
    parser.add_argument("--obj", default="scan")
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args(argv)
    found = find_video(args.obj, args.root)
    if found is None:
        path = record_absent(args.obj, args.root)
        sys.stdout.write(f"no video provided; COLMAP not started; wrote {path}\n")
        return 0
    sys.stdout.write(f"video {found}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
