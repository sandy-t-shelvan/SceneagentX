"""One D2 call for the bottle, using an existing orbit frame. Does not print secrets."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.orbit_blocks import fill_missing_d2

frames = Path("data/bottle/frames/orbit1")
preferred = frames / "frame_00113.png"
frame = preferred if preferred.is_file() else next(iter(sorted(frames.glob("*.png"))), None)
status = fill_missing_d2("bottle", Path("."), frame)
name = frame.name if frame is not None else "none"
print(f"D2_STATUS {status}")
print(f"D2_FRAME {name}")
