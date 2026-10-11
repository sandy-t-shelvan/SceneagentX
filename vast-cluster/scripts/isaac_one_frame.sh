#!/bin/bash
# One 120s attempt to render scan.usd. Does not loop and does not invent a frame.
set -u
cd /root/SceneagentX
echo "OPT"
ls /opt
echo "NVIDIA"
ls /opt/nvidia 2>/dev/null || true
PY=""
if [ -x /opt/isaac/bin/python3 ]; then
  PY="/opt/isaac/bin/python3"
elif [ -f /opt/IsaacLab/isaaclab.sh ]; then
  PY="/opt/IsaacLab/isaaclab.sh -p"
else
  found=$(find /opt/nvidia /opt/conda /root -maxdepth 4 \( -name isaaclab.sh -o -name python.sh \) 2>/dev/null | head -n 1 || true)
  echo "FOUND ${found:-none}"
  if [ -n "$found" ]; then
    case "$found" in
      *isaaclab.sh) PY="$found -p" ;;
      *) PY="$found" ;;
    esac
  fi
fi
for candidate in \
  /root/isaacsim/python.sh \
  /root/isaac-sim/python.sh \
  /isaac-sim/python.sh \
  /opt/isaac-sim/python.sh \
  /opt/nvidia/isaac-sim/python.sh \
  /root/IsaacLab/_isaac_sim/python.sh \
  /root/isaaclab/_isaac_sim/python.sh \
  /root/IsaacSim/python.sh
 do
  if [ -f "$candidate" ]; then
    PY="$candidate"
    break
  fi
done
if [ -z "$PY" ] && [ -f /opt/IsaacLab/isaaclab.sh ]; then
  PY="/opt/IsaacLab/isaaclab.sh -p"
fi
if [ -z "$PY" ]; then
  PY=$(find /opt /root -maxdepth 3 -name isaaclab.sh 2>/dev/null | head -n 1 || true)
  if [ -n "$PY" ]; then
    PY="$PY -p"
  fi
fi
echo "ISAAC_PY ${PY:-missing}"
mkdir -p data/scan/inspect
set +e
if [ -n "$PY" ]; then
  timeout --signal=KILL 120 $PY /root/isaac_one_frame.py
  code=$?
else
  code=127
fi
set -e
echo "ISAAC_CODE ${code}"
python3 - << PY
import json
from pathlib import Path
folder = Path("/root/SceneagentX/data/scan")
path = folder / "isaac_check.json"
record = {}
if path.is_file():
    loaded = json.loads(path.read_text())
    if isinstance(loaded, dict):
        record = loaded
png = folder / "inspect" / "isaac_rgb.png"
has_frame = png.is_file() and png.stat().st_size > 0 and int("${code}") == 0
record["rendered"] = bool(has_frame)
record["png"] = str(png) if has_frame else None
record["alignment"] = "skipped"
record["exit_code"] = int("${code}")
if not has_frame:
    if int("${code}") == 124 or int("${code}") == 137:
        record["error"] = "sim.reset did not return within 120 seconds"
    elif int("${code}") == 139:
        record["error"] = "GPU helper exited with signal 11"
    elif int("${code}") == 127:
        record["error"] = "Isaac python was not found; no RGB frame was written"
    else:
        record["error"] = "Isaac exited " + str(int("${code}")) + " and no RGB frame was written"
    record["reason"] = record["error"]
else:
    record["error"] = None
    record["reason"] = "one RGB frame written"
path.write_text(json.dumps(record, indent=2) + "\n")
print("ISAAC_RENDERED", record["rendered"])
PY
echo ISAAC_FRAME_DONE
