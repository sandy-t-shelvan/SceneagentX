#!/bin/bash
# One timed Isaac attempt. A timeout or a failed start removes any picture.
set -u
PNG=/root/SceneagentX/data/bottle/inspect/frame_000.png
NOTE=/root/SceneagentX/data/bottle/isaac_check.json
mkdir -p /root/SceneagentX/data/bottle/inspect
rm -f "$PNG"
export TERM=xterm
export OMNI_KIT_ACCEPT_EULA=YES
if ! mountpoint -q /root/jammy/proc; then
  mount --bind /proc /root/jammy/proc || true
fi
if ! mountpoint -q /root/jammy/dev; then
  mount --bind /dev /root/jammy/dev || true
fi
if ! mountpoint -q /root/jammy/tmp; then
  mkdir -p /root/jammy/tmp
  mount --bind /tmp /root/jammy/tmp || true
fi
mkdir -p /root/jammy/root/SceneagentX
if ! mountpoint -q /root/jammy/root/SceneagentX; then
  mount --bind /root/SceneagentX /root/jammy/root/SceneagentX || true
fi
code=127
if [ -x /root/jammy/usr/bin/python3.11 ]; then
  probe=$(timeout --signal=KILL 60 chroot /root/jammy /opt/isaac/bin/python -c 'import importlib.util; spec=importlib.util.find_spec("isaaclab"); print("ISAACLAB_SPEC", spec.origin if spec else "missing")' 2>&1) || true
  echo "$probe"
  if ! printf '%s' "$probe" | grep -q 'ISAACLAB_SPEC /'; then
    code=127
  else
  timeout --signal=KILL 600 chroot /root/jammy /opt/isaac/bin/python /root/isaac_bottle_frame.py \
    --headless \
    --usda /root/SceneagentX/data/bottle/scene.usda \
    --png /root/SceneagentX/data/bottle/inspect/frame_000.png \
    --pose /root/SceneagentX/data/bottle/2dgs/frame_00113_pose.json
  code=$?
  fi
fi
python3 - << PY
import json
from pathlib import Path
png = Path("/root/SceneagentX/data/bottle/inspect/frame_000.png")
code = int("${code}")
ok = png.is_file() and png.stat().st_size > 0 and code == 0
if not ok and png.is_file():
    png.unlink()
record = {
    "installed": True,
    "isaac_sim": "5.1.0-rc.19",
    "isaac_lab": "2.3.0",
    "rendered": bool(ok),
    "isaac": bool(ok),
    "png": "data/bottle/inspect/frame_000.png" if ok else None,
    "fake_image": False,
    "exit_code": code,
    "pose": "frame_00113",
    "host": "184.144.154.180",
    "instance_id": 55073623,
}
if not ok:
    if code in (124, 137):
        record["error"] = "timeout"
        record["reason"] = "Isaac did not return a frame within 600 seconds"
    elif code == 127:
        record["error"] = "Isaac python was not found"
        record["reason"] = "Isaac Sim 5.1 files are on the host but the interpreter did not start"
    else:
        record["error"] = f"Isaac exited {code} and no RGB frame was written"
        record["reason"] = record["error"]
else:
    record["error"] = None
    record["reason"] = "frame_000.png written from the frame_00113 COLMAP pose"
Path("/root/SceneagentX/data/bottle/isaac_check.json").write_text(json.dumps(record, indent=2) + "\n")
print("ISAAC_RESULT", json.dumps({"rendered": record["rendered"], "code": code, "error": record.get("error")}))
PY
echo ISAAC_BOTTLE_DONE
