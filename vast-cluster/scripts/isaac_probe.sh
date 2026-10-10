#!/bin/bash
set -eu
echo "OS"
. /etc/os-release
echo "$PRETTY_NAME"
echo "PYTHON $(python3 --version)"
echo "DISK"
df -h / /root /workspace 2>/dev/null || df -h /
echo "MEM"
free -g
echo "GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "ISAAC"
python3 - << 'PY'
import importlib.util
names = ("isaaclab", "omni.isaac.lab", "isaacsim")
found = [name for name in names if importlib.util.find_spec(name)]
print("isaac_modules", " ".join(found) if found else "none")
PY
echo PROBE_OK
