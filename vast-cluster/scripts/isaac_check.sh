#!/bin/bash
python - <<'PY'
import importlib.util
names = ("isaaclab", "omni.isaac.lab", "isaacsim")
found = [name for name in names if importlib.util.find_spec(name)]
print("isaac_modules", " ".join(found) if found else "none")
PY
find /opt /root /isaac-sim -maxdepth 3 -iname '*isaac*' -type d 2>/dev/null | head -n 20
echo ISAAC_CHECK_DONE
