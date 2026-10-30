#!/bin/bash
set -eu
base=/root/SceneagentX/data/bottle/colmap/orbit1
mapfile -t bins < <(find "$base" -name images.bin -type f | sort)
if [ "${#bins[@]}" -eq 0 ]; then
  echo "NO_MODEL"
  exit 2
fi
printf '%s\n' "${bins[@]}"
model=$(dirname "${bins[0]}")
micromamba run -n nerf colmap model_analyzer --path "$model" > /tmp/orbit1_analyzer.txt
python3 - "$model" /tmp/orbit1_models.json << 'PY'
import json
import sys
from pathlib import Path
model = Path(sys.argv[1])
sparse = model.parent
names = sorted(path.name for path in sparse.iterdir() if path.is_dir())
Path(sys.argv[2]).write_text(json.dumps(names) + "\n", encoding="utf-8")
print("MODELS", names)
print("ANALYZER", Path("/tmp/orbit1_analyzer.txt").stat().st_size)
PY
echo COLMAP_REPORT_OK
