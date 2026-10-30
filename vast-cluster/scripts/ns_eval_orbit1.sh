#!/bin/bash
# Held-out metrics for the existing orbit-1 splat. Does not train and does not run COLMAP.
set -u
cd /root/SceneagentX
echo "ORBIT_FILES"
ls -lh data/bottle/raw/orbit1.MOV data/bottle/raw/orbit2.MOV data/bottle/raw/orbit3.MOV
echo "CHECKPOINT"
ls -lh outputs/orbit1/splatfacto/2026-10-09_234601/config.yml
mkdir -p data/bottle/ingest
micromamba run -n nerf python - << 'PY'
import importlib.util
import json
from pathlib import Path

import numpy as np

root = Path("/root/SceneagentX")
frames = sorted((root / "data/bottle/frames/orbit1").glob("*.png"))
if not frames:
    frames = sorted((root / "data/bottle/colmap/orbit1/images").glob("*.png"))
    frames += sorted((root / "data/bottle/colmap/orbit1/images").glob("*.jpg"))
luminance = []
try:
    import cv2
except Exception:
    cv2 = None
for path in frames:
    image = None if cv2 is None else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        continue
    luminance.append(float(image.mean()))
values = np.asarray(luminance, dtype=np.float64)
reviewed = int(values.size) > 0
mean = float(values.mean()) if reviewed else None
std = float(values.std()) if reviewed else None
gate = bool(reviewed and mean and mean > 0 and std < 0.05 * mean)
record = {
    "exposure_reviewed": reviewed,
    "frames_reviewed": int(values.size),
    "luminance_mean": None if mean is None else round(mean, 6),
    "luminance_std": None if std is None else round(std, 6),
    "exposure_gate_passed": gate,
    "appearance_embeddings_retrained": False,
    "method": "existing orbit1 frames, luminance mean and std",
    "capture": "data/bottle/raw/orbit1.MOV",
    "existing_capture": True,
    "note": "Exposure was reviewed on the existing orbit-1 frames. The splat was not retrained.",
}
if not reviewed:
    record["note"] = "No existing orbit-1 frames were readable. Exposure was not reviewed. The splat was not retrained."
(root / "data/bottle/ingest/colour_calibration.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print("COLOUR_FRAMES", int(values.size))
print("LAMA", importlib.util.find_spec("lama") is not None)
print("DIFFUSERS", importlib.util.find_spec("diffusers") is not None)
PY
start=$(date +%s)
if ! micromamba run -n nerf ns-eval --load-config outputs/orbit1/splatfacto/2026-10-09_234601/config.yml --output-path data/bottle/ingest/ns_eval.json; then
  echo NS_EVAL_FAIL
  exit 1
fi
end=$(date +%s)
seconds=$((end - start))
micromamba run -n nerf python -c "import json; from pathlib import Path; Path('/root/SceneagentX/data/bottle/ingest/ns_eval_minutes.json').write_text(json.dumps({'gpu_minutes': round(${seconds}/60.0, 4), 'seconds': ${seconds}, 'ns_train': 'not_run', 'checkpoint': 'outputs/orbit1/splatfacto/2026-10-09_234601', 'capture': 'data/bottle/raw/orbit1.MOV', 'existing_capture': True}, indent=2)+'\n')"
echo NS_EVAL_DONE
