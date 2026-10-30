#!/bin/bash
set -euo pipefail
find /root/SceneagentX/third_party/PartField -name 'run_part_clustering.py' -o -name '*cluster*.py' | head -n 30
echo ---
micromamba run -n partfield python - <<'PY'
import numpy as np
a = np.load("/root/SceneagentX/third_party/PartField/exp_results/partfield_features/scan/part_feat_scan_0.npy")
print(type(a), getattr(a, "shape", None), getattr(a, "dtype", None))
PY
