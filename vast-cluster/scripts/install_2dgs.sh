#!/bin/bash
# Install the 2dgs env pin and clone 2DGS. Does not train a splat.
set -euo pipefail
ROOT=/root/SceneagentX
cd "$ROOT"

echo "=== pin check ==="
micromamba run -n 2dgs python - << 'PY'
import sys
import torch
version = sys.version.split()[0]
if not version.startswith("3.10."):
    raise SystemExit(f"python {version} is not 3.10")
if torch.__version__ != "2.1.2+cu121":
    raise SystemExit(f"torch {torch.__version__} is not 2.1.2+cu121")
if not torch.cuda.is_available():
    raise SystemExit("cuda is not available")
print("pin_ok", version, torch.__version__, torch.version.cuda)
PY

DEST="$ROOT/third_party/2d-gaussian-splatting"
if [ ! -d "$DEST/.git" ]; then
  echo "=== clone ==="
  git clone --recursive https://github.com/hbb1/2d-gaussian-splatting.git "$DEST"
else
  echo "=== repo present ==="
  git -C "$DEST" rev-parse --short HEAD
fi

echo "=== python deps ==="
micromamba run -n 2dgs python -m pip install --upgrade pip
micromamba run -n 2dgs python -m pip install \
  "numpy<2" plyfile tqdm opencv-python-headless "open3d>=0.17,<0.19" scipy scikit-image
micromamba run -n 2dgs python - << 'PY'
import torch
if torch.__version__ != "2.1.2+cu121":
    raise SystemExit(f"torch moved to {torch.__version__}")
print("torch_still", torch.__version__)
PY

echo "=== build rasterizer ==="
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
nvcc --version | tail -n 1
cd "$DEST"
micromamba run -n 2dgs python -m pip install --no-build-isolation ./submodules/diff-surfel-rasterization ./submodules/simple-knn

echo "=== import check ==="
micromamba run -n 2dgs python - << 'PY'
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2
print("rasterizer_ok", torch.__version__, torch.cuda.is_available())
PY

echo "=== orbit1 layout ==="
find "$ROOT/data/bottle/colmap/orbit1" -maxdepth 3 -type d | head -n 40
echo "=== images count ==="
find "$ROOT/data/bottle/colmap/orbit1" -type d -name images -o -type d -name input | head
find "$ROOT/data/bottle/colmap/orbit1" -name '*.png' -o -name '*.jpg' -o -name '*.JPG' | head
echo "=== isaac version ==="
cat /root/jammy/opt/IsaacLab/VERSION 2>/dev/null || true
find /root/jammy/opt/isaac -maxdepth 3 -name 'VERSION' -o -name 'version.txt' -o -name 'python.sh' | head -n 20
ls /root/jammy/opt/isaac | head
echo INSTALL_2DGS_DONE
