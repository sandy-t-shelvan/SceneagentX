#!/bin/bash
# Compile the 2DGS CUDA rasterizer into the existing 2dgs env.
# Does not change the torch 2.1.2+cu121 pin and does not train.
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.9"
cd /root/SceneagentX/third_party/2d-gaussian-splatting
micromamba run -n 2dgs python -m pip install "setuptools<81" wheel ninja
micromamba run -n 2dgs python - << 'PY'
import torch
if torch.__version__ != "2.1.2+cu121":
    raise SystemExit(f"torch moved to {torch.__version__}")
print("torch_still", torch.__version__, "cuda", torch.version.cuda)
PY
nvcc --version | tail -n 1
micromamba run -n 2dgs python -m pip install --no-build-isolation ./submodules/simple-knn ./submodules/diff-surfel-rasterization
micromamba run -n 2dgs python - << 'PY'
import torch
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from simple_knn._C import distCUDA2
if torch.__version__ != "2.1.2+cu121":
    raise SystemExit(f"torch moved to {torch.__version__}")
print("rasterizer_ok", torch.__version__, torch.cuda.is_available())
PY
echo BUILD_2DGS_DONE
