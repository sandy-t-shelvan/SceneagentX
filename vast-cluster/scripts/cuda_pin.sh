#!/bin/bash
set -u
echo NVCC
nvcc --version
echo DIRS
ls -d /usr/local/cuda* 2>/dev/null || true
for env in sa gsam2 nerf partfield trellis; do
  echo ENV "$env"
  micromamba run -n "$env" python -c 'import torch; print(torch.__version__); print(torch.version.cuda)' || echo FAIL
done
echo THIRD
ls /root/SceneagentX/third_party
