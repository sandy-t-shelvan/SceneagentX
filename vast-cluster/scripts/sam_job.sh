#!/bin/bash
set -euo pipefail
mkdir -p /root/SceneagentX/third_party
if [ ! -d /root/SceneagentX/third_party/Grounded-SAM-2/.git ]; then
  git clone --depth 1 https://github.com/IDEA-Research/Grounded-SAM-2.git /root/SceneagentX/third_party/Grounded-SAM-2
fi
cd /root/SceneagentX/third_party/Grounded-SAM-2
micromamba run -n gsam2 pip install -e .
mkdir -p checkpoints gdino_checkpoints
if [ ! -f checkpoints/sam2.1_hiera_large.pt ]; then
  curl -L -o checkpoints/sam2.1_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
fi
if [ ! -f gdino_checkpoints/groundingdino_swint_ogc.pth ]; then
  curl -L -o gdino_checkpoints/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
fi
micromamba run -n gsam2 python /root/SceneagentX/vast-cluster/scripts/run_semantics.py
echo SAM_DONE
