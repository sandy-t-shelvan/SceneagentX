#!/bin/bash
set -euo pipefail
cd /root/SceneagentX
mkdir -p data/bottle/raw data/bottle/colmap
echo "ns-process-data $(micromamba run -n nerf which ns-process-data)"
for n in 1 2 3; do
  echo "ORBIT ${n} COLMAP START"
  micromamba run -n nerf ns-process-data video \
    --data "data/bottle/raw/orbit${n}.MOV" \
    --output-dir "data/bottle/colmap/orbit${n}"
  echo "ORBIT ${n} COLMAP DONE"
done
echo "ORBIT 1 SPLATFACTO START"
micromamba run -n nerf ns-train splatfacto \
  --data data/bottle/colmap/orbit1 \
  --vis tensorboard \
  --viewer.quit-on-train-completion True
echo BOTTLE_PIPELINE_OK
