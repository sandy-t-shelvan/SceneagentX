#!/bin/bash
set -euo pipefail
cd /root/SceneagentX
rm -rf data/bottle/colmap
for n in 1 2 3; do
  frames="data/bottle/frames/orbit${n}"
  if [ ! -d "${frames}" ] || [ -z "$(ls -A "${frames}" 2>/dev/null)" ]; then
    mkdir -p "${frames}"
    echo "ORBIT ${n} EXTRACT START"
    ffmpeg -y -i "data/bottle/raw/orbit${n}.MOV" -vf fps=4 "${frames}/frame_%05d.png"
  fi
  echo "ORBIT ${n} FRAMES $(ls "${frames}" | wc -l)"
  echo "ORBIT ${n} COLMAP START"
  ns-process-data images --data "${frames}" --output-dir "data/bottle/colmap/orbit${n}" --matching-method sequential
  echo "ORBIT ${n} COLMAP DONE"
done
echo "ORBIT 1 SPLATFACTO START"
ns-train splatfacto --data data/bottle/colmap/orbit1 --vis tensorboard --viewer.quit-on-train-completion True
echo BOTTLE_PIPELINE_OK
