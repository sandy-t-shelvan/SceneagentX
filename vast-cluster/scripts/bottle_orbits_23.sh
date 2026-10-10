#!/bin/bash
set -euo pipefail
cd /root/SceneagentX
echo "WAIT_FOR_ORBIT1"
while pgrep -f "ns-train splatfacto --data data/bottle/colmap/orbit1" >/dev/null; do
  sleep 30
done
echo "ORBIT 2 SPLATFACTO START"
ns-train splatfacto --data data/bottle/colmap/orbit2 --vis tensorboard --viewer.quit-on-train-completion True
echo "ORBIT 2 SPLATFACTO DONE"
echo "ORBIT 3 SPLATFACTO START"
ns-train splatfacto --data data/bottle/colmap/orbit3 --vis tensorboard --viewer.quit-on-train-completion True
echo "ORBIT 3 SPLATFACTO DONE"
echo BOTTLE_ALL_SPLATS_OK
