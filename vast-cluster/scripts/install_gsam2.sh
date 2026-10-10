#!/bin/bash
set -euo pipefail
cd /root/SceneagentX/third_party/Grounded-SAM-2
nohup micromamba run -n gsam2 pip install -e . > /root/gsam2-install.log 2>&1 &
echo $! > /root/gsam2-install.pid
echo "started $(cat /root/gsam2-install.pid)"
