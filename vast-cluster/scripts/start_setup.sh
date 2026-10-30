#!/bin/bash
set -euo pipefail
mkdir -p /root/SceneagentX
tar -xf /root/sceneagentx-src.tar -C /root
mkdir -p /root/SceneagentX/data/scan/renders/synthetic
cd /root/SceneagentX
nohup bash setup.sh --no-status > /root/setup.log 2>&1 &
echo $! > /root/setup.pid
echo "started $(cat /root/setup.pid)"
