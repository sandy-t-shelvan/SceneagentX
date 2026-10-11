#!/bin/bash
set +e
echo "=== 2dgs torch ==="
micromamba run -n 2dgs python - << 'PY'
import sys
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
PY
echo "=== cuda homes ==="
ls -d /usr/local/cuda* 2>/dev/null
echo "=== nvcc default ==="
nvcc --version 2>/dev/null | tail -n 1
echo "=== transforms and ply ==="
find /root /opt /workspace -name 'transforms.json' -o -name 'dataparser_transforms.json' -o -name 'object.ply' -o -name 'cameras.bin' -o -name 'point_cloud.ply' 2>/dev/null | head -n 80
echo "=== isaac listing ==="
ls -la /root/jammy/opt 2>/dev/null
find /root/jammy/opt -maxdepth 4 \( -name 'isaaclab.sh' -o -name 'python.sh' -o -name 'VERSION' -o -name 'version.txt' -o -name 'VERSION.txt' \) 2>/dev/null | head -n 40
echo "=== data ==="
ls /root/SceneagentX/data 2>/dev/null
echo "=== third_party ==="
ls /root/SceneagentX/third_party
echo "=== 2dgs yml head ==="
sed -n '1,20p' /root/SceneagentX/env/2dgs.yml
echo DONE
