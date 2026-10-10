#!/bin/bash
set -euo pipefail
micromamba run -n sa pip install open_clip_torch torchvision
micromamba run -n sa python /root/SceneagentX/vast-cluster/scripts/run_codebook.py
echo CODEBOOK_DONE
