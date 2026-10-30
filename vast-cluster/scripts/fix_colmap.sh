#!/bin/bash
set -euo pipefail
micromamba install -n nerf -c conda-forge openimageio -y
echo LIBS
find /root/.local/share/mamba/envs/nerf -name 'libOpenImageIO.so*' | head
echo COLMAP
micromamba run -n nerf colmap help | head -n 8
echo COLMAP_FIXED
