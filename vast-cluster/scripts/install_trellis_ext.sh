#!/bin/bash
set -euo pipefail
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
echo "nvcc $(nvcc --version | tail -n 1)"
micromamba run -n trellis pip install --no-build-isolation /tmp/extensions/nvdiffrast
if [ ! -d /tmp/extensions/diffoctreerast/.git ]; then
  git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
fi
micromamba run -n trellis pip install --no-build-isolation /tmp/extensions/diffoctreerast
if [ ! -d /tmp/extensions/mip-splatting/.git ]; then
  git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
fi
micromamba run -n trellis pip install --no-build-isolation /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization
rm -rf /tmp/extensions/vox2seq
cp -a /root/SceneagentX/third_party/TRELLIS/extensions/vox2seq /tmp/extensions/vox2seq
micromamba run -n trellis pip install --no-build-isolation /tmp/extensions/vox2seq
echo TRELLIS_SETUP_OK
