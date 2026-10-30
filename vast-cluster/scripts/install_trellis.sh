#!/bin/bash
set -euo pipefail
cd /root/SceneagentX/third_party
if [ ! -d TRELLIS/.git ]; then
  git clone --depth 1 https://github.com/microsoft/TRELLIS.git TRELLIS
fi
cd /root/SceneagentX/third_party/TRELLIS
export ATTN_BACKEND=xformers
micromamba run -n trellis pip install numpy
micromamba run -n trellis pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
micromamba run -n trellis python -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
micromamba run -n trellis pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph "transformers==4.46.3"
micromamba run -n trellis pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
micromamba run -n trellis pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
micromamba run -n trellis pip install spconv-cu120
micromamba run -n trellis pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
mkdir -p /tmp/extensions
if [ ! -d /tmp/extensions/nvdiffrast/.git ]; then
  git clone --depth 1 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
fi
micromamba run -n trellis pip install /tmp/extensions/nvdiffrast
if [ ! -d /tmp/extensions/diffoctreerast/.git ]; then
  git clone --depth 1 --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
fi
micromamba run -n trellis pip install /tmp/extensions/diffoctreerast
if [ ! -d /tmp/extensions/mip-splatting/.git ]; then
  git clone --depth 1 https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
fi
micromamba run -n trellis pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization
rm -rf /tmp/extensions/vox2seq
cp -a extensions/vox2seq /tmp/extensions/vox2seq
micromamba run -n trellis pip install /tmp/extensions/vox2seq
echo TRELLIS_SETUP_OK
