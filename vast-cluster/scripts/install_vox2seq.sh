#!/bin/bash
set -euo pipefail
base="https://huggingface.co/spaces/dkatz2391/TRELLIS_TextTo3D_Try2/resolve/main"
root="/tmp/vox2seq-src"
files="
extensions/vox2seq/setup.py
extensions/vox2seq/src/api.cu
extensions/vox2seq/src/api.h
extensions/vox2seq/src/ext.cpp
extensions/vox2seq/src/hilbert.cu
extensions/vox2seq/src/hilbert.h
extensions/vox2seq/src/z_order.cu
extensions/vox2seq/src/z_order.h
extensions/vox2seq/vox2seq/__init__.py
extensions/vox2seq/vox2seq/pytorch/__init__.py
extensions/vox2seq/vox2seq/pytorch/default.py
extensions/vox2seq/vox2seq/pytorch/hilbert.py
extensions/vox2seq/vox2seq/pytorch/z_order.py
"
for rel in $files; do
  mkdir -p "$root/$(dirname "$rel")"
  curl -fsSL "$base/$rel" -o "$root/$rel"
done
export CUDA_HOME=/usr/local/cuda-12.1
export PATH="${CUDA_HOME}/bin:${PATH}"
micromamba run -n trellis pip install --no-build-isolation "$root/extensions/vox2seq"
echo VOX2SEQ_OK
