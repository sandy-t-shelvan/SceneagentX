#!/bin/bash
set -u
cd /root/SceneagentX/third_party/PartField
micromamba run -n partfield pip install lightning==2.2 h5py yacs trimesh scikit-image loguru plyfile einops
mkdir -p data/scan
cp /root/scan.ply data/scan/scan.ply 2>/dev/null || true
micromamba run -n partfield python partfield_inference.py -c configs/final/demo.yaml --opts continue_ckpt model/model_objaverse.ckpt result_name partfield_features/scan dataset.data_path data/scan
echo PARTFIELD_INFER_EXIT:$?
