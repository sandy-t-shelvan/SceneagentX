#!/bin/bash
set -euo pipefail
cd /root/SceneagentX/third_party/PartField
mkdir -p model
micromamba run -n sa python - <<'PY'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    "mikaelaangel/partfield-ckpt",
    "model_objaverse.ckpt",
    local_dir="model",
)
print(path)
PY
echo PARTFIELD_CKPT_OK
