#!/bin/bash
set -euo pipefail
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
cd /root/SceneagentX/third_party/TRELLIS
micromamba run -n trellis python -c 'import vox2seq; from trellis.pipelines import TrellisImageTo3DPipeline; print("import_ok")'
