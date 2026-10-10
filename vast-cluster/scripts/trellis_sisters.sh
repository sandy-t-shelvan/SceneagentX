#!/bin/bash
set -euo pipefail
export ATTN_BACKEND=xformers
export SPCONV_ALGO=native
cd /root/SceneagentX/third_party/TRELLIS
micromamba run -n trellis python /root/run_trellis_sisters.py
echo TRELLIS_SISTERS_OK
