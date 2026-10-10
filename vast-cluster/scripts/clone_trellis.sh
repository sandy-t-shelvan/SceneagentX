#!/bin/bash
set -euo pipefail
cd /root/SceneagentX/third_party
if [ ! -d TRELLIS/.git ]; then
  git clone --depth 1 https://github.com/microsoft/TRELLIS.git TRELLIS
fi
echo CLONED
ls TRELLIS | head
