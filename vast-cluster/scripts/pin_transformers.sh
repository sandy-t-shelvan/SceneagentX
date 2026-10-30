#!/bin/bash
set -euo pipefail
if [ -f /root/trellis-sisters.pid ]; then
  pid=$(cat /root/trellis-sisters.pid)
  kill "$pid" 2>/dev/null || true
  pkill -P "$pid" 2>/dev/null || true
fi
sleep 1
micromamba run -n trellis pip install "transformers==4.46.3"
echo PINNED
