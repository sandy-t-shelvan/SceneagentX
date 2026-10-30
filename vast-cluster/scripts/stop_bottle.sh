#!/bin/bash
for pat in ns-train ns-process-data ffmpeg colmap.real; do
  pkill -f "$pat" || true
done
echo STOPPED
