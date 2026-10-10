#!/bin/bash
real="/root/.local/share/mamba/envs/nerf/bin/colmap.real"
args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --SiftExtraction.use_gpu) args+=("--FeatureExtraction.use_gpu") ;;
    --SiftMatching.use_gpu) args+=("--FeatureMatching.use_gpu") ;;
    *) args+=("$1") ;;
  esac
  shift
done
exec "$real" "${args[@]}"
