#!/bin/bash
set -euo pipefail
cd /root/SceneagentX
export_one() {
  orbit="$1"
  config="$2"
  out="data/bottle/colmap/orbit${orbit}"
  tmp="/tmp/bottle-export-orbit${orbit}"
  rm -rf "$tmp"
  mkdir -p "$tmp" "$out"
  echo "EXPORT orbit${orbit} START"
  ns-export gaussian-splat --load-config "$config" --output-dir "$tmp"
  mv "$tmp/splat.ply" "$out/splat.ply"
  ls -lh "$out/splat.ply"
  echo "EXPORT orbit${orbit} DONE"
}
export_one 1 outputs/orbit1/splatfacto/2026-10-09_234601/config.yml
export_one 2 outputs/orbit2/splatfacto/2026-10-09_235735/config.yml
export_one 3 outputs/orbit3/splatfacto/2026-10-10_000656/config.yml
echo BOTTLE_EXPORT_OK
