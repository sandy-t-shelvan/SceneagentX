#!/usr/bin/env bash
# Create or update the seven SceneAgent envs on the Linux GPU box.
# If an env imports torch but torch.cuda.is_available() is false, fix that
# env's file under env/. Do not patch anything inside third_party/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if ! command -v micromamba >/dev/null 2>&1; then
  echo "micromamba is required on the GPU box." >&2
  echo "https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html" >&2
  exit 1
fi

START="$(date +%s)"
ENVS=(sa nerf gsam2 2dgs partfield physgauss trellis)
for name in "${ENVS[@]}"; do
  file="$ROOT/env/${name}.yml"
  if micromamba run -n "$name" python -c "import sys" >/dev/null 2>&1; then
    echo "Updating ${name}"
    micromamba install -y -n "$name" -f "$file"
  else
    echo "Creating ${name}"
    micromamba create -y -f "$file"
  fi
done

END="$(date +%s)"
MINUTES="$(python -c "print(round((${END} - ${START}) / 60.0, 2))")"
echo "setup_minutes=${MINUTES}"

OBJ="${SCENEAGENT_OBJ:-marker}"
if [[ "${1:-}" != "--no-status" ]]; then
  micromamba run -n sa python -m tools.common.status set-setup-minutes --obj "$OBJ" --minutes "$MINUTES"
fi

echo "Next, from this repo:"
echo "  micromamba run -n sa pytest tests -q"
echo "  python tools/check_env.py --obj ${OBJ}"
