#!/bin/bash
set -euo pipefail
pin() {
  env_name="$1"
  cuda_home="$2"
  env_root="/root/.local/share/mamba/envs/${env_name}"
  if [ ! -d "$env_root" ]; then
    echo "skip $env_name"
    return 0
  fi
  dest="${env_root}/etc/conda/activate.d"
  mkdir -p "$dest"
  cat > "${dest}/cuda-pin.sh" <<EOF
export CUDA_HOME=${cuda_home}
export PATH="\$CUDA_HOME/bin:\$PATH"
EOF
  echo "pinned $env_name -> $cuda_home"
}
pin sa /usr/local/cuda-13.0
pin gsam2 /usr/local/cuda-13.0
pin nerf /usr/local/cuda-12.1
pin partfield /usr/local/cuda-12.1
pin trellis /usr/local/cuda-12.1
pin 2dgs /usr/local/cuda-12.1
pin physgauss /usr/local/cuda-12.1
echo CHECK
for env in sa gsam2 nerf partfield trellis; do
  echo ENV "$env"
  micromamba run -n "$env" bash -c 'nvcc --version | tail -n 1; python -c "import torch; print(torch.__version__); print(torch.version.cuda)"'
done
