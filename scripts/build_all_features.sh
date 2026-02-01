#!/usr/bin/env bash
set -euo pipefail

ensure_cuda_toolkit() {
  if command -v nvcc >/dev/null 2>&1; then
    return 0
  fi

  echo "CUDA toolkit not found. Attempting to install..."
  if command -v apt-get >/dev/null 2>&1; then
    if command -v sudo >/dev/null 2>&1; then
      sudo apt-get update
      sudo apt-get install -y nvidia-cuda-toolkit
    elif [ "$(id -u)" -eq 0 ]; then
      apt-get update
      apt-get install -y nvidia-cuda-toolkit
    else
      echo "CUDA toolkit missing and no sudo privileges to install it."
      return 1
    fi
  else
    echo "CUDA toolkit missing and no supported package manager detected."
    return 1
  fi
}

if ! ensure_cuda_toolkit; then
  echo "Continuing without CUDA toolkit installation."
fi

if command -v nvcc >/dev/null 2>&1; then
  cuda_bin="$(dirname "$(command -v nvcc)")"
  export CUDA_TOOLKIT_ROOT_DIR="$(cd "${cuda_bin}/.." && pwd)"
  export CUDA_PATH="${CUDA_TOOLKIT_ROOT_DIR}"
  export CUDA_ROOT="${CUDA_TOOLKIT_ROOT_DIR}"
fi

cargo build --all-features --release
