#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys


def has_nvcc() -> bool:
    return shutil.which("nvcc") is not None


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def ensure_cuda_toolkit() -> bool:
    if has_nvcc():
        return True

    print("CUDA toolkit not found. Attempting to install...")
    if shutil.which("apt-get") is None:
        print("CUDA toolkit missing and no supported package manager detected.")
        return False

    if shutil.which("sudo") is not None:
        run(["sudo", "apt-get", "update"])
        run(["sudo", "apt-get", "install", "-y", "nvidia-cuda-toolkit"])
        return has_nvcc()

    if os.geteuid() == 0:
        run(["apt-get", "update"])
        run(["apt-get", "install", "-y", "nvidia-cuda-toolkit"])
        return has_nvcc()

    print("CUDA toolkit missing and no sudo privileges to install it.")
    return False


def export_cuda_env() -> None:
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return
    cuda_bin = os.path.dirname(nvcc_path)
    cuda_root = os.path.abspath(os.path.join(cuda_bin, os.pardir))
    os.environ["CUDA_TOOLKIT_ROOT_DIR"] = cuda_root
    os.environ["CUDA_PATH"] = cuda_root
    os.environ["CUDA_ROOT"] = cuda_root


def main() -> int:
    if not ensure_cuda_toolkit():
        print("Continuing without CUDA toolkit installation.")

    export_cuda_env()
    run(["cargo", "build", "--all-features", "--release"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
