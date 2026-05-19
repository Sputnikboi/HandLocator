#!/usr/bin/env bash
# Install CUDA 12 runtime libraries + cuDNN on Ubuntu 22.04.
# The driver is already installed; this only adds the CUDA runtime and cuDNN.
set -euo pipefail

ARCH=x86_64

# Map Ubuntu codename to NVIDIA repo slug (e.g. noble → ubuntu2404)
case "$(lsb_release -sc)" in
  noble)   REPO_SLUG="ubuntu2404" ;;
  jammy)   REPO_SLUG="ubuntu2204" ;;
  focal)   REPO_SLUG="ubuntu2004" ;;
  *)       echo "Unknown Ubuntu release, defaulting to ubuntu2404"; REPO_SLUG="ubuntu2404" ;;
esac

echo ">>> Adding NVIDIA CUDA apt repository (${REPO_SLUG}) ..."
KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
wget -O "/tmp/${KEYRING_DEB}" \
  "https://developer.download.nvidia.com/compute/cuda/repos/${REPO_SLUG}/${ARCH}/${KEYRING_DEB}"
sudo dpkg -i "/tmp/${KEYRING_DEB}"
sudo apt-get update -qq

echo ">>> Installing CUDA 12 runtime + cuDNN ..."
# cuda-libraries-12-x pulls in cudart, cublas, cufft, etc.
# The CUDA 13.2 driver is backward-compatible with CUDA 12 apps.
sudo apt-get install -y \
  cuda-libraries-12-6 \
  libcudnn9-cuda-12 \
  libcudnn9-dev-cuda-12

echo ">>> Adding CUDA libs to linker cache ..."
echo "/usr/local/cuda-12.6/lib64" | sudo tee /etc/ld.so.conf.d/cuda-12-6.conf
sudo ldconfig

echo "Done. Verify with: ldconfig -p | grep libcudart"
