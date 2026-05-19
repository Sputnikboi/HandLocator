#!/usr/bin/env bash
# Downloads the ONNX Runtime C++ prebuilt package (CUDA 12 / Linux x86_64)
# and extracts it to ./onnxruntime/ so CMake can find it.
set -euo pipefail

# Check https://github.com/microsoft/onnxruntime/releases for the latest version.
ORT_VERSION="1.26.0"
ORT_TARBALL="onnxruntime-linux-x64-gpu-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}"

echo ">>> Downloading ONNX Runtime ${ORT_VERSION} (CUDA 12) ..."
wget -q --show-progress -O "/tmp/${ORT_TARBALL}" "${ORT_URL}"

echo ">>> Extracting to ./onnxruntime/ ..."
mkdir -p onnxruntime
tar -xzf "/tmp/${ORT_TARBALL}" -C onnxruntime --strip-components=1

echo "Done. Headers: onnxruntime/include/  Libs: onnxruntime/lib/"
ls onnxruntime/lib/libonnxruntime*.so* 2>/dev/null
