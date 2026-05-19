#!/usr/bin/env bash
# Downloads palm_detection_lite.onnx and hand_landmark_lite.onnx
# from PINTO0309's model zoo (MediaPipe models converted to ONNX).
# Repo: https://github.com/PINTO0309/PINTO_model_zoo

set -euo pipefail
mkdir -p models
cd models

BASE="https://github.com/PINTO0309/PINTO_model_zoo/raw/main"

echo "Downloading palm_detection_lite.onnx ..."
curl -L -o palm_detection_lite.onnx \
  "${BASE}/030_BlazePalm/01_float32/palm_detection_lite.onnx"

echo "Downloading hand_landmark_lite.onnx ..."
curl -L -o hand_landmark_lite.onnx \
  "${BASE}/033_Hand_Detection_and_Tracking/01_float32/hand_landmark_lite.onnx"

echo "Done. Files in $(pwd):"
ls -lh *.onnx
