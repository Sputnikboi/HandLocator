# HandLocator Project State

## Project Location
- HandLocator: `/home/sputnikboi/Git/HandLocator`
- Build dir: `build/`
- Run: `cd build && ./hand_locator ../models/hand_yolov8n.onnx ../models/hand_landmark_lite.onnx`
- Semester project (DON'T TOUCH): `/home/sputnikboi/Git/Uni/Semesterprojekt-i-kontrol-og-regulering-af-robotsystemer`

## Current Status (2026-05-19)

### Working Features
- **YOLOv8n hand detector**: 93% mAP50 on original 500-frame dataset. ONNX at `models/hand_yolov8n.onnx`
- **Hand landmark detector**: Reads `Identity` (out[0]) for screen-space landmarks (0-224px), `Identity_3` (out[3]) for world-space 3D landmarks. Presence threshold = 0.6.
- **Multi-hand tracking**: Up to 2 hands tracked simultaneously with periodic re-detection (every 15 frames)
- **Landmark sanity check**: Drops hands if bbox < 10px or fully off-screen
- **3D GLFW viewer**: Separate OpenGL window with mouse-drag orbit, scroll zoom, ground grid, RGB axes, depth-shaded joints
- **Sunshine dual-GPU fix**: Systemd override forces NVENC on GPU 0 (GTX 1660), ML on GPU 1 (RTX 2070)

### Training In Progress
- **Run**: `runs/detect/runs/hand_detect_v2-2/` (note: `-2` suffix auto-appended by YOLO)
- **Dataset**: 1000 frames (800 train / 200 val) at `datasets/hand_data/`
  - Original 500: frontal views, open/closed hands
  - New 500: side views, fists, tilted, close/far, varied angles (captured with `capture_hands_v2.py`)
- **Config**: 80 epochs, patience=20, batch=16, imgsz=640, device=1 (RTX 2070)
- **cuDNN disabled**: `TORCH_CUDNN_V8_API_DISABLED=1` + `torch.backends.cudnn.enabled=False`
- **Early results**: Epoch 1: 80.6% mAP50, Epoch 2: 66%, Epoch 3: 58.7% (expected dip with harder data)

### After Training Completes
1. Export: `source .model_venv/bin/activate && python -c "from ultralytics import YOLO; YOLO('runs/detect/runs/hand_detect_v2-2/weights/best.pt').export(format='onnx', imgsz=640, opset=17)"`
   - NOTE: Check actual run dir — might be `hand_detect_v2-2` or different suffix
2. Copy: `cp runs/detect/runs/hand_detect_v2-2/weights/best.onnx models/hand_yolov8n.onnx`
3. Rebuild: `cd build && cmake --build . -j$(nproc)`
4. Test: `./hand_locator ../models/hand_yolov8n.onnx ../models/hand_landmark_lite.onnx`

## Architecture

### Source Files (`src/`)
- `main.cpp`: Entry point, camera loop, multi-hand tracking, 2D visualization, GLFW viewer integration
- `hand_detector.hpp/cpp`: YOLOv8 ONNX detector. Input: NCHW [1,3,640,640], letterbox pad=114, BGR→RGB, /255.0. Output: [1,5,8400]
- `hand_landmark.hpp/cpp`: MediaPipe landmark model. Input: NCHW [1,3,224,224]. Outputs by index:
  - out[0] = Identity [1,63]: screen landmarks 0-224px (USED for tracking)
  - out[1] = Identity_1 [1,1]: presence logit (sigmoid → threshold 0.6)
  - out[2] = Identity_2 [1,1]: handedness logit (sigmoid → >0.5 = right)
  - out[3] = Identity_3 [1,63]: world landmarks meters (USED for 3D view)
- `viewer3d.hpp/cpp`: GLFW+OpenGL 3D viewer. Mouse drag orbit, scroll zoom, grid, axes, depth-shaded spheres at joints, fingertips highlighted.
- `types.hpp`: PalmDetection, HandLandmarks (with world_landmarks), enums, HAND_CONNECTIONS
- `palm_detector.hpp/cpp`: Old BlazePalm, NOT compiled, kept for reference

### Build Dependencies
- OpenCV (core, dnn, highgui, imgproc, videoio)
- ONNX Runtime (local `./onnxruntime/` dir, CUDA provider on GPU 1)
- GLFW3, GLEW, OpenGL (for 3D viewer)
- CMakeLists.txt links all of the above

### Key Technical Decisions
- cuDNN broken (version mismatch) — always disable for PyTorch training
- GPU 0 (GTX 1660) = display/Sunshine encoding, GPU 1 (RTX 2070) = ML inference
- Sunshine fix: `CUDA_VISIBLE_DEVICES=0` + `CUDA_DEVICE_ORDER=PCI_BUS_ID` in systemd override
- Landmark presence threshold 0.6 (not 0.5) — model outputs ~0.5 for garbage input
- ROI from landmarks: no center shift (unlike BlazePalm), 1.6× expansion
- palm_detection_full model incompatible — different architecture despite same output shape
