import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, time, sys

model_path = "models/hand_landmarker.task"

out_img_train = "datasets/hand_data/images/train"
out_lbl_train = "datasets/hand_data/labels/train"
out_img_val   = "datasets/hand_data/images/val"
out_lbl_val   = "datasets/hand_data/labels/val"
for d in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
    os.makedirs(d, exist_ok=True)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
no_hand = 0
target = 500

print("=" * 50)
print("HAND CAPTURE")
print("Hold your hands in front of camera NOW!")
print("Move them: open, closed, fist, different angles")
print("Ctrl+C to stop early")
print("=" * 50)

try:
    while count < target:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = detector.detect(mp_image)

        h, w = frame.shape[:2]

        if results.hand_landmarks:
            labels = []
            for hand_lms in results.hand_landmarks:
                xs = [lm.x for lm in hand_lms]
                ys = [lm.y for lm in hand_lms]
                x_min = max(0, min(xs) - 0.03)
                x_max = min(1, max(xs) + 0.03)
                y_min = max(0, min(ys) - 0.03)
                y_max = min(1, max(ys) + 0.03)

                bw = x_max - x_min
                bh = y_max - y_min

                if bw > 0.05 and bh > 0.05:
                    cx = (x_min + x_max) / 2
                    cy = (y_min + y_max) / 2
                    labels.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if labels:
                is_val = (count % 5 == 0)
                img_d = out_img_val if is_val else out_img_train
                lbl_d = out_lbl_val if is_val else out_lbl_train

                fname = f"frame_{count:05d}"
                cv2.imwrite(f"{img_d}/{fname}.jpg", frame)
                with open(f"{lbl_d}/{fname}.txt", 'w') as f:
                    f.write('\n'.join(labels))
                count += 1
                no_hand = 0
                sys.stdout.write(f"\r  Captured: {count}/{target}  ({len(labels)} hand(s))   ")
                sys.stdout.flush()
            else:
                no_hand += 1
        else:
            no_hand += 1

        if no_hand > 0 and no_hand % 30 == 0:
            sys.stdout.write(f"\r  Waiting for hands... ({no_hand} frames without detection)   ")
            sys.stdout.flush()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nStopped early.")

cap.release()
detector.close()

n_train = len(os.listdir(out_img_train))
n_val = len(os.listdir(out_img_val))
print(f"\n\nDone! Captured {count} frames")
print(f"  Train: {n_train}")
print(f"  Val:   {n_val}")
if count < 50:
    print("\nWARNING: Very few captures. Make sure your hands are visible to the webcam!")
