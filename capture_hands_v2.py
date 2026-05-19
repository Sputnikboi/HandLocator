"""
Supplemental hand capture — focuses on hard angles, side views, varied lighting.
Appends to the existing dataset (doesn't overwrite).
"""
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, time, sys, glob

model_path = "models/hand_landmarker.task"

out_img_train = "datasets/hand_data/images/train"
out_lbl_train = "datasets/hand_data/labels/train"
out_img_val   = "datasets/hand_data/images/val"
out_lbl_val   = "datasets/hand_data/labels/val"
for d in [out_img_train, out_lbl_train, out_img_val, out_lbl_val]:
    os.makedirs(d, exist_ok=True)

# Find the highest existing frame number to avoid overwriting
existing = glob.glob(f"{out_img_train}/frame_*.jpg") + glob.glob(f"{out_img_val}/frame_*.jpg")
start_idx = 0
for p in existing:
    base = os.path.splitext(os.path.basename(p))[0]
    try:
        num = int(base.split("_")[1])
        start_idx = max(start_idx, num + 1)
    except (IndexError, ValueError):
        pass

print(f"Existing frames detected — starting at index {start_idx}")

# Lower confidence thresholds to catch harder poses (side views, partial occlusion)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.4,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

target = 500
count = 0
no_hand = 0
idx = start_idx

# Pose prompts to cycle through
poses = [
    "SIDE VIEW — show the edge/karate-chop side of your hand",
    "SIDE VIEW — other hand, edge view",
    "FIST from the side",
    "POINTING — index finger out, from the side",
    "FLAT HAND tilted 45 degrees",
    "HAND FAR AWAY — arm fully extended",
    "HAND VERY CLOSE — fill the frame",
    "TWO HANDS — both from the side",
    "HAND MOVING — wave slowly side to side",
    "BACK OF HAND — knuckles facing camera",
    "FINGERS SPREAD — from an angle",
    "RELAXED HAND — natural hanging position",
]

print("=" * 60)
print("SUPPLEMENTAL HAND CAPTURE — HARD ANGLES")
print("=" * 60)
print(f"Target: {target} new frames")
print()
print("Cycle through these poses (switch every ~40 frames):")
for i, p in enumerate(poses):
    print(f"  {i+1:2d}. {p}")
print()
print("Starting in 3 seconds — get your hands ready!")
time.sleep(3)

current_pose = 0
frames_this_pose = 0

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

                if bw > 0.04 and bh > 0.04:
                    cx_n = (x_min + x_max) / 2
                    cy_n = (y_min + y_max) / 2
                    labels.append(f"0 {cx_n:.6f} {cy_n:.6f} {bw:.6f} {bh:.6f}")

            if labels:
                is_val = (count % 5 == 0)
                img_d = out_img_val if is_val else out_img_train
                lbl_d = out_lbl_val if is_val else out_lbl_train

                fname = f"frame_{idx:05d}"
                cv2.imwrite(f"{img_d}/{fname}.jpg", frame)
                with open(f"{lbl_d}/{fname}.txt", 'w') as f:
                    f.write('\n'.join(labels))
                count += 1
                idx += 1
                no_hand = 0
                frames_this_pose += 1

                # Prompt next pose
                if frames_this_pose >= 40:
                    frames_this_pose = 0
                    current_pose = (current_pose + 1) % len(poses)
                    print(f"\n>>> SWITCH POSE: {poses[current_pose]}")

                pose_name = poses[current_pose][:40]
                sys.stdout.write(f"\r  [{count}/{target}] {len(labels)} hand(s) | Pose: {pose_name}...   ")
                sys.stdout.flush()
            else:
                no_hand += 1
        else:
            no_hand += 1

        if no_hand > 0 and no_hand % 30 == 0:
            sys.stdout.write(f"\r  Waiting for hands... ({no_hand} empty) | Try: {poses[current_pose][:50]}   ")
            sys.stdout.flush()

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n\nStopped early.")

cap.release()
detector.close()

n_train = len(os.listdir(out_img_train))
n_val = len(os.listdir(out_img_val))
print(f"\n\nDone! Captured {count} new frames (total dataset: {n_train + n_val})")
print(f"  Train: {n_train}")
print(f"  Val:   {n_val}")
