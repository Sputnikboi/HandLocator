"""
One-time model preparation: downloads MediaPipe hand models as .tflite
(via mediapipe Python) and converts them to ONNX for the C++ runtime.
"""
import os, sys, subprocess, pathlib, shutil, glob, urllib.request

MODELS_DIR = pathlib.Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
VENV_DIR   = pathlib.Path(__file__).parent / ".model_venv"

# ── 1. Re-launch inside venv if needed ───────────────────────────────────────
if sys.prefix == sys.base_prefix:
    if not (VENV_DIR / "bin" / "python").exists():
        print(f"Creating venv at {VENV_DIR} ...")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV_DIR)])
    venv_python = VENV_DIR / "bin" / "python"
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

def pip(*args):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *args])

# ── 2. Install deps ───────────────────────────────────────────────────────────
print("Installing mediapipe + tflite2onnx ...")
pip("mediapipe", "tflite2onnx")

import mediapipe as mp        # noqa: E402
import tflite2onnx            # noqa: E402

# ── 3. Trigger mediapipe model download via legacy solutions API ──────────────
print("Initialising mp.solutions.hands to trigger model download ...")
try:
    _hands = mp.solutions.hands.Hands(static_image_mode=True)
    _hands.close()
except Exception as e:
    print(f"  (init raised {e} — continuing anyway)")

# ── 4. Locate all .tflite files that appeared anywhere we care about ──────────
search_roots = [
    str(VENV_DIR),
    os.path.expanduser("~/.cache"),
    "/tmp",
    str(pathlib.Path(mp.__file__).parent),
]

wanted = {
    "palm_detection_lite":  "palm_detection_lite",
    "palm_detection_full":  "palm_detection_lite",   # accept full as lite
    "hand_landmark_lite":   "hand_landmark_lite",
    "hand_landmark_full":   "hand_landmark_lite",
}

found: dict[str, pathlib.Path] = {}  # onnx_stem → source_path

for root in search_roots:
    for path_str in glob.glob(f"{root}/**/*.tflite", recursive=True):
        p = pathlib.Path(path_str)
        stem = p.stem.lower()
        for key, val in wanted.items():
            if key in stem and val not in found:
                found[val] = p
                print(f"  Found {val}: {p}")

# ── 5. Fallback: download tflite directly from Google Storage ─────────────────
GOOGLE_URLS = {
    "palm_detection_lite": (
        "https://storage.googleapis.com/mediapipe-assets/palm_detection_lite.tflite"
    ),
    "hand_landmark_lite": (
        "https://storage.googleapis.com/mediapipe-assets/hand_landmark_lite.tflite"
    ),
}

for stem, url in GOOGLE_URLS.items():
    if stem not in found:
        dest = MODELS_DIR / f"{stem}.tflite"
        print(f"  Downloading {stem}.tflite from Google Storage ...")
        try:
            urllib.request.urlretrieve(url, dest)
            if dest.stat().st_size < 10_000:
                print(f"  WARNING: {dest} looks too small ({dest.stat().st_size} B) — may be 404")
            else:
                found[stem] = dest
                print(f"  Saved: {dest}")
        except Exception as e:
            print(f"  WARNING: download failed: {e}")

if not found:
    sys.exit("ERROR: Could not locate any tflite model files. Check your internet connection.")

# ── 6. Convert tflite → ONNX ─────────────────────────────────────────────────
for onnx_stem, tflite_path in found.items():
    tmp   = MODELS_DIR / tflite_path.name
    out   = MODELS_DIR / f"{onnx_stem}.onnx"
    if tmp != tflite_path:
        shutil.copy(tflite_path, tmp)
    print(f"\nConverting {tflite_path.name} → {out.name} ...")
    try:
        tflite2onnx.convert(str(tmp), str(out))
        print(f"  Saved: {out}  ({out.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"  FAILED: {e}")
    finally:
        if tmp != tflite_path and tmp.exists():
            tmp.unlink()

# ── 7. Summary ────────────────────────────────────────────────────────────────
print("\nModels directory:")
for f in sorted(MODELS_DIR.iterdir()):
    if f.suffix == ".onnx":
        print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
