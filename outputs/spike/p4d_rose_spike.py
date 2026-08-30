"""Tier D — perception spike on ROSE-Youtu BEFORE the long extraction.

The stack was validated on 1280x720 landscape DMD video. ROSE is mobile-captured:
640x480, 1280x720, and PORTRAIT 480x640 / 720x1280. Mask attacks may also defeat
face detection outright. Check detection rate and signal sanity per species on a
few clips each — 5 minutes now versus discovering it 2 hours in.
"""
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
ROSE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")
CODES = {"G": "genuine", "Ps": "print_still", "Pq": "print_quiver", "Vl": "replay_lenovo",
         "Vm": "replay_mac", "Mc": "mask_cropped", "Mf": "mask_full", "Mu": "mask_upper"}

EAR_L = [362, 385, 387, 263, 373, 380]
EAR_R = [33, 160, 158, 133, 153, 144]

def make_landmarker():
    """A FRESH landmarker per clip: VIDEO mode requires monotonically increasing
    timestamps, and each clip restarts at t=0. (The notebook's extract_signals
    already creates one per video; this spike originally reused a single instance
    and hit 'Input timestamp must be monotonically increasing'.)"""
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))


def ear_of(px, ids):
    p = px[ids]
    d = np.linalg.norm(p[0] - p[3])
    return np.nan if d < 1e-9 else float(
        (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * d))


by_code = defaultdict(list)
for v in sorted(ROSE.rglob("*.mp4")):
    by_code[v.stem.split("_")[0]].append(v)

rng = random.Random(42)
print(f"{'species':<16}{'clips':>6}{'frames':>8}{'detected':>10}{'rate':>8}"
      f"{'meanEAR':>9}{'resolution':>12}")
for code, name in CODES.items():
    clips = rng.sample(by_code[code], min(3, len(by_code[code])))
    tot = det = 0
    ears, res = [], set()
    for clip in clips:
        landmarker = make_landmarker()
        cap = cv2.VideoCapture(str(clip))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok or i >= 120:
                break
            h, w = bgr.shape[:2]
            res.add(f"{w}x{h}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            r = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
                int(i / fps * 1000))
            tot += 1
            if r.face_landmarks:
                det += 1
                px = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
                ears.append(np.nanmean([ear_of(px, EAR_L), ear_of(px, EAR_R)]))
            i += 1
        cap.release()
    rate = det / max(tot, 1)
    print(f"{name:<16}{len(clips):>6}{tot:>8}{det:>10}{rate:>8.1%}"
          f"{(np.nanmean(ears) if ears else float('nan')):>9.3f}{sorted(res)[0]:>12}")

print("\nnotes: a low detection rate on mask species is EXPECTED and is itself a")
print("PAD signal; a low rate on GENUINE would invalidate the corpus for our stack.")
