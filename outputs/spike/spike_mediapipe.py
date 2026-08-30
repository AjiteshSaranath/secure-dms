"""P0 spike 1: MediaPipe Face Landmarker on cached DMD frames.

Verifies: model loads, detects a face on real project data, returns 478
landmarks (incl. iris), and that EAR/MAR-relevant landmark indices exist.
"""
import glob
import sys

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\models\face_landmarker.task"
CACHE_GLOB = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_project\outputs\dmd_cache\*.npz"

caches = sorted(glob.glob(CACHE_GLOB))
if not caches:
    sys.exit("FAIL: no DMD cache npz found")
data = np.load(caches[0])
frames = data["frames"]
print(f"cache: {caches[0].rsplit(chr(92), 1)[-1]}  frames={frames.shape} dtype={frames.dtype}")

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL),
    num_faces=1,
    output_facial_transformation_matrixes=True,
)
landmarker = vision.FaceLandmarker.create_from_options(options)

detected = 0
n_landmarks = None
for idx in range(0, min(len(frames), 50), 10):
    frame = frames[idx]
    if frame.dtype != np.uint8:
        frame = (frame * 255).clip(0, 255).astype(np.uint8) if frame.max() <= 1.5 else frame.astype(np.uint8)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
    result = landmarker.detect(img)
    if result.face_landmarks:
        detected += 1
        n_landmarks = len(result.face_landmarks[0])

print(f"frames tried: 5, faces detected: {detected}, landmarks per face: {n_landmarks}")

# EAR (Soukupova & Cech 2016 measure) on MediaPipe's eye landmark indices
if detected:
    lm = result.face_landmarks[0]
    LEFT = [362, 385, 387, 263, 373, 380]  # p1..p6, left eye
    RIGHT = [33, 160, 158, 133, 153, 144]  # p1..p6, right eye

    def ear(ids):
        p = [np.array([lm[i].x, lm[i].y]) for i in ids]
        return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * np.linalg.norm(p[0] - p[3]))

    print(f"sample EAR  left={ear(LEFT):.3f} right={ear(RIGHT):.3f}")
    iris = [lm[i] for i in (468, 473)] if n_landmarks and n_landmarks >= 478 else None
    print(f"iris landmarks present: {iris is not None}")
    tm = result.facial_transformation_matrixes
    print(f"head transformation matrix available: {bool(tm)}")

print("PASS" if detected >= 4 else "MARGINAL" if detected else "FAIL")
