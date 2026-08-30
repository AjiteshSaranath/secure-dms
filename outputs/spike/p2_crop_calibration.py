"""P2 — geometric calibration of MediaPipe face box -> RetinaFace convention.

The fast gaze path (p2_gaze_fastpath.py) is 46x faster but showed a systematic
+5.3 deg yaw bias, because L2CS-Net was trained on RetinaFace crops and
MediaPipe's landmark bounding box uses a different convention.

This measures the fixed geometric relationship between the two boxes on real
DMD frames and derives a constant transform. Note what this is and is not:
  - It IS an implementation-detail calibration between two face detectors,
    using no labels and no ground truth, leaving all model weights untouched.
  - It is NOT model training or threshold tuning; the no-training constraint
    (plan DP-8 scope) is unaffected.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from face_detection import RetinaFace

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")

videos = sorted(DMD_DROW.rglob("*rgb_face.mp4"))[:3]
FRAMES_PER_VIDEO = 40
STRIDE = 37  # spread across the session to sample varied head poses

landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
        num_faces=1))
detector = RetinaFace(gpu_id=0)

records = []
for video in videos:
    cap = cv2.VideoCapture(str(video))
    for k in range(FRAMES_PER_VIDEO):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k * STRIDE)
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.ascontiguousarray(rgb)))
        if not res.face_landmarks:
            continue
        lm = res.face_landmarks[0]
        xs, ys = np.array([p.x for p in lm]) * w, np.array([p.y for p in lm]) * h
        mp_box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])

        faces = detector(bgr)
        if faces is None or len(faces) == 0:
            continue
        box, _, score = max(faces, key=lambda f: f[2])
        if score < 0.5:
            continue
        rf_box = np.array([max(box[0], 0), max(box[1], 0), box[2], box[3]])

        mp_w, mp_h = mp_box[2] - mp_box[0], mp_box[3] - mp_box[1]
        records.append({
            # RetinaFace box expressed in units of the MediaPipe box
            "dx0": (rf_box[0] - mp_box[0]) / mp_w,
            "dy0": (rf_box[1] - mp_box[1]) / mp_h,
            "dx1": (rf_box[2] - mp_box[2]) / mp_w,
            "dy1": (rf_box[3] - mp_box[3]) / mp_h,
        })
    cap.release()

print(f"paired detections: {len(records)}")
cal = {}
for k in ("dx0", "dy0", "dx1", "dy1"):
    v = np.array([r[k] for r in records])
    cal[k] = float(np.median(v))
    print(f"  {k}: median {np.median(v):+.4f}  mean {v.mean():+.4f}  sd {v.std():.4f}")

print("\nDerived transform (MediaPipe landmark box -> RetinaFace-equivalent box):")
print(f"  x0' = x0 + {cal['dx0']:+.4f}*W   y0' = y0 + {cal['dy0']:+.4f}*H")
print(f"  x1' = x1 + {cal['dx1']:+.4f}*W   y1' = y1 + {cal['dy1']:+.4f}*H")
print("  (W, H = MediaPipe box width/height)")

with (PROJ / "outputs" / "p2_crop_calibration.json").open("w", encoding="utf-8") as fh:
    json.dump({"n_paired_detections": len(records), "transform": cal,
               "videos": [v.name for v in videos],
               "note": "median offsets of RetinaFace box relative to MediaPipe "
                       "landmark box, in units of the MediaPipe box size"}, fh, indent=2)
print(f"\nwritten: {PROJ / 'outputs' / 'p2_crop_calibration.json'}")
