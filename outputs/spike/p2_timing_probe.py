"""P2 probe — throughput and detection quality on a real DMD video.

Sizes the full extraction before committing hours of compute, and checks that
the three pretrained models behave on 1280x720 driver footage (all P0 spikes
ran on 224x224 cached crops, which is a different regime).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")

video = sorted(DMD_DROW.rglob("*rgb_face.mp4"))[0]
print(f"video: {video.name}")

N = 300  # ~10 s at 29.8 fps

# --- MediaPipe (VIDEO mode: uses temporal tracking, faster + smoother) -----
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_facial_transformation_matrixes=True))

cap = cv2.VideoCapture(str(video))
fps = cap.get(cv2.CAP_PROP_FPS)
frames_bgr, detected = [], 0
t0 = time.perf_counter()
for i in range(N):
    ok, bgr = cap.read()
    if not ok:
        break
    frames_bgr.append(bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
        int(i / fps * 1000))
    if res.face_landmarks:
        detected += 1
cap.release()
t_mp = time.perf_counter() - t0
print(f"MediaPipe VIDEO mode: {len(frames_bgr)} frames in {t_mp:.1f}s "
      f"= {len(frames_bgr)/t_mp:.1f} fps | face detected {detected}/{len(frames_bgr)} "
      f"({detected/len(frames_bgr):.1%})")

# --- face crop for the torch models (they expect a face box) ---------------
# Use MediaPipe's landmarks to crop, avoiding a second face detector.
def crop_from_landmarks(bgr, lm, pad=0.35):
    h, w = bgr.shape[:2]
    xs = np.array([p.x for p in lm]) * w
    ys = np.array([p.y for p in lm]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1 = int(max(0, x0 - mx)), int(min(w, x1 + mx))
    y0, y1 = int(max(0, y0 - my)), int(min(h, y1 + my))
    return bgr[y0:y1, x0:x1]

device = torch.device("cuda")

from sixdrepnet import SixDRepNet
pose_model = SixDRepNet()
sample = frames_bgr[:60]
crops = []
cap = cv2.VideoCapture(str(video))
landmarker2 = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
        num_faces=1))
for bgr in sample:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    r = landmarker2.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=np.ascontiguousarray(rgb)))
    if r.face_landmarks:
        crops.append(crop_from_landmarks(bgr, r.face_landmarks[0]))
cap.release()
print(f"face crops obtained: {len(crops)}/{len(sample)} | "
      f"median crop size {np.median([c.shape[0] for c in crops]):.0f}x"
      f"{np.median([c.shape[1] for c in crops]):.0f}")

t0 = time.perf_counter()
poses = [pose_model.predict(c) for c in crops]
t_pose = time.perf_counter() - t0
p, y, r = (np.array([float(np.ravel(v[i])[0]) for v in poses]) for i in range(3))
print(f"6DRepNet: {len(crops)} crops in {t_pose:.1f}s = {len(crops)/t_pose:.1f} fps | "
      f"pitch {p.mean():+.1f}+-{p.std():.1f} yaw {y.mean():+.1f}+-{y.std():.1f} "
      f"roll {r.mean():+.1f}+-{r.std():.1f}")

from l2cs import Pipeline as GazePipeline
gaze = GazePipeline(weights=str(PROJ / "models" / "L2CSNet_gaze360.pkl"),
                    arch="ResNet50", device=device)
t0 = time.perf_counter()
pitches, yaws, ok_n = [], [], 0
for bgr in sample:
    try:
        g = gaze.step(bgr)
        if len(g.pitch):
            pitches.append(float(np.degrees(g.pitch[0])))
            yaws.append(float(np.degrees(g.yaw[0])))
            ok_n += 1
    except ValueError:
        pass
t_gaze = time.perf_counter() - t0
print(f"L2CS-Net: {len(sample)} frames in {t_gaze:.1f}s = {len(sample)/t_gaze:.1f} fps | "
      f"detected {ok_n}/{len(sample)} | gaze pitch {np.mean(pitches):+.1f}+-{np.std(pitches):.1f} "
      f"yaw {np.mean(yaws):+.1f}+-{np.std(yaws):.1f}")

# --- projection --------------------------------------------------------------
total_drow = 87071
total_dist = 82 * 6777
slowest = min(len(frames_bgr)/t_mp, len(crops)/t_pose, len(sample)/t_gaze)
print(f"\nfull-pipeline throughput bound: {slowest:.1f} fps")
print(f"projected: drowsiness {total_drow:,} frames -> {total_drow/slowest/60:.0f} min")
print(f"projected: distraction {total_dist:,} frames -> {total_dist/slowest/3600:.1f} h")
