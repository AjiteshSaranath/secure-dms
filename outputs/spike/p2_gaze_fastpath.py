"""P2 — fast L2CS-Net path, validated against the reference implementation.

Problem: l2cs.Pipeline runs RetinaFace on every full 1280x720 frame -> 0.8 fps,
i.e. ~192 h for the DMD distraction set. But MediaPipe (DR-1) already localises
the face, so the second detector is redundant.

Fix: call the L2CS model directly on MediaPipe-derived crops, batched on GPU,
reproducing the paper's decoding exactly (softmax over 90 bins, *4 - 180 deg).

Fidelity is not assumed — it is measured. L2CS was trained on RetinaFace-style
crops, so this script quantifies the agreement between the fast path and the
reference Pipeline on identical frames before the fast path is adopted.
"""
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from l2cs.utils import getArch, prep_input_numpy
from l2cs import Pipeline as RefPipeline

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
WEIGHTS = PROJ / "models" / "L2CSNet_gaze360.pkl"
device = torch.device("cuda")
N = 120

video = sorted(DMD_DROW.rglob("*rgb_face.mp4"))[0]
cap = cv2.VideoCapture(str(video))
frames = []
for _ in range(N):
    ok, bgr = cap.read()
    if not ok:
        break
    frames.append(bgr)
cap.release()
print(f"{len(frames)} frames from {video.name}")

# --- MediaPipe face boxes ---------------------------------------------------
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))


# Calibrated MediaPipe -> RetinaFace box transform (p2_crop_calibration.py,
# 120 paired detections). Offsets are in units of the MediaPipe box size.
CAL = {"dx0": -0.0031, "dy0": -0.1193, "dx1": -0.0038, "dy1": +0.0234}


def face_box(lm, w, h):
    xs, ys = np.array([p.x for p in lm]) * w, np.array([p.y for p in lm]) * h
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = x1 - x0, y1 - y0
    return (int(max(0, x0 + CAL["dx0"] * bw)), int(max(0, y0 + CAL["dy0"] * bh)),
            int(min(w, x1 + CAL["dx1"] * bw)), int(min(h, y1 + CAL["dy1"] * bh)))


crops, idx_ok = [], []
for i, bgr in enumerate(frames):
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)),
        int(i / 29.8 * 1000))
    if res.face_landmarks:
        x0, y0, x1, y1 = face_box(res.face_landmarks[0], w, h)
        crop = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        crops.append(cv2.resize(crop, (224, 224)))
        idx_ok.append(i)
print(f"MediaPipe crops: {len(crops)}/{len(frames)}")

# --- fast path: model called directly, batched -----------------------------
model = getArch("ResNet50", 90)
model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
model.to(device).eval()
softmax = nn.Softmax(dim=1)
idx_tensor = torch.arange(90, dtype=torch.float32, device=device)


@torch.no_grad()
def gaze_batch(crop_list, batch_size=32):
    out_p, out_y = [], []
    for i in range(0, len(crop_list), batch_size):
        batch = np.stack(crop_list[i:i + batch_size])
        x = prep_input_numpy(batch, device)
        gp, gy = model(x)
        # published L2CS decoding: expectation over 90 bins of 4 deg, offset -180
        pitch = torch.sum(softmax(gp) * idx_tensor, dim=1) * 4 - 180
        yaw = torch.sum(softmax(gy) * idx_tensor, dim=1) * 4 - 180
        out_p.append(pitch.cpu().numpy())
        out_y.append(yaw.cpu().numpy())
    return np.concatenate(out_p), np.concatenate(out_y)


torch.cuda.synchronize()
t0 = time.perf_counter()
fast_pitch, fast_yaw = gaze_batch(crops)
torch.cuda.synchronize()
t_fast = time.perf_counter() - t0
print(f"FAST path: {len(crops)} frames in {t_fast:.2f}s = {len(crops)/t_fast:.0f} fps")

# --- reference path for fidelity comparison --------------------------------
ref = RefPipeline(weights=str(WEIGHTS), arch="ResNet50", device=device)
t0 = time.perf_counter()
ref_pitch, ref_yaw, ref_idx = [], [], []
for i in idx_ok:
    try:
        g = ref.step(frames[i])
        if len(g.pitch):
            ref_pitch.append(np.degrees(float(g.pitch[0])))
            ref_yaw.append(np.degrees(float(g.yaw[0])))
            ref_idx.append(i)
    except ValueError:
        pass
t_ref = time.perf_counter() - t0
print(f"REF path : {len(ref_idx)} frames in {t_ref:.2f}s = {len(ref_idx)/t_ref:.1f} fps")
print(f"speed-up : {(len(crops)/t_fast) / (len(ref_idx)/t_ref):.0f}x")

# --- agreement --------------------------------------------------------------
pos = {v: k for k, v in enumerate(idx_ok)}
sel = [pos[i] for i in ref_idx]
fp, fy = fast_pitch[sel], fast_yaw[sel]
rp, ry = np.array(ref_pitch), np.array(ref_yaw)

print("\n" + "=" * 66)
print("FIDELITY: fast path (MediaPipe crops) vs reference (RetinaFace crops)")
print("=" * 66)
for name, a, b in (("pitch", fp, rp), ("yaw", fy, ry)):
    diff = a - b
    print(f"{name}: fast {a.mean():+.2f}+-{a.std():.2f} | ref {b.mean():+.2f}+-{b.std():.2f} | "
          f"bias {diff.mean():+.2f} deg | MAE {np.abs(diff).mean():.2f} deg | "
          f"r {np.corrcoef(a, b)[0,1]:.4f}")

# Acceptance criteria, justified rather than arbitrary:
#  (1) the fast-vs-reference discrepancy must be small relative to L2CS-Net's OWN
#      published accuracy - 10.41 deg mean angular error on Gaze360 (ledger R29).
#      A discrepancy well inside the model's own error budget cannot change any
#      downstream decision the model is trusted to make. Threshold: < 20 % = 2.08 deg.
#  (2) temporal dispersion must be preserved, because the PAD module consumes gaze
#      VARIABILITY over time, not absolute angles. Threshold: sd ratio within 10 %.
PUBLISHED_ERROR_DEG = 10.41
mae = max(np.abs(fp - rp).mean(), np.abs(fy - ry).mean())
sd_ratio = max(abs(fp.std() / rp.std() - 1), abs(fy.std() / ry.std() - 1))

print(f"\nacceptance vs L2CS published Gaze360 error {PUBLISHED_ERROR_DEG} deg (ledger R29):")
print(f"  (1) MAE {mae:.2f} deg = {mae / PUBLISHED_ERROR_DEG:.0%} of model's own error "
      f"(limit 20%) -> {'ok' if mae < 0.2 * PUBLISHED_ERROR_DEG else 'FAIL'}")
print(f"  (2) temporal sd preserved within {sd_ratio:.1%} (limit 10%) "
      f"-> {'ok' if sd_ratio < 0.10 else 'FAIL'}")
print("PASS — fast path adopted" if (mae < 0.2 * PUBLISHED_ERROR_DEG and sd_ratio < 0.10)
      else "REVIEW — discrepancy is material; document or recalibrate")
