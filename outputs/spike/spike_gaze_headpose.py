"""P0 spike 2: L2CS-Net gaze + 6DRepNet head pose on cached DMD frames.

Verifies: weights load (torch>=2.6 weights_only pickle safety), CUDA inference
works, outputs are physiologically plausible angles on real project data.
"""
import glob

import numpy as np
import cv2
import torch

CACHE_GLOB = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_project\outputs\dmd_cache\*.npz"
WEIGHTS = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\models\L2CSNet_gaze360.pkl"

frames = np.load(sorted(glob.glob(CACHE_GLOB))[0])["frames"]
samples = [np.ascontiguousarray(frames[i]) for i in range(0, 50, 10)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# --- L2CS-Net ---------------------------------------------------------------
from l2cs import Pipeline

gaze_pipeline = Pipeline(weights=WEIGHTS, arch="ResNet50", device=device)
ok = 0
for frame in samples:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    try:
        res = gaze_pipeline.step(bgr)
        if len(res.pitch):
            p, y = float(np.degrees(res.pitch[0])), float(np.degrees(res.yaw[0]))
            print(f"L2CS gaze: pitch={p:+.1f} deg yaw={y:+.1f} deg")
            if abs(p) < 90 and abs(y) < 90:
                ok += 1
    except ValueError:
        print("L2CS: no face detected in frame")
print(f"L2CS plausible outputs: {ok}/{len(samples)}")

# --- 6DRepNet ---------------------------------------------------------------
from sixdrepnet import SixDRepNet

pose_model = SixDRepNet()  # auto-downloads official weights on first use
ok6 = 0
for frame in samples:
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    pitch, yaw, roll = pose_model.predict(bgr)
    p, y, r = (float(np.ravel(v)[0]) for v in (pitch, yaw, roll))
    print(f"6DRepNet pose: pitch={p:+.1f} yaw={y:+.1f} roll={r:+.1f}")
    if all(abs(v) < 100 for v in (p, y, r)):
        ok6 += 1
print(f"6DRepNet plausible outputs: {ok6}/{len(samples)}")

print("PASS" if ok >= 3 and ok6 >= 4 else "FAIL")
