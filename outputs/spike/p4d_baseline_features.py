"""Gate G3, step 1 — extract face volumes and LBP features for the DR-4 baselines.

Two published texture baselines, re-implemented from their source papers:

  * Maatta, Hadid & Pietikainen (IJCB 2011, R73) — MULTI-SCALE LBP on the face
    image. Faithful form: uniform LBP(8,1) computed over a 3x3 grid of
    overlapping blocks (spatially enhanced), concatenated with whole-face
    uniform LBP(8,2) and LBP(16,2) histograms -> SVM. Single-frame method.

  * de Freitas Pereira, Anjos, De Martino & Marcel (ACCV-W 2012, R74) — LBP-TOP:
    uniform LBP on Three Orthogonal Planes (XY, XT, YT) over a spatio-temporal
    volume, the three histograms concatenated -> SVM. Temporal method, and the
    like-for-like comparison for our temporal cues.

Protocol (DP-8 Option C): SUBJECT-DISJOINT split. Baselines are fitted only on
train subjects; every reported number is on held-out test subjects. Our own
method is re-evaluated on the SAME test subjects so the comparison is fair —
its threshold calibration likewise uses only TRAIN genuine.

Faces are located with the project's own MediaPipe landmarker (the same
perception front-end the rest of the pipeline uses), cropped to 64x64 grayscale.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from skimage.feature import local_binary_pattern

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
ROSE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")
CODES = {"G": "genuine", "Ps": "print_still", "Pq": "print_quiver", "Vl": "replay_lenovo",
         "Vm": "replay_mac", "Mc": "mask_cropped", "Mf": "mask_full", "Mu": "mask_upper"}
CROP, VOL_T = 64, 16          # 64x64 grayscale crops; 16-frame volumes for LBP-TOP
VOLS_PER_CLIP = 3
CLIPS_PER_SPECIES = 90
SEED = 42

# Subject-disjoint split (sorted numerically for determinism)
SUBJECTS = sorted([d.name for d in ROSE.iterdir() if d.is_dir()], key=int)
TRAIN_SUBJ = set(SUBJECTS[:10])
TEST_SUBJ = set(SUBJECTS[10:])


def make_landmarker():
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
            running_mode=vision.RunningMode.VIDEO, num_faces=1))


def face_volumes(clip: Path):
    """Return up to VOLS_PER_CLIP volumes of (VOL_T, CROP, CROP) uint8 grayscale."""
    lm = make_landmarker()
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    crops, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        r = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=np.ascontiguousarray(rgb)),
                                int(i / fps * 1000))
        if r.face_landmarks:
            px = np.array([[p.x * w, p.y * h] for p in r.face_landmarks[0]])
            x0, y0 = max(int(px[:, 0].min()), 0), max(int(px[:, 1].min()), 0)
            x1, y1 = min(int(px[:, 0].max()), w), min(int(px[:, 1].max()), h)
            if x1 > x0 + 16 and y1 > y0 + 16:
                g = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                crops.append(cv2.resize(g, (CROP, CROP), interpolation=cv2.INTER_AREA))
        i += 1
    cap.release()
    if len(crops) < VOL_T:
        return []
    starts = np.linspace(0, len(crops) - VOL_T, VOLS_PER_CLIP).astype(int)
    return [np.stack(crops[s:s + VOL_T]) for s in sorted(set(starts.tolist()))]


def _uniform_hist(plane, P, R):
    lbp = local_binary_pattern(plane, P, R, method="uniform")
    h, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2))
    return h.astype(np.float32) / (h.sum() + 1e-9)


def maatta_features(img):
    """R73: spatially-enhanced LBP(8,1) over a 3x3 grid + whole-face (8,2) and (16,2)."""
    feats, s = [], CROP // 3
    for by in range(3):
        for bx in range(3):
            block = img[by * s:(by + 1) * s, bx * s:(bx + 1) * s]
            feats.append(_uniform_hist(block, 8, 1))
    feats.append(_uniform_hist(img, 8, 2))
    feats.append(_uniform_hist(img, 16, 2))
    return np.concatenate(feats)


def lbptop_features(vol):
    """R74: uniform LBP on the XY, XT and YT planes of a spatio-temporal volume."""
    t, h, w = vol.shape
    xy = _uniform_hist(vol[t // 2], 8, 1)                    # central spatial slice
    xt = _uniform_hist(vol[:, h // 2, :], 8, 1)              # horizontal slice over time
    yt = _uniform_hist(vol[:, :, w // 2], 8, 1)              # vertical slice over time
    return np.concatenate([xy, xt, yt])


def main():
    rng = random.Random(SEED)
    by_sp_subj = defaultdict(lambda: defaultdict(list))
    for v in sorted(ROSE.rglob("*.mp4")):
        sp = CODES.get(v.stem.split("_")[0])
        if sp:
            by_sp_subj[sp][v.parent.name].append(v)

    rows = []
    for sp, per_subj in sorted(by_sp_subj.items()):
        picked = []
        subs = sorted(per_subj, key=int)
        pools = {s: rng.sample(per_subj[s], len(per_subj[s])) for s in subs}
        i = 0
        while len(picked) < CLIPS_PER_SPECIES and any(pools[s] for s in subs):
            s = subs[i % len(subs)]
            if pools[s]:
                picked.append(pools[s].pop())
            i += 1
        for n, clip in enumerate(picked, 1):
            vols = face_volumes(clip)
            for vol in vols:
                rows.append({
                    "species": sp, "label": 0 if sp == "genuine" else 1,
                    "subject": clip.parent.name,
                    "split": "train" if clip.parent.name in TRAIN_SUBJ else "test",
                    "maatta": maatta_features(vol[VOL_T // 2]).tolist(),
                    "lbptop": lbptop_features(vol).tolist(),
                })
            if n % 30 == 0:
                print(f"  {sp:<15} {n}/{len(picked)} clips, {len(rows)} volumes", flush=True)

    out = PROJ / "outputs" / "p4d_baseline_features.json"
    out.write_text(json.dumps({
        "seed": SEED, "crop": CROP, "vol_t": VOL_T,
        "train_subjects": sorted(TRAIN_SUBJ, key=int),
        "test_subjects": sorted(TEST_SUBJ, key=int),
        "n": len(rows), "rows": rows}), encoding="utf-8")
    n_tr = sum(r["split"] == "train" for r in rows)
    print(f"\n{len(rows)} volumes ({n_tr} train / {len(rows)-n_tr} test); "
          f"maatta dim {len(rows[0]['maatta'])}, lbptop dim {len(rows[0]['lbptop'])}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
