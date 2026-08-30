"""P1 — core PAD hypothesis test: non-rigid deformation vs rigid motion.

Kollreider et al. [R25] and Anjos et al. [R27] characterise a photo presentation
attack as a *planar, rigid* object moved in front of the camera: all facial
points share one global transform. A live face additionally deforms
*non-rigidly* (eyelids, mouth, cheeks) — motion that no similarity transform
can explain.

Test: for consecutive detected frames within a capture, fit the optimal
similarity transform (Procrustes) between the two landmark constellations and
measure the RESIDUAL after alignment. Live > attack is the hypothesis.

This decides whether the P4 PAD module's rigidity cue is viable.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project")
PROJ = ROOT / "dms_project_fixed_v3" / "dms_jupyter"
NUAA = ROOT / "datasets" / "NormalizedFace"
OUT = PROJ / "outputs"

FNAME = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.bmp$", re.IGNORECASE)
# Landmark subsets: rigid skeleton (bone-anchored) vs deformable (soft tissue)
RIGID_IDS = [1, 4, 6, 9, 33, 133, 168, 263, 362]          # nose bridge, eye corners
DEFORM_IDS = [159, 145, 386, 374,                          # upper/lower eyelids
              13, 14, 61, 291,                             # lips
              50, 280]                                     # cheeks


def procrustes_residual(a: np.ndarray, b: np.ndarray) -> float:
    """Residual RMS after best similarity (rotation+scale+translation) fit a→b."""
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    norm_a = np.linalg.norm(a_c)
    if norm_a < 1e-9:
        return 0.0
    a_n, b_n = a_c / norm_a, b_c / (np.linalg.norm(b_c) + 1e-12)
    u, s, vt = np.linalg.svd(a_n.T @ b_n)
    r = (u @ vt).T
    aligned = a_n @ r.T * s.sum()
    return float(np.sqrt(np.mean(np.sum((aligned - b_n) ** 2, axis=1))))


def index_captures(root: Path):
    caps = defaultdict(list)
    for p in root.rglob("*.bmp"):
        m = FNAME.match(p.name)
        if m:
            caps[m.groups()[:4]].append((int(m.group(5)), p))
    for k in caps:
        caps[k].sort()
    return caps


options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
    num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)


def analyse(caps, label, max_captures=25, max_frames=40):
    rng = np.random.default_rng(42)
    keys = sorted(caps.keys())
    if len(keys) > max_captures:
        keys = [keys[i] for i in rng.choice(len(keys), max_captures, replace=False)]

    rows = []
    for key in keys:
        seq = []
        for _, path in caps[key][:max_frames]:
            img = cv2.imread(str(path))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
            if res.face_landmarks:
                seq.append(np.array([[p.x, p.y] for p in res.face_landmarks[0]]))
        if len(seq) < 6:
            continue

        full_res, deform_res, rigid_res = [], [], []
        for i in range(len(seq) - 1):
            full_res.append(procrustes_residual(seq[i], seq[i + 1]))
            deform_res.append(procrustes_residual(seq[i][DEFORM_IDS], seq[i + 1][DEFORM_IDS]))
            rigid_res.append(procrustes_residual(seq[i][RIGID_IDS], seq[i + 1][RIGID_IDS]))

        rows.append({
            "capture": "_".join(key),
            "pairs": len(full_res),
            "residual_full_median": float(np.median(full_res)),
            "residual_deformable_median": float(np.median(deform_res)),
            "residual_rigid_median": float(np.median(rigid_res)),
            # key discriminator: soft tissue moves MORE than bone on a live face
            "deform_over_rigid": float(np.median(deform_res) / (np.median(rigid_res) + 1e-9)),
        })
    print(f"{label}: {len(rows)} captures analysed")
    return rows


client = analyse(index_captures(NUAA / "ClientNormalized"), "NUAA client (live)")
imposter = analyse(index_captures(NUAA / "ImposterNormalized"), "NUAA imposter (print attack)")

print("\n" + "=" * 72)
print("NON-RIGID DEFORMATION TEST  (hypothesis: live > attack)")
print("=" * 72)
summary = {}
for key in ("residual_full_median", "residual_deformable_median",
            "residual_rigid_median", "deform_over_rigid"):
    c = np.array([r[key] for r in client])
    i = np.array([r[key] for r in imposter])
    pooled = np.sqrt((c.var() + i.var()) / 2)
    d = float((c.mean() - i.mean()) / pooled) if pooled > 1e-12 else 0.0
    # rank-based separability (AUC of the single cue), robust to outliers
    allv = np.concatenate([c, i])
    ranks = allv.argsort().argsort().astype(float) + 1
    auc = float((ranks[:len(c)].sum() - len(c) * (len(c) + 1) / 2) / (len(c) * len(i)))
    summary[key] = {"client_median": round(float(np.median(c)), 5),
                    "imposter_median": round(float(np.median(i)), 5),
                    "cohens_d": round(d, 3), "auc": round(auc, 3)}
    print(f"{key:>28}: live {np.median(c):.5f} | attack {np.median(i):.5f} | "
          f"d {d:+.3f} | AUC {auc:.3f}")

with (OUT / "p1_rigidity_results.json").open("w", encoding="utf-8") as fh:
    json.dump({"summary": summary, "client": client, "imposter": imposter}, fh, indent=2)
print(f"\nwritten: {OUT / 'p1_rigidity_results.json'}")
