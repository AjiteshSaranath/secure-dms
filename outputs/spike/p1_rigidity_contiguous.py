"""P1 — diagnostic: is the negative rigidity result caused by NUAA's sampling gaps?

p1_rigidity.py compared CONSECUTIVE STORED frames, but stored frames are not
consecutive CAPTURED frames: client captures are 61.3 % contiguous, imposter
captures only 0.8 % (p1_eda.py). A gap of 6 frames lets the attacker move the
photo much further, inflating apparent deformation for the attack class.

This script recomputes using ONLY pairs whose frame numbers differ by exactly 1.
If the imposter class has too few such pairs to measure, that is itself the
finding: NUAA cannot support temporal PAD evaluation.
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
DEFORM_IDS = [159, 145, 386, 374, 13, 14, 61, 291, 50, 280]
RIGID_IDS = [1, 4, 6, 9, 33, 133, 168, 263, 362]


def procrustes_residual(a, b):
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    na = np.linalg.norm(a_c)
    if na < 1e-9:
        return 0.0
    a_n, b_n = a_c / na, b_c / (np.linalg.norm(b_c) + 1e-12)
    u, s, vt = np.linalg.svd(a_n.T @ b_n)
    aligned = a_n @ (u @ vt).T.T * s.sum()
    return float(np.sqrt(np.mean(np.sum((aligned - b_n) ** 2, axis=1))))


def index_captures(root):
    caps = defaultdict(list)
    for p in root.rglob("*.bmp"):
        m = FNAME.match(p.name)
        if m:
            caps[m.groups()[:4]].append((int(m.group(5)), p))
    for k in caps:
        caps[k].sort()
    return caps


landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(PROJ / "models" / "face_landmarker.task")),
        num_faces=1))


def landmarks_of(path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(rgb)))
    if not res.face_landmarks:
        return None
    return np.array([[p.x, p.y] for p in res.face_landmarks[0]])


def analyse(caps, label, max_captures=40, max_pairs_per_capture=25):
    rng = np.random.default_rng(42)
    keys = sorted(caps.keys())
    if len(keys) > max_captures:
        keys = [keys[i] for i in rng.choice(len(keys), max_captures, replace=False)]

    total_adjacent = 0
    deform, rigid, ratio = [], [], []
    for key in keys:
        frames = caps[key]
        pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)
                 if frames[i + 1][0] - frames[i][0] == 1][:max_pairs_per_capture]
        total_adjacent += len(pairs)
        for (_, pa), (_, pb) in pairs:
            la, lb = landmarks_of(pa), landmarks_of(pb)
            if la is None or lb is None:
                continue
            d = procrustes_residual(la[DEFORM_IDS], lb[DEFORM_IDS])
            r = procrustes_residual(la[RIGID_IDS], lb[RIGID_IDS])
            deform.append(d)
            rigid.append(r)
            ratio.append(d / (r + 1e-9))

    print(f"{label}: {len(keys)} captures, {total_adjacent} truly-adjacent pairs, "
          f"{len(deform)} usable (both frames detected)")
    return {"adjacent_pairs": total_adjacent, "usable_pairs": len(deform),
            "deform": deform, "rigid": rigid, "ratio": ratio}


client = analyse(index_captures(NUAA / "ClientNormalized"), "NUAA client (live)")
imposter = analyse(index_captures(NUAA / "ImposterNormalized"), "NUAA imposter (attack)")

print("\n" + "=" * 72)
print("CONTIGUOUS-PAIRS-ONLY COMPARISON")
print("=" * 72)

out = {"client_pairs": client["usable_pairs"], "imposter_pairs": imposter["usable_pairs"]}
if imposter["usable_pairs"] < 30:
    print(f"INSUFFICIENT DATA: only {imposter['usable_pairs']} usable contiguous attack pairs "
          f"(need >=30 for a meaningful comparison).")
    print("=> CONFIRMED: NUAA's imposter set has no temporal continuity; the earlier")
    print("   negative result is a sampling artefact, NOT evidence against the cue.")
    print("=> GATE G1 VERDICT: NUAA cannot serve as the temporal-PAD corpus.")
    out["verdict"] = "NUAA lacks temporal continuity for attack class; cannot test temporal cues"
else:
    for name in ("deform", "rigid", "ratio"):
        c, i = np.array(client[name]), np.array(imposter[name])
        pooled = np.sqrt((c.var() + i.var()) / 2)
        d = float((c.mean() - i.mean()) / pooled) if pooled > 1e-12 else 0.0
        allv = np.concatenate([c, i])
        ranks = allv.argsort().argsort().astype(float) + 1
        auc = float((ranks[:len(c)].sum() - len(c) * (len(c) + 1) / 2) / (len(c) * len(i)))
        out[name] = {"client_median": round(float(np.median(c)), 5),
                     "imposter_median": round(float(np.median(i)), 5),
                     "cohens_d": round(d, 3), "auc": round(auc, 3)}
        print(f"{name:>10}: live {np.median(c):.5f} | attack {np.median(i):.5f} | "
              f"d {d:+.3f} | AUC {auc:.3f}")

with (OUT / "p1_rigidity_contiguous.json").open("w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=2)
print(f"\nwritten: {OUT / 'p1_rigidity_contiguous.json'}")
