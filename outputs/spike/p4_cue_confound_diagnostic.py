"""P4 diagnostic — two PAD cues are confounded by instrument motion.

First PAD run: ACER 0.478, with APCER 0.917 on handheld print attacks. The blink
cue works (89 % live vs 0-8 % attack); the other two invert or collapse:

  GAZE VARIABILITY   fires "live" on 75-100 % of ATTACK windows.
    Cause: gaze is estimated in CAMERA coordinates, so it is head orientation
    combined with eye-in-head rotation. Moving a photograph changes the apparent
    head orientation, hence the apparent gaze — the cue is measuring instrument
    motion, not ocular motion. The gaze-PAD literature (R26, R28) measures OCULAR
    movement specifically. Correct measure: gaze MINUS head pose, which is
    constant for a photograph because the eyes are painted on the surface.
    (This is the same camera-frame confound found in P3 for distraction.)

  DEFORMATION        fires "live" on 92 % of HANDHELD attack windows but only
    0 % of static ones — i.e. it tracks how much the instrument moved, not
    whether the face deformed. Larger warps mean more resampling and landmark
    jitter. Correct measure: the residual of DEFORMABLE landmarks RELATIVE to
    RIGID ones, which normalises out overall motion (the deform/rigid ratio
    already computed in the P1 rigidity analysis).

Both corrections are definitional — they are how the cited cues are defined —
rather than threshold tuning. This script measures the corrected cues on both
classes so thresholds can then be argued from physics rather than fitted.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
TIER_C = PROJ / "outputs" / "tier_c_attacks"
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"
WINDOW_S = 10.0


def cache_path(v):
    key = f"{v.name}|{v.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def procrustes_residual(a, b):
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    na = np.linalg.norm(a_c)
    if na < 1e-9:
        return 0.0
    a_n, b_n = a_c / na, b_c / (np.linalg.norm(b_c) + 1e-12)
    u, s, vt = np.linalg.svd(a_n.T @ b_n)
    return float(np.sqrt(np.mean(np.sum((a_n @ (u @ vt).T.T * s.sum() - b_n) ** 2, axis=1))))


def window_cues(d, lo, hi):
    gy, gp = d["gaze_yaw"][lo:hi], d["gaze_pitch"][lo:hi]
    hy, hp = d["head_yaw"][lo:hi], d["head_pitch"][lo:hi]
    ok = (np.abs(gy) <= 90) & (np.abs(gp) <= 90) & (np.abs(hy) <= 90) & (np.abs(hp) <= 90)
    if ok.sum() < 3:
        return None
    # camera-frame variability (the broken cue)
    cam = float(np.hypot(np.std(gy[ok]), np.std(gp[ok])))
    # eye-in-head variability (the corrected cue)
    eih = float(np.hypot(np.std((gy - hy)[ok]), np.std((gp - hp)[ok])))

    dfm, rgd = d["deformable"][lo:hi], d["rigid"][lo:hi]
    dres = [procrustes_residual(dfm[i], dfm[i + 1]) for i in range(len(dfm) - 1)]
    rres = [procrustes_residual(rgd[i], rgd[i + 1]) for i in range(len(rgd) - 1)]
    dmed = float(np.median(dres)) if dres else 0.0
    rmed = float(np.median(rres)) if rres else 0.0
    return {"gaze_camera": cam, "gaze_eye_in_head": eih,
            "deform_abs": dmed, "deform_ratio": dmed / (rmed + 1e-9)}


def windows_of(path):
    p = cache_path(path)
    if not p.exists():
        return []
    d = np.load(p)
    fps = float(d["fps"][0])
    w = max(int(round(WINDOW_S * fps)), 2)
    n = len(d["ear_left"])
    out = []
    for lo in range(0, n - w + 1, w):
        c = window_cues(d, lo, lo + w)
        if c:
            out.append(c)
    return out


groups = defaultdict(list)
for split in ("Drowsiness", "Distraction"):
    for v in sorted((DMD / split).rglob("*rgb_face.mp4")):
        groups["bona fide"].extend(windows_of(v))
manifest = json.loads((TIER_C / "manifest.json").read_text(encoding="utf-8"))
for item in manifest["items"]:
    groups[item["species"]].extend(windows_of(TIER_C / item["file"]))

print(f"{'group':<18}{'windows':>9}{'gaze CAM':>11}{'gaze EYE-IN-HEAD':>18}"
      f"{'deform abs':>13}{'deform RATIO':>14}")
print("-" * 83)
for name, ws in groups.items():
    if not ws:
        continue
    print(f"{name:<18}{len(ws):>9}"
          f"{np.median([w['gaze_camera'] for w in ws]):>11.3f}"
          f"{np.median([w['gaze_eye_in_head'] for w in ws]):>18.3f}"
          f"{np.median([w['deform_abs'] for w in ws]):>13.6f}"
          f"{np.median([w['deform_ratio'] for w in ws]):>14.3f}")

# separability of each candidate cue (rank AUC, bona fide = positive/live)
bona = groups["bona fide"]
atk = [w for k, ws in groups.items() if k != "bona fide" for w in ws]
print(f"\nsingle-cue separability (AUC, live vs all attacks; 1.0 = perfect):")
for key in ("gaze_camera", "gaze_eye_in_head", "deform_abs", "deform_ratio"):
    b = np.array([w[key] for w in bona])
    a = np.array([w[key] for w in atk])
    allv = np.concatenate([b, a])
    ranks = allv.argsort().argsort().astype(float) + 1
    auc = (ranks[:len(b)].sum() - len(b) * (len(b) + 1) / 2) / (len(b) * len(a))
    print(f"  {key:<20} AUC {auc:.4f}"
          f"{'   <- currently deployed' if key in ('gaze_camera','deform_abs') else '   <- corrected'}")
