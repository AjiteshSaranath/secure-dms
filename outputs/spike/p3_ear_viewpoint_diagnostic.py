"""P3 diagnostic — EAR > 1.0 on 23 distraction sessions: bug or viewpoint effect?

The plausibility guard fired on 23 of 82 distraction sessions with "EAR exceeds
1.0". That guard was written for the p2.1 coordinate bug, which is fixed, so the
message is now misleading and the cause must be established.

Hypothesis: EAR is a ratio of VERTICAL eyelid separation to HORIZONTAL corner
separation. Under large head yaw the eye foreshortens horizontally far more than
vertically, so the denominator shrinks and EAR inflates. Distraction sessions
contain large head turns (reach_side, talking_to_passenger) that the drowsiness
sessions did not — which is why this surfaced only now.

If confirmed this is the viewpoint dependence R14 documents ("EAR is inherently
viewpoint dependent, relying on 2D landmark distances that distort under head
rotation"), i.e. a real limitation to mask and report, not a coding error.
"""
import hashlib
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DIST = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Distraction")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def collect(root, limit=None):
    out = []
    for video in sorted(root.rglob("*rgb_face.mp4")):
        p = cache_path(video)
        if p.exists():
            out.append((("/".join(video.parts[-4:-1])), np.load(p)))
        if limit and len(out) >= limit:
            break
    return out


dist = collect(DMD_DIST)
drow = collect(DMD_DROW)
print(f"cached sessions — distraction {len(dist)}, drowsiness {len(drow)}\n")

# --- head-yaw distributions: is distraction genuinely more extreme? --------
for label, rows in (("drowsiness", drow), ("distraction", dist)):
    yaw = np.concatenate([np.abs(d["head_yaw"]) for _, d in rows])
    print(f"{label:<12} |head yaw|: mean {yaw.mean():5.1f}  p95 {np.percentile(yaw,95):5.1f}  "
          f"p99 {np.percentile(yaw,99):5.1f}  max {yaw.max():5.1f} deg  "
          f"| frames >40 deg: {np.mean(yaw > 40):.1%}")

# --- does EAR inflation track head yaw? -----------------------------------
ear_all, yaw_all = [], []
for _, d in dist:
    ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)
    ear_all.append(ear)
    yaw_all.append(np.abs(d["head_yaw"]))
ear = np.concatenate(ear_all)
yaw = np.concatenate(yaw_all)
ok = np.isfinite(ear) & np.isfinite(yaw)
ear, yaw = ear[ok], yaw[ok]

print(f"\ndistraction frames analysed: {len(ear):,}")
print(f"EAR > 1.0: {np.sum(ear > 1.0):,} frames ({np.mean(ear > 1.0):.3%})")
print(f"EAR > 0.6: {np.sum(ear > 0.6):,} frames ({np.mean(ear > 0.6):.3%})")

print(f"\n{'|head yaw| band':>18}{'frames':>10}{'mean EAR':>10}{'p99 EAR':>9}{'% EAR>0.6':>11}")
bands = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 90)]
for lo, hi in bands:
    m = (yaw >= lo) & (yaw < hi)
    if m.sum() < 50:
        continue
    print(f"{f'{lo}-{hi}':>18}{m.sum():>10,}{ear[m].mean():>10.3f}"
          f"{np.percentile(ear[m], 99):>9.3f}{np.mean(ear[m] > 0.6):>10.2%}")

r = float(np.corrcoef(yaw, ear)[0, 1])
print(f"\ncorrelation |head yaw| vs EAR: {r:+.4f}")

hi_ear = ear > 1.0
if hi_ear.any():
    print(f"mean |head yaw| where EAR>1.0: {yaw[hi_ear].mean():.1f} deg "
          f"vs {yaw[~hi_ear].mean():.1f} deg elsewhere")

# --- what yaw limit keeps EAR physiological? ------------------------------
print(f"\n{'yaw limit':>10}{'frames kept':>13}{'% kept':>9}{'max EAR kept':>14}")
for lim in (25, 30, 35, 40, 45, 50, 90):
    keep = yaw <= lim
    print(f"{lim:>10}{keep.sum():>13,}{keep.mean():>8.1%}{ear[keep].max():>14.3f}")
