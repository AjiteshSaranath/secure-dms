"""P9 step 5 — what changing the yawn threshold does to the DEPLOYED verdict.

The deployed rule is "closure proportion OR sustained yawn", and Sec. VIII
reports its flag rates.  Changing the yawn threshold therefore moves a headline
number, so the consequence is computed before anything is changed.

Reproduces the deployed driver-state logic on the cached signals.
"""
import json, hashlib
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "outputs" / "signal_cache"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes"
           r"\Case Study in Cybersecurity Analytics\DMD Dataset")
EXTRACTOR_VERSION = "p2.3"

OPEN_EYE_PERCENTILE = 85.0
PERCLOS_CLOSURE_FRACTION = 0.20
PERCLOS_WINDOW_S = 60.0
PERCLOS_DROWSY = 0.15
MIN_MS = 1000.0


def cache_path(v):
    key = f"{v.name}|{v.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def runs_of_true(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def perclos_flag(ear, fps):
    ok = np.isfinite(ear)
    if ok.sum() < 10:
        return False
    base = np.percentile(ear[ok], OPEN_EYE_PERCENTILE)
    closed = np.zeros(len(ear), bool)
    closed[ok] = ear[ok] <= PERCLOS_CLOSURE_FRACTION * base
    w = max(int(round(PERCLOS_WINDOW_S * fps)), 2)
    if len(closed) < w:
        return closed.mean() > PERCLOS_DROWSY
    c = np.convolve(closed.astype(float), np.ones(w) / w, mode="valid")
    return bool(np.nanmax(c) > PERCLOS_DROWSY)


def yawn_flag(mar, fps, thr):
    ok = np.isfinite(mar); m = np.zeros(len(mar), bool)
    m[ok] = mar[ok] >= thr
    return any((b - a + 1) / fps * 1000.0 >= MIN_MS for a, b in runs_of_true(m))


def load(split):
    out = []
    for v in sorted((DMD / split).rglob("*rgb_face.mp4")):
        cp = cache_path(v)
        if cp.exists():
            d = np.load(cp)
            ear = np.nanmean(np.vstack([d["ear_left"].astype(float),
                                        d["ear_right"].astype(float)]), axis=0)
            out.append((ear, d["mar"].astype(float), float(d["fps"][0])))
    return out


drowsy, control = load("drowsiness"), load("distraction")
print(f"{len(drowsy)} drowsiness, {len(control)} control sessions\n")

pd_ = [perclos_flag(e, f) for e, m, f in drowsy]
pc_ = [perclos_flag(e, f) for e, m, f in control]
print(f"closure proportion alone:  drowsy {np.mean(pd_)*100:5.1f}%   "
      f"control {np.mean(pc_)*100:5.1f}%   sep {(np.mean(pd_)-np.mean(pc_))*100:5.1f}")

print(f"\n{'MAR thr':>8}{'yawn drowsy':>13}{'yawn ctrl':>11}"
      f"{'OR drowsy':>11}{'OR ctrl':>9}{'OR sep':>8}")
print("-" * 62)
res = {}
for thr in (0.30, 0.35, 0.40, 0.50, 0.60):
    yd = [yawn_flag(m, f, thr) for e, m, f in drowsy]
    yc = [yawn_flag(m, f, thr) for e, m, f in control]
    od = np.mean([a or b for a, b in zip(pd_, yd)])
    oc = np.mean([a or b for a, b in zip(pc_, yc)])
    res[f"{thr:.2f}"] = dict(yawn_drowsy=float(np.mean(yd)), yawn_control=float(np.mean(yc)),
                             or_drowsy=float(od), or_control=float(oc),
                             or_sep=float(od - oc))
    print(f"{thr:>8.2f}{np.mean(yd)*100:>12.1f}%{np.mean(yc)*100:>10.1f}%"
          f"{od*100:>10.1f}%{oc*100:>8.1f}%{(od-oc)*100:>8.1f}")

json.dump({"perclos_alone": {"drowsy": float(np.mean(pd_)), "control": float(np.mean(pc_))},
           "by_mar_threshold": res},
          open(PROJ / "outputs" / "p9_verdict_impact.json", "w"), indent=1)
print("\nwrote outputs/p9_verdict_impact.json")
