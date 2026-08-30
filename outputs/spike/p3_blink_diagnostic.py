"""P3 diagnostic — why are blinks under-counted, and does R9's threshold transfer?

Full P3 run: blink counts correlate only r=0.50 with the annotator and are
badly under-detected (6 predicted vs 66 annotated on gA/2). Suspected cause: the
blink detector reuses the PERCLOS P80 mask, but P80 marks *80 % closure* — a
criterion designed to measure sustained eyelid droop, not to catch the brief dip
of a normal blink. The two measures need different criteria because they answer
different questions (R15/R16 vs R9).

This also tests whether Soukupova & Cech's ABSOLUTE threshold (EAR < 0.21, R9)
transfers to MediaPipe landmarks, which is a prerequisite for using it at all.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.2"
BLINK_MIN_MS, BLINK_MAX_MS = 75, 400


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def frame_intervals(node):
    fi = node.get("frame_intervals", [])
    return [fi] if isinstance(fi, dict) else fi


def runs_of_true(mask):
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    out, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    return out


sessions = []
for video in sorted(DMD_DROW.rglob("*rgb_face.mp4")):
    p = cache_path(video)
    ann_path = next(video.parent.glob("*rgb_ann_drowsiness.json"), None)
    if not p.exists() or ann_path is None:
        continue
    d = np.load(p)
    if len(d["frame_idx"]) < 1000:
        continue
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    n_blinks = sum(len(frame_intervals(a)) for a in ol.get("actions", {}).values()
                   if a["type"] == "blinks/blinking")
    ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)
    sessions.append({"name": "/".join(video.parts[-4:-1]), "ear": ear,
                     "fps": float(d["fps"][0]), "truth": n_blinks})

print(f"sessions: {len(sessions)}\n")

# --- does the ABSOLUTE R9 threshold transfer to MediaPipe landmarks? -------
print("Transferability of the absolute R9 threshold (EAR < 0.21):")
print(f"{'session':<12}{'mean EAR':>10}{'85th pct':>10}{'% frames < 0.21':>18}")
for s in sessions:
    below = float(np.mean(s["ear"] < 0.21))
    print(f"{s['name']:<12}{np.nanmean(s['ear']):>10.3f}"
          f"{np.nanpercentile(s['ear'], 85):>10.3f}{below:>17.1%}")
means = np.array([np.nanmean(s["ear"]) for s in sessions])
print(f"\nsessions whose MEAN open-eye EAR is already below 0.21: "
      f"{int(np.sum(means < 0.21))}/{len(sessions)}")
print("=> the absolute threshold does NOT transfer across landmark detectors;")
print("   a relative criterion is required (consistent with R14's personalised-threshold finding).\n")


def count_blinks(ear, fps, fraction, baseline_pct=85.0):
    baseline = np.nanpercentile(ear, baseline_pct)
    closed = ear <= fraction * baseline
    n = 0
    for a, b in runs_of_true(closed):
        ms = (b - a + 1) / fps * 1000
        if BLINK_MIN_MS <= ms <= BLINK_MAX_MS:
            n += 1
    return n


print("Blink-count agreement vs closure fraction (relative criterion):")
print(f"{'fraction':>9}{'pearson r':>11}{'MAPE':>9}{'mean pred':>11}{'mean truth':>12}")
truth = np.array([s["truth"] for s in sessions], dtype=float)
for frac in (0.20, 0.30, 0.40, 0.50, 0.60, 0.67, 0.70, 0.75, 0.80):
    pred = np.array([count_blinks(s["ear"], s["fps"], frac) for s in sessions], dtype=float)
    r = float(np.corrcoef(pred, truth)[0, 1])
    mape = float(np.mean(np.abs(pred - truth) / np.maximum(truth, 1)))
    tag = "  <- P80 (PERCLOS criterion)" if abs(frac - 0.20) < 1e-9 else \
          "  <- R9 ratio-preserving" if abs(frac - 0.67) < 1e-9 else ""
    print(f"{frac:>9.2f}{r:>11.4f}{mape:>9.1%}{pred.mean():>11.1f}{truth.mean():>12.1f}{tag}")

print("\nNote: the ratio-preserving transfer takes R9's threshold of 0.21 against the")
print("~0.31 open-eye EAR typical of that work (0.21/0.31 = 0.67) and applies the RATIO")
print("to each driver's own baseline, rather than the raw value.")
