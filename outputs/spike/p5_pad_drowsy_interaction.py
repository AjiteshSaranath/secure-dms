"""P5 diagnostic — does the PAD liveness gate reject DROWSY drivers as spoofs?

The end-to-end smoke run rejected 1,502/1,800 frames (83 %) of a genuine
DROWSINESS session at the PAD stage — far above the P4 BPCER of 0.152 measured
over the pooled corpus. Hypothesis: the deployed liveness cues (>=1 blink AND
ocular movement per 10 s window) conflate DROWSINESS with a spoof, because a
drowsy driver blinks less and moves their eyes less. If so this is a
safety-critical architecture interaction, not a tuning issue: PAD sits BEFORE
driver-state in the pipeline (rate->integrity->access->PAD->driver-state), so a
PAD rejection means the drowsiness detector never runs on that driver — the gate
would suppress exactly the alerts the system exists to raise.

Test: PAD window-failure rate on drowsiness sessions vs distraction sessions,
and its correlation with each window's own eye-closure fraction. Both come from
the already-cached signals; no re-extraction.
"""
import hashlib
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"
WINDOW_S = 10.0
# calibrated thresholds from the last §5 eval run (5% target BPCER)
GAZE_THR, DEFORM_THR = 6.896, 0.000520
EAR_YAW_LIMIT = 45.0


def cache_path(v):
    key = f"{v.name}|{v.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def runs_of_true(mask):
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    out, s = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            out.append((s, a)); s = b
    out.append((s, idx[-1]))
    return out


def window_liveness(d, lo, hi):
    ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)[lo:hi]
    hy = d["head_yaw"][lo:hi]
    ear = np.where(np.abs(hy) <= EAR_YAW_LIMIT, ear, np.nan)
    base = np.nanpercentile(ear, 85) if np.isfinite(ear).sum() > 2 else np.nan
    closed = np.nan_to_num(ear, nan=np.inf) <= 0.20 * base if np.isfinite(base) else np.zeros_like(ear, bool)
    closed_frac = float(np.mean(closed))
    # blink: EAR<=0.67*baseline runs of 75-400 ms
    fps = float(d["fps"][0])
    blink_mask = np.nan_to_num(ear, nan=np.inf) <= 0.67 * base if np.isfinite(base) else np.zeros_like(ear, bool)
    n_blinks = sum(1 for a, b in runs_of_true(blink_mask)
                   if 75 <= (b - a + 1) / fps * 1000 <= 400)
    gy = (d["gaze_yaw"] - d["head_yaw"])[lo:hi]
    gp = (d["gaze_pitch"] - d["head_pitch"])[lo:hi]
    ok = (np.abs(d["gaze_yaw"][lo:hi]) <= 90) & (np.abs(d["head_yaw"][lo:hi]) <= 90)
    ocular = float(np.hypot(np.std(gy[ok]), np.std(gp[ok]))) if ok.sum() > 2 else 0.0
    is_live = (n_blinks >= 1) and (ocular >= GAZE_THR)
    return is_live, closed_frac, n_blinks, ocular


def analyse(split):
    rej_rates, all_closed, all_live = [], [], []
    for v in sorted((DMD / split).rglob("*rgb_face.mp4")):
        p = cache_path(v)
        if not p.exists():
            continue
        d = np.load(p)
        fps = float(d["fps"][0]); w = max(int(round(WINDOW_S * fps)), 2)
        n = len(d["ear_left"])
        wins = [window_liveness(d, lo, lo + w) for lo in range(0, n - w + 1, w)]
        if not wins:
            continue
        rej = np.mean([not x[0] for x in wins])
        rej_rates.append(rej)
        all_closed.extend(x[1] for x in wins)
        all_live.extend(x[0] for x in wins)
    return np.array(rej_rates), np.array(all_closed), np.array(all_live, dtype=bool)


for split in ("Drowsiness", "Distraction"):
    rej, closed, live = analyse(split)
    print(f"{split:<12}: {len(rej)} sessions | PAD window-rejection rate per session "
          f"mean {rej.mean():.1%} median {np.median(rej):.1%} max {rej.max():.1%}")
    # correlation of rejection with eye-closure at window level
    rejected = ~live
    print(f"{'':12}  window eye-closure: rejected windows {closed[rejected].mean():.3f} "
          f"vs admitted {closed[live].mean():.3f}")

print("\n=> if drowsiness rejection >> distraction rejection, the liveness gate")
print("   conflates drowsiness with spoofing and would gate out drowsy drivers.")
