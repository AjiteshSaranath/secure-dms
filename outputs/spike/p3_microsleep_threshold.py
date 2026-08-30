"""P3 diagnostic — the microsleep cue fires on 54.9 % of the non-drowsy control.

The ablation shows the deployed fused rule (PERCLOS OR microsleep) is WORSE than
PERCLOS alone, because the microsleep term has poor specificity. Two candidate
causes, and they call for different responses:

  (a) Wrong threshold. This implementation used >400 ms ("beyond a normal
      blink"), but the drowsiness literature defines a BEHAVIOURAL MICROSLEEP as
      an eye closure exceeding ~500 ms (R17). If 500 ms fixes it, the fault was a
      mis-set published value — correct it.
  (b) The cue is genuinely weak at session level. Then it should be dropped from
      the rule on evidence, not re-tuned until it looks good.

Sweeping duration here is diagnostic. The deployed value must remain a published
one; the sweep exists to show where the published value sits, not to pick a value.
"""
import hashlib
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"
P80, OPEN_PCT, EAR_YAW_LIMIT = 0.20, 85.0, 45.0


def cache_path(v):
    key = f"{v.name}|{v.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


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


def load(split):
    rows = []
    for v in sorted((DMD / split).rglob("*rgb_face.mp4")):
        p = cache_path(v)
        if not p.exists():
            continue
        d = np.load(p)
        ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)
        valid = (np.abs(d["head_yaw"]) <= EAR_YAW_LIMIT) & (np.abs(d["head_pitch"]) <= 90)
        ear = np.where(valid, ear, np.nan)
        rows.append((ear, float(d["fps"][0])))
    return rows


drowsy, control = load("Drowsiness"), load("Distraction")
print(f"drowsiness {len(drowsy)} | distraction (control) {len(control)}\n")


def fires(rows, min_ms):
    hits = []
    for ear, fps in rows:
        base = np.nanpercentile(ear, OPEN_PCT)
        closed = np.nan_to_num(ear, nan=np.inf) <= P80 * base
        longest = max(((b - a + 1) / fps * 1000 for a, b in runs_of_true(closed)), default=0.0)
        hits.append(longest > min_ms)
    return float(np.mean(hits))


print(f"{'min closure (ms)':>18}{'drowsy fires':>14}{'control fires':>15}{'separation':>12}")
for ms in (400, 500, 600, 800, 1000, 1500, 2000):
    d, c = fires(drowsy, ms), fires(control, ms)
    tag = ("  <- used (beyond-blink)" if ms == 400 else
           "  <- published microsleep (R17)" if ms == 500 else "")
    print(f"{ms:>18}{d:>13.1%}{c:>15.1%}{d - c:>+12.1%}{tag}")

# How long ARE the longest closures in each group?
for name, rows in (("drowsiness", drowsy), ("control", control)):
    longest = []
    for ear, fps in rows:
        base = np.nanpercentile(ear, OPEN_PCT)
        closed = np.nan_to_num(ear, nan=np.inf) <= P80 * base
        longest.append(max(((b - a + 1) / fps * 1000 for a, b in runs_of_true(closed)), default=0.0))
    longest = np.array(longest)
    print(f"\n{name}: longest closure per session — median {np.median(longest):.0f} ms, "
          f"p25 {np.percentile(longest,25):.0f}, p75 {np.percentile(longest,75):.0f}, "
          f"max {longest.max():.0f} ms")
