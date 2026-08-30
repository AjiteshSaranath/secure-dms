"""Tier D — WHY did the PAD module fail on ROSE-Youtu? (ACER 0.592)

Two very different explanations, with different consequences:

  (A) PHYSICS: the cues cannot separate these attacks in principle. Masks with
      eye-holes expose the wearer's REAL eyes (real blinks); video replay shows a
      REAL person blinking and moving their eyes. Both defeat blink/ocular cues
      by construction — the predicted failure mode.

  (B) MEASUREMENT: the cue DETECTORS degrade on ROSE's sensor regime (480p-720p
      mobile, different framing) versus DMD's 1280x720 cabin video, so genuine
      windows lose their blinks and get rejected — a cross-domain transfer
      failure of the measurement, not of the physics.

The tell: genuine blink-fire rate. On DMD it was 89.5 %; on ROSE it is 41.6 %.
If genuine EAR dynamics are visibly weaker/flatter on ROSE, (B) is implicated —
which matters enormously, because (B) is the SAME cross-database fragility this
project cited (R21, R22) as its reason for abandoning texture methods.
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
ROSE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"
CODES = {"G": "genuine", "Ps": "print_still", "Pq": "print_quiver", "Vl": "replay_lenovo",
         "Vm": "replay_mac", "Mc": "mask_cropped", "Mf": "mask_full", "Mu": "mask_upper"}


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


def ear_stats(d):
    """Per-clip EAR dynamics: are blinks physically present in the signal?"""
    ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)
    hy = d["head_yaw"]
    ear = np.where(np.abs(hy) <= 45, ear, np.nan)
    fin = ear[np.isfinite(ear)]
    if len(fin) < 30:
        return None
    fps = float(d["fps"][0])
    base = np.nanpercentile(ear, 85)
    dips = np.nan_to_num(ear, nan=np.inf) <= 0.67 * base
    blinks = [(a, b) for a, b in runs_of_true(dips)
              if 75 <= (b - a + 1) / fps * 1000 <= 400]
    minutes = len(ear) / fps / 60
    return {
        "ear_mean": float(np.mean(fin)), "ear_sd": float(np.std(fin)),
        # dynamic range of the eyelid signal — a flat signal cannot show blinks
        "ear_p5_p95": float(np.percentile(fin, 95) - np.percentile(fin, 5)),
        "ear_min_over_base": float(np.min(fin) / max(base, 1e-6)),
        "blinks_per_min": len(blinks) / max(minutes, 1e-6),
        "fps": fps, "n": len(fin),
    }


rng = random.Random(42)
groups = defaultdict(list)

# ROSE, by species
by_sp = defaultdict(list)
for v in sorted(ROSE.rglob("*.mp4")):
    sp = CODES.get(v.stem.split("_")[0])
    if sp and cache_path(v).exists():
        by_sp[sp].append(v)
for sp, vs in by_sp.items():
    for v in rng.sample(vs, min(60, len(vs))):
        s = ear_stats(np.load(cache_path(v)))
        if s:
            groups[f"ROSE {sp}"].append(s)

# DMD genuine, for the cross-domain comparison
dmd_v = sorted((DMD / "Drowsiness").rglob("*rgb_face.mp4")) + \
        sorted((DMD / "Distraction").rglob("*rgb_face.mp4"))
for v in rng.sample([x for x in dmd_v if cache_path(x).exists()], 40):
    s = ear_stats(np.load(cache_path(v)))
    if s:
        groups["DMD genuine"].append(s)

print(f"{'group':<22}{'clips':>6}{'fps':>6}{'EAR mean':>10}{'EAR sd':>9}"
      f"{'p5-p95':>9}{'min/base':>10}{'blinks/min':>12}")
summary = {}
for g, rows in sorted(groups.items()):
    def m(k):
        return float(np.mean([r[k] for r in rows]))
    summary[g] = {k: m(k) for k in ("ear_mean", "ear_sd", "ear_p5_p95",
                                    "ear_min_over_base", "blinks_per_min", "fps")}
    print(f"{g:<22}{len(rows):>6}{m('fps'):>6.0f}{m('ear_mean'):>10.3f}{m('ear_sd'):>9.3f}"
          f"{m('ear_p5_p95'):>9.3f}{m('ear_min_over_base'):>10.3f}{m('blinks_per_min'):>12.1f}")

print("\ninterpretation:")
rg, dg = summary.get("ROSE genuine"), summary.get("DMD genuine")
if rg and dg:
    print(f"  genuine blink rate  DMD {dg['blinks_per_min']:.1f}/min  vs  "
          f"ROSE {rg['blinks_per_min']:.1f}/min  (literature 15-30, R20)")
    print(f"  genuine EAR range   DMD {dg['ear_p5_p95']:.3f}       vs  "
          f"ROSE {rg['ear_p5_p95']:.3f}")
    if rg["blinks_per_min"] < 12:
        print("  => ROSE genuine blink rate falls BELOW the physiological range:")
        print("     the DETECTOR is missing real blinks on this sensor regime (B),")
        print("     not that the subjects failed to blink.")
    print(f"  mask_cropped blinks/min {summary.get('ROSE mask_cropped',{}).get('blinks_per_min',float('nan')):.1f} "
          f"| replay_mac {summary.get('ROSE replay_mac',{}).get('blinks_per_min',float('nan')):.1f}")
    print("     masks with eye-holes and replays contain REAL blinking eyes (A).")

Path(PROJ / "outputs" / "p4d_failure_diagnostic.json").write_text(
    json.dumps(summary, indent=2), encoding="utf-8")
