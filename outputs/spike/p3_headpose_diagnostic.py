"""P3 diagnostic — head-pose values beyond +-90 deg on distraction sessions.

The first distraction session tripped the |head_pitch| > 90 guard. Drowsiness
sessions never did. Distraction recordings contain large head turns (reaching for
objects, talking to a passenger), so the question is whether these are:
  (a) genuine extreme poses the cabin camera can still observe,
  (b) Euler-angle wraparound / gimbal artefacts of the 6D->Euler conversion, or
  (c) estimator failure on profile views.

This matters because head pose is the FALLBACK for the off-road decision when
gaze is invalid (§4.2), and distraction is exactly where that fallback is used.
"""
import hashlib
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DIST = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Distraction")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


rows = []
for video in sorted(DMD_DIST.rglob("*rgb_face.mp4")):
    p = cache_path(video)
    if not p.exists():
        continue
    d = np.load(p)
    rows.append(("/".join(video.parts[-4:-1]), d))

print(f"cached distraction sessions: {len(rows)}\n")
if not rows:
    raise SystemExit("nothing cached yet")

for name, d in rows:
    n = len(d["head_pitch"])
    print(f"session {name}  ({n:,} frames)")
    for key in ("head_pitch", "head_yaw", "head_roll"):
        v = d[key]
        over = int(np.sum(np.abs(v) > 90))
        print(f"  {key:<11} range [{v.min():+7.1f}, {v.max():+7.1f}]  "
              f"|.|>90: {over:>5} ({over/n:.3%})  sd {v.std():5.1f}")

    # Are extreme values isolated spikes (artefact) or sustained (real pose)?
    pitch = d["head_pitch"]
    mask = np.abs(pitch) > 90
    if mask.any():
        idx = np.flatnonzero(mask)
        runs, start = [], idx[0]
        for a, b in zip(idx, idx[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        lens = [b - a + 1 for a, b in runs]
        print(f"  out-of-range pitch: {len(runs)} runs, longest {max(lens)} frames, "
              f"median {int(np.median(lens))}")
        # Wraparound signature: values cluster near +-180 rather than just past 90
        vals = np.abs(pitch[mask])
        print(f"  |pitch| on those frames: min {vals.min():.1f} max {vals.max():.1f} "
              f"median {np.median(vals):.1f}")
        near180 = float(np.mean(vals > 150))
        print(f"  fraction beyond 150 deg: {near180:.1%} "
              f"({'wraparound signature' if near180 > 0.5 else 'not wraparound'})")
        yaw_on = np.abs(d["head_yaw"][mask])
        print(f"  mean |head yaw| on those frames: {yaw_on.mean():.1f} deg vs "
              f"{np.abs(d['head_yaw'][~mask]).mean():.1f} elsewhere")
    print()
