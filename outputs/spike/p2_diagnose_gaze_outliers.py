"""P2 diagnostic — gaze estimates beyond +-90 deg.

The plausibility guard failed on two sessions. L2CS-Net was trained on Gaze360,
whose label space spans the full 360 deg (including gaze behind the head), so
|pitch| > 90 deg is a value the model CAN emit — but it is not physically
meaningful for a driver facing a cabin camera. Question: is this a small set of
degenerate frames (mask them) or a systematic failure (fix the pipeline)?
"""
import hashlib
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.2"


def cache_path(video: Path) -> Path:
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


videos = sorted(DMD_DROW.rglob("*rgb_face.mp4"))
print(f"{'session':<34}{'frames':>7}{'|pitch|>90':>11}{'rate':>8}{'|yaw|>90':>10}{'rate':>8}")
print("-" * 78)

totals = {"frames": 0, "bad_pitch": 0, "bad_yaw": 0, "bad_any": 0}
per_session = []
for v in videos:
    p = cache_path(v)
    if not p.exists():
        continue
    d = np.load(p)
    gp, gy = d["gaze_pitch"], d["gaze_yaw"]
    bad_p = int(np.sum(np.abs(gp) > 90))
    bad_y = int(np.sum(np.abs(gy) > 90))
    bad_any = int(np.sum((np.abs(gp) > 90) | (np.abs(gy) > 90)))
    n = len(gp)
    totals["frames"] += n
    totals["bad_pitch"] += bad_p
    totals["bad_yaw"] += bad_y
    totals["bad_any"] += bad_any
    name = f"{v.parent.parent.parent.name}/{v.parent.parent.name}/{v.parent.name}"
    print(f"{name:<34}{n:>7}{bad_p:>11}{bad_p/n:>8.2%}{bad_y:>10}{bad_y/n:>8.2%}")
    per_session.append((name, n, bad_p, bad_y, gp, gy, d))

if totals["frames"]:
    print("-" * 78)
    print(f"{'TOTAL':<34}{totals['frames']:>7}{totals['bad_pitch']:>11}"
          f"{totals['bad_pitch']/totals['frames']:>8.2%}{totals['bad_yaw']:>10}"
          f"{totals['bad_yaw']/totals['frames']:>8.2%}")
    print(f"\nframes with any out-of-range gaze angle: {totals['bad_any']:,} "
          f"({totals['bad_any']/totals['frames']:.3%})")

# Characterise the offending frames: are they isolated spikes or sustained runs?
worst = max(per_session, key=lambda r: r[2]) if per_session else None
if worst and worst[2]:
    name, n, bad_p, bad_y, gp, gy, d = worst
    mask = np.abs(gp) > 90
    idx = np.flatnonzero(mask)
    runs, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    runs.append((start, idx[-1]))
    print(f"\nworst session: {name}")
    print(f"  out-of-range values: min {gp[mask].min():.1f} max {gp[mask].max():.1f} deg")
    print(f"  in-range values:     min {gp[~mask].min():.1f} max {gp[~mask].max():.1f} deg")
    print(f"  contiguous runs: {len(runs)} | longest {max(b-a+1 for a, b in runs)} frames "
          f"| median {int(np.median([b-a+1 for a, b in runs]))} frames")
    ear = (d["ear_left"] + d["ear_right"]) / 2
    print(f"  mean EAR on flagged frames {np.nanmean(ear[mask]):.3f} vs "
          f"{np.nanmean(ear[~mask]):.3f} elsewhere")
    hy = d["head_yaw"]
    print(f"  mean |head yaw| on flagged frames {np.nanmean(np.abs(hy[mask])):.1f} vs "
          f"{np.nanmean(np.abs(hy[~mask])):.1f} deg elsewhere")
