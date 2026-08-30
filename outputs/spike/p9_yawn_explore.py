"""P9 step 1 — what does MAR actually look like, per driver?

Before designing a label-free yawn criterion we need to know whether the
driver's own MAR distribution separates yawns from ordinary mouth movement.
This script ONLY LOOKS.  It fits nothing and decides nothing.

Yawn annotations are used here strictly to *describe* the distribution, so we
can see whether a label-free rule is viable.  The rule itself must not use them.
"""
import json, hashlib, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "outputs" / "signal_cache"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes"
           r"\Case Study in Cybersecurity Analytics\DMD Dataset")
EXTRACTOR_VERSION = "p2.3"
YAWN_TYPES = ("yawning/Yawning with hand", "yawning/Yawning without hand")


def frame_intervals(obj):
    return obj.get("frame_intervals", [])


def cache_path(video: Path):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def yawn_mask(ann_path: Path, n_frames: int):
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    m = np.zeros(n_frames, bool)
    for act in ol.get("actions", {}).values():
        if act["type"] in YAWN_TYPES:
            for iv in frame_intervals(act):
                lo, hi = max(iv["frame_start"], 0), min(iv["frame_end"] + 1, n_frames)
                m[lo:hi] = True
    return m


drows = DMD / "drowsiness"
videos = sorted(drows.rglob("*rgb_face.mp4"))
anns = sorted(drows.rglob("*rgb_ann_*.json"))
print(f"{len(videos)} videos, {len(anns)} annotations\n")

rows = []
for v, a in zip(videos, anns):
    cp = cache_path(v)
    if not cp.exists():
        print(f"  MISS cache for {v.name}")
        continue
    d = np.load(cp)
    mar = d["mar"].astype(float)
    fidx = d["frame_idx"].astype(int)
    n_read = int(d["total_frames_read"][0])
    ym = yawn_mask(a, max(n_read, fidx.max() + 1))
    y = ym[fidx]                                  # align labels to detected frames
    ok = np.isfinite(mar)
    mar, y = mar[ok], y[ok]
    if y.sum() == 0:
        continue
    sess = "/".join(v.parts[-3:-1])
    rows.append(dict(
        sess=sess, n=len(mar), yawn_frac=y.mean(),
        p15=np.percentile(mar, 15), p50=np.percentile(mar, 50),
        p85=np.percentile(mar, 85), p95=np.percentile(mar, 95),
        p99=np.percentile(mar, 99),
        med_yawn=np.median(mar[y]), med_rest=np.median(mar[~y]),
        p10_yawn=np.percentile(mar[y], 10), p90_rest=np.percentile(mar[~y], 90),
        max_rest=mar[~y].max(),
    ))

print(f"{'session':<12}{'yawn%':>7}{'p50':>7}{'p85':>7}{'p95':>7}"
      f"{'medRest':>9}{'medYawn':>9}{'p90rest':>9}{'p10yawn':>9}{'ratio':>7}")
print("-" * 84)
for r in rows:
    ratio = r["med_yawn"] / r["p50"] if r["p50"] > 0 else float("nan")
    print(f"{r['sess']:<12}{r['yawn_frac']*100:>6.1f}%{r['p50']:>7.3f}{r['p85']:>7.3f}"
          f"{r['p95']:>7.3f}{r['med_rest']:>9.3f}{r['med_yawn']:>9.3f}"
          f"{r['p90_rest']:>9.3f}{r['p10_yawn']:>9.3f}{ratio:>7.2f}")

if rows:
    print("\nsummary across sessions")
    for k in ("p50", "p85", "p95", "med_rest", "med_yawn", "p90_rest", "p10_yawn"):
        v = np.array([r[k] for r in rows])
        print(f"  {k:<9} min {v.min():.3f}  median {np.median(v):.3f}  max {v.max():.3f}")
    ratio = np.array([r["med_yawn"] / r["p50"] for r in rows])
    print(f"  med_yawn / p50 :  min {ratio.min():.2f}  median {np.median(ratio):.2f}"
          f"  max {ratio.max():.2f}")
    # How well would the CURRENT fixed 0.60 do, per session?
    print("\n  overlap check: fraction of REST frames above a fixed 0.60 —",
          f"{np.mean([ (r['p90_rest'] > 0.60) for r in rows]):.2f} of sessions have p90(rest) > 0.60")
