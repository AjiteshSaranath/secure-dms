"""P9 step 2 — event-level view, and the with-hand/without-hand split.

Step 1 showed that annotated yawn INTERVALS contain many frames whose MAR is
indistinguishable from a closed mouth.  Two candidate explanations:

  (a) DMD annotates the whole yawn event, including onset and offset, so most
      frames in the interval are not "mouth wide open";
  (b) "Yawning with hand" occludes the mouth, so the landmarks cannot see the
      opening at all — a physical limit no threshold can fix.

This separates them.  Per annotated event we take the PEAK MAR, which is what a
sustained-opening detector actually keys on, and split by annotation type.
Labels are used to describe, not to fit.
"""
import json, hashlib
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "outputs" / "signal_cache"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes"
           r"\Case Study in Cybersecurity Analytics\DMD Dataset")
EXTRACTOR_VERSION = "p2.3"
WITH_HAND = "yawning/Yawning with hand"
NO_HAND = "yawning/Yawning without hand"


def cache_path(video: Path):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def yawn_events(ann_path: Path):
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    out = []
    for act in ol.get("actions", {}).values():
        if act["type"] in (WITH_HAND, NO_HAND):
            for iv in act.get("frame_intervals", []):
                out.append((iv["frame_start"], iv["frame_end"], act["type"]))
    return sorted(out)


drows = DMD / "drowsiness"
videos = sorted(drows.rglob("*rgb_face.mp4"))
anns = sorted(drows.rglob("*rgb_ann_*.json"))

peaks = {WITH_HAND: [], NO_HAND: []}
rest_max, n_ev = [], {WITH_HAND: 0, NO_HAND: 0}
per_sess = []

for v, a in zip(videos, anns):
    cp = cache_path(v)
    if not cp.exists():
        continue
    d = np.load(cp)
    mar = d["mar"].astype(float)
    fidx = d["frame_idx"].astype(int)
    pos = {int(f): i for i, f in enumerate(fidx)}
    evs = yawn_events(a)
    covered = np.zeros(len(mar), bool)
    sess_peaks = []
    for lo, hi, typ in evs:
        idx = [pos[f] for f in range(lo, hi + 1) if f in pos]
        if not idx:
            continue
        idx = np.array(idx)
        covered[idx] = True
        seg = mar[idx]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            peaks[typ].append(seg.max())
            sess_peaks.append((seg.max(), typ))
        n_ev[typ] += 1
    rest = mar[~covered]
    rest = rest[np.isfinite(rest)]
    rest_max.append(rest.max())
    per_sess.append(("/".join(v.parts[-3:-1]), len(evs), rest.max(),
                     np.percentile(rest, 99.5), sess_peaks))

print(f"annotated yawn events: with-hand {n_ev[WITH_HAND]}, "
      f"without-hand {n_ev[NO_HAND]}\n")

for typ, name in ((NO_HAND, "without hand"), (WITH_HAND, "with hand")):
    p = np.array(peaks[typ])
    if p.size:
        print(f"peak MAR within event, {name} (n={p.size}):")
        print(f"   min {p.min():.3f}  p25 {np.percentile(p,25):.3f}  "
              f"median {np.median(p):.3f}  p75 {np.percentile(p,75):.3f}  max {p.max():.3f}")
        print(f"   fraction of events peaking above 0.60: {(p>0.60).mean():.2f}")
        print(f"   fraction peaking above 0.30:           {(p>0.30).mean():.2f}")
        print(f"   fraction peaking above 0.15:           {(p>0.15).mean():.2f}\n")

rm = np.array(rest_max)
print(f"max MAR OUTSIDE any annotated yawn, per session:")
print(f"   min {rm.min():.3f}  median {np.median(rm):.3f}  max {rm.max():.3f}")
print(f"   sessions whose non-yawn max exceeds 0.60: {(rm>0.60).sum()} of {len(rm)}")
print(f"   sessions whose non-yawn max exceeds 0.30: {(rm>0.30).sum()} of {len(rm)}")

print(f"\n{'session':<12}{'events':>7}{'restMax':>9}{'rest p99.5':>11}"
      f"{'peaks>0.6':>11}{'peaks>0.3':>11}")
print("-" * 62)
for s, n, rmx, r995, sp in per_sess:
    pk = np.array([p for p, _ in sp]) if sp else np.array([0.0])
    print(f"{s:<12}{n:>7}{rmx:>9.3f}{r995:>11.3f}"
          f"{(pk>0.6).sum():>6}/{len(pk):<4}{(pk>0.3).sum():>6}/{len(pk):<4}")
