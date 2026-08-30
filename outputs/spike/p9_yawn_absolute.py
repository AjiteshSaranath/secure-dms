"""P9 step 4 — absolute MAR sweep, with a subject-disjoint check.

Step 3 tested the per-driver relative criterion (Option B) and it FAILED: it
flags 96-100 % of the non-drowsy control group.  The reason is circular.  The
reference is a high percentile of the driver's own MAR, but in a session with no
yawn that percentile is ordinary talking, so half of it is crossed constantly.
The eye measure escapes this only because eyes really are open most of the time.

So the threshold stays absolute.  MAR is already normalised by face width, and a
yawn opens the jaw close to its anatomical limit, so an absolute value transfers
better here than it does for EAR.  The value is swept against DMD's yawn
annotation and reported, which makes it CALIBRATED AGAINST ANNOTATION — a weaker
claim than the eye baseline's label-free calibration, and it must be labelled as
such.

To show the sweep is not merely fitted, it is repeated on a subject-disjoint
split: choose on half the sessions, report on the other half.
"""
import json, hashlib
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "outputs" / "signal_cache"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes"
           r"\Case Study in Cybersecurity Analytics\DMD Dataset")
EXTRACTOR_VERSION = "p2.3"
WITH_HAND, NO_HAND = "yawning/Yawning with hand", "yawning/Yawning without hand"
MIN_MS = 1000.0
GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]


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


def yawn_events(p):
    ol = json.loads(p.read_text(encoding="utf-8"))["openlabel"]
    return sorted((iv["frame_start"], iv["frame_end"], a["type"])
                  for a in ol.get("actions", {}).values()
                  if a["type"] in (WITH_HAND, NO_HAND)
                  for iv in a.get("frame_intervals", []))


def detect(mar, fps, thr):
    ok = np.isfinite(mar); m = np.zeros(len(mar), bool)
    m[ok] = mar[ok] >= thr
    return [(a, b) for a, b in runs_of_true(m)
            if (b - a + 1) / fps * 1000.0 >= MIN_MS]


def load(split):
    out = []
    for v in sorted((DMD / split).rglob("*rgb_face.mp4")):
        cp = cache_path(v)
        if cp.exists():
            d = np.load(cp)
            out.append(("/".join(v.parts[-3:-1]), d["mar"].astype(float),
                        d["frame_idx"].astype(int), float(d["fps"][0])))
    return out


drowsy = load("drowsiness")
anns = sorted((DMD / "drowsiness").rglob("*rgb_ann_*.json"))
control = load("distraction")


def score(thr, idxs=None):
    hit = {WITH_HAND: 0, NO_HAND: 0}; tot = {WITH_HAND: 0, NO_HAND: 0}
    false_ev, flagged, n = 0, 0, 0
    for k, ((s, mar, fidx, fps), a) in enumerate(zip(drowsy, anns)):
        if idxs is not None and k not in idxs:
            continue
        n += 1
        det = detect(mar, fps, thr)
        flagged += bool(det)
        pos = {int(f): i for i, f in enumerate(fidx)}
        matched = set()
        for lo, hi, typ in yawn_events(a):
            tot[typ] += 1
            idx = {pos[f] for f in range(lo, hi + 1) if f in pos}
            ov = [j for j, (dl, dh) in enumerate(det) if idx & set(range(dl, dh + 1))]
            if ov:
                hit[typ] += 1; matched.update(ov)
        false_ev += len(det) - len(matched)
    ctrl = sum(bool(detect(m, f, thr)) for _, m, _, f in control)
    tt = tot[NO_HAND] + tot[WITH_HAND]
    return dict(thr=thr, n_sess=n,
                recall=(hit[NO_HAND] + hit[WITH_HAND]) / max(tt, 1),
                recall_nohand=hit[NO_HAND] / max(tot[NO_HAND], 1),
                recall_hand=hit[WITH_HAND] / max(tot[WITH_HAND], 1),
                false_events=false_ev,
                drowsy_flag=flagged / max(n, 1),
                control_flag=ctrl / len(control))


print("ABSOLUTE MAR SWEEP — all 16 drowsiness sessions, 82 controls\n")
print(f"{'MAR':>6}{'recall':>9}{'noHand':>9}{'withHand':>10}{'falseEv':>9}"
      f"{'drowsy%':>9}{'ctrl%':>8}{'sep':>8}")
print("-" * 68)
rows = []
for t in GRID:
    r = score(t); rows.append(r)
    sep = (r["drowsy_flag"] - r["control_flag"]) * 100
    print(f"{t:>6.2f}{r['recall']:>9.3f}{r['recall_nohand']:>9.3f}"
          f"{r['recall_hand']:>10.3f}{r['false_events']:>9}"
          f"{r['drowsy_flag']*100:>9.1f}{r['control_flag']*100:>8.1f}{sep:>8.1f}")

# subject-disjoint check: choose on even-indexed sessions, report on odd
even = set(range(0, len(drowsy), 2))
odd = set(range(1, len(drowsy), 2))
pick = max(GRID, key=lambda t: (score(t, even)["drowsy_flag"]
                                - score(t, even)["control_flag"]))
held = score(pick, odd)
print(f"\nsubject-disjoint check: chosen on 8 sessions -> MAR >= {pick:.2f}")
print(f"   held-out 8 sessions: recall {held['recall']:.3f} "
      f"(no-hand {held['recall_nohand']:.3f}), drowsy flag "
      f"{held['drowsy_flag']*100:.1f}%, control {held['control_flag']*100:.1f}%")

json.dump({"grid": rows, "held_out": {"chosen": pick, **held},
           "note": "relative per-driver criterion rejected: see p9_yawn_calibration.json"},
          open(PROJ / "outputs" / "p9_yawn_absolute.json", "w"), indent=1)
print("\nwrote outputs/p9_yawn_absolute.json")
