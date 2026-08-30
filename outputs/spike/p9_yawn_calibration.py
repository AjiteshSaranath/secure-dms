"""P9 step 3 — a label-free yawn criterion, and its validation.

DESIGN (Option B).  The eye measure sets its reference from the driver's own
signal: eyes are open most of the time, so a high percentile of EAR is "open",
and closure is a fraction of that.  The mouth is the mirror image — closed most
of the time — so the reference cannot be a low percentile: dividing by a
near-zero closed value is numerically unstable (step 1 measured ratios from 4x
to 56x across drivers).

Instead the reference is the driver's own WIDEST opening, estimated as a high
percentile of their MAR, and a yawn is an opening that reaches a fraction of it
and is sustained.  The fraction is the natural midpoint, 0.5: half of the
driver's own maximum opening.  No annotation is used to place it.

Annotations are used ONLY to validate afterwards, never to choose P or f.
The sweep below is reported for sensitivity, exactly as the closure sweep is.
"""
import json, hashlib, sys
from pathlib import Path
import numpy as np

PROJ = Path(__file__).resolve().parents[2]
CACHE = PROJ / "outputs" / "signal_cache"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes"
           r"\Case Study in Cybersecurity Analytics\DMD Dataset")
EXTRACTOR_VERSION = "p2.3"
WITH_HAND, NO_HAND = "yawning/Yawning with hand", "yawning/Yawning without hand"
MIN_MS = 1000.0            # sustained >= 1 s, unchanged
FIXED_OLD = 0.60           # the value being replaced


def cache_path(video: Path):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def runs_of_true(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def yawn_events(ann_path: Path):
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    return sorted((iv["frame_start"], iv["frame_end"], act["type"])
                  for act in ol.get("actions", {}).values()
                  if act["type"] in (WITH_HAND, NO_HAND)
                  for iv in act.get("frame_intervals", []))


def mouth_reference(mar, pct):
    """The driver's own widest opening — label-free."""
    m = mar[np.isfinite(mar)]
    return float(np.percentile(m, pct)) if m.size else np.nan


def detect(mar, fps, thr):
    ok = np.isfinite(mar)
    m = np.zeros(len(mar), bool)
    m[ok] = mar[ok] >= thr
    return [(a, b) for a, b in runs_of_true(m)
            if (b - a + 1) / fps * 1000.0 >= MIN_MS]


def load(split):
    root = DMD / split
    out = []
    for v in sorted(root.rglob("*rgb_face.mp4")):
        cp = cache_path(v)
        if not cp.exists():
            continue
        d = np.load(cp)
        out.append(("/".join(v.parts[-3:-1]), d["mar"].astype(float),
                    d["frame_idx"].astype(int), float(d["fps"][0]), v))
    return out


drowsy = load("drowsiness")
anns = sorted((DMD / "drowsiness").rglob("*rgb_ann_*.json"))
control = load("distraction")
print(f"loaded {len(drowsy)} drowsiness + {len(control)} distraction sessions\n")


def evaluate(pct, frac, fixed=None):
    """Event recall vs annotation, false events, and session flag rates."""
    hit = {WITH_HAND: 0, NO_HAND: 0}
    tot = {WITH_HAND: 0, NO_HAND: 0}
    false_ev, flagged = 0, 0
    for (sess, mar, fidx, fps, v), a in zip(drowsy, anns):
        thr = fixed if fixed is not None else frac * mouth_reference(mar, pct)
        det = detect(mar, fps, thr)
        flagged += bool(det)
        pos = {int(f): i for i, f in enumerate(fidx)}
        det_idx = set()
        for lo, hi in det:
            det_idx.update(range(lo, hi + 1))
        matched = set()
        for lo, hi, typ in yawn_events(a):
            tot[typ] += 1
            idx = {pos[f] for f in range(lo, hi + 1) if f in pos}
            ov = [k for k, (dl, dh) in enumerate(det)
                  if idx & set(range(dl, dh + 1))]
            if ov:
                hit[typ] += 1
                matched.update(ov)
        false_ev += len(det) - len(matched)
    ctrl = sum(bool(detect(mar, fps,
                           fixed if fixed is not None else frac * mouth_reference(mar, pct)))
               for _, mar, _, fps, _ in control)
    r_nh = hit[NO_HAND] / max(tot[NO_HAND], 1)
    r_wh = hit[WITH_HAND] / max(tot[WITH_HAND], 1)
    r_all = (hit[NO_HAND] + hit[WITH_HAND]) / max(tot[NO_HAND] + tot[WITH_HAND], 1)
    return dict(recall_nohand=r_nh, recall_hand=r_wh, recall=r_all,
                false_events=false_ev, drowsy_flag=flagged / len(drowsy),
                control_flag=ctrl / len(control))


print("BASELINE — the fixed threshold being replaced")
b = evaluate(None, None, fixed=FIXED_OLD)
print(f"  MAR >= {FIXED_OLD}:  event recall {b['recall']:.3f} "
      f"(no-hand {b['recall_nohand']:.3f} / with-hand {b['recall_hand']:.3f})  "
      f"false events {b['false_events']}")
print(f"     session flag rate: drowsy {b['drowsy_flag']*100:.1f}%  "
      f"control {b['control_flag']*100:.1f}%\n")

print("SWEEP — threshold = frac x percentile(MAR) of the driver's own signal")
print(f"{'pct':>6}{'frac':>7}{'recall':>9}{'noHand':>9}{'withHand':>10}"
      f"{'falseEv':>9}{'drowsy%':>9}{'ctrl%':>8}")
print("-" * 68)
best = None
for pct in (98.0, 99.0, 99.5):
    for frac in (0.3, 0.4, 0.5, 0.6, 0.7):
        r = evaluate(pct, frac)
        print(f"{pct:>6.1f}{frac:>7.2f}{r['recall']:>9.3f}{r['recall_nohand']:>9.3f}"
              f"{r['recall_hand']:>10.3f}{r['false_events']:>9}"
              f"{r['drowsy_flag']*100:>9.1f}{r['control_flag']*100:>8.1f}")
    print()

print("DEPLOYED CHOICE — percentile 99.5, fraction 0.50 (natural midpoint)")
d = evaluate(99.5, 0.5)
for k, v in d.items():
    print(f"   {k:<16} {v}")

json.dump({"fixed_old": b,
           "sweep": {f"p{p}_f{f}": evaluate(p, f)
                     for p in (98.0, 99.0, 99.5) for f in (0.3, 0.4, 0.5, 0.6, 0.7)},
           "deployed": {"percentile": 99.5, "fraction": 0.5, **d}},
          open(PROJ / "outputs" / "p9_yawn_calibration.json", "w"), indent=1)
print("\nwrote outputs/p9_yawn_calibration.json")
