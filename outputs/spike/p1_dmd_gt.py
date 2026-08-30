"""P1 (rerun of Q2 only) — DMD drowsiness annotation ground truth.

Confirms per-frame eye-state / blink / yawn ground truth exists and computes
the reference PERCLOS per session (the target our EAR-based PERCLOS must track).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
OUT = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\outputs")


def intervals_of(action):
    return action.get("frame_intervals", [])


def total_frames(intervals):
    return sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in intervals)


rows = []
vocab = defaultdict(int)
for ann_path in sorted((DMD / "Drowsiness").rglob("*rgb_ann_drowsiness.json")):
    with ann_path.open(encoding="utf-8") as fh:
        ol = json.load(fh)["openlabel"]

    fi = ol.get("frame_intervals", [])
    if isinstance(fi, dict):
        fi = [fi]
    n_frames = total_frames(fi)

    per_label = {}
    for act in ol.get("actions", {}).values():
        vocab[act["type"]] += 1
        per_label[act["type"]] = per_label.get(act["type"], 0) + total_frames(intervals_of(act))

    closed = per_label.get("eyes_state/close", 0)
    closing = per_label.get("eyes_state/closing", 0)
    opening = per_label.get("eyes_state/opening", 0)
    yawn = (per_label.get("yawning/Yawning with hand", 0)
            + per_label.get("yawning/Yawning without hand", 0))
    n_blink_intervals = sum(
        len(intervals_of(a)) for a in ol.get("actions", {}).values()
        if a["type"] == "blinks/blinking")

    session = "/".join(ann_path.parts[-4:-1])
    rows.append({
        "session": session,
        "frames": n_frames,
        "eyes_closed": closed,
        "perclos_strict": round(closed / n_frames, 4) if n_frames else None,
        # P80-style: eye substantially closed incl. transitions (R16)
        "perclos_incl_transitions": round((closed + closing + opening) / n_frames, 4) if n_frames else None,
        "yawn_frames": yawn,
        "blink_events": n_blink_intervals,
    })

print(f"annotated drowsiness sessions: {len(rows)}")
print(f"label vocabulary: {dict(vocab)}\n")
print(f"{'session':<22}{'frames':>8}{'closed':>8}{'PERCLOS':>9}{'+trans':>8}{'blinks':>8}{'yawnF':>7}")
for r in rows:
    print(f"{r['session']:<22}{r['frames']:>8}{r['eyes_closed']:>8}"
          f"{r['perclos_strict']:>9.4f}{r['perclos_incl_transitions']:>8.4f}"
          f"{r['blink_events']:>8}{r['yawn_frames']:>7}")

tot = sum(r["frames"] for r in rows)
closed_tot = sum(r["eyes_closed"] for r in rows)
blinks_tot = sum(r["blink_events"] for r in rows)
fps = 29.8
print(f"\ntotal annotated frames: {tot:,} ({tot / fps / 60:.1f} min @ {fps} fps)")
print(f"overall PERCLOS ground truth: {closed_tot / tot:.4f}")
print(f"total annotated blink events: {blinks_tot:,} "
      f"(= {blinks_tot / (tot / fps / 60):.1f} blinks/min — literature range 15-30, ledger R20)")

pc = np.array([r["perclos_strict"] for r in rows])
print(f"per-session PERCLOS: min {pc.min():.4f} max {pc.max():.4f} "
      f"mean {pc.mean():.4f} std {pc.std():.4f}")

with (OUT / "p1_dmd_ground_truth.json").open("w", encoding="utf-8") as fh:
    json.dump({"sessions": rows, "total_frames": tot,
               "overall_perclos": round(closed_tot / tot, 4),
               "total_blink_events": blinks_tot,
               "blinks_per_min": round(blinks_tot / (tot / fps / 60), 1)}, fh, indent=2)
print(f"\nwritten: {OUT / 'p1_dmd_ground_truth.json'}")
