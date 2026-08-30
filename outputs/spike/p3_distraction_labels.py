"""P3 (distraction branch) — survey the DMD distraction annotation vocabulary.

Runs on the JSON only, so it can be done while video extraction is in flight.

The key question for DP-2a: DMD annotates `gaze_on_road/{looking_road,
not_looking_road}` per frame. If that is well populated it is DIRECT ground truth
for visual distraction — a far stronger validation target than inferring
attention from manual-action classes, and it validates the DP-2a decision to
scope the claim to visual distraction rather than all distraction.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DMD_DIST = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Distraction")
OUT = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\outputs")


def frame_intervals(node):
    fi = node.get("frame_intervals", [])
    return [fi] if isinstance(fi, dict) else fi


def total(intervals):
    return sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in intervals)


label_frames = defaultdict(int)
label_sessions = defaultdict(int)
sessions = []

for path in sorted(DMD_DIST.rglob("*rgb_ann_distraction.json")):
    ol = json.loads(path.read_text(encoding="utf-8"))["openlabel"]
    n_frames = total(frame_intervals(ol))
    seen, counts = set(), defaultdict(int)
    for act in ol.get("actions", {}).values():
        t = act["type"]
        n = total(frame_intervals(act))
        label_frames[t] += n
        counts[t] += n
        seen.add(t)
    for t in seen:
        label_sessions[t] += 1
    sessions.append({"session": "/".join(path.parts[-4:-1]), "frames": n_frames,
                     "counts": dict(counts)})

total_frames = sum(s["frames"] for s in sessions)
print(f"distraction sessions: {len(sessions)} | total annotated frames: {total_frames:,}")
print(f"({total_frames / 29.76 / 60:.0f} min of video)\n")

print(f"{'label':<38}{'sessions':>9}{'frames':>12}{'% of all':>10}")
print("-" * 69)
for t, n in sorted(label_frames.items(), key=lambda kv: -kv[1]):
    print(f"{t:<38}{label_sessions[t]:>9}{n:>12,}{n/total_frames:>9.1%}")

# --- the DP-2a target -------------------------------------------------------
on = label_frames.get("gaze_on_road/looking_road", 0)
off = label_frames.get("gaze_on_road/not_looking_road", 0)
print("\n" + "=" * 69)
print("DP-2a VISUAL-DISTRACTION GROUND TRUTH")
print("=" * 69)
if on + off:
    print(f"looking_road      {on:>10,} frames  ({on/(on+off):.1%} of gaze-labelled)")
    print(f"not_looking_road  {off:>10,} frames  ({off/(on+off):.1%} of gaze-labelled)")
    print(f"gaze labelling covers {(on+off)/total_frames:.1%} of all distraction frames")
    print(f"sessions carrying gaze labels: {label_sessions.get('gaze_on_road/not_looking_road', 0)}"
          f"/{len(sessions)}")
    print("\n=> direct per-frame ground truth for the visual-distraction branch;")
    print("   off_road_mask() can be scored against it with F1/precision/recall,")
    print("   and the class balance is far healthier than drowsiness (~10% positives).")
else:
    print("no gaze_on_road labels found — visual distraction would need proxy labels")

# --- classes our gaze/head stack CANNOT see (report separately, DP-2a) -----
manual = {t: n for t, n in label_frames.items()
          if t.startswith("driver_actions/") and "safe_drive" not in t
          and "unclassified" not in t}
print("\nmanual/cognitive action classes (no gaze evidence — reported separately per DP-2a):")
for t, n in sorted(manual.items(), key=lambda kv: -kv[1]):
    print(f"  {t:<40}{n:>10,} frames")

with (OUT / "p3_distraction_labels.json").open("w", encoding="utf-8") as fh:
    json.dump({"n_sessions": len(sessions), "total_frames": total_frames,
               "label_frames": dict(label_frames),
               "label_sessions": dict(label_sessions)}, fh, indent=2)
print(f"\nwritten: {OUT / 'p3_distraction_labels.json'}")
