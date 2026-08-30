"""P3 diagnostic — is the off-road cone anchored to the wrong origin?

Visual-distraction F1 is only 0.445, with phone calls flagged off-road 19.4 % of
the time against a ground truth of 1.7 %. Suspected cause is a coordinate-frame
error rather than a model failure:

L2CS-Net reports gaze in CAMERA coordinates. The published +-30 deg forward cone
(R19, R44) is defined relative to the ROAD-AHEAD direction. The DMD face camera is
dash-mounted and off-axis, so gaze_yaw = 0 does NOT mean "looking at the road" —
there is an unknown per-installation offset. Applying a road-referenced threshold
to camera-referenced angles mis-centres the cone for every session.

A principled fix without touching labels: drivers look at the road for the large
majority of driving time, so the MODE of each session's own gaze distribution
estimates the road-ahead direction. This is the same class of self-normalisation
already used for the PERCLOS open-eye baseline (R15, R16) and is standard in the
gaze-zone literature — it uses the subject's own signal, never the annotation.

This script quantifies the offset and tests whether re-centring explains the gap.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DIST = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Distraction")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"
YAW_CONE, PITCH_CONE = 30.0, 20.0


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def frame_intervals(node):
    fi = node.get("frame_intervals", [])
    return [fi] if isinstance(fi, dict) else fi


def load_truth(ann_path, frame_idx):
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    n = sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in frame_intervals(ol))
    lab = np.full(n, "", dtype=object)
    for act in ol.get("actions", {}).values():
        t = act["type"]
        if t in ("gaze_on_road/looking_road", "gaze_on_road/not_looking_road"):
            v = "off" if t.endswith("not_looking_road") else "on"
            for iv in frame_intervals(act):
                lab[max(iv["frame_start"], 0):min(iv["frame_end"] + 1, n)] = v
    idx = frame_idx[frame_idx < n]
    return lab[idx], idx


def mode_angle(x, lo=-60, hi=60, bins=121):
    """Modal direction via histogram peak — robust to the off-road tail."""
    h, edges = np.histogram(x[(x > lo) & (x < hi)], bins=bins, range=(lo, hi))
    if h.sum() == 0:
        return 0.0
    k = int(np.argmax(h))
    return float((edges[k] + edges[k + 1]) / 2)


def scores(pred, truth):
    tp = np.sum(pred & truth); fp = np.sum(pred & ~truth)
    fn = np.sum(~pred & truth); tn = np.sum(~pred & ~truth)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return ((tp + tn) / len(truth), prec, rec,
            2 * prec * rec / (prec + rec) if prec + rec else 0.0)


rows = []
for video in sorted(DMD_DIST.rglob("*rgb_face.mp4")):
    p = cache_path(video)
    ann = next(video.parent.glob("*rgb_ann_distraction.json"), None)
    if not p.exists() or ann is None:
        continue
    d = np.load(p)
    lab, idx = load_truth(ann, d["frame_idx"])
    keep = lab != ""
    if keep.sum() < 100:
        continue
    n = len(idx)
    rows.append({
        "name": "/".join(video.parts[-4:-1]),
        "gy": d["gaze_yaw"][:n][keep], "gp": d["gaze_pitch"][:n][keep],
        "truth": (lab[keep] == "off"),
    })

print(f"sessions: {len(rows)}\n")

offsets_y = [mode_angle(r["gy"]) for r in rows]
offsets_p = [mode_angle(r["gp"]) for r in rows]
print("Per-session modal gaze direction (estimate of road-ahead in camera frame):")
print(f"  yaw   mean {np.mean(offsets_y):+6.1f}  sd {np.std(offsets_y):5.1f}  "
      f"range [{min(offsets_y):+.1f}, {max(offsets_y):+.1f}] deg")
print(f"  pitch mean {np.mean(offsets_p):+6.1f}  sd {np.std(offsets_p):5.1f}  "
      f"range [{min(offsets_p):+.1f}, {max(offsets_p):+.1f}] deg")
print("\n(a non-zero, consistent offset indicates a fixed camera mounting angle;")
print(" spread across sessions indicates per-installation/seat-position variation)\n")

# --- compare: raw camera frame vs per-session re-centred -------------------
for label, recentre in (("raw camera frame (current)", False),
                        ("re-centred on session modal gaze", True)):
    preds, truths = [], []
    for r in rows:
        gy, gp = r["gy"], r["gp"]
        if recentre:
            gy = gy - mode_angle(gy)
            gp = gp - mode_angle(gp)
        preds.append((np.abs(gy) > YAW_CONE) | (np.abs(gp) > PITCH_CONE))
        truths.append(r["truth"])
    acc, prec, rec, f1 = scores(np.concatenate(preds), np.concatenate(truths))
    print(f"{label:<36} acc {acc:.4f}  precision {prec:.4f}  "
          f"recall {rec:.4f}  F1 {f1:.4f}")

t_all = np.concatenate([r["truth"] for r in rows])
print(f"\nmajority-class baseline               acc {1 - t_all.mean():.4f}  F1 0.0000")
print(f"prevalence (not looking road): {t_all.mean():.1%}")
