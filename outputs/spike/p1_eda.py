"""P1 — Data inventory + EDA + gate G1.

Three questions this answers, before any PAD/driver-state code is written:
  Q1 (inventory)  What video + annotation material actually exists?
  Q2 (DMD)        Do DMD annotations give per-frame eye/yawn ground truth
                  usable for PERCLOS/EAR/MAR validation?
  Q3 (gate G1)    Can NUAA support *temporal* PAD — i.e. do its frames form
                  usable sequences with measurable blink / motion signal?

Writes: outputs/p1_eda_results.json + console log (provenance rule).
"""
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project")
PROJ = ROOT / "dms_project_fixed_v3" / "dms_jupyter"
DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
NUAA = ROOT / "datasets" / "NormalizedFace"
MODEL = PROJ / "models" / "face_landmarker.task"
OUT = PROJ / "outputs"

results = {"generated": time.strftime("%Y-%m-%d %H:%M:%S")}


def log(*p):
    line = " ".join(str(x) for x in p)
    print(line, flush=True)


# --------------------------------------------------------------------------
# Q1 + Q2 — DMD inventory and annotation ground truth
# --------------------------------------------------------------------------
log("=" * 72)
log("Q1/Q2 — DMD inventory & annotation ground truth")
log("=" * 72)

dmd_stats = {}
for split, folder in (("drowsiness", "Drowsiness"), ("distraction", "Distraction")):
    videos = sorted((DMD / folder).rglob("*rgb_face.mp4"))
    anns = sorted((DMD / folder).rglob("*rgb_ann_*.json"))
    dmd_stats[split] = {"rgb_face_videos": len(videos), "annotations": len(anns)}
    log(f"{split}: {len(videos)} rgb_face videos, {len(anns)} annotation files")

    if videos:
        cap = cv2.VideoCapture(str(videos[0]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        dmd_stats[split]["sample_video"] = {
            "name": videos[0].name, "fps": round(fps, 2), "frames": n,
            "resolution": f"{w}x{h}", "duration_s": round(n / fps, 1) if fps else None,
        }
        log(f"  sample: {w}x{h} @ {fps:.1f} fps, {n} frames "
            f"({n / fps:.0f} s)" if fps else "  sample: fps unknown")

# Annotation label vocabulary + PERCLOS-relevant coverage
label_counts = defaultdict(int)
eye_closed_frames = 0
total_ann_frames = 0
per_session = []
for ann_path in sorted((DMD / "Drowsiness").rglob("*rgb_ann_drowsiness.json")):
    with ann_path.open(encoding="utf-8") as fh:
        ol = json.load(fh)["openlabel"]
    fi = ol.get("frame_intervals", [])
    if isinstance(fi, dict):
        fi = [fi]
    n_frames = sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in fi)
    closed = 0
    for act in ol.get("actions", {}).values():
        label_counts[act["type"]] += 1
        if act["type"] == "eyes_state/close":
            closed = sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in act["frame_intervals"])
    eye_closed_frames += closed
    total_ann_frames += n_frames
    per_session.append({"session": ann_path.parent.name, "frames": n_frames,
                        "eyes_closed_frames": closed,
                        "perclos_gt": round(closed / n_frames, 4) if n_frames else None})

log(f"\nDrowsiness annotation vocabulary: {dict(label_counts)}")
log(f"annotated sessions: {len(per_session)}, total frames: {total_ann_frames:,}")
if total_ann_frames:
    log(f"ground-truth eye-closure fraction (PERCLOS reference): "
        f"{eye_closed_frames / total_ann_frames:.4f} "
        f"({eye_closed_frames:,}/{total_ann_frames:,} frames)")
    log("per-session PERCLOS ground truth: " +
        ", ".join(f"{s['perclos_gt']:.3f}" for s in per_session if s["perclos_gt"] is not None))

dmd_stats["annotation_vocabulary"] = dict(label_counts)
dmd_stats["annotated_sessions"] = len(per_session)
dmd_stats["total_annotated_frames"] = total_ann_frames
dmd_stats["gt_eye_closure_fraction"] = (
    round(eye_closed_frames / total_ann_frames, 4) if total_ann_frames else None)
dmd_stats["per_session"] = per_session
results["dmd"] = dmd_stats

# --------------------------------------------------------------------------
# Q3 — GATE G1: NUAA temporal viability
# --------------------------------------------------------------------------
log("\n" + "=" * 72)
log("Q3 / GATE G1 — NUAA temporal viability for behavioural PAD")
log("=" * 72)

# Filename format (NUAA readme): ID_glasses_pos_session_picNo.bmp
# A "capture" = (subject, glasses, pos, session); picNo orders frames within it.
FNAME = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.bmp$", re.IGNORECASE)


def index_captures(root: Path):
    captures = defaultdict(list)
    for path in root.rglob("*.bmp"):
        m = FNAME.match(path.name)
        if m:
            sid, glasses, pos, sess, pic = m.groups()
            captures[(sid, glasses, pos, sess)].append((int(pic), path))
    for key in captures:
        captures[key].sort()
    return captures


client_caps = index_captures(NUAA / "ClientNormalized")
imposter_caps = index_captures(NUAA / "ImposterNormalized")
log(f"client captures: {len(client_caps)} | imposter captures: {len(imposter_caps)}")


def gap_stats(caps):
    """Frame-number gaps within a capture reveal whether frames are contiguous."""
    gaps, lengths = [], []
    for frames in caps.values():
        nums = [n for n, _ in frames]
        lengths.append(len(nums))
        gaps.extend(np.diff(nums).tolist())
    return np.array(gaps), np.array(lengths)


for name, caps in (("client", client_caps), ("imposter", imposter_caps)):
    gaps, lengths = gap_stats(caps)
    if len(gaps):
        contiguous = float(np.mean(gaps == 1))
        log(f"{name}: capture length median {np.median(lengths):.0f} "
            f"(min {lengths.min()}, max {lengths.max()}) | "
            f"frame-number gap median {np.median(gaps):.0f}, mean {gaps.mean():.1f} | "
            f"contiguous(gap==1) {contiguous:.1%}")
        results.setdefault("nuaa_sequence_structure", {})[name] = {
            "captures": len(caps),
            "median_capture_length": float(np.median(lengths)),
            "median_frame_gap": float(np.median(gaps)),
            "mean_frame_gap": round(float(gaps.mean()), 2),
            "fraction_contiguous": round(contiguous, 4),
        }

# --- measure EAR / motion signal on reconstructed sequences ---------------
options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)), num_faces=1)
landmarker = vision.FaceLandmarker.create_from_options(options)

LEFT = [362, 385, 387, 263, 373, 380]
RIGHT = [33, 160, 158, 133, 153, 144]


def ear_of(lm, ids):
    p = [np.array([lm[i].x, lm[i].y]) for i in ids]
    denom = np.linalg.norm(p[0] - p[3])
    if denom < 1e-6:
        return np.nan
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * denom)


def analyse(caps, label, max_captures=25, max_frames=60):
    """Per-capture: EAR trace stats, landmark motion, detection rate."""
    rng = np.random.default_rng(42)
    keys = sorted(caps.keys())
    if len(keys) > max_captures:
        keys = [keys[i] for i in rng.choice(len(keys), max_captures, replace=False)]

    rows = []
    for key in keys:
        frames = caps[key][:max_frames]
        ears, centroids, detected = [], [], 0
        for _, path in frames:
            img_bgr = cv2.imread(str(path))
            if img_bgr is None:
                continue
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                             data=np.ascontiguousarray(rgb)))
            if not res.face_landmarks:
                continue
            detected += 1
            lm = res.face_landmarks[0]
            ears.append(np.nanmean([ear_of(lm, LEFT), ear_of(lm, RIGHT)]))
            pts = np.array([[p.x, p.y] for p in lm])
            centroids.append(pts.mean(axis=0))

        if len(ears) < 5:
            continue
        ears = np.array(ears, dtype=float)
        centroids = np.array(centroids)
        motion = np.linalg.norm(np.diff(centroids, axis=0), axis=1) if len(centroids) > 1 else np.array([0.0])
        rows.append({
            "capture": "_".join(key),
            "n_frames": len(frames),
            "detection_rate": round(detected / max(len(frames), 1), 3),
            "ear_mean": round(float(np.nanmean(ears)), 4),
            "ear_std": round(float(np.nanstd(ears)), 4),
            "ear_min": round(float(np.nanmin(ears)), 4),
            # blink-like: EAR dips below Soukupova-Cech-style threshold (R9)
            "frames_below_0p21": int(np.sum(ears < 0.21)),
            "motion_median": round(float(np.median(motion)), 5),
            "motion_p95": round(float(np.percentile(motion, 95)), 5),
        })
    log(f"\n{label}: analysed {len(rows)} captures")
    if rows:
        for k in ("detection_rate", "ear_std", "frames_below_0p21", "motion_median"):
            vals = np.array([r[k] for r in rows], dtype=float)
            log(f"  {k:>18}: median {np.median(vals):.4f}  mean {vals.mean():.4f}")
    return rows


client_rows = analyse(client_caps, "NUAA client (bona fide)")
imposter_rows = analyse(imposter_caps, "NUAA imposter (print attack)")
results["nuaa_client_captures"] = client_rows
results["nuaa_imposter_captures"] = imposter_rows


def summarise(rows, key):
    vals = np.array([r[key] for r in rows], dtype=float)
    return {"median": round(float(np.median(vals)), 5),
            "mean": round(float(vals.mean()), 5),
            "std": round(float(vals.std()), 5)}


log("\n" + "-" * 72)
log("GATE G1 SUMMARY — bona fide vs attack separability of temporal cues")
log("-" * 72)
g1 = {}
for key in ("detection_rate", "ear_std", "frames_below_0p21", "motion_median", "motion_p95"):
    c, i = summarise(client_rows, key), summarise(imposter_rows, key)
    cv = np.array([r[key] for r in client_rows], dtype=float)
    iv = np.array([r[key] for r in imposter_rows], dtype=float)
    pooled = np.sqrt((cv.var() + iv.var()) / 2)
    d = float((cv.mean() - iv.mean()) / pooled) if pooled > 1e-9 else 0.0
    g1[key] = {"client": c, "imposter": i, "cohens_d": round(d, 3)}
    log(f"{key:>18}: client {c['median']:.4f} | imposter {i['median']:.4f} | Cohen's d {d:+.3f}")
results["gate_g1"] = g1

OUT.mkdir(exist_ok=True)
with (OUT / "p1_eda_results.json").open("w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=2)
log(f"\nwritten: {OUT / 'p1_eda_results.json'}")
