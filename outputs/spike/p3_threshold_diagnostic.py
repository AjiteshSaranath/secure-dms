"""P3 diagnostic — is the eye-closure criterion implemented correctly?

First P3 run gave recall 1.000 but precision 0.297 and a +0.25 PERCLOS bias:
the criterion fires on far too many frames. Suspected cause: the implementation
blended two DIFFERENT published criteria into one formula --

    ear <= (1 - P80) * baseline + EAR_CLOSED * P80        # <- incoherent

mixing Soukupova & Cech's ABSOLUTE blink threshold (R9) with the RELATIVE P80
criterion of PERCLOS (R15, R16). With baseline ~0.30 that evaluates to ~0.228,
which sits right at the median open-eye EAR, so most frames are marked closed.

Correct P80 (R15, R16): the eye counts as closed when its aperture falls to
<=20 % of the individual's fully-open aperture, i.e.  ear <= 0.20 * baseline.

This script checks the corrected formula against ground truth and sweeps both
the fraction and the baseline percentile to show the criterion is not
knife-edge (feeds the §8 sensitivity analysis; it is a diagnostic, not tuning).
"""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.2"


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def frame_intervals(node):
    fi = node.get("frame_intervals", [])
    return [fi] if isinstance(fi, dict) else fi


def load_truth(ann_path, frame_idx):
    ol = json.loads(ann_path.read_text(encoding="utf-8"))["openlabel"]
    n = sum(iv["frame_end"] - iv["frame_start"] + 1 for iv in frame_intervals(ol))
    timeline = np.zeros(n, dtype=bool)
    for act in ol.get("actions", {}).values():
        if act["type"] == "eyes_state/close":
            for iv in frame_intervals(act):
                timeline[max(iv["frame_start"], 0):min(iv["frame_end"] + 1, n)] = True
    idx = frame_idx[frame_idx < n]
    return timeline[idx], len(idx)


# --- gather cached sessions -------------------------------------------------
sessions = []
for video in sorted(DMD_DROW.rglob("*rgb_face.mp4")):
    p = cache_path(video)
    if not p.exists():
        continue
    ann = next(video.parent.glob("*rgb_ann_drowsiness.json"), None)
    if ann is None:
        continue
    d = np.load(p)
    if len(d["frame_idx"]) < 1000:            # skip the 300-frame smoke-test cache
        continue
    ear = np.nanmean(np.stack([d["ear_left"], d["ear_right"]]), axis=0)
    truth, n = load_truth(ann, d["frame_idx"])
    sessions.append({"name": "/".join(video.parts[-4:-1]), "ear": ear[:n], "truth": truth})

print(f"sessions available: {len(sessions)}")


def scores(pred, truth):
    tp = np.sum(pred & truth); fp = np.sum(pred & ~truth); fn = np.sum(~pred & truth)
    tn = np.sum(~pred & ~truth)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (tp + tn) / len(truth), prec, rec, f1


def evaluate(fraction, percentile):
    preds, truths, bias = [], [], []
    for s in sessions:
        baseline = np.nanpercentile(s["ear"], percentile)
        pred = s["ear"] <= fraction * baseline
        preds.append(pred); truths.append(s["truth"])
        bias.append(pred.mean() - s["truth"].mean())
    p, t = np.concatenate(preds), np.concatenate(truths)
    acc, prec, rec, f1 = scores(p, t)
    return acc, prec, rec, f1, float(np.mean(bias))


print("\n--- OLD (incoherent blended) formula, for the record ---")
preds, truths = [], []
for s in sessions:
    baseline = np.nanpercentile(s["ear"], 85)
    preds.append(s["ear"] <= 0.20 * baseline + 0.21 * 0.80)
    truths.append(s["truth"])
acc, prec, rec, f1 = scores(np.concatenate(preds), np.concatenate(truths))
print(f"acc {acc:.4f} precision {prec:.4f} recall {rec:.4f} F1 {f1:.4f}")

print("\n--- CORRECTED P80: ear <= 0.20 * open-baseline (R15, R16) ---")
acc, prec, rec, f1, bias = evaluate(0.20, 85)
print(f"acc {acc:.4f} precision {prec:.4f} recall {rec:.4f} F1 {f1:.4f} PERCLOS bias {bias:+.4f}")

print("\n--- sensitivity: closure fraction (baseline = 85th pct) ---")
print(f"{'fraction':>9}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'PERCLOS bias':>14}")
for frac in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
    acc, prec, rec, f1, bias = evaluate(frac, 85)
    mark = "  <- P80 (published)" if abs(frac - 0.20) < 1e-9 else ""
    print(f"{frac:>9.2f}{acc:>8.4f}{prec:>8.4f}{rec:>8.4f}{f1:>8.4f}{bias:>+14.4f}{mark}")

print("\n--- sensitivity: open-eye baseline percentile (fraction = 0.20) ---")
print(f"{'pctile':>9}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'PERCLOS bias':>14}")
for pct in (75, 80, 85, 90, 95, 99):
    acc, prec, rec, f1, bias = evaluate(0.20, pct)
    print(f"{pct:>9}{acc:>8.4f}{prec:>8.4f}{rec:>8.4f}{f1:>8.4f}{bias:>+14.4f}")
