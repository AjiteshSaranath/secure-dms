"""Tier D — ROSE-Youtu inventory and feasibility check BEFORE extraction.

Decisive questions, each of which has already cost a re-run once if assumed:
  1. Duration vs the 10 s PAD window — Tier-C clips rendered at exactly one
     window length produced ZERO windows. If ROSE clips are ~5 s the window must
     be re-specified for this corpus (and stated), not silently mismatched.
  2. Resolution / face size — ROSE is mobile-captured; the perception stack was
     validated on 1280x720 DMD video.
  3. Class balance per attack species (ISO 30107-3 reports per species).
  4. Extraction cost at the measured ~40 fps.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

ROSE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")
OUT = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\outputs")

# Official ROSE-Youtu naming: L_S_D_x_E_p_N  (rose1.ntu.edu.sg)
SPECIES = {
    "G":  ("genuine", "genuine person"),
    "Ps": ("print_still", "still printed paper"),
    "Pq": ("print_quiver", "quivering printed paper"),
    "Vl": ("replay_lenovo", "video replayed on a Lenovo LCD"),
    "Vm": ("replay_mac", "video replayed on a Mac LCD"),
    "Mc": ("mask_cropped", "paper mask, eyes+mouth cropped out"),
    "Mf": ("mask_full", "paper mask, no cropping"),
    "Mu": ("mask_upper", "paper mask, upper part cut"),
    "Ml": ("mask_lower", "paper mask, lower part cut"),
}
DEVICES = {"HS": "Hasee", "HW": "Huawei", "IP": "iPad", "5s": "iPhone 5s", "ZTE": "ZTE"}

videos = sorted(ROSE.rglob("*.mp4"))
by_species, by_subject, by_device = Counter(), Counter(), Counter()
parsed = []
for v in videos:
    parts = v.stem.split("_")
    code = parts[0]
    species = SPECIES.get(code, ("unknown", code))[0]
    device = parts[2] if len(parts) > 2 else "?"
    subject = v.parent.name
    by_species[species] += 1
    by_subject[subject] += 1
    by_device[device] += 1
    parsed.append({"path": v, "species": species, "code": code,
                   "device": device, "subject": subject})

print(f"ROSE-Youtu: {len(videos):,} videos, {len(by_subject)} subjects\n")
print(f"{'species':<16}{'count':>7}   description")
for code, (name, desc) in SPECIES.items():
    if by_species.get(name):
        print(f"{name:<16}{by_species[name]:>7}   {desc}")
n_attack = sum(c for s, c in by_species.items() if s != "genuine")
print(f"\ngenuine {by_species['genuine']:,} | attack {n_attack:,} "
      f"(prevalence {n_attack / len(videos):.1%})")
print(f"devices: {dict(by_device)}")

# ── sample video properties (stratified across species) ────────────────────
rng = random.Random(42)
by_sp = defaultdict(list)
for p in parsed:
    by_sp[p["species"]].append(p)

print(f"\n{'species':<16}{'n':>4}{'fps':>7}{'frames':>8}{'seconds':>9}{'resolution':>12}")
durations, all_props = [], {}
for sp, items in sorted(by_sp.items()):
    sample = rng.sample(items, min(6, len(items)))
    fps_l, fr_l, res_l = [], [], []
    for it in sample:
        cap = cv2.VideoCapture(str(it["path"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps > 0 and n > 0:
            fps_l.append(fps); fr_l.append(n); res_l.append(f"{w}x{h}")
    if not fps_l:
        continue
    mfps = sum(fps_l) / len(fps_l)
    mfr = sum(fr_l) / len(fr_l)
    secs = mfr / mfps
    durations.append(secs)
    all_props[sp] = {"fps": round(mfps, 2), "frames": round(mfr), "seconds": round(secs, 1),
                     "resolutions": sorted(set(res_l))}
    print(f"{sp:<16}{len(sample):>4}{mfps:>7.1f}{mfr:>8.0f}{secs:>9.1f}{res_l[0]:>12}")

med = sorted(durations)[len(durations) // 2]
print(f"\nmedian clip duration across species: {med:.1f} s")
print(f"PAD window currently 10 s -> {'OK' if med > 11 else 'TOO LONG for this corpus'}")
if med <= 11:
    print(f"  => Tier-D evaluation needs a shorter window (e.g. {max(3, int(med) - 1)} s).")
    print("     Window length must then be reported per corpus, and the blink criterion")
    print("     (>=1 blink per window) rescaled: 15-30 blinks/min => a 5 s window")
    print("     expects 1.25-2.5 blinks, so >=1 remains physiologically sound.")

est_frames = sum(all_props[s]["frames"] * by_species[s] for s in all_props if by_species.get(s))
print(f"\nestimated total frames: {est_frames:,} -> ~{est_frames / 40 / 3600:.1f} h at 40 fps")

(OUT / "p4d_rose_inventory.json").write_text(json.dumps(
    {"n_videos": len(videos), "n_subjects": len(by_subject),
     "by_species": dict(by_species), "by_device": dict(by_device),
     "properties": all_props, "median_seconds": med,
     "estimated_frames": est_frames}, indent=2), encoding="utf-8")
print(f"\nwritten: {OUT / 'p4d_rose_inventory.json'}")
