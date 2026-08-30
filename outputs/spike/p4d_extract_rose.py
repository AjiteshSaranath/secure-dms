"""Tier D — extract signal timelines for a stratified ROSE-Youtu sample.

Sampling protocol (seeded, documented, reproducible):
  * All 20 subjects are represented.
  * Per species, videos are drawn round-robin ACROSS subjects so no subject
    dominates a species; genuine is sampled more heavily because it is the
    calibration class (thresholds are set on the genuine distribution alone).
  * Seed 42.
Rationale for sampling rather than all 3,497 clips (~7.2 h): ~2 windows/clip at a
5 s window gives >=200 windows per attack species, which is ample for stable
per-species APCER, at ~1/3 the compute. The full corpus can be extracted later by
raising the caps — the cache key makes that incremental.

Reuses the notebook's OWN extract_signals (executed headlessly), so Tier-D
signals are produced by exactly the same code path as DMD and Tier C.
"""
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import nbformat

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
ROSE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")

CODES = {"G": "genuine", "Ps": "print_still", "Pq": "print_quiver", "Vl": "replay_lenovo",
         "Vm": "replay_mac", "Mc": "mask_cropped", "Mf": "mask_full", "Mu": "mask_upper"}
N_GENUINE, N_PER_ATTACK = 240, 120
SEED = 42

# ── load the notebook's own definitions (sections 0-3) ─────────────────────
nb = nbformat.read(PROJ / "Secure_DMS_Rework.ipynb", as_version=4)
ns = {"__name__": "__nb__"}
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    src = cell.source
    if "smoke test" in src:
        src = src.split("# --- smoke test")[0]
    if src.strip().startswith("#") and len(src.strip().splitlines()) == 1:
        continue
    # stop after the extraction runner + validity masks (cell 3.3/3.4)
    exec(compile(src, "<nb>", "exec"), ns)
    if "gaze_validity_mask" in src and "assert_signals_plausible" in src:
        break

extract_signals = ns["extract_signals"]

# ── stratified sample ──────────────────────────────────────────────────────
rng = random.Random(SEED)
by_species_subject = defaultdict(lambda: defaultdict(list))
for v in sorted(ROSE.rglob("*.mp4")):
    code = v.stem.split("_")[0]
    if code in CODES:
        by_species_subject[CODES[code]][v.parent.name].append(v)


def stratified(species, n_target):
    """Round-robin across subjects so no subject dominates the species."""
    per_subject = {s: rng.sample(vs, len(vs)) for s, vs in by_species_subject[species].items()}
    order = sorted(per_subject)
    picked, i = [], 0
    while len(picked) < n_target and any(per_subject[s] for s in order):
        s = order[i % len(order)]
        if per_subject[s]:
            picked.append(per_subject[s].pop())
        i += 1
    return picked


selection = {sp: stratified(sp, N_GENUINE if sp == "genuine" else N_PER_ATTACK)
             for sp in CODES.values()}
total = sum(len(v) for v in selection.values())
print(f"stratified sample: {total} clips "
      + " | ".join(f"{k} {len(v)}" for k, v in selection.items()), flush=True)

# ── extract ────────────────────────────────────────────────────────────────
t0 = time.perf_counter()
done = failed = 0
manifest = []
for species, clips in selection.items():
    for n, clip in enumerate(clips, 1):
        try:
            sig = extract_signals(clip)
            n_det = len(sig["frame_idx"])
            n_read = int(sig["total_frames_read"][0])
            manifest.append({"file": clip.name, "subject": clip.parent.name,
                             "species": species, "detected": n_det, "read": n_read})
            done += 1
            if n % 20 == 0 or n == len(clips):
                el = time.perf_counter() - t0
                print(f"  [{species:<14} {n:>3}/{len(clips)}] {clip.name[:34]:<34} "
                      f"{n_det:>4}/{n_read:>4} frames | {done}/{total} done, "
                      f"{el/60:.1f} min elapsed", flush=True)
        except Exception as exc:                       # noqa: BLE001
            failed += 1
            print(f"  [{species}] {clip.name}: FAILED {type(exc).__name__}: {exc}", flush=True)

import json
(PROJ / "outputs" / "p4d_rose_manifest.json").write_text(
    json.dumps({"seed": SEED, "n_genuine": N_GENUINE, "n_per_attack": N_PER_ATTACK,
                "extracted": done, "failed": failed, "items": manifest}, indent=2),
    encoding="utf-8")
print(f"\ndone: {done} extracted, {failed} failed in "
      f"{(time.perf_counter()-t0)/60:.1f} min", flush=True)
