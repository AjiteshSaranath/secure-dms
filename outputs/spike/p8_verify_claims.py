"""P8 — cross-verify Section VII prose against what the code actually does.

Checks each factual claim in the paper text against the artefacts on disk and
the notebook source, so that no sentence survives that the implementation does
not support.
"""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import nbformat

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
TIER_C = PROJ / "outputs" / "tier_c_attacks"
NB = PROJ / "Secure_DMS_Rework.ipynb"

print("=" * 72)
print("CLAIM 1 — synthesised presentations (Sec. VII-A3)")
print("=" * 72)
man = json.loads((TIER_C / "manifest.json").read_text(encoding="utf-8"))
items = man["items"]
by_sp = Counter(i["species"] for i in items)
print(f'"thirty-six attack sequences"            -> {len(items)} rendered')
print(f'"three instrument species, equal number" -> {dict(by_sp)}')
print(f'"20 s at 29.76 Hz"                       -> manifest says '
      f'{man["duration_s"]} s @ {man["fps"]} Hz')

# what the WRITTEN containers actually report (the render fps and the muxed fps
# are not necessarily identical — this bit us once before)
fps_seen, frames_seen, res_seen = Counter(), Counter(), Counter()
for it in items:
    p = TIER_C / it["file"]
    if not p.exists():
        continue
    cap = cv2.VideoCapture(str(p))
    fps_seen[round(cap.get(cv2.CAP_PROP_FPS), 2)] += 1
    frames_seen[int(cap.get(cv2.CAP_PROP_FRAME_COUNT))] += 1
    res_seen[f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
             f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"] += 1
    cap.release()
print(f"   containers report fps={dict(fps_seen)} frames={dict(frames_seen)}")
print(f"   -> actual duration = {list(frames_seen)[0] / list(fps_seen)[0]:.2f} s")
print(f"   resolutions: {dict(res_seen)}")

srcs = {i["source_session"] for i in items}
print(f'"source frames drawn from the same corpus as the genuine class"')
print(f"   -> {len(srcs)} distinct DMD sessions used as sources, e.g. "
      f"{sorted(srcs)[:4]}")
print(f"   -> resolution matches DMD native (1280x720): "
      f"{'YES' if all(r == '1280x720' for r in res_seen) else 'NO ' + str(dict(res_seen))}")

print()
print("=" * 72)
print("CLAIM 2 — reproducibility machinery (Sec. VII-E)")
print("=" * 72)
nb = nbformat.read(NB, as_version=4)
src = {c.get("id"): c.source for c in nb.cells if c.cell_type == "code"}
alltext = "\n".join(src.values())


def show(label, pattern, cell_hint=None):
    hits = [(cid, ln.strip()) for cid, s in src.items()
            for ln in s.splitlines() if re.search(pattern, ln)]
    print(f"\n{label}")
    if not hits:
        print("   *** NOT FOUND IN NOTEBOOK ***")
    for cid, ln in hits[:4]:
        print(f"   [{cid}] {ln[:88]}")


show('"All random seeds are fixed"', r"random\.seed|np\.random\.seed|SEED\s*=|manual_seed")
show('"weight file verified against a recorded SHA-256 digest"',
     r"EXPECTED_HASHES|sha256\(path\.read_bytes|ALL_ARTEFACTS_OK")
show('"cached under keys derived from source video, extractor version, parameters"',
     r"def _cache_path|EXTRACTOR_VERSION\s*=|key = f")
show('"emitted to a timestamped run log"', r"RUN_LOG\s*=|def log\(")

print("\n--- torch determinism check (claim says ALL seeds) ---")
if "torch.manual_seed" in alltext:
    print("   torch.manual_seed present")
else:
    print("   *** torch.manual_seed ABSENT — the phrase 'all random seeds are")
    print("       fixed' overstates what the notebook does. Either seed torch")
    print("       or reword. NOTE: every model runs under torch.no_grad() in")
    print("       eval mode with no dropout/sampling, so outputs are")
    print("       deterministic in practice — but the sentence should say that")
    print("       rather than claim a seed that is not set.")

print("\n--- where randomness actually enters ---")
for cid, s in src.items():
    for ln in s.splitlines():
        if re.search(r"default_rng|rng\.|random\.Random|np\.random", ln):
            print(f"   [{cid}] {ln.strip()[:88]}")
            break
