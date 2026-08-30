"""P4 diagnostic — attack videos yielded 0 PAD windows. Extraction or windowing?"""
import hashlib
import json
from pathlib import Path

import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
TIER_C = PROJ / "outputs" / "tier_c_attacks"
CACHE = PROJ / "outputs" / "signal_cache"
EXTRACTOR_VERSION = "p2.3"


def cache_path(v):
    key = f"{v.name}|{v.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


manifest = json.loads((TIER_C / "manifest.json").read_text(encoding="utf-8"))
print(f"{'file':<26}{'rendered':>10}{'detected':>10}{'rate':>8}{'window need':>13}")
need = None
for item in manifest["items"][:8]:
    p = TIER_C / item["file"]
    c = cache_path(p)
    if not c.exists():
        print(f"{item['file']:<26}  NO CACHE (extraction failed or not run)")
        continue
    d = np.load(c)
    n_det = len(d["frame_idx"])
    n_read = int(d["total_frames_read"][0])
    fps = float(d["fps"][0])
    need = max(int(round(10.0 * fps)), 2)
    print(f"{item['file']:<26}{n_read:>10}{n_det:>10}{n_det/max(n_read,1):>8.1%}{need:>13}")

if need:
    print(f"\nwindow size needed: {need} frames of detected face for ONE 10 s window.")
    print("Attack clips were rendered at exactly 10 s, so a single undetected frame")
    print("drops the count below one full window and `range(0, n-w+1, w)` is empty.")
    print("=> windowing/duration bug, not a face-detection failure.")
