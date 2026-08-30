"""P2 — batch extraction of per-frame signal timelines for all DMD sessions.

Populates outputs/signal_cache/ so the notebook's `Run All` is fast and every
downstream phase (P3 driver state, P4 PAD) reads cached signals rather than
re-decoding video. Extraction logic is imported from the notebook to guarantee
the cache matches what the notebook itself would produce (single source of truth).
"""
import sys
import time
from pathlib import Path

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")

import nbformat

nb = nbformat.read(PROJ / "Secure_DMS_Rework.ipynb", as_version=4)
# Execute notebook sections 0-3 in this process to reuse their definitions verbatim.
ns = {"__name__": "__nb__"}
for cell in nb.cells:
    if cell.cell_type != "code":
        continue
    src = cell.source
    if src.strip().startswith("#") and len(src.strip().splitlines()) == 1:
        continue
    if "smoke test" in src:                     # skip the demo, we do the real run below
        src = src.split("# --- smoke test")[0]
    exec(compile(src, "<nb>", "exec"), ns)

extract_signals = ns["extract_signals"]
assert_signals_plausible = ns["assert_signals_plausible"]
dmd_inventory = ns["dmd_inventory"]

split = sys.argv[1] if len(sys.argv) > 1 else "drowsiness"
videos = dmd_inventory[split]["videos"]
print(f"\n=== extracting {len(videos)} {split} sessions ===", flush=True)

t_start = time.perf_counter()
total_frames = 0
for n, video in enumerate(videos, 1):
    t0 = time.perf_counter()
    try:
        sig = extract_signals(video)
        assert_signals_plausible(sig, video.name[:16])
        dt = time.perf_counter() - t0
        n_read = int(sig["total_frames_read"][0])
        total_frames += n_read
        ear = (sig["ear_left"] + sig["ear_right"]) / 2
        print(f"[{n:>2}/{len(videos)}] {video.parent.parent.parent.name}/"
              f"{video.parent.parent.name}/{video.parent.name} "
              f"{len(sig['frame_idx']):>5}/{n_read:>5} frames "
              f"({len(sig['frame_idx'])/n_read:.1%} detected) "
              f"EAR {ear.mean():.3f} | {dt:>5.1f}s = {n_read/dt:>5.1f} fps", flush=True)
    except Exception as exc:                     # noqa: BLE001 — report and continue
        print(f"[{n:>2}/{len(videos)}] {video.name}: FAILED — {type(exc).__name__}: {exc}",
              flush=True)

elapsed = time.perf_counter() - t_start
print(f"\ndone: {total_frames:,} frames in {elapsed/60:.1f} min "
      f"= {total_frames/elapsed:.1f} fps overall", flush=True)
