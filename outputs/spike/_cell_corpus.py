"""3.5 — corpus extraction: make the notebook self-describing about its inputs.

Everything downstream reads cached per-frame signals. If a session is not cached
the validation cells skip it, which fails QUIETLY — an incomplete corpus looks
like a small corpus rather than an error. This cell makes coverage explicit.

Extraction of the full corpus takes roughly 8 hours (98 sessions, 1.37 M frames
at ~41 fps), so it is NOT triggered automatically by `Run All`. Set
EXTRACT_MISSING = True to run it in-notebook, or use the headless batch runner:

    .venv\\Scripts\\python outputs\\spike\\p2_extract_all.py drowsiness
    .venv\\Scripts\\python outputs\\spike\\p2_extract_all.py distraction

The batch runner executes THIS notebook's cells — it is the same code driven
headlessly, not a second implementation.
"""
EXTRACT_MISSING = False       # set True to extract in-notebook (~8 h from cold)

CORPUS = {split: dmd_inventory[split]["videos"] for split in ("drowsiness", "distraction")}

coverage, missing_total = {}, 0
for split, videos in CORPUS.items():
    missing = [v for v in videos if not _cache_path(v).exists()]
    coverage[split] = {"total": len(videos), "cached": len(videos) - len(missing),
                       "missing": missing}
    missing_total += len(missing)
    log(f"corpus {split:<12} {len(videos) - len(missing):>3}/{len(videos):<3} sessions cached")

if missing_total == 0:
    n_files = len(list(SIGNAL_CACHE.glob("*.npz")))
    mb = sum(f.stat().st_size for f in SIGNAL_CACHE.glob("*.npz")) / 1024 / 1024
    log(f"  full corpus available ({n_files} cache files, {mb:.0f} MB) — "
        f"downstream sections will run on all sessions")
elif EXTRACT_MISSING:
    log(f"  extracting {missing_total} missing sessions — this will take hours")
    for split, info in coverage.items():
        for n, video in enumerate(info["missing"], 1):
            sig = extract_signals(video)
            qc = assert_signals_plausible(sig, video.name[:16])
            log(f"    [{split} {n}/{len(info['missing'])}] {video.name[:34]} "
                f"{len(sig['frame_idx']):,} frames, "
                f"invalid gaze {qc['invalid_gaze_rate']:.2%}")
else:
    log(f"  WARNING: {missing_total} sessions NOT cached. Downstream validation will")
    log(f"  silently SKIP them and report a partial corpus. Set EXTRACT_MISSING = True")
    log(f"  or run outputs/spike/p2_extract_all.py before trusting any results below.")
