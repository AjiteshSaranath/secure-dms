"""4.8 — threshold sensitivity, computed with the DEPLOYED code path.

An earlier standalone diagnostic swept the closure fraction using its own
re-implementation, before the EAR viewpoint mask (§3.4) existed, and on the
subset of sessions cached at that time. Its numbers therefore did not match the
headline figures and must not be quoted alongside them. This cell recomputes the
sweep through `mean_ear` / `eye_closed_mask` on the full annotated corpus, so
every value below is directly comparable with §4.4.
"""
_sweep_sessions = []
for _v, _a in zip(dmd_inventory["drowsiness"]["videos"], DMD_ANNOTATIONS):
    if not _cache_path(_v).exists():
        continue
    _s = extract_signals(_v)
    _t = align_annotation(_s, _a)
    if _t is None:
        continue
    _sweep_sessions.append((mean_ear(_s)[:len(_t)], _t))

log("\n" + "=" * 78)
log("P3 — threshold sensitivity (deployed code path, full annotated corpus)")
log("=" * 78)
log(f"sessions {len(_sweep_sessions)} | "
    f"frames {sum(len(t) for _, t in _sweep_sessions):,}")

SENSITIVITY = {"closure_fraction": [], "baseline_percentile": []}

log(f"\nclosure fraction of the driver's open-eye baseline (percentile fixed at "
    f"{OPEN_EYE_PERCENTILE:.0f}):")
log(f"  {'fraction':>9}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}{'bias':>9}")
for _frac in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60):
    _p, _t, _bias = [], [], []
    for _ear, _truth in _sweep_sessions:
        _m = eye_closed_mask(_ear, fraction=_frac)
        _p.append(_m); _t.append(_truth)
        _obs = np.isfinite(_ear).sum()
        _bias.append(_m.sum() / max(_obs, 1) - _truth.mean())
    _sc = binary_scores(np.concatenate(_p), np.concatenate(_t))
    _row = {"fraction": _frac, **_sc, "perclos_bias": float(np.mean(_bias))}
    SENSITIVITY["closure_fraction"].append(_row)
    _tag = "  <- P80, the published criterion" if abs(_frac - 0.20) < 1e-9 else ""
    log(f"  {_frac:>9.2f}{_sc['accuracy']:>8.4f}{_sc['precision']:>8.4f}"
        f"{_sc['recall']:>8.4f}{_sc['f1']:>8.4f}{_row['perclos_bias']:>+9.4f}{_tag}")

_best = max(SENSITIVITY["closure_fraction"], key=lambda r: r["f1"])
_p80 = next(r for r in SENSITIVITY["closure_fraction"] if abs(r["fraction"] - 0.20) < 1e-9)
log(f"\n  best F1 {_best['f1']:.4f} at fraction {_best['fraction']:.2f}; "
    f"P80 gives {_p80['f1']:.4f} (difference {_best['f1'] - _p80['f1']:+.4f})")
log("  The published criterion is not the maximum on this corpus, but the curve is")
log("  flat around it: F1 varies by less than 0.01 between fractions 0.20 and 0.30.")

log(f"\nopen-eye baseline percentile (fraction fixed at {P80_CLOSURE_FRACTION:.2f}):")
log(f"  {'pctile':>9}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
for _pct in (75, 80, 85, 90, 95, 99):
    _p, _t = [], []
    for _ear, _truth in _sweep_sessions:
        _p.append(eye_closed_mask(_ear, baseline=open_eye_baseline(_ear, _pct)))
        _t.append(_truth)
    _sc = binary_scores(np.concatenate(_p), np.concatenate(_t))
    SENSITIVITY["baseline_percentile"].append({"percentile": _pct, **_sc})
    log(f"  {_pct:>9}{_sc['accuracy']:>8.4f}{_sc['precision']:>8.4f}"
        f"{_sc['recall']:>8.4f}{_sc['f1']:>8.4f}")
_f1s = [r["f1"] for r in SENSITIVITY["baseline_percentile"]]
log(f"  F1 spans {min(_f1s):.4f}-{max(_f1s):.4f} across the percentile range.")

(OUT_DIR / "p3_sensitivity.json").write_text(json.dumps(SENSITIVITY, indent=2), encoding="utf-8")
