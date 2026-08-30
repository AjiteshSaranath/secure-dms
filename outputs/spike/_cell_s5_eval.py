"""5.2 — PAD evaluation on Tier A (bona fide) vs Tier C (simulated attacks).

Domain control (P1_FINDINGS §4.1): the attack videos are RENDERED FROM the same
DMD footage that supplies the bona-fide class, so camera, subjects, lighting and
resolution are identical and only the presentation physics differ. Pairing DMD
bona fide against an external corpus would instead let a detector separate the
classes on capture domain — the failure the v7 paper diagnosed for NUAA (D3).

Protocol order matters and is enforced by the code: cues are computed for every
window, thresholds are calibrated on the GENUINE windows alone, and only then are
attack windows judged. No attack data informs any threshold.

Read the numbers as MECHANISM VALIDATION, not field performance — see §5.3.
"""
TIER_C_DIR = OUT_DIR / "tier_c_attacks"
TIER_C_MANIFEST = json.loads((TIER_C_DIR / "manifest.json").read_text(encoding="utf-8"))

# --- 1. cues for every window ---------------------------------------------
bona_cues, bona_sessions = [], 0
for split in ("drowsiness", "distraction"):
    for video in dmd_inventory[split]["videos"]:
        if not _cache_path(video).exists():
            continue
        bona_cues.extend(pad_windows(extract_signals(video), label=video.name))
        bona_sessions += 1

attack_cues, attack_sessions = defaultdict(list), defaultdict(int)
for item in TIER_C_MANIFEST["items"]:
    path = TIER_C_DIR / item["file"]
    if not path.exists():
        continue
    attack_cues[item["species"]].extend(
        pad_windows(extract_signals(path), label=item["file"]))
    attack_sessions[item["species"]] += 1

# --- 2. calibrate on GENUINE windows only ---------------------------------
cal = calibrate_pad_thresholds(bona_cues)
log("\n" + "=" * 78)
log("P4 — PAD: Tier A bona fide (real DMD video) vs Tier C simulated attacks")
log("=" * 78)
log(f"thresholds calibrated on {cal['n_calibration_windows']:,} GENUINE windows only "
    f"(target {cal['target_bpcer_per_cue']:.0%} BPCER per cue):")
log(f"  ocular variability >= {cal['gaze_variability_deg']:.3f} deg")
log(f"  deformation residual >= {cal['deformation_residual']:.6f}")
log(f"  blink count >= {PAD_MIN_BLINKS} per {PAD_WINDOW_S:.0f} s window (R20 physiology)")

# --- 3. judge every window -------------------------------------------------
bona_fide = apply_pad_decisions(bona_cues)
attack_windows = {sp: apply_pad_decisions(ws) for sp, ws in attack_cues.items()}
PAD_RESULTS = iso_30107_metrics(bona_fide, attack_windows)

log(f"\nbona fide: {bona_sessions} sessions -> {len(bona_fide):,} windows "
    f"of {PAD_WINDOW_S:.0f} s")
for sp, ws in attack_windows.items():
    log(f"attack   : {sp:<16} {attack_sessions[sp]:>2} videos -> {len(ws):>4} windows")

log("\nISO/IEC 30107-3 metrics (R32):")
for sp, v in PAD_RESULTS["apcer_per_species"].items():
    log(f"  APCER {sp:<16} {v:.4f}   (attack windows wrongly accepted as live)")
log(f"  APCER worst-case      {PAD_RESULTS['apcer_worst']:.4f}   (the standard reports the worst species)")
log(f"  BPCER                 {PAD_RESULTS['bpcer']:.4f}   (genuine drivers wrongly rejected)")
log(f"  ACER                  {PAD_RESULTS['acer']:.4f}")

# --- per-cue contribution --------------------------------------------------
log("\ncue firing rates (fraction of windows where the cue says 'live'):")
log(f"  {'group':<22}{'blink':>9}{'ocular':>10}{'deform':>9}{'votes>=2':>10}")
groups = [("bona fide (real)", bona_fide)] + list(attack_windows.items())
for name, ws in groups:
    if not ws:
        continue
    log(f"  {name:<22}{np.mean([w['blink_ok'] for w in ws]):>9.2%}"
        f"{np.mean([w['gaze_ok'] for w in ws]):>10.2%}"
        f"{np.mean([w['deform_ok'] for w in ws]):>9.2%}"
        f"{np.mean([w['is_live'] for w in ws]):>10.2%}")

log("\nunderlying cue magnitudes (median per window):")
log(f"  {'group':<22}{'blinks':>9}{'ocular sd deg':>15}{'deform resid':>14}")
for name, ws in groups:
    if not ws:
        continue
    log(f"  {name:<22}{np.median([w['n_blinks'] for w in ws]):>9.1f}"
        f"{np.median([w['gaze_variability_deg'] for w in ws]):>15.3f}"
        f"{np.median([w['deformation_residual'] for w in ws]):>14.6f}")

# --- operating-point trade-off (BPCER vs APCER), as R32 expects ------------
log("\noperating-point sweep — target BPCER per cue vs resulting worst-case APCER:")
log(f"  {'target BPCER':>13}{'actual BPCER':>14}{'APCER worst':>13}{'ACER':>8}")
for target in (0.01, 0.02, 0.05, 0.10, 0.20):
    calibrate_pad_thresholds(bona_cues, target)
    m = iso_30107_metrics(apply_pad_decisions(bona_cues),
                          {sp: apply_pad_decisions(ws) for sp, ws in attack_cues.items()})
    log(f"  {target:>13.0%}{m['bpcer']:>14.4f}{m['apcer_worst']:>13.4f}{m['acer']:>8.4f}")
calibrate_pad_thresholds(bona_cues)      # restore the reported operating point


# --- cue ablation: which physics is actually carrying the decision? --------
def decide_with(cue_subset, cues: dict) -> bool:
    """Liveness using only `cue_subset`; live if a majority of those cues fire."""
    fired = {
        "blink": cues["n_blinks"] >= PAD_MIN_BLINKS,
        "ocular": cues["gaze_variability_deg"] >= PAD_CALIBRATION["gaze_variability_deg"],
        "deform": cues["deformation_residual"] >= PAD_CALIBRATION["deformation_residual"],
    }
    votes = sum(fired[c] for c in cue_subset)
    return votes > len(cue_subset) / 2     # strict majority; for 2 cues, both


log("\nPAD cue ablation (same calibrated thresholds; subsets of the same three cues):")
log(f"  {'cue set':<26}{'BPCER':>8}{'static':>9}{'handheld':>10}{'screen':>9}{'ACER':>8}")
for subset in (("blink",), ("ocular",), ("deform",),
               ("blink", "ocular"), ("blink", "deform"), ("ocular", "deform"),
               ("blink", "ocular", "deform")):
    bp = float(np.mean([not decide_with(subset, w) for w in bona_cues]))
    ap = {sp: float(np.mean([decide_with(subset, w) for w in ws]))
          for sp, ws in attack_cues.items()}
    worst = max(ap.values())
    name = "+".join(subset) + (" (deployed)" if set(subset) == {"blink", "ocular"} else "")
    log(f"  {name:<26}{bp:>8.4f}{ap['print_static']:>9.4f}"
        f"{ap['print_handheld']:>10.4f}{ap['screen_replay']:>9.4f}{(worst + bp) / 2:>8.4f}")
log("  (a cue that fires equally on both classes adds nothing; one that fires MORE")
log("   on attacks than on genuine windows actively degrades the fusion)")
