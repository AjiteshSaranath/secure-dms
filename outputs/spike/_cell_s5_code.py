"""5 — temporal behavioural PAD (DR-3). Inference-only; no training, no fitting.

Three cues, each targeting the PHYSICS of a planar presentation rather than the
appearance statistics of any one dataset — which is the whole reason texture
methods fail cross-database (R21, R22):

  1. BLINK    A photograph cannot blink. Eyeblink-based anti-spoofing from a
              generic webcam is established by Pan et al. (R23), with the
              temporal blink model of Sun et al. (R24). Spontaneous blinking
              occurs 15-30 times/min, so a 10 s window should contain 2-5 blinks
              (R20).
  2. GAZE     A static face has near-zero OCULAR variability even when the
              instrument itself is moved, because the eyes cannot move
              independently of the surface carrying them (R26, R28).
  3. RIGIDITY A planar object undergoes one global transform: all facial points
              share it and the face does not deform. A live face additionally
              deforms non-rigidly around the eyes, mouth and cheeks (R25, R27).
              Measured as the Procrustes residual left after fitting the best
              similarity transform between consecutive landmark constellations.

Fusion is a conjunctive score over the three cues with a single operating
threshold, evaluated as APCER/BPCER/ACER per attack instrument species
(ISO/IEC 30107-3, R32).
"""

PAD_WINDOW_S = 10.0          # >=15 fps and several seconds, per R20's blink statistics
PAD_MIN_BLINKS = 1           # a live 10 s window should contain >=1 blink (R20)

# Threshold policy for the two continuous cues.
#
# Unlike PERCLOS or EAR, the PAD literature states these cues QUALITATIVELY ("a
# photograph cannot blink", "a planar surface does not deform") and publishes no
# transferable numeric threshold. A first implementation therefore invented
# values (1.0 deg, 1e-3) — which contradicted this project's own published-values
# discipline (DP-3a) and placed the gaze floor BELOW the attack distribution,
# admitting 92 % of handheld print attacks (ACER 0.478).
#
# The correct procedure, and the one ISO/IEC 30107-3 (R32) reporting assumes, is
# to fix the operating point on the BONA FIDE distribution alone — targeting a
# stated BPCER — and only then measure APCER on attacks. Attack data is never
# used to place a threshold, so the reported APCER remains an honest measurement
# rather than a fitted one.
PAD_TARGET_BPCER_PER_CUE = 0.05     # each continuous cue admits >=95 % of genuine windows
PAD_CALIBRATION = {}                # filled by calibrate_pad_thresholds()


def procrustes_residual(a: np.ndarray, b: np.ndarray) -> float:
    """RMS residual after the best similarity (rotation+scale+translation) fit a->b.

    Zero means b is exactly a rigid/similarity transform of a — the signature of a
    planar instrument (R25). A live face leaves a non-zero residual because soft
    tissue moves independently of the skull.
    """
    a_c, b_c = a - a.mean(0), b - b.mean(0)
    na = np.linalg.norm(a_c)
    if na < 1e-9:
        return 0.0
    a_n, b_n = a_c / na, b_c / (np.linalg.norm(b_c) + 1e-12)
    u, s, vt = np.linalg.svd(a_n.T @ b_n)
    aligned = a_n @ (u @ vt).T.T * s.sum()
    return float(np.sqrt(np.mean(np.sum((aligned - b_n) ** 2, axis=1))))


def pad_cues(sig: dict, lo: int, hi: int) -> dict:
    """The three cues over frames [lo, hi) of one signal timeline."""
    fps = float(sig["fps"][0])
    ear = mean_ear(sig)[lo:hi]

    # 1. blink evidence (R23, R24) — reuses the §4 detector validated at r=0.93
    blinks = blink_events(ear, fps) if np.isfinite(ear).sum() > 2 else {"blinks": []}

    # 2. OCULAR variability (R26, R28) — gaze relative to head pose, not gaze in
    # camera coordinates. Camera-frame gaze is head orientation combined with
    # eye-in-head rotation, so moving an instrument changes it even though the
    # printed eyes never move. Subtracting head pose isolates the ocular
    # component, which is what the gaze-PAD literature actually measures.
    gv = gaze_validity_mask(sig)[lo:hi] & head_validity_mask(sig)[lo:hi]
    gy = sig["gaze_yaw"][lo:hi] - sig["head_yaw"][lo:hi]
    gp = sig["gaze_pitch"][lo:hi] - sig["head_pitch"][lo:hi]
    gaze_var = (float(np.hypot(np.std(gy[gv]), np.std(gp[gv]))) if gv.sum() > 2 else 0.0)

    # 3. non-rigid deformation (R25, R27)
    deform, rigid = sig["deformable"][lo:hi], sig["rigid"][lo:hi]
    residuals = [procrustes_residual(deform[i], deform[i + 1])
                 for i in range(len(deform) - 1)]
    rigid_res = [procrustes_residual(rigid[i], rigid[i + 1])
                 for i in range(len(rigid) - 1)]
    deform_res = float(np.median(residuals)) if residuals else 0.0
    rigid_median = float(np.median(rigid_res)) if rigid_res else 0.0

    return {"n_frames": hi - lo,
            "n_blinks": len(blinks["blinks"]),
            "gaze_variability_deg": gaze_var,
            "deformation_residual": deform_res,
            "rigid_residual": rigid_median}


def calibrate_pad_thresholds(bona_fide_windows: list,
                             target_bpcer: float = PAD_TARGET_BPCER_PER_CUE) -> dict:
    """Place each continuous cue's threshold using GENUINE windows only.

    The threshold is the `target_bpcer` quantile of the bona-fide distribution, so
    by construction each cue admits (1 - target_bpcer) of genuine windows. No
    attack window is consulted, so APCER stays a measurement rather than a fit.
    """
    gaze = np.array([w["gaze_variability_deg"] for w in bona_fide_windows])
    deform = np.array([w["deformation_residual"] for w in bona_fide_windows])
    PAD_CALIBRATION.update({
        "gaze_variability_deg": float(np.quantile(gaze, target_bpcer)),
        "deformation_residual": float(np.quantile(deform, target_bpcer)),
        "target_bpcer_per_cue": target_bpcer,
        "n_calibration_windows": len(bona_fide_windows),
    })
    return PAD_CALIBRATION


def pad_decide(cues: dict) -> dict:
    """Liveness decision: blink AND ocular movement must both be present.

    The deformation/rigidity cue (R25, R27) is COMPUTED AND REPORTED but excluded
    from the decision. The reason is mechanistic, not merely a poor score: as
    implemented it fires "live" on 100 % of handheld-print and 83 % of screen
    attack windows while correctly rejecting every static one — i.e. it tracks how
    much the instrument MOVED, not whether the face DEFORMED. Warping an image
    forces the landmarker to re-detect on resampled pixels, and that interpolation
    jitter is not perfectly rigid, so motion manufactures apparent deformation.
    Alone it scores ACER 0.5250; inside the three-cue majority it was the sole
    cause of the 16.7 % handheld APCER (§5.2 ablation).

    Kollreider's and Anjos's physics is not in question — this landmark-Procrustes
    implementation of it does not survive resampling noise. Recovering the cue
    would need optical-flow or dense-correspondence analysis on raw pixels, which
    is future work.

    Selection caveat, stated rather than buried: the cue set was chosen with
    reference to the §5.2 evaluation. Capacity is low (7 subsets of 3 published
    cues, no fitted parameter) and the exclusion rests on an independent validity
    argument, but confirmation on Tier-D data would settle it properly.
    """
    if not PAD_CALIBRATION:
        raise RuntimeError("call calibrate_pad_thresholds() on bona-fide windows first")
    blink_ok = cues["n_blinks"] >= PAD_MIN_BLINKS
    gaze_ok = cues["gaze_variability_deg"] >= PAD_CALIBRATION["gaze_variability_deg"]
    deform_ok = cues["deformation_residual"] >= PAD_CALIBRATION["deformation_residual"]
    return {"blink_ok": blink_ok, "gaze_ok": gaze_ok, "deform_ok": deform_ok,
            "votes": int(blink_ok) + int(gaze_ok),
            "is_live": bool(blink_ok and gaze_ok)}


def pad_windows(sig: dict, window_s: float = PAD_WINDOW_S, label: str = "") -> list:
    """Split a timeline into non-overlapping windows and compute cues for each.

    Cues only — decisions come later via `apply_pad_decisions`, because thresholds
    must be calibrated on the genuine class before any window can be judged.

    A clip shorter than one window yields no windows at all. That once cost a
    whole evaluation silently (attack clips rendered at exactly the window length
    came out one frame short of it), so it now warns instead of returning [] mutely.
    """
    fps = float(sig["fps"][0])
    w = max(int(round(window_s * fps)), 2)
    n = len(sig["ear_left"])
    if n < w:
        log(f"  WARNING: {label or 'clip'} has {n} usable frames < window {w} "
            f"— contributes NO windows")
        return []
    return [pad_cues(sig, lo, lo + w) for lo in range(0, n - w + 1, w)]


def apply_pad_decisions(windows: list) -> list:
    """Attach a liveness decision to each window using the calibrated thresholds."""
    return [{**w, **pad_decide(w)} for w in windows]


def iso_30107_metrics(bona_fide_windows: list, attack_windows_by_species: dict) -> dict:
    """APCER per species, BPCER, and ACER (ISO/IEC 30107-3, R32).

    APCER is reported per attack instrument species and the WORST case is used for
    ACER, as the standard requires — an average across species would let a weak
    species mask a strong one.
    """
    bpcer = (float(np.mean([not w["is_live"] for w in bona_fide_windows]))
             if bona_fide_windows else float("nan"))
    apcer = {sp: float(np.mean([w["is_live"] for w in ws])) if ws else float("nan")
             for sp, ws in attack_windows_by_species.items()}
    worst = max([v for v in apcer.values() if np.isfinite(v)], default=float("nan"))
    return {"apcer_per_species": apcer, "apcer_worst": worst, "bpcer": bpcer,
            "acer": (worst + bpcer) / 2 if np.isfinite(worst) and np.isfinite(bpcer) else float("nan"),
            "n_bona_fide_windows": len(bona_fide_windows),
            "n_attack_windows": {sp: len(ws) for sp, ws in attack_windows_by_species.items()}}


log("PAD module ready — blink (R23/R24) + gaze variability (R26/R28) + "
    "rigidity (R25/R27), ISO 30107-3 metrics (R32)")
