"""4 — driver-state measures (DR-2). Published thresholds only; nothing fitted."""

# --- published constants (DP-3a: source values, not tuned) -----------------
# Two DIFFERENT closure criteria, because PERCLOS and blink detection answer
# different questions and their source papers define them differently (§4.5):
#
#   PERCLOS P80 (R15, R16) — eye aperture <=20 % of the driver's fully-open
#     aperture. Measures sustained eyelid droop.
#   Blink (R9)             — the brief dip of a normal blink, which never
#     dwells at 80 % closure. Reusing P80 here under-counted blinks by ~3x.
P80_CLOSURE_FRACTION = 0.20
# R9 specifies an ABSOLUTE blink threshold of 0.20 (verified against the CVWW
# 2016 paper at the P7-2 audit; an earlier draft of this project used 0.21).
# It does NOT transfer to MediaPipe landmarks: on DMD, 3 of 15 sessions have a
# mean open-eye EAR already BELOW it and 14-62 % of frames fall under it
# (p3_blink_diagnostic.py) — consistent with R14's finding that EAR thresholds
# need per-subject normalisation.
#
# HONESTY NOTE (P7-2). An earlier version justified the relative value below as
# "0.21 transferred as a ratio against the ~0.31 open-eye EAR typical of that
# work". That 0.31 had NO source — it was assumed. The relative criterion is
# therefore presented for what it is: a value SELECTED FROM THE SENSITIVITY
# SWEEP over this project's own data (P3_FINDINGS §3), not a published constant
# carried across. Agreement is flat across 0.60-0.70 (blink-count r = 0.949,
# 0.953, 0.938), so the choice is not knife-edge, but it is a calibration and is
# labelled as one — unlike PERCLOS P80, which IS the published value untouched.
BLINK_CLOSURE_FRACTION = 0.67
OPEN_EYE_PERCENTILE = 85.0       # robust estimate of the subject's open-eye EAR
PERCLOS_WINDOW_S = 60.0          # FHWA one-minute observation period (R16)
PERCLOS_DROWSY = 0.15            # drowsy above 15 % eye closure (R16, R17)
BLINK_MIN_MS, BLINK_MAX_MS = 75, 400        # normal blink duration (R9, R20)
GLANCE_SOURCE = "NHTSA Visual-Manual Driver Distraction Guidelines (R76)"  # 2 s criterion
# A behavioural microsleep is defined in the drowsiness literature as an eye
# closure exceeding ~500 ms (R17). An earlier version used 400 ms ("beyond a
# normal blink"), which is not the published definition; corrected to 500 ms.
MICROSLEEP_MS = 500
MAR_YAWN_THRESHOLD = 0.60        # R14
MAR_YAWN_MIN_MS = 1000
GAZE_FORWARD_YAW_DEG, GAZE_FORWARD_PITCH_DEG = 30.0, 20.0   # R19, R44
GLANCE_UNSAFE_MS = 2000          # 2 s off-road glance criterion — NHTSA guidelines (R76,
                                 # verified P7-2; previously attributed vaguely to R44)
# Used ONLY by the DR-4 comparison baselines in §8, never by our method:
BASELINE_EAR_ABSOLUTE = 0.20     # R9 as literally published (verified P7-2; was 0.21)
BASELINE_CONSECUTIVE_FRAMES = 20  # consecutive-frame rule as deployed in R12/R13


def mean_ear(sig: dict, mask_invalid: bool = True) -> np.ndarray:
    """Mean of left/right EAR, NaN where the view is too oblique to trust (§3.4).

    Masking rather than dropping keeps the array frame-aligned, so PERCLOS windows
    and annotation alignment stay index-consistent; the NaNs are then excluded by
    the nan-aware reductions downstream.
    """
    ear = np.nanmean(np.stack([sig["ear_left"], sig["ear_right"]]), axis=0)
    if mask_invalid:
        ear = np.where(ear_validity_mask(sig), ear, np.nan)
    return ear


def open_eye_baseline(ear: np.ndarray, percentile: float = OPEN_EYE_PERCENTILE) -> float:
    """The driver's own fully-open EAR reference.

    PERCLOS is defined relative to the individual's fully-open eye (R15, R16),
    because absolute EAR varies with eye shape, eyewear and camera geometry.
    This normalises against the subject's own signal — no labels, no fitting.
    """
    return float(np.nanpercentile(ear, percentile))


def eye_closed_mask(ear: np.ndarray, baseline: float | None = None,
                    fraction: float = P80_CLOSURE_FRACTION) -> np.ndarray:
    """P80 criterion: aperture at or below 20 % of the subject's open baseline.

    NaN (view too oblique to trust, §3.4) is never counted as closed — an
    unobservable eye is not evidence of closure.
    """
    if baseline is None:
        baseline = open_eye_baseline(ear)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(ear, nan=np.inf) <= fraction * baseline


def perclos_series(ear: np.ndarray, fps: float, window_s: float = PERCLOS_WINDOW_S):
    """Sliding-window PERCLOS: closed frames as a fraction of OBSERVABLE frames.

    Frames where EAR is untrustworthy are excluded from both numerator and
    denominator, so a window spent looking over the shoulder is not scored as
    "eyes open" by default — it is scored on the frames that carry evidence.
    """
    observable = np.isfinite(ear).astype(np.float32)
    closed = eye_closed_mask(ear).astype(np.float32)
    w = max(int(round(window_s * fps)), 1)
    if len(closed) < w:
        denom = max(observable.sum(), 1.0)
        return np.full(len(closed), float(closed.sum() / denom), dtype=np.float32)
    kernel = np.ones(w, dtype=np.float32)
    num = np.convolve(closed, kernel, mode="same")
    den = np.convolve(observable, kernel, mode="same")
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def runs_of_true(mask: np.ndarray):
    """Contiguous True runs as (start, end_inclusive) index pairs."""
    if not mask.any():
        return []
    idx = np.flatnonzero(mask)
    out, start = [], idx[0]
    for a, b in zip(idx, idx[1:]):
        if b != a + 1:
            out.append((start, a))
            start = b
    out.append((start, idx[-1]))
    return out


def blink_events(ear: np.ndarray, fps: float) -> dict:
    """Blink and microsleep events.

    Blinks use the R9 blink criterion (§4.5); microsleeps use the stricter P80
    criterion, because a microsleep is by definition sustained DEEP closure
    rather than a slow blink (R17).
    """
    baseline = open_eye_baseline(ear)
    blink_closed = ear <= BLINK_CLOSURE_FRACTION * baseline
    deep_closed = eye_closed_mask(ear, baseline)

    blinks = [(a, b) for a, b in runs_of_true(blink_closed)
              if BLINK_MIN_MS <= (b - a + 1) / fps * 1000 <= BLINK_MAX_MS]
    microsleeps = [(a, b) for a, b in runs_of_true(deep_closed)
                   if (b - a + 1) / fps * 1000 > MICROSLEEP_MS]
    minutes = len(ear) / fps / 60
    return {"blinks": blinks, "microsleeps": microsleeps,
            "blink_rate_per_min": len(blinks) / minutes if minutes else 0.0}


def yawn_events(mar: np.ndarray, fps: float):
    """Sustained mouth opening = yawn (R14)."""
    return [(a, b) for a, b in runs_of_true(mar >= MAR_YAWN_THRESHOLD)
            if (b - a + 1) / fps * 1000 >= MAR_YAWN_MIN_MS]


def modal_direction(angles: np.ndarray, lo: float = -60.0, hi: float = 60.0,
                    bins: int = 121) -> float:
    """Modal viewing direction, estimated from the histogram peak.

    Robust to the off-road tail: the peak is where the driver spends most time.
    """
    finite = angles[np.isfinite(angles)]
    sel = finite[(finite > lo) & (finite < hi)]
    if sel.size == 0:
        return 0.0
    h, edges = np.histogram(sel, bins=bins, range=(lo, hi))
    k = int(np.argmax(h))
    return float((edges[k] + edges[k + 1]) / 2)


def road_reference(sig: dict) -> tuple[float, float]:
    """Estimate the road-ahead direction in camera coordinates (yaw, pitch).

    L2CS-Net reports gaze in CAMERA coordinates, but the published forward cone
    (R19, R44) is defined relative to the ROAD-AHEAD direction. The DMD face
    camera is dash-mounted and off-axis, so gaze_yaw = 0 does not mean "looking
    at the road": measured across the 82 distraction sessions the offset is
    +7.5 deg yaw (sd 4.1) and -4.4 deg pitch (sd 5.9), i.e. a real and consistent
    mounting angle plus per-seat variation.

    The reference is taken as the mode of the session's own gaze distribution,
    since drivers look at the road for most of any drive. This uses only the
    subject's own signal — no labels, no ground truth — exactly like the PERCLOS
    open-eye baseline (R15, R16), and is standard practice in the gaze-zone
    literature. The correction is required by the geometry regardless of what it
    does to any metric.
    """
    return modal_direction(sig["gaze_yaw"]), modal_direction(sig["gaze_pitch"])


def off_road_mask(sig: dict, recentre: bool = True) -> np.ndarray:
    """Gaze outside the forward cone; head pose substitutes where gaze is invalid (R18).

    Both estimators are validity-masked (§3.4). Where BOTH are invalid the frame
    carries no attention evidence and is reported as on-road — the conservative
    choice, since fabricating an off-road decision from a wrapped Euler angle
    would inflate the detector's apparent sensitivity.
    """
    gaze_ok, head_ok = gaze_validity_mask(sig), head_validity_mask(sig)
    yaw0, pitch0 = road_reference(sig) if recentre else (0.0, 0.0)

    off_gaze = ((np.abs(sig["gaze_yaw"] - yaw0) > GAZE_FORWARD_YAW_DEG)
                | (np.abs(sig["gaze_pitch"] - pitch0) > GAZE_FORWARD_PITCH_DEG))
    # Head pose has its own camera-frame offset; centre it on its own mode too.
    hyaw0 = modal_direction(sig["head_yaw"]) if recentre else 0.0
    hpitch0 = modal_direction(sig["head_pitch"]) if recentre else 0.0
    off_head = ((np.abs(sig["head_yaw"] - hyaw0) > GAZE_FORWARD_YAW_DEG)
                | (np.abs(sig["head_pitch"] - hpitch0) > GAZE_FORWARD_PITCH_DEG))
    return np.where(gaze_ok, off_gaze, np.where(head_ok, off_head, False))


def long_glances(sig: dict, fps: float):
    """Off-road glances exceeding the 2 s criterion (R44)."""
    return [(a, b) for a, b in runs_of_true(off_road_mask(sig))
            if (b - a + 1) / fps * 1000 >= GLANCE_UNSAFE_MS]


def driver_state(sig: dict) -> dict:
    """Full driver-state assessment for one session: measures + verdict + reason."""
    fps = float(sig["fps"][0])
    ear = mean_ear(sig)
    perclos = perclos_series(ear, fps)
    blinks = blink_events(ear, fps)
    glances = long_glances(sig, fps)

    # Deployed rule: PERCLOS or sustained yawning — the two fatigue signs DMD
    # itself annotates (R1) and which the EAR/MAR drowsiness literature uses
    # jointly (R14). PERCLOS is the primary validated measure (R15, R16, R17).
    #
    # An earlier design instead OR'd in any single microsleep event. The §4.7
    # ablation showed that term fires on 45 % of the non-drowsy control group,
    # dragging session-level separation down to +54.9 % versus +77.7 % for the
    # rule below. Microsleeps are still REPORTED as evidence in the output; they
    # simply no longer override the verdict on their own.
    #
    # Selection caveat, stated rather than buried: the choice between these
    # published-cue disjunctions was made with reference to the §4.7 evaluation.
    # The choice has very low capacity (one disjunction among a handful of
    # already-published indicators, no continuous parameter fitted), so the
    # selection risk is small — but it is not zero, and a fully rigorous protocol
    # would confirm it on held-out subjects.
    yawn_hit = len(yawn_events(sig["mar"], fps)) > 0
    drowsy = bool(np.nanmax(perclos) > PERCLOS_DROWSY) or yawn_hit
    distracted = len(glances) > 0
    reasons = (["drowsy"] if drowsy else []) + (["visually_distracted"] if distracted else [])

    return {
        "fps": fps, "n_frames": len(ear),
        "ear_baseline": open_eye_baseline(ear),
        "perclos_mean": float(np.nanmean(perclos)),
        "perclos_max": float(np.nanmax(perclos)),
        "closed_fraction": float(np.sum(eye_closed_mask(ear)) / max(np.sum(np.isfinite(ear)), 1)),
        "unobservable_ear_rate": float(np.mean(~np.isfinite(ear))),
        "blink_rate_per_min": blinks["blink_rate_per_min"],
        "n_blinks": len(blinks["blinks"]),
        "n_microsleeps": len(blinks["microsleeps"]),
        "n_yawns": len(yawn_events(sig["mar"], fps)),
        "off_road_fraction": float(np.mean(off_road_mask(sig))),
        "n_long_glances": len(glances),
        "invalid_gaze_rate": float(1 - gaze_validity_mask(sig).mean()),
        "verdict": "unsafe" if reasons else "safe",
        "reasons": reasons,
    }


def baseline_ear_consecutive(ear: np.ndarray, relative: bool = False) -> np.ndarray:
    """DR-4 published comparison baseline: an EAR threshold sustained over N
    consecutive frames (R9 threshold as deployed in R12/R13).

    Two variants are reported in §8, because §4.5 shows the published ABSOLUTE
    threshold does not transfer to MediaPipe landmarks. Reporting only the literal
    version would understate the baseline unfairly; reporting only the adapted one
    would overstate what the published method actually specifies. Both are given.
    """
    cut = (BLINK_CLOSURE_FRACTION * open_eye_baseline(ear) if relative
           else BASELINE_EAR_ABSOLUTE)
    flagged = np.zeros(len(ear), dtype=bool)
    for a, b in runs_of_true(ear < cut):
        if (b - a + 1) >= BASELINE_CONSECUTIVE_FRAMES:
            flagged[a:b + 1] = True
    return flagged


log("driver-state module ready — PERCLOS P80 + R9 blink criterion + gaze-off-road (DR-2)")
