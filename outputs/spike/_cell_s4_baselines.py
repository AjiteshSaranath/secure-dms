"""4.7 — DR-4 baselines, single-cue ablations, and verdict-level metrics (P3).

Three questions, each needing a different comparison:

(a) BASELINES. Does the proposed P80 criterion beat the published simple method?
    Per DR-4 the baselines are published methods re-implemented from source, and
    a baseline win is an acceptable, reportable outcome. The EAR-threshold method
    is reported in BOTH variants (§4.5): as literally published (absolute 0.21),
    and ratio-adapted, because §3 showed the absolute value does not transfer to
    MediaPipe landmarks. Reporting only one would be unfair in one direction or
    the other.

(b) ABLATIONS. Which cue actually carries the drowsiness signal? Reporting the
    fused rule without this would assert that fusion helps rather than show it.

(c) VERDICT. Until now every drowsiness session returned "unsafe", so the module
    had no demonstrated discrimination. The distraction corpus supplies a control
    group — but that it is genuinely non-drowsy is an ASSUMPTION, and it is
    tested here rather than assumed.
"""

# ---------------------------------------------------------------- (a) baselines
def compare_baselines(videos, annotations) -> dict:
    """Frame-level eye-closure: proposed P80 vs published EAR baselines vs majority."""
    preds = {"proposed (P80, R15/R16)": [], "baseline EAR absolute (R9 literal)": [],
             "baseline EAR ratio-adapted": []}
    truths = []

    for video, ann in zip(videos, annotations):
        if not _cache_path(video).exists():
            continue
        sig = extract_signals(video)
        truth = align_annotation(sig, ann)
        if truth is None:
            continue
        ear = mean_ear(sig)[:len(truth)]
        truths.append(truth)
        preds["proposed (P80, R15/R16)"].append(eye_closed_mask(ear))
        preds["baseline EAR absolute (R9 literal)"].append(
            baseline_ear_consecutive(ear, relative=False))
        preds["baseline EAR ratio-adapted"].append(
            baseline_ear_consecutive(ear, relative=True))

    if not truths:
        return {}
    t = np.concatenate(truths)
    out = {name: binary_scores(np.concatenate(p), t) for name, p in preds.items()}
    out["majority class"] = binary_scores(np.full_like(t, t.mean() >= 0.5, dtype=bool), t)
    return out


# ---------------------------------------------------------------- (b) ablations
def drowsy_by_cue(sig: dict) -> dict:
    """Each cue's independent drowsiness verdict, plus the fused rule."""
    fps = float(sig["fps"][0])
    ear = mean_ear(sig)
    blinks = blink_events(ear, fps)
    perclos_hit = bool(np.nanmax(perclos_series(ear, fps)) > PERCLOS_DROWSY)
    micro_hit = len(blinks["microsleeps"]) > 0
    yawn_hit = len(yawn_events(sig["mar"], fps)) > 0
    # Abnormal blink rate: outside the ~15-20/min band typical for alert adults
    # (R77; P7-2 corrected this from "15-30/min, R20", which was unsupportable and
    # misattributed). Note the literature also reports very large inter-individual
    # variance (13.8 +- 9.7, range 2.8-48), which is itself a reason to expect this
    # cue to discriminate poorly — as the ablation below confirms (+3.5 % only).
    rate_hit = not (15.0 <= blinks["blink_rate_per_min"] <= 20.0)
    return {"PERCLOS only": perclos_hit, "microsleep only": micro_hit,
            "yawn only": yawn_hit, "blink-rate only": rate_hit,
            "PERCLOS OR microsleep": perclos_hit or micro_hit,
            "PERCLOS OR yawn (deployed)": perclos_hit or yawn_hit}


# ------------------------------------------------- (c) verdict + control group
def session_measures(videos, limit=None) -> list:
    rows = []
    for video in videos[:limit] if limit else videos:
        if not _cache_path(video).exists():
            continue
        sig = extract_signals(video)
        rows.append({"session": "/".join(video.parts[-4:-1]),
                     "state": driver_state(sig), "cues": drowsy_by_cue(sig)})
    return rows


DROWSY_SESSIONS = session_measures(dmd_inventory["drowsiness"]["videos"])
CONTROL_SESSIONS = session_measures(dmd_inventory["distraction"]["videos"])

log("\n" + "=" * 78)
log("P3 (a) — DR-4 baselines: frame-level eye closure")
log("=" * 78)
BASELINES = compare_baselines(dmd_inventory["drowsiness"]["videos"], DMD_ANNOTATIONS)
log(f"{'method':<38}{'acc':>8}{'prec':>8}{'recall':>8}{'F1':>8}")
for name, s in BASELINES.items():
    log(f"{name:<38}{s['accuracy']:>8.4f}{s['precision']:>8.4f}"
        f"{s['recall']:>8.4f}{s['f1']:>8.4f}")

log("\n" + "=" * 78)
log("P3 (c) — is the distraction corpus a valid non-drowsy control?")
log("=" * 78)
d_perclos = np.array([r["state"]["closed_fraction"] for r in DROWSY_SESSIONS])
c_perclos = np.array([r["state"]["closed_fraction"] for r in CONTROL_SESSIONS])
pooled_sd = np.sqrt((d_perclos.var() + c_perclos.var()) / 2)
cohens_d = float((d_perclos.mean() - c_perclos.mean()) / pooled_sd) if pooled_sd > 1e-9 else 0.0
log(f"  drowsiness sessions  n={len(d_perclos):>3}  measured PERCLOS "
    f"{d_perclos.mean():.4f} +- {d_perclos.std():.4f}")
log(f"  distraction sessions n={len(c_perclos):>3}  measured PERCLOS "
    f"{c_perclos.mean():.4f} +- {c_perclos.std():.4f}")
log(f"  separation: Cohen's d {cohens_d:+.3f}")
log("  (a control group is only valid if it is measurably less drowsy; a d near")
log("   zero would mean the 'contrast' is illusory and verdict metrics meaningless)")

log("\n" + "=" * 78)
log("P3 (b,c) — cue ablation and verdict discrimination")
log("=" * 78)
log(f"{'cue':<26}{'drowsy flagged':>16}{'control flagged':>17}{'separation':>12}")
for cue in ("PERCLOS only", "microsleep only", "yawn only", "blink-rate only",
            "PERCLOS OR microsleep", "PERCLOS OR yawn (deployed)"):
    d_hit = np.mean([r["cues"][cue] for r in DROWSY_SESSIONS])
    c_hit = np.mean([r["cues"][cue] for r in CONTROL_SESSIONS])
    log(f"{cue:<26}{d_hit:>15.1%}{c_hit:>17.1%}{d_hit - c_hit:>+12.1%}")
log("  'separation' = true-positive rate minus false-positive rate at session level;")
log("  a cue firing equally on both groups contributes nothing regardless of its rate.")

P3_BASELINES = {"frame_level": BASELINES,
                "control_cohens_d": cohens_d,
                "drowsy_perclos_mean": float(d_perclos.mean()),
                "control_perclos_mean": float(c_perclos.mean())}
