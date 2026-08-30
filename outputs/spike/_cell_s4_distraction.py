"""4.6 — visual-distraction validation against DMD gaze_on_road labels (P3, DP-2a).

DMD annotates `gaze_on_road/{looking_road, not_looking_road}` on 100 % of frames
in all 82 distraction sessions (822,521 frames; 19.1 % not-looking-road). That is
DIRECT per-frame ground truth for exactly what the gaze/head stack predicts, and a
far stronger validation target than inferring attention from manual-action labels.
It also vindicates DP-2a: the claim was scoped to VISUAL distraction, and visual
distraction is precisely what DMD labels independently of the manual actions.

Two analyses:
  (a) off_road_mask() vs not_looking_road — the headline claim.
  (b) per manual-action class, how often gaze is genuinely off-road, and how much
      of that our detector recovers. This quantifies INDIRECT coverage: texting
      pulls the eyes down (detectable), a hands-free phone call may not (not
      detectable). Reporting this is how DP-2a's scope limit is made concrete
      rather than asserted.
"""

DISTRACTION_GAZE_LABELS = {
    "gaze_on_road/looking_road": "on_road",
    "gaze_on_road/not_looking_road": "off_road",
}
MANUAL_ACTION_PREFIX = "driver_actions/"
MANUAL_EXCLUDE = {"driver_actions/safe_drive", "driver_actions/unclassified"}


def load_distraction_annotation(path: Path) -> dict:
    """Parse a distraction OpenLABEL file into per-frame gaze + action timelines."""
    with path.open(encoding="utf-8") as fh:
        ol = json.load(fh)["openlabel"]
    n_frames = interval_frames(frame_intervals(ol))

    gaze = np.full(n_frames, "", dtype=object)
    actions = {}
    for act in ol.get("actions", {}).values():
        t = act["type"]
        ivs = frame_intervals(act)
        if t in DISTRACTION_GAZE_LABELS:
            for iv in ivs:
                lo, hi = max(iv["frame_start"], 0), min(iv["frame_end"] + 1, n_frames)
                gaze[lo:hi] = DISTRACTION_GAZE_LABELS[t]
        elif t.startswith(MANUAL_ACTION_PREFIX) and t not in MANUAL_EXCLUDE:
            mask = actions.setdefault(t, np.zeros(n_frames, dtype=bool))
            for iv in ivs:
                mask[max(iv["frame_start"], 0):min(iv["frame_end"] + 1, n_frames)] = True

    return {"path": path, "n_frames": n_frames, "gaze": gaze, "actions": actions}


def validate_distraction(videos) -> dict:
    """Score off_road_mask() against DMD gaze_on_road ground truth."""
    pooled_pred, pooled_truth = [], []
    per_session = []
    action_stats = defaultdict(lambda: {"frames": 0, "truth_off": 0, "pred_off": 0})

    for video in videos:
        if not _cache_path(video).exists():
            continue
        ann_path = next(video.parent.glob("*rgb_ann_distraction.json"), None)
        if ann_path is None:
            continue
        sig = extract_signals(video)
        ann = load_distraction_annotation(ann_path)

        idx = sig["frame_idx"]
        idx = idx[idx < ann["n_frames"]]
        if len(idx) == 0:
            continue
        labelled = ann["gaze"][idx] != ""
        if not labelled.any():
            continue

        truth = (ann["gaze"][idx] == "off_road")[labelled]
        pred = off_road_mask(sig)[:len(idx)][labelled]
        pooled_pred.append(pred)
        pooled_truth.append(truth)
        per_session.append({
            "session": "/".join(video.parts[-4:-1]),
            "frames": int(labelled.sum()),
            "off_road_truth": float(truth.mean()),
            "off_road_pred": float(pred.mean()),
            **binary_scores(pred, truth),
        })

        # (b) indirect coverage of manual-action classes
        for action, mask in ann["actions"].items():
            m = mask[idx][labelled]
            if not m.any():
                continue
            st = action_stats[action]
            st["frames"] += int(m.sum())
            st["truth_off"] += int(truth[m].sum())
            st["pred_off"] += int(pred[m].sum())

    if not per_session:
        return {"per_session": []}

    p, t = np.concatenate(pooled_pred), np.concatenate(pooled_truth)
    majority = np.full_like(t, t.mean() >= 0.5, dtype=bool)
    return {
        "per_session": per_session,
        "pooled": binary_scores(p, t),
        "majority_baseline": binary_scores(majority, t),
        "prevalence": float(t.mean()),
        "action_stats": {k: dict(v) for k, v in action_stats.items()},
    }


P3_DISTRACTION = validate_distraction(dmd_inventory["distraction"]["videos"])

if P3_DISTRACTION["per_session"]:
    res, maj = P3_DISTRACTION["pooled"], P3_DISTRACTION["majority_baseline"]
    n = res["tp"] + res["tn"] + res["fp"] + res["fn"]
    log("\n" + "=" * 78)
    log("P3 (distraction) — visual distraction vs DMD gaze_on_road ground truth")
    log("=" * 78)
    log(f"sessions {len(P3_DISTRACTION['per_session'])} | frames {n:,} | "
        f"not-looking-road prevalence {P3_DISTRACTION['prevalence']:.1%}")
    log(f"  accuracy {res['accuracy']:.4f} | precision {res['precision']:.4f} "
        f"| recall {res['recall']:.4f} | F1 {res['f1']:.4f}")
    log(f"  majority-class baseline (DR-4): accuracy {maj['accuracy']:.4f} F1 {maj['f1']:.4f}")

    log("\nindirect coverage of manual-action classes (DP-2a scope limit, made concrete):")
    log(f"  {'action':<38}{'frames':>10}{'gaze off-road':>15}{'we detect':>11}")
    for action, st in sorted(P3_DISTRACTION["action_stats"].items(),
                             key=lambda kv: -kv[1]["frames"]):
        if st["frames"] < 500:
            continue
        log(f"  {action:<38}{st['frames']:>10,}"
            f"{st['truth_off'] / st['frames']:>14.1%}"
            f"{st['pred_off'] / st['frames']:>11.1%}")
    log("  (a class with low 'gaze off-road' is one a gaze/head stack cannot see by design)")
else:
    log("P3 distraction validation skipped — distraction signal cache not yet populated")
