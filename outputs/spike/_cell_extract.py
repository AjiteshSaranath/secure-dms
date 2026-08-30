"""3.3 — extraction runner: video -> cached per-frame signal timeline.

Caches are content-addressed (video, mtime, extractor version, AND the extraction
window), so a re-run reuses them and `Run All` stays cheap, while any change to
the extractor or to how much of the video was read invalidates them automatically
(reproducibility rule, ledger R68).

The cache stores RAW model output. Validity masking (§3.4) is applied downstream
as a documented analysis step, so the cached record stays a faithful measurement
and the masking rule can be revised without re-decoding video.
"""
# p2.1 -> p2.2: EAR/MAR moved from normalised to pixel coordinates. On 1280x720
# video the normalised convention inflated both ratios by w/h = 1.78 (observed
# EAR mean 0.492, max 1.120 — physiologically impossible), which would have
# invalidated every published threshold.
# p2.2 -> p2.3: max_frames/frame_stride added to the cache key. Previously the
# 300-frame smoke test poisoned the cache for its session, and the full run then
# silently reused those 300 frames (gA/1/s5 appeared in P3 with 300 frames and a
# PERCLOS of 0.023 against a ground truth of 0.120).
EXTRACTOR_VERSION = "p2.3"
SIGNAL_CACHE = OUT_DIR / "signal_cache"
SIGNAL_CACHE.mkdir(exist_ok=True)


def _cache_path(video: Path, max_frames=None, frame_stride: int = 1) -> Path:
    key = (f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}"
           f"|{max_frames}|{frame_stride}")
    return SIGNAL_CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


def extract_signals(video: Path, max_frames: int | None = None,
                    frame_stride: int = 1, use_cache: bool = True) -> dict:
    """Single pass over a video -> per-frame EAR/MAR/gaze/pose/constellation arrays."""
    cache = _cache_path(video, max_frames, frame_stride)
    if use_cache and cache.exists():
        d = np.load(cache, allow_pickle=False)
        return {k: d[k] for k in d.files}

    landmarker = make_landmarker(video_mode=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.8

    frame_idx, rows, crops = [], [], []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and len(frame_idx) >= max_frames):
            break
        if i % frame_stride == 0:
            sig = perceive_frame(landmarker, bgr, int(i / fps * 1000))
            if sig is not None:
                x0, y0, x1, y1 = sig["box"]
                if x1 > x0 + 8 and y1 > y0 + 8:
                    frame_idx.append(i)
                    rows.append(sig)
                    crops.append(cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()

    if not rows:
        raise RuntimeError(f"no faces detected in {video.name}")

    g_pitch, g_yaw = gaze_angles(crops)
    h_pitch, h_yaw, h_roll = head_pose_angles(crops)

    out = {
        "frame_idx": np.array(frame_idx, dtype=np.int32),
        "fps": np.array([fps]),
        "total_frames_read": np.array([i]),
        "ear_left": np.array([r["ear_left"] for r in rows], dtype=np.float32),
        "ear_right": np.array([r["ear_right"] for r in rows], dtype=np.float32),
        "mar": np.array([r["mar"] for r in rows], dtype=np.float32),
        "iris": np.stack([r["iris"] for r in rows]).astype(np.float32),
        "gaze_pitch": g_pitch.astype(np.float32),
        "gaze_yaw": g_yaw.astype(np.float32),
        "head_pitch": h_pitch.astype(np.float32),
        "head_yaw": h_yaw.astype(np.float32),
        "head_roll": h_roll.astype(np.float32),
        "deformable": np.stack([r["deformable"] for r in rows]).astype(np.float32),
        "rigid": np.stack([r["rigid"] for r in rows]).astype(np.float32),
    }
    np.savez_compressed(cache, **out)
    return out


# --- 3.4 validity masking ---------------------------------------------------
# Both angle estimators can emit values that are not physically meaningful for a
# forward-facing cabin camera, for different reasons:
#
# GAZE (L2CS-Net): trained on Gaze360, whose label space spans the full 360 deg,
#   so |angle| > 90 deg is inside its output range. Diagnosed on 33,138 drowsiness
#   frames (p2_diagnose_gaze_outliers.py): 4 frames, 0.012 %, isolated 1-2 frame
#   spikes, occurring when the eyes are nearly shut AND the head is turned away
#   (mean EAR 0.116 vs 0.212; mean |head yaw| 38.0 vs 8.1 deg elsewhere).
#
# HEAD POSE (6DRepNet): the network predicts a CONTINUOUS 6D rotation — that is
#   precisely the contribution of R30 — but the conversion to Euler angles can
#   wrap, expressing a modest rotation as a near-180 deg flip. Diagnosed on the
#   first distraction session (p3_headpose_diagnostic.py): 1 frame in 6,635
#   (0.015 %), pitch +160.5 deg with roll -124.2 deg, i.e. the wraparound
#   signature (100 % of offending values beyond 150 deg). Head YAW — the primary
#   off-road signal — was entirely in range.
#
# Such frames are marked invalid rather than clipped, so they neither bias means
# nor vanish silently, and the rate is reported as a data-quality statistic.
ANGLE_PLAUSIBLE_LIMIT_DEG = 90.0


def gaze_validity_mask(sig: dict) -> np.ndarray:
    """True where the gaze estimate is physically meaningful for a cabin camera."""
    return ((np.abs(sig["gaze_pitch"]) <= ANGLE_PLAUSIBLE_LIMIT_DEG)
            & (np.abs(sig["gaze_yaw"]) <= ANGLE_PLAUSIBLE_LIMIT_DEG))


def head_validity_mask(sig: dict) -> np.ndarray:
    """True where the head-pose Euler conversion has not wrapped."""
    return ((np.abs(sig["head_pitch"]) <= ANGLE_PLAUSIBLE_LIMIT_DEG)
            & (np.abs(sig["head_yaw"]) <= ANGLE_PLAUSIBLE_LIMIT_DEG)
            & (np.abs(sig["head_roll"]) <= ANGLE_PLAUSIBLE_LIMIT_DEG))


# EAR is a ratio of VERTICAL eyelid separation to HORIZONTAL corner separation.
# Under head yaw the eye foreshortens horizontally roughly as cos(yaw), so the
# denominator shrinks and EAR inflates by ~1/cos(yaw): +41 % at 45 deg, +100 % at
# 60 deg. That is larger than the entire P80 closure margin, so a closed eye
# viewed in profile can read as open. This is the viewpoint dependence documented
# in R14; it is masked, not corrected, because correcting it would mean inventing
# a method rather than applying a published one.
#
# Measured across 803,044 distraction frames (p3_ear_viewpoint_diagnostic.py):
#   |yaw| band   0-10   10-20  20-30  30-40  40-50  50-60  60-90
#   mean EAR     0.268  0.255  0.245  0.275  0.321  0.383  0.480
#   % EAR>0.6    0.00   0.00   0.00   0.02   0.22   5.42   19.94
# Frames with EAR > 1.0 have mean |head yaw| 68.5 deg vs 13.6 deg elsewhere.
#
# The 45 deg limit is set by the geometry (cos 45 = 0.71, i.e. the point at which
# inflation reaches ~40 %), not fitted to results; it retains 95.2 % of frames.
EAR_MAX_HEAD_YAW_DEG = 45.0


def ear_validity_mask(sig: dict) -> np.ndarray:
    """True where EAR is geometrically trustworthy (near-frontal view)."""
    return head_validity_mask(sig) & (np.abs(sig["head_yaw"]) <= EAR_MAX_HEAD_YAW_DEG)


def assert_signals_plausible(sig: dict, label: str = "",
                             max_invalid_rate: float = 0.01) -> dict:
    """Guard against silent geometry errors (e.g. the p2.1 aspect-ratio bug).

    EAR bounds are physiological, not tuned: a fully open human eye has EAR well
    below 0.6 (R9). Angles are checked as RATES, because isolated degenerate
    frames are an expected estimator/representation limitation, whereas a high
    rate would indicate a real pipeline fault (e.g. a wrong crop convention).
    """
    # EAR is checked only on frames where it is geometrically meaningful; profile
    # views legitimately inflate it (see EAR_MAX_HEAD_YAW_DEG above), so applying
    # the physiological bound to raw EAR would flag valid data as corrupt.
    valid_ear = ear_validity_mask(sig)
    if valid_ear.any():
        ear_lr = np.stack([sig["ear_left"], sig["ear_right"]])[:, valid_ear]
        ear = float(np.nanmean(ear_lr))
        assert 0.10 < ear < 0.60, (
            f"{label}: implausible mean EAR {ear:.3f} on near-frontal frames "
            f"— check coordinate space")
        assert np.nanmax(ear_lr) < 1.0, (
            f"{label}: EAR exceeds 1.0 on a near-frontal frame — geometry error")

    rates = {"invalid_ear_rate": float(1 - valid_ear.mean())}
    for name, mask in (("gaze", gaze_validity_mask(sig)), ("head", head_validity_mask(sig))):
        rate = float(1 - mask.mean())
        rates[f"invalid_{name}_rate"] = rate
        assert rate <= max_invalid_rate, (
            f"{label}: {rate:.2%} of {name} estimates out of range "
            f"(limit {max_invalid_rate:.0%}) — investigate before trusting this session")
    rates["n_frames"] = len(sig["gaze_pitch"])
    return rates


# --- smoke test: a short prefix, cached under its own key (never reused for
#     full-session analysis — that confusion was the p2.2 -> p2.3 bug) --------
_demo_video = dmd_inventory["drowsiness"]["videos"][0]
_sig = extract_signals(_demo_video, max_frames=300)
_qc = assert_signals_plausible(_sig, _demo_video.name[:16])
_ear = (_sig["ear_left"] + _sig["ear_right"]) / 2
log(f"\nP2 smoke test — {_demo_video.name[:28]}… (first 300 frames)")
log(f"  frames with face: {len(_sig['frame_idx'])}/{int(_sig['total_frames_read'][0])} "
    f"({len(_sig['frame_idx'])/int(_sig['total_frames_read'][0]):.1%})")
log(f"  EAR   mean {np.nanmean(_ear):.3f} sd {np.nanstd(_ear):.3f} "
    f"min {np.nanmin(_ear):.3f} max {np.nanmax(_ear):.3f}  (open-eye literature ~0.2-0.35, R9)")
log(f"  MAR   mean {np.nanmean(_sig['mar']):.3f} sd {np.nanstd(_sig['mar']):.3f}")
log(f"  gaze  pitch {_sig['gaze_pitch'].mean():+.1f}+-{_sig['gaze_pitch'].std():.1f} "
    f"yaw {_sig['gaze_yaw'].mean():+.1f}+-{_sig['gaze_yaw'].std():.1f} deg")
log(f"  head  pitch {_sig['head_pitch'].mean():+.1f} yaw {_sig['head_yaw'].mean():+.1f} "
    f"roll {_sig['head_roll'].mean():+.1f} deg")
log(f"  quality guard: PASS (invalid gaze {_qc['invalid_gaze_rate']:.3%}, "
    f"invalid head pose {_qc['invalid_head_rate']:.3%})")
