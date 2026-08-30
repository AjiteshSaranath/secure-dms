"""5.3 — TIER D: evaluation on ROSE-Youtu (REAL attack instruments).

This is the evaluation P4 could not do. ROSE-Youtu supplies genuine AND attack
video from the SAME capture setup (20 subjects, 5 mobile devices), so it is
internally domain-controlled — the protocol P1_FINDINGS §4.1 requires — and it
contains the attack class this project predicted would defeat the deployed cues:
VIDEO REPLAY of a live face, which reproduces both blinking and eye movement.

Protocol differences from Tier C, each forced by the corpus and stated rather
than silently applied:

  * WINDOW = 5 s, not 10 s. ROSE clips run ~10-12 s and `replay_mac` averages
    9.8 s, so a 10 s window would yield ZERO windows for much of the hardest
    species — the failure that silently emptied the first Tier-C evaluation.
    5 s remains physiologically sound for the blink criterion: at ~15-20
    blinks/min (R77, corrected at P7-2 from an unsupportable "15-30/min") a 5 s
    window expects 1.25-1.67 blinks, so ">=1 blink" still holds.
  * RECALIBRATION on ROSE genuine. Thresholds are always set on the genuine
    class of the corpus under evaluation (never on attacks) — ROSE is 480p-720p
    mobile capture, a different sensor regime from DMD's 1280x720 cabin video.
  * NO-FACE CLIPS COUNT AS DETECTED ATTACKS. 29 replay clips yielded no
    detectable face at all. Dropping them would silently remove the hardest
    attacks from APCER and flatter the result; operationally a presentation in
    which no face can be found cannot be confirmed live, so it is a rejection.
"""
from collections import Counter

ROSE_DIR = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\datasets\ROSE-Youtu Extracted Dataset")
ROSE_CODES = {"G": "genuine", "Ps": "print_still", "Pq": "print_quiver",
              "Vl": "replay_lenovo", "Vm": "replay_mac",
              "Mc": "mask_cropped", "Mf": "mask_full", "Mu": "mask_upper"}
TIER_D_WINDOW_S = 5.0
_rose_manifest = json.loads((OUT_DIR / "p4d_rose_manifest.json").read_text(encoding="utf-8"))
_selected = {(i["species"], i["file"]) for i in _rose_manifest["items"]}

# ── gather cues per species (only clips in the seeded sample) ──────────────
rose_cues, rose_clips, rose_noface = defaultdict(list), Counter(), Counter()
for _v in sorted(ROSE_DIR.rglob("*.mp4")):
    _sp = ROSE_CODES.get(_v.stem.split("_")[0])
    if _sp is None or (_sp, _v.name) not in _selected:
        continue
    rose_clips[_sp] += 1
    if not _cache_path(_v).exists():
        # extraction found no face anywhere in the clip -> cannot be confirmed live
        rose_noface[_sp] += 1
        continue
    rose_cues[_sp].extend(pad_windows(extract_signals(_v), window_s=TIER_D_WINDOW_S,
                                      label=_v.name))

# ── calibrate on ROSE GENUINE only, then judge ─────────────────────────────
_prev_cal = dict(PAD_CALIBRATION)
_cal_d = calibrate_pad_thresholds(rose_cues["genuine"])
_bona = apply_pad_decisions(rose_cues["genuine"])
_atk = {sp: apply_pad_decisions(ws) for sp, ws in rose_cues.items() if sp != "genuine"}

log("\n" + "=" * 82)
log("P4-D — TIER D: ROSE-Youtu, REAL attack instruments (20 subjects, 5 devices)")
log("=" * 82)
log(f"window {TIER_D_WINDOW_S:.0f} s | thresholds calibrated on {len(_bona):,} GENUINE "
    f"ROSE windows only: ocular >= {_cal_d['gaze_variability_deg']:.3f} deg, "
    f"deform >= {_cal_d['deformation_residual']:.6f}")

# APCER per species, counting no-face clips as correctly rejected
_apcer_d, _rows = {}, []
for _sp, _ws in sorted(_atk.items()):
    _accepted = sum(w["is_live"] for w in _ws)
    _nf = rose_noface[_sp]
    _apcer = _accepted / max(len(_ws) + _nf, 1)      # no-face windows count as rejected
    _apcer_d[_sp] = _apcer
    _rows.append((_sp, rose_clips[_sp], _nf, len(_ws), _accepted, _apcer))

_bpcer_d = float(np.mean([not w["is_live"] for w in _bona]))
_worst_d = max(_apcer_d.values())
_acer_d = (_worst_d + _bpcer_d) / 2

log(f"\n{'species':<16}{'clips':>6}{'no-face':>9}{'windows':>9}{'accepted':>10}{'APCER':>9}")
for _sp, _nc, _nf, _nw, _acc, _ap in _rows:
    log(f"{_sp:<16}{_nc:>6}{_nf:>9}{_nw:>9}{_acc:>10}{_ap:>9.4f}")
log(f"\nBPCER (genuine windows wrongly rejected): {_bpcer_d:.4f}")
log(f"APCER worst-case: {_worst_d:.4f}  ->  ACER {_acer_d:.4f}")

# ── cue-level breakdown: WHICH physics survives real instruments? ──────────
log("\ncue firing rates ('live' votes) — the decisive comparison:")
log(f"  {'group':<16}{'blink':>9}{'ocular':>9}{'deform':>9}{'accepted':>10}")
for _name, _ws in [("genuine", _bona)] + sorted(_atk.items()):
    if not _ws:
        continue
    log(f"  {_name:<16}{np.mean([w['blink_ok'] for w in _ws]):>9.1%}"
        f"{np.mean([w['gaze_ok'] for w in _ws]):>9.1%}"
        f"{np.mean([w['deform_ok'] for w in _ws]):>9.1%}"
        f"{np.mean([w['is_live'] for w in _ws]):>10.1%}")

# ── the deformation cue on REAL attacks (resolves P4_FINDINGS §4.1) ────────
def _auc(pos, neg):
    a, b = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([a, b])
    r = allv.argsort().argsort().astype(float) + 1
    return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))

_g_def = [w["deformation_residual"] for w in _bona]
_a_def = [w["deformation_residual"] for ws in _atk.values() for w in ws]
_g_oc = [w["gaze_variability_deg"] for w in _bona]
_a_oc = [w["gaze_variability_deg"] for ws in _atk.values() for w in ws]
_g_bl = [w["n_blinks"] for w in _bona]
_a_bl = [w["n_blinks"] for ws in _atk.values() for w in ws]
TIER_D_CUE_AUC = {"blink": _auc(_g_bl, _a_bl), "ocular": _auc(_g_oc, _a_oc),
                  "deformation": _auc(_g_def, _a_def)}
log("\nsingle-cue separability on REAL attacks (AUC, genuine=positive):")
for _k, _v in TIER_D_CUE_AUC.items():
    log(f"  {_k:<14}{_v:.4f}")
log("  (P4_FINDINGS §4.1 left the deformation cue's exclusion PROVISIONAL because")
log("   its failure on Tier C could have been an artefact of synthetic warping;")
log("   this is the measurement that settles it.)")

# ── cue ablation on REAL attacks (same discipline as Tier C) ───────────────
def _decide_subset(subset, c):
    fired = {"blink": c["n_blinks"] >= PAD_MIN_BLINKS,
             "ocular": c["gaze_variability_deg"] >= PAD_CALIBRATION["gaze_variability_deg"],
             "deform": c["deformation_residual"] >= PAD_CALIBRATION["deformation_residual"]}
    return sum(fired[k] for k in subset) > len(subset) / 2


log("\nTier-D cue ablation (same calibrated thresholds; APCER worst-case shown):")
log(f"  {'cue set':<24}{'BPCER':>8}{'APCER worst':>13}{'ACER':>8}   worst species")
TIER_D_ABLATION = {}
for _subset in (("blink",), ("ocular",), ("deform",), ("blink", "ocular"),
                ("blink", "ocular", "deform")):
    _bp = float(np.mean([not _decide_subset(_subset, w) for w in rose_cues["genuine"]]))
    _aps = {}
    for _sp, _ws in rose_cues.items():
        if _sp == "genuine":
            continue
        _acc = sum(_decide_subset(_subset, w) for w in _ws)
        _aps[_sp] = _acc / max(len(_ws) + rose_noface[_sp], 1)
    _wsp = max(_aps, key=_aps.get)
    _name = "+".join(_subset) + (" (deployed)" if set(_subset) == {"blink", "ocular"} else "")
    TIER_D_ABLATION[_name] = {"bpcer": _bp, "apcer_worst": _aps[_wsp],
                              "acer": (_aps[_wsp] + _bp) / 2, "worst_species": _wsp}
    log(f"  {_name:<24}{_bp:>8.4f}{_aps[_wsp]:>13.4f}{(_aps[_wsp]+_bp)/2:>8.4f}   {_wsp}")

TIER_D_RESULTS = {"window_s": TIER_D_WINDOW_S, "calibration": _cal_d,
                  "ablation": TIER_D_ABLATION,
                  "apcer_per_species": _apcer_d, "apcer_worst": _worst_d,
                  "bpcer": _bpcer_d, "acer": _acer_d,
                  "clips": dict(rose_clips), "no_face_clips": dict(rose_noface),
                  "n_genuine_windows": len(_bona),
                  "n_attack_windows": {k: len(v) for k, v in _atk.items()},
                  "cue_auc": TIER_D_CUE_AUC}
(OUT_DIR / "p4d_tierd_results.json").write_text(json.dumps(TIER_D_RESULTS, indent=2),
                                                encoding="utf-8")
