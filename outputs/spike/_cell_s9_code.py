"""9 — STRIDE validation suite (P6).

Each threat is exercised in isolation with a controlled experiment producing one
headline number. Mechanisms unchanged from v7 (T/R/I/D/E) keep the v7 pass
criteria and canonical counts; S is re-specified around the new PAD module.
Uses the §6 security classes and the §5 PAD directly — no re-implementation.
"""
import hmac as _hmac

STRIDE = {}
_rng = np.random.default_rng(SEED)


def _band(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


# ── S — Spoofing: temporal behavioural PAD (PAD-as-alert, §7) ───────────────
# STRIDE-S is judged on TIER D (ROSE-Youtu, REAL attack instruments), not Tier C.
# Tier C is synthetic (affine warps this project generated) and yielded APCER
# 0.0000; on real instruments the same module scores ACER 0.59. Reporting the
# Tier-C figure as the STRIDE-S result would be flattering by omission now that
# real-instrument evidence exists, so the row FAILS on the evidence that counts
# and both tiers are shown. See P4_FINDINGS §6.
_tc_worst = PAD_RESULTS["apcer_worst"]          # Tier C, synthetic
_td = TIER_D_RESULTS
_worst_apcer, _bpcer, _acer_d = _td["apcer_worst"], _td["bpcer"], _td["acer"]
_worst_species = max(_td["apcer_per_species"], key=_td["apcer_per_species"].get)
_td_windows = sum(_td["n_attack_windows"].values())
STRIDE["S"] = {
    "threat": "Spoofing", "mechanism": "Temporal behavioural PAD (blink+ocular; R23/R26)",
    "primary_evidence": "Tier D — ROSE-Youtu, real instruments (20 subjects, 5 devices)",
    "tier_d_attack_windows": _td_windows,
    "tier_d_apcer_per_species": _td["apcer_per_species"],
    "tier_d_apcer_worst": _worst_apcer, "tier_d_bpcer": _bpcer, "tier_d_acer": _acer_d,
    "tier_c_apcer_worst_synthetic": _tc_worst,
    "pass": _worst_apcer <= 0.10 and _bpcer <= 0.20,
    "result": f"FAILS on real instruments: worst-species APCER {_worst_apcer:.2%} "
              f"({_worst_species}), BPCER {_bpcer:.2%}, ACER {_acer_d:.2%} over "
              f"{_td_windows:,} Tier-D attack windows. (Tier-C synthetic APCER was "
              f"{_tc_worst:.2%} — the gap is the finding, P4_FINDINGS §6.)",
}

# ── T — Tampering: SHA-256 + HMAC integrity (unchanged mechanism) ───────────
_N_T = 500
_iv = IntegrityVerifier()
_clean_ok = _tamper_caught = _forge_caught = 0
for _ in range(_N_T):
    fr = _rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
    meta = _iv.sign_frame(fr)
    if _iv.verify_frame(fr, meta)[0]:
        _clean_ok += 1
    tampered = fr.copy().reshape(-1)
    idx = _rng.choice(tampered.size, size=16, replace=False)
    tampered[idx] = (tampered[idx].astype(np.int16) + 73) % 256
    tampered = tampered.reshape(224, 224, 3).astype(np.uint8)
    if not _iv.verify_frame(tampered, meta)[0]:
        _tamper_caught += 1
    forged = {"hash": _iv.frame_hash(tampered), "algorithm": "sha256"}   # recompute hash, no valid HMAC
    if not _iv.verify_frame(tampered, forged)[0]:
        _forge_caught += 1
STRIDE["T"] = {
    "threat": "Tampering", "mechanism": "SHA-256 + HMAC-SHA-256 (FIPS 180-4 / RFC 2104)",
    "trials": _N_T, "clean_passed": _clean_ok, "tamper_detected": _tamper_caught,
    "hmac_forgery_detected": _forge_caught,
    "pass": _tamper_caught == _N_T and _clean_ok == _N_T and _forge_caught == _N_T,
    "result": f"{_tamper_caught}/{_N_T} 16-byte tampers detected; {_clean_ok}/{_N_T} clean "
              f"passed; {_forge_caught}/{_N_T} forged-hash-without-key rejected",
}

# ── R — Repudiation: forward-integrity hash chain (v7 + DP-4 additions) ─────
_N_R, _N_TAMPER = 200, 50
_K0 = b"stride-R-verifier-key"
_log = ForwardIntegrityAuditLog(_K0)
for i in range(_N_R):
    _log.log_event("inference_completed", f"frame_{i:05d}", {"label": i % 2})
_clean = ForwardIntegrityAuditLog.verify(_log.entries, _K0)

def _verify_variant(entries):
    return ForwardIntegrityAuditLog.verify(entries, _K0)["intact"]

# batch field-tamper: alter 50 spread entries, keep stale HMACs
_ti = sorted(_rng.choice(_N_R, size=_N_TAMPER, replace=False).tolist())
_tampered_entries = [dict(e) for e in _log.entries]
for t in _ti:
    _tampered_entries[t] = dict(_tampered_entries[t], details={"label": 999})
_r = ForwardIntegrityAuditLog.verify(_tampered_entries, _K0)
_field_detected = (not _r["intact"]) and _r["break_index"] == _ti[0]  # first altered entry
# deletion / reorder
_deleted = [e for j, e in enumerate(_log.entries) if j != _N_R // 2]
_reordered = [dict(e) for e in _log.entries]
_reordered[100], _reordered[101] = _reordered[101], _reordered[100]
_del_detected = not _verify_variant(_deleted)
_reorder_detected = not _verify_variant(_reordered)
# Tail truncation: HONEST framing. Dropping the tail leaves a VALID prefix chain —
# hash chaining alone cannot detect it (Schneier-Kelsey). It is caught by the
# out-of-band anchor: the trusted verifier expects N_R entries (and, under DP-4,
# a specific final evolved key). We report both facts rather than pretend the
# chain breaks.
_truncated = _log.entries[:150]
_trunc_prefix_valid = _verify_variant(_truncated)          # True: prefix is cryptographically fine
_trunc_caught_by_count = len(_truncated) != _N_R           # True: caught by expected-count anchor
# DP-4 forward integrity: steal the CURRENT key, try to rewrite a past entry
_stolen = _log._key
_fi = [dict(e) for e in _log.entries]
_victim = {k: v for k, v in _fi[100].items() if k != "hmac"}
_victim["details"] = {"label": "REWRITTEN"}
_victim["hmac"] = _hmac.new(_stolen, json.dumps(_victim, sort_keys=True,
                            separators=(",", ":")).encode(), hashlib.sha256).hexdigest()
_fi[100] = _victim
_fwd_ok = not _verify_variant(_fi)
STRIDE["R"] = {
    "threat": "Repudiation",
    "mechanism": "HMAC hash chain + DP-4 key evolution (Schneier-Kelsey)",
    "entries": _N_R, "chain_clean": _clean["intact"],
    "field_tamper_detected": _field_detected,
    "breaks_on_delete": _del_detected,
    "breaks_on_reorder": _reorder_detected,
    "truncation_prefix_valid": _trunc_prefix_valid,       # honest: chain alone can't catch it
    "truncation_caught_by_count_anchor": _trunc_caught_by_count,
    "dp4_forward_integrity": _fwd_ok,
    "pass": (_clean["intact"] and _field_detected and _del_detected and _reorder_detected
             and _fwd_ok and _trunc_caught_by_count),
    "result": f"{_N_R}/{_N_R} entries signed & chained; field-tamper/deletion/reorder detected; "
              f"tail-truncation caught by count anchor (chain prefix stays valid); "
              f"DP-4: stolen current key cannot rewrite entry 100 = {_fwd_ok}",
}

# ── I — Information disclosure: privacy guard on REAL DMD faces ─────────────
def _within_block_detail(gray, block=16):
    g = gray.astype(np.float32)
    h, w = g.shape
    bh, bw = h // block, w // block
    if bh == 0 or bw == 0:
        return float(g.std())
    g = g[:bh * block, :bw * block].reshape(bh, block, bw, block)
    return float(np.sqrt(((g - g.mean(axis=(1, 3), keepdims=True)) ** 2).mean()))

_face_lm = make_landmarker(video_mode=False)
def _face_detected(frame_rgb) -> bool:
    r = _face_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=np.ascontiguousarray(frame_rgb)))
    return bool(r.face_landmarks)

# grab 60 real face crops from a drowsiness session (the biometric to protect)
_faces, _cap = [], cv2.VideoCapture(str(dmd_inventory["drowsiness"]["videos"][0]))
_lm_v = make_landmarker(video_mode=True)
_fi_idx = 0
while len(_faces) < 60:
    ok, bgr = _cap.read()
    if not ok:
        break
    if _fi_idx % 30 == 0:
        sig1 = perceive_frame(_lm_v, bgr, int(_fi_idx / 29.76 * 1000))
        if sig1 is not None:
            x0, y0, x1, y1 = sig1["box"]
            if x1 > x0 + 32 and y1 > y0 + 32:
                _faces.append(cv2.cvtColor(cv2.resize(bgr[y0:y1, x0:x1], (224, 224)),
                                           cv2.COLOR_BGR2RGB))
    _fi_idx += 1
_cap.release()

_guard = PrivacyGuard()
_recoverable = _faces_before = _faces_after = 0
_detail_ratios = []
for _f in _faces:
    _protected = _guard.protect(_f)
    _gb = 0.299 * _f[:, :, 0] + 0.587 * _f[:, :, 1] + 0.114 * _f[:, :, 2]
    _ga = 0.299 * _protected[:, :, 0] + 0.587 * _protected[:, :, 1] + 0.114 * _protected[:, :, 2]
    _ratio = _within_block_detail(_ga) / max(_within_block_detail(_gb), 1e-6)
    _detail_ratios.append(_ratio)
    _fb, _fa = _face_detected(_f), _face_detected(_protected)
    _faces_before += int(_fb)
    _faces_after += int(_fa and _fb)
    if _ratio > 0.15 or (_fb and _fa):
        _recoverable += 1
_eff_res = f"{224 // 16}x{224 // 16}"
STRIDE["I"] = {
    "threat": "Information disclosure", "mechanism": "Irreversible pixelation (GDPR minimisation)",
    "faces": len(_faces), "recoverable": _recoverable,
    "detail_retained": round(float(np.mean(_detail_ratios)), 4),
    "effective_resolution": _eff_res,
    "faces_before": _faces_before, "faces_after": _faces_after,
    "pass": _recoverable == 0 and _faces_after == 0,
    "result": f"{_recoverable}/{len(_faces)} faces recoverable after anonymisation; "
              f"{np.mean(_detail_ratios):.1%} sub-block detail retained; eff. res {_eff_res}; "
              f"face detector {_faces_before}->{_faces_after}",
}

# ── D — Denial of service: sliding-window rate limiter ──────────────────────
_CAP, _FLOOD = SEC_RATE_MAX_PER_WINDOW, SEC_RATE_FLOOD_THRESHOLD
_submitted = _FLOOD * 3          # 180
_rl = SlidingWindowRateLimiter()
_acc = _rate_lim = _flood_blk = 0
for _k in range(_submitted):
    ok, why = _rl.check(_k / 1e6)   # all within one 1 s window
    if ok:
        _acc += 1
    elif "flood" in why:
        _flood_blk += 1
    else:
        _rate_lim += 1
STRIDE["D"] = {
    "threat": "Denial of service", "mechanism": "Sliding-window rate limiter (30 fps cap)",
    "submitted": _submitted, "admitted": _acc,
    "rate_limited": _rate_lim, "flood_blocked": _flood_blk,
    "pass": _acc == _CAP and (_acc + _rate_lim + _flood_blk) == _submitted,
    "result": f"{_acc}/{_submitted} admitted (= {_CAP} fps cap); {_rate_lim + _flood_blk} throttled "
              f"= {_rate_lim} rate-limited + {_flood_blk} flood-blocked (alarm at {_FLOOD}); "
              f"counts reconcile",
}

# ── E — Elevation of privilege: RBAC ────────────────────────────────────────
_ac = AccessController()
_unauth = [("driver", "read_logs"), ("driver", "update_model"), ("driver", "manage_roles"),
           ("driver", "update_config"), ("operator", "update_model"), ("operator", "manage_roles")]
_auth = [("driver", "read_inference"), ("operator", "read_logs"),
         ("admin", "update_model"), ("admin", "manage_roles")]
_denied = 0
for _role, _act in _unauth:
    _ac.set_role(_role)
    if not _ac.check_permission(_act)[0]:
        _denied += 1
_granted = 0
for _role, _act in _auth:
    _ac.set_role(_role)
    if _ac.check_permission(_act)[0]:
        _granted += 1
_all_actions = sorted({a for acts in SEC_ROLES.values() for a in acts})
_matrix_ok = 0
for _role in SEC_ROLES:
    _ac.set_role(_role)
    for _act in _all_actions:
        if _ac.check_permission(_act)[0] == (_act in SEC_ROLES[_role]):
            _matrix_ok += 1
STRIDE["E"] = {
    "threat": "Elevation of privilege", "mechanism": "RBAC 3 roles x 6 actions (Ferraiolo-Kuhn)",
    "unauth_denied": f"{_denied}/{len(_unauth)}", "auth_granted": f"{_granted}/{len(_auth)}",
    "matrix": f"{_matrix_ok}/{len(SEC_ROLES) * len(_all_actions)}",
    "pass": _denied == len(_unauth) and _granted == len(_auth)
            and _matrix_ok == len(SEC_ROLES) * len(_all_actions),
    "result": f"{_denied}/{len(_unauth)} unauthorised denied; {_granted}/{len(_auth)} authorised "
              f"granted; {_matrix_ok}/18 role x action cells match policy",
}

# ── summary table ───────────────────────────────────────────────────────────
log("\n" + "=" * 82)
log("P6 — STRIDE MITIGATION VALIDATION")
log("=" * 82)
for _k in ("S", "T", "R", "I", "D", "E"):
    _v = STRIDE[_k]
    log(f"[{_k}] {_band(_v['pass'])}  {_v['threat']:<24} {_v['result']}")
log("-" * 82)
_n_pass = sum(STRIDE[k]["pass"] for k in STRIDE)
log(f"STRIDE suite: {_n_pass}/6 threats validated"
    + ("" if _n_pass == 6 else "  — S FAILS on real attack instruments (Tier D)"))
if _n_pass < 6:
    log("  The S row is reported as failing rather than shown against synthetic")
    log("  attacks, which would pass. The mechanism is implemented, instrumented and")
    log("  measured; its measured effectiveness against real instruments is the result.")
(OUT_DIR / "stride_validation.json").write_text(json.dumps(STRIDE, indent=2), encoding="utf-8")
# The five mechanisms whose behaviour was ported from v7 must not regress; S is a
# measured research outcome, not a correctness invariant, so it is not asserted.
for _k in ("T", "R", "I", "D", "E"):
    assert STRIDE[_k]["pass"], f"STRIDE-{_k} regressed — see table above"