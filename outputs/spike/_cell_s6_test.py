"""6.1 — security-mechanism self-tests.

Functional checks that each mechanism does what its section claims, run on every
`Run All`. These are NOT the STRIDE validation suite (§9 runs the full attack
scenarios with canonical counts); they are the precondition for assembling the
pipeline at all. Every check asserts — a regression fails the notebook loudly.
"""
_rng = np.random.default_rng(SEED)
_frame = _rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)

# ── integrity: tamper + forge detection ─────────────────────────────────────
_iv = IntegrityVerifier()
_meta = _iv.sign_frame(_frame)
assert _iv.verify_frame(_frame, _meta) == (True, "OK")
_tampered = _frame.copy(); _tampered[0, 0, 0] ^= 1          # single-bit flip
ok, why = _iv.verify_frame(_tampered, _meta)
assert not ok and "hash" in why
_forged = dict(_meta, hash=_iv.frame_hash(_tampered))        # attacker recomputes hash only
ok, why = _iv.verify_frame(_tampered, _forged)
assert not ok and "HMAC" in why, "forged hash must fail without the shared secret"
log("6.1 integrity: 1-bit tamper detected; forged-hash-without-key detected")

# ── rate limiter: normal rate passes, flood throttled, deterministic ────────
_rl = SlidingWindowRateLimiter()
_normal = [_rl.check(i / 29.76)[0] for i in range(60)]        # 2 s at sensor rate
assert all(_normal), "legitimate 30 fps stream must fully pass"
_rl.reset()
_flood = [_rl.check(i / 180.0)[0] for i in range(180)]        # 180 frames in 1 s
_admitted = sum(_flood)
assert _admitted == SEC_RATE_MAX_PER_WINDOW, \
    f"flood must admit exactly the {SEC_RATE_MAX_PER_WINDOW}-frame cap, got {_admitted}"
assert _rl.flood_events > 0
log(f"6.1 rate-limit: 30 fps passes; 180-frame flood -> {_admitted} admitted, "
    f"{_rl.blocked_total} blocked ({_rl.flood_events} flood events)")

# ── RBAC: full role x action matrix, exhaustively ──────────────────────────
_ac = AccessController()
_all_actions = sorted({a for acts in SEC_ROLES.values() for a in acts})
_matrix_checks = 0
for _role in SEC_ROLES:
    _ac.set_role(_role)
    for _action in _all_actions:
        _granted, _ = _ac.check_permission(_action)
        assert _granted == (_action in SEC_ROLES[_role])
        _matrix_checks += 1
assert _matrix_checks == len(SEC_ROLES) * len(_all_actions)   # 3 x 6 = 18
log(f"6.1 RBAC: {_matrix_checks}/18 role x action cells behave per matrix; "
    f"{_ac.violations} violations correctly flagged")

# ── audit chain: tamper / delete / reorder / truncate / forward integrity ──
_K0 = b"k0-held-by-trusted-verifier"
def _fresh_log(n=20):
    lg = ForwardIntegrityAuditLog(_K0)
    for i in range(n):
        lg.log_event("frame_decision", f"f{i:04d}", {"verdict": "safe", "i": i})
    return lg

_lg = _fresh_log()
assert ForwardIntegrityAuditLog.verify(_lg.entries, _K0)["intact"]

_e = [dict(x) for x in _lg.entries]; _e[7]["details"] = {"verdict": "unsafe", "i": 7}
_r = ForwardIntegrityAuditLog.verify(_e, _K0)
assert not _r["intact"] and _r["break_index"] == 7

_e = [dict(x) for x in _lg.entries]; del _e[5]
assert not ForwardIntegrityAuditLog.verify(_e, _K0)["intact"]

_e = [dict(x) for x in _lg.entries]; _e[3], _e[4] = _e[4], _e[3]
assert not ForwardIntegrityAuditLog.verify(_e, _K0)["intact"]

# DP-4 forward integrity: attacker compromises the device AFTER entry 10, i.e.
# steals the CURRENT key K_15, and tries to rewrite history at entry 10 with it.
_lg2 = _fresh_log(15)
_stolen_key = _lg2._key                                   # what a t=15 compromise yields
_e = [dict(x) for x in _lg2.entries]
_victim = {k: v for k, v in _e[10].items() if k != "hmac"}
_victim["details"] = {"verdict": "REWRITTEN", "i": 10}
_victim["hmac"] = hmac_lib.new(
    _stolen_key, json.dumps(_victim, sort_keys=True, separators=(",", ":")).encode(),
    hashlib.sha256).hexdigest()
_e[10] = _victim
_r = ForwardIntegrityAuditLog.verify(_e, _K0)
assert not _r["intact"] and _r["break_index"] == 10, \
    "DP-4: stolen current key must NOT be able to re-sign pre-compromise entries"
log("6.1 audit chain: tamper/delete/reorder detected at exact index; "
    "DP-4 forward integrity holds (stolen K_15 cannot rewrite entry 10)")

# ── privacy guard: irreversibility proxy ────────────────────────────────────
_pg = PrivacyGuard()
_protected = _pg.protect(_frame)
_unique_blocks = len(np.unique(_protected.reshape(-1, 3), axis=0))
_orig_blocks = len(np.unique(_frame.reshape(-1, 3), axis=0))
assert _protected.shape == _frame.shape
assert _unique_blocks < _orig_blocks / 50, \
    "pixelation must collapse information content (~256x fewer distinct values)"
assert not np.array_equal(_protected, _frame)
log(f"6.1 privacy: pixelation collapses {_orig_blocks:,} distinct colours -> "
    f"{_unique_blocks:,} (information destruction; recoverability tested in §9)")

log("6.1 ALL SECURITY SELF-TESTS PASS")