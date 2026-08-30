"""7 — secure pipeline assembly (P5): the seven stages in the v7 order (D12).

  rate-limit -> integrity/HMAC -> access control -> PAD -> driver-state
  -> hash-chained audit -> privacy guard

Runtime semantics:
- SECURITY stages (rate-limit, integrity, access) run PER FRAME on the raw video
  bytes and HARD-GATE: a frame that is flooded, tampered, or unauthorised is
  invalid input and is dropped, anonymised, and audited.
- PAD and DRIVER STATE are WINDOW-LEVEL decisions (10 s / 60 s) fed by the
  perception signals of §§2-3, reusing the validated signal cache by frame index.

**PAD does NOT hard-gate driver-state — a deliberate, evidence-driven design
decision (see P4_FINDINGS + the P5 diagnostic).** Measured genuine window-level
BPCER is ~15 % pooled and up to ~73 % on individual sessions. A hard PAD gate in
front of the safety-critical drowsiness detector would therefore drop most frames
of some genuine drivers before they were ever monitored — the security control
would suppress the very alerts the system exists to raise. The diagnostic also
refuted the initial worry that PAD conflates drowsiness with spoofing (eye-closure
is identical in rejected vs admitted genuine windows, 0.118 vs 0.117), so the
issue is baseline liveness-cue noise, not a drowsiness bias.

Resolution (consistent with the paper's decision D14, liveness as defence-in-depth,
not a single point of failure): a not-live window raises an AUDITED SPOOF ALERT
and is counted, but driver-state inference STILL RUNS on those frames. The three
input-validity stages fail closed (drop); the PAD classification fails safe
(alert, keep monitoring). A deployed system would additionally require sustained
multi-window spoof evidence before escalating — stated in the P6 STRIDE-S framing.
"""


class SecurePipeline:
    STAGES = ("rate_limit", "integrity", "access", "pad", "driver_state",
              "audit", "privacy")

    def __init__(self, verifier_key: bytes = b"k0-held-by-trusted-verifier"):
        self.integrity = IntegrityVerifier()
        self.rate_limiter = SlidingWindowRateLimiter()
        self.access = AccessController()
        self.audit = ForwardIntegrityAuditLog(verifier_key)
        self.privacy = PrivacyGuard()
        self.counters = {s: {"pass": 0, "reject": 0} for s in self.STAGES}
        self.spoof_alerts = 0          # PAD not-live windows: alerted, NOT gated
        self.anonymised_examples: list[np.ndarray] = []

    def process_video(self, video: Path, sig: dict, max_frames: int | None = None,
                      role: str = SEC_DEFAULT_ROLE) -> dict:
        """Run one session end-to-end. `sig` is the cached signal timeline."""
        self.access.set_role(role)
        fps = float(sig["fps"][0])
        pad_w = max(int(round(PAD_WINDOW_S * fps)), 2)

        # window-level PAD verdicts over the cached signals (§5)
        pad_verdicts = apply_pad_decisions(pad_windows(sig, label=video.name))
        state = driver_state(sig)                      # session driver-state (§4)
        sig_pos = {int(f): i for i, f in enumerate(sig["frame_idx"])}

        cap = cv2.VideoCapture(str(video))
        frame_i, results = 0, {"admitted": 0, "rejected": 0}
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and frame_i >= max_frames):
                break
            fid = f"{video.stem[:18]}:{frame_i:05d}"
            meta = self.integrity.sign_frame(frame)    # trusted producer side
            disposition, reason = self._process_frame(
                frame, meta, frame_i / fps, fid,
                pad_verdicts[frame_i // pad_w] if frame_i // pad_w < len(pad_verdicts) else None,
                state, frame_i in sig_pos)
            results["admitted" if disposition else "rejected"] += 1
            frame_i += 1
        cap.release()

        chain = ForwardIntegrityAuditLog.verify(self.audit.entries,
                                                b"k0-held-by-trusted-verifier")
        return {"frames": frame_i, **results, "counters": self.counters,
                "spoof_alerts": self.spoof_alerts,
                "session_state": {k: state[k] for k in ("verdict", "reasons",
                                  "perclos_max", "n_yawns")},
                "audit_entries": len(self.audit.entries), "chain": chain}

    def _process_frame(self, frame, meta, t_s, fid, pad_verdict, state, has_face):
        # 1. DoS
        ok, reason = self.rate_limiter.check(t_s)
        if not self._tally("rate_limit", ok):
            return self._reject(frame, fid, "rate_limit", reason)
        # 2. Tampering / authenticity
        ok, reason = self.integrity.verify_frame(frame, meta)
        if not self._tally("integrity", ok):
            return self._reject(frame, fid, "integrity", reason)
        # 3. Elevation of privilege — the inference action itself is guarded
        ok, reason = self.access.check_permission("read_inference")
        if not self._tally("access", ok):
            return self._reject(frame, fid, "access", reason)
        # 4. Spoofing — PAD FLAGS but does not gate (fail-safe; see class docstring)
        live = bool(pad_verdict["is_live"]) if pad_verdict is not None else True
        self.counters["pad"]["pass" if live else "reject"] += 1
        if not live:
            self.spoof_alerts += 1
            self.audit.log_event("spoof_alert", fid, {
                "pad_votes": pad_verdict["votes"], "note": "monitoring continues"})
        # 5. Driver state runs REGARDLESS of the PAD flag (safety function)
        self.counters["driver_state"]["pass"] += 1
        # 6. Audit the admission
        self.audit.log_event("frame_admitted", fid, {
            "driver_state": state["verdict"], "spoof_flag": not live})
        self.counters["audit"]["pass"] += 1
        return True, "spoof_flagged" if not live else "admitted"

    def _tally(self, stage, ok):
        self.counters[stage]["pass" if ok else "reject"] += 1
        return ok

    def _reject(self, frame, fid, stage, reason):
        # rejected frames: audit the rejection, anonymise before any retention
        self.audit.log_event("frame_rejected", fid, {"stage": stage, "reason": reason})
        self.counters["audit"]["pass"] += 1
        protected = self.privacy.protect(frame)
        self.counters["privacy"]["pass"] += 1
        if len(self.anonymised_examples) < 3:
            self.anonymised_examples.append(protected)
        return False, reason


# ── end-to-end smoke run: one genuine session, 60 s ─────────────────────────
_demo = dmd_inventory["drowsiness"]["videos"][0]
_pipe = SecurePipeline()
P5_DEMO = _pipe.process_video(_demo, extract_signals(_demo), max_frames=1800)

log("\n" + "=" * 78)
log("P5 — secure pipeline end-to-end smoke run (60 s of a genuine session)")
log("=" * 78)
log(f"session: {_demo.name[:40]}…  role: {SEC_DEFAULT_ROLE}")
log(f"frames processed {P5_DEMO['frames']:,} | admitted (monitored) {P5_DEMO['admitted']:,} "
    f"| hard-rejected {P5_DEMO['rejected']:,} | spoof-flagged {P5_DEMO['spoof_alerts']:,}")
for s in SecurePipeline.STAGES:
    c = P5_DEMO["counters"][s]
    tag = "  (flag only, not gated)" if s == "pad" else ""
    log(f"  stage {s:<13} pass {c['pass']:>6,}  reject {c['reject']:>5,}{tag}")
log(f"driver state: {P5_DEMO['session_state']}")
log(f"audit: {P5_DEMO['audit_entries']:,} chained entries | "
    f"chain intact: {P5_DEMO['chain']['intact']}")
assert P5_DEMO["chain"]["intact"], "audit chain must verify after a clean run"
assert P5_DEMO["frames"] == P5_DEMO["admitted"] + P5_DEMO["rejected"]
# On a GENUINE, well-formed session no frame is flooded/tampered/unauthorised, so
# every frame must reach driver-state; PAD may still raise spoof flags (defence-in-depth).
assert P5_DEMO["rejected"] == 0, "genuine session must not be hard-rejected by the pipeline"
log(f"P5 smoke run PASS — every genuine frame monitored; {P5_DEMO['spoof_alerts']:,} "
    f"PAD spoof flags raised (defence-in-depth, not gated). Full STRIDE suite = §9 (P6)")