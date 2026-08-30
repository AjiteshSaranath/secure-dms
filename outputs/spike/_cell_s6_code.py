"""6 — security layer (P5). Five mechanisms ported from the frozen v7 modules,
behaviour-preserving, plus the approved DP-4 forward-integrity upgrade.

Ports (v7 file -> here): utils/integrity.py -> IntegrityVerifier ·
utils/rate_limiter.py -> SlidingWindowRateLimiter · utils/access_control.py ->
AccessController · utils/logger.py AuditLogger -> ForwardIntegrityAuditLog
(+ DP-4) · utils/privacy.py -> PrivacyGuard. Literature justification for every
mechanism vs its alternatives: plan §6.2-6.7.
"""
import hmac as hmac_lib
from collections import deque

# ── configuration (v7 values preserved; sources noted) ──────────────────────
SEC_HMAC_SECRET = "dms-prototype-shared-secret"   # prototype key; production => TEE/TPM (plan §6.3.4)
SEC_RATE_MAX_PER_WINDOW = 30        # legitimate sensor contract: 30 fps
SEC_RATE_WINDOW_S = 1.0
SEC_RATE_FLOOD_THRESHOLD = 60       # above => declared flood (DoS taxonomy, R59)
SEC_ROLES = {                       # v7 role x action matrix, unchanged (18 cells, STRIDE-E)
    "driver":   ["read_inference", "read_status"],
    "operator": ["read_inference", "read_status", "read_logs"],
    "admin":    ["read_inference", "read_status", "read_logs",
                 "update_model", "update_config", "manage_roles"],
}
SEC_DEFAULT_ROLE = "driver"
SEC_ANON_METHOD = "pixelate"        # irreversible downsampling (plan §6.7; GDPR minimisation R64)
SEC_ANON_STRENGTH = 16              # v7 value -> 14x14 effective resolution on 224px faces (R65)


# ── Tampering: SHA-256 frame integrity + HMAC authenticity (plan §6.2-6.3) ──
class IntegrityVerifier:
    """SHA-256 frame hash (FIPS 180-4, R40) + HMAC-SHA-256 over the hash
    (RFC 2104 / FIPS 198-1, R48/R49; PRF security per Bellare, R51).

    The bare hash detects accidental/blind modification; the HMAC prevents an
    attacker who can rewrite BOTH frame and metadata from forging a valid pair,
    and sidesteps Merkle-Damgard length extension (R46) by construction.
    """

    def __init__(self, secret: str = SEC_HMAC_SECRET):
        self._secret = secret.encode()

    @staticmethod
    def frame_hash(frame: np.ndarray) -> str:
        return hashlib.sha256(frame.tobytes()).hexdigest()

    def _hmac(self, frame_hash_hex: str) -> str:
        return hmac_lib.new(self._secret, frame_hash_hex.encode(), hashlib.sha256).hexdigest()

    def sign_frame(self, frame: np.ndarray) -> dict:
        """Producer side (simulates the trusted sensor driver)."""
        h = self.frame_hash(frame)
        return {"hash": h, "hmac": self._hmac(h), "algorithm": "sha256"}

    def verify_frame(self, frame: np.ndarray, metadata: dict) -> tuple[bool, str]:
        if not metadata:
            return False, "missing integrity metadata"
        recomputed = self.frame_hash(frame)
        if not hmac_lib.compare_digest(recomputed, metadata.get("hash", "")):
            return False, "hash mismatch - frame tampered"
        if not hmac_lib.compare_digest(self._hmac(recomputed), metadata.get("hmac", "")):
            return False, "HMAC mismatch - metadata forged"
        return True, "OK"


# ── DoS: sliding-window rate limiter (plan §6.5) ────────────────────────────
class SlidingWindowRateLimiter:
    """Strict per-window cap fits a fixed-rate sensor (30 fps contract); the
    sliding window avoids the fixed-window boundary-burst artefact (R58), and
    burst tolerance - token bucket's defining feature - is exactly what a
    frame-flood attacker would exploit here. Graduated response per R59.

    Port note: v7 used wall-clock time.monotonic(); this port injects the frame
    timestamp instead, so behaviour is identical but tests are deterministic and
    a flood can be SIMULATED by timestamps rather than by real-time busy-waiting.
    """

    def __init__(self, max_per_window: int = SEC_RATE_MAX_PER_WINDOW,
                 window_seconds: float = SEC_RATE_WINDOW_S,
                 flood_threshold: int = SEC_RATE_FLOOD_THRESHOLD):
        self.max_per_window = max_per_window
        self.window_seconds = window_seconds
        self.flood_threshold = flood_threshold
        self._timestamps = deque()
        self.blocked_total = 0
        self.flood_events = 0

    def check(self, now_s: float) -> tuple[bool, str]:
        while self._timestamps and self._timestamps[0] < now_s - self.window_seconds:
            self._timestamps.popleft()
        current = len(self._timestamps)
        self._timestamps.append(now_s)
        if current >= self.flood_threshold:
            self.flood_events += 1
            self.blocked_total += 1
            return False, f"DoS flood ({current}/{self.window_seconds:.0f}s)"
        if current >= self.max_per_window:
            self.blocked_total += 1
            return False, f"rate limit ({current}/{self.max_per_window} fps)"
        return True, "OK"

    def reset(self):
        self._timestamps.clear()
        self.blocked_total = 0
        self.flood_events = 0


# ── Elevation of privilege: RBAC (plan §6.6) ────────────────────────────────
class AccessController:
    """Role-based access control (Ferraiolo & Kuhn R60; Sandhu et al. R61).
    Small static role x action matrix => completely verifiable by inspection,
    which is the auditability argument against ABAC here (SP 800-162, R63).
    Session role stands in for TEE/token attestation (prototype scope)."""

    def __init__(self, roles: dict = None, default_role: str = SEC_DEFAULT_ROLE):
        self.roles = dict(roles or SEC_ROLES)
        self.current_role = default_role
        self.violations = 0
        self.access_log: list[dict] = []

    def set_role(self, role: str):
        if role not in self.roles:
            raise ValueError(f"unknown role: {role}")
        self.current_role = role

    def check_permission(self, action: str) -> tuple[bool, str]:
        granted = action in self.roles.get(self.current_role, [])
        self.access_log.append({"role": self.current_role, "action": action,
                                "granted": granted})
        if not granted:
            self.violations += 1
            return False, f"'{action}' not permitted for role '{self.current_role}'"
        return True, "OK"


# ── Repudiation: forward-integrity hash-chained audit log (plan §6.4 + DP-4) ─
class ForwardIntegrityAuditLog:
    """HMAC-signed, hash-chained audit log (Schneier & Kelsey, R54; linkage
    lineage Haber & Stornetta, R55) with the approved DP-4 upgrade: per-entry
    KEY EVOLUTION K_{i+1} = SHA-256(K_i), old key destroyed.

    Properties (each maps to a STRIDE-R test in §9):
      - field tampering    -> that entry's HMAC fails
      - deletion/reorder   -> prev-HMAC linkage breaks
      - truncation of the  -> verifier's expected chain length / final key
        tail                  mismatch (chain is append-only, count is logged)
      - DP-4 forward integrity: an attacker who compromises the system at time t
        obtains only K_t; keys K_0..K_{t-1} are DESTROYED, so pre-compromise
        entries cannot be re-signed. The v7 fixed-key chain lacked exactly this.

    The verifier is assumed to hold K_0 in a separate trust domain (e.g. the
    OEM's audit service); verification replays the evolution from K_0.
    """

    GENESIS = "GENESIS"

    def __init__(self, initial_key: bytes):
        self._key = hashlib.sha256(initial_key).digest()   # K_0 (never stored here)
        self.entries: list[dict] = []
        self._last_hmac = self.GENESIS

    @staticmethod
    def _evolve(key: bytes) -> bytes:
        return hashlib.sha256(key).digest()

    def _sign(self, payload: dict, key: bytes) -> str:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hmac_lib.new(key, body.encode(), hashlib.sha256).hexdigest()

    def log_event(self, event: str, frame_id: str, details: dict):
        entry = {"seq": len(self.entries), "event": event, "frame_id": frame_id,
                 "details": details, "prev_hmac": self._last_hmac}
        entry["hmac"] = self._sign(entry, self._key)
        self.entries.append(entry)
        self._last_hmac = entry["hmac"]
        self._key = self._evolve(self._key)      # DP-4: destroy K_i, hold K_{i+1}

    @classmethod
    def verify(cls, entries: list[dict], initial_key: bytes) -> dict:
        """Trusted-verifier check: replay the key evolution from K_0."""
        key = hashlib.sha256(initial_key).digest()
        prev = cls.GENESIS
        for i, stored in enumerate(entries):
            entry = {k: v for k, v in stored.items() if k != "hmac"}
            if entry.get("prev_hmac") != prev:
                return {"intact": False, "break_index": i,
                        "reason": "broken link (deletion/reordering)"}
            expected = hmac_lib.new(
                key, json.dumps(entry, sort_keys=True, separators=(",", ":")).encode(),
                hashlib.sha256).hexdigest()
            if not hmac_lib.compare_digest(stored.get("hmac", ""), expected):
                return {"intact": False, "break_index": i,
                        "reason": "HMAC mismatch (tampered or wrong-epoch key)"}
            prev = stored["hmac"]
            key = hashlib.sha256(key).digest()
        return {"intact": True, "entries": len(entries), "break_index": None}


# ── Information disclosure: privacy guard (plan §6.7) ───────────────────────
class PrivacyGuard:
    """Irreversible pixelation of any frame the pipeline stores or rejects.
    Data minimisation by architecture (GDPR Art. 5(1)(c), R64); block size 16 on
    a ~224px face gives ~14x14 effective facial resolution, inside the
    very-low-resolution regime R65 defines as "lower than 16 x 16". Note R65
    characterises that regime as HARD (and proposes super-resolution for it) —
    it does not claim recognition is impossible there, so the resolution figure
    is context, not proof. Non-recoverability is TESTED in STRIDE-I (§9), and
    that measurement is what carries the claim."""

    def __init__(self, method: str = SEC_ANON_METHOD, strength: int = SEC_ANON_STRENGTH):
        self.method = method
        self.strength = strength
        self.anonymised = 0

    def protect(self, frame: np.ndarray) -> np.ndarray:
        self.anonymised += 1
        h, w = frame.shape[:2]
        if self.method == "blackout":
            return np.zeros_like(frame)
        if self.method == "blur":
            k = self.strength | 1
            return cv2.GaussianBlur(frame, (k, k), 0)
        small = cv2.resize(frame, (max(1, w // self.strength), max(1, h // self.strength)),
                           interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


log("security layer ready — integrity(SHA-256+HMAC) · rate-limit · RBAC · "
    "forward-integrity audit chain (DP-4) · privacy guard")