## 6. Security layer (P5)

The five STRIDE mechanisms, ported **behaviour-preserving** from the frozen v7 modules
(`../dms_project/utils/{integrity, rate_limiter, access_control, logger, privacy}.py`), plus the
approved **DP-4 upgrade**. Full "why this and not the alternatives" justifications with literature:
plan §6.2–§6.7. Summary of the choices being implemented:

| STRIDE threat | Mechanism | Key evidence (ledger) | Rejected alternatives (plan) |
|---|---|---|---|
| Tampering | SHA-256 frame hash | FIPS 180-4 (R40); no practical attacks, hardware support | MD5/SHA-1 broken (R42, R43); SHA-3 adds nothing here (R41); BLAKE not FIPS-approved (R45) |
| + Authenticity | HMAC-SHA-256 over the hash | RFC 2104/FIPS 198-1 (R48, R49); PRF-secure even without collision resistance (Bellare, R51); kills length-extension (R46) | CMAC (second primitive family), GMAC (nonce-fragile), Poly1305 (one-time keys), signatures (no third-party verifier in scope) |
| Repudiation | HMAC-signed hash chain **+ DP-4 key evolution** `K_{i+1}=H(K_i)` | Schneier & Kelsey (R54); Haber & Stornetta (R55) | Merkle/history trees (built for untrusted third-party loggers, R56); blockchain (no consensus problem exists) |
| DoS | Sliding-window rate limiter, 30 fps cap | boundary-burst artefact of fixed windows (R58); flood taxonomy (R59) | Token bucket — its burst tolerance is the attack surface here; leaky bucket queues what should be rejected |
| Elevation of privilege | RBAC, 3 roles × 6 actions | Ferraiolo & Kuhn (R60); Sandhu (R61) | ABAC oversized for a static enumerable matrix (SP 800-162, R63); ACLs obscure role semantics |
| Information disclosure | Irreversible pixelation (→ ~14×14 face) + minimisation | GDPR Art. 5(1)(c) (R64); 14×14 sits inside the very-low-resolution regime R65 defines as "< 16×16" — **and non-recoverability is measured in §9, not inferred** | Encryption-only retains the biometric; k-Same needs generative models (R66); blackout destroys audit utility |

**What DP-4 adds over v7.** The v7 chain used one fixed key: an attacker who obtained it could
re-sign a *rewritten history*. With per-entry key evolution the signer holds only the current key
K_t and destroys predecessors, so compromise at time t cannot forge any entry before t — the chain
gains **forward integrity**, verified by a trusted party replaying the evolution from K_0
(held off-device, e.g. by the OEM's audit service). Cost: one extra SHA-256 per entry.

**Port deviations (all deliberate, all behaviour-preserving):** the rate limiter takes an injected
timestamp instead of wall-clock `time.monotonic()`, so DoS tests are deterministic and simulate
floods by timestamps rather than real-time busy-waiting; the audit log keeps entries in memory for
the in-notebook verifier (v7 appended to a file); logging goes through the notebook's provenance
`log()`. The §6.1 self-test cell must pass before the pipeline is assembled.