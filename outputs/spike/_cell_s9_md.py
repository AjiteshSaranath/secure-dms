## 9. STRIDE validation suite (P6)

Each of the six STRIDE threats is exercised in isolation with a controlled experiment producing one
headline number and a PASS/FAIL. This is the central thesis contribution's evidence table. The suite
uses the §6 security classes and the §5 PAD directly — no re-implementation — and asserts, so any
regression fails the notebook.

**Continuity with v7.** T, R, I, D, E were ported behaviour-preserving (P5), so they keep the v7
pass criteria and canonical counts. **S is re-specified** around the new temporal-behavioural PAD
(the old Laplacian+LBP gate is gone, DR-4) and around the PAD-as-alert semantics (P5_FINDINGS §2):
a spoof "rejection" is an audited alert that does not suppress monitoring, so STRIDE-S validates that
the alert fires on attacks and stays quiet on genuine input.

| Threat | Mechanism | Experiment | v7 canonical target |
|---|---|---|---|
| **S** Spoofing | temporal behavioural PAD (blink+ocular) | Tier-C attack windows vs genuine; ISO 30107-3 | *(new: worst-species APCER = 0)* |
| **T** Tampering | SHA-256 + HMAC | 500 frames: flip 16 bytes; forge hash without key | 500/500 + 500/500 clean |
| **R** Repudiation | HMAC chain **+ DP-4** | 200 entries: tamper 50, delete, reorder, truncate, steal key | 200/200, 50/50, breaks |
| **I** Info disclosure | pixelation → 14×14 | 60 real DMD face crops through the guard | 0/60 recoverable, faces→0 |
| **D** DoS | sliding-window limiter | 180-frame flood in one window | 30 admitted, 150 throttled |
| **E** Elevation | RBAC 3×6 | 6 unauthorised + 4 authorised + full matrix | 6/6, 4/4, 18/18 |

**Two honest refinements over v7**, both stated in the results:
- **R / truncation.** Dropping the log tail leaves a cryptographically *valid prefix* — hash chaining
  alone cannot detect it (Schneier–Kelsey). It is caught only by the verifier's out-of-band anchor
  (expected entry count / final evolved key). Reported as such, not as a chain break.
- **R / DP-4.** Beyond v7's delete+reorder, the suite verifies forward integrity: an attacker who
  steals the *current* evolved key cannot re-sign a pre-compromise entry — the property v7 lacked.