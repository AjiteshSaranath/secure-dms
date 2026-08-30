## 7. Secure pipeline assembly (P5)

7-stage order preserved (plan §1.3, v7 decision D12):
rate-limit -> integrity -> access -> PAD -> driver-state -> audit -> privacy.

**Fail-closed vs fail-safe (P5_FINDINGS §2).** The three input-validity stages (rate-limit, integrity,
access) hard-gate: a flooded, tampered, or unauthorised frame is dropped, anonymised, and audited.
**PAD does NOT hard-gate** — a not-live window raises an audited spoof alert but driver-state still
runs, because a hard PAD gate before the safety function dropped 83 % of a genuine session's frames.
Matches paper decision D14 (liveness as defence-in-depth, not a single point of failure).