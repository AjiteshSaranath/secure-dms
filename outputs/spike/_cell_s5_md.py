## 5. Presentation attack detection (DR-3, P4)

Passive temporal behavioural PAD — blink evidence (ledger R23, R24), ocular variability over time
(R26, R28), non-rigid deformation (R25, R27) — decided per 10 s window; ISO/IEC 30107-3 metrics (R32).
Full results and caveats: [`P4_FINDINGS.md`](P4_FINDINGS.md). Deployed decision = **blink AND ocular**
(deformation cue computed but excluded — provisional, P4_FINDINGS §4.1).

**Attack data (DP-1 as modified, refined by the P1 domain-control rule — `P1_FINDINGS.md` §4.1):**
Tier A = DMD video (bona fide) · Tier B = NUAA stills (frame-level print attacks, evaluated against
the published single-frame baseline) · **Tier C = simulated rigid-planar presentations rendered
FROM DMD frames** (R25, R27) — *not* from NUAA. This corrects the literal DP-1 wording ("from NUAA
imposter frames"): rendering the attack from the same DMD footage as the bona-fide class keeps camera,
subject, lighting and resolution identical, so only the presentation physics differ. Sourcing from
NUAA (64×64 webcam) against DMD (720p) bona fide would reintroduce the capture-domain confound the
domain-control rule exists to prevent (the v7 NUAA F1=1.0 artefact, D3). A synthetic print/screen
appearance filter supplies the instrument texture. Tier D = Replay-Attack [R71] if access is granted.
No self-recorded media; simulated and real attacks are never pooled in reporting.

**Baselines (DR-4):** LBP-TOP [R74] and Määttä et al. [R73] — published methods, not the previous
implementation's gate. **Deferred to Tier D** (P4_FINDINGS §5): on synthetic warps of genuine frames
they would measure the rendering, not the method.