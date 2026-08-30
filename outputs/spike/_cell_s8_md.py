## 8. Evaluation — consolidated results (P7)

**Structural note (deviation from plan §8.2, deliberate).** The plan sketched a single Evaluation
section. In the implementation, evaluation lives **next to what it evaluates**, so it can be read
and re-run in context: driver-state metrics + DR-4 baselines + ablations in **§4.4/§4.6/§4.7**, PAD
ISO/IEC 30107-3 metrics + cue ablation in **§5.2**, and the STRIDE suite in **§9**. This section is
reserved for the **consolidated results summary table** to be assembled in P7 (results
consolidation), drawing those numbers together for the thesis.

**Baseline / no-training policy that governs all of the above (DR-4 + DP-8, Option C):** every
comparative baseline is a peer-reviewed published method re-implemented from source (no
previous-implementation result is used — DP-5 revoked). The deployed pipeline is 100 % training-free;
baselines are *evaluation instruments* fitted per their source papers on their own train splits,
never on test data, with published figures tabulated alongside as an external check (fallback A drops
that column if a protocol cannot be matched). A baseline outperforming the proposed system is an
acceptable, reportable outcome (D2, D9, D15, D16). PAD baselines (LBP-TOP R74, Määttä R73) are
deferred to Tier D — on synthetic warps they would measure the rendering, not the method.