## 4. Driver-state module (DR-2)

Plan §4. Every measure below is a **published** one applied to the pretrained perception signals
from §§2–3. Nothing here is learned, fitted, or tuned — thresholds are the values stated in the
source papers (DP-3a); §4.5 reports how sensitive the results are to them.

### 4.1 Drowsiness

| Measure | Definition used | Threshold | Source |
|---|---|---|---|
| **PERCLOS (P80)** | fraction of a 60 s window in which eye aperture is ≤20 % of that driver's fully-open aperture | drowsy above 0.15 | Wierwille (R15); Dinges & Grace (R16) |
| **Blink** | contiguous closure run of 75–400 ms | — | Soukupová & Čech (R9); R20 |
| **Microsleep** | closure run beyond 400 ms | any occurrence ⇒ drowsy | Abe (R17) |
| **MAR** | mouth aspect ratio sustained ≥1 s | yawn above 0.60 | R14 |

**Why P80 is expressed relative to the driver's own open-eye baseline.** PERCLOS is defined as
percentage *eyelid closure*, i.e. aperture relative to the fully-open eye (R15, R16). Absolute EAR
varies with eye shape, eyewear and camera geometry, so an absolute cut cannot express "80 % closed".
The baseline is the 85th percentile of that session's own EAR — normalisation against the subject's
own signal, using **no labels and no ground truth**, so the no-training constraint holds.

**Why a 60 s window:** the FHWA validation defines PERCLOS over a one-minute observation period
(R16). DMD sessions run ~184 s; a sliding window keeps the measure continuously available as it
would be in deployment.

### 4.2 Visual distraction (DP-2a)

Scope is deliberately limited to **visual** distraction — where the driver is looking — because a
gaze/head-pose stack has no evidence about the hands. DMD's manual-action classes (phone, drinking,
hair) are reported separately rather than claimed.

| Measure | Threshold | Source |
|---|---|---|
| Gaze outside the forward cone | beyond ±30° yaw, ±20° pitch | gaze-zone literature (R19, R44) |
| Head deviation (fallback when gaze invalid) | same cone | Fridman et al. (R18); R44 |
| Off-road glance duration | >2 s single glance ⇒ unsafe | 2 s glance criterion (R44) |

Head pose substitutes when the gaze estimate is invalid (§3.4): the "owl vs lizard" finding (R18)
is that drivers redirect attention with both head and eyes, making head yaw an accepted coarse proxy
where the iris is not resolvable.

### 4.3 Output
Per session: `safe`/`unsafe` plus a reason code (`drowsy`, `visually_distracted`, or both) and the
contributing measures — richer and more auditable than a binary CNN score. The reason code is what
the §6 audit log records.

### 4.4 Baseline (DR-4)
`baseline_ear_consecutive()` implements the published simple alternative: an **absolute** EAR
threshold (0.21, R9) sustained over 20 consecutive frames, as deployed in the real-time EAR
drowsiness literature (R12, R13). It is deliberately distinct from our method — one cue, absolute
threshold, no subject normalisation, no windowing — so §8 can test whether the extra machinery earns
its complexity. Per DR-4, it is entirely possible that it does not; that would be reported as-is.

### 4.5 A bug this section caught, and the sensitivity evidence

The first P3 run scored recall 1.000 but precision 0.297, with PERCLOS over-predicted by +0.254 on
every session. The cause was an implementation error rather than a modelling one: the closure
criterion had been written as a blend of two *different* published thresholds —

```
ear <= (1 - P80) * baseline + EAR_absolute * P80        # incoherent
```

mixing Soukupová & Čech's **absolute** blink threshold (R9) with PERCLOS's **relative** P80
criterion (R15, R16). With a typical baseline of 0.30 that evaluates to ≈0.228, which sits at the
median open-eye EAR — so roughly half of all frames were labelled "closed". Corrected to the
published definition (`ear ≤ 0.20 × baseline`), pooled accuracy rises 0.745 → 0.908, F1 0.424 →
0.618, and the PERCLOS bias falls from +0.254 to +0.028.

The sweep below (`outputs/spike/p3_threshold_diagnostic.py`, 8 sessions, 43,923 frames) is reported
as sensitivity evidence, **not** as tuning — the deployed value stays the published 0.20:

| Closure fraction | Accuracy | Precision | Recall | F1 | PERCLOS bias |
|---|---|---|---|---|---|
| 0.10 | 0.9056 | 0.5833 | 0.4201 | 0.4884 | −0.0294 |
| 0.15 | 0.9112 | 0.5804 | 0.6194 | 0.5993 | +0.0076 |
| **0.20 (published P80)** | **0.9076** | **0.5550** | **0.6970** | **0.6180** | **+0.0278** |
| 0.25 | 0.9009 | 0.5269 | 0.7421 | 0.6163 | +0.0442 |
| 0.30 | 0.8918 | 0.4970 | 0.7684 | 0.6036 | +0.0590 |
| 0.50 | 0.8542 | 0.4109 | 0.8300 | 0.5497 | +0.1099 |

Two points worth making to an examiner. First, **the published P80 value lands at the F1 optimum**
without any tuning — the strongest possible evidence for DP-3a's "published defaults only" policy.
Second, varying the open-eye baseline percentile from 75 to 99 moves F1 only between 0.617 and 0.621,
so the criterion is not knife-edge on that choice either.