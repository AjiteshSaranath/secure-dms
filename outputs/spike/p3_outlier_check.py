"""P3 — is the modest PERCLOS correlation driven by one session?

gB/8/s5 predicts PERCLOS 0.037 against a ground truth of 0.158 — the largest
miss in the set, and the session with the lowest mean EAR (0.186). Hypothesis:
the P80 baseline is estimated from the session's own 85th-percentile EAR, so for
a driver whose eyes are narrow throughout, the "fully open" reference is itself
low and the closure threshold ends up too low to fire. This is a circularity in
relative PERCLOS, not a coding error.

Note the control that identifies the mechanism: gB/8/s5 is NOT the most drowsy
session — gC/12/s5 (truth 0.1757) and gB/10/s5 (0.1734) are both higher, and
gC/12/s5 is predicted accurately (0.1955) on an ordinary aperture of 0.257. So
high annotated closure does not by itself cause the miss; narrow aperture does,
which is what the error/mean-EAR correlation below measures.
"""
import numpy as np

sessions = [
    # name,        pred,   truth,  mean EAR (from extraction log)
    ("gA/1/s5",  0.1564, 0.1204, 0.287),
    ("gA/2/s5",  0.1017, 0.0779, 0.238),
    ("gA/3/s5",  0.1698, 0.0874, 0.258),
    ("gA/4/s5",  0.1075, 0.0937, 0.199),
    ("gA/5/s5",  0.2041, 0.1040, 0.217),
    ("gB/10/s5", 0.1737, 0.1734, 0.212),
    ("gB/10/s5", 0.1056, 0.1133, 0.223),
    ("gB/6/s5",  0.0491, 0.0325, 0.310),
    ("gB/7/s5",  0.1617, 0.1140, 0.194),
    ("gB/8/s5",  0.0368, 0.1576, 0.186),
    ("gB/9/s5",  0.1514, 0.1221, 0.221),
    ("gC/11/s5", 0.0838, 0.0750, 0.276),
    ("gC/12/s5", 0.1955, 0.1757, 0.257),
    ("gC/13/s5", 0.0790, 0.0307, 0.274),
    ("gC/14/s5", 0.0567, 0.0855, 0.227),
    ("gC/15/s5", 0.1114, 0.0809, 0.226),
]

names = [s[0] for s in sessions]
pred = np.array([s[1] for s in sessions])
truth = np.array([s[2] for s in sessions])
ear = np.array([s[3] for s in sessions])

r_all = np.corrcoef(pred, truth)[0, 1]
print(f"all 16 sessions:      Pearson r {r_all:+.4f}  MAE {np.mean(np.abs(pred-truth)):.4f}")

err = pred - truth
order = np.argsort(-np.abs(err))
print("\nlargest absolute errors:")
for i in order[:4]:
    print(f"  {names[i]:<10} pred {pred[i]:.4f} truth {truth[i]:.4f} "
          f"err {err[i]:+.4f}  mean EAR {ear[i]:.3f}")

keep = np.ones(len(pred), dtype=bool)
keep[order[0]] = False
r_drop1 = np.corrcoef(pred[keep], truth[keep])[0, 1]
print(f"\nexcluding {names[order[0]]}: Pearson r {r_drop1:+.4f} "
      f"MAE {np.mean(np.abs(pred[keep]-truth[keep])):.4f}")

# Does prediction error track the driver's overall eye aperture?
r_err_ear = np.corrcoef(err, ear)[0, 1]
print(f"\ncorrelation between signed PERCLOS error and session mean EAR: {r_err_ear:+.4f}")
print("(positive => sessions with WIDER eyes are over-predicted, narrower under-predicted,")
print(" which is the signature of baseline contamination rather than random noise)")

lo, hi = ear < np.median(ear), ear >= np.median(ear)
print(f"\nnarrow-eyed half (mean EAR < {np.median(ear):.3f}): mean signed error {err[lo].mean():+.4f}")
print(f"wide-eyed half:                          mean signed error {err[hi].mean():+.4f}")
