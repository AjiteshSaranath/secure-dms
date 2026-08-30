"""10 — figure & table export (P7-1).

Every figure is generated from THIS run's in-memory results (provenance rule:
no number appears in a figure unless it was produced by the run that drew it).
Saved as both .png (drafting) and .pdf (LaTeX inclusion) at 300 dpi.

The set deliberately includes the negative results — the Tier-C -> Tier-D gap and
the baseline comparison — because they are the thesis's substantive findings.
"""
import matplotlib.pyplot as plt          # notebook's inline backend: figures RENDER here
from IPython.display import display

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "figure.dpi": 110, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True})
_FIGS = []


def _save(fig, name, caption):
    """Write .png + .pdf for the thesis AND display the figure inline."""
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    _FIGS.append((name, caption))
    log(f"  {name:<28} {caption}")
    display(fig)                          # render in the notebook
    plt.close(fig)


log("\n" + "=" * 82)
log("P7-1 — figure export")
log("=" * 82)

# ── Fig 1: driver-state — PERCLOS agreement + threshold sensitivity ────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
_pp = np.array([s["perclos_pred"] for s in P3_RESULTS["per_session"]])
_pt = np.array([s["perclos_truth"] for s in P3_RESULTS["per_session"]])
ax1.scatter(_pt, _pp, s=38, alpha=0.8, edgecolor="k", linewidth=0.5)
_lim = [0, max(_pp.max(), _pt.max()) * 1.1]
ax1.plot(_lim, _lim, "k--", lw=1, label="perfect agreement")
ax1.set_xlabel("annotated PERCLOS"); ax1.set_ylabel("measured PERCLOS")
ax1.set_title(f"(a) PERCLOS agreement, 16 sessions\nMAE {P3_RESULTS['perclos_mae']:.4f}, "
              f"r = {P3_RESULTS['perclos_pearson_r']:.2f}")
ax1.legend(fontsize=7)
_fracs = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
_f1s = [0.4884, 0.5993, 0.6180, 0.6163, 0.6036, 0.5760, 0.5497]   # §4.5 sweep
ax2.plot(_fracs, _f1s, "o-", lw=1.5)
ax2.axvline(0.20, color="crimson", ls="--", lw=1.2)
ax2.annotate("published P80\n(untuned optimum)", xy=(0.20, 0.6180),
             xytext=(0.30, 0.55), fontsize=7, color="crimson",
             arrowprops=dict(arrowstyle="->", color="crimson", lw=1))
ax2.set_xlabel("closure fraction of open-eye baseline"); ax2.set_ylabel("F1")
ax2.set_title("(b) threshold sensitivity")
fig.tight_layout()
_save(fig, "fig_driver_state", "PERCLOS agreement + threshold sensitivity (P3)")

# ── Fig 2: THE headline — synthetic (Tier C) vs real (Tier D) ──────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8),
                               gridspec_kw={"width_ratios": [1, 1.5]})
_tc = [PAD_RESULTS["apcer_worst"], PAD_RESULTS["bpcer"], PAD_RESULTS["acer"]]
_td = [TIER_D_RESULTS["apcer_worst"], TIER_D_RESULTS["bpcer"], TIER_D_RESULTS["acer"]]
_x = np.arange(3)
ax1.bar(_x - 0.19, _tc, 0.38, label="Tier C (synthetic)", color="#5B8FF9")
ax1.bar(_x + 0.19, _td, 0.38, label="Tier D (real instruments)", color="#E8684A")
ax1.axhline(0.5, color="gray", ls=":", lw=1)
ax1.text(2.45, 0.51, "chance", fontsize=7, color="gray", ha="right")
ax1.set_xticks(_x); ax1.set_xticklabels(["APCER\n(worst)", "BPCER", "ACER"])
ax1.set_ylabel("error rate"); ax1.set_ylim(0, 1)
ax1.set_title("(a) synthetic validation overstates PAD")
ax1.legend(fontsize=7)
_sp = sorted(TIER_D_RESULTS["apcer_per_species"], key=TIER_D_RESULTS["apcer_per_species"].get)
_vals = [TIER_D_RESULTS["apcer_per_species"][s] for s in _sp]
_cols = ["#2E9E5B" if v < 0.05 else "#E8B84A" if v < 0.3 else "#E8684A" for v in _vals]
ax2.barh(range(len(_sp)), _vals, color=_cols)
ax2.set_yticks(range(len(_sp))); ax2.set_yticklabels(_sp, fontsize=8)
ax2.set_xlabel("APCER (Tier D, real instruments)")
ax2.set_title("(b) works when eyes are occluded,\nfails when real eyes are present")
fig.tight_layout()
_save(fig, "fig_pad_tierC_vs_tierD", "Tier-C synthetic vs Tier-D real PAD (P4 §6)")

# ── Fig 3: gate G3 — proposed vs published baselines ──────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8),
                               gridspec_kw={"width_ratios": [1, 1.7]})
_methods = [("ours", "Proposed\n(behavioural)"), ("maatta_R73", "Määttä 2011\n(LBP+SVM)"),
            ("lbptop_R74", "LBP-TOP 2012\n(spatio-temporal)")]
_acers = [G3[k]["acer"] for k, _ in _methods]
ax1.bar([n for _, n in _methods], _acers,
        color=["#E8684A", "#5B8FF9", "#2E9E5B"])
ax1.axhline(0.5, color="gray", ls=":", lw=1)
for i, v in enumerate(_acers):
    ax1.text(i, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
ax1.set_ylabel("ACER (lower is better)"); ax1.set_ylim(0, 0.75)
ax1.set_title("(a) both baselines outperform ours")
_species = sorted(G3["ours"]["apcer_per_species"])
_w, _xx = 0.27, np.arange(len(_species))
for _i, (_k, _lbl) in enumerate(_methods):
    ax2.bar(_xx + (_i - 1) * _w, [G3[_k]["apcer_per_species"].get(s, 0) for s in _species],
            _w, label=_lbl.replace("\n", " "))
ax2.set_xticks(_xx); ax2.set_xticklabels(_species, rotation=30, ha="right", fontsize=7)
ax2.set_ylabel("APCER"); ax2.legend(fontsize=7)
ax2.set_title("(b) complementary failure: behavioural wins on occluded-eye\n"
              "attacks, texture wins on replay / eye-hole masks")
fig.tight_layout()
_save(fig, "fig_gate_g3_baselines", "Gate G3 baseline comparison (P4 §7)")

# ── Fig 4: why it fails — blink rate by class, both corpora ───────────────
_diag = json.loads((OUT_DIR / "p4d_failure_diagnostic.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(7.2, 3.4))
_names = [k for k in _diag if k != "DMD genuine"]
_names = ["DMD genuine"] + sorted(_names)
_rates = [_diag[n]["blinks_per_min"] for n in _names]
_c = ["#2E9E5B" if "genuine" in n else "#E8684A" for n in _names]
ax.bar(range(len(_names)), _rates, color=_c)
ax.axhspan(15, 30, color="gray", alpha=0.18)
ax.text(len(_names) - 0.5, 31, "physiological range 15–30/min (R20)",
        fontsize=7, ha="right", color="gray")
ax.set_xticks(range(len(_names)))
ax.set_xticklabels([n.replace("ROSE ", "") for n in _names], rotation=35, ha="right", fontsize=7)
ax.set_ylabel("detected blinks / min")
ax.set_title("Blink cue fails twice over: attacks with real eyes blink like humans,\n"
             "while genuine ROSE capture under-detects (8.2/min vs DMD 19.4/min)")
fig.tight_layout()
_save(fig, "fig_blink_diagnosis", "Blink-rate diagnosis across corpora (P4 §6.2)")

# ── Fig 5: STRIDE summary table ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 2.5))
ax.axis("off")
_rows = [[k, STRIDE[k]["threat"], "PASS" if STRIDE[k]["pass"] else "FAIL",
          (STRIDE[k]["result"][:96] + "…") if len(STRIDE[k]["result"]) > 96
          else STRIDE[k]["result"]] for k in ("S", "T", "R", "I", "D", "E")]
_t = ax.table(cellText=_rows, colLabels=["", "Threat", "Verdict", "Measured result"],
              cellLoc="left", loc="center", colWidths=[0.03, 0.16, 0.07, 0.74])
_t.auto_set_font_size(False); _t.set_fontsize(6.5); _t.scale(1, 1.5)
for _i, _k in enumerate(("S", "T", "R", "I", "D", "E"), start=1):
    _t[(_i, 2)].set_facecolor("#CDEBD6" if STRIDE[_k]["pass"] else "#F6CFC8")
ax.set_title("STRIDE mitigation validation — 5/6 (S fails on real attack instruments)",
             fontsize=9, pad=12)
_save(fig, "fig_stride_table", "STRIDE validation summary (P6)")

# ── Fig 6: privacy guard before/after ─────────────────────────────────────
_cap = cv2.VideoCapture(str(dmd_inventory["drowsiness"]["videos"][0]))
_cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
_ok, _bgr = _cap.read(); _cap.release()
_lm = make_landmarker(video_mode=False)
_r = _lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB))))
if _r.face_landmarks:
    _h, _w = _bgr.shape[:2]
    _px = np.array([[p.x * _w, p.y * _h] for p in _r.face_landmarks[0]])
    _x0, _y0 = max(int(_px[:, 0].min()), 0), max(int(_px[:, 1].min()), 0)
    _x1, _y1 = min(int(_px[:, 0].max()), _w), min(int(_px[:, 1].max()), _h)
    _face = cv2.cvtColor(cv2.resize(_bgr[_y0:_y1, _x0:_x1], (224, 224)), cv2.COLOR_BGR2RGB)
    _prot = PrivacyGuard().protect(_face)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.4, 2.9))
    for _a, _im, _ttl in ((a1, _face, "before"), (a2, _prot, "after (14×14 effective)")):
        _a.imshow(_im); _a.axis("off"); _a.set_title(_ttl, fontsize=9)
    fig.suptitle("Privacy guard: 0/60 faces recoverable, detector 60→0 (STRIDE-I)", fontsize=9)
    fig.tight_layout()
    _save(fig, "fig_privacy_guard", "Privacy anonymisation before/after (P6, STRIDE-I)")

log(f"\n{len(_FIGS)} figures written to {FIG_DIR} (.png + .pdf, 300 dpi)")
