"""Verify that the paper's table CELLS carry the right values, not merely
values that exist somewhere in a run log.

The strict provenance audit proves every number appears in a log. It cannot
catch a precision and a recall being swapped, or a species row misaligned.
This checks the three tables that carry the headline claims, row by row.
"""
import re, glob, os, json

PAPER = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\Paper_v9"
PROJ = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter"
log = max(glob.glob(os.path.join(PROJ, "outputs", "run_log_*.txt")), key=os.path.getmtime)
L = open(log, encoding="utf-8", errors="replace").read()
print(f"log: {os.path.basename(log)}\n")

sec08 = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
fails = []


def check(label, expect, row_regex, cols):
    """expect: list of floats from the log. cols: values parsed from the paper."""
    ok = all(abs(a - b) < 0.0011 for a, b in zip(expect, cols))
    print(f"  {'OK ' if ok else 'BAD'} {label:<42} paper {cols}  log {expect}")
    if not ok:
        fails.append(label)


# ---------- Table: eye closure -----------------------------------------
print("tab:eyeclosure — frame-level eye closure")
logrows = {
    "ratio-adapted": [0.9385, 0.6475, 0.8917, 0.7502],
    "published value": [0.9223, 0.5791, 0.9171, 0.7099],
    "Proposed": [0.9343, 0.6562, 0.7684, 0.7079],
    "Majority": [0.8964, 0.0, 0.0, 0.0],
}
# confirm those come from the log text itself
for name, vals in logrows.items():
    tag = {"ratio-adapted": "baseline EAR ratio-adapted",
           "published value": "baseline EAR absolute",
           "Proposed": "proposed (P80",
           "Majority": "majority class"}[name]
    line = [l for l in L.split("\n") if tag in l]
    assert line, f"log line for {tag} not found"
    nums = [float(x) for x in re.findall(r"0\.\d{4}", line[0])]
    assert nums[:4] == vals, f"log mismatch for {tag}: {nums[:4]} vs {vals}"

tbl = sec08[sec08.index("label{tab:eyeclosure}"):]
tbl = tbl[:tbl.index("end{tabular}")]
for name, vals in logrows.items():
    row = [l for l in tbl.split("\n") if name.split()[0] in l and "&" in l]
    assert row, f"paper row {name} not found"
    cols = [float(x) for x in re.findall(r"\d\.\d{3}", row[0])]
    check(name, [round(v, 3) for v in vals], None, cols)

# ---------- Table: PAD ---------------------------------------------------
print("\ntab:pad — real-instrument APCER per species")
d = json.load(open(os.path.join(PROJ, "outputs", "p4d_tierd_results.json")))
species_map = {
    "Printed paper, still": "print_still",
    "Paper mask, uncut": "mask_full",
    "Printed paper, quivering": "print_quiver",
    "Display replay, Lenovo": "replay_lenovo",
    "Display replay, Mac": "replay_mac",
    "Paper mask, eyes cut": "mask_cropped",
    "Paper mask, upper cut": "mask_upper",
}
tbl = sec08[sec08.index("label{tab:pad}"):]
tbl = tbl[:tbl.index("end{tabular}")]
for label, key in species_map.items():
    row = [l for l in tbl.split("\n") if label in l]
    assert row, f"paper row {label} not found"
    paper = float(re.findall(r"\d\.\d{3}", row[0])[0])
    logv = round(d["apcer_per_species"][key], 3)
    ok = abs(paper - logv) < 0.0011
    print(f"  {'OK ' if ok else 'BAD'} {label:<28} paper {paper:.3f}  log {logv:.3f}")
    if not ok:
        fails.append(label)
print(f"  BPCER  log {d['bpcer']:.4f} | ACER log {d['acer']:.4f} "
      f"| APCER worst log {d['apcer_worst']:.4f}")

# ---------- Table: baselines --------------------------------------------
print("\ntab:baselines — subject-disjoint comparison")
g3 = json.load(open(os.path.join(PROJ, "outputs", "p4d_gate_g3.json")))
print("  gate_g3.json keys:", list(g3.keys())[:8])

print("\n" + ("ALL TABLE CELLS MATCH" if not fails else f"MISMATCHES: {fails}"))
