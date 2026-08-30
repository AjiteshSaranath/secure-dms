import re, json, os

PAPER = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\Paper_v9"
PROJ = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter"
g3 = json.load(open(os.path.join(PROJ, "outputs", "p4d_gate_g3.json")))

print("gate_g3.json structure:")
for k, v in g3.items():
    print(f"  {k}: {list(v.keys()) if isinstance(v, dict) else v}")

sec08 = open(os.path.join(PAPER, "main.tex"), encoding="utf-8").read()
tbl = sec08[sec08.index("label{tab:baselines}"):]
tbl = tbl[:tbl.index("end{tabular}")]

rows = {
    "Printed paper, still": "print_still",
    "Paper mask, uncut": "mask_full",
    "Printed paper, quivering": "print_quiver",
    "Display replay, Mac": "replay_mac",
    "Display replay, Lenovo": "replay_lenovo",
    "Paper mask, eyes cut": "mask_cropped",
    "Paper mask, upper cut": "mask_upper",
}
order = ["ours", "maatta_R73", "lbptop_R74"]   # paper column order
fails = []
print("\ntab:baselines — per-species APCER (ours | Maatta | LBP-TOP)")
for label, key in rows.items():
    line = [l for l in tbl.split("\n") if label in l]
    assert line, label
    paper = [float(x) for x in re.findall(r"\d\.\d{3}", line[0])]
    logv = [round(g3[m]["apcer_per_species"][key], 3) for m in order]
    ok = all(abs(a - b) < 0.0011 for a, b in zip(paper, logv))
    print(f"  {'OK ' if ok else 'BAD'} {label:<26} paper {paper}  log {logv}")
    if not ok:
        fails.append(label)

for metric, tag in (("bpcer", "BPCER"), ("acer", "ACER")):
    line = [l for l in tbl.split("\n") if l.strip().startswith(tag)]
    assert line, tag
    paper = [float(x) for x in re.findall(r"\d\.\d{3}", line[0])]
    logv = [round(g3[m][metric], 3) for m in order]
    ok = all(abs(a - b) < 0.0011 for a, b in zip(paper, logv))
    print(f"  {'OK ' if ok else 'BAD'} {tag:<26} paper {paper}  log {logv}")
    if not ok:
        fails.append(tag)

print("\n" + ("BASELINE TABLE MATCHES" if not fails else f"MISMATCHES: {fails}"))
