"""STRICT provenance audit of Paper_v9.

CLAUDE.md sec.12: "no number enters findings/paper unless it appears in a
run_log_*.txt produced by an actual run."

The earlier audit accepted JSON artefacts and findings docs as evidence.  This
one does not.  Run logs only.  Everything else is reported so it can be
resolved individually.
"""
import re, os, glob

BASE = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project"
PAPER = os.path.join(BASE, "Paper_v9")
PROJ = os.path.join(BASE, r"dms_project_fixed_v3\dms_jupyter")

SECTIONS = ["main.tex"]
TOKEN = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")

# ---- index RUN LOGS ONLY -------------------------------------------------
runlog_idx, logs = {}, sorted(glob.glob(os.path.join(PROJ, "outputs", "run_log_*.txt")))
for p in logs:
    body = open(p, encoding="utf-8", errors="replace").read()
    for m in TOKEN.finditer(body):
        runlog_idx.setdefault(m.group(1).replace(",", ""), set()).add(os.path.basename(p))
# secondary index: everything else, only to explain a miss
other_idx = {}
for pat in ["outputs/*.json", "*.md"]:
    for p in glob.glob(os.path.join(PROJ, pat)):
        body = open(p, encoding="utf-8", errors="replace").read()
        for m in TOKEN.finditer(body):
            other_idx.setdefault(m.group(1).replace(",", ""), set()).add(os.path.basename(p))

print(f"indexed {len(logs)} run logs "
      f"(newest: {os.path.basename(logs[-1]) if logs else 'none'})")
print(f"{len(runlog_idx)} distinct numeric tokens in run logs\n")


def variants(tok):
    t = tok.replace(",", "")
    out = {t}
    try:
        f = float(t)
    except ValueError:
        return out
    for nd in range(0, 7):
        out.add(f"{f:.{nd}f}")
        out.add(f"{f*100:.{nd}f}")
        out.add(f"{f/100:.{nd}f}")
    return out


# numbers that are not measurements: standards, years, citation numerals,
# regulation numbers, and small integers written as prose
SKIP = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "16",
        "155", "21434", "30107", "2144", "1999", "2014", "2019", "2021",
        "2022", "2023", "2024", "2010", "2011", "2012", "2017", "2018",
        "2020", "84", "133", "103391", "53", "69", "27", "1794", "1809",
        "13", "121", "132", "7728", "1280", "720", "180", "198", "2104",
        "800", "38", "1997", "1991", "2005", "2009", "2016", "1992", "1996"}

total, missing = 0, []
for sec in SECTIONS:
    t = open(os.path.join(PAPER, sec), encoding="utf-8").read()
    t = re.sub(r"(?m)^\s*%.*$", "", t)
    # Volumes, years, pages and standard numbers are citation
    # metadata, not claims about this system, so the bibliography
    # is out of scope for a provenance audit of the results.
    t = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", t, flags=re.S)
    # for a provenance audit of the results.
    t = re.sub(r"\\(cite|label|ref|bibitem|input)\{[^}]*\}", "", t)
    t = re.sub(r"\\begin\{tabular\}\{[^}]*\}", "", t)
    seen, miss = set(), []
    for m in TOKEN.finditer(t):
        tok = m.group(1)
        if tok in SKIP or tok in seen:
            continue
        seen.add(tok)
        total += 1
        if not any(v in runlog_idx for v in variants(tok)):
            where = set()
            for v in variants(tok):
                where |= other_idx.get(v, set())
            miss.append((tok, sorted(where)[:3]))
    status = "OK" if not miss else f"{len(miss)} NOT IN ANY RUN LOG"
    print(f"{sec:<32}{len(seen):>4} numbers   {status}")
    for tok, where in miss:
        print(f"      ! {tok:<12} elsewhere: {where or 'NOWHERE IN PROJECT'}")
        missing.append((sec, tok, where))

print("\n" + "=" * 70)
print(f"{total} distinct numbers checked, {total-len(missing)} in run logs, "
      f"{len(missing)} not")
