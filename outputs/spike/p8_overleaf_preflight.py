"""Overleaf preflight for Paper_v8.

No LaTeX toolchain on this machine, so this catches the error classes a
compile would catch: undefined macros (package not loaded), dangling
refs/cites, unbalanced braces and environments, unescaped specials, and
siunitx leftovers.
"""
import re, os, sys
from collections import Counter

# Folder to check; defaults to the live (simplified) paper.
#   python p8_overleaf_preflight.py            -> Paper_v8_simple  (live)
#   python p8_overleaf_preflight.py <folder>   -> any other folder
BASE = r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project"
os.chdir(sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "Paper_v9"))
print("checking:", os.getcwd())
FILES = ["main.tex"]

# macro -> package that must be loaded for it
NEEDS = {
    r"\toprule": "booktabs", r"\midrule": "booktabs", r"\bottomrule": "booktabs",
    r"\cmidrule": "booktabs", r"\multirow": "multirow",
    r"\includegraphics": "graphicx", r"\resizebox": "graphicx",
    r"\text": "amsmath", r"\lor": "(kernel)", r"\circ": "(kernel)",
    r"\pm": "(kernel)", r"\SI": "siunitx", r"\num": "siunitx",
    r"\qty": "siunitx", r"\si": "siunitx", r"\ang": "siunitx",
    r"\url": "hyperref", r"\href": "hyperref",
    r"\citep": "natbib", r"\citet": "natbib", r"\textcolor": "xcolor",
    r"\toprule": "booktabs", r"\SIrange": "siunitx", r"\numrange": "siunitx",
    r"\degree": "siunitx", r"\percent": "siunitx", r"\celsius": "siunitx",
    r"\bm": "bm", r"\algorithmic": "algorithmicx", r"\tikz": "tikz",
}

src = {f: open(f, encoding="utf-8").read() for f in FILES}
preamble = src["main.tex"]
loaded = set()
for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", preamble):
    loaded |= {p.strip() for p in m.group(1).split(",")}
loaded |= {"(kernel)"}
print("packages loaded:", sorted(loaded - {"(kernel)"}))

def strip_comments(t):
    return re.sub(r"(?<!\\)%.*", "", t)

problems = []

# --- 1. macros needing an unloaded package -------------------------------
for f, t in src.items():
    body = strip_comments(t)
    for macro, pkg in NEEDS.items():
        if re.search(re.escape(macro) + r"(?![a-zA-Z])", body) and pkg not in loaded:
            problems.append(f"{f}: uses {macro} but '{pkg}' is not loaded")

# --- 2. refs / cites ------------------------------------------------------
labels, refs, cites, bibs = set(), [], [], set()
for f, t in src.items():
    b = strip_comments(t)
    labels |= set(re.findall(r"\\label\{([^}]*)\}", b))
    refs += [(f, r) for r in re.findall(r"\\ref\{([^}]*)\}", b)]
    cites += [(f, c) for c in re.findall(r"\\cite\{([^}]*)\}", b)]
    bibs |= set(re.findall(r"\\bibitem\{([^}]*)\}", b))
for f, r in refs:
    if r not in labels:
        problems.append(f"{f}: \\ref{{{r}}} has no \\label")
used = {k.strip() for _, c in cites for k in c.split(",")}
for k in sorted(used - bibs):
    problems.append(f"\\cite{{{k}}} has no \\bibitem")
for k in sorted(bibs - used):
    problems.append(f"WARN bibitem '{k}' never cited (IEEEtran prints it anyway)")
dupes = [l for l, n in Counter(
    l for f, t in src.items() for l in re.findall(r"\\label\{([^}]*)\}", strip_comments(t))
).items() if n > 1]
for d in dupes:
    problems.append(f"duplicate \\label{{{d}}}")

# --- 3. braces / environments / math ------------------------------------
for f, t in src.items():
    b = strip_comments(t)
    b2 = re.sub(r"\\[{}$&#_%]", "", b)          # drop escaped specials
    depth = 0
    for ch in b2:
        depth += (ch == "{") - (ch == "}")
    if depth:
        problems.append(f"{f}: brace imbalance {depth:+d}")
    be = Counter(re.findall(r"\\begin\{([^}]*)\}", b))
    en = Counter(re.findall(r"\\end\{([^}]*)\}", b))
    for k, v in (be - en).items():
        problems.append(f"{f}: \\begin{{{k}}} x{v} never closed")
    for k, v in (en - be).items():
        problems.append(f"{f}: \\end{{{k}}} x{v} never opened")
    if b2.count("$") % 2:
        problems.append(f"{f}: odd number of $ (unclosed math)")

# --- 4. unescaped specials outside math ---------------------------------
for f, t in src.items():
    for i, line in enumerate(t.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        code = re.sub(r"(?<!\\)%.*", "", line)
        code = re.sub(r"\$[^$]*\$", "", code)        # drop inline math
        code = re.sub(r"\\[%&#_$]", "", code)        # drop escaped
        # filename/key arguments are never typeset, so '_' in them is legal
        code = re.sub(r"\\(begin|end|label|ref|cite|input|includegraphics|"
                      r"IfFileExists|texttt|url|bibitem)"
                      r"(?:\[[^\]]*\])?\{[^}]*\}", "", code)
        for ch in ["_", "#"]:
            if ch in code:
                problems.append(f"{f}:{i}: unescaped '{ch}': {line.strip()[:70]}")
        if re.search(r"(?<!\\)&", code) and "\\\\" not in line and "tabular" not in line:
            pass  # & is legal in tables; too noisy to flag

# --- 5. table column counts ---------------------------------------------
for f, t in src.items():
    for m in re.finditer(r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}",
                         strip_comments(t), re.S):
        ncol = len(re.findall(r"[lcr]|p\{[^}]*\}", m.group(1)))
        for row in m.group(2).split("\\\\"):
            row = row.strip()
            if not row or row.startswith("\\") and "&" not in row:
                continue
            if "multicolumn" in row or "multirow" in row or "cmidrule" in row:
                continue
            n = len(re.split(r"(?<!\\)&", row))
            if n != ncol:
                problems.append(
                    f"{f}: tabular({ncol} cols) row has {n}: {row[:60]}")

print("\n" + "=" * 62)
if problems:
    for p in problems:
        print(("  ! " if not p.startswith("WARN") else "  ~ ") + p)
else:
    print("  no problems found")
print("=" * 62)
print(f"{len([p for p in problems if not p.startswith('WARN')])} error-class, "
      f"{len([p for p in problems if p.startswith('WARN')])} warnings")
