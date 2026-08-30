"""P7-2 — read a downloaded paper PDF and grep it for the numbers we quote.

Used to verify citations against the SOURCE rather than against secondary
summaries. Usage:
    python p7_read_paper.py <pdf> [regex ...]
Prints the paper's title block plus every line matching any regex, with context.
"""
import re
import sys
from pathlib import Path

import pdfplumber

pdf_path = Path(sys.argv[1])
patterns = sys.argv[2:] or [r"\d+\.\d+\s*%", r"HTER", r"EER"]

with pdfplumber.open(pdf_path) as pdf:
    pages = [p.extract_text() or "" for p in pdf.pages]

print(f"=== {pdf_path.name} — {len(pages)} pages ===\n")
print("--- first 25 lines (title/authors/venue) ---")
for line in pages[0].splitlines()[:25]:
    print(f"  {line}")

print("\n--- matches ---")
seen = set()
for pno, text in enumerate(pages, 1):
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                key = (pno, i)
                if key in seen:
                    continue
                seen.add(key)
                ctx = lines[max(0, i - 1):i + 2]
                print(f"\n[p{pno}] " + "\n        ".join(c.strip() for c in ctx))
                break
