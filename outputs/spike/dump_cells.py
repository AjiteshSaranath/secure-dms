"""Dump the source of specific cells by id."""
import sys
from pathlib import Path
import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")
want = set(sys.argv[1:])
nb = nbformat.read(NB, as_version=4)
for c in nb.cells:
    if c.get("id") in want:
        print(f"===== {c['id']} ({c.cell_type}) =====")
        print(c.source)
        print()
