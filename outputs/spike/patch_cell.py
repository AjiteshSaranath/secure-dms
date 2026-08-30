"""Targeted find/replace inside one notebook cell (safer than re-injecting).

Usage: python patch_cell.py <cell_id> <old_file> <new_file>
Both files hold the exact literal text to match and to substitute.
"""
import sys
from pathlib import Path

import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")

cell_id = sys.argv[1]
old = Path(sys.argv[2]).read_text(encoding="utf-8").rstrip("\n")
new = Path(sys.argv[3]).read_text(encoding="utf-8").rstrip("\n")

nb = nbformat.read(NB, as_version=4)
for cell in nb.cells:
    if cell.get("id") != cell_id:
        continue
    if old not in cell.source:
        sys.exit(f"pattern not found in cell '{cell_id}'")
    if cell.source.count(old) > 1:
        sys.exit(f"pattern appears {cell.source.count(old)}x in '{cell_id}' — ambiguous")
    cell.source = cell.source.replace(old, new)
    if cell.cell_type == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    nbformat.write(nb, NB)
    print(f"patched cell '{cell_id}' ({len(old.splitlines())} -> "
          f"{len(new.splitlines())} lines)")
    break
else:
    sys.exit(f"cell '{cell_id}' not found")
