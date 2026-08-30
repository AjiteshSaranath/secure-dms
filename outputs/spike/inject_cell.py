"""Replace a notebook cell's source from a .py file.

The master notebook has grown past what can be edited in one piece, so cell
bodies are authored as plain .py files and injected here. Usage:

    python inject_cell.py <cell_id> <source.py>
"""
import sys
from pathlib import Path

import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")

cell_id, src_path = sys.argv[1], Path(sys.argv[2])
source = src_path.read_text(encoding="utf-8").rstrip("\n")

nb = nbformat.read(NB, as_version=4)
for cell in nb.cells:
    if cell.get("id") == cell_id:
        cell["source"] = source
        if cell.cell_type == "code":
            # markdown cells must NOT carry these keys — nbformat rejects them
            cell["outputs"] = []
            cell["execution_count"] = None
        else:
            cell.pop("outputs", None)
            cell.pop("execution_count", None)
        nbformat.write(nb, NB)
        print(f"injected {len(source.splitlines())} lines into cell '{cell_id}'")
        break
else:
    ids = [c.get("id") for c in nb.cells]
    sys.exit(f"cell '{cell_id}' not found. available: {ids}")
