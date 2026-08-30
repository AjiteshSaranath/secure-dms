"""Insert a new cell after an existing one, with source from a file.

Usage: python insert_cell.py <after_cell_id> <new_cell_id> <source_file> [code|markdown]
Cell type defaults to 'code'; pass 'markdown' as the 4th arg for a markdown cell.
"""
import sys
from pathlib import Path

import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")

after_id, new_id, src_path = sys.argv[1], sys.argv[2], Path(sys.argv[3])
cell_type = sys.argv[4] if len(sys.argv) > 4 else "code"
source = src_path.read_text(encoding="utf-8").rstrip("\n")

nb = nbformat.read(NB, as_version=4)
if any(c.get("id") == new_id for c in nb.cells):
    sys.exit(f"cell id '{new_id}' already exists — use inject_cell.py to replace it")

for i, cell in enumerate(nb.cells):
    if cell.get("id") == after_id:
        cell_new = (nbformat.v4.new_markdown_cell(source=source) if cell_type == "markdown"
                    else nbformat.v4.new_code_cell(source=source))
        cell_new["id"] = new_id
        nb.cells.insert(i + 1, cell_new)
        nbformat.write(nb, NB)
        print(f"inserted cell '{new_id}' after '{after_id}' "
              f"({len(source.splitlines())} lines); notebook now {len(nb.cells)} cells")
        break
else:
    sys.exit(f"anchor cell '{after_id}' not found")
