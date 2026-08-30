"""List the notebook's cells so the split between notebook and helper scripts is auditable."""
from pathlib import Path

import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")

nb = nbformat.read(NB, as_version=4)
print(f"{'id':<16}{'type':<10}{'lines':>6}  first line")
print("-" * 88)
for c in nb.cells:
    src = c.source.splitlines()
    head = next((l for l in src if l.strip()), "")
    print(f"{c.get('id',''):<16}{c.cell_type:<10}{len(src):>6}  {head[:58]}")
print(f"\ntotal cells: {len(nb.cells)} "
      f"({sum(1 for c in nb.cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown)")
print(f"total code lines: {sum(len(c.source.splitlines()) for c in nb.cells if c.cell_type=='code')}")
