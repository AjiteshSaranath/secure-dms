"""Strip execution outputs from the master notebook.

The run_log_*.txt files are the provenance record (every reported number appears
there), so the notebook itself is kept output-free: it stays diffable, editable,
and forces `Run All` to regenerate results rather than displaying stale ones
(Rule et al. 2019, ledger R68).
"""
from pathlib import Path

import nbformat

NB = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\Secure_DMS_Rework.ipynb")

nb = nbformat.read(NB, as_version=4)
cleared = fixed = 0
for cell in nb.cells:
    if cell.cell_type == "code":
        if cell.get("outputs"):
            cleared += 1
        cell["outputs"] = []
        cell["execution_count"] = None
    else:
        # markdown cells must not carry these keys — nbformat rejects them.
        # Pop both unconditionally; `or` would short-circuit and skip the second.
        had = ("outputs" in cell) or ("execution_count" in cell)
        cell.pop("outputs", None)
        cell.pop("execution_count", None)
        if had:
            fixed += 1
nbformat.validate(nb)
nbformat.write(nb, NB)
print(f"cleared outputs from {cleared} code cells; "
      f"stripped invalid keys from {fixed} markdown cells; {len(nb.cells)} cells total")
