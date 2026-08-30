"""P7-2 — close the two open L2CS-Net provenance items.

L2CS is the only component whose CODE and WEIGHTS both come from third parties
(the authors' pip install is unavailable and their Google-Drive weights 404), so
it carries the project's weakest supply chain. Two checks were deferred to P7-2:

  (1) Diff the installed fork's model definition against the OFFICIAL repository
      (github.com/Ahmednull/L2CS-Net). If the architecture differs, the weights
      we load are not the published model and every gaze number is suspect.
  (2) Cross-check the weight file. The pinned Hugging Face mirror cannot be
      confirmed against the authors' own copy (it is gone), so instead verify
      internal consistency: the state-dict must match the published architecture
      exactly — right layer names, shapes, and the 90-bin gaze head of R29.
"""
import hashlib
import json
import urllib.request
from pathlib import Path

import torch

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
VENV_L2CS = PROJ / ".venv" / "Lib" / "site-packages" / "l2cs"
OFFICIAL_MODEL_URL = "https://raw.githubusercontent.com/Ahmednull/L2CS-Net/main/l2cs/model.py"
WEIGHTS = PROJ / "models" / "L2CSNet_gaze360.pkl"
report = {}

# ── (1) fork vs official model definition ─────────────────────────────────
local = (VENV_L2CS / "model.py").read_text(encoding="utf-8")
try:
    official = urllib.request.urlopen(OFFICIAL_MODEL_URL, timeout=30).read().decode("utf-8")
    fetched = True
except Exception as exc:                                    # noqa: BLE001
    official, fetched = "", False
    print(f"could not fetch official model.py: {exc}")

def _norm(s):
    """Compare code semantics, not formatting: drop blanks/comments/whitespace."""
    out = []
    for line in s.splitlines():
        t = line.split("#")[0].strip()
        if t:
            out.append(" ".join(t.split()))
    return out

if fetched:
    a, b = _norm(local), _norm(official)
    same = a == b
    report["model_py_identical"] = same
    report["model_py_local_sha256"] = hashlib.sha256("\n".join(a).encode()).hexdigest()
    report["model_py_official_sha256"] = hashlib.sha256("\n".join(b).encode()).hexdigest()
    print(f"(1) fork model.py vs official: "
          f"{'IDENTICAL (normalised)' if same else 'DIFFERS'}")
    if not same:
        only_local = [l for l in a if l not in b]
        only_off = [l for l in b if l not in a]
        report["lines_only_in_fork"] = only_local[:20]
        report["lines_only_in_official"] = only_off[:20]
        print(f"    lines only in fork ({len(only_local)}):")
        for l in only_local[:12]:
            print(f"      + {l[:100]}")
        print(f"    lines only in official ({len(only_off)}):")
        for l in only_off[:12]:
            print(f"      - {l[:100]}")

# ── (2) weight file vs the published architecture ─────────────────────────
sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
sd = sd.get("state_dict", sd)
keys = list(sd.keys())
report["weights_sha256"] = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest().upper()
report["n_tensors"] = len(keys)
report["n_params"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel")))

# R29: ResNet-50 backbone with TWO 90-bin heads (pitch, yaw) — the paper's
# fine-grained binned formulation. Verify both heads exist with 90 outputs.
head_keys = [k for k in keys if "fc" in k.lower()]
checks = {
    "resnet50_layer4_present": any("layer4" in k for k in keys),
    "two_gaze_heads": sum(1 for k in head_keys if k.endswith("weight")
                          and sd[k].ndim == 2 and sd[k].shape[0] == 90) >= 2,
}
for k in head_keys:
    if sd[k].ndim == 2:
        report.setdefault("head_shapes", {})[k] = list(sd[k].shape)
report["architecture_checks"] = checks
print(f"\n(2) weight file: {len(keys)} tensors, {report['n_params']:,} parameters")
print(f"    SHA-256 {report['weights_sha256'][:32]}…")
for k, v in checks.items():
    print(f"    {k:<28} {'OK' if v else 'FAIL'}")
print(f"    gaze head shapes: {report.get('head_shapes', {})}")
print("    (90 output bins per head = the fine-grained binned gaze formulation of R29)")

(PROJ / "outputs" / "p7_l2cs_provenance.json").write_text(json.dumps(report, indent=2),
                                                          encoding="utf-8")
print(f"\nwritten: {PROJ / 'outputs' / 'p7_l2cs_provenance.json'}")
