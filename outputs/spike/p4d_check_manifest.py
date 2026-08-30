"""Summarise the Tier-D extraction manifest: coverage and per-species detection."""
import collections
import json
from pathlib import Path

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
m = json.loads((PROJ / "outputs" / "p4d_rose_manifest.json").read_text(encoding="utf-8"))

agg = collections.defaultdict(lambda: {"clips": 0, "detected": 0, "read": 0, "subjects": set()})
for it in m["items"]:
    a = agg[it["species"]]
    a["clips"] += 1
    a["detected"] += it["detected"]
    a["read"] += it["read"]
    a["subjects"].add(it["subject"])

print(f"extracted {m['extracted']} clips, {m['failed']} failed (seed {m['seed']})\n")
print(f"{'species':<16}{'clips':>6}{'subj':>6}{'detected':>10}{'read':>10}{'rate':>8}")
for sp, a in sorted(agg.items()):
    print(f"{sp:<16}{a['clips']:>6}{len(a['subjects']):>6}{a['detected']:>10,}"
          f"{a['read']:>10,}{a['detected']/max(a['read'],1):>8.1%}")

tot_det = sum(a["detected"] for a in agg.values())
tot_read = sum(a["read"] for a in agg.values())
print(f"\ntotal {tot_det:,}/{tot_read:,} frames with a detected face ({tot_det/tot_read:.1%})")

# how many 5 s windows will each species yield? (assume ~25 fps => 125 frames/window)
print(f"\n{'species':<16}{'est. 5s windows (detected frames / 125)':>44}")
for sp, a in sorted(agg.items()):
    print(f"{sp:<16}{a['detected'] // 125:>44,}")
