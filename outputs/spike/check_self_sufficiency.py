"""Is the notebook self-sufficient? Check cache coverage for every DMD session.

The validation cells skip any video without a cached signal file, so if the cache
is incomplete the notebook silently reports fewer sessions instead of failing.
This checks coverage explicitly.
"""
import hashlib
from pathlib import Path

DMD = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset")
CACHE = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter\outputs\signal_cache")
EXTRACTOR_VERSION = "p2.3"


def cache_path(video):
    key = f"{video.name}|{video.stat().st_mtime_ns}|{EXTRACTOR_VERSION}|None|1"
    return CACHE / f"{hashlib.sha256(key.encode()).hexdigest()[:16]}.npz"


total_missing = 0
for split in ("Drowsiness", "Distraction"):
    videos = sorted((DMD / split).rglob("*rgb_face.mp4"))
    missing = [v for v in videos if not cache_path(v).exists()]
    total_missing += len(missing)
    print(f"{split:<14} {len(videos) - len(missing):>3}/{len(videos):<3} cached"
          f"{'' if not missing else '  MISSING: ' + ', '.join(v.name[:18] for v in missing[:3])}")

files = list(CACHE.glob("*.npz"))
size_mb = sum(f.stat().st_size for f in files) / 1024 / 1024
stale = len(files) - (98 - total_missing) - 1   # -1 for the 300-frame smoke-test cache
print(f"\ncache: {len(files)} files, {size_mb:.0f} MB "
      f"({stale} from superseded extractor versions — inert but reclaimable)")
print(f"\nnotebook self-sufficient on THIS machine: {'YES' if total_missing == 0 else 'NO'}")
print("notebook self-sufficient on a FRESH machine: NO — the validation cells skip")
print("  uncached sessions, and nothing in the notebook extracts the full corpus.")
