"""P4 step 1 — Tier C: synthesise rigid-planar presentation attacks.

Kollreider et al. [R25] and Anjos et al. [R27] define a photo presentation attack
as a PLANAR surface undergoing RIGID motion in front of the camera: every facial
point shares one global transform, and the face does not deform. Rendering that
transform is implementing the published attack model, not inventing one — and the
project has precedent, since v7's `evaluation/attack_simulator.py` generated its
STRIDE attacks programmatically.

Domain control (P1_FINDINGS §4.1): the attack is rendered FROM the same DMD
footage that supplies the bona-fide class, so capture domain, subject, lighting
and camera are identical and only the presentation physics differ. That is the
property external corpora cannot give us, and it is why this tier is worth having
even if Replay-Attack arrives.

Attack instrument species generated (reported separately, per ISO/IEC 30107-3 [R32]):
  print_static   — printed photo held nearly still (small tremor only)
  print_handheld — printed photo with visible hand motion, translation + rotation
  screen_replay  — photo displayed on a screen: adds moire-like pattern, gamma
                   shift and a slight specular gradient, still rigid-planar

Each is written as an .mp4 so the PAD module consumes it through exactly the same
video path as genuine footage — no special-casing that could leak the label.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Adaptive Cybersecurity Project\dms_project_fixed_v3\dms_jupyter")
DMD_DROW = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Drowsiness")
DMD_DIST = Path(r"C:\Users\Ajitesh\Documents\Galway Notes\Case Study in Cybersecurity Analytics\DMD Dataset\Distraction")
OUT_DIR = PROJ / "outputs" / "tier_c_attacks"
SEED = 42
FPS = 29.76
# Clips must be comfortably LONGER than one PAD window (10 s), otherwise a clip
# that renders exactly one window's worth yields zero windows: the container's
# reported fps can differ slightly from the render fps, so the window size comes
# out a frame or two larger than the frame count. 20 s gives two clean windows
# per clip and removes the off-by-one entirely.
DURATION_S = 20.0
SPECIES = ("print_static", "print_handheld", "screen_replay")


def source_frame(video: Path, rng) -> np.ndarray | None:
    """Grab one frame from a session — the 'photograph' the attacker holds."""
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng.integers(int(n * 0.2), int(n * 0.8))))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def rigid_trajectory(n_frames: int, species: str, rng) -> list:
    """Per-frame (dx, dy, angle, scale) for a planar object held by a hand.

    Modelled as slow drift plus physiological hand tremor. Tremor is ~8-12 Hz in
    the literature on postural hand tremor; amplitudes below differ by species to
    represent a photo braced against a surface vs held freely vs a phone/tablet.
    """
    cfg = {                    # drift_px, tremor_px, drift_deg, tremor_deg, scale_amp
        "print_static":   (2.0, 0.4, 0.6, 0.15, 0.004),
        "print_handheld": (14.0, 1.8, 4.0, 0.8, 0.030),
        "screen_replay":  (8.0, 1.1, 2.2, 0.5, 0.018),
    }[species]
    drift_px, tremor_px, drift_deg, tremor_deg, scale_amp = cfg

    t = np.arange(n_frames) / FPS
    # low-frequency drift: a couple of independent slow sinusoids per axis
    def drift(amp):
        f1, f2 = rng.uniform(0.05, 0.15, 2)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        return amp * (0.6 * np.sin(2 * np.pi * f1 * t + p1)
                      + 0.4 * np.sin(2 * np.pi * f2 * t + p2))

    def tremor(amp):
        f = rng.uniform(8.0, 12.0)          # physiological hand tremor band
        return amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))

    dx = drift(drift_px) + tremor(tremor_px)
    dy = drift(drift_px * 0.7) + tremor(tremor_px * 0.8)
    ang = drift(drift_deg) + tremor(tremor_deg)
    scale = 1.0 + drift(scale_amp) / max(scale_amp, 1e-9) * scale_amp
    return list(zip(dx, dy, ang, scale))


def apply_instrument(img: np.ndarray, species: str, rng) -> np.ndarray:
    """Appearance of the attack instrument itself (before motion)."""
    out = img.astype(np.float32)
    if species.startswith("print"):
        # print: slight blur from paper + ink, mild contrast loss, paper grain
        out = cv2.GaussianBlur(out, (3, 3), 0.8)
        out = (out - 128) * 0.92 + 128
        out += rng.normal(0, 2.0, out.shape).astype(np.float32)
    else:
        # screen: gamma shift, faint scan-line/moire pattern, specular gradient
        out = 255.0 * np.power(np.clip(out / 255.0, 0, 1), 1.12)
        h, w = out.shape[:2]
        rows = np.arange(h).reshape(-1, 1, 1)
        out += 3.0 * np.sin(2 * np.pi * rows / rng.uniform(3.0, 5.0))
        gx = np.linspace(-1, 1, w).reshape(1, -1, 1)
        out += 6.0 * np.clip(gx, 0, None) ** 2
    return np.clip(out, 0, 255).astype(np.uint8)


def render(frame: np.ndarray, species: str, out_path: Path, rng) -> int:
    h, w = frame.shape[:2]
    n_frames = int(DURATION_S * FPS)
    instrument = apply_instrument(frame, species, rng)
    traj = rigid_trajectory(n_frames, species, rng)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for dx, dy, ang, scale in traj:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(ang), float(scale))
        m[0, 2] += float(dx)
        m[1, 2] += float(dy)
        # BORDER_REPLICATE keeps the frame full — a black border would be a
        # trivial giveaway unrelated to liveness.
        warped = cv2.warpAffine(instrument, m, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        writer.write(warped)
    writer.release()
    return n_frames


def main(n_per_species: int = 12):
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(DMD_DROW.rglob("*rgb_face.mp4")) + sorted(DMD_DIST.rglob("*rgb_face.mp4"))
    # Use distinct source sessions so no subject appears in two species by accident
    picks = rng.permutation(len(videos))[: n_per_species * len(SPECIES)]
    manifest = []
    k = 0
    for species in SPECIES:
        for i in range(n_per_species):
            src = videos[picks[k]]
            k += 1
            frame = source_frame(src, rng)
            if frame is None:
                continue
            name = f"{species}_{i:02d}.mp4"
            n = render(frame, species, OUT_DIR / name, rng)
            manifest.append({"file": name, "species": species,
                             "source_session": "/".join(src.parts[-4:-1]),
                             "source_video": src.name, "frames": n})
            print(f"  {name:<24} {n:>4} frames  from {src.parts[-4]}/{src.parts[-3]}", flush=True)

    (OUT_DIR / "manifest.json").write_text(json.dumps(
        {"seed": SEED, "fps": FPS, "duration_s": DURATION_S,
         "attack_model": "rigid-planar presentation (R25, R27)",
         "domain_control": "rendered from DMD frames — same camera/subjects as bona fide",
         "items": manifest}, indent=2), encoding="utf-8")
    print(f"\nwrote {len(manifest)} attack videos + manifest to {OUT_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
