## Pretrained model code — sources and provenance

Every model in this project is used **inference-only** (no training, no fine-tuning). The *code* that
defines and runs each model comes from the sources below; the *weight files* and their SHA-256 hashes
are recorded separately in `REFERENCES.md` and verified in §0 on every run. Full versions/commits are
pinned in `requirements.lock.txt`.

| Model / role | Paper | Code obtained from | Official? | Licence |
|---|---|---|---|---|
| **MediaPipe Face Landmarker** — face detection + 478 landmarks (DR-1) | R5, R6, R7 | PyPI `mediapipe==1.0.0` · repo <https://github.com/google/mediapipe> | Official (Google) | Apache-2.0 |
| **L2CS-Net** — gaze estimation (DR-3) | R29 | `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@4a0f978d` — **third-party fork** (the official README itself points here for pip install) | **No** — official (not pip-installable): <https://github.com/Ahmednull/L2CS-Net> | MIT |
| **6DRepNet** — head-pose estimation (DR-3) | R30 | PyPI `sixdrepnet==0.1.6` · repo <https://github.com/thohemp/6DRepNet> | **Official** — published by the paper's own author (T. Hempel, OvGU Magdeburg) | MIT |
| **RetinaFace** — face detector, used **only** for the §2.1 crop calibration and the §3.2 fidelity reference (not a cited method) | — | `pip install git+https://github.com/elliottzheng/face-detection@786fbab7` | Third-party packaging of RetinaFace | MIT |
| **PyTorch / TorchVision** — inference runtime | — | `--index-url https://download.pytorch.org/whl/cu128` (`torch==2.11.0+cu128`, `torchvision==0.26.0+cu128`) | Official | BSD-3-Clause |

**Weight-file download URLs** (integrity-checked in §0; hashes in `REFERENCES.md`):
- `face_landmarker.task` — `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task` (official Google)
- `L2CSNet_gaze360.pkl` — official Google-Drive folder is **dead (404)**; obtained from a commit-pinned
  Hugging Face mirror `https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models` (**open item —
  cross-verify the hash + diff the fork's `model.py` against the official repo at P7-2**)
- `6DRepNet_300W_LP_AFLW2000.pth` — `https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth`
  (the authors' institution — official)

> **Note on the L2CS-Net fork.** The authors' official repo is not pip-installable, so the
> `edavalosanaya` fork it recommends is used, **pinned to an exact commit**, loaded with
> `weights_only=True`, and its fast inference path was validated against the fork's *own* reference
> `Pipeline` on identical frames (§3.2: MAE 1.08°/1.68°, i.e. 16 % of L2CS-Net's own published error).
> L2CS is the only component whose code *and* weights both come from third-party mirrors — flagged for
> the P7-2 reference/provenance audit.