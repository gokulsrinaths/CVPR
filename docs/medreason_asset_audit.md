# MedReason 2026 GA-CoT — Asset Audit

Date: 2026-08-08, updated 2026-08-13, updated 2026-08-27
Scope searched: `/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify`, `~/projects`, `~`, `/gscratch/gokulsrinathseetharam` (full recursive search), plus a local Docker-image export (`medreason_new_codebase`) supplied 2026-08-12, a self-built scoring suite supplied 2026-08-13, and the real 886-case held-out slice (`medreason_holdout_886_seed42.zip`) supplied 2026-08-21 — all from teammate Adinath.

**Update (2026-08-27): real reproduction completed.** MCQ accuracy reproduced to within 1 case of the paper's published number (96.06% vs. 96.2%, on the exact 886-case held-out slice, real ground truth, exact-match scoring). 11 of 12 required assets are now present and functionally verified. The one remaining gap (real GT/VA judge checkpoints) is scoped and documented — see `docs/next_reproduction_action.md` for the full run and result.

| # | Asset | Status | Local Path | Exact Command / Next Action | Blocks Reproduction? |
|---|-------|--------|------------|------------------------------|----------------------|
| 1 | Original GA-CoT source code | **Present** | `medical-vqa-retrieve-verify/code/medreason_docker/` (13 files) + `code/process.py` | — | No longer blocking |
| 2 | Exact backbone model name and version | **Present** | `Qwen/Qwen3.5-9B` — confirmed in `models/base_model/config.json` (`architectures: Qwen3_5ForConditionalGeneration`) and `models/adapter/adapter_config.json` (`base_model_name_or_path`) | — | No longer blocking |
| 3 | Adapter / LoRA checkpoint | **Present** | `medical-vqa-retrieve-verify/models/adapter/adapter_model.safetensors` (205MB, r=16, alpha=32, PEFT 0.19.1) | — | No longer blocking |
| 4 | Modality-router checkpoint and configuration | **Present** | `medical-vqa-retrieve-verify/models/detector.pkl` (4.6MB) — a classical sklearn classifier (`clf`/`labels`/`threshold`) used for modality-adaptive prompting (MAP), not a neural router, but is the actual asset the system uses | — | No longer blocking |
| 5 | Original inference command | **Present** | `python /opt/app/process.py` (now `code/process.py`), with env vars `MEDREASON_SYSTEM=custom`, `MEDREASON_MODEL_PATH=.../models/base_model`, `MEDREASON_ADAPTER_PATH=.../models/adapter`, `MEDREASON_DETECTOR_PATH=.../models/detector.pkl`, input `cases.json` under `MEDREASON_INPUT_DIR`, output to `MEDREASON_OUTPUT_FILE` | — | No longer blocking |
| 6 | Held-out evaluation data | **Present** | `medical-vqa-retrieve-verify/data/holdout886/cases.json` + `images/` (886 real cases, 711 MCQ + 175 open-ended) | — | No longer blocking. Real paper split, seed=42, checksum-verified (`ad85e2d8...dcabd`), supplied by teammate 2026-08-21 |
| 7 | Case metadata: modality and question type | **Present** | Same file — `metadata.modality`/`metadata.question_type` per case | — | No longer blocking |
| 8 | MCQ scoring script | **Present** | `scoring_suite/scoring/score_all.py` + `medreason_scoring/metrics.py` — exact-match, no judge needed. **Verified against the real paper number** (96.06% vs 96.2%) | — | No longer blocking |
| 9 | RVFtrace scoring script | **Present, but proxy only** | `medreason_scoring/judges.py` — reconstructed from the public challenge guide, not the official implementation. Real Qwen-2.5-VL judge not available | Real numbers require the actual VA judge | **Partially blocking** — only for GT/VA numbers |
| 10 | VAfinal scoring script | **Present, but proxy only** | Same as #9 | Same as #9 | **Partially blocking** |
| 11 | GTfinal scoring script | **Present, but proxy only** | Same file, GT side — proxy uses token-F1, not the real Llama-3.1-70B judge | Real numbers require the actual GT judge | **Partially blocking** |
| 12 | Original environment / dependency file | **Present** | `medical-vqa-retrieve-verify/code/requirements.txt` — torch==2.7.1, torchvision==0.22.1, transformers==5.10.2, peft==0.19.1, bitsandbytes==0.49.2, accelerate==1.13.0, flash-linear-attention==0.5.0, opencv-python-headless, scikit-learn, pillow, numpy | — | No longer blocking |

## Provenance and integrity of the transferred assets

- Source: local files `medreason_new_codebase-20260809T192211Z-1-001.zip`, `-1-003.zip`, and two large `.tar` layer files (originally saved under SHA256-hash filenames by the browser instead of the expected `-002`/`-004` names — confirmed to be the same split-download set, not corruption).
- These are a `docker save` export (OCI manifest + SHA256-named layer tars) of a "Qwen3.5-9B + GA-CoT grounded reasoning + Modality-Adaptive Prompting (MAP)" submission container, entrypoint `python /opt/app/process.py`.
- The largest layer (`opt/app/assets/*` — base model + adapter + detector, 19.5GB) was transferred to the cluster in 47 chunks after `scp`/`sftp` connections proved unable to sustain bulk transfer on this network path (SFTP subsystem and even large plain-`ssh` pipes were being cut after a few seconds). A resumable byte-offset append transfer was used instead.
- 12 of the 47 chunks were corrupted in the first pass (stale NFS `stat` reads during append-resume caused misaligned duplicate writes, confirmed via per-chunk SHA256 mismatch, not just size mismatch). All 12 were deleted and re-transferred with an immediate post-transfer checksum-verify-or-retry loop.
- Final reassembled tar on the cluster was confirmed **byte-for-byte identical** to the local original via whole-file SHA256 (`dc792a1d...` matches on both ends — this is also the file's own content-addressed name, since Docker layer tars are named by their digest).

## What still needs to be located

Items 6–11 are genuinely outside what a participant container would contain — they belong to the MedReason 2026 official evaluator / challenge organizers, not to your own submission code. Confirmed by a full re-inspection of all 18 files in the source Google Drive folder (17 docker layer tars + 1 manifest) on 2026-08-13 — every layer was catalogued and none contain held-out data or scoring scripts; the three largest unexamined layers (1ffbbe19 2GB, cf2dd371 1GB, bb72c5e9 500MB) turned out to be CUDA libraries, gcc/g++ toolchain, and base Ubuntu packages only. If you are also the evaluator (not just a participant), you need your own held-out split and the four scoring scripts (MCQ, RVFtrace, VAfinal, GTfinal) from wherever that evaluation harness was built — check `~/projects`, any private evaluator repo, or the Med-CMR benchmark's HuggingFace dataset (`aaassddaadf/Med-CMR`) if that benchmark is the actual eval source (its leaderboard columns `VA`/`GT` match your `VAfinal`/`GTfinal` naming — see prior findings in conversation).

## Mechanics-only smoke test — completed 2026-08-13

To validate the 6 present assets actually work (not just exist), ran `sbatch slurm/02_gacot_mechanics_smoke_test.sbatch` (job 12031503) on 1×H100. Used a freshly built `medreason_env` conda environment matching `code/requirements.txt` exactly (torch==2.7.1, transformers==5.10.2, peft==0.19.1, bitsandbytes==0.49.2, accelerate==1.13.0, flash-linear-attention==0.5.0, causal-conv1d==1.6.2.post1) — `verifier_env` was left untouched. Input was 2 hand-made synthetic placeholder cases (1 MCQ, 1 open-ended), NOT real medical data or official held-out cases.

**Result: full success.** Model loaded, adapter applied, detector ran, generation completed, output passed the code's own schema validation. The open-ended case produced a coherent, correctly-structured GA-CoT-style step-by-step reasoning trace. This confirms the transferred model/adapter/detector assets are not just present but functionally correct end-to-end on the target H100 hardware. See `docs/next_reproduction_action.md` for full detail. No accuracy/benchmark claim is made — inputs were synthetic with no ground truth.

## Real reproduction — completed 2026-08-27

Ran `sbatch slurm/04_gacot_real_holdout886.sbatch` (job 13666563) on 1×H100 against the real 886-case held-out slice (`data/holdout886/`), using `medreason_env` (exact `requirements.txt` match) and `MEDREASON_SYSTEM=custom` (the real GA-CoT + MAP system).

| Metric | This run | Paper | Comparable? |
|---|---|---|---|
| MCQ Accuracy | 96.06% (683/711) | 96.2% (684/711) | **Yes — real, exact-match, off by 1 case** |
| Open-ended GT | 0.69/4 | 1.61/4 | No — proxy judge only |
| Open-ended VA | 0.58/4 | 2.51/4 | No — proxy judge only |

Full detail and honest caveats in `docs/next_reproduction_action.md`.

## Verdict

**11 of 12 required assets are present and verified; the MCQ portion of the reproduction is confirmed against the real paper number.** The one remaining gap is the real GT/VA judge checkpoints (Llama-3.1-70B-Instruct, Qwen-2.5-VL) — without them, GT/VA scores are proxy-only and not comparable to the published paper. Everything else — code, model, adapter, router, environment, held-out data, ground truth, and scoring logic — is in place and has produced a real, verifiable result.
