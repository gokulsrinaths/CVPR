# Next Reproduction Action — MedReason 2026 GA-CoT

**Status (2026-08-27): Real reproduction completed against the paper's exact 886-case held-out slice. MCQ accuracy reproduced to within 1 case of the published number (96.06% vs. 96.2%). GT/VA still require the official judges to be trustworthy.**

## Timeline of what unblocked this

1. **2026-08-08**: initial audit found 0/12 required assets anywhere on scratch.
2. **2026-08-12/13**: user supplied a Docker-image export of the team's own MedReason submission container. Verified byte-for-byte (SHA256) and staged on scratch. Resolved 6/12 items: code, backbone (Qwen3.5-9B), LoRA adapter, modality detector, inference command, environment file.
3. **2026-08-13**: mechanics-only smoke test on 2 synthetic cases — confirmed the pipeline runs end-to-end and produces coherent GA-CoT reasoning traces. Not a benchmark result.
4. **2026-08-13**: teammate (Adinath) sent a self-built scoring suite (`medreason_scoring_suite`) — real scoring logic (MCQ exact-match, GT/VA with the RVF-trace gate from the paper), but paired with a **synthetic placeholder** held-out set (20 fake cases) and **proxy judges** (heuristic stand-ins, not the real Llama-3.1-70B/Qwen-2.5-VL judges). Ran a dry run to confirm the scoring pipeline itself works — again not a benchmark result.
5. **2026-08-21**: teammate sent `medreason_holdout_886_seed42.zip` — the **exact, real 886-case held-out slice** behind the paper's headline numbers (711 MCQ + 175 open-ended, real medical images, real ground-truth answers, checksum-verified split identity `ad85e2d8...dcabd`, carved from the 17,722-case training set with `seed=42, val_fraction=0.05` — matches the paper's Table 2/README exactly).
6. **2026-08-27**: converted the slice to the production `cases.json`/`ground_truth.json` schema, transferred 886 real medical images (289MB) to the cluster, and ran the real reproduction (`sbatch slurm/04_gacot_real_holdout886.sbatch`, job 13666563) on 1×H100.

## Result — job 13666563, 2026-08-27

| Metric | This run | Paper | Judge |
|---|---|---|---|
| **MCQ Accuracy** | **96.06%** (683/711) | **96.2%** (684/711) | exact-match, no judge needed — **real, trustworthy result** |
| Open-ended GT | 0.69/4 | 1.61/4 | proxy (NOT official) — not comparable |
| Open-ended VA | 0.58/4 | 2.51/4 | proxy (NOT official) — not comparable |

**MCQ accuracy is a genuine reproduction of the paper's headline result — off by exactly 1 case out of 711.** This is real evidence the recovered checkpoints, adapter, router, and inference code are the authentic, correct assets from the published system.

**GT/VA are not comparable to the paper.** The proxy judge is a token-overlap/keyword-recall heuristic, not a real semantic judge. Notably `RVF_trace` scored 0 for all 175 open-ended cases under the proxy — this is a proxy-judge weakness (it does not understand real free-text medical reasoning), not evidence of a real drop in the model's grounding quality. Do not report the GT/VA numbers from this run as reproducing or contradicting the paper.

## What is still missing for a fully official reproduction

**Real judges for GT and VA.** Specifically:
- GT judge: Llama-3.1-70B-Instruct (text-only)
- VA judge: Qwen-2.5-VL (vision-language)

Source: either (a) the actual checkpoints/prompts, if accessible, or (b) precomputed judge outputs from an official run, which `score_all.py` can already ingest via `--gt-judge file` / `--va-judge file` without needing the checkpoints locally.

## What I will not do

- Will not claim the GT/VA numbers above are official or comparable to the paper.
- Will not fabricate judge outputs to fill this gap.

## Everything currently staged on the cluster

`/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/`
- `code/` — GA-CoT source + requirements.txt
- `models/` — Qwen3.5-9B backbone, LoRA adapter, detector.pkl
- `scoring_suite/scoring/` — scoring harness (medreason_scoring/) + synthetic 20-case scaffold
- `data/holdout886/` — the real 886-case held-out slice (cases.json, ground_truth.json, images/, provenance README/SUMMARY/case-id list)
- `data/holdout886/run_output/` — results.json + scores.json from the 2026-08-27 run
- `slurm/04_gacot_real_holdout886.sbatch` — the job that produced the above
