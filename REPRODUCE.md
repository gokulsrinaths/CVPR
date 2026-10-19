# Reproducing the results

This guide explains how to rebuild the paper's results from this repository. The repository holds code, Slurm drivers, pre-registration documents, and paper sources. It does not hold benchmark data, model weights, or per-item outputs. Each of those must be obtained separately, as described below.

## 1. What you need before starting

| Item | Source | Access |
|---|---|---|
| Med-CMR (MedReason 2026 release) | Official MedReason challenge, Synapse | Synapse account and challenge registration. Released data is CC BY-NC 4.0 (research use, with attribution). |
| Med-CMR public mirror used by `scripts/fetch_public_medcmr.py` | Hugging Face dataset `aaassddaadf/Med-CMR` | Public download. Its terms are not verified. Prefer the official MedReason release. |
| PMC-VQA | Its own release (Hugging Face card states CC BY-SA 4.0) | Public. Check the card before use. |
| OmniMedVQA, A-OKVQA, MMMU, MedXpertQA-MM | Each dataset's own release | Public. Check each licence. A-OKVQA's repository is Apache-2.0; confirm it covers the data files. |
| Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct | Hugging Face `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct` | Public. |
| InternVL3-2B | Hugging Face (OpenGVLab) | Public. |
| Released MedReason adapter (Qwen3.5-9B, 4-bit LoRA) | The challenge's released container | Needs the challenge materials. The Docker-based MedReason system is not in this repository. |
| Compute | Slurm cluster with H100 or A30 GPUs | The Slurm files assume a cluster with `effectorfold` account and `mb-h100` partition. Edit for your cluster. |

## 2. Environment

Install the pinned packages in `code/requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
```

The file pins `torch==2.7.1`, `transformers==5.10.2`, `peft==0.19.1`, and `bitsandbytes==0.49.2`, among others. These versions are as recorded for the original runs. Check that they install on your platform before running experiments.

## 3. Paths

Scripts and Slurm files contain absolute paths under `/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify/`. Before running anything:

1. Choose a project root on your system.
2. Replace the prefix in every `scripts/*.py`, `slurm/*.sbatch`, and `docs/*.md` path with your root.
3. Create `logs/`, `data/`, and `models/` inside the root.

Search for the prefix with `grep -rl "/gscratch/gokulsrinathseetharam/medical-vqa-retrieve-verify" .` to find every occurrence.

## 4. Data

1. Obtain each dataset from its official source (section 1).
2. Place Med-CMR under `data/public_medcmr/`. The fetch script writes there.
3. Build the derived splits with the build scripts (`scripts/a21_build.py`, `scripts/omni_build.py`, `scripts/pmc_build.py`, `scripts/a37_build.py`, `scripts/ss_build_split.py`). Check each build script header for its inputs and outputs.

The held-out 711 questions come from the official MedReason split. Confirm the split against the frozen manifests in `docs/` before trusting any number.

## 5. Model weights

Base models download automatically from Hugging Face when a script loads them. Use `scripts/fetch_qwen25vl.py` and `scripts/fetch_qwen25vl3b.py` to pre-download and verify.

The released MedReason adapter cannot be obtained from this repository. Without it, the A8, A12, and A39 analyses of the released system cannot be rerun. The analyses of our own fine-tuned models do not depend on it.

## 6. Run order

The pre-registration in `docs/` defines each experiment and its amendment. Run them in amendment order:

1. Fine-tuning studies: `scripts/ft_run*.py`, submitted through `slurm/16_ft.sbatch` onward.
2. Dose-response and phase experiments: `scripts/dose_*.py`, `scripts/a17_*`, `scripts/a21_*` through `scripts/a25_*`.
3. Intervention experiments: `scripts/a12_*`, `scripts/a18_*`, `scripts/a20_*`, `scripts/a24_*`, `scripts/a26_*`.
4. Blind ceiling and headroom: `scripts/a36_blind.py`, `scripts/a36_countercue.py`.
5. Released-system analyses: `scripts/a39_released.py`, `scripts/a41_*`, `scripts/a44_*`.
6. Analysis summaries: `scripts/*_analysis.py` and `scripts/*_summarize.py`.

Each Slurm file sets its own time limit and GPU request. Submit with `sbatch slurm/<file>.sbatch`.

Do not change a script after its amendment has been hashed. The hash chain in `docs/` records which version of each script governed which run.

## 7. What you should get

The paper's numbers come from analysis output files, which are not in this repository. Expect these checks to pass if the setup is correct:

- Headroom for Med-CMR, OmniMedVQA, PMC-VQA, and A-OKVQA: −0.1, +4.9, +13.7, +33.2 points (A36).
- PMC-VQA image benefit falls from +13.5 to +0.1 points across doses 0.25 to 0.92 (A17).
- Counter-cue accuracy on 206 FRESH questions: 68.0% (Qwen-3B), 73.3% (Qwen-7B) (A36).

Small differences from sampling and hardware are expected. Compare to within the reported intervals.

## 8. Verifying your copy of excluded files

Once a checksum list of the excluded files is added, compare your downloads against it to confirm they match the original records. The list will contain names and hashes only.

## 9. Known gaps

- Model checkpoints from our own fine-tuning are not in the repository. Retraining reproduces them only approximately.
- Some scripts reference the released MedReason container, which is not available here.
- The Hugging Face Med-CMR mirror's licence is not verified. Use the official release where possible.
- Licence for OmniMedVQA, MMMU, and MedXpertQA-MM is not verified.
- No licence has been chosen for the repository. Do not redistribute until one is agreed.
