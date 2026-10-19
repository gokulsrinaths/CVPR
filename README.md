# CVPR 2027 submission: "When Text Locks Out Vision" (private working repository)

Measuring visual headroom in multiple-choice VQA. This repository is for internal collaboration. It is private. Do not make it public while the paper is under double-blind review.

## Contents

| Folder | What it holds | Notes |
|---|---|---|
| `paper/` | LaTeX sources of the main paper and supplement (`main.tex`, `supp.tex`, `sec/`, `fig/`, `main.bib`, `cvpr.sty`, `ieeenat_fullname.bst`, `preamble.tex`) | Compiles with pdflatex plus BibTeX. The `fig/` folder includes the illustrative Figure 1 (`f1_teaser_illustrative.png`), which contains invented text. |
| `scripts/` | Analysis and experiment-driver scripts, one per pre-registered amendment (for example `a17_*`, `a21_*`, `a36_*`, `a39_*`, `a44_*`) | Written for the Medicine Bow cluster. Paths point to `/gscratch/...` and need editing to run elsewhere. |
| `slurm/` | Slurm batch scripts used to run the experiments | Account and partition names are cluster-specific. |
| `docs/` | Pre-registration document and dated amendments, protocols, and status notes | The amendment chain is append-only. Do not edit an amendment after it has been hashed. |
| `code/` | `process.py` and `requirements.txt` | Included as received. Check before relying on them. |

## What is deliberately not in this repository

- **Benchmark data.** Med-CMR (through MedReason 2026), PMC-VQA, OmniMedVQA, A-OKVQA, MMMU, and MedXpertQA-MM are not included, and no question, answer, image, or derived subset is included.
- **Model weights and checkpoints.** The released adapter, the fine-tuned checkpoints, and the base models are not included.
- **Result records that contain benchmark item identifiers.** Per-item predictions and the files that reference individual items are excluded.
- **Teammate code.** A Docker-based MedReason system from a teammate is excluded. Its licence and permissions have not been confirmed.
- **Cache and environment folders.** `__pycache__`, virtual environments, and Hugging Face or pip caches are excluded.

Files that reference benchmark items were excluded in full: `docs/evaluation_rules_frozen.md`, `scripts/claim_check.py`, `scripts/len_rand.py`, and `scripts/phase2d_annotate.py`.

## Reproducing the data setup

1. Register on the MedReason 2026 challenge Synapse page and accept its terms. The released data is governed by CC BY-NC 4.0 (research use, with attribution).
2. Obtain PMC-VQA (CC BY-SA 4.0), A-OKVQA (repository licence Apache-2.0; confirm the dataset terms), and OmniMedVQA, MMMU, and MedXpertQA-MM from their own sources. Check each licence before use.
3. Place the data at the paths expected by the scripts. Paths are set inside the scripts and need editing.

## Status

- The paper is a CVPR 2027 submission candidate. It has not been submitted or accepted.
- Pending: the MedReason organisers' response on independent publication approval, and the CVPR 2027 LLM policy and template.
- Unverified: licences for OmniMedVQA, MMMU, and MedXpertQA-MM.

## Licence

Not chosen. Do not redistribute the code or paper until a licence is agreed.

## Citation

Withheld during double-blind review.
