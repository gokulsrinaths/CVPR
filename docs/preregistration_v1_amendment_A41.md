# Amendment A41 to pre-registration v1: zero-shot Qwen2.5-VL-3B, the size of our fine-tuned models

Appended 2026-10-12, before any run below. v1 and A0-A40 unchanged. Motivation: A40 tests the two routes to image-blind accuracy with zero-shot 7B (complete) and 32B (queued) models, but every fine-tuned system of the controlled experiments is a Qwen2.5-VL-3B. A zero-shot 3B run on the same six benchmarks with the same arms gives the direct before-and-after comparison for fine-tuning at one size, and does not wait for the 32B queue.

## What has been seen (not predictions)
A40 (`analysis/a40_interim.md`): zero-shot 7B black-image accuracy 27.7% (Med-CMR, chance 20.0), 57.2% (OmniMedVQA, 25.0), 36.1 (PMC-VQA), 46.2 (A-OKVQA), 23.5 (MedXpertQA-MM), 45.5 (MMMU health). Fine-tuned Qwen-3B black-image accuracy: 95.6% (Med-CMR, A9 FRESH3000; the TEST711 value is in the A9 rows), 93.2% (OmniMedVQA E1, A16), 38.1% (PMC-VQA E1, A11), 49.6% (A-OKVQA E1 with the original options at d = 0.25, A21). No zero-shot 3B result on these six benchmarks has been seen.

## Runs (`scripts/a41_zeroshot.py`, a copy of `a40_zeroshot.py` with the model choice 3b and the tag a41; `slurm/50_a41_zeroshot.sbatch`; `scripts/a41_analysis.py`; all hashed before the first submission)
Zero-shot Qwen2.5-VL-3B-Instruct, same test sets, prompt, readout and arms (true, black, told) as A40.

## Rules
- **Y1 (artifact route at 3B):** the blind ceiling (A36) exceeds the zero-shot 3B black-image accuracy by at least 30.0 points on both Med-CMR and OmniMedVQA.
- **Y2 (fine-tuning installs the artifact):** the black-image accuracy of the fine-tuned Qwen-3B exceeds that of the zero-shot Qwen-3B by at least 40.0 points on Med-CMR (both on the 711 held-out MCQs; the fine-tuned model is the A9 3B reference) and by at least 20.0 points on OmniMedVQA (both on E1).
Reported, not tested: zero-shot 3B true, black and told accuracy on all six benchmarks, true minus black, the same comparison for PMC-VQA and A-OKVQA, and the Spearman correlation of the blind-ceiling excess with the black-image excess.

## Reading rule (fixed)
- Y1 and Y2 hold: the paper states that at 3B, fine-tuning raises image-blind accuracy on Med-CMR and OmniMedVQA from near chance or moderate levels to the blind ceiling, i.e. the artifact is installed by training on the benchmark; either fails: the paper reports the values and drops the claim for that benchmark.
No run is repeated or tuned after seeing its result.

## Limits (stated in advance)
One model size and family; letter readout, not free generation; the fine-tuned and zero-shot black-image accuracies are on the same question sets but differ in prompt (the fine-tuned models were trained and evaluated with the medical system prompt and the zero-shot ones with the generic prompt).

## Amendments log
(A0-A41.)
