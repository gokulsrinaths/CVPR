# Amendment A40 to pre-registration v1: two routes to image-blind accuracy, tested zero-shot on six benchmarks

Appended 2026-10-12, before any run below. v1 and A0-A39 unchanged. Motivation: a concurrent study (Asadi et al., "MIRAGE", arXiv 2603.21687) reports that frontier models describe and answer questions about images never provided, and that a text-only model tops a chest X-ray VQA benchmark; it does not separate where that accuracy comes from. We distinguish a *knowledge route* (a model answers without the image from medical or world knowledge, without having seen the benchmark) from an *artifact route* (the benchmark's own text artifacts, learned from its training split), and test both on six benchmarks with open models.

## What has been seen (not predictions)
Blind ceilings (A36; an option-only classifier trained on the benchmark's own training questions): Med-CMR 93.7, OmniMedVQA 94.1, PMC-VQA 29.5, A-OKVQA 40.1, MedXpertQA-MM 22.9, MMMU health 23.5 (the last two cross-validated within the test set). Fine-tuned Qwen-3B black-image accuracy: 95.6, 93.2, 38.1, 49.6 on the first four benchmarks. Earlier zero-shot evidence: the zero-shot Qwen models pick the shortest Med-CMR option in 23-25% of the cases (A9), and the zero-shot image gaps on PMC-VQA are 14-16 points (A11). No zero-shot black-image or no-image result exists for MedXpertQA-MM or MMMU health, and no zero-shot "image unavailable" arm exists anywhere. A smoke test (12 questions per benchmark, numbers unused) checked the pipeline.

## Runs (`scripts/a40_zeroshot.py`, `slurm/49_a40_zeroshot.sbatch`, `scripts/a40_analysis.py`; hashed before the first submission)
Zero-shot Qwen2.5-VL-7B-Instruct (A30 GPU) and Qwen2.5-VL-32B-Instruct (H100), no training. Test sets: Med-CMR held-out MCQs (711), PMC-VQA E1 (1,999), OmniMedVQA E1 (2,000), A-OKVQA E1 (1,145), MedXpertQA-MM test (2,000; all images of the question), MMMU health multiple-choice validation questions (145; first three images). Prompt: question, lettered options, "Answer with only the letter"; letter log-probability readout. Arms: **true** image(s); **black** image(s) of the same size; **told**: no image, with the sentence "Note: the image could not be loaded. Answer using the question and options only."

## Rules
- **Z1 (knowledge route):** the 32B model's black-image accuracy exceeds chance by at least 10.0 points on at least two of PMC-VQA, MedXpertQA-MM and MMMU health.
- **Z2 (artifact route):** the blind ceiling exceeds the 7B model's zero-shot black-image accuracy by at least 30.0 points on both Med-CMR and OmniMedVQA (the benchmark's own text artifacts are not available to a model that has not seen its training split).
- **Z3 (explicit absence lowers accuracy, as MIRAGE reports for frontier models):** for the 32B model the "told" accuracy is lower than the black-image accuracy on at least four of the six benchmarks.
Reported, not tested: true-minus-black for each benchmark and model (zero-shot visual headroom), the Spearman correlation across the six benchmarks between the blind-ceiling excess over chance and the black-image excess over chance, and all intervals.

## Reading rule (fixed)
- Z1 and Z2 hold: the paper states that image-blind accuracy has two routes, one measured by a zero-shot no-image run (knowledge) and one by the blind ceiling (benchmark artifacts), and that Med-CMR and OmniMedVQA lie on the artifact route; Z1 or Z2 fails: the paper reports the numbers and does not claim two routes.
- Z3 holds: the paper states that the MIRAGE effect (lower accuracy when told that the image is missing) also appears in open models on these benchmarks; it fails: it states that it did not replicate in letter-readout multiple choice.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
Letter log-probability readout, not free generation (the "mirage" of free-text descriptions is not tested here); two open models of one family, not the frontier models of MIRAGE; benchmark sets differ in question style; MMMU uses at most three images; blind ceilings of two benchmarks are cross-validated within the test set; the black image is not the same as an absent image.

## Amendments log
(A0-A40.)
