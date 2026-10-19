# Amendment A30 to pre-registration v1: does the gold-unlikelihood intervention (UL) transfer to Qwen2.5-VL-7B on Med-CMR?

Appended 2026-10-05, before any run below. v1 and A0-A29 unchanged. Motivation: the one intervention that restored image use on the real shortcut (A26 UL, then A27) was tested on a single model size (Qwen2.5-VL-3B). The paper lists this as a limit; A30 tests the same method, unchanged, on the 7B model.

## What has been seen (not predictions)
A26 (`analysis/a26_results.md`): UL on Qwen2.5-VL-3B, seeds 41/42/43, FRESH3000 accuracy with the true image 91.2 / 90.3 / 89.5%, image effect (true minus black) +52 / +69 / +71 points, F1 and F2 met in 3 of 3 seeds. `analysis/price_of_vision_results.md`: true-image accuracy minus accuracy with a plausible wrong (nearest-neighbour) image +11.6 / +11.5 / +11.8 points (random-other-image accuracy 12.7 to 16.2%, below chance, so the black-image effect is partly an artefact). A27 (`analysis/a27_results.md`): accuracy by blur radius 32 / 16 / 8 / 4 / 2 / intact for UL seed 41 is 43.8 / 61.7 / 81.3 / 89.0 / 90.6 / 91.2%; R8 >= 0.8 in two of three seeds. No UL run on a 7B model exists. The 7B model without intervention (A9-style, seeds in A16) has FRESH accuracy around 96% and no image effect on text-decisive questions.

## Runs
`scripts/ft_run10.py` unchanged, `--model 7b --mode satw --method ul --gc`, A26 recipe (lambda = 1, Qwen2.5-VL-7B, learning rate 1e-4, one epoch, gradient checkpointing) at seeds 20261051 and 20261052; evaluation on FRESH3000 and TEST711 with arms true, black, random other image, plausible wrong image, exactly as A26; then the A27 blur evaluation (`scripts/ft_eval_blur.py --model 7b`, evalblur mode) of the two saved adapters. 4 jobs on A30 GPUs (two training runs, two blur evaluations); job files `slurm/42_a30_train.sbatch`, `slurm/43_a30_blur.sbatch`, `slurm/44_finalize_a30.sbatch`. The config of each training run names the A26 amendment because the script is unchanged; this amendment governs these runs. Analysis script `scripts/a30_analysis.py`, hashed before the first submission.

## Rules (per seed; FRESH3000)
- **H1 (transfer of image reliance):** the A26 rules F1 (image effect over all questions >= +3.0 points with exact McNemar p < 0.025) and F2 (true-image accuracy >= 88.0%) both hold.
- **H2 (not only a black-image artefact):** accuracy with the true image minus accuracy with the plausible wrong image is >= +5.0 points (the 3B models gave +11.5 to +11.8).
- **H3 (graded in sharpness):** rule C of A27: accuracy rises by at least 10.0 points from blur 32 to the intact image and never falls by more than 1.0 point from one level to the next. R8 = (acc(blur 8) - acc(black)) / (acc(intact) - acc(black)) is reported with the reading rule of A27 (R8 >= 0.8 means mostly coarse image identity).

## Reading rule (fixed)
- H1 holds in both seeds: the paper states that UL restores image use on the real shortcut for a second model size, and reports H2, H3 and the accuracy price for both sizes.
- H1 holds in exactly one seed: mixed; the paper reports both seeds and does not claim transfer.
- H1 fails in both seeds: UL is reported as not transferring to 7B at these settings (a negative result), with the accuracy and image effect obtained.
- H2 fails where H1 holds: the paper states that the black-image effect at 7B is mostly an artefact of the penalty.
No run is repeated or tuned after seeing its result.

## Expectation, not a rule
UL transfers with an image effect of the same order; the 88% accuracy floor may fail for 7B.

## Limits (stated in advance)
Two seeds; one method setting (lambda = 1); same Med-CMR evaluation sets as A26, which are not fresh; InternVL3-2B is not covered (the UL code path is specific to Qwen2.5-VL); blur dependence does not separate image content from coarse image identity.

## Amendments log
(A0-A30.)
