# Amendment A32 to pre-registration v1: a model-independent stratifier for text saturation

Appended 2026-10-05, before the analysis below was run. v1 and A0-A31 unchanged. Motivation: the paper defines text saturation from the fine-tuned model's own confidence under a black image (strata HI/MID/LO). A reviewer can object that this stratifier depends on the model and on an out-of-distribution input, so that "no answer changes in the confident stratum" is partly built in. A32 stratifies by a quantity that uses neither the model nor any image.

## What has been seen (not predictions)
A31 (`analysis/a31_results.md`): a bag-of-words option classifier with no image and no question identifies the correct option in 90.0% of the held-out Med-CMR questions. The A9 runs (`experiments/pilot/ft_3b_s0_22569048_*`, `ft_7b_s0_22569049_*`) have per-question predictions on FRESH3000 with the true, black, random-other and plausible-wrong image; their own black-image strata are reported in the paper (82-84% HI; image effect on the LO stratum -4.1 to +2.6 points). The A32 analysis has not been run.

## Method (`scripts/a32_stratifier.py`, hashed before its first run)
Text-only classifier: word and bigram TF-IDF of each option, option-level logistic regression (balanced classes, C = 1); out-of-fold scores (5 folds grouped by near-duplicate component, `split_v1/components.json`) over all Med-CMR training-selection MCQs that are not in the 886 held-out cases; the **text confidence** of a question is the top probability of the within-question softmax over its five option scores. FRESH3000 questions are split in terciles of the text confidence (bottom, middle, top). For each A9 run (Qwen-3B, Qwen-7B) and each tercile: accuracy with the true and with the black image, image effect, exact McNemar counts, and the share of predictions that change between the true image and each counterfactual (black, random other, plausible wrong). Agreement with the model's own stratifier: Spearman correlation between the text confidence and the model's black-image margin.

## Rules
- **S1 (the finding does not depend on the model-based stratifier):** in the top tercile, the share of predictions changed by the true image relative to each of the three counterfactual images is at most 2.0% for both models.
- **S3 (the two stratifiers agree):** Spearman correlation between the text confidence and the model's black-image margin is at least 0.4 for both models.
The image effect in the bottom tercile is reported, not tested.

## Reading rule (fixed)
- S1 holds: the paper states that the result "text-confident questions are not changed by any image" holds under a stratifier that uses neither the model nor any image (and reports S3).
- S1 fails: the paper states that the claim holds only for the model-based stratifier, gives the shares, and removes the stronger wording.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
A single text classifier; terciles are coarse; only the two A9 reference models are covered; the text confidence is high for most questions because the classifier exploits the same cue.

## Amendments log
(A0-A32.)
