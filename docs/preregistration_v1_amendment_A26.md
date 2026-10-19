# Amendment A26 to pre-registration v1: making the real Med-CMR shortcut yield image use, and measuring the price (gold-unlikelihood under a mismatched image; more seeds for the combined method)

Appended 2026-10-05, before any run below. v1 and A0-A25 unchanged (A25 runs are still finishing; its rules are not modified).

## What has been seen (not predictions), and why this amendment changes the criterion
A24 (analysis/a24_results.md) failed its criterion for all three methods (image effect >= +5 points on the 410-question A9 low-confidence set with FRESH3000 accuracy >= 93.5%). An exploratory re-analysis of the stored A24 outputs (scripts/cue_contra_runs.py, run before this amendment) shows that the criterion measured the wrong thing: for the combined method (LP+MU, seeds 31/32/33) the image effect over the 2,765 FRESH3000 questions on which the correct option is the shortest is +8.5 / +5.3 / +0.1 points (black-image accuracy 72.9 / 75.4 / 80.5%, true-image accuracy 81.4 / 80.8 / 80.6%, p < 1e-11 in the first two seeds), i.e. the intervention restores image use in two of three seeds and costs about 17 points of accuracy there (reference 98%); in the third seed accuracy is lost without image use. MU alone gives +3.4 / +4.7 / +3.4 points on the 235 questions on which the cue points away, with 5-8 points lower accuracy there; the length prior (LP) alone and the tempered expert of A20 give about 0. The thresholds of this amendment (below) were therefore chosen after seeing A24 and are not independent of it. The evaluation sets (FRESH3000, TEST711) were used by A9, A18, A20 and A24.

## Methods (Qwen2.5-VL-3B, A9 recipe, A30 GPUs; script `scripts/ft_run10.py`)
- **UL (gold-unlikelihood under a mismatched image):** standard loss plus `lambda * (-log(1 - p_gold(mismatched)))`, where p_gold(mismatched) is the model's probability of the gold letter for the same question shown with the image of another training case of the same modality and organ system; `lambda = 1`. Unlike MU it does not ask for a uniform answer distribution, only that the gold option cannot be recognised from the text alone.
- **LP+UL:** UL together with the explicit length prior as bias expert of A24 (`w = 1`).
- **LP+MU (A24, "both"):** two further seeds (20261034, 20261035) to measure how often it works.
Seeds 20261041, 20261042, 20261043 for UL and for LP+UL (6 runs); 20261034 and 20261035 for LP+MU (2 runs). 8 runs.

## Rules (per method; level 0.025 for the two new methods)
- **F1 (image reliance restored):** in a given seed, the image effect (true minus black accuracy) over all FRESH3000 questions is at least +3.0 points with exact McNemar p < 0.025.
- **F2 (accuracy floor):** in the same seed, FRESH3000 accuracy with the true image is at least 88.0%.
- **Success of a new method (UL, LP+UL):** F1 and F2 in at least two of three seeds.
- **LP+MU reliability:** with the A24 seeds (31-33) and the new seeds (34, 35), the number of the five seeds that satisfy F1 and F2 is reported (no success rule).
Reported, not tested: image effect separately on cue-following and cue-contradicting questions, shortest-option picks on TEST711, TEST711 accuracy, text-decisive answers changed by a counterfactual image, and the accuracy-versus-image-effect trade-off over all methods of A18-A26.
**Expectation, not a rule:** UL is more stable than MU; the accuracy floor of 88% is not reached by LP+MU.

## Reading rule (fixed)
- A new method succeeds: the paper reports it as an intervention that restores image use on the real shortcut at the stated accuracy cost, together with all other methods tried (none omitted) and the post-hoc origin of the criterion.
- No method succeeds: the paper reports the trade-off (accuracy versus image effect) over all methods and states that at these settings image reliance cannot be bought without losing more than 7 points of accuracy.
Either way the paper states the price of image reliance on Med-CMR as measured. No run is repeated or tuned after seeing its result; any further method needs its own amendment and counts as another attempt.

## Limits (stated in advance)
One model size and recipe; three seeds per new method; thresholds chosen after A24; the evaluation sets are not fresh; the image effect over all questions can be driven by the 2,765 cue-following questions.

## Amendments log
(A0-A26.)
