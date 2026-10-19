# Amendment A24 to pre-registration v1: interventions on the real Med-CMR shortcut (explicit length prior, mismatched-image uniformity)

Appended 2026-10-05, before any run below. v1 and A0-A23 unchanged (A23 runs are in progress; its rules are not modified).

## What has been seen (not predictions)
Five earlier attempts to act on Med-CMR's own option-length cue failed (A6, A7 not evaluable; A12 padding: models read through the filler; A18 product-of-experts with the cross-fitted neural text-only expert: collapse to chance; A20 tempered expert: image effect +0.2, -0.2, -0.2 points on the A9 low-confidence set, 90% shortest-option picks). The A9 Qwen-3B reference model: FRESH3000 accuracy 95.5%, image effect on its own text-undecided set (410 questions, "A9-LO set") -0.98 points, shortest-option picks 91-92%. The evaluation sets used below (A9-LO set, FRESH3000, TEST711) were therefore already used to evaluate A9, A18 and A20; no hyper-parameter below was tuned on them. Nothing of the A24 methods has been run on them. Reasons for the earlier failures, as far as known: the neural text-only expert saturated (median margin 0.9995) and learned more than length (A18, A20); a blank image can be recognised, so penalising behaviour on it would only teach the model to recognise the blank image (not tried, by design).

## Methods (all: Qwen2.5-VL-3B, the A9 recipe: LoRA r=16, one epoch, lr 1e-4, images <= 448 px, loss on the answer letter; training set TR of A9; script `scripts/ft_run9.py`)
- **M1 `lenprior` (explicit length prior as bias expert):** product-of-experts loss `-log softmax(log p_main + w * log q)[gold]` with `q_i = P(correct | length rank of option i)`, the empirical probability, estimated on TR (add-one smoothing), that the option of that character-length rank (ties by letter) is correct; `w = 1`. At test time the main model alone. The expert cannot saturate beyond that probability (about 0.9) and uses only length.
- **M2 `mu` (mismatched-image uniformity):** the standard loss with the true image plus `lambda * KL(uniform || softmax(z_mis))`, where `z_mis` are the five option-letter logits for the same question with the image of another training case of the **same modality and organ system** (different image), `lambda = 1`; this requires the model to compare image and question instead of answering from the text, and a blank image is not used.
- **M3 `both`:** M1 and M2 together (`w = 1`, `lambda = 1`).
No other value of `w` or `lambda` is run.

## Runs (9; A30 GPUs)
Seeds 20261031, 20261032, 20261033 for each method; evaluation as in A20 (`scripts/a12_summarize.py m2`): TEST711 and FRESH3000 with true / black / mismatched images, strata with the model's own margins.

## Rules (per method; Bonferroni over the three methods)
- **E1 (success):** in at least two of three seeds, the image effect (true minus black accuracy) on the A9-LO set is at least +5.0 points with exact McNemar p < 0.0167 **and** FRESH3000 accuracy with the true image is at least 93.5%.
Reported, not tested: shortest-option pick rate on TEST711, share of text-decisive answers changed by a counterfactual image, image effect on the model's own text-undecided stratum, TEST711 accuracy, training loss.

## Reading rule (fixed)
- At least one method meets E1: the paper reports that a method, tried together with the other two (all three reported, none omitted), restores visual use on the real shortcut in the stated sense, with its accuracy cost; the number of methods tried (three) is stated in the abstract/outcomes.
- None meets E1: the paper reports eight failed attempts on the real shortcut, states which part failed (effect or accuracy), and keeps the remedy as a negative result.
- A method that meets the effect criterion but not the accuracy criterion is reported as such (a trade-off), not as success.
Results are reported whatever they are; no run is repeated or tuned after seeing its result; any further method would need its own amendment and counts as another attempt in the text.

## Limits (stated in advance)
One model size and one recipe; three seeds; the evaluation sets were seen by earlier methods (see above); the A9-LO set is defined by the A9 reference model, so a successful method changes the questions on which the image is decisive; training cost of M2/M3 about twice that of M1.

## Amendments log
(A0-A24.)
