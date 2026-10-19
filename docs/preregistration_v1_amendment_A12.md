# Amendment A12 to pre-registration v1: an intervention on option length, and two methods that use text saturation

Appended 2026-10-03, before any run below. v1 and A0-A11 unchanged.

## Motivation
A9-A11 show that the models trained on Med-CMR are text-saturated, pick the shortest option 91-92% of the time and gain nothing from the image on text-undecided questions, while the same models on PMC-VQA (no length cue) look at the image. Two attempts to test the length cue causally with an LLM rewriter failed their gates (A6, A7). A12 replaces the rewriter by a mechanical, label-blind intervention and adds two methods that follow from the finding.

## What has been seen (not predictions)
All A8-A11 results (summaries in `analysis/`). Not seen: any output of the interventions, of the text-only bias models, of the weighted model, or of the alarm combiner below.

## Part L: option-length intervention by neutral filler (`scripts/len_rand.py`, hashed)
For every question, every option is extended by a deterministic string of neutral adverbs ("often", "typically", "especially", ...) so that the five final option lengths are exactly s_max + 4j characters (j = 1..5, s_max the longest original option); all five options therefore carry filler, each by at least 4 characters. Two modes:
- **pres** (cue preserved): the original length order is kept (ties broken by a seeded coin). The control for the filler itself.
- **rand** (cue randomised): the rank of every option is a uniformly random permutation (seeded per question), so the correct option's final length rank is uniform on 1-5 (shortest with probability 0.2).
Limit stated in advance: the amount of filler is a function of the original length and the assigned rank, so a model could in principle strip the filler and recover the original length; the diagnostic "prediction equals the original-shortest option" measures this.
**L1 (test-time intervention).** The A9 adapters (Qwen2.5-VL-3B and 7B, trained on original options) are evaluated on TEST711 and FRESH3000, conditions orig (TEST711 only), pres and rand, arms true and black image. Predictions (each model, each set):
- **V (validity of the filler):** accuracy in pres is within 3.0 points of the A9 accuracy on the same questions, and accuracy in orig equals the A9 value within 0.5 points (adapter loaded correctly). If V fails the L1 predictions are not evaluable.
- **L1a:** accuracy(pres) - accuracy(rand) >= 20.0 points.
- **L1b:** under rand, the share of predictions equal to the (final) shortest option >= 0.50.
**L2 (training-time intervention).** Qwen2.5-VL-3B and 7B are fine-tuned with the A9 recipe (same data, order, seed) on the rand version of the training questions, and evaluated on the rand version of VAL700, TEST711 and FRESH3000, arms true and black. Predictions (each model):
- **L2a (habit):** the share of TEST711 predictions equal to the final shortest option <= 0.35.
- **L2b (text no longer decisive):** the share of FRESH3000 questions with a black-image margin >= 0.99 is <= 0.50 (A9: 0.76-0.79).
- **L2c (the image matters):** accuracy(true) - accuracy(black) over all FRESH3000 questions (rand) >= +5.0 points, exact McNemar p < 0.05 (A9: -0.1 and +0.2).
- Reported without prediction: accuracy, strata, shortest-original diagnostic, undecided-stratum effect.
**Reading rule.** L2a and L2c hold: with the length cue removed the image-blindness goes away, which is causal evidence that the cue drives it (within the limit above). L2a holds and L2c fails: the models stop following length but remain image-indifferent, so the image-blindness does not depend on length alone. L2a fails: the intervention did not remove the cue for the learner (the filler can be stripped), and no causal claim is made. L1 holds: the models use length (a causal statement about use, not about training).

## Part M1: counterfactual-evidence alarm (CPU only; `scripts/m1_alarm.py`, hashed)
The margin alarm (A5) is blind to the image. M1 adds image-evidence features computed from the four arms already available for each question: true margin, black-image margin, Jensen-Shannon divergence between the true-image and black-image distributions, agreement of the true-image prediction with the random-image and nearest-neighbour-image predictions, JS divergence to each of those, and the top-1 probability. A logistic regression (L2, C = 1, standardised features) on these eight features predicts whether the true-image answer is wrong. PMC-VQA: fit on E2, test on E1 (article-disjoint). Med-CMR (fresh 3,000): two folds by a hash of the case id, out-of-fold predictions. Systems: the six runs of A9/A11 (Qwen 3B and 7B, InternVL3-2B, on both benchmarks). Predictions:
- **M1a:** on PMC-VQA E1, AUROC(combined) - AUROC(margin) >= 0.03 for each of the three systems.
- **M1b:** on PMC-VQA E1, the share of errors in the 15% highest-scored questions improves by >= 3.0 points for each of the three systems.
- **M1c (no harm):** on Med-CMR, AUROC(combined) >= AUROC(margin) - 0.01 for each of the three systems.
- Reported with bootstrap intervals (1,000 resamples) for each AUROC difference.

## Part M2: saturation-aware training (Qwen2.5-VL-3B, Med-CMR)
Idea: questions that a model with no image already answers confidently teach the image nothing; downweight them. Procedure (frozen): (1) split the training questions by a hash of the case id into folds A and B; (2) train a **text-only bias model** on fold A (A9 recipe, but the image is always a black 224x224 image), score fold B with a black image, and vice versa, so every training question gets a cross-fitted text-only margin m_cf; (3) train the **weighted model** exactly as A9 (same data, order, seed, true images) but multiply each example's loss by w = 1.0 if m_cf < 0.90 and by w = 0.25 otherwise; (4) evaluate as A9 on VAL700, TEST711 and FRESH3000 (arms true, black, random, nearest-neighbour). The comparison model is the A9 3B model. The **A9-LO set** is the set of FRESH3000 questions whose black-image margin under the A9 3B model is below 0.90 (n about 400); all primary comparisons use these same questions. Predictions:
- **M2a:** on the A9-LO set, accuracy(true) - accuracy(black) of the weighted model >= +5.0 points with p < 0.05 (A9 3B: about -1).
- **M2b:** accuracy(true) of the weighted model on the A9-LO set exceeds the A9 3B model's by >= 3.0 points.
- **M2c (no cost):** overall FRESH3000 accuracy of the weighted model >= A9 3B accuracy - 1.0 point.
- Reported without prediction: share of training questions with m_cf < 0.90, own strata, shortest-option share on TEST711, training loss.
**Reading rule.** M2a and M2b hold: text-saturation-aware weighting restores image use on text-undecided questions at no cost (M2c), a positive method result. Otherwise the weighting does not help in this setting and the paper reports it as a negative result.

## Limits (stated in advance)
One recipe and one seed per run; the filler is artificial; M2 uses one model size; the text-only models see half of the training data; the 7B models are used only in Part L; no tuning of any threshold (0.90, 0.25, 0.03, 3.0, 20.0, 5.0) after seeing results.

## Amendments log
(A0-A12.)
