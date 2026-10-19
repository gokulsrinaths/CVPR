# Amendment A18 to pre-registration v1: shortcut-rescue training (a diagnosis-guided remedy) and a training-free baseline

Appended 2026-10-04, before any run or analysis below. v1 and A0-A17 unchanged. Numbering: the held-out-prediction amendment promised in A17 for InternVL3-2B and Qwen-7B will be **A19** (it must still be written and hashed before any such run).

## Motivation
A17 batch 1 shows a dose-response: as a planted length cue gets stronger, the image benefit of Qwen2.5-VL-3B on PMC-VQA falls to zero (seed 20261011: +13.8, +13.4, +7.8, +3.6, +0.3, 0.0 points at 25-92%). If the diagnosis is right, a model that is **prevented from using the shortcut during training** should keep its visual control. A18 tests a standard remedy (product-of-experts debiasing with a text-only bias expert, after Clark et al. 2019 and Cadene et al. 2019) in the controlled setting, and on the real Med-CMR shortcut. The technique is not new; the contribution tested here is that the diagnosis predicts when it is needed and that the dose-controlled setting verifies that it works.

## What has been seen (not predictions)
A9-A15 and A17 batch 1 (length shortcut, seed 20261011, all six doses; summary above). A17 batch 2 and all position-shortcut runs are not yet analysed. A12 M2 (text-only models and loss weighting on Med-CMR) was negative; the text-only models of A12 (fold A/B on Med-CMR, 3B) exist and their cross-fitted log-probabilities are reused here. Not seen: any output of the methods below.

## Method (frozen)
**Bias expert.** A text-only model (the question and options, black image; Qwen2.5-VL-3B, A11 recipe, LoRA) trained on one half of the training questions (fold by hash of the case id) scores the other half, so every training question gets bias log-probabilities l_b over the option letters from a model that has not seen it (cross-fitting; for Med-CMR the A12 models are reused).
**Rescue loss (PoE).** For a training question with the true image, the main model's option-letter log-probabilities l_m are combined with the bias expert's, s = log_softmax(l_m) + w * l_b (w = 1, renormalised), and the loss is -log softmax(s)[gold]; the bias term is a constant (no gradient). **At test time the main model is used alone** (no bias term). A **control** uses the same letter-restricted cross-entropy with w = 0 (isolates the effect of the bias expert from the change of loss form). The baseline ("naive") is the A17 run of the same dose and seed.
**Systems and runs.** (a) PMC-VQA with the length shortcut planted at d = 0.92 (A17 mechanism and split): bias experts (2 folds), PoE main models at seeds 20261011, 20261012, 20261013, control (w = 0) at seed 20261011. (b) d = 0.70: bias experts (2 folds), PoE at seed 20261011. (c) Med-CMR (real cue): PoE main models (Qwen 3B, A9 recipe and data, A12 cross-fitted bias log-probabilities) at seeds 20261011-13, evaluated as A9 (VAL700, TEST711, FRESH3000; arms true, black, random, nearest-neighbour). All on A30 GPUs, no gradient checkpointing for 3B unless it does not fit. Scripts `scripts/rescue_run.py`, `scripts/ft_run7.py`, `scripts/a18_contrastive.py`, `scripts/a18_analysis.py`.
**Training-free baseline (CPU).** On the stored outputs of the naive A17 runs, test-time contrastive scoring argmax_a [ l(a | true image) - lambda * l(a | black image) ] with lambda in {0.5, 1.0}.

## Endpoints and predictions
Planted PMC-VQA d = 0.92, E1 (planted options) and E1 original options; g(intact) = accuracy(true) - accuracy(black), paired McNemar and bootstrap as in A17.
- **M-A (visual control restored):** in every seed g(intact) of the PoE model >= +5.0 points with exact McNemar p < 0.05 (naive: about 0.0).
- **M-B (cue-free accuracy recovered):** in every seed accuracy of the PoE model on the original (cue-free) E1 options is at least 8.0 points above that of the naive model of the same seed (naive about 38%; the cue-free-trained model of A11 reaches about 52%).
- **M-D (attribution to the bias expert; seed 20261011):** g(intact) of PoE minus g(intact) of the w = 0 control >= +4.0 points.
- **M-E (intermediate dose; seed 20261011):** at d = 0.70, g(intact) of PoE >= naive + 3.0 points.
- **M-F (real shortcut; Med-CMR, Qwen 3B):** in at least two of three seeds, on the 410 FRESH questions that are text-undecided for the A9 3B model (the A9-LO set, A12) the image effect of the PoE model is >= +5.0 points with p < 0.05, **and** its overall FRESH3000 accuracy is within 2.0 points of the naive model (A9/A14 three-seed mean, 95.5%).
- **M-G (training-free baseline does not rescue):** for the naive models at d = 0.92 (three seeds), contrastive scoring with the better lambda gives g(intact) < +5.0 points on planted E1 and improves the original-E1 accuracy by less than 8.0 points over the naive true-image accuracy.
- **Reported without prediction:** accuracy on planted E1 (cue present; expected to fall for PoE), strata, text-margin distribution, shortest-pick rate, training loss, bias-expert accuracy.

## Reading rule (fixed)
M-A, M-B and M-D hold: the paper reports shortcut-rescue training as a **diagnosis-guided remedy validated in the controlled setting**, with M-E and M-F stating how far it carries (to an intermediate dose; to the real shortcut). M-F failing is reported as a limit: the remedy works when the shortcut is learnable by a text-only expert from other questions, and does not restore visual control on Med-CMR in this setting. M-A failing: the remedy is reported as negative and the diagnosis is not weakened by it (the dose-response is independent evidence). The method is described as a known technique applied to a diagnosed problem, not as a new algorithm.

## Limits (stated in advance)
PoE needs a text-only expert and cross-fitting (two extra trainings per dose); only the Qwen 3B model, one benchmark for the controlled part and one real shortcut; cue-present test accuracy is expected to fall; w = 1 is not tuned; one control seed.

## Amendments log
(A0-A18.)
