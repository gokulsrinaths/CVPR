# Amendment A34 to pre-registration v1: existing sample-level measures against the stratified measures

Appended 2026-10-05, before the analysis below was run. v1 and A0-A33 unchanged. Motivation: contribution (i) of the paper says that the stratified protocol "separates systems with the same headline gap". A reviewer will ask what existing sample-level measures (the accuracy drop without the image, visual information gain as the gold log-probability gain from the image, the share of answers changed by blanking the image) already show on the same systems.

## What has been seen (not predictions)
Accuracy drops of these fine-tuned systems are in the paper (Med-CMR +0.3 for Qwen-3B and -7B; PMC-VQA +13.7, +14.9, +14.4; OmniMedVQA +5.7, +7.1, +16.2 points). Per-question log-probabilities with the true and the black image exist for all of them (`experiments/pilot/ft_3b_s0_*`, `ft_7b_s0_*`, `pmc_*_s0_*`, `omni16_*`). The VIG and the answer-change rate per system have not been computed.

## Method (`scripts/a34_baseline.py`, hashed before its first run)
Eight fine-tuned systems: Med-CMR (FRESH3000, Qwen-3B, Qwen-7B), PMC-VQA (E2, three families), OmniMedVQA (E2, three families), one seed each, true and black image only. Per system: accuracy drop (true minus black, points); mean and median VIG (gold log-probability with the true image minus with the black image, nats); share of answers changed by the black image; text-decisive share (black-image margin at or above the 15th percentile of the VAL700 margins, as in the paper); share changed within the text-decisive questions; image effect on the text-undecided questions.

## Rule
- **R1 (conventional measures carry the same information):** Spearman correlation across the eight systems of the accuracy drop with the mean VIG and with the share of changed answers is at least 0.9 for both.
Reported, not tested: the text-decisive share and the text-undecided image effect, and whether any conventional measure separates Qwen-3B from Qwen-7B on Med-CMR.

## Reading rule (fixed)
- R1 holds: the paper states that the existing sample-level measures rank these systems like the accuracy drop; what the stratified protocol adds is the split by text saturation and the dose-response, not a better ranking of systems by average image use; contribution (i) is reworded accordingly.
- R1 fails: the paper reports which conventional measure departs and by how much.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
Eight systems, one seed each; the released adapter and the 235-question cue-contradicting analysis are not part of this comparison (different run formats); no conventional per-question measure other than VIG and the change rate is computed.

## Amendments log
(A0-A34.)
