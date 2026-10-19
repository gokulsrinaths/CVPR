# Amendment A27 to pre-registration v1: does the image dependence of the gold-unlikelihood model rest on image content?

Appended 2026-10-05, before any run below. v1 and A0-A26 unchanged. Motivation: the paper reports that the UL models lose about 11.6 points when the image is replaced by a plausible wrong one but cannot say whether they read image content or only detect a mismatch of modality or organ.

## What has been seen (not predictions)
A26 (analysis/a26_results.md, price_of_vision_results.md): FRESH3000 accuracy of UL seeds 41/42/43 with the true / black / random-other / plausible-wrong image: 91.2/38.8/16.2/79.6, 90.3/21.6/13.9/78.8, 89.5/18.5/12.7/77.7 %; of LP+MU seeds 31/32: 79.9/71.6/48.6/76.7 and 79.4/74.2/56.8/77.8 %; of the A9 reference: 95.5/95.6/95.6/95.5 %. No blurred-image evaluation of these models exists.

## Evaluations (6; evaluation only, saved adapters, no training; A30 GPUs; script `scripts/ft_eval_blur.py`, mode evalblur)
FRESH3000 (as in A24/A26) with the true image blurred by Gaussian radius 32, 16, 8, 4, 2 px and intact (arm base), for: UL seeds 20261041, 20261042, 20261043; LP+MU seeds 20261031, 20261032; the A9 reference. The black-image accuracy of each model is taken from its own earlier evaluation. Analysis script `scripts/a27_analysis.py`, hashed before the first run.

## Rule (per UL seed)
- **C (graded content dependence):** accuracy rises from blur 32 to the intact image by at least 10.0 points, and from one ladder level to the next it never falls by more than 1.0 point. UL is called *content-dependent* if C holds in at least two of three seeds.
## Reading rule (fixed)
For each model, let R8 = (acc(blur 8) - acc(black)) / (acc(intact) - acc(black)) and R2 likewise for blur 2 (reported, not tested).
- C holds in at least two UL seeds and R8 < 0.8: the dependence needs image detail beyond coarse image identity; the paper states that UL's dependence is graded in image fidelity.
- R8 >= 0.8 in at least two UL seeds (a heavily blurred true image is almost as good as the intact one): the dependence is mostly on coarse image identity (modality, organ, global appearance), consistent with detection of a mismatch; the paper states this limit.
- Neither: reported as mixed with the numbers.
The A9 reference is expected to be flat (no dependence) and is reported as a control; LP+MU is reported without a rule. No evaluation is repeated or tuned after seeing its result.

## Limits (stated in advance)
Blur removes detail but keeps modality, organ and global layout, so a graded response to blur does not rule out dependence on global appearance; one model size and recipe; three UL seeds.

## Amendments log
(A0-A27.)
