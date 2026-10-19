# Amendment A19 to pre-registration v1: held-out prediction of the shortcut dose-response for InternVL3-2B and Qwen2.5-VL-7B

Appended 2026-10-04, before any run below, after the Qwen2.5-VL-3B dose-response of A17 was analysed. v1 and A0-A18 unchanged. This is the held-out-prediction amendment announced in A17 (there called "A18").

## What has been seen (not predictions)
The complete A17 analysis for Qwen 3B (three seeds, six doses of the length shortcut; second shortcut, position, at three doses): image benefit g(intact), mean over seeds 20261011-13 at d = 0.25 / 0.40 / 0.55 / 0.70 / 0.85 / 0.92: **13.48 / 13.21 / 8.03 / 3.35 / 0.38 / 0.10** points; visual veto boundary V* (blur ladder, 7 = none) 3-4 at d = 0.25, 5 at 0.55-0.70, 7 at 0.85-0.92; H1, H2, H3a, H4, H5 and the position-shortcut predictions held, H3b did not. The A11 original-option models (cue 20-25%, no filler) of InternVL3-2B and Qwen-7B had gaps +14.4 and +14.9 points; A13 (92% planted, seed 20261003) gave gaps +0.2 and about 0 for InternVL3-2B and Qwen-7B. **Not seen:** any dose-response run for InternVL3-2B or Qwen-7B at the doses below, or any such model with a blur ladder.

## Held-out systems and runs
InternVL3-2B and Qwen2.5-VL-7B, the A11 recipe, PMC-VQA split of A11, the A17 length shortcut (`scripts/dose_plant.py`, filler on all options) at **d in {0.25, 0.70, 0.92}**, seed **20261011**, one run per model and dose (6 runs), arms and ladder as in A17 (`scripts/dose_run.py`, summary `scripts/dose_summarize.py`). A30 GPUs with gradient checkpointing (memory; same gradients). No tuning, no repeated run, no change of the predictions below after any output is seen.

## Quantitative predictions (fixed from the Qwen-3B fit; each model, each dose)
The Qwen-3B curve (means above) is used as the prediction for g(intact), rounded: **g_pred = 13.5 (d = 0.25), 3.4 (d = 0.70), 0.1 (d = 0.92) points**; the predicted veto boundary is V*(0.25) <= 6, V*(0.70) in {4, 5, 6, 7}, V*(0.92) = 7.
- **A (quantitative transfer):** |g_obs - g_pred| <= 4.0 points at each of the three doses.
- **B (boundary):** V*(0.25) <= 6 and V*(0.92) = 7.
- **C (monotone):** g_obs(0.25) - g_obs(0.70) >= 2.0 points, g_obs(0.70) - g_obs(0.92) >= 1.0 point, and g_obs(0.92) <= +2.0 points.
A model **passes** if A, B and C hold. Each model is reported separately; the held-out claim is "the dose-response transfers" only if both pass; if one passes, the paper states which. Where A fails but B and C hold, the paper reports the shift of the curve (the dose at which g drops below +2 points is bracketed by the three doses) as a model-dependent difference in sensitivity, not as a replication of the quantitative curve.

## Reading rule (fixed)
Both models pass: the paper states that the dose-response of the Qwen-3B transfers quantitatively to two further models (one smaller of a different family, one larger of the same family) at the tested doses. One passes: it states that it transfers to one of them. Neither: the dose-response is reported as qualitative (shape preserved if C holds; model-dependent location), and the manuscript's claim is restricted to one model family.

## Limits (stated in advance)
Three doses and one seed per model; the 3B curve is estimated with three seeds (sd of g about 0.4-1.1 points at the mid doses), so a 4-point tolerance is wide in some places and tight in the steep region (d = 0.55-0.70); gradient checkpointing on A30; one benchmark.

## Amendments log
(A0-A19.)
