# Amendment A8 to pre-registration v1: confirmatory test of "the image is a tie-breaker with no veto" on fresh data

Appended 2026-10-03, before any model has been run on the fresh sample. v1, A0-A7 unchanged. CPU selftest, then one GPU job.

## What has been seen (not confirmatory)
`analysis/exploratory_conditional_reliance.json` (sha256 `220eb21b...`), computed on TEST711 and DEV700 base/black arms already on disk: when the black-image ("text-only") margin is at least tau_m the image changed no prediction
(0 of 595 on test, 0 of 581 on dev); when it is below 0.90 the image was worth +11.1 points (test) and +13.8 points (dev). These hypotheses were formed from those numbers, so they are tested here only on **fresh cases never
used in any analysis**. Not seen: any model output on the fresh sample.

## Fresh sample
FRESH3000 = 3,000 MCQs drawn with `random.Random(20261003).sample` (ids sorted first) from the MCQ ids of R_test (`split_v1/r_test_ids.json`, i.e. the pool minus the 62 same-source cases) after removing the 700 ids of DEV700 (H2) and the 700 ids
of POOL700 (J1). Labels come from the official train JSON and are used only in the summary. **Caveat stated in advance:** these are pool cases, which the validity gate flagged as in-sample suspected for the adapter; the hypotheses are about
structure (how the image effect depends on text-only confidence), not about absolute accuracy, and every result is labelled in-sample suspected.

## Job `condrel_fresh_s0` (frozen adapter pipeline, teacher-forced `direct` readout, adapter ON)
Arms per case: `base` (true image); `black` (black 224x224); `mismatch_rand` (image of another sampled case, fixed cyclic derangement, seeded by 20261003); `mismatch_nn` (the image of the case's nearest neighbour by PubMedCLIP cosine
similarity within the same `primary_modality`, searched in R_test and excluding the case itself and any case in its same-source component: a plausible but wrong image). Donor images go through the target case's modality route and
preprocessing, as in Gate A. Readouts: `direct` (primary) and `cue`.

## Definitions (fixed)
Prediction = argmax of the `direct` letter log-probabilities. m_b = margin (top-1 minus top-2 probability) under the **black** arm; m_t = margin under the true-image arm; tau_m = 0.994782414189317 (A1).
Strata by m_b: **HI** m_b >= tau_m; **MID** 0.90 <= m_b < tau_m; **LO** m_b < 0.90. "Changed" = the prediction under one arm differs from the prediction under the true-image arm. Exact McNemar (two-sided) for paired comparisons, Wilson / Clopper-Pearson intervals.

## Predictions (each falsifiable)
- **C1 (no effect when text is decisive):** in HI, the share of cases whose prediction changes between `base` and `black` is at most 0.01. Falsified above 0.01.
- **C2 (tie-breaker when text is undecided):** in LO, accuracy(base) - accuracy(black) >= 5.0 percentage points with exact McNemar p < 0.05. Falsified otherwise.
- **C3 (concentration):** at least 90% of all `base`-vs-`black` prediction changes in the whole sample occur at m_b < tau_m (MID or LO). Falsified below 90%.
- **C4 (no veto even by a plausible wrong image):** in HI, the share of cases whose prediction changes between `base` and `mismatch_nn` is at most 0.02. Falsified above 0.02.
- **C5 (the headline hides it):** over all 3,000 cases, accuracy(base) - accuracy(black) < 3.0 points, while C2 holds. Falsified if the headline gap is 3.0 points or more or C2 fails.
- **C6 (the error alarm replicates):** the cases with m_t < tau_m (the flag of A1) make up between 10% and 25% of the sample and contain at least 80% of the `base` errors. Falsified otherwise.

## Reported without prediction
Stratum sizes; accuracy and image effect per stratum and per arm; change rates of `mismatch_rand` and `mismatch_nn` per stratum; **image influence in log-odds** (median and 90th percentile of |change in the log-odds margin between top-2 letters, base
vs black| per stratum) to separate "the image has no influence" from "the image has influence but cannot overturn a saturated text answer"; results on the shortest-wrong subset; the `cue` readout.

## Reading rule
C1-C4 and C6 all supported and C2 supported => "tie-breaker with no veto" confirmed for this model on fresh in-sample-suspected data. C2 falsified => the tie-breaker effect is not confirmed and the exploratory result is reported as chance variation. C1 or C4 falsified => the "no veto"
statement is withdrawn. Whatever the outcome, the paper reports it; the claim is limited to this one fine-tuned model until other models are tested (not part of this amendment).

## Amendments log
(A0, A1, A2, A3, A4, A5, A6, A7, A8.)
