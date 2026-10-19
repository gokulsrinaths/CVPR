# Amendment A5 to pre-registration v1: selective prediction (using the margin as an error alarm)

Appended 2026-10-02, before any selective-prediction statistic has been computed. v1, A0-A4 unchanged. CPU only; it re-reads existing run outputs
and runs no model.

## What has already been seen (not confirmatory)
On TEST711 the Gate D summary showed that M0 margins below tau_m = 0.994782 (the 15th percentile of the unlabeled VAL700 margins, set in A1) flag 112 cases
containing all 28 baseline errors. Gate A also reported a margin AUROC of 0.967 on the test set. Therefore **TEST711 results below are exploratory**. Also seen: the
aggregate margin distribution of POOL700 and VAL700 (Gate B J1: KS test, share with margin >= 0.90). **Not seen:** capture, selective accuracy or AUROC at tau_m on
DEV700 (H2 run) or POOL700 (J1 run). These two labelled samples are the confirmatory evidence, with the caveat that both are labelled in-sample suspected.

## Data (all read from existing runs; margins from the `direct` readout of the true-image arm)
TEST711 (Gate A, job 21406682), DEV700 (job 22454795), POOL700 (job 22436140), VAL700 (job 22436140, unlabeled, only for tau_m). Labels from the official train JSON.
DEV700 and POOL700 are separate samples that may overlap; they are never pooled.

## Scores compared (higher = more confident)
S1 margin = top-1 minus top-2 probability over the five letters. S2 top-1 probability. S3 negative entropy of the five-letter distribution.
S4 shortcut score = 1 if the predicted option is the shortest option (A0 rule), else 0. S5 random (seeded 20261002).

## Metrics
AUROC of each score for predicting a correct answer (ties 0.5), with a 2,000-sample bootstrap 95% interval (seed 20261002); risk-coverage accuracy at score thresholds
set to give about 99%, 95%, 90%, 84% and 70% coverage (actual coverage reported; ties included whole); AURC (area under the risk-coverage curve, ties averaged over 200 seeded
shuffles); and, at the fixed operating point tau_m: flagged fraction, errors captured, answered-set accuracy with exact (Clopper-Pearson) 95% intervals. Also on the
subset where the shortest option is wrong (n = 63 test, about 57 dev): AUROC of S1 and S4. And an image-sensitivity check: the share of cases with margin >= tau_m, the
median margin, and the share of unchanged predictions under the black-image arm (Gate A for test, job 22454795 for dev).

## Predictions (each falsifiable; evaluated on DEV700 and POOL700 separately; TEST711 reported as exploratory)
- **P-SP1 (the threshold transfers):** at tau_m, on each of DEV700 and POOL700: flagged fraction between 0.10 and 0.25, the flagged set contains at least 60% of the errors,
  and the answered (unflagged) set has accuracy at least 0.99. Supported only if all three hold on both samples; otherwise falsified.
- **P-SP2 (not just the shortcut):** AUROC(S1) minus AUROC(S4) >= 0.05 on each of DEV700 and POOL700. Falsified if smaller on either.
- **P-SP3 (works where the shortcut fails):** AUROC(S1) >= 0.80 on the shortest-wrong subset of DEV700. Falsified below 0.80 (small n; interval reported).
- **Reported without prediction:** all other scores, the TEST711 versions, the image-sensitivity check.

## Reading rule
The 28 errors on the test set keep any claim from it uncertain; the paper must give the interval (exact 28/28 lower bound is 87.7%) and state that TEST711 is exploratory.
If P-SP1 fails on either sample, the margin gate is reported as not transferring, whatever the test set showed.

## Amendments log
(A0, A1, A2, A3, A4, A5.)
