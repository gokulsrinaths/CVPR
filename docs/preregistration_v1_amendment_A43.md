# Amendment A43 to pre-registration v1: a frozen one-parameter hypothesis for the lock-out strength, with a prospective test

Appended 2026-10-12, before the prediction below was computed. v1 and A0-A42 unchanged. Motivation: the paper measures the strength d* at which a planted length shortcut locks out the image (about 0.85 for Qwen-3B on PMC-VQA, 0.99 on A-OKVQA) but cannot explain it; one attempted predictor (the image benefit without a cue, A29) failed on OmniMedVQA. A43 states a mechanistic hypothesis in advance, calibrates its single constant on one condition, and tests it on others, one of them with new runs.

## Hypothesis (fixed here)
Let V be the mean over test questions of log p(gold | true image) - log p(gold | black image), in nats, with the option letters of the question normalised, of the **control model** (d = 0.25, no cue; planted options of E1). Let S(d) = log(d (k-1) / (1-d)) be the log-odds evidence that a cue of strength d gives for the option that carries it (k = 4 options; S = 0 at d = 1/k). **H:** the image is locked out (visual veto boundary V* = none in every seed) where S(d) exceeds lambda * V, with one constant lambda common to all benchmarks and models; lambda is set once so that lambda * V(PMC-VQA, Qwen-3B) equals the mean of S at the highest unlocked and the lowest locked tested dose of that condition.

## What has been seen (not predictions)
The observed lock-out brackets of A17, A19, A21-A23 and A25 are known: PMC-VQA Qwen-3B unlocked at d = 0.70 and locked at 0.85 (all three seeds); PMC-VQA Qwen-7B and InternVL3-2B unlocked at 0.70 and locked at 0.92 (two seeds each); A-OKVQA Qwen-3B unlocked at 0.97 and locked at 0.99; Qwen-7B unlocked at 0.92 and locked at 0.99; InternVL3-2B not locked at 0.99. OmniMedVQA Qwen-3B (A29, two seeds, d up to 0.85) was not locked at 0.85 (g = +3.25 and +2.60). Therefore P1 and P2 below are **retrospective** (the brackets are known; V is computed only from the d = 0.25 runs and the hypothesis and tolerance are fixed here), and only P3 is prospective. No V has been computed before this amendment.

## Procedure (`scripts/a43_predict.py`, hashed before its first run)
1. `predict` computes V for each condition from the d = 0.25 control runs (mean over seeds of the per-seed mean), calibrates lambda on Qwen-3B / PMC-VQA and writes the predicted S and d* of every condition, including OmniMedVQA Qwen-3B, to `analysis/a43_prediction.json`; this file is hashed **before** the new runs are submitted.
2. New runs (`scripts/a29_run.py` unchanged, `slurm/41_a29_a30.sbatch`): OmniMedVQA, Qwen2.5-VL-3B, length shortcut, d = 0.92, 0.97 and 0.99, seeds 20261091 and 20261092 (6 runs, A30), summarised by `scripts/a29_summarize.py`.
3. `evaluate` recomputes the observed brackets (including the new doses) and scores the predictions.

## Rules (tolerance: 0.5 nats on the S axis around the observed bracket; a condition that is never locked up to its highest tested dose needs a predicted S at least S(highest dose) - 0.5)
- **P1 (retrospective):** the prediction for A-OKVQA Qwen-3B lies inside its observed bracket (calibration used PMC-VQA Qwen-3B only).
- **P2 (retrospective):** at least three of the four predictions for PMC-VQA Qwen-7B, PMC-VQA InternVL3-2B, A-OKVQA Qwen-7B and A-OKVQA InternVL3-2B lie inside their brackets.
- **P3 (prospective):** the frozen prediction for OmniMedVQA Qwen-3B lies inside its observed bracket after the new runs.
Reported, not tested: V of every condition, lambda, and the predicted d* of every condition.

## Reading rule (fixed)
- P1, P2 and P3 hold: the paper states that a one-constant evidence-competition hypothesis (shortcut log-odds against the visual log-odds gain) explains the lock-out strengths of seven conditions, one predicted before its runs; P3 fails: the paper reports the retrospective fits as consistent but not as predictive, and states that the prospective test failed; P1 or P2 fails: the paper states that the hypothesis does not fit the known brackets.
- The hypothesis is not modified or re-calibrated after seeing any result.

## Limits (stated in advance)
One constant fitted to a bracket (not a point); brackets are coarse (the dose grid has 3 to 7 values per condition); V* = none is the lock-out criterion of the paper (a +2-point criterion would differ); two seeds for the new runs; one shortcut (length); V is estimated on E1, the same set on which lock-out is evaluated.

## Amendments log
(A0-A43.)
