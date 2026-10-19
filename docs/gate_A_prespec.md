# Gate A' (mvqa_diag): pre-specified questions, predictions and decision rules

Written 2026-09-30, BEFORE the job is submitted or any result exists. Do not edit after results arrive; add an
"amendments" section instead.

Job: `slurm/05_mvqa_diag.sbatch` -> `scripts/mvqa_diag.py`. Inputs: the 711 MCQ cases of the frozen held-out slice and
the frozen baseline pipeline (unmodified). Outputs: `experiments/pilot/mvqa_diag_holdout711_s0_<jobid>_<ts>/`
(`summary.json`, `env.json`, `greedy_repeat.jsonl`, `forward_arms.jsonl`, `config.json`, `errors.jsonl`).
Estimated cost: ~1-1.3 H100-hours (estimate from the baseline's ~2 s per MCQ case, not measured for this job).
No sampling; the only randomness is the fixed cyclic derangement used for the mismatched-image arm (seed 20260930).

## Validity gate (decides whether every other number can be trusted)

The teacher-forced answer distribution reconstructs the baseline's decision by forcing the model's own constant stub.
- **Pass:** base-arm argmax agrees with the frozen prediction on >= 99% of the 711 cases (direct or cue variant).
- **Partial:** 95-99% -> report, but the margins are only approximate confidence.
- **Fail:** < 95% -> the reconstruction is not faithful; discard the logit-based analyses (greedy-repeat still stands).
- **Early abort:** after 20 cases, abort if greedy flips > 3/20 or base-arm agreement < 0.90.
- **Letter mass:** if the mean probability mass on A-E at the direct position is < 0.5, use the cue variant and say so.

## Questions and what each outcome means

1. **Determinism / the 683 vs 684 question.** Greedy repeat vs the frozen predictions.
   - 0 flips on the same GPU type -> decoding is deterministic here; the paper's extra correct case is not explained by noise on this stack.
   - >= 1 flip -> greedy decoding is non-deterministic or hardware-dependent at the size of the observed gap; report the flipped case ids and their margins.
   - Flips concentrated in low-margin cases would support numerical non-determinism.
2. **Image reliance.** Accuracy with a black image and with another case's image (direct variant).
   - Either arm >= 90% -> MCQ accuracy is mostly text/prior-driven on this slice; every image-related claim must be qualified.
   - Both arms <= 60% -> the model genuinely depends on the image.
   - In between -> report as partial reliance; compare against the length-only control (91.1%).
3. **Option-order robustness.** Accuracy of perm1-4 (mapped back) vs base; position bias; permutation-averaged decision.
   - Permutation self-consistency changes the decision on a case only if the averaged probability over the 5 orderings flips it.
   - Report fixed and broken counts against the exact-McNemar table (6 fixes/0 broken, 8/1, 10/2, 12/3, 15/5 for p<0.05).
   - A gain that a shortest-option rule also reproduces is NOT evidence for the method: report `picks_equal_to_shortest_option`.
4. **Confidence signal.** AUROC of the base-arm probability margin for predicting correctness, and median margin of the 28 errors vs the
   correct cases.
   - AUROC >= 0.80 -> margin is a usable trigger for uncertainty-gated retrieval (R2) and the router.
   - 0.65-0.80 -> weak trigger; say so.
   - < 0.65 -> do not build an uncertainty-gated design on it.
5. **Reasoning-stub check.** Fraction of greedy raw generations whose think-block equals the stub (expected ~100%) and whether the
   forced-conclusion fallback ever fired.

## Not claimed by this job

It does not evaluate retrieval, self-consistency by sampling, open-ended answers, or any method; it measures the baseline
and its failure signals only. Nothing here is a tuned or cherry-picked result: all 711 cases, all arms, reported as run.
