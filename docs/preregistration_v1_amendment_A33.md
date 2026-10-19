# Amendment A33 to pre-registration v1: the wording signal of Med-CMR at scale

Appended 2026-10-05, before the analysis below was run. v1 and A0-A32 unchanged. Motivation: A31 found that the option wording identifies the correct answer beyond length, but on only 56 cue-contradicting test questions (wide intervals). A33 repeats the question on all cue-contradicting questions of the benchmark with out-of-fold predictions, and asks which kinds of wording carry the signal.

## What has been seen (not predictions)
A31 (`analysis/a31_results.md`): on 56 cue-contradicting test questions a TF-IDF model reached 51.8% [39.3, 64.3] (trained on cue-contradicting questions only) and 64.3% [51.8, 76.8] (trained on all), chance 25%; style features alone (all groups together, trained on all questions) 23.2%, trained on cue-contradicting questions only 35.7% [23.2, 50.0]. The A33 script has not been run.

## Method (`scripts/a33_wording.py`, hashed before its first run)
All Med-CMR training-selection MCQs (14,398); a question is *cue-contradicting* if its correct option is not the (tied) shortest (about 1,000 questions). Option-level logistic regression, 5 folds grouped by near-duplicate component, out-of-fold predictions for every question. Models: TF-IDF trained on all training folds; TF-IDF trained on the cue-contradicting training folds only; each of six single style-feature groups (hedging words, commas per word, word overlap with the question, word overlap with the other options, negation at the start, digits) and all of them together, trained on the cue-contradicting training folds only. Accuracy on the cue-contradicting questions with a 95% bootstrap interval resampled by component (1,000 resamples), and the share of predictions equal to the shortest option. Descriptive: the 25 strongest word or bigram coefficients for the correct and for the wrong options.

## Rules
- **V1 (the signal replicates at scale):** TF-IDF trained on all training folds reaches at least 40.0% on the cue-contradicting questions with lower bound at least 35.0%, **and** TF-IDF trained on cue-contradicting questions only reaches at least 35.0% with lower bound at least 30.0% (chance 25%).
- **V2 (which kind of wording):** the single style-feature groups whose lower bound exceeds 25% are listed (reported, not a success rule).

## Reading rule (fixed)
- V1 holds: the paper states that the wording of the correct option identifies it beyond character length on all cue-contradicting questions of the benchmark (not only the held-out 56), with the interval, and names the style groups of V2 and the strongest terms descriptively, without claiming that they cause the model's behaviour.
- V1 fails: the paper reduces the A31 statement to a pointer from 56 questions that did not replicate at scale and says the remaining text-side barrier is not identified.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
Bag-of-words and six hand-made style groups; it does not test what a language model reads; a model that sees the question could find more or less; descriptive terms are associations in the benchmark's own writing, not causes; near-duplicate partners are handled by component grouping only.

## Amendments log
(A0-A33.)
