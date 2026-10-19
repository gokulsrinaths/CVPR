# Amendment A31 to pre-registration v1: what in the option text of Med-CMR identifies the correct answer, beyond character length?

Appended 2026-10-05, before the analysis below was run. v1 and A0-A30 unchanged. Motivation: the paper shows that the length cue is sufficient to remove image use (planted cue) but not the only barrier on Med-CMR: models trained on length-randomised options still pick the originally shortest option in about 91% of questions (A12, L2: they read through the filler), and the image effect on the 235 cue-contradicting questions is -1.3 (3B) and 0.0 (7B) points (`analysis/cue_contra_results.md`). The remaining text-side signal has not been identified. A31 asks what in the option text carries it. Text only: no image, no model fine-tuning, CPU.

## What has been seen (not predictions)
Length statistics only: correct option the shortest of five in 91.9% of the 13,687 pool (A28 audit: 91.9% of the 14,398 training MCQs). `analysis/a12_results.md`, `analysis/cue_contra_results.md`. No wording or style feature has been examined, and the script `scripts/a31_option_audit.py` has not been run.

## Data and models (`scripts/a31_option_audit.py`, hashed before its first run)
Train: the MCQ cases of the training selection that are not in the 886-case held-out list; test: the 711 held-out MCQs. A question is *cue-contradicting* if its correct option is not the (tied) shortest; for these, chance among the four non-shortest options is 25% and the shortest-option rule scores 0%. Option-level logistic regression (balanced classes, C = 1; the predicted option is the highest score within the question) with features of each option: M1 length features (length, words, length rank, within-question z-score: sanity check, expected near the shortest rule); M2 word and bigram TF-IDF on all training questions; **M3 word and bigram TF-IDF trained only on cue-contradicting training questions**; M4 non-length style features (hedge-word share, comma share, word overlap with the question, share of words that occur in the other options, mean Jaccard overlap with the other options, negation start, digit flag; centred within the question) on all training questions; M5 the same style features on cue-contradicting training questions only; M6 TF-IDF plus style, cue-contradicting training questions only. The question text is used only for the overlap feature of M4-M6; no image is used. Evaluation: accuracy on all 711 test questions and on the cue-contradicting test questions (95% bootstrap interval, 1,000 resamples), and the share of predictions equal to the shortest option.

## Rules
- **W1 (wording signal beyond length):** M3 accuracy on the cue-contradicting test questions is at least 35.0% with the lower bound of its bootstrap interval at least 28.0% (chance 25%).
- **W2 (non-length style is enough):** M4 accuracy on all 711 test questions is at least 60.0% (chance 20%).
M1 (sanity), M2, M5 and M6 are reported, not tested.

## Reading rule (fixed)
- W1 holds: the paper states that the option wording of the questions carries a signal to the correct answer beyond character length (answer written without the image), names the feature groups that carry it according to M5/M6 versus M3, and says this is a text-side barrier distinct from length. W2 is reported alongside.
- W1 fails and W2 holds: the signal beyond length is recoverable from style features across all questions but not within cue-contradicting questions; reported as such.
- Both fail: the paper states that the remaining text-side barrier is not recoverable from option wording with these models (it may need the question-option interaction or medical knowledge), and leaves it unexplained.
No analysis is repeated or tuned after seeing its result; any further feature set needs its own amendment.

## Limits (stated in advance)
Linear models on hand-chosen features; the cue-contradicting training set is small (about 8% of the training questions); wording that a language model reads (as opposed to bag-of-words) is not tested here; case-disjointness only (no patient identifiers; 47 of the 711 test MCQs have a near-duplicate partner in the pool); the training-selection MCQs include cases outside the models' training split of earlier amendments.

## Amendments log
(A0-A31.)
