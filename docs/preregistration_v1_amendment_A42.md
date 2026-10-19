# Amendment A42 to pre-registration v1: a natural dose-response, without planted shortcuts

Appended 2026-10-12, before the analysis below was run. v1 and A0-A41 unchanged. Motivation: the dose-response of the paper is obtained by planting a shortcut, and a reviewer can argue that a deliberately introduced shortcut is predictably used. A42 asks whether the same competition between nonvisual and visual evidence appears in the unmodified benchmarks: whether the image benefit falls with how predictable a question is from option text alone, with nothing planted.

## What has been seen (not predictions)
Blind ceilings (A36): Med-CMR 93.7%, OmniMedVQA 94.1%, PMC-VQA 29.5%, A-OKVQA 40.1%. Image benefits over all questions of the fine-tuned models: Med-CMR about 0 (A9), OmniMedVQA +5.7 to +16.2, PMC-VQA +12.6 to +14.9, A-OKVQA about +34 (original options). A32 (Med-CMR, FRESH3000, terciles of the text confidence): image effect -0.5 and +0.9 points in the least text-predictable tercile, 0.0 and -0.2 in the most predictable. No image benefit has been computed by quintile of text predictability for PMC-VQA, OmniMedVQA or A-OKVQA.

## Method (`scripts/a42_natural.py`, hashed before its first run)
*Predictability* of a question: the top probability of the within-question softmax over the option scores of an option-level logistic regression (word and bigram TF-IDF of the option plus four within-question length features; no question, no image) trained on the benchmark's own training split (PMC-VQA TR, OmniMedVQA TR, A-OKVQA TR; Med-CMR: out-of-fold over the MCQs outside the held-out list, folds grouped by near-duplicate component). Evaluation questions: Med-CMR FRESH3000; PMC-VQA E1 and E2 (4,999); OmniMedVQA E1 and E2 (5,000); A-OKVQA E1 (1,145; original options). Systems: A9 Qwen-3B and -7B (Med-CMR); the three fine-tuned families of A11 (PMC-VQA) and A16 (OmniMedVQA), one seed each; the three families of A21 trained with the d = 0.25 control and evaluated on the original options, seeds 21 and 22 (A-OKVQA). Within each benchmark the questions are split into quintiles of predictability (quintile 1 least, quintile 5 most predictable); the image benefit (accuracy with the true image minus with a black image, points) is averaged over the models of the benchmark within each quintile (95% bootstrap interval resampling questions).

## Rules (each over the four benchmarks)
- **N1 (monotone):** the Spearman correlation between the quintile index and the image benefit over the five quintiles is at most -0.8, in at least three of the four benchmarks.
- **N2 (size):** the image benefit of quintile 1 exceeds that of quintile 5 by at least 5.0 points, in at least three of the four benchmarks.
Reported, not tested: all quintile values, the true-image accuracy per quintile, and the benchmarks that fail.

## Reading rule (fixed)
- N1 and N2 both hold: the paper states that the image benefit falls with the text predictability of the questions in the unmodified benchmarks, so that the competition between nonvisual and visual evidence found with planted shortcuts is also present naturally (correlational).
- One of them fails: the paper states which benchmarks show the natural trend and which do not, and that the natural data corroborate the planted dose-response only in part.
- Both fail: the paper states that the natural variation in text predictability did not reproduce the planted dose-response.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
Correlational: predictability is confounded with question type and difficulty; the predictability measure captures only artifacts learnable from the benchmark's own training split (the artifact route), not knowledge the model brings; one definition of predictability; models pooled within a benchmark (the same questions for all); a flat curve on Med-CMR is expected because the image benefit is near zero throughout (A32), so the rules are defined on three of four benchmarks.

## Amendments log
(A0-A42.)
