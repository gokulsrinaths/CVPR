# Amendment A36 to pre-registration v1: a counter-cue evaluation split and a blind ceiling for seven benchmarks

Appended 2026-10-06, before the analyses below were run. v1 and A0-A35 unchanged. Motivation: two quantities would turn the diagnostic findings into a reusable instrument: (1) how accurate the Med-CMR systems are on the questions on which the length cue points away (counter-cue questions), which the paper shows only through the image effect; (2) for each benchmark, a blind ceiling (what option text alone can reach) and the visual headroom above it.

## What has been seen (not predictions)
Counter-cue questions exist (A28, A33: 1,023 of 14,398 training MCQs; 235 of the 3,000 FRESH questions; 56 of the 711 TEST questions). Of the errors of the fine-tuned Qwen models, 106 of 108 (7B seed 1) and 119 of 121 (seed 2) are picks of the shortest option (A16), which is why the accuracy on counter-cue questions is expected to be much lower than on the others; no accuracy by counter-cue status has been computed. The black-image accuracy of the fine-tuned Qwen-3B is about 96% (Med-CMR), 37-39% (PMC-VQA), 93% (OmniMedVQA) and 49% (A-OKVQA, planted filler at d = 0.25); A28 gives the length statistics of the benchmarks. The option-only classifier of A31/A33 reaches 90.0% on TEST711 and 55.0% on the counter-cue questions. No blind ceiling has been computed for the other benchmarks.

## Part 1: counter-cue accuracy (`scripts/a36_countercue.py`, hashed before its first run)
Systems with per-question predictions on FRESH3000: A9 Qwen-3B and -7B, UL Qwen-3B (seeds 41-43) and Qwen-7B (seeds 51-52), LP+MU Qwen-3B (seeds 31-35). For each: accuracy with the true and the black image on counter-cue and on cue-following questions (95% bootstrap interval).
- **C1:** for both A9 models, accuracy on cue-following questions minus accuracy on counter-cue questions (true image) is at least 30.0 points.
Reported, not tested: all other systems and the black-image rows.

## Part 2: blind ceiling and visual headroom (`scripts/a36_blind.py`)
Seven benchmarks: Med-CMR (train: MCQs outside the 886 held-out cases; test: TEST711), PMC-VQA (30,000 random training questions; test: clean test, 2,000), OmniMedVQA open access (a random 2,000 as test, 30,000 other questions as training), A-OKVQA (official train, validation), MedXpertQA-MM and MMMU health (5-fold cross-validation within the test set, since no training split is used). Classifier: option-level logistic regression on word and bigram TF-IDF of the option plus four within-question length features; no question text, no image. **Blind ceiling** B_opt = its accuracy on the test set (95% bootstrap interval). **Visual headroom** = accuracy of the fine-tuned Qwen-3B with the true image minus the larger of B_opt and its accuracy with a black image, for the four benchmarks with such a model.
- **H1:** B_opt is at least 85.0% on Med-CMR and at most 45.0% on both PMC-VQA and A-OKVQA.
- **H2:** Med-CMR has the smallest visual headroom of the four.
Reported, not tested: B_opt of the other benchmarks and the headroom values.

## Reading rule (fixed)
- C1 holds: the paper states the counter-cue accuracy (with the interval) next to the overall accuracy and releases the split definition; C1 fails: the paper reports the accuracies as they are and says the counter-cue gap is smaller than expected.
- H1 and H2 hold: the paper presents the blind ceiling and visual headroom as the benchmark-level measures; either fails: the paper reports the values and drops the claim that they order the benchmarks.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
The blind classifier does not see the question, so for benchmarks whose answers depend on it (PMC-VQA, A-OKVQA) B_opt is a lower bound of what text alone can reach; one fine-tuned model per benchmark; the training sets of the classifiers differ in size and origin; the counter-cue split is defined by character length only (the wording cue of A33 is not part of the definition); one seed.

## Amendments log
(A0-A36.)
