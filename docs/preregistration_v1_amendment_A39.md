# Amendment A39 to pre-registration v1: the released MedReason adapter on the counter-cue and blind-fail held-out questions

Appended 2026-10-12, before the analysis below was run. v1 and A0-A38 unchanged. Motivation: the accuracy of the challenge's own released system where the length cue points away has not been computed (A36 used our fine-tuned systems), and a concurrent study (Asadi et al., "MIRAGE", arXiv 2603.21687) cleans benchmarks of non-visual inference; we ask what such a cleaning does to the accuracy of the released system and of ours on the held-out MCQs.

## What has been seen (not predictions)
A36: counter-cue accuracy of our fine-tuned Qwen-3B and -7B on FRESH3000 is 68.0% and 73.3% against 97.5-97.7% on the others (206 counter-cue questions). The released adapter reaches 683/711 (96.06%) on the 711 held-out MCQs (A8, `data/holdout886/run_output/results.json`); its accuracy by counter-cue status has not been computed. Of the 711 held-out MCQs, 56 are counter-cue (A31).

## Method (`scripts/a39_released.py`, hashed before its first run)
Test set: the 711 held-out MCQs (the released adapter never trained on them). *Counter-cue*: the correct option is not the tied-shortest. *Blind-fail*: the option-level classifier of A36 (word and bigram TF-IDF of the option plus four within-question length features, no question and no image, trained on the MCQs outside the 886 held-out cases) picks a wrong option; this is a model-free cleaning in the spirit of removing questions that text alone solves. For the released adapter and for our systems with held-out predictions (A9 Qwen-3B and -7B, UL Qwen-3B seeds 41-43 and Qwen-7B seeds 51-52): accuracy on all questions, counter-cue, cue-following and blind-fail questions, with 95% bootstrap intervals.

## Rule
- **C2:** the released adapter's accuracy on the counter-cue questions is at most 80.0% and its accuracy on the cue-following questions exceeds it by at least 15.0 points.
Reported, not tested: all other systems and subsets, and whether the order of the systems on all questions equals their order on the counter-cue and the blind-fail questions.

## Reading rule (fixed)
- C2 holds: the paper states the released system's accuracy on counter-cue questions next to its 96.1% overall accuracy.
- C2 fails: the paper reports the numbers and says the released system loses less than our systems where length points away.
No analysis is repeated or tuned after seeing its result.

## Limits (stated in advance)
56 counter-cue and about 45 blind-fail questions, so wide intervals; the two subsets overlap; one released system; the blind-fail definition depends on one classifier.

## Amendments log
(A0-A39.)
