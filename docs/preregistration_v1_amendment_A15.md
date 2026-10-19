# Amendment A15 to pre-registration v1: a text-vision competition score for predicting visual overrides

Appended 2026-10-03, before any quantity below has been computed or any run below has been made. v1 and A0-A14 unchanged.

## Motivation
An outside review of v4 said the technical object (a black-image confidence margin) is simple. A15 tests whether a quantity that combines text-side decision strength with counterfactual visual evidence predicts, better than text confidence alone, whether the true image changes a model's answer, and whether this holds across datasets, architectures and the planted-cue intervention (A13). The test is built so that it can fail: the visual evidence is measured on **other** images (random and nearest-neighbour wrong images) and used to predict the effect of the **true** image. Measuring the pressure on the same image whose override it predicts would make the prediction a tautology and is not done.

## What has been seen (not predictions)
All A8-A14 results, including that on Med-CMR fewer than 2% of questions change answer between the black and the true image, that on PMC-VQA about 40% do, that the planted cue removes the image gap (A13), and that an image-evidence combiner did not improve the **error** alarm (M1, a different target). Not seen: any competition score, any AUROC for override prediction, any output on blurred images, any wrong-image output of the planted-cue models.

## Definitions (frozen; log-probabilities over the K option letters, renormalised: l(a|I))
For a question and a system: a* = the answer under the black image; **M_T** = l(a*|black) - max_{a != a*} l(a|black) (text-side decision margin, in log-odds; m_b is the paper's probability margin under the black image). For an image I': **Delta(I')** = M_T + V(I'), where V(I') = max_{a != a*} l(a|I') - l(a*|I'): how far I' moves the log-odds against a*. **Delta_wrong** = max(Delta(random image), Delta(nearest-neighbour image)). **Override** O = [prediction under the true image != a*]. Scores for predicting O (higher = more likely): S1 = -m_b (text saturation alone, as in the paper); S2 = -M_T; S3 = Delta_wrong (visual evidence alone); **S4 = Delta_wrong / (max(M_T, 0) + 0.1)** (competition ratio R); S5 = Delta_wrong - M_T (deficit form, minus D = M_T - Delta_wrong). epsilon = 0.1 is fixed.

## Systems (10)
Med-CMR, FRESH3000, four image arms: released adapter (A8), Qwen2.5-VL-3B and 7B (A9), InternVL3-2B (A11 Part 2). PMC-VQA E1, four arms: Qwen 3B, 7B, InternVL3-2B (A11). PMC-VQA planted-cue E1 (planted options): the three A13 models, **evaluated for the wrong-image arms in Part 2 below** (arms base, black, random, nearest neighbour; adapters from A13).

## Part 1: prediction of O (CPU only, `scripts/a15_competition.py`)
- **C1 (beyond saturation):** for each system with at least 40 overrides, AUROC(S4) - max(AUROC(S1), AUROC(S2)) >= +0.02 with the lower end of a paired bootstrap 95% interval (500 resamples) above 0; C1 holds if this is true for at least 70% of the evaluable systems.
- **C2 (transfer across datasets and architectures):** for each held-out system j (leave-one-system-out; training pool = the other nine systems, each subsampled to at most 1,500 questions with seed 20261004, equal weight), fit a logistic regression P(O) = sigma(a + b x), (i) with x = M_T (text-only model) and (ii) with x = log10 R clipped to [-3, 3] (competition model), and compute the log-loss on all questions of system j. C2 holds if the competition model has the lower log-loss in at least 8 of 10 held-out systems and the mean relative reduction is at least 5%.
- **Reported without prediction:** all AUROCs (S1-S5), AUPRC, expected calibration error (10 bins) of the LOSO models, a two-feature additive model (M_T, log Delta_wrong) as a comparison, binned P(O) against R and against m_b for each system (the "collapse" plot).
- **Reading rule:** C1 and C2 both hold: the paper is rebuilt around the competition score as its technical contribution. C2 fails but C1 holds: the score is a within-system predictor only, and the paper reports it as a secondary analysis. Neither holds: the paper reports the negative result and keeps its current structure.

## Part 2: wrong-image arms for the planted-cue models (small GPU job)
The A13 adapters (Qwen 3B, 7B, InternVL3-2B) are evaluated on E1 with planted options, arms base, black, mismatch_rand, mismatch_nn (donors of A11), `scripts/a15_eval_pmc.py`, no training. Part 1 is run on all ten systems after this part.
- **C3 (mechanism):** for each family, the median M_T of the planted-cue model on E1 is larger, and the median R smaller, than that of the A11 model of the same family.

## Part 3: visual-strength factorial (small GPU jobs)
Visual evidence strength is manipulated on the **true image** by Gaussian blur of radius 8 px and 24 px (PIL, on the image downscaled to at most 448 px), giving image strengths black < blur24 < blur8 < true. Text strength: Med-CMR (cue present), PMC-VQA original, PMC-VQA planted cue. Systems: the A9 Qwen 3B/7B and A11 InternVL3-2B models on Med-CMR FRESH3000 (arms base, blur8, blur24, plus the existing black), the A11 models on PMC-VQA E1 and the A13 models on planted PMC-VQA E1 (arms base, blur8, blur24, black). The image benefit g(I) = accuracy(I) - accuracy(black). Predictions (per family):
- **F1 (unlocked: PMC-VQA original):** g(blur24) < g(blur8) < g(true).
- **F2 (locked: Med-CMR and planted PMC-VQA):** |g(I)| <= 1.0 point for I in {blur24, blur8, true} for Med-CMR, and <= 1.0 point for planted PMC-VQA.
- **F3 (interaction):** [g(true) - g(blur24)] on PMC-VQA original is at least 3 times that on planted PMC-VQA (for each family).
- Reported: override rate [prediction != answer under black] by image strength and condition; accuracy by image strength.

## Execution note (written before any run)
The H100 queue was saturated (estimated start about ten hours later), so the evaluation-only jobs of Parts 2 and 3 are run on NVIDIA A30 (24 GB) nodes of the same cluster (partition beartooth-gpu), bf16, batch size one, same code path. bf16 results can differ in the last digits between GPU types. To avoid mixing hardware inside one comparison, every Part 2 and Part 3 job evaluates **all** arms it needs itself (base, black, blur8, blur24, and for PMC-VQA also mismatch_rand and mismatch_nn), and the analyses of Part 3 and of the planted-cue systems use only the rows of these jobs; the seven systems of Part 1 that were run earlier on H100 keep their stored rows.

## Limits (stated in advance)
Log-probabilities of the letter at the answer position are the "scores"; blur is one way to weaken visual evidence and also a distribution shift; wrong-image influence is only a proxy for visual counterevidence on the true image; one seed per system; ten systems are not ten independent datasets (three families x three conditions, plus the adapter).

## Amendments log
(A0-A15.)
