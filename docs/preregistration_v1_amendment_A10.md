# Amendment A10 to pre-registration v1: recipe robustness of the text-undecided image effect

Appended 2026-10-03, before any variant has been trained. v1, A0-A9 unchanged. A9 falsified the prediction that the image is worth at least +5 points on text-undecided (LO) questions in two public models fine-tuned with one short recipe.
The obvious objection is that the weak image use belongs to that recipe. A10 tests it with the three most plausible changes, without any selection among them.

## What has been seen (not predictions)
A9 results for the one-epoch, 448 px, language-layers-only recipe (analysis/ft_study_results.md): LO image effect -0.98 (3B) and +1.81 (7B) points. **Not seen:** any output of the variants below.

## Variants (fixed; each is trained once; no tuning, no second run unless a crash)
All use A9's training set TR (9,277 MCQs), prompt, loss (answer letter and end token), optimiser (AdamW, lr 1e-4, warm-up 3%, cosine), batch 16, seed 20261003, and readout, and differ in exactly one factor from A9:
- **V1 (more training):** Qwen2.5-VL-3B, **3 epochs** (reshuffled each epoch with seed + epoch).
- **V2 (more visual detail):** Qwen2.5-VL-3B, images downscaled to a longest side of **896 px** (training and evaluation).
- **V3 (trainable vision tower):** Qwen2.5-VL-3B, LoRA (r = 16, alpha = 32) on the **language model and the vision tower** linear layers (qkv, proj, gate/up/down).
- **V4 (larger and longer):** Qwen2.5-VL-**7B**, **3 epochs**.

## Evaluation (per variant)
VAL700 `base` (unlabeled, sets the model's own tau_m as in A9); TEST711 `base` and `black`; FRESH3000 `base` and `black` (the same fresh questions as A8/A9, unseen in training). Strata, margins, McNemar and intervals as in A8/A9.

## Predictions (each falsifiable; evaluated per variant; the conclusion is "robust" only if it holds for all four)
- **R1 (the weak image benefit is robust to recipe):** in LO, accuracy(base) - accuracy(black) is **below +5.0 points** (or has exact McNemar p >= 0.05). **Falsified for a variant** if its LO effect is at least +5.0 points with p < 0.05.
- **R2 (the habit is robust):** the share of TEST711 predictions equal to the shortest option (A0 rule) is at least 0.80.
- **R3 (saturation is robust):** in HI, the share of FRESH3000 cases whose prediction changes between `base` and `black` is at most 0.01.
- **R4 (error alarm):** flagged questions (true-image margin below the variant's tau_m) make up 10% to 25% of FRESH3000 and contain at least 80% of the errors.
- **Reported without prediction:** FRESH3000 and TEST711 accuracy, strata sizes, headline true-vs-black gap, training time and loss.

## Reading rule (fixed)
All R1 hold => the failure of C2 in A9 is not a recipe artefact for the tested factors, and the paper says so. Any variant with R1 falsified => image use on undecided questions can be induced by that factor, which the paper reports as the main caveat to A9
and as a lead (not as a general claim). Whatever happens is reported. Limits stated in advance: one family, three single-factor changes, one seed per variant.

## Amendments log
(A0-A10.)
